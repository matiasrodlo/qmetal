// capi.cpp — the C ABI, wrapping the C++ classes.
//
// Every entry point catches. An exception reaching a C caller is undefined
// behaviour, so the boundary converts each one into a negative return code and
// a thread-local message.

#include "qmetal/qmetal.h"

#include <cstring>
#include <new>
#include <string>

#include "qmetal/circuit.h"
#include "qmetal/plan.h"
#include "qmetal/simulator.h"

using namespace qmetal;

namespace {

thread_local std::string g_error;

void set_error(const char *what) { g_error = what ? what : "unknown error"; }

// Wrap a call so nothing escapes. Returns 0 on success, code on failure.
template <typename Fn>
int guard(Fn &&fn, int code = -1) {
  try {
    fn();
    return 0;
  } catch (const std::exception &e) {
    set_error(e.what());
    return code;
  } catch (...) {
    set_error("unknown exception");
    return code;
  }
}

GateKind to_kind(qm_gate_kind k) {
  // The C enum is deliberately its own numbering so the C++ one can be
  // reordered without breaking the ABI.
  switch (k) {
    case QM_X: return G_X;
    case QM_Y: return G_Y;
    case QM_Z: return G_Z;
    case QM_H: return G_H;
    case QM_S: return G_S;
    case QM_SDG: return G_SDG;
    case QM_T: return G_T;
    case QM_TDG: return G_TDG;
    case QM_RX: return G_RX;
    case QM_RY: return G_RY;
    case QM_RZ: return G_RZ;
    case QM_P: return G_P;
    case QM_CX: return G_CX;
    case QM_CY: return G_CY;
    case QM_CZ: return G_CZ;
    case QM_SWAP: return G_SWAP;
    case QM_CP: return G_CP;
    case QM_CCX: return G_CCX;
  }
  throw std::runtime_error("unknown gate code");
}

Hamiltonian to_hamiltonian(const qm_hamiltonian *h) {
  Hamiltonian out;
  if (!h) return out;
  if (h->num_terms && (!h->coeffs || !h->x_masks || !h->y_masks || !h->z_masks))
    throw std::runtime_error("hamiltonian: null array with non-zero num_terms");
  for (uint32_t i = 0; i < h->num_terms; i++) {
    PauliString p;
    p.x_mask = h->x_masks[i];
    p.y_mask = h->y_masks[i];
    p.z_mask = h->z_masks[i];
    out.add(h->coeffs[i], p);
  }
  return out;
}

}  // namespace

struct qm_circuit_s {
  Circuit c;
};
struct qm_simulator_s {
  Simulator *s;
};

// --- diagnostics -----------------------------------------------------------

extern "C" void qm_version(int *major, int *minor, int *patch) {
  if (major) *major = QMETAL_VERSION_MAJOR;
  if (minor) *minor = QMETAL_VERSION_MINOR;
  if (patch) *patch = QMETAL_VERSION_PATCH;
}

extern "C" const char *qm_last_error(void) { return g_error.c_str(); }

extern "C" int qm_is_available(void) {
  try {
    return Simulator::available() ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

extern "C" const char *qm_device_name(void) {
  static thread_local std::string name;
  try {
    name = Simulator::device_name();
  } catch (...) {
    name = "none";
  }
  return name.c_str();
}

extern "C" uint64_t qm_max_buffer_length(void) {
  try { return Simulator::max_buffer_length(); } catch (...) { return 0; }
}
extern "C" uint64_t qm_recommended_working_set(void) {
  try { return Simulator::recommended_working_set(); } catch (...) { return 0; }
}
extern "C" uint32_t qm_max_qubits(void) { return kMaxQubits; }

// --- circuits --------------------------------------------------------------

extern "C" qm_circuit qm_circuit_create(uint32_t num_qubits) {
  auto *c = new (std::nothrow) qm_circuit_s;
  if (!c) { set_error("out of memory"); return nullptr; }
  c->c.num_qubits = num_qubits;
  return c;
}

extern "C" void qm_circuit_destroy(qm_circuit c) { delete c; }

extern "C" int qm_circuit_add(qm_circuit c, qm_gate_kind kind,
                              const uint32_t *qubits, uint32_t nq,
                              const double *params, uint32_t np) {
  if (!c) { set_error("null circuit"); return -1; }
  return guard([&] {
    if (nq == 0 || nq > 3) throw std::runtime_error("gate arity must be 1..3");
    if (np > 3) throw std::runtime_error("at most three parameters");
    if (!qubits) throw std::runtime_error("null qubit array");
    if (np && !params) throw std::runtime_error("null parameter array");
    Gate g{};
    g.kind = to_kind(kind);
    g.num_qubits = static_cast<uint8_t>(nq);
    g.num_params = static_cast<uint8_t>(np);
    for (uint32_t i = 0; i < nq; i++) {
      if (qubits[i] >= c->c.num_qubits)
        throw std::runtime_error("qubit index out of range");
      g.qubits[i] = qubits[i];
    }
    for (uint32_t i = 0; i < np; i++) g.params[i] = params[i];
    c->c.add(g);
  });
}

extern "C" size_t qm_circuit_size(qm_circuit c) { return c ? c->c.size() : 0; }

extern "C" size_t qm_circuit_plan_size(qm_circuit c, int fuse) {
  if (!c) return 0;
  try {
    FusionOptions o = fuse ? FusionOptions{} : FusionOptions::none();
    return build_plan(c->c, o).size();
  } catch (const std::exception &e) {
    set_error(e.what());
    return 0;
  }
}

// --- simulator -------------------------------------------------------------

extern "C" qm_simulator qm_create(uint32_t num_qubits) {
  auto *h = new (std::nothrow) qm_simulator_s;
  if (!h) { set_error("out of memory"); return nullptr; }
  h->s = nullptr;
  try {
    h->s = new Simulator(num_qubits);
  } catch (const std::exception &e) {
    set_error(e.what());
    delete h;
    return nullptr;
  }
  return h;
}

extern "C" void qm_destroy(qm_simulator s) {
  if (!s) return;
  delete s->s;
  delete s;
}

extern "C" int qm_reset(qm_simulator s) {
  if (!s) { set_error("null simulator"); return -1; }
  return guard([&] { s->s->reset(); });
}

extern "C" int qm_run(qm_simulator s, qm_circuit c) {
  if (!s || !c) { set_error("null argument"); return -1; }
  return guard([&] { s->s->run(c->c); });
}

extern "C" int qm_run_unfused(qm_simulator s, qm_circuit c) {
  if (!s || !c) { set_error("null argument"); return -1; }
  return guard([&] { s->s->run(c->c, FusionOptions::none()); });
}

extern "C" int qm_synchronize(qm_simulator s) {
  if (!s) { set_error("null simulator"); return -1; }
  return guard([&] { s->s->synchronize(); });
}

extern "C" uint32_t qm_num_qubits(qm_simulator s) {
  return s ? s->s->num_qubits() : 0;
}
extern "C" uint64_t qm_dim(qm_simulator s) { return s ? s->s->dim() : 0; }
extern "C" uint64_t qm_dispatch_count(qm_simulator s) {
  return s ? s->s->dispatch_count() : 0;
}

extern "C" const float *qm_amplitudes(qm_simulator s) {
  if (!s) { set_error("null simulator"); return nullptr; }
  try {
    return s->s->raw();
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  }
}

extern "C" int qm_norm(qm_simulator s, double *out) {
  if (!s || !out) { set_error("null argument"); return -1; }
  return guard([&] { *out = s->s->norm(); });
}

extern "C" int qm_expectation_pauli(qm_simulator s, uint64_t x, uint64_t y,
                                    uint64_t z, double *out) {
  if (!s || !out) { set_error("null argument"); return -1; }
  return guard([&] {
    PauliString p;
    p.x_mask = x;
    p.y_mask = y;
    p.z_mask = z;
    *out = s->s->expectation(p);
  });
}

extern "C" int qm_expectation(qm_simulator s, const qm_hamiltonian *h,
                              double *out) {
  if (!s || !h || !out) { set_error("null argument"); return -1; }
  return guard([&] { *out = s->s->expectation(to_hamiltonian(h)); });
}

extern "C" int qm_sample(qm_simulator s, size_t shots, uint64_t seed,
                         uint64_t *out) {
  if (!s || (shots && !out)) { set_error("null argument"); return -1; }
  return guard([&] {
    auto v = s->s->sample(shots, seed);
    if (!v.empty()) std::memcpy(out, v.data(), v.size() * sizeof(uint64_t));
  });
}

extern "C" int qm_energy_and_gradient(qm_simulator s, qm_circuit c,
                                      const qm_hamiltonian *h,
                                      double *out_energy, double *out_grad) {
  if (!s || !c || !h || !out_grad) { set_error("null argument"); return -1; }
  return guard([&] {
    auto [e, g] = s->s->energy_and_gradient(c->c, to_hamiltonian(h));
    if (out_energy) *out_energy = e;
    if (!g.empty()) std::memcpy(out_grad, g.data(), g.size() * sizeof(double));
  });
}
