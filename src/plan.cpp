#include "qmetal/plan.h"

#include <array>
#include <cmath>
#include <algorithm>
#include <map>
#include <vector>
#include <stdexcept>

#include "qmetal/reference.h"  // gate_matrix_1q

namespace qmetal {
namespace {

// Product of two row-major 2x2 matrices, `second` applied after `first`.
void mat_mul(const cdouble *second, const cdouble *first, cdouble *out) {
  cdouble t[4];
  t[0] = second[0] * first[0] + second[1] * first[2];
  t[1] = second[0] * first[1] + second[1] * first[3];
  t[2] = second[2] * first[0] + second[3] * first[2];
  t[3] = second[2] * first[1] + second[3] * first[3];
  for (int i = 0; i < 4; i++) out[i] = t[i];
}

// Identity test with rtol=0. A relative tolerance silently discards genuine
// small-angle rotations: RZ(1e-6) is a real gate whose matrix sits ~1e-6 from
// the identity, while true algebraic cancellation (H*H) lands within ~1e-15.
// Only an absolute threshold separates them.
bool is_identity(const cdouble *m) {
  const double atol = 1e-12;
  return std::abs(m[0] - cdouble(1, 0)) <= atol &&
         std::abs(m[1]) <= atol && std::abs(m[2]) <= atol &&
         std::abs(m[3] - cdouble(1, 0)) <= atol;
}

PlanOp make_dense1q(uint32_t q, const cdouble *m) {
  PlanOp op;
  op.kind = PlanOp::Kind::Dense1q;
  op.qubits[0] = q;
  for (int i = 0; i < 4; i++) op.matrix[i] = m[i];
  return op;
}

// Lower one gate with no fusion at all.
PlanOp lower(const Gate &g) {
  PlanOp op;
  switch (g.num_qubits) {
    case 1: {
      cdouble m[4];
      gate_matrix_1q(g.kind, g.params, m);
      return make_dense1q(g.qubits[0], m);
    }
    case 2:
      switch (g.kind) {
        case G_CX:
        case G_CY: {
          op.kind = PlanOp::Kind::Controlled1q;
          op.qubits[0] = g.qubits[0];
          op.qubits[1] = g.qubits[1];
          gate_matrix_1q(g.kind == G_CX ? G_X : G_Y, nullptr, op.matrix);
          return op;
        }
        case G_CZ:
        case G_CP: {
          op.kind = PlanOp::Kind::DiagonalLayer;
          op.global_phase = diagonal_term(g, op.masks, op.angles);
          return op;
        }
        case G_SWAP:
          op.kind = PlanOp::Kind::Swap;
          op.qubits[0] = g.qubits[0];
          op.qubits[1] = g.qubits[1];
          return op;
        default:
          break;
      }
      break;
    case 3:
      if (g.kind == G_CCX) {
        op.kind = PlanOp::Kind::CCX;
        for (int i = 0; i < 3; i++) op.qubits[i] = g.qubits[i];
        return op;
      }
      break;
    default:
      break;
  }
  throw std::runtime_error(std::string("build_plan: unsupported gate ") +
                           gate_name(g.kind));
}

}  // namespace

bool is_diagonal(const Gate &g) {
  switch (g.kind) {
    case G_Z: case G_S: case G_SDG: case G_T: case G_TDG:
    case G_RZ: case G_P:
      return g.num_qubits == 1;
    case G_CZ: case G_CP:
      return g.num_qubits == 2;
    default:
      return false;
  }
}

double diagonal_term(const Gate &g, std::vector<uint64_t> &masks,
                     std::vector<double> &angles) {
  const uint64_t b0 = 1ull << g.qubits[0];
  switch (g.kind) {
    // diag(1, e^{i a}): one term on the |1> subspace, no global phase.
    case G_Z:   masks.push_back(b0); angles.push_back(M_PI);        return 0.0;
    case G_S:   masks.push_back(b0); angles.push_back(M_PI / 2);    return 0.0;
    case G_SDG: masks.push_back(b0); angles.push_back(-M_PI / 2);   return 0.0;
    case G_T:   masks.push_back(b0); angles.push_back(M_PI / 4);    return 0.0;
    case G_TDG: masks.push_back(b0); angles.push_back(-M_PI / 4);   return 0.0;
    case G_P:   masks.push_back(b0); angles.push_back(g.params[0]); return 0.0;
    // RZ = diag(e^{-it/2}, e^{it/2}) = e^{-it/2} * diag(1, e^{it}). The global
    // factor is tracked rather than dropped: it is unobservable alone but not
    // once the gate sits under a control.
    case G_RZ:
      masks.push_back(b0);
      angles.push_back(g.params[0]);
      return -g.params[0] / 2.0;
    case G_CZ:
      masks.push_back(b0 | (1ull << g.qubits[1]));
      angles.push_back(M_PI);
      return 0.0;
    case G_CP:
      masks.push_back(b0 | (1ull << g.qubits[1]));
      angles.push_back(g.params[0]);
      return 0.0;
    default:
      throw std::runtime_error("diagonal_term: gate is not diagonal");
  }
}

// CX(a,b) . RZ_b(t) . CX(a,b) is diagonal, even though none of the three gates
// is. Conjugating by CX makes the RZ see the parity x_a ^ x_b, so the block is
// exp(-i t/2 Z_a Z_b) -- the standard emission for a ZZ coupling, and the
// reason a Trotter or QAOA layer looks unfusable while being entirely phase.
//
// The mask form here tests "all bits set", i.e. AND, so the parity is expanded:
//   x_a ^ x_b = x_a + x_b - 2 x_a x_b
// giving terms (a, t), (b, t), (ab, -2t) and a global -t/2.
static bool match_zz(const Circuit &c, size_t i, uint32_t *qa, uint32_t *qb,
                     double *theta) {
  if (i + 2 >= c.ops.size()) return false;
  const Gate &g0 = c.ops[i], &g1 = c.ops[i + 1], &g2 = c.ops[i + 2];
  if (g0.kind != G_CX || g2.kind != G_CX || g1.kind != G_RZ) return false;
  if (g0.qubits[0] != g2.qubits[0] || g0.qubits[1] != g2.qubits[1])
    return false;
  if (g1.qubits[0] != g0.qubits[1]) return false;  // RZ must sit on the target
  *qa = g0.qubits[0];
  *qb = g0.qubits[1];
  *theta = g1.params[0];
  return true;
}

// Length of the diagonal-representable block starting at i: 3 for a ZZ
// conjugation, 1 for a plain diagonal gate, 0 if neither.
static size_t diagonal_block_len(const Circuit &c, size_t i,
                                 bool allow_zz) {
  uint32_t a, b;
  double t;
  if (allow_zz && match_zz(c, i, &a, &b, &t)) return 3;
  return is_diagonal(c.ops[i]) ? 1 : 0;
}

static void append_diagonal_block(const Circuit &c, size_t i, size_t len,
                                  PlanOp &op) {
  if (len == 3) {
    uint32_t a, b;
    double t;
    match_zz(c, i, &a, &b, &t);
    const uint64_t ma = 1ull << a, mb = 1ull << b;
    op.masks.push_back(ma);       op.angles.push_back(t);
    op.masks.push_back(mb);       op.angles.push_back(t);
    op.masks.push_back(ma | mb);  op.angles.push_back(-2.0 * t);
    op.global_phase += -t / 2.0;
    return;
  }
  op.global_phase += diagonal_term(c.ops[i], op.masks, op.angles);
}

// ---------------------------------------------------------------------------

Plan build_plan(const Circuit &c, const FusionOptions &opts) {
  Plan plan;
  plan.num_qubits = c.num_qubits;

  const size_t n_ops = c.ops.size();

  // Pass 1 (marking): find maximal runs of consecutive diagonal gates of length
  // two or more. Marking rather than emitting matters, because the window pass
  // that follows must not consume a diagonal gate that belongs to a run -- a
  // lone RZ is better merged into its wire's 2x2 product, but an RZ sitting in
  // a phase ladder is better left to the ladder.
  std::vector<bool> in_run(n_ops, false);
  std::vector<uint8_t> block_len(n_ops, 0);
  if (opts.fuse_diagonals) {
    size_t i = 0;
    while (i < n_ops) {
      size_t len = diagonal_block_len(c, i, opts.recognize_zz);
      if (len == 0) { i++; continue; }
      // Extend across consecutive diagonal blocks.
      size_t j = i, blocks = 0;
      while (j < n_ops) {
        size_t l = diagonal_block_len(c, j, opts.recognize_zz);
        if (l == 0) break;
        block_len[j] = static_cast<uint8_t>(l);
        j += l;
        blocks++;
      }
      // A lone plain gate is left to the window pass, which may merge it into
      // its wire's 2x2 product. A lone ZZ triple is still worth collapsing:
      // three passes become one.
      if (blocks >= 2 || block_len[i] == 3)
        for (size_t k = i; k < j; k++) in_run[k] = true;
      i = j;
    }
  }

  // Pass 2 (emission): walk the marked circuit, collapsing runs and windows.
  size_t i = 0;
  while (i < n_ops) {
    // --- marked diagonal run ---------------------------------------------
    if (in_run[i]) {
      size_t j = i;
      PlanOp op;
      op.kind = PlanOp::Kind::DiagonalLayer;
      while (j < n_ops && in_run[j]) {
        size_t len = block_len[j] ? block_len[j] : 1;
        append_diagonal_block(c, j, len, op);
        j += len;
      }
      plan.ops.push_back(std::move(op));
      i = j;
      continue;
    }

    // --- single-qubit window ---------------------------------------------
    // Operations on distinct wires commute, so a maximal window of 1-qubit
    // gates can be regrouped by wire without changing the unitary. Each wire's
    // subsequence collapses to one 2x2 product, so a window of W gates over w
    // wires costs w passes rather than W.
    if (opts.collapse_1q_windows && c.ops[i].num_qubits == 1) {
      size_t j = i;
      while (j < n_ops && c.ops[j].num_qubits == 1 && !in_run[j]) j++;
      if (j - i >= 2) {
        // Ordered by first appearance, so the emitted plan is deterministic.
        std::vector<uint32_t> order;
        std::map<uint32_t, std::array<cdouble, 4>> prod;
        for (size_t k = i; k < j; k++) {
          const Gate &g = c.ops[k];
          uint32_t q = g.qubits[0];
          cdouble m[4];
          gate_matrix_1q(g.kind, g.params, m);
          auto it = prod.find(q);
          if (it == prod.end()) {
            order.push_back(q);
            prod[q] = {m[0], m[1], m[2], m[3]};
          } else {
            mat_mul(m, it->second.data(), it->second.data());
          }
        }
        // Surviving wires, in ascending order: the group kernel inserts bits
        // at ascending positions and the plan should not depend on the order
        // gates happened to appear in.
        std::vector<uint32_t> live;
        for (uint32_t q : order)
          if (!is_identity(prod[q].data())) live.push_back(q);  // cancels drop
        std::sort(live.begin(), live.end());

        const size_t K = std::max<size_t>(1, opts.group_1q);
        for (size_t b = 0; b < live.size(); b += K) {
          size_t take = std::min(K, live.size() - b);
          if (take == 1) {
            plan.ops.push_back(make_dense1q(live[b], prod[live[b]].data()));
            continue;
          }
          PlanOp op;
          op.kind = PlanOp::Kind::Dense1qGroup;
          for (size_t k = 0; k < take; k++) {
            uint32_t q = live[b + k];
            op.group_qubits.push_back(q);
            const cdouble *m = prod[q].data();
            op.group_matrices.insert(op.group_matrices.end(), m, m + 4);
          }
          plan.ops.push_back(std::move(op));
        }
        i = j;
        continue;
      }
    }

    plan.ops.push_back(lower(c.ops[i]));
    i++;
  }
  return plan;
}

}  // namespace qmetal
