// qmetal.h — stable C ABI.
//
// The only surface other languages should bind to. Plain C: opaque handles,
// flat arrays, integer return codes, no exceptions crossing the boundary.
// Anything here is expected to keep working; the C++ headers are not.
//
// Convention: return 0 on success and a negative code on failure, with
// qm_last_error() holding the message for the calling thread. Every entry
// point catches; nothing throws across the ABI.
//
// Index order is LSB-first: qubit q occupies bit q of the basis index.
// Amplitudes are complex64 stored interleaved (re, im, re, im, ...).

#ifndef QMETAL_H
#define QMETAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QMETAL_VERSION_MAJOR 0
#define QMETAL_VERSION_MINOR 1
#define QMETAL_VERSION_PATCH 0

// Gate codes. These are ABI: append only, never renumber.
typedef enum {
  QM_X = 0, QM_Y = 1, QM_Z = 2, QM_H = 3,
  QM_S = 4, QM_SDG = 5, QM_T = 6, QM_TDG = 7,
  QM_RX = 8, QM_RY = 9, QM_RZ = 10, QM_P = 11,
  QM_CX = 12, QM_CY = 13, QM_CZ = 14, QM_SWAP = 15,
  QM_CP = 16,
  QM_CCX = 17
} qm_gate_kind;

typedef struct qm_simulator_s *qm_simulator;
typedef struct qm_circuit_s *qm_circuit;

// A Hamiltonian as parallel arrays: one coefficient and three Pauli masks per
// term. Passing masks rather than per-qubit codes keeps a term at 32 bytes
// whatever its weight.
typedef struct {
  uint32_t num_terms;
  const double *coeffs;
  const uint64_t *x_masks;
  const uint64_t *y_masks;
  const uint64_t *z_masks;
} qm_hamiltonian;

// --- diagnostics -----------------------------------------------------------

void qm_version(int *major, int *minor, int *patch);
const char *qm_last_error(void);          // thread-local, valid until next call
int qm_is_available(void);                // 1 if a Metal device exists
const char *qm_device_name(void);
uint64_t qm_max_buffer_length(void);
uint64_t qm_recommended_working_set(void);
uint32_t qm_max_qubits(void);

// --- circuits --------------------------------------------------------------

qm_circuit qm_circuit_create(uint32_t num_qubits);
void qm_circuit_destroy(qm_circuit c);
int qm_circuit_add(qm_circuit c, qm_gate_kind kind, const uint32_t *qubits,
                   uint32_t num_qubits, const double *params,
                   uint32_t num_params);
size_t qm_circuit_size(qm_circuit c);

// Number of GPU passes the circuit lowers to. With fuse = 0 this equals the
// gate count; with fuse = 1 it is what execution will actually cost.
size_t qm_circuit_plan_size(qm_circuit c, int fuse);

// --- simulator -------------------------------------------------------------

qm_simulator qm_create(uint32_t num_qubits);
void qm_destroy(qm_simulator s);

int qm_reset(qm_simulator s);
int qm_run(qm_simulator s, qm_circuit c);         // fused
int qm_run_unfused(qm_simulator s, qm_circuit c); // ablation baseline
int qm_synchronize(qm_simulator s);

uint32_t qm_num_qubits(qm_simulator s);
uint64_t qm_dim(qm_simulator s);
uint64_t qm_dispatch_count(qm_simulator s);

// Zero-copy view of the interleaved state, valid until the next mutation.
// Length is 2 * qm_dim(). Returns NULL on error.
const float *qm_amplitudes(qm_simulator s);

int qm_norm(qm_simulator s, double *out);
int qm_expectation_pauli(qm_simulator s, uint64_t x_mask, uint64_t y_mask,
                         uint64_t z_mask, double *out);
int qm_expectation(qm_simulator s, const qm_hamiltonian *h, double *out);

// Draws `shots` basis-state indices without collapsing the state. `out` must
// hold `shots` entries. Identical seeds give identical samples.
int qm_sample(qm_simulator s, size_t shots, uint64_t seed, uint64_t *out);

// Adjoint gradient. `out_grad` must hold qm_circuit_size(c) entries; gates
// carrying no parameter get 0. Fails if a parameterised gate has no native
// generator (only RX, RY, RZ do).
int qm_energy_and_gradient(qm_simulator s, qm_circuit c,
                           const qm_hamiltonian *h, double *out_energy,
                           double *out_grad);

#ifdef __cplusplus
}
#endif

#endif  // QMETAL_H
