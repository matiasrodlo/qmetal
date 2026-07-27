// kernels.metal — in-place statevector kernels.
//
// Convention: LSB-first. Qubit q occupies bit q of the basis index, stride
// 1 << q. Amplitudes are complex64 stored as float2 (re, im).
//
// Every index is 64-bit. At n = 33 a single-qubit gate needs 2^32 threads,
// which does not fit a 32-bit grid coordinate, so dispatches are 2-D and the
// linear thread id is reassembled as ulong. Nothing here may assume 32 bits.
//
// All kernels are in place: one read and one write of each touched amplitude,
// no destination buffer. That is worth one qubit of capacity and ~14% of
// throughput on unified memory, both measured.

#include <metal_stdlib>
using namespace metal;

struct Gate2x2 {
    float2 m00, m01, m10, m11;
};

static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Reassemble a 2-D grid position into a 64-bit linear index. grid_w is always a
// power of two, so this is exact.
static inline ulong lin(uint2 gid, uint grid_w) {
    return (ulong)gid.y * (ulong)grid_w + (ulong)gid.x;
}

// Insert a zero bit at position p, shifting higher bits up.
static inline ulong insert_bit(ulong x, uint p) {
    ulong low = x & ((1ul << p) - 1ul);
    return ((x >> p) << (p + 1)) | low;
}

// ---------------------------------------------------------------------------

kernel void init_basis(device float2 *state [[buffer(0)]],
                       constant uint &grid_w [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    state[t] = float2(t == 0 ? 1.0f : 0.0f, 0.0f);
}

// Dense single-qubit gate. One thread owns the pair (i, i + 2^q).
// Threads: 2^(n-1).
kernel void apply_1q(device float2 *state [[buffer(0)]],
                     constant Gate2x2 &g [[buffer(1)]],
                     constant uint &q [[buffer(2)]],
                     constant uint &grid_w [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    ulong stride = 1ul << q;
    ulong i = insert_bit(t, q);
    ulong j = i | stride;
    float2 a = state[i];
    float2 b = state[j];
    state[i] = cmul(g.m00, a) + cmul(g.m01, b);
    state[j] = cmul(g.m10, a) + cmul(g.m11, b);
}

// K independent single-qubit gates in ONE pass. A thread owns the 2^K
// amplitudes that differ only in the K target bits, loads them into registers,
// applies every gate there, and writes them back. Total traffic is one read and
// one write of the state regardless of K, so K full-state passes collapse to
// one. Wires must arrive sorted ascending: the bit inserts assume it.
//
// K is a macro rather than a parameter so the register array is indexed by
// compile-time constants and never spills to memory.
#define DEFINE_GROUP_KERNEL(K, NAME)                                          \
kernel void NAME(device float2 *state [[buffer(0)]],                          \
                 constant Gate2x2 *gates [[buffer(1)]],                       \
                 constant uint *qs [[buffer(2)]],                             \
                 constant uint &grid_w [[buffer(3)]],                         \
                 uint2 gid [[thread_position_in_grid]]) {                     \
    ulong base = lin(gid, grid_w);                                            \
    for (uint j = 0; j < K; j++) base = insert_bit(base, qs[j]);              \
    ulong idx[1u << K];                                                       \
    float2 a[1u << K];                                                        \
    for (uint l = 0; l < (1u << K); l++) {                                    \
        ulong x = base;                                                       \
        for (uint j = 0; j < K; j++) if ((l >> j) & 1u) x |= (1ul << qs[j]);  \
        idx[l] = x;                                                           \
        a[l] = state[x];                                                      \
    }                                                                         \
    for (uint j = 0; j < K; j++) {                                            \
        uint bit = 1u << j;                                                   \
        Gate2x2 g = gates[j];                                                 \
        for (uint l = 0; l < (1u << K); l++) {                                \
            if (l & bit) continue;                                            \
            float2 x = a[l], y = a[l | bit];                                  \
            a[l]       = cmul(g.m00, x) + cmul(g.m01, y);                     \
            a[l | bit] = cmul(g.m10, x) + cmul(g.m11, y);                     \
        }                                                                     \
    }                                                                         \
    for (uint l = 0; l < (1u << K); l++) state[idx[l]] = a[l];                \
}

DEFINE_GROUP_KERNEL(2, apply_1q_group2)
DEFINE_GROUP_KERNEL(3, apply_1q_group3)
DEFINE_GROUP_KERNEL(4, apply_1q_group4)

// Cache-level blocking. A threadgroup stages one contiguous 2^b block of the
// state into threadgroup memory, applies every gate in the run to it, and
// writes it back once. Traffic is one read and one write of the state no matter
// how long the run is -- the same bargain the register groups make, one level
// further out in the hierarchy.
//
// Every gate must target a wire below b, so its amplitude pairs stay inside the
// block. Runs that leave the local window are not blocked.
kernel void apply_1q_blocked(device float2 *state [[buffer(0)]],
                             constant Gate2x2 *gates [[buffer(1)]],
                             constant uint *qs [[buffer(2)]],
                             constant uint &count [[buffer(3)]],
                             constant uint &b [[buffer(4)]],
                             threadgroup float2 *tile [[threadgroup(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint tgid [[threadgroup_position_in_grid]],
                             uint tgsize [[threads_per_threadgroup]]) {
    uint block = 1u << b;
    ulong base = (ulong)tgid * (ulong)block;

    for (uint i = tid; i < block; i += tgsize) tile[i] = state[base + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pairs = block >> 1;
    for (uint k = 0; k < count; k++) {
        uint q = qs[k];
        uint stride = 1u << q;
        Gate2x2 g = gates[k];
        for (uint t = tid; t < pairs; t += tgsize) {
            uint i = ((t >> q) << (q + 1)) | (t & (stride - 1));
            uint j = i | stride;
            float2 x = tile[i], y = tile[j];
            tile[i] = cmul(g.m00, x) + cmul(g.m01, y);
            tile[j] = cmul(g.m10, x) + cmul(g.m11, y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < block; i += tgsize) state[base + i] = tile[i];
}

// Controlled single-qubit gate: apply g to the target only where the control
// bit is set. One thread owns one (control=1) target pair, so only a quarter of
// the state is touched. Threads: 2^(n-2).
kernel void apply_controlled_1q(device float2 *state [[buffer(0)]],
                                constant Gate2x2 &g [[buffer(1)]],
                                constant uint &ctrl [[buffer(2)]],
                                constant uint &targ [[buffer(3)]],
                                constant uint &grid_w [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    uint lo = min(ctrl, targ), hi = max(ctrl, targ);
    ulong base = insert_bit(insert_bit(t, lo), hi);
    ulong i = base | (1ul << ctrl);
    ulong j = i | (1ul << targ);
    float2 a = state[i];
    float2 b = state[j];
    state[i] = cmul(g.m00, a) + cmul(g.m01, b);
    state[j] = cmul(g.m10, a) + cmul(g.m11, b);
}

// Diagonal two-qubit gate: multiply the |11> subspace by a phase. Covers CZ
// (phase -1) and CP(theta). Touches a quarter of the state, read-modify-write.
// Threads: 2^(n-2).
kernel void apply_diagonal_2q(device float2 *state [[buffer(0)]],
                              constant float2 &phase [[buffer(1)]],
                              constant uint &qa [[buffer(2)]],
                              constant uint &qb [[buffer(3)]],
                              constant uint &grid_w [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    uint lo = min(qa, qb), hi = max(qa, qb);
    ulong i = insert_bit(insert_bit(t, lo), hi) | (1ul << qa) | (1ul << qb);
    state[i] = cmul(phase, state[i]);
}

// SWAP: exchange the |10> and |01> amplitudes. Threads: 2^(n-2).
kernel void apply_swap(device float2 *state [[buffer(0)]],
                       constant uint &qa [[buffer(1)]],
                       constant uint &qb [[buffer(2)]],
                       constant uint &grid_w [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    uint lo = min(qa, qb), hi = max(qa, qb);
    ulong base = insert_bit(insert_bit(t, lo), hi);
    ulong i = base | (1ul << qa);
    ulong j = base | (1ul << qb);
    float2 tmp = state[i];
    state[i] = state[j];
    state[j] = tmp;
}

// Fused diagonal layer. An arbitrary run of commuting diagonal gates becomes
// ONE pass: the phase of basis index i is the sum of the angles whose mask is
// fully set in i. A QFT stage's entire controlled-phase ladder collapses here.
//
// Terms arrive sorted by ascending |angle| so the float32 accumulation adds
// small contributions to each other before they meet large ones. Without that
// ordering the tail of a phase ladder (pi/2^k for large k) is lost outright.
kernel void apply_diagonal_layer(device float2 *state [[buffer(0)]],
                                 constant ulong *masks [[buffer(1)]],
                                 constant float *angles [[buffer(2)]],
                                 constant uint &nterms [[buffer(3)]],
                                 constant float2 &global_phase [[buffer(4)]],
                                 constant uint &grid_w [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    ulong i = lin(gid, grid_w);
    float phi = 0.0f;
    for (uint t = 0; t < nterms; t++) {
        ulong m = masks[t];
        if ((i & m) == m) phi += angles[t];
    }
    float2 p = cmul(global_phase, float2(metal::cos(phi), metal::sin(phi)));
    state[i] = cmul(p, state[i]);
}

// ---------------------------------------------------------------------------
// Reductions
//
// Both use a grid-stride loop over a fixed 1-D grid rather than one thread per
// amplitude. That keeps the dispatch inside 32 bits at any n and lets the
// threadgroup count be tuned independently of the state size.
//
// Precision: each thread accumulates in float32, then one value per threadgroup
// is written out and the host finishes the sum in double. Apple GPUs have no
// fp64, so partial-sum widening on the host is the only lever available. With a
// few thousand threadgroups the per-thread run is short enough that the float
// stage stays near 1e-6 relative.

// Sum one float across a threadgroup: SIMD-group reduction, then a second pass
// over the per-SIMD-group results.
static inline float threadgroup_sum(float v, threadgroup float *scratch,
                                    uint tid, uint tgsize, uint lane,
                                    uint sg) {
    float s = simd_sum(v);
    if (lane == 0) scratch[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = (tgsize + 31u) / 32u;
        float x = (lane < nsg) ? scratch[lane] : 0.0f;
        return simd_sum(x);
    }
    return 0.0f;
}

// sum |amp|^2 over the state, one partial per threadgroup.
kernel void reduce_abs2(device const float2 *state [[buffer(0)]],
                        device float *partial [[buffer(1)]],
                        constant ulong &count [[buffer(2)]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tgsize [[threads_per_threadgroup]],
                        uint ntg [[threadgroups_per_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    ulong stride = (ulong)ntg * (ulong)tgsize;
    float acc = 0.0f;
    for (ulong i = (ulong)tgid * tgsize + tid; i < count; i += stride) {
        float2 a = state[i];
        acc = fma(a.x, a.x, fma(a.y, a.y, acc));
    }
    float total = threadgroup_sum(acc, scratch, tid, tgsize, lane, sg);
    if (tid == 0) partial[tgid] = total;
}

// Fused Pauli expectation: sum conj(psi_i) * sign(i) * psi_{i ^ flip} in one
// pass. No |P psi> is ever materialised. The i^ny global factor is applied on
// the host, where it is exact.
kernel void reduce_pauli(device const float2 *state [[buffer(0)]],
                         device float2 *partial [[buffer(1)]],
                         constant ulong &count [[buffer(2)]],
                         constant ulong &flip [[buffer(3)]],
                         constant ulong &sign_mask [[buffer(4)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint tgsize [[threads_per_threadgroup]],
                         uint ntg [[threadgroups_per_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    ulong stride = (ulong)ntg * (ulong)tgsize;
    float accr = 0.0f, acci = 0.0f;
    for (ulong i = (ulong)tgid * tgsize + tid; i < count; i += stride) {
        ulong j = i ^ flip;
        float2 a = state[i];              // psi_i
        float2 b = state[j];              // psi_j
        // The coefficient belongs to the SOURCE index: (P psi)_i = c(j) psi_j.
        // Evaluating it at i instead is correct whenever flip and sign_mask are
        // disjoint -- pure X or pure Z strings -- and silently sign-flips every
        // Y, since Y sets a bit in both masks.
        ulong m = j & sign_mask;
        uint pc = popcount((uint)(m & 0xFFFFFFFFul)) +
                  popcount((uint)(m >> 32));
        float s = (pc & 1u) ? -1.0f : 1.0f;
        // conj(a) * b * s
        accr += s * (a.x * b.x + a.y * b.y);
        acci += s * (a.x * b.y - a.y * b.x);
    }
    float tr = threadgroup_sum(accr, scratch, tid, tgsize, lane, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ti = threadgroup_sum(acci, scratch, tid, tgsize, lane, sg);
    if (tid == 0) partial[tgid] = float2(tr, ti);
}

// lambda += coeff * P psi, for building the adjoint co-state |lambda> = H|psi>
// one Hamiltonian term at a time. (P psi)_i = c(i ^ flip) psi_{i ^ flip}, so a
// term costs one pass and no intermediate buffer. coeff carries the real
// coefficient times i^ny, computed exactly on the host.
kernel void pauli_accum(device float2 *lambda [[buffer(0)]],
                        device const float2 *psi [[buffer(1)]],
                        constant ulong &flip [[buffer(2)]],
                        constant ulong &sign_mask [[buffer(3)]],
                        constant float2 &coeff [[buffer(4)]],
                        constant uint &grid_w [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    ulong i = lin(gid, grid_w);
    ulong j = i ^ flip;
    ulong m = j & sign_mask;
    uint pc = popcount((uint)(m & 0xFFFFFFFFul)) + popcount((uint)(m >> 32));
    float2 v = psi[j];
    if (pc & 1u) v = -v;
    lambda[i] += cmul(coeff, v);
}

// <lambda|P|psi> across two states, for the per-parameter adjoint overlap.
// Same reduction as reduce_pauli but reading a bra and a ket that differ.
kernel void reduce_pauli2(device const float2 *bra [[buffer(0)]],
                          device float2 *partial [[buffer(1)]],
                          constant ulong &count [[buffer(2)]],
                          device const float2 *ket [[buffer(3)]],
                          constant ulong &flip [[buffer(4)]],
                          constant ulong &sign_mask [[buffer(5)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tgsize [[threads_per_threadgroup]],
                          uint ntg [[threadgroups_per_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    ulong stride = (ulong)ntg * (ulong)tgsize;
    float accr = 0.0f, acci = 0.0f;
    for (ulong i = (ulong)tgid * tgsize + tid; i < count; i += stride) {
        ulong j = i ^ flip;
        float2 a = bra[i];
        float2 b = ket[j];
        ulong m = j & sign_mask;  // source index, as above
        uint pc = popcount((uint)(m & 0xFFFFFFFFul)) + popcount((uint)(m >> 32));
        float s = (pc & 1u) ? -1.0f : 1.0f;
        accr += s * (a.x * b.x + a.y * b.y);
        acci += s * (a.x * b.y - a.y * b.x);
    }
    float tr = threadgroup_sum(accr, scratch, tid, tgsize, lane, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ti = threadgroup_sum(acci, scratch, tid, tgsize, lane, sg);
    if (tid == 0) partial[tgid] = float2(tr, ti);
}

// Zero a buffer, for co-state initialisation.
kernel void zero_state(device float2 *v [[buffer(0)]],
                       constant uint &grid_w [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    v[lin(gid, grid_w)] = float2(0.0f, 0.0f);
}

// Per-block |amp|^2 sums for sampling: one value per fixed-size contiguous
// block, so a shot can be resolved by binary search over the blocks and a walk
// inside the chosen one, touching a single page of the shared buffer.
kernel void block_abs2(device const float2 *state [[buffer(0)]],
                       device float *sums [[buffer(1)]],
                       constant ulong &count [[buffer(2)]],
                       constant uint &block [[buffer(3)]],
                       constant uint &grid_w [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    ulong b = lin(gid, grid_w);
    ulong start = b * (ulong)block;
    ulong end = min(start + (ulong)block, count);
    float acc = 0.0f;
    for (ulong i = start; i < end; i++) {
        float2 a = state[i];
        acc = fma(a.x, a.x, fma(a.y, a.y, acc));
    }
    sums[b] = acc;
}

// ---------------------------------------------------------------------------

// Toffoli: flip the target where both controls are set. Threads: 2^(n-3).
kernel void apply_ccx(device float2 *state [[buffer(0)]],
                      constant uint &c0 [[buffer(1)]],
                      constant uint &c1 [[buffer(2)]],
                      constant uint &targ [[buffer(3)]],
                      constant uint &grid_w [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    ulong t = lin(gid, grid_w);
    uint a = min(min(c0, c1), targ);
    uint c = max(max(c0, c1), targ);
    uint b = c0 + c1 + targ - a - c;
    ulong base = insert_bit(insert_bit(insert_bit(t, a), b), c);
    ulong i = base | (1ul << c0) | (1ul << c1);
    ulong j = i | (1ul << targ);
    float2 tmp = state[i];
    state[i] = state[j];
    state[j] = tmp;
}
