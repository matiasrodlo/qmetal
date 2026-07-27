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
        float2 a = state[i];              // psi_i
        float2 b = state[i ^ flip];       // psi_{i ^ flip}
        // popcount over a 64-bit mask, as two 32-bit halves.
        ulong m = i & sign_mask;
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
