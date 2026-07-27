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
