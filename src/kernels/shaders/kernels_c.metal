// Kernel set C — Metal shaders (MoE hot path).
//
// Contains: expert_mlp_swiglu_q8 — fused SwiGLU expert MLP over a dispatch-pair
// list, Q8_0 weights, f32 activations/accumulation (ADR-005 §1).
//
// The router (routerTopK) intentionally has NO Metal kernel: its outputs are
// host slices by contract (contracts.RouterArgs.out_ids/out_weights), the
// per-token work is a tiny n_experts × dim matvec, and ADR-001 rule 1 makes
// bit-level control of the top-k selection worth far more than the microseconds
// a GPU version would save. The CPU reference in src/kernels/cpu/kernels_c.zig
// is the permanent implementation.
//
// Dispatch story (bindings, params layout, grid geometry) is documented in
// PORTING-moe.md next to this file; the structs below must stay byte-identical
// to the Zig side (contracts.PairDispatch is frozen at 12 bytes).
//
// Numerics notes:
//   - silu uses precise::exp so results track the f32 CPU oracle within the
//     expert_mlp tolerance (atol 1e-3, rtol 5e-3) even if the pipeline is
//     compiled with fast math.
//   - out is accumulated with device atomic_float adds (relaxed order):
//     multiple pairs target the same token row, and pairs run in independent
//     threadgroups. Requires MSL 2.4+ / Apple7+ GPUs — true for every DS5 node.
//   - Q8_0 dequant is exactly f32(f16 scale) * i8 q. The f16 scale is read by
//     byte assembly (as_type<half>) so blocks need no alignment guarantee
//     beyond byte addressing.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared layout constants
// ---------------------------------------------------------------------------

constant constexpr uint Q8_BLOCK       = 32; // elements per Q8_0 block
constant constexpr uint Q8_BLOCK_BYTES = 34; // half scale + 32 × int8

// Upper bound on expert_ffn_dim across DS5 targets (235B: 1536, 30B: 768,
// synthetic: 128). Threadgroup h buffer: 1536 × 4 B = 6 KiB (limit 32 KiB).
constant constexpr uint DS5_MAX_FFN = 1536;

// Must match the ExpertMlpParams the glue writes into buffer(0). All fields
// uint32; total 20 bytes (Metal constant buffers need no trailing padding).
struct ExpertMlpParams {
    uint dim;      // hidden size: gate/up row length, out row length
    uint ffn_dim;  // expert FFN size: down row length, gate/up row count
    uint n_experts;
    uint n_pairs;  // pairs[] entries; grid.x may over-provision
    uint tile_out; // output columns per threadgroup along grid.y (see doc)
};

// Byte-identical to contracts.PairDispatch (extern struct, 12 bytes).
struct PairDispatch {
    uint  token;
    uint  expert;
    float weight;
};

// ---------------------------------------------------------------------------
// Q8_0 row dot products. `row` points at the first block of one weight row.
// value = f32(f16 scale) * q, accumulated in f32 (block-factored form:
// scale * Σ q·x — reassociation is inside the expert_mlp tolerance).
// ---------------------------------------------------------------------------

static inline half q8_scale(const device uchar* block) {
    return as_type<half>(ushort(block[0] | (ushort(block[1]) << 8)));
}

// x in device memory (activation row) — used for gate/up.
static inline float q8_dot_device(const device uchar* row,
                                  const device float* x,
                                  uint k) {
    float acc = 0.0f;
    const uint n_blocks = k / Q8_BLOCK;
    for (uint b = 0; b < n_blocks; ++b) {
        const device uchar* blk = row + b * Q8_BLOCK_BYTES;
        const device char*  qs  = (const device char*)(blk + 2);
        float s = 0.0f;
        for (uint i = 0; i < Q8_BLOCK; ++i) {
            s = fma(float(qs[i]), x[b * Q8_BLOCK + i], s);
        }
        acc = fma(float(q8_scale(blk)), s, acc);
    }
    return acc;
}

// x in threadgroup memory (the shared h vector) — used for down.
static inline float q8_dot_tg(const device uchar* row,
                              const threadgroup float* x,
                              uint k) {
    float acc = 0.0f;
    const uint n_blocks = k / Q8_BLOCK;
    for (uint b = 0; b < n_blocks; ++b) {
        const device uchar* blk = row + b * Q8_BLOCK_BYTES;
        const device char*  qs  = (const device char*)(blk + 2);
        float s = 0.0f;
        for (uint i = 0; i < Q8_BLOCK; ++i) {
            s = fma(float(qs[i]), x[b * Q8_BLOCK + i], s);
        }
        acc = fma(float(q8_scale(blk)), s, acc);
    }
    return acc;
}

static inline float silu_f32(float v) {
    return v / (1.0f + precise::exp(-v));
}

// ---------------------------------------------------------------------------
// expert_mlp_swiglu_q8
//
//   out[token] += weight * down( silu(gate(x[token])) ⊙ up(x[token]) )
//
// One threadgroup per (pair, output tile):
//   grid  = (n_pairs, ceil(dim / tile_out), 1) threadgroups
//   tg    = (TG_THREADS, 1, 1); TG_THREADS = 256 recommended
//
// Phase 1: the threadgroup cooperatively computes the full h vector
// (ffn_dim elements; each thread handles rows tid, tid+tg, ...) into
// threadgroup memory. Phase 2 (after a barrier): each thread produces output
// columns of this tile as weight-scaled down-row dots and atomically adds
// them into out. v1 tradeoff: with more than one tile per pair the h phase is
// recomputed per tile — prefer tile_out >= dim (grid.y == 1) until profiling
// says otherwise. See PORTING-moe.md.
// ---------------------------------------------------------------------------

kernel void expert_mlp_swiglu_q8(
    constant ExpertMlpParams&  p     [[buffer(0)]],
    const device float*        x     [[buffer(1)]], // f32 [n_tokens, dim]
    const device uchar*        gate  [[buffer(2)]], // Q8_0 bank ne={dim, ffn_dim, E}
    const device uchar*        up    [[buffer(3)]], // Q8_0 bank ne={dim, ffn_dim, E}
    const device uchar*        down  [[buffer(4)]], // Q8_0 bank ne={ffn_dim, dim, E}
    const device PairDispatch* pairs [[buffer(5)]], // n_pairs entries
    device atomic_float*       out   [[buffer(6)]], // f32 [n_tokens, dim], pre-zeroed
    uint2 tgid    [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    threadgroup float h[DS5_MAX_FFN];

    const uint pair_idx = tgid.x;
    if (pair_idx >= p.n_pairs) {
        return; // uniform per threadgroup: no thread reaches the barrier
    }
    const PairDispatch pr = pairs[pair_idx];

    // Bank strides (ADR-005 §1): expert e of ne={k, n, E} starts at
    // e * rowBytes(k) * n.
    const uint  row_h_bytes = (p.dim / Q8_BLOCK) * Q8_BLOCK_BYTES;     // gate/up rows
    const uint  row_f_bytes = (p.ffn_dim / Q8_BLOCK) * Q8_BLOCK_BYTES; // down rows
    const ulong gate_stride = ulong(row_h_bytes) * p.ffn_dim;
    const ulong down_stride = ulong(row_f_bytes) * p.dim;

    const device uchar* g_e = gate + ulong(pr.expert) * gate_stride;
    const device uchar* u_e = up   + ulong(pr.expert) * gate_stride;
    const device uchar* d_e = down + ulong(pr.expert) * down_stride;
    const device float* xr  = x    + ulong(pr.token) * p.dim;

    // Phase 1: h = silu(gate(x)) ⊙ up(x), cooperatively across the group.
    for (uint j = tid; j < p.ffn_dim; j += tg_size.x) {
        const float gv = q8_dot_device(g_e + ulong(j) * row_h_bytes, xr, p.dim);
        const float uv = q8_dot_device(u_e + ulong(j) * row_h_bytes, xr, p.dim);
        h[j] = silu_f32(gv) * uv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: out[token, i] += weight * dot(h, down_row_i) for this tile.
    const uint tile0    = tgid.y * p.tile_out;
    const uint tile_end = min(tile0 + p.tile_out, p.dim);
    for (uint i = tile0 + tid; i < tile_end; i += tg_size.x) {
        const float acc = q8_dot_tg(d_e + ulong(i) * row_f_bytes, h, p.ffn_dim);
        atomic_fetch_add_explicit(&out[ulong(pr.token) * p.dim + i],
                                  pr.weight * acc,
                                  memory_order_relaxed);
    }
}
