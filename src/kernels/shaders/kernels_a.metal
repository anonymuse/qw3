// Kernel set A — Metal shaders (dense per-layer ops).
//
// Contains: rmsnorm_f32, rope_f32, matmul_q8_0, matmul_f32, kv_append_f32,
// add_f32. Math is identical to the CPU reference in
// src/kernels/cpu/kernels_a.zig (the permanent comparator); layout rules are
// ADR-005 §1 (ne[0] contiguous, Q8_0 = f32(f16 scale)·i8 q in 34-byte blocks,
// f32 activations and accumulation).
//
// Dispatch story (bindings, params layouts, grid geometry) is frozen for the
// W2 glue in PORTING-kernels-a.md next to this file. All params structs are
// POD with 4-byte-aligned uint/float fields — bind with setBytes at index 0.
//
// Numerics notes:
//   - Accumulation is f32 everywhere. rmsnorm reduces squares in f32 with a
//     threadgroup tree reduction; the row order differs from the CPU's serial
//     sum but stays well inside atol 1e-5 / rtol 1e-4 for dim <= 4096.
//   - rope uses precise::pow/cos/sin so fast-math pipelines cannot bend the
//     angle math past the rope tolerance; rsqrt path in rmsnorm likewise uses
//     precise::sqrt (one divide per row — cost is irrelevant).
//   - Q8_0 scales are read by byte assembly (as_type<half>) so blocks need no
//     alignment guarantee beyond byte addressing.
//   - matmul dots use block-factored FMA (scale · Σ q·x); reassociation vs.
//     the elementwise CPU oracle is inside atol 5e-4 / rtol 2e-3.
//   - No thread writes an element another thread reads: rmsnorm/add may alias
//     out onto x (contract-allowed) because each thread reads x[i] only to
//     produce out[i] (rmsnorm reads the row for the reduction BEFORE the
//     barrier that precedes any write).

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared layout constants
// ---------------------------------------------------------------------------

constant constexpr uint Q8_BLOCK       = 32; // elements per Q8_0 block
constant constexpr uint Q8_BLOCK_BYTES = 34; // half scale + 32 × int8

// Threads per threadgroup for the reduction kernel (power of two, simd-multiple).
constant constexpr uint RMSNORM_TG = 256;

// ---------------------------------------------------------------------------
// Params structs. Must stay byte-identical to what the glue writes into
// buffer(0) (see PORTING-kernels-a.md). All fields 4-byte aligned.
// ---------------------------------------------------------------------------

struct RmsNormParams {   // 12 bytes
    uint  n_rows;        // RmsNormArgs.n_rows
    uint  dim;           // RmsNormArgs.dim
    float eps;           // RmsNormArgs.eps
};

struct RopeParams {      // 20 bytes
    uint  n_tokens;      // RopeArgs.n_tokens
    uint  n_heads;       // RopeArgs.n_heads
    uint  head_dim;      // RopeArgs.head_dim (even)
    float theta;         // RopeArgs.theta
    float freq_scale;    // RopeArgs.freq_scale (1.0 in v1)
};

struct MatmulParams {    // 12 bytes
    uint m;              // MatmulArgs.m — rows of x / out
    uint n;              // MatmulArgs.n — weight rows / out columns
    uint k;              // MatmulArgs.k — row length (multiple of 32 for Q8_0)
};

struct KvAppendParams {  // 24 bytes
    uint n_tokens;       // KvAppendArgs.n_tokens
    uint n_kv_heads;     // KvAppendArgs.n_kv_heads
    uint head_dim;       // KvAppendArgs.head_dim
    uint pos;            // KvAppendArgs.pos (glue asserts pos+n_tokens <= max_ctx)
    uint max_ctx;        // KvAppendArgs.max_ctx
    uint kv_dtype;       // KvAppendArgs.kv_dtype (0=f32, 1=f16, etc.)
};

struct AddParams {       // 4 bytes
    uint n_elems;        // AddArgs.n_elems (glue asserts < 2^32 per dispatch)
};

// ---------------------------------------------------------------------------
// Q8_0 row dot product. `row` points at the first block of one weight row.
// value = f32(f16 scale) · q, accumulated in f32 (block-factored form).
// ---------------------------------------------------------------------------

static inline half q8_scale(const device uchar* block) {
    return as_type<half>(ushort(block[0] | (ushort(block[1]) << 8)));
}

static inline float q8_dot(const device uchar* row,
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

static inline float f32_dot(const device float* row,
                            const device float* x,
                            uint k) {
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) {
        acc = fma(row[i], x[i], acc);
    }
    return acc;
}

// ---------------------------------------------------------------------------
// rmsnorm_f32 — y = x · rsqrt(mean(x²) + eps) · w, rowwise.
//
//   grid = (n_rows, 1, 1) threadgroups, tg = (RMSNORM_TG, 1, 1)
//
// One threadgroup per row. Phase 1: each thread accumulates the squares of a
// strided slice of the row; tree reduction in threadgroup memory. Phase 2:
// every thread scales its slice. Works for any dim (dim < tg leaves threads
// idle; dim > tg strides). out may alias x.
// ---------------------------------------------------------------------------

kernel void rmsnorm_f32(
    constant RmsNormParams& p      [[buffer(0)]],
    const device float*     x      [[buffer(1)]], // f32 [n_rows, dim]
    const device float*     weight [[buffer(2)]], // f32 [dim]
    device float*           out    [[buffer(3)]], // f32 [n_rows, dim], may be x
    uint  row      [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  tg_size  [[threads_per_threadgroup]])
{
    threadgroup float partial[RMSNORM_TG];

    const device float* xr = x   + ulong(row) * p.dim;
    device float*       yr = out + ulong(row) * p.dim;

    // Phase 1: sum of squares, f32.
    float s = 0.0f;
    for (uint i = tid; i < p.dim; i += tg_size) {
        s = fma(xr[i], xr[i], s);
    }
    partial[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // rsqrt(mean + eps) via precise sqrt + divide: matches the CPU reference
    // (1.0 / @sqrt(...)) and is immune to fast-math rsqrt approximation.
    const float inv = 1.0f / precise::sqrt(partial[0] / float(p.dim) + p.eps);

    // Phase 2: scale. Alias-safe: thread t only writes indices it read.
    for (uint i = tid; i < p.dim; i += tg_size) {
        yr[i] = xr[i] * inv * weight[i];
    }
}

// ---------------------------------------------------------------------------
// rope_f32 — in-place NeoX rotation, element i pairs with i + head_dim/2.
//
//   dispatchThreads: grid = (head_dim/2, n_heads, n_tokens)
//   tg = (min(head_dim/2, 64), 1, 1) or any divisor-friendly shape
//
// One thread per (pair i, head h, token t):
//   angle = pos[t] · freq_scale / theta^(2i/head_dim)
//   x'[i]        = x[i]·cos − x[i+h/2]·sin
//   x'[i+h/2]    = x[i]·sin + x[i+h/2]·cos
// Each thread owns both elements of its pair — no races, in place.
// ---------------------------------------------------------------------------

kernel void rope_f32(
    constant RopeParams& p         [[buffer(0)]],
    device float*        x         [[buffer(1)]], // f32 [n_tokens, n_heads, head_dim]
    const device int*    positions [[buffer(2)]], // i32 [n_tokens]
    uint3 gid [[thread_position_in_grid]])
{
    const uint half_dim = p.head_dim / 2;
    const uint i = gid.x; // pair index in [0, head_dim/2)
    const uint h = gid.y; // head
    const uint t = gid.z; // token
    if (i >= half_dim || h >= p.n_heads || t >= p.n_tokens) {
        return; // guards non-uniform / over-provisioned grids
    }

    const float pos   = float(positions[t]);
    const float expnt = float(2 * i) / float(p.head_dim);
    const float angle = pos * p.freq_scale * precise::pow(p.theta, -expnt);
    const float c = precise::cos(angle);
    const float s = precise::sin(angle);

    device float* head = x + (ulong(t) * p.n_heads + h) * p.head_dim;
    const float x0 = head[i];
    const float x1 = head[i + half_dim];
    head[i]            = x0 * c - x1 * s;
    head[i + half_dim] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------------------
// matmul_q8_0 / matmul_f32 — out[m,n] = x[m,k] · Wᵀ.
//
//   dispatchThreads: grid = (n, m, 1); tg = (256, 1, 1) recommended
//
// One thread per output element: thread (ni, mi) computes the full k-length
// dot of x row mi against W row ni. W rows are GGUF layout (ne = {k, n}):
// row ni starts at byte offset ni · rowBytes(k). v1 is deliberately the
// simplest correct shape; see PORTING-kernels-a.md for the planned tiling.
// ---------------------------------------------------------------------------

kernel void matmul_q8_0(
    constant MatmulParams& p [[buffer(0)]],
    const device float*    x [[buffer(1)]], // f32 [m, k]
    const device uchar*    w [[buffer(2)]], // Q8_0, n rows × k cols (34 B / 32 elems)
    device float*          out [[buffer(3)]], // f32 [m, n]
    uint2 gid [[thread_position_in_grid]])
{
    const uint ni = gid.x;
    const uint mi = gid.y;
    if (ni >= p.n || mi >= p.m) {
        return;
    }
    const ulong row_bytes = ulong(p.k / Q8_BLOCK) * Q8_BLOCK_BYTES;
    const float acc = q8_dot(w + ulong(ni) * row_bytes,
                             x + ulong(mi) * p.k,
                             p.k);
    out[ulong(mi) * p.n + ni] = acc;
}

kernel void matmul_f32(
    constant MatmulParams& p [[buffer(0)]],
    const device float*    x [[buffer(1)]], // f32 [m, k]
    const device float*    w [[buffer(2)]], // f32, n rows × k cols
    device float*          out [[buffer(3)]], // f32 [m, n]
    uint2 gid [[thread_position_in_grid]])
{
    const uint ni = gid.x;
    const uint mi = gid.y;
    if (ni >= p.n || mi >= p.m) {
        return;
    }
    const float acc = f32_dot(w + ulong(ni) * p.k,
                              x + ulong(mi) * p.k,
                              p.k);
    out[ulong(mi) * p.n + ni] = acc;
}

// ---------------------------------------------------------------------------
// kv_append_f32 — scatter step K/V into the per-layer cache at `pos`.
//
//   dispatchThreads: grid = (head_dim, n_kv_heads, n_tokens)
//
// One thread per element; copies its K and V lanes. Source layout
// [n_tokens, n_kv_heads, head_dim]; cache layout [n_kv_heads, max_ctx,
// head_dim] (ADR-005 §1). Disjoint destinations — no synchronization needed.
// ---------------------------------------------------------------------------

kernel void kv_append_f32(
    constant KvAppendParams& p       [[buffer(0)]],
    const device float*      k_new   [[buffer(1)]], // f32 [n_tokens, n_kv_heads, head_dim]
    const device float*      v_new   [[buffer(2)]], // f32 [n_tokens, n_kv_heads, head_dim]
    device float*            k_cache [[buffer(3)]], // f32 [n_kv_heads, max_ctx, head_dim]
    device float*            v_cache [[buffer(4)]], // f32 [n_kv_heads, max_ctx, head_dim]
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (d >= p.head_dim || h >= p.n_kv_heads || t >= p.n_tokens) {
        return;
    }
    const ulong src = (ulong(t) * p.n_kv_heads + h) * p.head_dim + d;
    const ulong dst = (ulong(h) * p.max_ctx + (p.pos + t)) * p.head_dim + d;
    k_cache[dst] = k_new[src];
    v_cache[dst] = v_new[src];
}

// kvAppend for f16 cache: write f32 inputs as half-precision to cache.
kernel void kv_append_f16(
    constant KvAppendParams& p       [[buffer(0)]],
    const device float*      k_new   [[buffer(1)]], // f32 [n_tokens, n_kv_heads, head_dim]
    const device float*      v_new   [[buffer(2)]], // f32 [n_tokens, n_kv_heads, head_dim]
    device half*             k_cache [[buffer(3)]], // f16 [n_kv_heads, max_ctx, head_dim]
    device half*             v_cache [[buffer(4)]], // f16 [n_kv_heads, max_ctx, head_dim]
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (d >= p.head_dim || h >= p.n_kv_heads || t >= p.n_tokens) {
        return;
    }
    const ulong src = (ulong(t) * p.n_kv_heads + h) * p.head_dim + d;
    const ulong dst = (ulong(h) * p.max_ctx + (p.pos + t)) * p.head_dim + d;
    k_cache[dst] = half(k_new[src]);
    v_cache[dst] = half(v_new[src]);
}

// ---------------------------------------------------------------------------
// add_f32 — out = x + y elementwise. out may alias either input (each thread
// reads and writes only index gid).
//
//   dispatchThreads: grid = (n_elems, 1, 1); tg = (256, 1, 1)
// ---------------------------------------------------------------------------

kernel void add_f32(
    constant AddParams& p [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* y [[buffer(2)]],
    device float*       out [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n_elems) {
        return;
    }
    out[gid] = x[gid] + y[gid];
}
