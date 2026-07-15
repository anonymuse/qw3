// Kernel set B — Metal shaders (attention).
//
// Contains: gqa_attention_f32 — causal grouped-query attention over the frozen
// KV cache layout (f32 [n_kv_heads, max_ctx, head_dim], K and V in separate
// buffers), matching contracts.AttnArgs semantics exactly:
//
//   - q is f32 [n_tokens, n_q_heads, head_dim], post-rope post-qknorm.
//   - Query token t sits at absolute position pos + t and attends to cache
//     positions 0..pos+t inclusive.
//   - GQA mapping: q head h reads kv head h / (n_q_heads / n_kv_heads).
//   - scores = scale * (q . k) pre-softmax; softmax and all accumulation f32.
//   - out is f32 [n_tokens, n_q_heads * head_dim].
//
// Cache addressing ALWAYS strides by max_ctx from the params — fixture caches
// happen to be packed (max_ctx == pos + n_tokens) but real caches are
// allocated at full max_ctx with only pos + n_tokens positions valid.
//
// Dispatch choice (v1): one threadgroup per (query token, q-head) —
// grid = (n_tokens, n_q_heads). Scores for a row can be 32K long on real
// models (32K ctx), which does not fit threadgroup memory, so the kernel
// streams the cache in DS5_ATTN_CHUNK-position chunks with an online
// (flash-attention-style) softmax: running row max m and denominator l are
// carried across chunks and the output accumulator is rescaled by
// exp(m_old - m_new) whenever the max improves. This is algebraically the
// exact stable softmax the CPU reference computes — no approximation — and
// per-chunk reassociation of the sums is well inside the attention tolerance
// (atol 1e-4 / rtol 1e-3).
//
// Numerics notes:
//   - precise::exp everywhere in the softmax; fast-math exp drifts on
//     large-magnitude scores and can break the tolerance.
//   - exp(-INFINITY) == 0 handles the first chunk (m_run starts at -inf)
//     without a special case; ctx_len >= 1 always (a token attends to itself)
//     so m_new is finite from the first chunk on.
//   - All threadgroup reductions (row max, denominator) are simd_max/simd_sum
//     followed by a cross-simdgroup pass over a tiny scratch array; every
//     thread recomputes the combined value so no extra broadcast barrier is
//     needed.
//
// Dispatch story (params layout, binding indices, grid geometry, glue-side
// preconditions) is frozen for W2 in PORTING-attention.md next to this file.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Layout constants — must match PORTING-attention.md and the glue's dispatch.
// ---------------------------------------------------------------------------

// Threads per threadgroup. The glue MUST dispatch exactly this many (see
// PORTING-attention.md); must be a multiple of the simd width (32).
constant constexpr uint DS5_ATTN_TG = 128;

// Cache positions scored per streaming pass. 1024 * 4 B = 4 KiB of the 32 KiB
// threadgroup budget; total static usage ~5.1 KiB (s_sh + q_sh + o_sh + red).
constant constexpr uint DS5_ATTN_CHUNK = 1024;

// Upper bound on head_dim across DS5 targets (real models: 128, synthetic
// fixture model: 64). The glue must reject head_dim > this (see PORTING doc).
constant constexpr uint DS5_MAX_HEAD_DIM = 128;

// Must match the GqaAttnParams the glue writes into buffer(0). Seven uint32
// fields then one float; 32 bytes total, no padding.
struct GqaAttnParams {
    uint  n_q_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  pos;       // tokens already in the cache
    uint  n_tokens;  // query tokens this dispatch
    uint  max_ctx;   // cache stride in positions (NOT the valid length)
    uint  kv_dtype;  // AttnArgs.kv_dtype (0=f32, 1=f16, etc.)
    float scale;     // multiplies scores pre-softmax (1/sqrt(head_dim))
};

// ---------------------------------------------------------------------------
// Threadgroup-wide reductions. Each thread contributes one partial value;
// every thread returns the combined result (no broadcast step). `scratch`
// needs DS5_ATTN_TG / 32 floats.
// ---------------------------------------------------------------------------

static inline float tg_reduce_max(float v,
                                  threadgroup float* scratch,
                                  uint simd_lane, uint simd_id, uint n_simd) {
    v = simd_max(v);
    if (simd_lane == 0) scratch[simd_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = scratch[0];
    for (uint s = 1; s < n_simd; ++s) m = max(m, scratch[s]);
    return m;
}

static inline float tg_reduce_sum(float v,
                                  threadgroup float* scratch,
                                  uint simd_lane, uint simd_id, uint n_simd) {
    v = simd_sum(v);
    if (simd_lane == 0) scratch[simd_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint s = 0; s < n_simd; ++s) acc += scratch[s];
    return acc;
}

// ---------------------------------------------------------------------------
// gqa_attention_f32
//
//   grid  = (n_tokens, n_q_heads, 1) threadgroups; tgid.x = query token t,
//           tgid.y = q head h. Over-provisioning either axis is allowed:
//           out-of-range threadgroups return before the first barrier.
//   tg    = (DS5_ATTN_TG, 1, 1)
//
// Per (t, h): ctx_len = pos + t + 1 cache positions are streamed in chunks.
// Phase A of a chunk: thread j scores positions base+j, base+j+TG, ... into
// threadgroup s_sh (full head_dim dot per thread against the q row cached in
// q_sh). Phase B: online-softmax bookkeeping (chunk max -> m_new, rescale of
// o_sh and l_run, exponentiation in place in s_sh, denominator update).
// Phase C: thread d accumulates output dims d, d+TG, ... as
// o_sh[d] += sum_j s_sh[j] * V[base+j][d]. After the last chunk each owned
// output element is written as o_sh[d] / l_run.
// ---------------------------------------------------------------------------

kernel void gqa_attention_f32(
    constant GqaAttnParams& p       [[buffer(0)]],
    const device float*     q       [[buffer(1)]], // f32 [n_tokens, n_q_heads, head_dim]
    const device float*     k_cache [[buffer(2)]], // f32 [n_kv_heads, max_ctx, head_dim]
    const device float*     v_cache [[buffer(3)]], // f32 [n_kv_heads, max_ctx, head_dim]
    device float*           out     [[buffer(4)]], // f32 [n_tokens, n_q_heads * head_dim]
    // MSL rule: thread-position attribute parameters must be all scalar or
    // all same-width vectors. The vector-valued ones here are uniformly
    // uint3; tid/simd_lane/simd_id are inherently scalar index attributes.
    uint3 tgid      [[threadgroup_position_in_grid]],
    uint3 tptg      [[threads_per_threadgroup]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    const uint tg_size = tptg.x; // tg is (DS5_ATTN_TG, 1, 1)

    threadgroup float q_sh[DS5_MAX_HEAD_DIM];      // this (t, h)'s q row
    threadgroup float o_sh[DS5_MAX_HEAD_DIM];      // unnormalized output accumulator
    threadgroup float s_sh[DS5_ATTN_CHUNK];        // one chunk of scores / weights
    threadgroup float red_max[DS5_ATTN_TG / 32];   // reduction scratch (row max)
    threadgroup float red_sum[DS5_ATTN_TG / 32];   // reduction scratch (denominator)

    const uint t = tgid.x; // query token
    const uint h = tgid.y; // q head
    if (t >= p.n_tokens || h >= p.n_q_heads) {
        return; // uniform per threadgroup: no thread reaches a barrier
    }

    const uint hd      = p.head_dim;
    const uint kh      = h / (p.n_q_heads / p.n_kv_heads); // GQA head mapping
    const uint ctx_len = p.pos + t + 1;                    // causal: 0..pos+t inclusive
    const uint n_simd  = tg_size / 32;

    const device float* q_row  = q + (ulong(t) * p.n_q_heads + h) * hd;
    const device float* k_head = k_cache + ulong(kh) * p.max_ctx * hd; // max_ctx stride
    const device float* v_head = v_cache + ulong(kh) * p.max_ctx * hd;

    // Stage the q row once; zero the output accumulator.
    for (uint d = tid; d < hd; d += tg_size) {
        q_sh[d] = q_row[d];
        o_sh[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -INFINITY; // running row max
    float l_run = 0.0f;      // running softmax denominator (at scale exp(-m_run))

    for (uint base = 0; base < ctx_len; base += DS5_ATTN_CHUNK) {
        const uint n = min(DS5_ATTN_CHUNK, ctx_len - base); // n >= 1

        // Phase A: scores for this chunk. Thread j handles positions
        // base + j, base + j + TG, ...
        for (uint j = tid; j < n; j += tg_size) {
            const device float* k_row = k_head + ulong(base + j) * hd;
            float acc = 0.0f;
            for (uint d = 0; d < hd; ++d) acc = fma(q_sh[d], k_row[d], acc);
            s_sh[j] = acc * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase B: online softmax bookkeeping.
        float local_max = -INFINITY;
        for (uint j = tid; j < n; j += tg_size) local_max = max(local_max, s_sh[j]);
        const float m_chunk = tg_reduce_max(local_max, red_max, simd_lane, simd_id, n_simd);
        const float m_new   = max(m_run, m_chunk);
        const float rescale = precise::exp(m_run - m_new); // 0 on the first chunk

        float local_sum = 0.0f;
        for (uint j = tid; j < n; j += tg_size) {
            const float e = precise::exp(s_sh[j] - m_new);
            s_sh[j] = e;
            local_sum += e;
        }
        const float l_chunk = tg_reduce_sum(local_sum, red_sum, simd_lane, simd_id, n_simd);
        // (the barrier inside tg_reduce_sum also fences the s_sh writes above,
        // so Phase C below may read the whole chunk of weights)

        l_run = l_run * rescale + l_chunk;
        m_run = m_new;

        // Phase C: accumulate V. Thread d owns output dims d, d+TG, ... —
        // the same ownership as the rescale, so no barrier is needed between
        // rescaling o_sh[d] and adding this chunk's contribution to it.
        for (uint d = tid; d < hd; d += tg_size) {
            float acc = o_sh[d] * rescale;
            for (uint j = 0; j < n; ++j) {
                acc = fma(s_sh[j], v_head[ulong(base + j) * hd + d], acc);
            }
            o_sh[d] = acc;
        }
        // Next chunk's Phase A overwrites s_sh; every thread must be done
        // reading this chunk's weights first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // out[t, h*hd + d] = o_sh[d] / l_run
    const float inv_l = 1.0f / l_run;
    device float* out_row = out + ulong(t) * p.n_q_heads * hd + ulong(h) * hd;
    for (uint d = tid; d < hd; d += tg_size) {
        out_row[d] = o_sh[d] * inv_l;
    }
}

// gqaAttention for f16 KV cache: loads half-precision cache into f32 for computation.
kernel void gqa_attention_f16(
    constant GqaAttnParams& p       [[buffer(0)]],
    const device float*     q       [[buffer(1)]], // f32 [n_tokens, n_q_heads, head_dim]
    const device half*      k_cache [[buffer(2)]], // f16 [n_kv_heads, max_ctx, head_dim]
    const device half*      v_cache [[buffer(3)]], // f16 [n_kv_heads, max_ctx, head_dim]
    device float*           out     [[buffer(4)]], // f32 [n_tokens, n_q_heads * head_dim]
    uint3 tgid      [[threadgroup_position_in_grid]],
    uint3 tptg      [[threads_per_threadgroup]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    const uint tg_size = tptg.x;

    threadgroup float q_sh[DS5_MAX_HEAD_DIM];
    threadgroup float o_sh[DS5_MAX_HEAD_DIM];
    threadgroup float s_sh[DS5_ATTN_CHUNK];
    threadgroup float red_max[DS5_ATTN_TG / 32];
    threadgroup float red_sum[DS5_ATTN_TG / 32];

    const uint t = tgid.x;
    const uint h = tgid.y;
    if (t >= p.n_tokens || h >= p.n_q_heads) {
        return;
    }

    const uint hd      = p.head_dim;
    const uint kh      = h / (p.n_q_heads / p.n_kv_heads);
    const uint ctx_len = p.pos + t + 1;
    const uint n_simd  = tg_size / 32;

    const device float* q_row  = q + (ulong(t) * p.n_q_heads + h) * hd;
    const device half*  k_head = k_cache + ulong(kh) * p.max_ctx * hd;
    const device half*  v_head = v_cache + ulong(kh) * p.max_ctx * hd;

    for (uint d = tid; d < hd; d += tg_size) {
        q_sh[d] = q_row[d];
        o_sh[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -INFINITY;
    float l_run = 0.0f;

    for (uint base = 0; base < ctx_len; base += DS5_ATTN_CHUNK) {
        const uint n = min(DS5_ATTN_CHUNK, ctx_len - base);

        for (uint j = tid; j < n; j += tg_size) {
            const device half* k_row = k_head + ulong(base + j) * hd;
            float acc = 0.0f;
            for (uint d = 0; d < hd; ++d) acc = fma(q_sh[d], float(k_row[d]), acc);
            s_sh[j] = acc * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = -INFINITY;
        for (uint j = tid; j < n; j += tg_size) local_max = max(local_max, s_sh[j]);
        const float m_chunk = tg_reduce_max(local_max, red_max, simd_lane, simd_id, n_simd);
        const float m_new   = max(m_run, m_chunk);
        const float rescale = precise::exp(m_run - m_new);

        float local_sum = 0.0f;
        for (uint j = tid; j < n; j += tg_size) {
            const float e = precise::exp(s_sh[j] - m_new);
            s_sh[j] = e;
            local_sum += e;
        }
        const float l_chunk = tg_reduce_sum(local_sum, red_sum, simd_lane, simd_id, n_simd);

        l_run = l_run * rescale + l_chunk;
        m_run = m_new;

        for (uint d = tid; d < hd; d += tg_size) {
            float acc = o_sh[d] * rescale;
            for (uint j = 0; j < n; ++j) {
                acc = fma(s_sh[j], float(v_head[ulong(base + j) * hd + d]), acc);
            }
            o_sh[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = 1.0f / l_run;
    device float* out_row = out + ulong(t) * p.n_q_heads * hd + ulong(h) * hd;
    for (uint d = tid; d < hd; d += tg_size) {
        out_row[d] = o_sh[d] * inv_l;
    }
}
