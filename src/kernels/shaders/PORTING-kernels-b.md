# Porting notes — kernel set B: causal GQA attention over the KV cache

Audience: the Metal-glue workstream (W2) integrating `kernels_b.metal` behind the
frozen `assertKernelApi` surface (`src/shared/contracts.zig`, ADR-005). The CPU
reference lives in `src/kernels/cpu/kernels_b.zig` and stays the comparator; its
fixture test (the six `op == "attention"` synthetic cases) is the template for the
Metal-vs-oracle gate, and CPU-vs-Metal diff of identical dispatches is the second
gate.

## `gqaAttention` dispatch contract

Implements `contracts.AttnArgs` exactly:

- `q` f32 `[n_tokens, n_q_heads, head_dim]`, post-rope post-qknorm.
- K/V caches: specified dtype (f32 or f16 per `AttnArgs.kv_dtype`) `[n_kv_heads, max_ctx, head_dim]`, separate buffers. Addressing strides by `max_ctx` from the params — **never** by the valid length. Fixture caches happen to be packed (`max_ctx == pos + n_tokens`); real caches are not. **When kv_dtype is f16, the kernel loads half-precision cache values into f32 registers before dot products** (standard load-convert pattern).
- GQA mapping: q head `h` reads kv head `h / (n_q_heads / n_kv_heads)`.
- Causal: query token `t` (absolute position `pos + t`) attends to cache
  positions `0..pos+t` inclusive.
- `scores = scale · (q·k)` pre-softmax; softmax and all accumulation f32.
- `out` f32 `[n_tokens, n_q_heads · head_dim]`.

### Buffer bindings

| index | binding | type | contents |
|---|---|---|---|
| 0 | `params` | constant `GqaAttnParams` | 32 bytes, see below |
| 1 | `q` | `const device float*` | f32 `[n_tokens, n_q_heads, head_dim]` |
| 2 | `k_cache` | `const device float*` or `const device half*` | kv_dtype: f32 or f16 `[n_kv_heads, max_ctx, head_dim]` |
| 3 | `v_cache` | `const device float*` or `const device half*` | kv_dtype: f32 or f16 `[n_kv_heads, max_ctx, head_dim]` |
| 4 | `out` | `device float*` | f32 `[n_tokens, n_q_heads · head_dim]` |

**Dispatch note:** The kernel's buffer-type selection (how it interprets bindings 2 and 3) is determined by `kv_dtype` in the params. Glue provides the matching device buffer type for each binding before encoding.

### Params layout (buffer 0)

```c
struct GqaAttnParams {     // seven uint32 then one float, 32 bytes total
    uint  n_q_heads;       // AttnArgs.n_q_heads
    uint  n_kv_heads;      // AttnArgs.n_kv_heads
    uint  head_dim;        // AttnArgs.head_dim
    uint  pos;             // AttnArgs.pos    (tokens already in cache)
    uint  n_tokens;        // AttnArgs.n_tokens
    uint  max_ctx;         // AttnArgs.max_ctx (cache stride, NOT valid length)
    uint  kv_dtype;        // AttnArgs.kv_dtype (Dtype enum: 0=f32, 1=f16, etc.)
    float scale;           // AttnArgs.scale  (multiplies scores pre-softmax)
};
```

Use `setBytes:length:atIndex:0` — no dedicated buffer needed.

### Grid geometry

```
threadsPerThreadgroup = (128, 1, 1)               // must equal DS5_ATTN_TG
threadgroupsPerGrid   = (n_tokens, n_q_heads, 1)  // dispatchThreadgroups
```

One threadgroup per (query token, q-head); `tgid.x = t`, `tgid.y = h`. The kernel
uses threadgroup barriers, so it is dispatched by **threadgroups** (like
`rmsnorm_f32`), not `dispatchThreads`. Over-provisioning either grid axis is
allowed: out-of-range threadgroups return uniformly before the first barrier.
`DS5_ATTN_TG` must stay a multiple of the simd width (32); if you change it,
change the shader constant and this doc together.

MSL note: thread-position attribute parameters must be all scalar or all
same-width vectors. In this kernel the vector-valued ones
(`threadgroup_position_in_grid`, `threads_per_threadgroup`) are uniformly
`uint3`; `thread_index_in_threadgroup` and the simdgroup indices are inherently
scalar. Keep any position attribute you add `uint3`.

### Glue-side preconditions (reject with `ShapeMismatch`/`UnsupportedDtype` before encoding)

- `head_dim <= 128` (`DS5_MAX_HEAD_DIM` bounds the static threadgroup arrays;
  real models are exactly 128, synthetic 64 — raise the constant only with a
  matching doc edit).
- `n_q_heads % n_kv_heads == 0`; `pos + n_tokens <= max_ctx`; all counts non-zero.
- `kv_dtype` must be f32 (0) or f16 (1); reject others with `UnsupportedDtype`.
- Buffer lengths: `q`/`out` at least `n_tokens·n_q_heads·head_dim·4` bytes,
  each cache at least `n_kv_heads·max_ctx·head_dim·(4 if kv_dtype==f32 else 2)` bytes.

The CPU reference performs the same checks; keeping them glue-side for Metal
preserves identical error behavior across backends.

### Algorithm (why results still match the oracle exactly-in-tolerance)

Score rows reach `pos + n_tokens` positions — up to 32K on real models — which
does not fit the 32 KiB threadgroup memory, so the kernel streams the cache in
`DS5_ATTN_CHUNK = 1024`-position chunks with an **online softmax**
(flash-attention style): running max `m` and denominator `l` carry across
chunks, and the unnormalized output accumulator is rescaled by `exp(m_old −
m_new)` when the max improves. This is algebraically the same stable softmax
(rowmax subtraction) the CPU reference computes — no approximation. Differences
vs the CPU are only floating-point reassociation (per-chunk sums, simd
reductions, `fma` in the dots), well inside the attention tolerance
(atol 1e-4 / rtol 1e-3).

Per chunk: Phase A — thread `j` scores positions `base+j, base+j+128, …`
(full-`head_dim` dot against the q row staged in threadgroup memory); Phase B —
threadgroup max/sum reductions (`simd_max`/`simd_sum` + a cross-simdgroup pass)
update `m`/`l` and exponentiate the chunk in place; Phase C — thread `d`
accumulates output dims `d, d+128, …` over the chunk's V rows. Static
threadgroup usage: scores 4 KiB + q row 512 B + accumulator 512 B + reduction
scratch — ~5.1 KiB total.

### Numerics

- `precise::exp` everywhere in the softmax; fast-math `exp` drifts on
  large-magnitude scores. Keep it even though pipelines default to fast math.
- `exp(-INFINITY) == 0` makes the first chunk need no special case; `ctx_len =
  pos + t + 1 >= 1` always (a token attends to itself), so the running max is
  finite from the first chunk on.
- Output is deterministic run-to-run (no atomics; fixed reduction shape).

### Performance notes (v1 tradeoffs, correctness unaffected)

- Decode (`n_tokens = 1`) launches only `n_q_heads` threadgroups (64 on 235B).
  That under-occupies big GPUs; the natural v2 is splitting `ctx_len` across
  threadgroups with a second-pass merge of `(m, l, o)` partials. Do this only
  after fixtures pass, as a separate kernel or a `grid.z` extension.
- Phase C reads V column-wise (stride `head_dim` between consecutive `j`),
  which is not coalesced; K reads in Phase A are fully coalesced per thread.
  Acceptable for bring-up; revisit with simdgroup matrices later.
- Prefill launches `n_tokens · n_q_heads` threadgroups and parallelizes well as
  is.

### Validation

Fixtures: `tests/fixtures/synthetic/manifest.json`, the attention cases —
- Base cases (kv_dtype f32): `l0/l2/l3_attn_prefill` (`pos = 0`, `n_tokens = 17`) and `l0/l2/l3_attn_decode` (`pos = 16`, `n_tokens = 1`), all with `n_q_heads = 4`, `n_kv_heads = 2`, `head_dim = 64`, `max_ctx = 17`, `scale = 0.125`, tolerance atol 1e-4 / rtol 1e-3.
- f16 KV cache variants: same cases with f16 k_cache/v_cache (fixtures regenerated with f16 tensors, tolerance unchanged).

Tensor roles: `q`, `k_cache`, `v_cache` (inputs), `output` (oracle). The CPU-side fixture test in `kernels_b.zig` is the template. Because fixture caches are packed, they cannot catch a kernel that wrongly strides by valid length — additionally run the CPU reference's "ragged pos" scenario (cache with `max_ctx >` valid length, NaN tail) as a CPU-vs-Metal diff before calling the port done. Test both f32 and f16 variants to verify dtype dispatch logic.
