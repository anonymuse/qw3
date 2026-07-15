# Porting notes — kernel set A: rmsNorm, rope, matmul (Q8_0/f32), kvAppend, add

Audience: the Metal-glue workstream (W2) integrating `kernels_a.metal` behind the
frozen `assertKernelApi` surface (`src/shared/contracts.zig`, ADR-005). The CPU
reference lives in `src/kernels/cpu/kernels_a.zig` and stays the comparator; the
fixture tests there (rmsnorm 6, rope 3, matmul_quant 4 synthetic cases) are the
template for the Metal-vs-oracle gate, and CPU-vs-Metal diff of identical dispatches
is the second gate.

Conventions shared by all six functions:

- Params structs are POD, all fields 4-byte-aligned `uint`/`float`. Bind with
  `setBytes:length:atIndex:0` — no dedicated buffer needed. Layouts below must stay
  byte-identical to the structs at the top of `kernels_a.metal`.
- Tensor buffers bind `contracts.Buf` (handle + byte offset) at the indices listed;
  all activations f32, weights in artifact dtype, accumulation f32 (ADR-005 §1).
- Grid shapes below use `dispatchThreads` (non-uniform grids, fine on Apple GPUs).
  Every kernel bounds-checks, so over-provisioned grids are safe; only `rmsnorm_f32`
  uses threadgroup barriers, and it is dispatched by threadgroups, not threads.
- `matmul` with any `w_dtype` other than `f32`/`q8_0`, and unsupported ops
  generally, are rejected glue-side with `UnsupportedDtype` — there is no dtype
  switch in-shader.

## rmsnorm_f32

`y = x · rsqrt(mean(x²) + eps) · w`, rowwise. Serves the hidden-dim norms and the
per-head Q/K norm (`dim = head_dim`, `n_rows = n_tokens·n_heads`). `out` may alias
`x` (the forward recipe uses both forms).

| index | binding | type | contents |
|---|---|---|---|
| 0 | `params` | constant `RmsNormParams` | 12 B: `uint n_rows, uint dim, float eps` |
| 1 | `x` | `const device float*` | f32 `[n_rows, dim]` |
| 2 | `weight` | `const device float*` | f32 `[dim]` |
| 3 | `out` | `device float*` | f32 `[n_rows, dim]`; may be the same Buf as `x` |

```
threadsPerThreadgroup = (256, 1, 1)      // must equal RMSNORM_TG in the shader
threadgroupsPerGrid   = (n_rows, 1, 1)   // dispatchThreadgroups, one group per row
```

One threadgroup per row: strided square-sum, tree reduction in threadgroup memory,
then strided scale. The reduction order differs from the CPU's serial sum;
tolerance (atol 1e-5 / rtol 1e-4) absorbs it up to dim 4096. `tg_size` must be a
power of two (the reduction halves it) — keep 256.

## rope_f32

In-place NeoX rotation, element `i` pairs with `i + head_dim/2`, angle
`pos · freq_scale / theta^(2i/head_dim)`. `freq_scale` is 1.0 in v1 but is part of
the params so the glue passes `RopeArgs.freq_scale` through verbatim.

| index | binding | type | contents |
|---|---|---|---|
| 0 | `params` | constant `RopeParams` | 20 B: `uint n_tokens, uint n_heads, uint head_dim, float theta, float freq_scale` |
| 1 | `x` | `device float*` | f32 `[n_tokens, n_heads, head_dim]`, rotated in place |
| 2 | `positions` | `const device int*` | i32 `[n_tokens]` absolute positions |

```
dispatchThreads: grid = (head_dim/2, n_heads, n_tokens)
threadsPerThreadgroup = (min(head_dim/2, 64), 1, 1)   // any shape works; bounds-checked
```

One thread per (pair, head, token); each thread owns both elements of its pair, so
in-place is race-free. `head_dim` must be even (true for all targets: 64/128).
Uses `precise::pow/cos/sin` — keep even under fast-math pipeline options.

## matmul_q8_0 / matmul_f32

`out[m,n] = x[m,k] · Wᵀ`; W is GGUF layout ne = `{k, n}` (n rows of k contiguous
elements), row `ni` at byte offset `ni · rowBytes(k)` where `rowBytes(k) = k/32·34`
for Q8_0 and `k·4` for f32. Q8_0 dequant is exactly `f32(f16 scale) · i8 q`.
Glue selects the pipeline by `MatmulArgs.w_dtype` (`.q8_0` → `matmul_q8_0`,
`.f32` → `matmul_f32`, anything else → `UnsupportedDtype` before encoding).

| index | binding | type | contents |
|---|---|---|---|
| 0 | `params` | constant `MatmulParams` | 12 B: `uint m, uint n, uint k` |
| 1 | `x` | `const device float*` | f32 `[m, k]` |
| 2 | `w` | `const device uchar*` (q8_0) / `const device float*` (f32) | n rows × k cols |
| 3 | `out` | `device float*` | f32 `[m, n]` |

```
dispatchThreads: grid = (n, m, 1)
threadsPerThreadgroup = (256, 1, 1)      // clamp x-extent to n for tiny n
```

One thread per output element, full-k dot per thread. `k` must be a multiple of 32
for Q8_0 (holds for every DS5 tensor; glue may assert). Deliberately the simplest
correct v1 — decode-time matvecs (m = 1) keep the GPU busy via n (256…151936 rows).
Post-correctness optimization candidates (do NOT change semantics): simdgroup
reductions per row, threadgroup x-tiling to reuse activations, `char4`/`float4`
vector loads. Any rework re-runs the matmul_quant fixtures (atol 5e-4 / rtol 2e-3),
which absorb dot-product reassociation.

## kvAppend (kv_dtype dispatch)

Scatter this step's K and V `[n_tokens, n_kv_heads, head_dim]` into the frozen
per-layer cache layout `[n_kv_heads, max_ctx, head_dim]` at positions
`pos..pos+n_tokens-1`. Glue asserts `pos + n_tokens <= max_ctx` before encoding
(the CPU reference returns ShapeMismatch; the shader has no error channel).

**Dispatch by `KvAppendArgs.kv_dtype`:** The kernel reads this field at runtime and
selects the cache write path (f32 or f16). Both paths accept the same k_new/v_new (f32
input), but write cache buffers in the specified dtype. Glue provides the appropriate
Metal device buffer type to each binding based on kv_dtype before encoding.

| index | binding | type | contents |
|---|---|---|---|
| 0 | `params` | constant `KvAppendParams` | 24 B: `uint n_tokens, n_kv_heads, head_dim, pos, max_ctx, uint kv_dtype` |
| 1 | `k_new` | `const device float*` | f32 `[n_tokens, n_kv_heads, head_dim]` |
| 2 | `v_new` | `const device float*` | f32 `[n_tokens, n_kv_heads, head_dim]` |
| 3 | `k_cache` | `device float*` or `device half*` | kv_dtype: f32 or f16 `[n_kv_heads, max_ctx, head_dim]` |
| 4 | `v_cache` | `device float*` or `device half*` | kv_dtype: f32 or f16 `[n_kv_heads, max_ctx, head_dim]` |

```
dispatchThreads: grid = (head_dim, n_kv_heads, n_tokens)
threadsPerThreadgroup = (64, 1, 1)
```

Pure elementwise copy to disjoint destinations; no synchronization. When kv_dtype is
f16, kernels convert f32 input to f16 before write (standard precision loss); bit-exact
vs. CPU reference for matching dtype (validate with exact compare, atol 0).

## add_f32

`out = x + y` elementwise; `out` may alias either input.

| index | binding | type | contents |
|---|---|---|---|
| 0 | `params` | constant `AddParams` | 4 B: `uint n_elems` |
| 1 | `x` | `const device float*` | f32 `[n_elems]` |
| 2 | `y` | `const device float*` | f32 `[n_elems]` |
| 3 | `out` | `device float*` | f32 `[n_elems]`; may be `x` or `y` |

```
dispatchThreads: grid = (n_elems, 1, 1)
threadsPerThreadgroup = (256, 1, 1)
```

`AddArgs.n_elems` is u64 in the contract; the params field is u32. Glue asserts
`n_elems < 2^32` per dispatch (worst real case is n_tokens·hidden_dim — orders of
magnitude below) or splits into multiple dispatches with adjusted buffer offsets.
Bit-exact vs. CPU (single f32 add per element); validate exact.

## Validation

- Fixtures: `tests/fixtures/synthetic/manifest.json` — `rmsnorm` ×6
  (`l{0,2,3}_attn_norm`, `l{0,2,3}_q_norm`; atol 1e-5 / rtol 1e-4), `rope` ×3
  (`l{0,2,3}_rope_q`; atol 1e-5 / rtol 1e-4; copy input into a device buffer, run
  in place, compare), `matmul_quant` ×4 (`l{0,2,3}_attn_q`, `lm_head`; atol 5e-4 /
  rtol 2e-3). Tensor roles and ne[] orders are exactly as consumed by the fixture
  tests in `kernels_a.zig` — reuse them as the template.
- kvAppend/add have no fixture cases; port the direct unit tests from
  `kernels_a.zig` (sentinel-filled cache round-trip; add vs scalar loop) and
  compare exactly.
- Second gate: CPU-vs-Metal diff of identical dispatches — rmsnorm/rope/matmul
  within fixture tolerances, kv_append/add bit-exact.
