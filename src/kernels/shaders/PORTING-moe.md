# Porting notes — kernel set C (MoE): router top-k + expert SwiGLU MLP

Audience: the Metal-glue workstream (W2) integrating `kernels_c.metal` behind the
frozen `assertKernelApi` surface (`src/shared/contracts.zig`, ADR-005). The CPU
reference lives in `src/kernels/cpu/kernels_c.zig` and stays the comparator.

## 1. Router: CPU-only, permanently

`routerTopK` has **no Metal kernel** and does not need one:

- The contract already lands results in **host** slices (`RouterArgs.out_ids: []u16`,
  `out_weights: []f32`) — the scheduler and activation-packet path consume them from
  CPU memory, so a GPU version would round-trip anyway.
- Per token the work is one `n_experts × dim` matvec (128 × 4096 for 235B) plus a
  softmax over 128 floats and a k=8 selection — microseconds on a P-core.
- ADR-001 rule 1 / ADR-005 §2 make the top-k semantics the most correctness-critical
  code in the project. Keeping the single implementation on the CPU removes an entire
  class of GPU-reassociation and cross-backend-divergence bugs.

Integration: the provider module's `routerTopK` should call the CPU implementation
directly (it needs the router weight bytes host-visible, which holds — weight buffers
are wrapped mmapped GGUF bytes via `bufferFromBytes`). Revisit only if profiling
shows the router on the critical path (it will not be; the expert MLP dominates FLOPs
by ~3 orders of magnitude).

Exact frozen semantics (never altered): logits `x·Wᵀ` in f32 → softmax over ALL
`n_experts` in f32 → top-k by probability descending, ties → LOWER expert id → iff
`norm_topk_prob`, divide the k weights by their sum.

## 2. `expert_mlp_swiglu_q8` dispatch contract

Computes, for every `PairDispatch {token, expert, weight}`:

```
out[token] += weight * down( silu(gate(x[token])) ⊙ up(x[token]) )
```

`out` must be zeroed before the dispatch (frozen contract: caller pre-zeroes).

### Buffer bindings

| index | binding | type | contents |
|---|---|---|---|
| 0 | `params` | constant `ExpertMlpParams` | 20 bytes, see below |
| 1 | `x` | `const device float*` | f32 `[n_tokens, dim]` activations |
| 2 | `gate` | `const device uchar*` | Q8_0 bank, ne = `{dim, ffn_dim, n_experts}` |
| 3 | `up` | `const device uchar*` | Q8_0 bank, ne = `{dim, ffn_dim, n_experts}` |
| 4 | `down` | `const device uchar*` | Q8_0 bank, ne = `{ffn_dim, dim, n_experts}` |
| 5 | `pairs` | `const device PairDispatch*` | `n_pairs` × 12-byte records |
| 6 | `out` | `device atomic_float*` | f32 `[n_tokens, dim]`, **pre-zeroed** |

`pairs` is a host slice in `ExpertMlpArgs`; the glue uploads it into a transient
device buffer (`n_pairs * 12` bytes, `PairDispatch` is a frozen 12-byte extern
struct — upload with a straight memcpy, no repacking).

Expert bank addressing matches ADR-005 §1: expert `e` of a bank with ne =
`{k, n, E}` starts at byte offset `e · rowBytes(k) · n`, where `rowBytes(k) =
k/32 · 34` for Q8_0. Pass bank base pointers; the kernel computes strides from
`dim`/`ffn_dim`. `dim` and `ffn_dim` must be multiples of 32 (true for all targets).

### Params layout (buffer 0)

```c
struct ExpertMlpParams {   // all uint32, 20 bytes total
    uint dim;              // ExpertMlpArgs.dim
    uint ffn_dim;          // ExpertMlpArgs.ffn_dim
    uint n_experts;        // ExpertMlpArgs.n_experts
    uint n_pairs;          // pairs.len
    uint tile_out;         // output columns per grid.y step (see geometry)
};
```

Use `setBytes:length:atIndex:0` — no dedicated buffer needed.

### Grid geometry

```
threadsPerThreadgroup = (256, 1, 1)            // TG_THREADS; simd-multiple
threadgroupsPerGrid   = (n_pairs, ceil(dim / tile_out), 1)
```

One threadgroup per (pair, output tile):

1. **Phase 1** — the group cooperatively computes the whole
   `h[ffn_dim] = silu(gate·x) ⊙ (up·x)` vector into threadgroup memory
   (thread `t` handles FFN rows `t, t+256, …`). `DS5_MAX_FFN = 1536` bounds the
   static threadgroup array (6 KiB; raise the constant if a future target exceeds it).
2. **barrier**
3. **Phase 2** — thread `t` produces output columns `tile0+t, tile0+t+256, …` of its
   tile as `weight * dot(h, down_row_i)` and adds them to `out` with
   `atomic_fetch_add_explicit(..., memory_order_relaxed)`.

**Recommended v1 launch: `tile_out = dim` (grid.y == 1).** Phase 1 is recomputed per
tile, so multiple tiles per pair only pay off if a single-tile launch under-occupies
the GPU (tiny batches). Correctness is identical either way.

Over-provisioning grid.x is allowed: threadgroups with `tgid.x >= n_pairs` return
before the barrier uniformly (whole group exits — no barrier divergence).

### Why atomics on `out`

Different pairs can target the same token row (every token has `top_k` pairs), and
pairs run in independent threadgroups with no ordering. Float atomic adds make the
accumulation correct without a reduction pass. Requirements:

- `device atomic_float` + `atomic_fetch_add_explicit`: MSL 2.4+ (Metal 3), Apple7+
  GPU family — satisfied by every DS5 node (M-series).
- Consequence: **summation order is nondeterministic across runs.** Tolerances
  (atol 1e-3 / rtol 5e-3) absorb this. If bit-exact reruns are ever needed, switch to
  a deterministic variant: sort pairs by token and dispatch one threadgroup per
  (token, tile) iterating that token's pairs serially in slot order.

### Numerics

- All accumulation in f32; weights dequantized in-shader as
  `f32(f16 scale) · i8 q` (frozen Q8_0 rule). Scales are read by byte assembly
  (`as_type<half>` of two uchars), so no alignment requirement beyond byte offsets.
- `silu` uses `precise::exp`; keep it even if the pipeline compiles with fast math
  (default), otherwise fast `exp` can drift on large-magnitude gate values.
- Dot products use block-factored FMA (`scale · Σ q·x`); reassociation vs. the
  elementwise CPU oracle is well inside the expert_mlp tolerance.

### Validation

Fixtures: `tests/fixtures/synthetic/manifest.json`, the three `op == "expert_mlp"`
cases (`l0/l2/l3_experts`): Q8_0 banks ne `{256,128,8}` / `{128,256,8}`, 17 tokens,
top_k 4 → 68 pairs, tolerance atol 1e-3 / rtol 5e-3. Build the pair list token-major
from the `expert_ids`/`gate_weights` tensors (ne = `{top_k, n_tokens}`, ne[0]
fastest): pair `t·top_k + j` = `{token: t, expert: ids[t·top_k+j], weight:
w[t·top_k+j]}` — exactly what the CPU-side fixture test in `kernels_c.zig` does;
reuse it as the template. CPU-vs-Metal diff of the same dispatch is the second gate.
