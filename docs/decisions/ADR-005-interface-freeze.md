# ADR-005: Week-1 Interface Freeze

**Status:** Accepted
**Date:** 2026-07-09
**Depends on:** ADR-001 (model), ADR-002 (kernel strategy), ADR-003 (bring-up model)

## Decision

All parallel kernel/parser/glue workstreams build against the frozen contracts in
[`src/shared/contracts.zig`](../../src/shared/contracts.zig) plus the schemas and semantic
rules in this document. **Changing any frozen item requires an orchestrator decision
recorded as an amendment at the bottom of this file.** A branch that edits a contract to
make itself compile is rejected at merge.

Frozen surfaces:

1. `contracts.zig` — dtypes/quant-block geometry, `TensorDesc` layout convention,
   `ModelConfig`, `Buf`, all kernel argument structs, the `assertKernelApi` /
   `assertGpuApi` / `assertGgufApi` signature sets, DS5T fixture header.
2. Fixture manifest JSON schema (§3 below) and default tolerances (§4).
3. GGUF tensor naming + metadata key mapping (§5).
4. The Qwen3-MoE forward-pass recipe (§6) — the order and semantics of ops per layer.
5. Wire contracts already frozen at M0: `protocol.zig` framing, `activation_packet.zig`
   header (80 bytes), CRC32 checksums.

Conformance is mechanical: each impl module's tests run
`comptime contracts.assertKernelApi(@This(), Ctx)` (or the gguf/gpu variant), and every
kernel test loads golden tensors via `src/shared/fixture.zig` so the pass rule
(`|actual − oracle| ≤ atol + rtol·|oracle|`, every element) is implemented once.

## 1. Layout conventions

- GGUF/GGML dimension order: `ne[0]` is the contiguous in-row dimension. A weight of
  `ne = {k, n}` is `n` rows × `k` cols; `matmul` computes `y[m,n] = x[m,k] · Wᵀ`.
- Quant blocks never straddle rows; `ne[0]` is always a multiple of the block size.
- Q8_0 dequant semantics: `value = f32(f16_scale) · i8_q` — the f16 rounding of the
  scale is part of the definition.
- Activations are f32 device buffers in M2; accumulation f32. (f16 activations are a
  post-correctness optimization, gated by a fixture re-run.)
- Expert banks are single 3-D tensors; expert `e` of `ne = {k, n, n_experts}` starts at
  byte offset `e · rowBytes(k) · n`.
- KV cache per layer: two f16 buffers `[n_kv_heads, max_ctx, head_dim]` (K and V). Attention loads f16 and accumulates in f32.
- All wire and file formats are little-endian / native Apple-Silicon layout (A-12).

## 2. Router semantics (never altered — ADR-001 rule 1)

```
logits = x · W_routerᵀ                (f32)
p      = softmax(logits)              over ALL n_experts, f32
top_k  = k largest p, descending;     ties → LOWER expert id
weights = norm_topk_prob ? top_p / sum(top_p) : top_p
```

Fixture gate: expert IDs match the oracle **100%**; gate weights within atol 1e-5.
The oracle (`make_fixtures.py`) asserts no two selected/boundary probabilities are
within 1e-6 of each other on fixture inputs, so the tie-break rule never actually
decides a fixture case.

## 3. Fixture format

A fixture *set* is a directory under `tests/fixtures/` (e.g. `synthetic/`,
`qwen3-30b-a3b/`) containing one `manifest.json` and flat `.ds5t` tensor files
(binary: 64-byte header per `contracts.FixtureHeader`, then raw row-major data).

```jsonc
{
  "ds5_fixture_version": 1,
  "model": {
    "name": "synthetic-tiny-qwen3moe",
    "config": {                    // field names == contracts.ModelConfig
      "n_layers": 4, "hidden_dim": 256, "n_q_heads": 4, "n_kv_heads": 2,
      "head_dim": 64, "n_experts": 8, "top_k": 4, "expert_ffn_dim": 128,
      "vocab_size": 512, "rms_eps": 1e-6, "rope_theta": 1000000.0,
      "norm_topk_prob": true, "max_ctx": 512
    }
  },
  "generator": { "tool": "make_fixtures.py", "git_commit": "...", "date": "...",
                 "seed": 0, "torch": "...", "transformers": "..." },
  "cases": [
    { "op": "rmsnorm", "name": "l0_attn_norm",
      "params": { "eps": 1e-6 },
      "tensors": { "input": "l0_attn_norm.input.ds5t",
                   "weight": "l0_attn_norm.weight.ds5t",
                   "output": "l0_attn_norm.output.ds5t" },
      "tolerance": { "atol": 1e-5, "rtol": 1e-4 } }
  ],
  "prompts": [
    { "name": "p0", "text": "...", "token_ids": [1,2,3],
      "greedy_tokens": [4,5,6],
      "logits": "p0.logits.ds5t" }   // f32 [vocab, n_positions], all positions
  ]
}
```

Tensor roles per op (all activation tensors f32 unless stated):

| op | params | tensor roles |
|---|---|---|
| `rmsnorm` | `eps` | `input [dim, n_rows]`†, `weight [dim]`, `output` |
| `rope` | `theta`, `n_heads`, `head_dim` | `input [head_dim, n_heads, n_tokens]`, `positions` (i32 `[n_tokens]`), `output` |
| `matmul_quant` | `m,n,k`, `w_dtype` | `input [k, m]`, `weight` (quant blocks, `[k, n]`), `output [n, m]` |
| `attention` | `n_q_heads,n_kv_heads,head_dim,scale,pos` | `q`, `k_cache`, `v_cache` (cache layout §1, first `pos+n_tokens` positions valid), `output` |
| `router` | `n_experts, top_k, norm_topk_prob`, `w_dtype` | `input [dim, n_tokens]`, `weight`, `expert_ids` (i32 `[top_k, n_tokens]`), `gate_weights` (f32 `[top_k, n_tokens]`) |
| `expert_mlp` | `ffn_dim, w_dtype` | `input`, `gate`/`up`/`down` (banks), `expert_ids`, `gate_weights`, `output` (accumulated MoE output, no residual) |
| `layer` | `layer_idx, pos` | `input [dim, n_tokens]`, `positions`, `output` — full transformer block, integration checkpoint |

† dims listed in `ne[]` order (`ne[0]` first).

For quantized-weight cases the oracle computes its golden output from the **dequantized
block values** (identical numbers to what the kernel reads), so quantization error never
pollutes the tolerance; it isolates kernel arithmetic only.

## 4. Default tolerances (f32 oracle vs f32-accumulating Metal kernel)

| op | atol | rtol | extra exact checks |
|---|---|---|---|
| rmsnorm | 1e-5 | 1e-4 | |
| rope | 1e-5 | 1e-4 | |
| matmul_quant (Q8_0) | 5e-4 | 2e-3 | |
| attention | 1e-4 | 1e-3 | |
| router gate weights | 1e-5 | 0 | expert IDs 100% |
| expert_mlp | 1e-3 | 5e-3 | |
| layer | 2e-3 | 1e-2 | |
| e2e logits (same-weights oracle) | 5e-2 | 5e-2 | greedy tokens 100% |

Greedy sampling rule: argmax over logits, ties → lowest token id. E2E logits tolerance
is a hard gate only when oracle and engine use identical weight values (synthetic, or
GGUF-sourced dequant oracle); vs a bf16 HF oracle it is diagnostic and only greedy
tokens gate. Tolerance retuning = amendment to this ADR.

## 5. GGUF mapping (qwen3moe, llama.cpp conventions)

Tensor names (`{i}` = layer):

| GGUF name | role |
|---|---|
| `token_embd.weight` | embeddings `[hidden, vocab]` |
| `output_norm.weight`, `output.weight` | final norm, lm_head (not tied) |
| `blk.{i}.attn_norm.weight` | pre-attention RMSNorm |
| `blk.{i}.attn_q.weight` / `attn_k.weight` / `attn_v.weight` / `attn_output.weight` | projections |
| `blk.{i}.attn_q_norm.weight` / `attn_k_norm.weight` | per-head Q/K RMSNorm `[head_dim]` |
| `blk.{i}.ffn_norm.weight` | pre-MoE RMSNorm |
| `blk.{i}.ffn_gate_inp.weight` | router `[hidden, n_experts]` |
| `blk.{i}.ffn_gate_exps.weight` / `ffn_up_exps.weight` | expert banks `[hidden, ffn, n_experts]` |
| `blk.{i}.ffn_down_exps.weight` | expert bank `[ffn, hidden, n_experts]` |

Metadata keys (`qwen3moe.` prefix): `block_count`, `embedding_length`,
`attention.head_count`, `attention.head_count_kv`, `attention.key_length`,
`attention.layer_norm_rms_epsilon`, `rope.freq_base`, `expert_count`,
`expert_used_count`, `expert_feed_forward_length`, `context_length`; plus
`tokenizer.ggml.*` for the vocab. `ModelConfig.norm_topk_prob` is true for all Qwen3
MoE (not a GGUF key; verify against HF config at fixture generation).

Expected real-model configs (planning values — **verify at parse**, A-05/A-07):

| field | 30B-A3B-Instruct-2507 | 235B-A22B-Instruct-2507 |
|---|---|---|
| n_layers | 48 | 94 |
| hidden_dim | 2048 | 4096 |
| n_q_heads / n_kv_heads | 32 / 4 | 64 / 4 |
| head_dim | 128 | 128 |
| n_experts / top_k | 128 / 8 | 128 / 8 |
| expert_ffn_dim | 768 | 1536 |
| vocab_size | 151936 | 151936 |

## 6. Forward-pass recipe (Qwen3-MoE, per token batch)

```
x = embed[token]                                  # row dequant, no scaling
for each layer i:
    h  = rmsNorm(x, attn_norm)
    q  = matmul(h, Wq)   → view [n_tokens, n_q_heads, head_dim]
    k  = matmul(h, Wk)   → view [n_tokens, n_kv_heads, head_dim]
    v  = matmul(h, Wv)
    q  = rmsNorm(q, q_norm)   per head (n_rows = n_tokens·n_q_heads, dim = head_dim)
    k  = rmsNorm(k, k_norm)   per head
    rope(q, pos); rope(k, pos)                    # NeoX pairing, full head_dim
    kvAppend(k, v → cache[i])
    a  = gqaAttention(q, cache[i])                # scale = 1/sqrt(head_dim)
    x  = x + matmul(a, Wo)
    h2 = rmsNorm(x, ffn_norm)
    ids, w = routerTopK(h2)                       # §2, host outputs
    x  = x + expertMlpSwiglu(h2, ids, w)          # silu(gate)·up → down, weighted
final: logits = matmul(rmsNorm(x, output_norm), output.weight)
```

The synthetic model (`contracts.SYNTH_TINY`) must stay byte-identical to
`SYNTH_CONFIG` in `tools/make_fixtures.py`. It uses `top_k = 4` of 8 experts so
top-k *selection* is actually exercised (k = n_experts would degenerate); real-model
fixtures pin the untouched 128-choose-8 path.

## 7. Change process

1. Any workstream needing a contract change stops and files the request with the
   orchestrator (in-session or PR comment).
2. Orchestrator decides; if accepted, contracts.zig + this ADR are amended in a single
   commit on the integration branch, and affected workstreams rebase.
3. Fixture regeneration counts as a contract change when it alters manifest schema,
   tolerances, or tensor roles — not when it merely adds cases.

## Amendments

*(none yet)*
