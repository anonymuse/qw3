# llama.cpp routing-telemetry instrumentation (M1 hour-zero task 3)

Produces the routing-telemetry JSON contract in
`docs/runbooks/expert-stats-capture.md` §2, consumed unmodified by
`tools/expert_stats/merge_stats.py`. This is an **offline oracle** per ADR-002:
it links against llama.cpp to produce a data file on a cluster node; it is
never linked into, or invoked by, the DS5 runtime (`src/`).

## Why this needs instrumentation at all

Stock llama.cpp has no CLI flag that dumps per-expert routing counts. It does,
however, already expose the per-layer MoE routing tensors by name to the
`ggml_backend_sched_eval_callback` hook — the same mechanism
`examples/eval-callback` and `tools/imatrix` use to read intermediate
activations. `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`) names
the tensor holding each token's top-8 expert ids `"ffn_moe_topk"`; the gate
weight tensor is named `"ffn_moe_weights"` pre-normalization and
`"ffn_moe_weights_norm"` post-normalization (only emitted when the arch sets
`norm_w=true` — Qwen3-MoE always does, per `src/models/qwen3moe.cpp`, so it is
always present here). `llama_context`'s graph-build code suffixes every named
tensor with its layer index (`ggml_format_name(cur, "%s-%d", name, il)`,
`src/llama-context.cpp`) — so at runtime they appear as `"ffn_moe_topk-0"`,
`"ffn_moe_weights_norm-0"`, `"ffn_moe_topk-1"`, ... one pair per layer. No
llama.cpp core patch is required — this is a hook, built as an ordinary
example program.

Verified against `ggml-org/llama.cpp` @ `6eddde06a4f25d55d538b5d15628dcc2b6882147`
(2026-07-13). Tensor names and the `cb_eval` callback shape are part of the
same debug/instrumentation surface `eval-callback` and `imatrix` depend on, so
they are reasonably stable, but **re-check `grep -n 'ffn_moe_topk\|ffn_moe_weights' src/llama-graph.cpp`
on the commit you actually build** before trusting the output — names have
shifted before (e.g. the imatrix output format default changed from `.dat` to
`.gguf`).

## Install into a llama.cpp checkout

```sh
cp telemetry-capture.cpp /tmp/llama.cpp/examples/telemetry-capture.cpp
mkdir -p /tmp/llama.cpp/examples/telemetry-capture
mv /tmp/llama.cpp/examples/telemetry-capture.cpp /tmp/llama.cpp/examples/telemetry-capture/telemetry-capture.cpp
cp CMakeLists.txt /tmp/llama.cpp/examples/telemetry-capture/CMakeLists.txt
```

Register the new example directory (one line, next to `eval-callback` in
`examples/CMakeLists.txt`):

```sh
cd /tmp/llama.cpp
sed -i.bak '/add_subdirectory(eval-callback)/a\
    add_subdirectory(telemetry-capture)
' examples/CMakeLists.txt
```

(Or edit `examples/CMakeLists.txt` by hand and add
`add_subdirectory(telemetry-capture)` on its own line.)

## Build

```sh
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j --target llama-telemetry-capture
# Binary: build/bin/llama-telemetry-capture
```

## Run — see runbook §2 for the full command with real paths/hashes

```sh
build/bin/llama-telemetry-capture \
  -m "$MODEL" \
  -f "$CORPUS" \
  -o bench/results/expert_stats/telemetry-<date>-run1.json \
  --source-id "telemetry-<date>-run1" \
  --model-id "Qwen3-235B-A22B-Instruct-2507" \
  --model-hash "sha256:<hex from shasum -a 256 on $MODEL>" \
  --corpus-id "router-calibration-corpus-v1" \
  --corpus-hash "sha256:<hex from shasum -a 256 on $CORPUS>" \
  --ngl 99
```

`--n-layer`/`--n-expert`/`--top-k` default to 94/128/8 (Qwen3-235B-A22B,
ADR-005 constants) — leave them at the default; if the `gguf_dump.py` sanity
check in runbook §0 doesn't show `94`/`128`/`8`, stop, don't override these
flags to force a mismatched artifact through.

`--chunks N` caps how many `n_ctx`-sized windows are processed (same idea as
`llama-imatrix --chunks`); omit it to process the whole corpus file. Each
window is decoded as an independent context (KV cache cleared between
windows) — the same fixed-size-chunk compromise `tools/perplexity` and
`tools/imatrix` already make; routing telemetry does not carry cross-window
attention context. Record this in the run log alongside the other documented
compromises (Q8_0-as-KLD-reference, etc.).

## What it measures, precisely

- `activation_counts[layer][expert]`: number of tokens (across the whole
  corpus, or the `--chunks`-limited prefix) for which this expert was in the
  layer's top-8 selection (`"ffn_moe_topk-<il>"`).
- `gate_weight_sums[layer][expert]`: sum, over those activations, of the
  **post-normalization** gate weight (`"ffn_moe_weights_norm-<il>"` — the
  value `build_moe_ffn` actually multiplies each selected expert's output by,
  after the top-8 softmax probabilities are renormalized to sum to 1). Qwen3-MoE
  runs with `w_scale = hparams.expert_weights_scale`; if that scale is
  non-default (1.0) for the 235B config, `"ffn_moe_weights_scaled-<il>"` would
  be the truly final value — check `expert_weights_scale` in the GGUF metadata
  (`gguf_dump.py` output) before assuming `ffn_moe_weights_norm` is the last
  word; swap the tracked tensor name in `telemetry_cb_eval` if so.
  `merge_stats.py` divides this sum by `activation_count` to get
  `mean_gate_weight`.
