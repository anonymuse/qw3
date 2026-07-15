# T05 — M2b: GPU forward pass (Metal) matches CPU forward pass

**Model:** Sonnet. **Branch:** `t05-gpu-forward` off `integration` (requires
T04 merged). Needs an Apple Silicon GPU (any — dev M5 Air OK).

## Goal

The T04 `Engine`, instantiated with the Metal glue `Ctx`
(`src/metal/metal.zig`) and a GPU kernel provider, produces the same synthetic
-model results as the CPU instantiation: all 5 fixture prompts within
tolerance, greedy tokens exact, and GPU-vs-CPU residual-stream trace diff
within ADR-005 §4 tolerances at every layer.

## Read first

`src/metal/metal.zig` (glue API + existing proof-kernel dispatch examples),
all three `src/kernels/shaders/PORTING-*.md` (frozen dispatch contracts:
params layouts, binding indices, grid geometry — implement EXACTLY these),
`docs/orchestration/HANDOFF.md` §5 (A-09: batch dispatches per command buffer;
buffer landmines), T04's engine + trace hook.

## Deliverables

- `src/kernels/gpu/kernels.zig`: kernel provider passing
  `comptime contracts.assertKernelApi(@This(), metal.Ctx)`; each op encodes
  into the glue's current batch per its PORTING doc. Router runs on CPU
  (frozen decision, PORTING-moe.md §1): download normed activations, run the
  CPU routerTopK, build PairDispatch lists host-side. `kvAppend`/`add`/`rope`
  etc. per kernels_a/b/c shaders (compile via glue `addLibrary`/`pipeline`).
- Command-buffer batching: everything between router boundaries in one batch;
  target ≤3 submits per layer, fewer if possible. Record per-layer
  gpuElapsedNs into the run-metadata JSON of `ds5 run`.
- Per-shader GPU-vs-fixture tests in a `zig build test-gpu` step (pattern:
  `test-metal` in build.zig): every fixture case that kernels A/B/C pass on
  CPU must pass through the real shader dispatch path.
- `ds5 run --backend metal` flag.

## Definition of done

`zig build test-metal` and `zig build test-gpu` green; 5/5 prompts logits
within tolerance and greedy exact on Metal; layerwise CPU-vs-GPU trace max
diff reported per layer. `zig build test` (CPU) untouched and green.

## Known first suspects when a shader diverges from its CPU twin

Threadgroup memory races (missing barrier), non-`precise::` math, f16
rounding in dequant (contract: `f32(f16 scale) * i8 q` — promote BEFORE
multiply), grid over-provisioning writing out of bounds, params struct
padding mismatch (MSL packs differently — PORTING docs specify exact layouts;
verify with `@sizeOf` on the Zig side vs a static_assert comment in MSL).

## Forbidden

Editing contracts or CPU kernels; loosening tolerances; making `zig build
test` require a GPU.
