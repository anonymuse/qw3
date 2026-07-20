# DS5 Synthetic Metal Soak

This document describes the deterministic synthetic Metal soak exposed by
`tools/run-metal-backend-remote.sh`. Despite the script's historical filename,
it is a synthetic stability check, not a model-serving or performance gate.

## Evidence boundary

Every console report and saved report carries these labels:

```text
evidence_class=synthetic_metal_soak
hardware_interpretable=false
real_model=false
```

The soak repeatedly runs the repository's synthetic `test-metal` suite. A pass
means that command exited successfully in every deterministic repetition.

It does **not** load real model weights, execute GGUF inference, validate model
quantization, perform token generation, or establish throughput, latency,
speedup, acceptance, or cross-machine performance. Any recorded duration is
operational metadata only and must not be interpreted as a hardware benchmark.

## Usage

Run the default 64-iteration soak on the current machine:

```bash
./tools/run-metal-backend-remote.sh
```

Choose an artifact directory and attach a node label:

```bash
./tools/run-metal-backend-remote.sh \
  --node max-2 \
  --output-dir ./bench/results/max-2-metal-soak
```

`--node` records provenance only. The script does not open an SSH connection;
the operator or orchestration layer must start it on the intended machine.
An actual run requires a new output path and refuses to overwrite any existing
file, directory, or symlink, preventing stale logs from being mixed into a run.

Validate parsing and inspect the plan without building or running Metal tests:

```bash
./tools/run-metal-backend-remote.sh --iterations 2 --dry-run
```

For a short local smoke, explicitly reduce the deterministic repetition count:

```bash
./tools/run-metal-backend-remote.sh \
  --iterations 1 \
  --output-dir ./bench/results/local-metal-smoke
```

The default remains 64 iterations for compatibility with existing launchers.

## Prerequisites

- An Apple machine with a working Metal device.
- The repository's supported Zig toolchain.
- The committed synthetic fixtures used by `zig build test-metal`.
- A clean enough workspace for the normal Zig build and test commands.

No external model directory is required.

## Artifacts

The output directory contains:

- `build.log`: raw build-preflight stdout and stderr.
- `iteration-NNN.log`: raw stdout and stderr from each test invocation.
- `iterations.tsv`: per-iteration exit-status and operational-duration index.
- `metal-soak-report.txt`: evidence labels, provenance, counts, and limitations.

The report is written for both build-preflight failure and completed test loops.
It records the full Git commit SHA and only a `git_dirty=true|false|unknown`
value; it does not emit the paths that made a worktree dirty.

The script trusts the exit status of each unpiped `zig build test-metal`
invocation. It does not infer success from output text and does not inject
random failures.

## Interpreting results

- `status=pass`: every synthetic Metal test invocation returned zero.
- `status=fail`: at least one invocation returned nonzero; inspect its raw log.
- `hardware_interpretable=false`: do not compare durations across machines or
  use this result for procurement, optimization, or performance claims.
- `real_model=false`: use a separately specified and evidence-labeled real-model
  run for model-quality or inference claims.

## Relevant code

- Metal context and dispatch glue: `src/metal/metal.zig`
- GPU kernel provider: `src/kernels/gpu/kernels.zig`
- Metal shader sources: `src/kernels/shaders/`
- Test/build wiring: `src/test_metal.zig`, `build.zig`
