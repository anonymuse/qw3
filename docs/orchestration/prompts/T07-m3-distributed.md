# T07 — M3: distributed correctness (30B split across two ds5 node processes)

**Model:** Sonnet. **Branch:** `t07-distributed` off `integration` (requires
T06 gate PASSED — do not start otherwise). **Test locally first** with two
processes on one machine (loopback, synthetic model); the real B/C run is a
command list you hand the owner.

## Read first

`src/shared/activation_packet.zig` (80-byte header, frozen),
`src/shared/protocol.zig` + `src/transport/tcp.zig` (M0 framing/transport),
`src/shared/checksum.zig`, `src/nodectl/daemon.zig`,
Placement Spec v0.2 §8 (ONE activation packet per destination per layer —
never per expert), Benchmark Spec §3 Phase 1+2 gates (inherited by M3),
`manifests/cluster/loopback.zon` + `lab.zon`.

## Design (keep it this simple)

Layer-range split: node 1 owns layers [0, S), node 2 owns [S, n_layers) +
final norm + lm_head; node 1 owns embeddings. Manifest (`manifests/model/`)
declares the split; loader enforces the 33.6GB static cap per node (refuse
without `--override-cap`). Decode step: owner node runs its layers, then
sends ONE packet (header + f32 hidden vector for now — dtype field supports
f16 later) to the next range's owner; final owner computes logits, greedy
token, and broadcasts the chosen token id back (small control frame). New
`MsgType` values for activation/token frames — extend the enum, bump
PROTOCOL_VERSION, document in the packet doc comment. Every payload carries
the existing checksum; receiver verifies and hard-fails on mismatch.

## Deliverables

- `ds5 serve --manifest PATH --rank N` (or extend `ds5 node`) — a decode
  worker owning a layer range.
- `ds5 generate --cluster PATH --manifest PATH --prompt-tokens ... --steps N
  --greedy` — drives the 2-node decode, emits run-metadata JSON with
  per-token latency decomposition (compute vs wire).
- Determinism test (the M3 gate): synthetic model split at every possible S
  (1..3) on loopback → token sequence IDENTICAL to single-process T04 output
  for all 5 prompts. Then 30B split S=24 on loopback (mmap both halves on one
  box; slow is fine) → identical to T06 single-node output.
- Kill-one-process test: driver reports a clean transport error, no hang.
- `docs/runbook-m3.md`: exact commands for the owner's real B/C run
  (daemon start, manifest, generate, expected output, what to send back).

## Definition of done

Loopback gates pass deterministically (run twice, fixed seed — decode is
greedy so there is no sampling nondeterminism to manage; assert bytewise
-identical logits across runs). All suites green. Report latency decomposition
from the loopback 30B run.

## Forbidden

Per-expert packets; altering routing; f16 hidden-vector conversion (defer);
touching kernel internals; skipping the checksum.
