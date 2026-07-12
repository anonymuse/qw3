# T01 — Finish the Zig GGUF parser (workstream W1)

**Model:** Haiku. **Branch:** `w1-gguf-parser` (exists; based on
`d1-interface-freeze`). Work in an isolated worktree on that branch.

## Context

A previous agent wrote most of a GGUF parser and was killed mid-way through
fixing error-path tests ("the file tail got garbled — fixing the error-path
test section"). Your job is to bring the branch to done. Inspect what exists
before writing anything: `git log --oneline`, `git status`, then read the
parser source (expected under `src/gguf/`).

## Read first (mandatory)

1. `src/shared/contracts.zig` — sections 1–3 and `assertGgufApi` (§5). Your
   `Model` type must pass `comptime contracts.assertGgufApi(@This())` in a test.
2. `docs/decisions/ADR-005-interface-freeze.md` §5 (GGUF key mapping table for
   qwen3moe → ModelConfig) and §7 (change process — you cannot edit contracts).
3. `docs/orchestration/HANDOFF.md` §1 and §5.

## Definition of done

- `Model.open` mmaps the file (raw libc via `src/shared/sys.zig` pattern —
  never `std.Io`); GGUF v3 header, metadata KV table, tensor index parsed;
  `TensorView.data` points into the mmap (zero-copy), 32-byte-aligned per GGUF
  spec (verify alignment key, default 32).
- Metadata getters (`metaU32/U64/F32/Bool/Str`) return null on missing key or
  wrong type; `config()` builds `ModelConfig` from the ADR-005 §5 key table
  and errors with `GgufError.MissingKey`/`BadMetadata` on absent/invalid keys.
- Dtype coverage for the tensor index: at minimum f32, f16, q8_0, q4_k, q2_k,
  iq2_xxs/xs/s — the index only needs dtype + shape + offset (byte math via
  `Dtype.rowBytes`); no dequantization in this task.
- Malformed-input tests: truncated header, bad magic, unsupported version,
  string running past EOF, tensor data offset past EOF. Parser must return
  errors, never crash or overflow (fuzz-ish: test each field boundary).
- A test builds a tiny synthetic GGUF **in memory or in a temp file from Zig**
  (do not require Python at test time) with known keys/tensors and round-trips
  it through `Model`.
- `zig build test` green from repo root with your module wired into
  `src/main.zig`'s test block (`_ = @import("gguf/gguf.zig");` — adjust path
  to reality). If a real 30B GGUF exists at
  `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/`, add an opt-in check
  (skip-if-absent) that opens it, prints tensor count, and validates
  `config()` against ADR-001 §2 values; do NOT fail CI when the file is absent.

## Scope-cut rule (if blocked > half a day)

Drop q4_k/q2_k/iq2 index support to "dtype recognized, byteSize computed" only
(they share the block-geometry table in contracts.zig — that is already most
of it). Q8_0/f16/f32 index support cannot be cut.

## Forbidden

Editing `src/shared/contracts.zig` or fixture files; adding dependencies;
`std.Io`; touching other workstreams' files. Report: files, test counts,
exact commands run, anything cut.
