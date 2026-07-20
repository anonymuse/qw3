# DS5 Assumptions Ledger

Every load-bearing number or behavior we have **not yet measured**. Each entry gets
replaced by a measurement (link the finding) or explicitly re-dated. Rule: nothing in
this file may appear in a public claim.

| # | Assumption | Source | Status (updated through 2026-07-16) |
|---|---|---|---|
| A-01 | Node A (M5 Pro) UMA bandwidth ≈ 307 GB/s; B/C (M5 Max) ≈ 614 GB/s | Apple spec sheets via v0.2 pack | Unmeasured |
| A-02 | TB5 IP-bridge link delivers multi-GB/s bandwidth and sub-millisecond RTT | Hope | Unmeasured — M0 `ds5 bench link` replaces this |
| A-03 | macOS 26.2+ exposes a Verbs-compatible, two-sided user-space RDMA API on Apple silicon Macs with Thunderbolt 5 | [Apple TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt) and [macOS 26.2 release notes](https://developer.apple.com/documentation/macos-release-notes/macos-26_2-release-notes) | Platform availability verified 2026-07-16; QW3 transport correctness and performance remain unmeasured. Use `tools/cluster/check-rdma-readiness.sh` for sanitized local preflight only. |
| A-04 | Qwen3-235B expert usage is skewed enough for hot/warm/cool/cold tiering to beat uniform placement | v0.2 tiering design | Unmeasured — thesis-critical; M1 telemetry capture replaces this |
| A-05 | 235B expert weights ≈ 227B params (94 layers × 128 experts × 3 × 4096 × 1536); budget closes only at ~2.4–2.6 bpw average | Arithmetic from HF config | Config values to be verified against downloaded artifact |
| A-06 | Per-node static cap 33.6GB (70% of 48GB) leaves enough runtime reserve for KV at 32K (~3.2GB/worker at FP16) | v0.2 pack | Planning value; verify under real Metal heap behavior |
| A-07 | Community GGUF artifacts (unsloth et al.) faithfully preserve Qwen3 router/gate tensors at high precision | ADR-002 reuse decision | Verify tensor-by-tensor at M2 GGUF parse |
| A-08 | macOS `F_NOCACHE`/`F_RDAHEAD` suffice for NVMe promotion I/O control (no `O_DIRECT` on Darwin) | Platform knowledge | Unmeasured (risk R-009); measure before M4 |
| A-09 | Metal command-buffer dispatch overhead is small enough for per-layer kernel launches at 94 layers/token | v0.2 risk R-007 | Measured 2026-07-11 (W2, M5 Air): ~380–590 µs per synchronous one-dispatch command buffer → per-layer sync buffers would cost ~40 ms/token at 94 layers. Kernels MUST batch many dispatches per command buffer; glue `begin()`/`submit()` supports this. Re-verify on M5 Max. |
| A-10 | Toolchain: Zig 0.16.0 (docs said "Zig 1.0", which does not exist); macOS on all nodes; Python 3.14 for tooling | This repo | Fact, re-verify on cluster nodes |
| A-11 | Dev machine (Apple M5, 24GB) is not a cluster node; loopback numbers from it are *not* mesh numbers | ADR-004 | Fact |
| A-12 | All cluster nodes are aarch64 little-endian; wire formats assume native byte order | Protocol design | Fact for Apple Silicon; revisit only if a foreign node ever joins the data plane (ADR-004 forbids it) |
