# ADR-004: Auxiliary Hardware Stays Off the Model Data Plane

**Status:** Accepted
**Date:** 2026-07-08
**Depends on:** ADR-001

## Decision

The DS5 model data plane is exactly three nodes: A (M5 Pro 48GB), B and C (M5 Max 48GB).
No other machine holds model weights, KV pages, or participates in decode.

Permitted auxiliary roles:

| Machine | Permitted roles |
|---|---|
| Mac minis (16GB) | Telemetry sink, log/dashboard host, benchmark runner, artifact serving. Nothing latency-coupled to decode. |
| RTX 5080 / 9700X / 64GB box | Offline tool bench only: oracle trace generation, quantization experiments, dataset/eval generation, reference comparisons. |
| Dev machine (M5 24GB) | Build, unit tests, loopback benchmarks, tooling. Not a cluster node. |

## Rationale

The v0.2 spec pack assumes a strict 3-node mesh; every memory budget, placement manifest,
and transport rule is written against it. Adding weak nodes to the data plane adds
failure modes and scheduling complexity without adding meaningful memory or bandwidth
(16GB minis cannot hold a useful expert shard under the 70% static cap rule).

## Review trigger

Revisit only if the 3-node static budget (100.8GB) proves insufficient after real
quantized artifacts are measured (risk R-001), and only then with a superseding ADR.
