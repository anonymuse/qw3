# DS5 Runbook — M0 Mesh Benchmark

How to produce the Phase 0 link measurements on the real 3-node cluster.

This covers only the 3 TB5-bridged compute nodes (A/B/C). The dev/management
laptop (Node D) is not TB5-connected and has no role in this benchmark — see
[`tools/cluster/README.md`](../tools/cluster/README.md) for its bring-up and
[`tools/cluster/topology.md`](../tools/cluster/topology.md) for the full
4-node picture.

## 0. One-time cluster network setup

On each Mac, after connecting the Thunderbolt 5 cables (full mesh: A–B, A–C, B–C):

1. System Settings → Network → Thunderbolt Bridge → Details → TCP/IP.
2. Configure IPv4 **Manually** with the addresses from
   [manifests/cluster/lab.zon](../manifests/cluster/lab.zon):
   - Node A (M5 Pro): `10.5.0.1`, subnet `255.255.255.0`
   - Node B (M5 Max): `10.5.0.2`
   - Node C (M5 Max): `10.5.0.3`
3. Verify with `ping -c 3 10.5.0.2` etc. from each node.

Note: with a full TB mesh, macOS bridges the two TB interfaces per machine.
If routing behaves oddly, benchmark pairwise (single cable per pair) first and
record which topology produced the numbers in the run label.

## 1. Build (every node)

```sh
brew install zig   # 0.16.0
git clone <this repo> && cd qw3
zig build
zig build test
```

## 2. Start daemons (all three nodes)

```sh
# on node A:
./zig-out/bin/ds5 node --name a
# on node B:
./zig-out/bin/ds5 node --name b
# on node C:
./zig-out/bin/ds5 node --name c
```

Sanity check from any node: `./zig-out/bin/ds5 health --host 10.5.0.1`

## 3. Run the benchmark (from each node, 3 runs each)

```sh
# on node A (targets b and c):
./zig-out/bin/ds5 bench link --cluster manifests/cluster/lab.zon --self a --label mesh-run1
# on node B:
./zig-out/bin/ds5 bench link --cluster manifests/cluster/lab.zon --self b --label mesh-run1
# on node C:
./zig-out/bin/ds5 bench link --cluster manifests/cluster/lab.zon --self c --label mesh-run1
```

Each run writes `bench/results/link-<epoch>.json` and prints a human summary.
Full run: 500 RTT iterations x 4 sizes, 3 bandwidth reps x 4 block sizes, 10 s
sustained. Use `--sustained-secs 60` for the recorded runs, `--quick` for smoke.

## 4. Accept / reject (M0 gate)

- Collect 3 runs per direction; variance must be <10% or the run is repeated
  with the machines idle (close browsers, disable Spotlight indexing spikes).
- Commit the JSON files to `bench/results/` and update
  [docs/assumptions.md](assumptions.md) A-02 with measured values.
- Failure signals (see execution plan v0.3): RTT p99 > ~1 ms sustained,
  bandwidth < ~1 GB/s, or unexplained multi-ms jitter spikes.

## 5. Hour-zero background tasks

On a worker node with >150 GB free disk:

```sh
./tools/download_models.sh
```
