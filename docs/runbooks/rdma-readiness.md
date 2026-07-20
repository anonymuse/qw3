# RDMA-over-Thunderbolt readiness preflight

**Tool:** `tools/cluster/check-rdma-readiness.sh`

**Scope:** one local Mac at a time

**Evidence label:** `rdma_preflight_only`
**Performance interpretation:** none (`hardware_interpretable=false`)

This runbook checks whether a node appears ready for Apple's
RDMA-over-Thunderbolt API. It does not enable RDMA, configure a cluster, send a
packet, measure a link, or establish an inference-performance result.

Apple documents RDMA over Thunderbolt as available starting with macOS 26.2 on
Apple silicon Macs with Thunderbolt 5. The authoritative setup and API reference
is [Apple Technote TN3205: Low-latency communication with RDMA over
Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## Safety contract

The preflight is deliberately read-only. It only invokes local inventory
commands and parses their output in memory. It never:

- runs SSH or contacts a peer;
- enables or disables RDMA;
- enables, disables, creates, or deletes Thunderbolt Bridge;
- changes a network service, route, address, DNS setting, or firewall;
- changes a sysctl, boot argument, Recovery setting, or SIP setting;
- installs a package or developer tool;
- changes sleep, login, startup, or power settings;
- claims throughput, latency, tokens per second, or cluster viability.

Raw outputs from `system_profiler`, `ibv_devices`, `ibv_devinfo`, `ifconfig`,
`networksetup`, `netstat`, and `route` can contain serials, UUIDs, GUIDs, MAC
addresses, IP addresses, interface names, usernames, or peer addresses. The tool
does not reproduce those outputs. Its default JSON exposes only statuses,
reason codes, booleans, and counts.

## What it checks

The checks that determine `overall_status` are:

1. Darwin on `arm64` (the script's Apple-silicon test).
2. macOS 26.2 or newer.
3. An explicit Thunderbolt 5 or 80 Gb/s capability marker in
   `system_profiler SPThunderboltDataType`.
4. `infiniband/verbs.h` and `librdma.tbd` in the selected macOS SDK or system
   locations.
5. Availability of both `ibv_devices` and `ibv_devinfo`.
6. At least one Thunderbolt RDMA device and at least one `PORT_ACTIVE` port.

It also reports informational IP-over-Thunderbolt facts:

- the count of local hardware interfaces whose hardware-port description
  contains `Thunderbolt`;
- how many of those interfaces appear active;
- the number of IPv4 route-table entries assigned to those interfaces;
- optionally, whether the local route table selects one of those interfaces
  for a numeric peer address.

IP route facts do **not** prove that RDMA messages use that route. TN3205 says
the RDMA interface is point-to-point and is paired with a Thunderbolt IP
interface, but the transports are separate. Route observations are therefore
informational and do not affect `overall_status`.

## Run the preflight

Run locally on each compute node:

```sh
bash tools/cluster/check-rdma-readiness.sh
```

The script writes JSON to stdout and uses these exit codes:

| Exit | `overall_status` | Meaning |
|---:|---|---|
| 0 | `ready` | Every required local prerequisite was positively observed. |
| 1 | `not_ready` | At least one required prerequisite was definitively unsupported, missing, or inactive. |
| 2 | `unknown` | No prerequisite was definitively negative, but at least one required observation could not be completed or interpreted. |
| 64 | no JSON contract | Invalid command-line usage. |

`ready` means only that the listed local technical prerequisites were observed.
Proceeding still requires a separately reviewed RDMA transport test and direct
authorization for any machine change.

### Check local route selection for peers

`--peer` accepts only a syntactically valid numeric IPv4 or IPv6 address and
performs `route -n get` against the local route table. Hostnames are rejected
before any route lookup, so a peer argument cannot initiate DNS or mDNS name
resolution. IPv6 accepts an optional numeric scope ID such as `%12`; a named
scope such as `%en0` is rejected.

The lookup does not ping, connect, resolve cluster state over SSH, or send a
benchmark payload. Repeat the flag for each peer:

```sh
bash tools/cluster/check-rdma-readiness.sh \
  --peer <numeric-peer-b-ip> \
  --peer <numeric-peer-c-ip>
```

The peer addresses and selected interface names remain `redacted`/`null` in
the default output. The status and `uses_thunderbolt_interface` boolean remain
available for comparison.

For a private, local diagnostic only, an operator can explicitly include those
network identifiers:

```sh
bash tools/cluster/check-rdma-readiness.sh \
  --peer <numeric-peer-ip> \
  --include-network-identifiers
```

Do not paste, publish, or commit that opt-in output. Serial numbers, UUIDs, MAC
addresses, and usernames are never emitted, but peer addresses and interface
names are.

### Evidence boundary in the JSON

Every report includes:

```json
{
  "evidence": {
    "label": "rdma_preflight_only",
    "hardware_interpretable": false,
    "inference_performance_claim": false
  }
}
```

This report must not be promoted to a link result, a hardware benchmark, or an
inference result. A real transport gate needs separately versioned evidence
with exact endpoints, routes, physical ports, cable topology, message sizes,
directions, concurrency, warmup, repetitions, and p50/p95/p99 latency and
throughput.

## Operator-only enablement from Apple TN3205

The following steps change machine state. The preflight does not perform them,
and an agent must not infer permission to perform them from this runbook.

Apple's current TN3205 enablement sequence is:

1. An operator reboots the Mac into macOS Recovery.
2. In Recovery, the operator opens Utilities → Terminal.
3. The operator runs `rdma_ctl enable`.
4. The operator reboots into macOS.
5. Back in macOS, the operator checks for interfaces with `ibv_devices` and
   port state with `ibv_devinfo`.

Apple's [WWDC26 distributed MLX
session](https://developer.apple.com/videos/play/wwdc2026/233/) also demonstrates
an “Enable RDMA over Thunderbolt” setting followed by a reboot. Use the surface
available on the exact macOS build and follow Apple's current instructions.
Do not automate Recovery, change SIP, add boot arguments, or use undocumented
enablement methods.

Before any operator changes a node:

- preserve an out-of-band management path over ordinary Wi-Fi or Ethernet;
- record the exact macOS build on all nodes;
- change and verify one node at a time;
- ensure an operator can regain local access after reboot;
- rerun this preflight locally after the reboot.

The preflight reporting `not_ready` is not authorization to repair or mutate
the machine.

## Full-mesh Thunderbolt Bridge warning

TN3205 says IP over Thunderbolt and RDMA can operate in parallel and that the
Thunderbolt hardware load-balances the protocols. It also warns that the
default Thunderbolt Bridge forwards Ethernet frames between ports. In a cable
topology containing a loop—such as the proposed three-node full mesh—the bridge
can circulate frames indefinitely, consume CPU, and compromise network
performance.

Apple's documented mitigation for a loop is an **operator** making
“Thunderbolt Bridge” inactive in System Settings → Network. Do this only after
the operator has verified out-of-band management connectivity and reviewed the
current TN3205. The preflight never changes the service.

Do not interpret these cases as contradictions:

- RDMA devices can be active while Thunderbolt Bridge is inactive.
- A peer's IP route can stop using Thunderbolt while RDMA remains available.
- Thunderbolt Bridge can be useful in a non-loop topology.
- `--peer` can report `not_ready` while the required RDMA checks remain
  `ready`; peer-route status is intentionally informational.

## API constraints that affect the transport design

TN3205 documents a Verbs-compatible API with these current limits:

- include `infiniband/verbs.h` and link `librdma.tbd` from the macOS SDK;
- send and receive operations only—no hardware-initiated remote writes;
- at most 10 unreliable-connection queue pairs;
- messages no larger than 16,773,120 bytes;
- at most 4,095 work requests at a time;
- page-aligned memory registered separately for each Thunderbolt controller's
  IOMMU/protection domain;
- send and receive work requests must agree on frame count/message length.

QW3 should preserve its framing, sequence numbers, payload lengths, checksums,
timeouts, and failure evidence above the transport. Any future large transfer
must be chunked below the documented message limit. None of these design
constraints imply direct Metal-buffer registration or GPUDirect behavior; that
would require its own measured experiment.

## Troubleshooting status codes

| Check/reason | Interpretation | Safe next step |
|---|---|---|
| `platform.status=not_ready` | Not Darwin arm64. | Stop. Do not attempt Apple RDMA setup. |
| `version_below_26_2` | The OS predates Apple's supported release. | Operator reviews Apple's supported update path; script does nothing. |
| `no_generation_marker_in_profiler_output` | The profiler completed but did not prove TB5. | Verify the exact Mac specification and profiler format manually; do not assume. |
| `link_library_or_header_missing` | The selected SDK did not expose both API artifacts. | Operator verifies Xcode/CLT and macOS SDK versions; do not auto-install. |
| `one_or_more_tools_missing` | The documented inventory commands are unavailable. | Operator verifies RDMA enablement and OS installation from TN3205. |
| `no_rdma_devices` | `ibv_devices` completed with no listed device. | Verify operator enablement, reboot, and physical TB5 cabling. |
| `no_active_rdma_ports` | Devices exist but no port is `PORT_ACTIVE`. | Verify RDMA is enabled at both cable endpoints and inspect each cable/port. |
| `ibv_inventory_or_inspection_failed` | A command existed but failed. | Preserve the sanitized JSON; inspect locally without publishing raw identifiers. |
| `route_does_not_use_thunderbolt_interface` | Local IP route selected another interface. | Treat as route evidence only; do not change routes automatically. |

## Regression tests

The tests inject mocked macOS commands through `PATH`; they do not inspect or
change the machine running the test:

```sh
bash tools/cluster/tests/test-check-rdma-readiness.sh
```

Coverage includes a ready node, old macOS, non-Apple hardware, failed probes,
inactive RDMA ports, missing SDK artifacts, default redaction, and explicit
network-identifier opt-in. Peer validation covers numeric IPv4, numeric IPv6,
numeric IPv6 scope IDs, hostname rejection, named-scope rejection, and malformed
or ambiguous numeric inputs; rejected peers never reach `route`.

## Official sources

- [Apple TN3205: Low-latency communication with RDMA over
  Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
- [macOS 26.2 release notes: RDMA over
  Thunderbolt](https://developer.apple.com/documentation/macos-release-notes/macos-26_2-release-notes)
- [WWDC26: Explore distributed inference and training with
  MLX](https://developer.apple.com/videos/play/wwdc2026/233/)

These links were rechecked on 2026-07-16. Recheck them before changing operator
procedures because Apple can revise platform guidance independently of this
repository.
