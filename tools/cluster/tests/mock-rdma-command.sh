#!/usr/bin/env bash
# PATH-injected command fixture for test-check-rdma-readiness.sh.

set -u

command_name="${0##*/}"
scenario="${MOCK_RDMA_SCENARIO:-ready}"

if [ -n "${MOCK_RDMA_CALL_LOG:-}" ]; then
  printf '%s\n' "$command_name" >>"$MOCK_RDMA_CALL_LOG"
fi

case "$command_name" in
  uname)
    if [ "$scenario" = "non_apple" ]; then
      case "${1:-}" in
        -s) printf 'Linux\n' ;;
        -m) printf 'x86_64\n' ;;
        *) printf 'Linux\n' ;;
      esac
    else
      case "${1:-}" in
        -s) printf 'Darwin\n' ;;
        -m) printf 'arm64\n' ;;
        *) printf 'Darwin\n' ;;
      esac
    fi
    ;;
  sw_vers)
    if [ "$scenario" = "old_os" ]; then
      printf '26.1.9\n'
    elif [ "$scenario" = "unknown_os" ]; then
      printf 'unavailable\n'
    else
      printf '26.2.1\n'
    fi
    ;;
  system_profiler)
    if [ "$scenario" = "unknown" ]; then
      printf 'profiler failed for SERIAL-SECRET\n' >&2
      exit 1
    fi
    cat <<'PROFILE'
Thunderbolt/USB4 Bus:
    Thunderbolt 5
    Speed: Up to 80 Gb/s x1
    Serial Number: SERIAL-SECRET
    Domain UUID: UUID-SECRET
    Owner: alice
PROFILE
    ;;
  xcrun)
    printf '%s\n' "${MOCK_RDMA_SDK_ROOT:?MOCK_RDMA_SDK_ROOT is required}"
    ;;
  ibv_devices)
    if [ "$scenario" = "unknown" ]; then
      printf 'device lookup failed for alice at 10.5.0.1\n' >&2
      exit 1
    fi
    cat <<'DEVICES'
    device                 node GUID
    ------              ----------------
    rdma_en2            e0c5797d9c8fac05
    rdma_en3            e1c5797d9c8fac05
DEVICES
    ;;
  ibv_devinfo)
    if [ "$scenario" = "unknown" ]; then
      printf 'UUID-SECRET\n' >&2
      exit 1
    fi
    state='PORT_ACTIVE (4)'
    if [ "$scenario" = "ports_down" ]; then
      state='PORT_DOWN (1)'
    fi
    cat <<DEVINFO
hca_id: rdma_en2
    transport: Thunderbolt (100)
    node_guid: UUID-SECRET
    port: 1
        state: $state
hca_id: rdma_en3
    transport: Thunderbolt (100)
    node_guid: UUID-SECRET-2
    port: 1
        state: $state
DEVINFO
    ;;
  networksetup)
    cat <<'NETWORK'
Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: aa:bb:cc:dd:ee:ff

Hardware Port: Wi-Fi
Device: en0
Ethernet Address: 11:22:33:44:55:66

Serial Number: SERIAL-SECRET
User: alice
NETWORK
    ;;
  ifconfig)
    cat <<'IFCONFIG'
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    ether aa:bb:cc:dd:ee:ff
    inet 10.5.0.1 netmask 0xffffff00 broadcast 10.5.0.255
    inet6 fe80::abcd:ef01:2345:6789%bridge0 prefixlen 64
    status: active
    uuid UUID-SECRET
IFCONFIG
    ;;
  netstat)
    cat <<'NETSTAT'
Routing tables
Internet:
Destination        Gateway            Flags               Netif Expire
default            192.168.1.1        UGScg                 en0
10.5.0/24          link#20            UCS               bridge0
10.5.0.2           aa:bb:cc:dd:ee:01  UHLWIi            bridge0
NETSTAT
    ;;
  route)
    route_interface=bridge0
    if [ "$scenario" = "wrong_route" ]; then
      route_interface=en0
    fi
    cat <<ROUTE
   route to: ${3:-10.5.0.2}
destination: ${3:-10.5.0.2}
    gateway: 10.5.0.1
  interface: $route_interface
      flags: <UP,HOST,DONE,LLINFO,WASCLONED,IFSCOPE,IFREF>
ROUTE
    ;;
  *)
    printf 'unexpected mock command: %s\n' "$command_name" >&2
    exit 127
    ;;
esac
