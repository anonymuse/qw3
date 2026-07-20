#!/usr/bin/env bash
# Read-only local preflight for Apple's RDMA-over-Thunderbolt prerequisites.
#
# This script never enables RDMA, changes Thunderbolt Bridge, touches routes,
# changes security settings, or contacts another host. It emits sanitized JSON
# only; command output that can contain host identifiers is parsed in memory and
# never copied to the report.

set -uo pipefail

MIN_MACOS_MAJOR=26
MIN_MACOS_MINOR=2
INCLUDE_NETWORK_IDENTIFIERS=0
PEERS=()

usage() {
  cat <<'USAGE'
Usage: check-rdma-readiness.sh [--peer <numeric-ip>]... [--include-network-identifiers]

Read-only checks for Apple Silicon, macOS 26.2+, Thunderbolt 5, the macOS
RDMA SDK/API, ibv tools and active RDMA ports. Optional --peer values must be
numeric IPv4 or IPv6 addresses and are looked up in the local route table only;
no packet is sent. An IPv6 scope suffix is accepted only when it is numeric.

Output is JSON. Network identifiers, including peer addresses and interface
names, are redacted unless --include-network-identifiers is explicitly set.

Exit status: 0 ready, 1 not_ready, 2 unknown, 64 usage error.
USAGE
}

is_valid_ipv4() {
  local address="$1" old_ifs octet

  case "$address" in
    ''|*[!0-9.]*|.*|*..*|*.) return 1 ;;
  esac

  old_ifs="$IFS"
  IFS=.
  set -- $address
  IFS="$old_ifs"
  [ "$#" -eq 4 ] || return 1

  for octet in "$@"; do
    [ -n "$octet" ] || return 1
    [ "${#octet}" -le 3 ] || return 1
    case "$octet" in
      *[!0-9]*) return 1 ;;
      0) ;;
      0*) return 1 ;;
    esac
    [ "$octet" -le 255 ] || return 1
  done

  return 0
}

# validate_ipv6_sequence <colon-delimited-parts> <allow-final-ipv4>
# Sets IPV6_SEQUENCE_UNITS to the number of 16-bit units represented. An
# embedded dotted-decimal tail counts as two units and is legal only at the end.
validate_ipv6_sequence() {
  local sequence="$1" allow_final_ipv4="$2" old_ifs part part_index

  IPV6_SEQUENCE_UNITS=0
  [ -n "$sequence" ] || return 0
  case "$sequence" in
    :*|*:|*::*) return 1 ;;
  esac

  old_ifs="$IFS"
  IFS=:
  set -- $sequence
  IFS="$old_ifs"
  part_index=0

  for part in "$@"; do
    part_index=$((part_index + 1))
    [ -n "$part" ] || return 1
    case "$part" in
      *.*)
        [ "$allow_final_ipv4" -eq 1 ] || return 1
        [ "$part_index" -eq "$#" ] || return 1
        is_valid_ipv4 "$part" || return 1
        IPV6_SEQUENCE_UNITS=$((IPV6_SEQUENCE_UNITS + 2))
        ;;
      *)
        case "$part" in
          *[!0-9A-Fa-f]*) return 1 ;;
        esac
        [ "${#part}" -le 4 ] || return 1
        IPV6_SEQUENCE_UNITS=$((IPV6_SEQUENCE_UNITS + 1))
        ;;
    esac
  done

  return 0
}

is_valid_ipv6() {
  local candidate="$1" address scope remainder left right left_units right_units total_units

  address="$candidate"
  case "$candidate" in
    *%*)
      address="${candidate%%\%*}"
      scope="${candidate#*%}"
      case "$scope" in
        ''|*[!0-9]*|*%*) return 1 ;;
      esac
      # Darwin route scope identifiers are interface indexes. Bound the text
      # length so an arbitrary-size decimal cannot reach the route parser.
      [ "${#scope}" -le 10 ] || return 1
      ;;
  esac

  [ -n "$address" ] || return 1
  case "$address" in
    *:*) ;;
    *) return 1 ;;
  esac
  case "$address" in
    *[!0-9A-Fa-f:.]*) return 1 ;;
  esac

  case "$address" in
    *::*)
      remainder="${address#*::}"
      case "$remainder" in
        *::*) return 1 ;;
      esac
      left="${address%%::*}"
      right="$remainder"

      validate_ipv6_sequence "$left" 0 || return 1
      left_units="$IPV6_SEQUENCE_UNITS"
      validate_ipv6_sequence "$right" 1 || return 1
      right_units="$IPV6_SEQUENCE_UNITS"
      total_units=$((left_units + right_units))

      # A double colon must replace at least one of the eight 16-bit units.
      [ "$total_units" -lt 8 ] || return 1
      ;;
    *)
      validate_ipv6_sequence "$address" 1 || return 1
      [ "$IPV6_SEQUENCE_UNITS" -eq 8 ] || return 1
      ;;
  esac

  return 0
}

is_valid_numeric_peer() {
  case "$1" in
    *:*) is_valid_ipv6 "$1" ;;
    *) is_valid_ipv4 "$1" ;;
  esac
}

command_available() {
  command -v "$1" >/dev/null 2>&1
}

version_at_least_26_2() {
  local version="$1" major minor rest

  case "$version" in
    *[!0-9.]*|.*|*..*|*.) return 2 ;;
  esac

  major="${version%%.*}"
  rest="${version#*.}"
  if [ "$rest" = "$version" ]; then
    minor=0
  else
    minor="${rest%%.*}"
  fi
  [ -n "$major" ] && [ -n "$minor" ] || return 2

  if [ "$major" -gt "$MIN_MACOS_MAJOR" ]; then
    return 0
  fi
  if [ "$major" -eq "$MIN_MACOS_MAJOR" ] && [ "$minor" -ge "$MIN_MACOS_MINOR" ]; then
    return 0
  fi
  return 1
}

interface_in_list() {
  local wanted="$1" candidate

  while IFS= read -r candidate; do
    [ -n "$candidate" ] || continue
    [ "$candidate" = "$wanted" ] && return 0
  done <<EOF
$tb_interfaces
EOF
  return 1
}

emit_interface_array() {
  local first=1 interface

  if [ "$INCLUDE_NETWORK_IDENTIFIERS" -eq 0 ]; then
    printf 'null'
    return
  fi

  printf '['
  while IFS= read -r interface; do
    [ -n "$interface" ] || continue
    if [ "$first" -eq 0 ]; then
      printf ','
    fi
    printf '"%s"' "$interface"
    first=0
  done <<EOF
$tb_interfaces
EOF
  printf ']'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --peer)
      if [ "$#" -lt 2 ] || ! is_valid_numeric_peer "$2"; then
        printf 'error: --peer requires numeric IPv4 or IPv6; an IPv6 scope ID must be numeric\n' >&2
        exit 64
      fi
      PEERS[${#PEERS[@]}]="$2"
      shift 2
      ;;
    --include-network-identifiers)
      INCLUDE_NETWORK_IDENTIFIERS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

# Defaults are deliberately conservative. A missing or unparsable observation
# is unknown; a proven unsupported prerequisite is not_ready.
kernel_raw="$(uname -s 2>/dev/null || true)"
arch_raw="$(uname -m 2>/dev/null || true)"
platform_kernel="non_darwin"
platform_architecture="non_arm64"
platform_status="not_ready"
apple_silicon=false

if [ "$kernel_raw" = "Darwin" ]; then
  platform_kernel="Darwin"
fi
if [ "$arch_raw" = "arm64" ]; then
  platform_architecture="arm64"
fi
if [ "$kernel_raw" = "Darwin" ] && [ "$arch_raw" = "arm64" ]; then
  platform_status="ready"
  apple_silicon=true
fi

macos_status="not_ready"
macos_version="unknown"
macos_reason="unsupported_platform"
sw_vers_available=false

if [ "$platform_status" = "ready" ]; then
  if command_available sw_vers; then
    sw_vers_available=true
    raw_version="$(sw_vers -productVersion 2>/dev/null || true)"
    case "$raw_version" in
      ''|*[!0-9.]*)
        macos_status="unknown"
        macos_reason="version_unavailable_or_unparsable"
        ;;
      *)
        macos_version="$raw_version"
        if version_at_least_26_2 "$raw_version"; then
          macos_status="ready"
          macos_reason="minimum_met"
        else
          version_result=$?
          if [ "$version_result" -eq 1 ]; then
            macos_status="not_ready"
            macos_reason="version_below_26_2"
          else
            macos_status="unknown"
            macos_reason="version_unavailable_or_unparsable"
            macos_version="unknown"
          fi
        fi
        ;;
    esac
  else
    macos_status="unknown"
    macos_reason="sw_vers_unavailable"
  fi
fi

eligible_for_rdma_checks=0
if [ "$platform_status" = "ready" ] && [ "$macos_status" = "ready" ]; then
  eligible_for_rdma_checks=1
fi

if [ "$platform_status" = "not_ready" ] || [ "$macos_status" = "not_ready" ]; then
  skipped_status="not_ready"
  skipped_reason="unsupported_platform_or_os"
else
  skipped_status="unknown"
  skipped_reason="platform_or_os_unconfirmed"
fi

tb5_status="$skipped_status"
tb5_reason="$skipped_reason"
system_profiler_available=false
system_profiler_completed=false

rdma_api_status="$skipped_status"
rdma_api_reason="$skipped_reason"
xcrun_available=false
sdk_detected=false
link_library_available=false
verbs_header_available=false

rdma_tools_status="$skipped_status"
rdma_tools_reason="$skipped_reason"
ibv_devices_available=false
ibv_devinfo_available=false

rdma_devices_status="$skipped_status"
rdma_devices_reason="$skipped_reason"
ibv_devices_completed=false
ibv_devinfo_completed=false
device_count_json=null
thunderbolt_transport_count_json=null
active_port_count_json=null

thunderbolt_network_status="$skipped_status"
thunderbolt_network_reason="$skipped_reason"
networksetup_available=false
networksetup_completed=false
ifconfig_available=false
netstat_available=false
route_available=false
tb_interfaces=""
interface_count=0
active_interface_count_json=null
route_count_json=null

if [ "$eligible_for_rdma_checks" -eq 1 ]; then
  # system_profiler can include serials, UUIDs, and attached-device names. Its
  # raw output is intentionally never emitted.
  if command_available system_profiler; then
    system_profiler_available=true
    if profiler_output="$(LC_ALL=C system_profiler SPThunderboltDataType 2>/dev/null)"; then
      system_profiler_completed=true
      if printf '%s\n' "$profiler_output" | LC_ALL=C grep -Ei \
        'Thunderbolt[[:space:]]*5|80[[:space:]]*Gb(/s|ps)' >/dev/null; then
        tb5_status="ready"
        tb5_reason="explicit_tb5_or_80_gbps_marker"
      elif printf '%s\n' "$profiler_output" | LC_ALL=C grep -Ei \
        'Thunderbolt[[:space:]]*[34]|40[[:space:]]*Gb(/s|ps)' >/dev/null; then
        tb5_status="not_ready"
        tb5_reason="explicit_pre_tb5_or_40_gbps_marker"
      else
        tb5_status="unknown"
        tb5_reason="no_generation_marker_in_profiler_output"
      fi
    else
      tb5_status="unknown"
      tb5_reason="system_profiler_failed"
    fi
  else
    tb5_status="unknown"
    tb5_reason="system_profiler_unavailable"
  fi

  if command_available xcrun; then
    xcrun_available=true
    sdk_root="$(LC_ALL=C xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
    case "$sdk_root" in
      /*)
        if [ -d "$sdk_root" ]; then
          sdk_detected=true
          if [ -e "$sdk_root/usr/lib/librdma.tbd" ] || \
             [ -e "$sdk_root/usr/lib/librdma.dylib" ]; then
            link_library_available=true
          fi
          if [ -e "$sdk_root/usr/include/infiniband/verbs.h" ]; then
            verbs_header_available=true
          fi
        fi
        ;;
    esac
  fi
  if [ -e /usr/lib/librdma.tbd ] || [ -e /usr/lib/librdma.dylib ]; then
    link_library_available=true
  fi
  if [ -e /usr/include/infiniband/verbs.h ]; then
    verbs_header_available=true
  fi

  if [ "$link_library_available" = true ] && [ "$verbs_header_available" = true ]; then
    rdma_api_status="ready"
    rdma_api_reason="link_library_and_header_found"
  else
    rdma_api_status="not_ready"
    rdma_api_reason="link_library_or_header_missing"
  fi

  if command_available ibv_devices; then
    ibv_devices_available=true
  fi
  if command_available ibv_devinfo; then
    ibv_devinfo_available=true
  fi
  if [ "$ibv_devices_available" = true ] && [ "$ibv_devinfo_available" = true ]; then
    rdma_tools_status="ready"
    rdma_tools_reason="both_tools_available"
  else
    rdma_tools_status="not_ready"
    rdma_tools_reason="one_or_more_tools_missing"
  fi

  if [ "$rdma_tools_status" = "ready" ]; then
    devices_output=""
    devinfo_output=""
    if devices_output="$(LC_ALL=C ibv_devices 2>/dev/null)"; then
      ibv_devices_completed=true
      device_count="$(printf '%s\n' "$devices_output" | LC_ALL=C awk '
        BEGIN { in_table = 0; count = 0 }
        /^[[:space:]]*device[[:space:]]+node[[:space:]]+GUID[[:space:]]*$/ {
          in_table = 1
          next
        }
        in_table && /^[[:space:]]*-+[[:space:]]+-+[[:space:]]*$/ { next }
        in_table && NF >= 2 { count += 1 }
        END { print count }
      ')"
      device_count_json="${device_count:-0}"
    fi

    if devinfo_output="$(LC_ALL=C ibv_devinfo 2>/dev/null)"; then
      ibv_devinfo_completed=true
      thunderbolt_transport_count="$(printf '%s\n' "$devinfo_output" | \
        LC_ALL=C grep -Eic 'transport:[[:space:]]*Thunderbolt' || true)"
      active_port_count="$(printf '%s\n' "$devinfo_output" | \
        LC_ALL=C grep -Eic 'state:[[:space:]]*PORT_ACTIVE' || true)"
      thunderbolt_transport_count_json="${thunderbolt_transport_count:-0}"
      active_port_count_json="${active_port_count:-0}"
    fi

    if [ "$ibv_devices_completed" != true ] || [ "$ibv_devinfo_completed" != true ]; then
      rdma_devices_status="unknown"
      rdma_devices_reason="ibv_inventory_or_inspection_failed"
    elif [ "$device_count_json" -eq 0 ]; then
      rdma_devices_status="not_ready"
      rdma_devices_reason="no_rdma_devices"
    elif [ "$thunderbolt_transport_count_json" -eq 0 ]; then
      rdma_devices_status="unknown"
      rdma_devices_reason="no_thunderbolt_transport_marker"
    elif [ "$active_port_count_json" -eq 0 ]; then
      rdma_devices_status="not_ready"
      rdma_devices_reason="no_active_rdma_ports"
    else
      rdma_devices_status="ready"
      rdma_devices_reason="thunderbolt_devices_and_active_ports_found"
    fi
  else
    rdma_devices_status="not_ready"
    rdma_devices_reason="ibv_tools_unavailable"
  fi

  if command_available networksetup; then
    networksetup_available=true
  fi
  if command_available ifconfig; then
    ifconfig_available=true
  fi
  if command_available netstat; then
    netstat_available=true
  fi
  if command_available route; then
    route_available=true
  fi

  if [ "$networksetup_available" = true ]; then
    if network_output="$(LC_ALL=C networksetup -listallhardwareports 2>/dev/null)"; then
      networksetup_completed=true
      tb_interfaces="$(printf '%s\n' "$network_output" | LC_ALL=C awk '
        /^Hardware Port:[[:space:]]*/ {
          thunderbolt = index($0, "Thunderbolt") > 0
          next
        }
        thunderbolt && /^Device:[[:space:]]*/ {
          sub(/^Device:[[:space:]]*/, "")
          if ($0 ~ /^[A-Za-z0-9._-]+$/ && !seen[$0]++) print $0
          thunderbolt = 0
        }
      ')"
      interface_count="$(printf '%s\n' "$tb_interfaces" | \
        LC_ALL=C awk 'NF { count += 1 } END { print count + 0 }')"
    fi
  fi

  if [ "$networksetup_completed" != true ]; then
    thunderbolt_network_status="unknown"
    thunderbolt_network_reason="network_hardware_inventory_unavailable"
  elif [ "$interface_count" -eq 0 ]; then
    thunderbolt_network_status="unknown"
    thunderbolt_network_reason="no_thunderbolt_ip_interface_found"
  else
    thunderbolt_network_status="ready"
    thunderbolt_network_reason="thunderbolt_ip_interface_inventory_found"
  fi

  if [ "$ifconfig_available" = true ] && [ "$interface_count" -gt 0 ]; then
    active_interface_count=0
    while IFS= read -r interface; do
      [ -n "$interface" ] || continue
      if interface_output="$(LC_ALL=C ifconfig "$interface" 2>/dev/null)"; then
        if printf '%s\n' "$interface_output" | LC_ALL=C grep -E \
          'status:[[:space:]]*active|flags=[^<]*<[^>]*UP' >/dev/null; then
          active_interface_count=$((active_interface_count + 1))
        fi
      fi
    done <<EOF
$tb_interfaces
EOF
    active_interface_count_json="$active_interface_count"
  fi

  if [ "$netstat_available" = true ] && [ "$interface_count" -gt 0 ]; then
    if route_table_output="$(LC_ALL=C netstat -rn -f inet 2>/dev/null)"; then
      route_count=0
      while IFS= read -r interface; do
        [ -n "$interface" ] || continue
        interface_route_count="$(printf '%s\n' "$route_table_output" | \
          LC_ALL=C awk -v interface="$interface" '$NF == interface { count += 1 } END { print count + 0 }')"
        route_count=$((route_count + interface_route_count))
      done <<EOF
$tb_interfaces
EOF
      route_count_json="$route_count"
    fi
  fi
fi

PEER_STATUSES=()
PEER_REASONS=()
PEER_USES_THUNDERBOLT=()
PEER_INTERFACES=()
peer_index=0
while [ "$peer_index" -lt "${#PEERS[@]}" ]; do
  peer="${PEERS[$peer_index]}"
  peer_status="$skipped_status"
  peer_reason="$skipped_reason"
  peer_uses_thunderbolt=null
  peer_interface=""

  if [ "$eligible_for_rdma_checks" -eq 1 ]; then
    if [ "$route_available" != true ]; then
      peer_status="unknown"
      peer_reason="route_command_unavailable"
    elif peer_route_output="$(LC_ALL=C route -n get "$peer" 2>/dev/null)"; then
      peer_interface="$(printf '%s\n' "$peer_route_output" | LC_ALL=C awk '
        /^[[:space:]]*interface:[[:space:]]*/ {
          sub(/^[[:space:]]*interface:[[:space:]]*/, "")
          if ($0 ~ /^[A-Za-z0-9._-]+$/) print $0
          exit
        }
      ')"
      if [ -z "$peer_interface" ]; then
        peer_status="unknown"
        peer_reason="route_interface_unavailable"
      elif interface_in_list "$peer_interface"; then
        peer_status="ready"
        peer_reason="route_uses_thunderbolt_interface"
        peer_uses_thunderbolt=true
      else
        peer_status="not_ready"
        peer_reason="route_does_not_use_thunderbolt_interface"
        peer_uses_thunderbolt=false
      fi
    else
      peer_status="unknown"
      peer_reason="route_lookup_failed"
    fi
  fi

  PEER_STATUSES[$peer_index]="$peer_status"
  PEER_REASONS[$peer_index]="$peer_reason"
  PEER_USES_THUNDERBOLT[$peer_index]="$peer_uses_thunderbolt"
  PEER_INTERFACES[$peer_index]="$peer_interface"
  peer_index=$((peer_index + 1))
done

overall_status="ready"
for prerequisite_status in \
  "$platform_status" "$macos_status" "$tb5_status" \
  "$rdma_api_status" "$rdma_tools_status" "$rdma_devices_status"; do
  if [ "$prerequisite_status" = "not_ready" ]; then
    overall_status="not_ready"
  elif [ "$prerequisite_status" = "unknown" ] && [ "$overall_status" = "ready" ]; then
    overall_status="unknown"
  fi
done

if [ "$INCLUDE_NETWORK_IDENTIFIERS" -eq 1 ]; then
  network_identifiers_redacted=false
  network_identifiers_included=true
else
  network_identifiers_redacted=true
  network_identifiers_included=false
fi

printf '{\n'
printf '  "schema_version": 1,\n'
printf '  "evidence": {"label": "rdma_preflight_only", "hardware_interpretable": false, "inference_performance_claim": false},\n'
printf '  "overall_status": "%s",\n' "$overall_status"
printf '  "redaction": {"network_identifiers_redacted": %s, "network_identifiers_included": %s, "serials_uuids_usernames_redacted": true},\n' \
  "$network_identifiers_redacted" "$network_identifiers_included"
printf '  "checks": {\n'
printf '    "platform": {"status": "%s", "kernel": "%s", "architecture": "%s", "apple_silicon": %s},\n' \
  "$platform_status" "$platform_kernel" "$platform_architecture" "$apple_silicon"
printf '    "macos": {"status": "%s", "reason": "%s", "version": "%s", "minimum_version": "26.2", "sw_vers_available": %s},\n' \
  "$macos_status" "$macos_reason" "$macos_version" "$sw_vers_available"
printf '    "thunderbolt5": {"status": "%s", "reason": "%s", "system_profiler_available": %s, "system_profiler_completed": %s},\n' \
  "$tb5_status" "$tb5_reason" "$system_profiler_available" "$system_profiler_completed"
printf '    "rdma_api": {"status": "%s", "reason": "%s", "xcrun_available": %s, "sdk_detected": %s, "link_library_available": %s, "verbs_header_available": %s},\n' \
  "$rdma_api_status" "$rdma_api_reason" "$xcrun_available" "$sdk_detected" \
  "$link_library_available" "$verbs_header_available"
printf '    "rdma_tools": {"status": "%s", "reason": "%s", "ibv_devices_available": %s, "ibv_devinfo_available": %s},\n' \
  "$rdma_tools_status" "$rdma_tools_reason" "$ibv_devices_available" "$ibv_devinfo_available"
printf '    "rdma_devices": {"status": "%s", "reason": "%s", "ibv_devices_completed": %s, "ibv_devinfo_completed": %s, "device_count": %s, "thunderbolt_transport_count": %s, "active_port_count": %s},\n' \
  "$rdma_devices_status" "$rdma_devices_reason" "$ibv_devices_completed" \
  "$ibv_devinfo_completed" "$device_count_json" "$thunderbolt_transport_count_json" \
  "$active_port_count_json"
printf '    "thunderbolt_network": {"status": "%s", "reason": "%s", "networksetup_available": %s, "networksetup_completed": %s, "ifconfig_available": %s, "netstat_available": %s, "route_available": %s, "interface_count": %s, "active_interface_count": %s, "route_count": %s, "interfaces": ' \
  "$thunderbolt_network_status" "$thunderbolt_network_reason" \
  "$networksetup_available" "$networksetup_completed" "$ifconfig_available" \
  "$netstat_available" "$route_available" "$interface_count" \
  "$active_interface_count_json" "$route_count_json"
emit_interface_array
printf '}\n'
printf '  },\n'
printf '  "peer_routes": ['
peer_index=0
while [ "$peer_index" -lt "${#PEERS[@]}" ]; do
  if [ "$peer_index" -gt 0 ]; then
    printf ','
  fi
  if [ "$INCLUDE_NETWORK_IDENTIFIERS" -eq 1 ]; then
    peer_target_json="\"${PEERS[$peer_index]}\""
    if [ -n "${PEER_INTERFACES[$peer_index]}" ]; then
      peer_interface_json="\"${PEER_INTERFACES[$peer_index]}\""
    else
      peer_interface_json=null
    fi
  else
    peer_target_json='"redacted"'
    peer_interface_json=null
  fi
  printf '\n    {"target": %s, "status": "%s", "reason": "%s", "uses_thunderbolt_interface": %s, "interface": %s}' \
    "$peer_target_json" "${PEER_STATUSES[$peer_index]}" \
    "${PEER_REASONS[$peer_index]}" "${PEER_USES_THUNDERBOLT[$peer_index]}" \
    "$peer_interface_json"
  peer_index=$((peer_index + 1))
done
if [ "${#PEERS[@]}" -gt 0 ]; then
  printf '\n  '
fi
printf ']\n'
printf '}\n'

case "$overall_status" in
  ready) exit 0 ;;
  not_ready) exit 1 ;;
  *) exit 2 ;;
esac
