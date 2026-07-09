//! Cluster manifest: the deterministic configuration file that names every node,
//! its role, address, and memory caps. Stored as ZON (`manifests/cluster/*.zon`) —
//! deterministic, zero-dependency, parseable by the Zig standard library.
//!
//! Hosts are IP addresses, not hostnames: the Thunderbolt bridge interfaces get
//! statically assigned IPs (see docs/runbook.md); avoiding DNS keeps benchmark
//! numbers free of resolution noise.

const std = @import("std");
const sys = @import("sys.zig");

pub const NodeRole = enum {
    control_plane,
    decode_worker,
    dev,
};

pub const NodeSpec = struct {
    name: []const u8,
    role: NodeRole,
    host: []const u8,
    port: u16 = 4750,
    chip: []const u8 = "unknown",
    memory_gb: f64 = 0,
    static_cap_gb: f64 = 0,
};

pub const ClusterManifest = struct {
    cluster_name: []const u8,
    manifest_version: u32 = 1,
    nodes: []const NodeSpec,

    pub fn findNode(self: *const ClusterManifest, name: []const u8) ?*const NodeSpec {
        for (self.nodes) |*n| {
            if (std.mem.eql(u8, n.name, name)) return n;
        }
        return null;
    }
};

pub fn parse(alloc: std.mem.Allocator, source: [:0]const u8) !ClusterManifest {
    return std.zon.parse.fromSliceAlloc(ClusterManifest, alloc, source, null, .{});
}

pub fn loadFile(alloc: std.mem.Allocator, path: []const u8) !ClusterManifest {
    const src = try sys.readFileAllocZ(alloc, path);
    return parse(alloc, src);
}

test "parse a three-node manifest" {
    const src: [:0]const u8 =
        \\.{
        \\    .cluster_name = "ds5-lab",
        \\    .manifest_version = 1,
        \\    .nodes = .{
        \\        .{ .name = "a", .role = .control_plane, .host = "10.5.0.1", .port = 4750, .chip = "M5 Pro", .memory_gb = 48.0, .static_cap_gb = 33.6 },
        \\        .{ .name = "b", .role = .decode_worker, .host = "10.5.0.2", .chip = "M5 Max", .memory_gb = 48.0, .static_cap_gb = 33.6 },
        \\        .{ .name = "c", .role = .decode_worker, .host = "10.5.0.3", .chip = "M5 Max", .memory_gb = 48.0, .static_cap_gb = 33.6 },
        \\    },
        \\}
    ;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const m = try parse(arena.allocator(), src);
    try std.testing.expectEqual(@as(usize, 3), m.nodes.len);
    try std.testing.expectEqualStrings("ds5-lab", m.cluster_name);
    try std.testing.expectEqual(NodeRole.control_plane, m.nodes[0].role);
    try std.testing.expectEqual(@as(u16, 4750), m.nodes[1].port);
    const c = m.findNode("c") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("10.5.0.3", c.host);
    try std.testing.expect(m.findNode("nope") == null);
}
