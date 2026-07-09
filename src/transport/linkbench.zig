//! `ds5 bench link` — measures what the interconnect actually does: RTT by
//! message size, bandwidth by block size, and sustained throughput, per target
//! node. Emits machine-readable JSON to bench/results/ plus a human summary.
//!
//! This is Phase 0 / milestone M0: these numbers replace assumptions A-01..A-03.

const std = @import("std");
const protocol = @import("../shared/protocol.zig");
const tcp = @import("tcp.zig");
const sys = @import("../shared/sys.zig");
const manifest = @import("../shared/manifest.zig");
const stats = @import("../shared/stats.zig");
const JsonBuf = @import("../shared/jsonbuf.zig").JsonBuf;
const version = @import("../shared/version.zig");
const out = @import("../shared/out.zig");

pub const Options = struct {
    cluster_path: []const u8,
    self_name: ?[]const u8 = null,
    out_dir: []const u8 = "bench/results",
    label: []const u8 = "",
    sustained_secs: u64 = 10,
    quick: bool = false,
};

const RTT_SIZES = [_]usize{ 64, 512, 4096, 8192 };
const RTT_WARMUP = 50;
const RTT_ITERS = 500;
const RTT_ITERS_QUICK = 100;

const BW_SIZES = [_]usize{ 1 << 20, 16 << 20, 64 << 20, 256 << 20 };
const BW_REPS = 3;
const SUSTAINED_BLOB = 64 << 20;

pub fn run(alloc: std.mem.Allocator, opts: Options) !void {
    const cluster = try manifest.loadFile(alloc, opts.cluster_path);

    var jb = JsonBuf.init(alloc);
    const epoch = sys.epochSeconds();

    try jb.print("{{\"run_id\":\"link-{d}\",\"benchmark\":\"link\",\"schema_version\":1,", .{epoch});
    try jb.print("\"epoch_seconds\":{d},", .{epoch});
    try jb.raw("\"git_commit\":");
    try jb.str(gitCommit(alloc));
    try jb.raw(",\"ds5_version\":");
    try jb.str(version.DS5_VERSION);
    try jb.raw(",\"label\":");
    try jb.str(opts.label);
    try jb.raw(",\"cluster\":");
    try jb.str(cluster.cluster_name);
    try jb.raw(",\"self_node\":");
    try jb.str(opts.self_name orelse "");
    try jb.print(",\"sustained_secs\":{d},\"targets\":[", .{opts.sustained_secs});

    var first = true;
    var benchmarked: usize = 0;
    for (cluster.nodes) |*node| {
        if (opts.self_name) |self| {
            if (std.mem.eql(u8, node.name, self)) continue;
        }
        if (!first) try jb.raw(",");
        first = false;
        try benchTarget(alloc, &jb, node, opts);
        benchmarked += 1;
    }
    try jb.raw("]}");

    if (benchmarked == 0) {
        out.status("no target nodes (all filtered by --self?)\n", .{});
        return error.NoTargets;
    }

    try sys.mkdirPath(alloc, opts.out_dir);
    var name_buf: [512]u8 = undefined;
    const file_name = try std.fmt.bufPrint(&name_buf, "{s}/link-{d}.json", .{ opts.out_dir, epoch });
    try sys.writeFileTrunc(alloc, file_name, jb.items());
    out.print("\nresults written to {s}\n", .{file_name});
}

fn benchTarget(alloc: std.mem.Allocator, jb: *JsonBuf, node: *const manifest.NodeSpec, opts: Options) !void {
    out.print("\n=== target '{s}' ({s}:{d}) ===\n", .{ node.name, node.host, node.port });
    try jb.raw("{\"node\":");
    try jb.str(node.name);
    try jb.raw(",\"host\":");
    try jb.str(node.host);

    const fd = tcp.connect(node.host, node.port) catch |err| {
        out.print("  UNREACHABLE: {s}\n", .{@errorName(err)});
        try jb.print(",\"reachable\":false,\"error\":\"{s}\"}}", .{@errorName(err)});
        return;
    };
    defer sys.closeFd(fd);
    try jb.raw(",\"reachable\":true");

    // Health snapshot from the daemon (already JSON — embed raw).
    try protocol.writeFrame(fd, .health_req, &.{});
    const hh = try protocol.readHeader(fd);
    const health = try alloc.alloc(u8, @intCast(hh.payload_len));
    try protocol.readPayload(fd, hh, health);
    try jb.raw(",\"health\":");
    try jb.raw(health);

    // --- RTT sweep ---
    try jb.raw(",\"rtt\":[");
    const iters: usize = if (opts.quick) RTT_ITERS_QUICK else RTT_ITERS;
    const samples = try alloc.alloc(u64, iters);
    const ping_buf = try alloc.alloc(u8, RTT_SIZES[RTT_SIZES.len - 1]);
    fillPattern(ping_buf);
    const pong_buf = try alloc.alloc(u8, ping_buf.len);

    for (RTT_SIZES, 0..) |size, si| {
        if (si != 0) try jb.raw(",");
        const payload = ping_buf[0..size];
        var i: usize = 0;
        while (i < RTT_WARMUP) : (i += 1) try pingOnce(fd, payload, pong_buf[0..size]);
        i = 0;
        while (i < iters) : (i += 1) {
            const t0 = sys.monotonicNs();
            try pingOnce(fd, payload, pong_buf[0..size]);
            samples[i] = sys.monotonicNs() - t0;
        }
        const s = stats.summarize(samples[0..iters]);
        out.print("  rtt {d:>5} B: p50 {d:>8.1} us  p95 {d:>8.1} us  p99 {d:>8.1} us  min {d:>8.1} us\n", .{
            size, usFloat(s.p50), usFloat(s.p95), usFloat(s.p99), usFloat(s.min),
        });
        try jb.print("{{\"payload_bytes\":{d},\"iters\":{d},\"min_ns\":{d},\"mean_ns\":{d},\"p50_ns\":{d},\"p95_ns\":{d},\"p99_ns\":{d},\"max_ns\":{d}}}", .{
            size, iters, s.min, s.mean, s.p50, s.p95, s.p99, s.max,
        });
    }
    try jb.raw("]");

    // --- Bandwidth sweep ---
    try jb.raw(",\"bandwidth\":[");
    const blob = try std.heap.page_allocator.alloc(u8, BW_SIZES[BW_SIZES.len - 1]);
    defer std.heap.page_allocator.free(blob);
    fillPattern(blob);

    for (BW_SIZES, 0..) |size, si| {
        if (si != 0) try jb.raw(",");
        var best_gbps: f64 = 0;
        var sum_gbps: f64 = 0;
        var rep: usize = 0;
        while (rep < BW_REPS) : (rep += 1) {
            const gbps = try blobOnce(fd, blob[0..size]);
            best_gbps = @max(best_gbps, gbps);
            sum_gbps += gbps;
        }
        const mean_gbps = sum_gbps / @as(f64, @floatFromInt(BW_REPS));
        out.print("  bw  {d:>4} MiB: best {d:>7.3} GB/s  mean {d:>7.3} GB/s\n", .{ size >> 20, best_gbps, mean_gbps });
        try jb.print("{{\"payload_bytes\":{d},\"reps\":{d},\"best_gbytes_per_s\":{d:.4},\"mean_gbytes_per_s\":{d:.4}}}", .{
            size, BW_REPS, best_gbps, mean_gbps,
        });
    }
    try jb.raw("]");

    // --- Sustained transfer ---
    const sustained_ns = opts.sustained_secs * std.time.ns_per_s;
    var blobs: u64 = 0;
    var min_gbps: f64 = std.math.floatMax(f64);
    var max_gbps: f64 = 0;
    const t_start = sys.monotonicNs();
    while (sys.monotonicNs() - t_start < sustained_ns) {
        const gbps = try blobOnce(fd, blob[0..SUSTAINED_BLOB]);
        min_gbps = @min(min_gbps, gbps);
        max_gbps = @max(max_gbps, gbps);
        blobs += 1;
    }
    const elapsed_s = @as(f64, @floatFromInt(sys.monotonicNs() - t_start)) / std.time.ns_per_s;
    const total_gb = @as(f64, @floatFromInt(blobs * SUSTAINED_BLOB)) / 1e9;
    const agg = total_gb / elapsed_s;
    out.print("  sustained {d:.1} s: {d:.3} GB/s aggregate ({d} x 64MiB blobs, per-blob min {d:.3} / max {d:.3})\n", .{
        elapsed_s, agg, blobs, min_gbps, max_gbps,
    });
    try jb.print(",\"sustained\":{{\"blob_bytes\":{d},\"blobs\":{d},\"elapsed_s\":{d:.3},\"gbytes_per_s\":{d:.4},\"per_blob_min_gbps\":{d:.4},\"per_blob_max_gbps\":{d:.4}}}", .{
        SUSTAINED_BLOB, blobs, elapsed_s, agg, min_gbps, max_gbps,
    });

    try jb.raw("}");
}

fn pingOnce(fd: sys.fd_t, payload: []const u8, recv_buf: []u8) !void {
    try protocol.writeFrame(fd, .ping, payload);
    const hdr = try protocol.readHeader(fd);
    if (hdr.payload_len != payload.len) return protocol.FrameError.BadMagic;
    try protocol.readPayload(fd, hdr, recv_buf);
}

/// Send one blob, wait for the ack, return application-level GB/s.
fn blobOnce(fd: sys.fd_t, payload: []const u8) !f64 {
    const t0 = sys.monotonicNs();
    try protocol.writeFrame(fd, .blob, payload);
    const hdr = try protocol.readHeader(fd);
    var ack: u64 = 0;
    try protocol.readPayload(fd, hdr, std.mem.asBytes(&ack));
    if (ack != payload.len) return protocol.FrameError.BadMagic;
    const ns = sys.monotonicNs() - t0;
    return @as(f64, @floatFromInt(payload.len)) / 1e9 / (@as(f64, @floatFromInt(ns)) / std.time.ns_per_s);
}

fn fillPattern(buf: []u8) void {
    for (buf, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
}

fn usFloat(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

/// Short commit hash read straight from .git — no subprocess, no dependency on
/// git being installed on a cluster node.
fn gitCommit(alloc: std.mem.Allocator) []const u8 {
    const head = sys.readFileAllocZ(alloc, ".git/HEAD") catch return "unknown";
    const trimmed = std.mem.trim(u8, head, " \n\r");
    if (std.mem.startsWith(u8, trimmed, "ref: ")) {
        const ref = trimmed[5..];
        const ref_path = std.fmt.allocPrint(alloc, ".git/{s}", .{ref}) catch return "unknown";
        const hash = sys.readFileAllocZ(alloc, ref_path) catch return "unknown";
        const h = std.mem.trim(u8, hash, " \n\r");
        return if (h.len >= 7) h[0..7] else "unknown";
    }
    return if (trimmed.len >= 7) trimmed[0..7] else "unknown";
}
