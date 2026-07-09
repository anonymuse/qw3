//! Thin libc layer. DS5 talks to the OS through std.c directly: sockets, file
//! descriptors, and clocks with no buffering or scheduling the project didn't
//! write. This keeps benchmark numbers honest and insulates the project from
//! std.Io API churn (Zig 0.15/0.16 rewrote the higher-level I/O stack).

const std = @import("std");
const c = std.c;

pub const fd_t = c.fd_t;

pub const Error = error{
    SyscallFailed,
    ConnectionClosed,
};

/// Monotonic nanoseconds; not correlated across nodes.
pub fn monotonicNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(.MONOTONIC_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Wall-clock seconds since the Unix epoch (run metadata only, never timing).
pub fn epochSeconds() i64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(.REALTIME, &ts);
    return @intCast(ts.sec);
}

pub fn closeFd(fd: fd_t) void {
    _ = c.close(fd);
}

pub fn writeAllFd(fd: fd_t, bytes: []const u8) Error!void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = c.write(fd, bytes.ptr + off, bytes.len - off);
        if (n < 0) {
            if (interrupted()) continue;
            return Error.SyscallFailed;
        }
        if (n == 0) return Error.ConnectionClosed;
        off += @intCast(n);
    }
}

pub fn readAllFd(fd: fd_t, buf: []u8) Error!void {
    var off: usize = 0;
    while (off < buf.len) {
        const n = c.read(fd, buf.ptr + off, buf.len - off);
        if (n < 0) {
            if (interrupted()) continue;
            return Error.SyscallFailed;
        }
        if (n == 0) return Error.ConnectionClosed;
        off += @intCast(n);
    }
}

fn interrupted() bool {
    return std.c._errno().* == @intFromEnum(c.E.INTR);
}

pub fn writeStdout(s: []const u8) void {
    writeAllFd(1, s) catch {};
}

/// Read a whole file into a null-terminated buffer.
pub fn readFileAllocZ(alloc: std.mem.Allocator, path: []const u8) ![:0]u8 {
    const pathz = try alloc.dupeZ(u8, path);
    defer alloc.free(pathz);
    const fd = c.open(pathz, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.OpenFailed;
    defer closeFd(fd);
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(alloc);
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = c.read(fd, &buf, buf.len);
        if (n < 0) {
            if (interrupted()) continue;
            return error.ReadFailed;
        }
        if (n == 0) break;
        try list.appendSlice(alloc, buf[0..@intCast(n)]);
    }
    return list.toOwnedSliceSentinel(alloc, 0);
}

/// Create/truncate a file and write all bytes.
pub fn writeFileTrunc(alloc: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    const pathz = try alloc.dupeZ(u8, path);
    defer alloc.free(pathz);
    const fd = c.open(pathz, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c_uint, 0o644));
    if (fd < 0) return error.OpenFailed;
    defer closeFd(fd);
    try writeAllFd(fd, bytes);
}

/// mkdir -p for relative paths. Existing directories are fine.
pub fn mkdirPath(alloc: std.mem.Allocator, path: []const u8) !void {
    var i: usize = 0;
    while (i <= path.len) : (i += 1) {
        if (i != path.len and path[i] != '/') continue;
        if (i == 0) continue;
        const prefixz = try alloc.dupeZ(u8, path[0..i]);
        defer alloc.free(prefixz);
        _ = c.mkdir(prefixz, 0o755);
    }
}

test "monotonic clock advances" {
    const a = monotonicNs();
    const b = monotonicNs();
    try std.testing.expect(b >= a);
}

test "mkdirPath and file round-trip" {
    const alloc = std.testing.allocator;
    try mkdirPath(alloc, ".zig-cache/tmp/ds5-sys-test");
    try writeFileTrunc(alloc, ".zig-cache/tmp/ds5-sys-test/x.txt", "hello");
    const back = try readFileAllocZ(alloc, ".zig-cache/tmp/ds5-sys-test/x.txt");
    defer alloc.free(back);
    try std.testing.expectEqualStrings("hello", back);
}
