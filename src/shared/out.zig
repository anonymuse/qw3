//! Stdout/stderr helpers that stay off the churning std.Io writer APIs.

const std = @import("std");
const sys = @import("sys.zig");

/// Print to stdout. Formats into a stack buffer; oversized output is truncated,
/// which is acceptable for human-readable summaries (JSON goes to files).
pub fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [8192]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..];
    sys.writeStdout(s);
}

/// Progress/status lines go to stderr so stdout stays pipeable.
pub fn status(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
}
