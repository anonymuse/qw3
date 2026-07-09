//! Minimal JSON accumulation buffer. Every DS5 benchmark emits machine-readable
//! JSON (coding standard #4); this keeps that dependency-free and immune to
//! std.Io writer churn. Values written via `print` are caller-controlled — this
//! is an output builder, not a general serializer.

const std = @import("std");

pub const JsonBuf = struct {
    alloc: std.mem.Allocator,
    list: std.ArrayList(u8) = .empty,

    pub fn init(alloc: std.mem.Allocator) JsonBuf {
        return .{ .alloc = alloc };
    }

    pub fn print(self: *JsonBuf, comptime fmt: []const u8, args: anytype) !void {
        const s = try std.fmt.allocPrint(self.alloc, fmt, args);
        defer self.alloc.free(s);
        try self.list.appendSlice(self.alloc, s);
    }

    pub fn raw(self: *JsonBuf, s: []const u8) !void {
        try self.list.appendSlice(self.alloc, s);
    }

    /// Append a JSON string literal with minimal escaping (quote and backslash).
    pub fn str(self: *JsonBuf, s: []const u8) !void {
        try self.list.append(self.alloc, '"');
        for (s) |c| {
            switch (c) {
                '"', '\\' => {
                    try self.list.append(self.alloc, '\\');
                    try self.list.append(self.alloc, c);
                },
                '\n' => try self.list.appendSlice(self.alloc, "\\n"),
                else => try self.list.append(self.alloc, c),
            }
        }
        try self.list.append(self.alloc, '"');
    }

    pub fn items(self: *const JsonBuf) []const u8 {
        return self.list.items;
    }
};

test "jsonbuf builds a small object" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var jb = JsonBuf.init(arena.allocator());
    try jb.raw("{\"name\":");
    try jb.str("node \"a\"");
    try jb.print(",\"port\":{d}}}", .{4750});
    try std.testing.expectEqualStrings("{\"name\":\"node \\\"a\\\"\",\"port\":4750}", jb.items());
}
