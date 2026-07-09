//! Host facts for health reporting and run metadata. Darwin-only where noted;
//! failures degrade to zero/unknown rather than aborting a benchmark.

const std = @import("std");

pub const HOSTNAME_MAX = 256;

pub fn hostname(buf: *[HOSTNAME_MAX]u8) []const u8 {
    if (std.c.gethostname(buf, buf.len) != 0) return "unknown";
    return std.mem.sliceTo(buf, 0);
}

/// Total physical memory in bytes (Darwin `hw.memsize`), or 0 if unavailable.
pub fn memTotalBytes() u64 {
    var value: u64 = 0;
    var len: usize = @sizeOf(u64);
    const rc = std.c.sysctlbyname("hw.memsize", &value, &len, null, 0);
    if (rc != 0) return 0;
    return value;
}

/// Machine chip string (Darwin `machdep.cpu.brand_string`), or "unknown".
pub fn chipBrand(buf: []u8) []const u8 {
    var len: usize = buf.len;
    const rc = std.c.sysctlbyname("machdep.cpu.brand_string", buf.ptr, &len, null, 0);
    if (rc != 0 or len == 0) return "unknown";
    // sysctl includes the trailing NUL in len
    return std.mem.sliceTo(buf[0..len], 0);
}

test "memTotalBytes returns something plausible on darwin" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    try std.testing.expect(memTotalBytes() > 1024 * 1024 * 1024);
}
