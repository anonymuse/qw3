//! Payload checksums for transport correctness tests (Benchmark Spec Phase 1 gate:
//! "activation packets are checksummed and traceable").

const std = @import("std");

pub fn crc32(bytes: []const u8) u32 {
    return std.hash.crc.Crc32.hash(bytes);
}

test "crc32 matches the IEEE check value" {
    // Standard CRC-32/ISO-HDLC check: crc("123456789") == 0xCBF43926
    try std.testing.expectEqual(@as(u32, 0xCBF43926), crc32("123456789"));
}
