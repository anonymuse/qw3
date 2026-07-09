//! Latency/throughput sample summaries. Percentiles use the nearest-rank method
//! on sorted samples; good enough for benchmark reporting and fully deterministic.

const std = @import("std");

pub const Summary = struct {
    n: usize,
    min: u64,
    max: u64,
    mean: u64,
    p50: u64,
    p95: u64,
    p99: u64,
};

/// Sorts `samples` in place and summarizes. Asserts non-empty.
pub fn summarize(samples: []u64) Summary {
    std.debug.assert(samples.len > 0);
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    var sum: u128 = 0;
    for (samples) |s| sum += s;
    return .{
        .n = samples.len,
        .min = samples[0],
        .max = samples[samples.len - 1],
        .mean = @intCast(sum / samples.len),
        .p50 = percentile(samples, 50),
        .p95 = percentile(samples, 95),
        .p99 = percentile(samples, 99),
    };
}

/// Classic nearest-rank percentile of an already-sorted slice:
/// rank = ceil(p/100 * n), clamped to [1, n].
pub fn percentile(sorted: []const u64, p: u64) u64 {
    std.debug.assert(sorted.len > 0 and p <= 100);
    const rank = @max((p * sorted.len + 99) / 100, 1);
    return sorted[@min(rank, sorted.len) - 1];
}

test "summary of a known distribution" {
    var samples = [_]u64{ 90, 10, 20, 30, 40, 50, 60, 70, 80, 100 };
    const s = summarize(&samples);
    try std.testing.expectEqual(@as(u64, 10), s.min);
    try std.testing.expectEqual(@as(u64, 100), s.max);
    try std.testing.expectEqual(@as(u64, 55), s.mean);
    try std.testing.expectEqual(@as(u64, 50), s.p50);
    try std.testing.expectEqual(@as(u64, 100), s.p99);
}

test "percentile of single sample" {
    const one = [_]u64{42};
    try std.testing.expectEqual(@as(u64, 42), percentile(&one, 99));
}
