//! DS5 activation packet header, per the Model Runtime and Placement Spec v0.2 §8.
//!
//! Rule (spec §8.2): a layer sends at most ONE activation packet per destination
//! node — never one packet per expert. The header is followed by an aligned
//! hidden-vector payload on the wire.
//!
//! Not used by the M0 benchmarks yet; defined and size-locked now so the wire
//! contract is versioned from day one.

const std = @import("std");

pub const HiddenDtype = enum(u16) {
    f16 = 1,
    bf16 = 2,
    f32 = 3,
};

pub const ActivationHeader = extern struct {
    version: u16,
    layer_id: u16,
    source_node: u16,
    destination_node: u16,
    hidden_dtype: u16,
    expert_count: u16,
    sequence_id: u32,
    token_id: u64,
    trace_id: u64,
    expert_ids: [8]u16,
    gate_weights: [8]f32,
};

pub const ACTIVATION_HEADER_SIZE: usize = 80;

test "activation header layout is stable at 80 bytes" {
    try std.testing.expectEqual(ACTIVATION_HEADER_SIZE, @sizeOf(ActivationHeader));
}

test "activation header round-trips through bytes" {
    var hdr = ActivationHeader{
        .version = 1,
        .layer_id = 46,
        .source_node = 1,
        .destination_node = 2,
        .hidden_dtype = @intFromEnum(HiddenDtype.f16),
        .expert_count = 8,
        .sequence_id = 12345,
        .token_id = 987654321,
        .trace_id = 0xDEADBEEF,
        .expert_ids = .{ 3, 17, 42, 63, 77, 90, 101, 127 },
        .gate_weights = .{ 0.31, 0.20, 0.14, 0.10, 0.09, 0.07, 0.05, 0.04 },
    };
    const bytes = std.mem.asBytes(&hdr);
    var back: ActivationHeader = undefined;
    @memcpy(std.mem.asBytes(&back), bytes);
    try std.testing.expectEqual(@as(u16, 46), back.layer_id);
    try std.testing.expectEqual(@as(u16, 127), back.expert_ids[7]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.31), back.gate_weights[0], 1e-6);
}
