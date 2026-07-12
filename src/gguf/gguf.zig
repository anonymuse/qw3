//! GGUF v3 parser (workstream W1). Implements the frozen `assertGgufApi`
//! contract from contracts.zig: mmap the artifact, index metadata + tensor
//! infos, and hand out zero-copy TensorViews whose `data` points into the map
//! (later wrapped by Metal's newBufferWithBytesNoCopy — the map base is
//! page-aligned, tensor offsets are `general.alignment`-multiples).
//!
//! File layout (all little-endian, GGUF v3):
//!   header   magic u32 = 0x46554747 ("GGUF"), version u32 = 3,
//!            n_tensors u64, n_kv u64
//!   kv       key (u64-len string), value_type u32, value; arrays are
//!            elem_type u32 + count u64 + elements and are skipped wholesale
//!            (tokenizer vocab lists can be megabytes we never look at)
//!   tensors  name, n_dims u32, ne[n_dims] u64 (ne[0] first), ggml type u32,
//!            offset u64 relative to the data section
//!   data     starts at alignForward(end of tensor infos, general.alignment
//!            [default 32]); every tensor offset is alignment-aligned
//!
//! Every tensor's byte size is validated against contracts.TensorDesc
//! .byteSize() and the file length at open() time with overflow-checked
//! arithmetic, so downstream code can trust TensorView.data.len blindly.
//! Quantized dtypes beyond Q8_0 (K-quants, I-quants) parse structurally via
//! the frozen block geometry in contracts.Dtype; dequantization is kernel
//! work, except the CPU Q8_0 row helper at the bottom (M2 embedding lookup).

const std = @import("std");
const contracts = @import("../shared/contracts.zig");
const sys = @import("../shared/sys.zig");

pub const GgufError = contracts.GgufError;

const MAGIC: u32 = 0x46554747;
const VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u32 = 32;
/// GGUF permits arrays of arrays; bound the nesting so a malicious file
/// cannot blow the parser stack.
const MAX_ARRAY_DEPTH = 8;

/// GGUF metadata value types (wire ids, GGUF v3).
const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,

    /// Wire size for fixed-size values; null for string/array.
    fn scalarSize(self: ValueType) ?usize {
        return switch (self) {
            .uint8, .int8, .boolean => 1,
            .uint16, .int16 => 2,
            .uint32, .int32, .float32 => 4,
            .uint64, .int64, .float64 => 8,
            .string, .array => null,
        };
    }
};

/// Decoded metadata value. Only the types the engine reads through the frozen
/// getters are kept; everything else (small ints, arrays, ...) is consumed
/// from the wire and recorded as `skipped` so lookups return null cleanly.
const MetaValue = union(enum) {
    u32v: u32,
    u64v: u64,
    f32v: f32,
    boolv: bool,
    str: []const u8, // slice into the mmap
    skipped,
};

const Kv = struct {
    key: []const u8, // slice into the mmap
    val: MetaValue,
};

/// Bounds-checked little-endian cursor over the mapped file.
const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn remaining(self: *const Reader) usize {
        return self.buf.len - self.pos;
    }

    fn take(self: *Reader, n: usize) GgufError![]const u8 {
        if (self.remaining() < n) return GgufError.Truncated;
        const s = self.buf[self.pos..][0..n];
        self.pos += n;
        return s;
    }

    fn takeU32(self: *Reader) GgufError!u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }

    fn takeU64(self: *Reader) GgufError!u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }

    /// u64-length-prefixed string; the slice aliases the map.
    fn takeStr(self: *Reader) GgufError![]const u8 {
        const len = try self.takeU64();
        if (len > self.remaining()) return GgufError.Truncated;
        return self.take(@intCast(len));
    }
};

fn parseValue(r: *Reader, vt_raw: u32, depth: u32) GgufError!MetaValue {
    const vt = std.enums.fromInt(ValueType, vt_raw) orelse return GgufError.BadMetadata;
    switch (vt) {
        .uint32 => return .{ .u32v = try r.takeU32() },
        .uint64 => return .{ .u64v = try r.takeU64() },
        .float32 => return .{ .f32v = @bitCast(try r.takeU32()) },
        .boolean => return .{ .boolv = (try r.take(1))[0] != 0 },
        .string => return .{ .str = try r.takeStr() },
        .array => {
            if (depth >= MAX_ARRAY_DEPTH) return GgufError.BadMetadata;
            const elem_raw = try r.takeU32();
            const count = try r.takeU64();
            const elem = std.enums.fromInt(ValueType, elem_raw) orelse return GgufError.BadMetadata;
            if (elem.scalarSize()) |sz| {
                const bytes = std.math.mul(u64, count, sz) catch return GgufError.Truncated;
                if (bytes > r.remaining()) return GgufError.Truncated;
                _ = try r.take(@intCast(bytes));
            } else {
                var i: u64 = 0;
                while (i < count) : (i += 1) _ = try parseValue(r, elem_raw, depth + 1);
            }
            return .skipped;
        },
        // Scalar types no frozen getter surfaces: consume and forget.
        else => {
            _ = try r.take(vt.scalarSize().?);
            return .skipped;
        },
    }
}

/// TensorDesc.byteSize() with overflow-checked arithmetic; null means the
/// ne[] values are unrepresentable (oversized or not block-aligned).
fn checkedByteSize(dtype: contracts.Dtype, ne: [contracts.MAX_DIMS]u64) ?u64 {
    const be = dtype.blockElems();
    if (ne[0] % be != 0) return null;
    var total = std.math.mul(u64, ne[0] / be, dtype.blockBytes()) catch return null;
    for (ne[1..]) |n| total = std.math.mul(u64, total, n) catch return null;
    return total;
}

fn findKv(kvs: []const Kv, key: []const u8) ?MetaValue {
    for (kvs) |kv| {
        if (std.mem.eql(u8, kv.key, key)) return kv.val;
    }
    return null;
}

pub const Model = struct {
    alloc: std.mem.Allocator,
    map: sys.MappedFile,
    kvs: []Kv,
    tensors: []contracts.TensorView,
    /// tensor name (slice into map) -> index into `tensors`
    names: std.StringHashMapUnmanaged(usize),

    pub fn open(alloc: std.mem.Allocator, path: []const u8) GgufError!Model {
        var map = sys.mmapFileRead(alloc, path) catch |err| return switch (err) {
            error.OpenFailed => GgufError.OpenFailed,
            error.MmapFailed => GgufError.MmapFailed,
            error.OutOfMemory => GgufError.OutOfMemory,
        };
        errdefer map.unmap();
        const buf = map.data;
        var r = Reader{ .buf = buf };

        if (try r.takeU32() != MAGIC) return GgufError.BadMagic;
        if (try r.takeU32() != VERSION) return GgufError.UnsupportedVersion;
        const n_tensors = try r.takeU64();
        const n_kv = try r.takeU64();
        // Cheapest possible encodings: kv >= 12 bytes, tensor info >= 32.
        // Rejecting absurd counts up front keeps allocations honest.
        if (n_kv > r.remaining() / 12) return GgufError.Truncated;
        if (n_tensors > r.remaining() / 32) return GgufError.Truncated;

        const kvs = alloc.alloc(Kv, @intCast(n_kv)) catch return GgufError.OutOfMemory;
        errdefer alloc.free(kvs);
        for (kvs) |*kv| {
            const key = try r.takeStr();
            const vt = try r.takeU32();
            kv.* = .{ .key = key, .val = try parseValue(&r, vt, 0) };
        }

        const alignment: u64 = switch (findKv(kvs, "general.alignment") orelse MetaValue{ .u32v = DEFAULT_ALIGNMENT }) {
            .u32v => |a| a,
            else => return GgufError.BadMetadata,
        };
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return GgufError.BadMetadata;

        const tensors = alloc.alloc(contracts.TensorView, @intCast(n_tensors)) catch return GgufError.OutOfMemory;
        errdefer alloc.free(tensors);
        // Data offsets are relative to the (not yet known) data section, so
        // stash them in `data.ptr`-free form first: reuse the view and patch
        // the slices after the section start is computed.
        const rel_offsets = alloc.alloc(u64, tensors.len) catch return GgufError.OutOfMemory;
        defer alloc.free(rel_offsets);

        var names: std.StringHashMapUnmanaged(usize) = .empty;
        errdefer names.deinit(alloc);
        const cap = std.math.cast(u32, tensors.len) orelse return GgufError.Truncated;
        names.ensureTotalCapacity(alloc, cap) catch return GgufError.OutOfMemory;

        for (tensors, rel_offsets, 0..) |*t, *rel, i| {
            const name = try r.takeStr();
            const n_dims = try r.takeU32();
            if (n_dims < 1 or n_dims > contracts.MAX_DIMS) return GgufError.BadMetadata;
            var ne = [_]u64{ 1, 1, 1, 1 };
            for (ne[0..n_dims]) |*d| {
                d.* = try r.takeU64();
                if (d.* == 0) return GgufError.BadMetadata;
            }
            const dtype_raw = try r.takeU32();
            const dtype = std.enums.fromInt(contracts.Dtype, dtype_raw) orelse return GgufError.BadMetadata;
            rel.* = try r.takeU64();
            if (rel.* % alignment != 0) return GgufError.BadMetadata;
            if (checkedByteSize(dtype, ne) == null) return GgufError.BadMetadata;
            t.* = .{
                .name = name,
                .desc = .{ .dtype = dtype, .n_dims = n_dims, .ne = ne },
                .data = &.{},
            };
            const gop = names.getOrPutAssumeCapacity(name);
            if (gop.found_existing) return GgufError.BadMetadata; // duplicate tensor name
            gop.value_ptr.* = i;
        }

        const data_start = std.mem.alignForward(usize, r.pos, @intCast(alignment));
        if (data_start > buf.len) return GgufError.Truncated;
        const data = buf[data_start..];
        for (tensors, rel_offsets) |*t, rel| {
            const size = t.desc.byteSize(); // overflow-checked above
            const end = std.math.add(u64, rel, size) catch return GgufError.Truncated;
            if (end > data.len) return GgufError.Truncated;
            t.data = data[@intCast(rel)..][0..@intCast(size)];
        }

        return .{ .alloc = alloc, .map = map, .kvs = kvs, .tensors = tensors, .names = names };
    }

    pub fn deinit(self: *Model) void {
        self.names.deinit(self.alloc);
        self.alloc.free(self.tensors);
        self.alloc.free(self.kvs);
        self.map.unmap();
        self.* = undefined;
    }

    pub fn tensorCount(self: *const Model) usize {
        return self.tensors.len;
    }

    pub fn tensorAt(self: *const Model, i: usize) contracts.TensorView {
        return self.tensors[i];
    }

    pub fn tensorByName(self: *const Model, name: []const u8) ?contracts.TensorView {
        const i = self.names.get(name) orelse return null;
        return self.tensors[i];
    }

    // Metadata getters: null on missing key or wrong wire type (frozen rule).

    pub fn metaU32(self: *const Model, key: []const u8) ?u32 {
        return switch (findKv(self.kvs, key) orelse return null) {
            .u32v => |v| v,
            else => null,
        };
    }

    pub fn metaU64(self: *const Model, key: []const u8) ?u64 {
        return switch (findKv(self.kvs, key) orelse return null) {
            .u64v => |v| v,
            else => null,
        };
    }

    pub fn metaF32(self: *const Model, key: []const u8) ?f32 {
        return switch (findKv(self.kvs, key) orelse return null) {
            .f32v => |v| v,
            else => null,
        };
    }

    pub fn metaBool(self: *const Model, key: []const u8) ?bool {
        return switch (findKv(self.kvs, key) orelse return null) {
            .boolv => |v| v,
            else => null,
        };
    }

    pub fn metaStr(self: *const Model, key: []const u8) ?[]const u8 {
        return switch (findKv(self.kvs, key) orelse return null) {
            .str => |v| v,
            else => null,
        };
    }

    fn requireU32(self: *const Model, key: []const u8) GgufError!u32 {
        return self.metaU32(key) orelse GgufError.MissingKey;
    }

    /// Build ModelConfig from the qwen3moe.* keys (ADR-005 §5). vocab_size
    /// falls back to token_embd.weight's row count when the key is absent
    /// (some converters only emit the tokenizer vocab array).
    pub fn config(self: *const Model) GgufError!contracts.ModelConfig {
        const arch = self.metaStr("general.architecture") orelse return GgufError.MissingKey;
        if (!std.mem.eql(u8, arch, "qwen3moe")) return GgufError.BadMetadata;
        const vocab = self.metaU32("qwen3moe.vocab_size") orelse vs: {
            const emb = self.tensorByName("token_embd.weight") orelse return GgufError.MissingKey;
            break :vs std.math.cast(u32, emb.desc.ne[1]) orelse return GgufError.BadMetadata;
        };
        return .{
            .n_layers = try self.requireU32("qwen3moe.block_count"),
            .hidden_dim = try self.requireU32("qwen3moe.embedding_length"),
            .n_q_heads = try self.requireU32("qwen3moe.attention.head_count"),
            .n_kv_heads = try self.requireU32("qwen3moe.attention.head_count_kv"),
            .head_dim = try self.requireU32("qwen3moe.attention.key_length"),
            .n_experts = try self.requireU32("qwen3moe.expert_count"),
            .top_k = try self.requireU32("qwen3moe.expert_used_count"),
            .expert_ffn_dim = try self.requireU32("qwen3moe.expert_feed_forward_length"),
            .vocab_size = vocab,
            .rms_eps = self.metaF32("qwen3moe.attention.layer_norm_rms_epsilon") orelse return GgufError.MissingKey,
            .rope_theta = self.metaF32("qwen3moe.rope.freq_base") orelse return GgufError.MissingKey,
            // Not a GGUF key; constant for the qwen3moe arch (ADR-005 §5).
            .norm_topk_prob = true,
            .max_ctx = try self.requireU32("qwen3moe.context_length"),
        };
    }
};

// ---------------------------------------------------------------------------
// CPU Q8_0 dequant helper (M2 embedding lookup). Frozen semantics (ADR-005
// §1): value = f32(f16 scale) · i8 q, blocks of 32. GPU kernels own all other
// dequant paths.
// ---------------------------------------------------------------------------

/// Dequantize one Q8_0 tensor row. `row_bytes` is out.len/32 blocks of 34
/// bytes (e.g. `view.data[row * desc.dtype.rowBytes(desc.ne[0]) ..]`).
pub fn dequantRowQ8_0(row_bytes: []const u8, out: []f32) void {
    std.debug.assert(out.len % 32 == 0);
    std.debug.assert(row_bytes.len == out.len / 32 * 34);
    var b: usize = 0;
    while (b * 32 < out.len) : (b += 1) {
        const blk = row_bytes[b * 34 ..][0..34];
        const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk[0..2], .little))));
        for (blk[2..34], out[b * 32 ..][0..32]) |q, *o| {
            o.* = scale * @as(f32, @floatFromInt(@as(i8, @bitCast(q))));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const fixture = @import("../shared/fixture.zig");

test "frozen gguf api conformance" {
    comptime contracts.assertGgufApi(@This());
}

const SYNTH_PATH = "tests/fixtures/synthetic/model.gguf";

fn openSynth(alloc: std.mem.Allocator) !?Model {
    // Fixture set is committed; skip only if it was stripped from a checkout.
    return Model.open(alloc, SYNTH_PATH) catch |err| switch (err) {
        GgufError.OpenFailed => null,
        else => err,
    };
}

test "synthetic model: config matches SYNTH_TINY field by field" {
    var m = try openSynth(testing.allocator) orelse return error.SkipZigTest;
    defer m.deinit();
    const cfg = try m.config();
    const want = contracts.SYNTH_TINY;
    try testing.expectEqual(want.n_layers, cfg.n_layers);
    try testing.expectEqual(want.hidden_dim, cfg.hidden_dim);
    try testing.expectEqual(want.n_q_heads, cfg.n_q_heads);
    try testing.expectEqual(want.n_kv_heads, cfg.n_kv_heads);
    try testing.expectEqual(want.head_dim, cfg.head_dim);
    try testing.expectEqual(want.n_experts, cfg.n_experts);
    try testing.expectEqual(want.top_k, cfg.top_k);
    try testing.expectEqual(want.expert_ffn_dim, cfg.expert_ffn_dim);
    try testing.expectEqual(want.vocab_size, cfg.vocab_size);
    try testing.expectEqual(want.rms_eps, cfg.rms_eps);
    try testing.expectEqual(want.rope_theta, cfg.rope_theta);
    try testing.expectEqual(want.norm_topk_prob, cfg.norm_topk_prob);
    try testing.expectEqual(want.max_ctx, cfg.max_ctx);
}

fn expectTensor(m: *const Model, name: []const u8, dtype: contracts.Dtype, dims: []const u64) !void {
    const t = m.tensorByName(name) orelse return error.TensorMissing;
    try testing.expectEqualStrings(name, t.name);
    try testing.expectEqual(dtype, t.desc.dtype);
    try testing.expectEqual(@as(u32, @intCast(dims.len)), t.desc.n_dims);
    const want = contracts.TensorDesc.init(dtype, dims);
    try testing.expectEqual(want.ne, t.desc.ne);
    try testing.expectEqual(want.byteSize(), t.data.len);
    // Zero-copy: data must point into the mmap.
    const base = @intFromPtr(m.map.data.ptr);
    try testing.expect(@intFromPtr(t.data.ptr) >= base);
    try testing.expect(@intFromPtr(t.data.ptr) + t.data.len <= base + m.map.data.len);
}

test "synthetic model: tensor index, dtypes, shapes" {
    var m = try openSynth(testing.allocator) orelse return error.SkipZigTest;
    defer m.deinit();
    try testing.expectEqual(@as(usize, 51), m.tensorCount());

    const hidden = contracts.SYNTH_TINY.hidden_dim; // 256
    const ffn = contracts.SYNTH_TINY.expert_ffn_dim; // 128
    try expectTensor(&m, "token_embd.weight", .q8_0, &.{ hidden, 512 });
    try expectTensor(&m, "output.weight", .q8_0, &.{ hidden, 512 });
    try expectTensor(&m, "output_norm.weight", .f32, &.{hidden});
    try expectTensor(&m, "blk.0.attn_q.weight", .q8_0, &.{ hidden, 256 });
    try expectTensor(&m, "blk.0.attn_k.weight", .q8_0, &.{ hidden, 128 });
    try expectTensor(&m, "blk.0.attn_v.weight", .q8_0, &.{ hidden, 128 });
    try expectTensor(&m, "blk.0.attn_output.weight", .q8_0, &.{ 256, hidden });
    try expectTensor(&m, "blk.0.attn_norm.weight", .f32, &.{hidden});
    try expectTensor(&m, "blk.0.attn_q_norm.weight", .f32, &.{64});
    try expectTensor(&m, "blk.0.attn_k_norm.weight", .f32, &.{64});
    try expectTensor(&m, "blk.0.ffn_norm.weight", .f32, &.{hidden});
    try expectTensor(&m, "blk.0.ffn_gate_inp.weight", .f32, &.{ hidden, 8 });
    try expectTensor(&m, "blk.0.ffn_gate_exps.weight", .q8_0, &.{ hidden, ffn, 8 });
    try expectTensor(&m, "blk.0.ffn_up_exps.weight", .q8_0, &.{ hidden, ffn, 8 });
    try expectTensor(&m, "blk.3.ffn_down_exps.weight", .q8_0, &.{ ffn, hidden, 8 });
    try testing.expect(m.tensorByName("blk.4.attn_q.weight") == null);
    try testing.expect(m.tensorByName("no.such.tensor") == null);

    // tensorAt covers every index and agrees with the name index.
    for (0..m.tensorCount()) |i| {
        const t = m.tensorAt(i);
        const again = m.tensorByName(t.name) orelse return error.TensorMissing;
        try testing.expectEqual(t.data.ptr, again.data.ptr);
    }
}

test "synthetic model: metadata getters" {
    var m = try openSynth(testing.allocator) orelse return error.SkipZigTest;
    defer m.deinit();
    try testing.expectEqualStrings("qwen3moe", m.metaStr("general.architecture").?);
    try testing.expectEqualStrings("ds5-synthetic-tiny", m.metaStr("general.name").?);
    try testing.expectEqual(@as(u32, 32), m.metaU32("general.alignment").?);
    try testing.expectEqual(@as(u32, 4), m.metaU32("qwen3moe.block_count").?);
    // Wrong-type and missing lookups return null, never error.
    try testing.expect(m.metaU64("qwen3moe.block_count") == null);
    try testing.expect(m.metaF32("general.architecture") == null);
    try testing.expect(m.metaBool("qwen3moe.block_count") == null);
    try testing.expect(m.metaU32("qwen3moe.not_a_key") == null);
}

test "synthetic model: tensor bytes match committed DS5T fixtures" {
    const alloc = testing.allocator;
    var m = try openSynth(alloc) orelse return error.SkipZigTest;
    defer m.deinit();
    const pairs = [_]struct { ds5t: []const u8, tensor: []const u8 }{
        .{ .ds5t = "tests/fixtures/synthetic/l0_attn_q.weight.ds5t", .tensor = "blk.0.attn_q.weight" },
        .{ .ds5t = "tests/fixtures/synthetic/l0_router.weight.ds5t", .tensor = "blk.0.ffn_gate_inp.weight" },
        .{ .ds5t = "tests/fixtures/synthetic/l0_attn_norm.weight.ds5t", .tensor = "blk.0.attn_norm.weight" },
        .{ .ds5t = "tests/fixtures/synthetic/l0_experts.gate.ds5t", .tensor = "blk.0.ffn_gate_exps.weight" },
    };
    for (pairs) |p| {
        var golden = try fixture.loadTensor(alloc, p.ds5t);
        defer golden.free(alloc);
        const t = m.tensorByName(p.tensor) orelse return error.TensorMissing;
        try testing.expectEqual(golden.desc.dtype, t.desc.dtype);
        try testing.expectEqualSlices(u8, golden.data, t.data);
    }
}

test "dequantRowQ8_0: hand-built blocks, exact semantics" {
    // Two blocks with different scales; q spans the i8 range.
    var row: [68]u8 = undefined;
    const s0: f16 = 0.5;
    const s1: f16 = -2.0;
    std.mem.writeInt(u16, row[0..2], @bitCast(s0), .little);
    for (0..32) |i| row[2 + i] = @bitCast(@as(i8, @intCast(@as(i32, @intCast(i)) - 16)));
    std.mem.writeInt(u16, row[34..36], @bitCast(s1), .little);
    for (0..32) |i| row[36 + i] = @bitCast(@as(i8, @intCast(@as(i32, @intCast(i)) * 3 - 48)));

    var out: [64]f32 = undefined;
    dequantRowQ8_0(&row, &out);
    for (0..32) |i| {
        const q0: f32 = @floatFromInt(@as(i32, @intCast(i)) - 16);
        const q1: f32 = @floatFromInt(@as(i32, @intCast(i)) * 3 - 48);
        try testing.expectEqual(@as(f32, 0.5) * q0, out[i]);
        try testing.expectEqual(@as(f32, -2.0) * q1, out[32 + i]);
    }
}

test "dequantRowQ8_0: synthetic embedding rows are sane" {
    var m = try openSynth(testing.allocator) orelse return error.SkipZigTest;
    defer m.deinit();
    const t = m.tensorByName("token_embd.weight") orelse return error.TensorMissing;
    const row_bytes = t.desc.dtype.rowBytes(t.desc.ne[0]); // 256/32*34
    var out: [256]f32 = undefined;
    var nonzero: usize = 0;
    for ([_]u64{ 0, 1, 511 }) |r| {
        const row = t.data[@intCast(r * row_bytes)..][0..@intCast(row_bytes)];
        dequantRowQ8_0(row, &out);
        for (out) |v| {
            try testing.expect(std.math.isFinite(v));
            try testing.expect(@abs(v) < 16.0); // synthetic init is ~N(0, small)
            if (v != 0) nonzero += 1;
        }
    }
    try testing.expect(nonzero > 512);
}

// --- hand-built GGUF buffers for the error paths ---------------------------

const TMP_DIR = ".zig-cache/tmp/ds5-gguf-test";

/// Minimal GGUF byte builder for tests.
const Builder = struct {
    alloc: std.mem.Allocator,
    list: std.ArrayList(u8) = .empty,

    fn deinit(self: *Builder) void {
        self.list.deinit(self.alloc);
    }

    fn u32v(self: *Builder, v: u32) !void {
        try self.list.appendSlice(self.alloc, std.mem.asBytes(&std.mem.nativeToLittle(u32, v)));
    }

    fn u64v(self: *Builder, v: u64) !void {
        try self.list.appendSlice(self.alloc, std.mem.asBytes(&std.mem.nativeToLittle(u64, v)));
    }

    fn bytes(self: *Builder, b: []const u8) !void {
        try self.list.appendSlice(self.alloc, b);
    }

    fn str(self: *Builder, s: []const u8) !void {
        try self.u64v(s.len);
        try self.bytes(s);
    }

    fn header(self: *Builder, n_tensors: u64, n_kv: u64) !void {
        try self.u32v(MAGIC);
        try self.u32v(VERSION);
        try self.u64v(n_tensors);
        try self.u64v(n_kv);
    }

    /// One f32 tensor `t` of ne = {2, 2}, offset 0, then the padded data
    /// section holding 16 bytes.
    fn oneTensorTail(self: *Builder) !void {
        try self.str("t");
        try self.u32v(2); // n_dims
        try self.u64v(2);
        try self.u64v(2);
        try self.u32v(@intFromEnum(contracts.Dtype.f32));
        try self.u64v(0); // offset
        try self.pad(DEFAULT_ALIGNMENT);
        const vals = [4]f32{ 1, 2, 3, 4 };
        try self.bytes(std.mem.sliceAsBytes(&vals));
    }

    fn pad(self: *Builder, alignment: usize) !void {
        while (self.list.items.len % alignment != 0)
            try self.list.append(self.alloc, 0);
    }

    fn write(self: *Builder, name: []const u8) ![]const u8 {
        try sys.mkdirPath(self.alloc, TMP_DIR);
        const path = try std.fmt.allocPrint(self.alloc, "{s}/{s}", .{ TMP_DIR, name });
        try sys.writeFileTrunc(self.alloc, path, self.list.items);
        return path;
    }
};

fn expectOpenError(path: []const u8, want: GgufError) !void {
    try testing.expectError(want, Model.open(testing.allocator, path));
}

test "hand-built gguf: valid file with skipped metadata types parses" {
    const alloc = testing.allocator;
    var b = Builder{ .alloc = alloc };
    defer b.deinit();
    try b.header(1, 12);
    // Every value type the getters do not surface must be skipped cleanly.
    try b.str("k.u8");
    try b.u32v(0);
    try b.bytes(&.{7});
    try b.str("k.i8");
    try b.u32v(1);
    try b.bytes(&.{0xff});
    try b.str("k.u16");
    try b.u32v(2);
    try b.bytes(&.{ 1, 0 });
    try b.str("k.i16");
    try b.u32v(3);
    try b.bytes(&.{ 1, 0 });
    try b.str("k.i32");
    try b.u32v(5);
    try b.u32v(123);
    try b.str("k.i64");
    try b.u32v(11);
    try b.u64v(123);
    try b.str("k.f64");
    try b.u32v(12);
    try b.u64v(0x3ff0000000000000);
    try b.str("k.arr_u32"); // array of 3 u32
    try b.u32v(9);
    try b.u32v(4);
    try b.u64v(3);
    try b.u32v(10);
    try b.u32v(20);
    try b.u32v(30);
    try b.str("k.arr_str"); // array of 2 strings (tokenizer-vocab shape)
    try b.u32v(9);
    try b.u32v(8);
    try b.u64v(2);
    try b.str("hello");
    try b.str("world");
    try b.str("k.arr_nested"); // array of 1 array of 2 u8
    try b.u32v(9);
    try b.u32v(9);
    try b.u64v(1);
    try b.u32v(0);
    try b.u64v(2);
    try b.bytes(&.{ 1, 2 });
    // ...and the ones we do read still resolve after all that skipping.
    try b.str("k.u32");
    try b.u32v(4);
    try b.u32v(77);
    try b.str("k.bool");
    try b.u32v(7);
    try b.bytes(&.{1});
    try b.oneTensorTail();
    const path = try b.write("skips.gguf");
    defer alloc.free(@constCast(path));

    var m = try Model.open(alloc, path);
    defer m.deinit();
    try testing.expectEqual(@as(u32, 77), m.metaU32("k.u32").?);
    try testing.expectEqual(true, m.metaBool("k.bool").?);
    // Skipped entries exist but surface as null through every getter.
    try testing.expect(m.metaU32("k.arr_u32") == null);
    try testing.expect(m.metaU64("k.i64") == null);
    try testing.expect(m.metaStr("k.arr_str") == null);
    const t = m.tensorAt(0);
    try testing.expectEqualStrings("t", t.name);
    const f = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, t.data)));
    try testing.expectEqual(@as(f32, 3), f[2]);
}

test "hand-built gguf: bad magic" {
    const alloc = testing.allocator;
    var b = Builder{ .alloc = alloc };
    defer b.deinit();
    try b.u32v(0xdeadbeef);
    try b.u32v(VERSION);
    try b.u64v(0);
    try b.u64v(0);
    const path = try b.write("badmagic.gguf");
    defer alloc.free(@constCast(path));
    try expectOpenError(path, GgufError.BadMagic);
}

test "hand-built gguf: unsupported version" {
    const alloc = testing.allocator;
    var b = Builder{ .alloc = alloc };
    defer b.deinit();
    try b.u32v(MAGIC);
    try b.u32v(2);
    try b.u64v(0);
    try b.u64v(0);
    const path = try b.write("badver.gguf");
    defer alloc.free(@constCast(path));
    try expectOpenError(path, GgufError.UnsupportedVersion);
}

test "hand-built gguf: truncated header and sections" {
    const alloc = testing.allocator;
    { // header cut mid-count
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.u32v(MAGIC);
        try b.u32v(VERSION);
        try b.bytes(&.{ 1, 2, 3 });
        const path = try b.write("trunc_hdr.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.Truncated);
    }
    { // promises 1 kv, delivers none
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(0, 1);
        const path = try b.write("trunc_kv.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.Truncated);
    }
    { // kv string length runs past EOF
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(0, 1);
        try b.u64v(1 << 40); // key length
        try b.bytes("xx");
        const path = try b.write("trunc_str.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.Truncated);
    }
    { // tensor data extends past the file end
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 0);
        try b.str("t");
        try b.u32v(2);
        try b.u64v(64);
        try b.u64v(64); // 16 KiB of f32
        try b.u32v(@intFromEnum(contracts.Dtype.f32));
        try b.u64v(0);
        try b.pad(DEFAULT_ALIGNMENT);
        try b.bytes(&.{ 1, 2, 3, 4 }); // nowhere near 16 KiB
        const path = try b.write("trunc_data.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.Truncated);
    }
}

test "hand-built gguf: bad tensor infos" {
    const alloc = testing.allocator;
    { // oversized ne: byte size overflows u64
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 0);
        try b.str("t");
        try b.u32v(2);
        try b.u64v(1 << 62);
        try b.u64v(1 << 62);
        try b.u32v(@intFromEnum(contracts.Dtype.f32));
        try b.u64v(0);
        const path = try b.write("oversized_ne.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.BadMetadata);
    }
    { // zero dimension
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 0);
        try b.str("t");
        try b.u32v(1);
        try b.u64v(0);
        try b.u32v(@intFromEnum(contracts.Dtype.f32));
        try b.u64v(0);
        const path = try b.write("zero_ne.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.BadMetadata);
    }
    { // n_dims out of range
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 0);
        try b.str("t");
        try b.u32v(5);
        for (0..5) |_| try b.u64v(1);
        try b.u32v(@intFromEnum(contracts.Dtype.f32));
        try b.u64v(0);
        const path = try b.write("bad_ndims.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.BadMetadata);
    }
    { // unknown ggml dtype id
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 0);
        try b.str("t");
        try b.u32v(1);
        try b.u64v(32);
        try b.u32v(999);
        try b.u64v(0);
        const path = try b.write("bad_dtype.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.BadMetadata);
    }
    { // quant row not block-aligned: 33 elems of q8_0
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 0);
        try b.str("t");
        try b.u32v(1);
        try b.u64v(33);
        try b.u32v(@intFromEnum(contracts.Dtype.q8_0));
        try b.u64v(0);
        const path = try b.write("unaligned_row.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.BadMetadata);
    }
    { // tensor offset not a multiple of general.alignment
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 0);
        try b.str("t");
        try b.u32v(1);
        try b.u64v(4);
        try b.u32v(@intFromEnum(contracts.Dtype.f32));
        try b.u64v(7);
        const path = try b.write("bad_offset.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.BadMetadata);
    }
    { // duplicate tensor name
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(2, 0);
        for (0..2) |_| {
            try b.str("t");
            try b.u32v(1);
            try b.u64v(4);
            try b.u32v(@intFromEnum(contracts.Dtype.f32));
            try b.u64v(0);
        }
        try b.pad(DEFAULT_ALIGNMENT);
        const vals = [4]f32{ 1, 2, 3, 4 };
        try b.bytes(std.mem.sliceAsBytes(&vals));
        const path = try b.write("dup_name.gguf");
        defer alloc.free(@constCast(path));
        try expectOpenError(path, GgufError.BadMetadata);
    }
}

test "hand-built gguf: unknown metadata value type is an error" {
    // Type ids above 12 have unknown wire size — skipping is impossible.
    const alloc = testing.allocator;
    var b = Builder{ .alloc = alloc };
    defer b.deinit();
    try b.header(0, 1);
    try b.str("k.mystery");
    try b.u32v(99);
    try b.u64v(0);
    const path = try b.write("bad_vt.gguf");
    defer alloc.free(@constCast(path));
    try expectOpenError(path, GgufError.BadMetadata);
}

test "hand-built gguf: k-quant and i-quant tensors parse structurally" {
    const alloc = testing.allocator;
    const cases = [_]contracts.Dtype{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .iq2_xxs, .iq2_xs, .iq2_s, .iq4_xs, .f16, .bf16 };
    var b = Builder{ .alloc = alloc };
    defer b.deinit();
    try b.header(cases.len, 0);
    var offset: u64 = 0;
    var total: u64 = 0;
    inline for (cases, 0..) |dtype, i| {
        const ne0: u64 = 2 * dtype.blockElems(); // two blocks per row
        const size = dtype.rowBytes(ne0) * 3;
        try b.str(std.fmt.comptimePrint("t{d}", .{i}));
        try b.u32v(2);
        try b.u64v(ne0);
        try b.u64v(3);
        try b.u32v(@intFromEnum(dtype));
        try b.u64v(offset);
        offset = std.mem.alignForward(u64, offset + size, DEFAULT_ALIGNMENT);
        total = offset;
    }
    try b.pad(DEFAULT_ALIGNMENT);
    for (0..@intCast(total)) |i| try b.list.append(alloc, @truncate(i));
    const path = try b.write("quants.gguf");
    defer alloc.free(@constCast(path));

    var m = try Model.open(alloc, path);
    defer m.deinit();
    try testing.expectEqual(cases.len, m.tensorCount());
    inline for (cases, 0..) |dtype, i| {
        const t = m.tensorAt(i);
        try testing.expectEqual(dtype, t.desc.dtype);
        try testing.expectEqual(dtype.rowBytes(2 * dtype.blockElems()) * 3, t.data.len);
    }
}

test "config: missing keys and wrong architecture" {
    const alloc = testing.allocator;
    { // no architecture at all
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 0);
        try b.oneTensorTail();
        const path = try b.write("cfg_nokeys.gguf");
        defer alloc.free(@constCast(path));
        var m = try Model.open(alloc, path);
        defer m.deinit();
        try testing.expectError(GgufError.MissingKey, m.config());
    }
    { // wrong architecture string
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 1);
        try b.str("general.architecture");
        try b.u32v(8);
        try b.str("llama");
        try b.oneTensorTail();
        const path = try b.write("cfg_badarch.gguf");
        defer alloc.free(@constCast(path));
        var m = try Model.open(alloc, path);
        defer m.deinit();
        try testing.expectError(GgufError.BadMetadata, m.config());
    }
    { // right arch, hyperparameters absent
        var b = Builder{ .alloc = alloc };
        defer b.deinit();
        try b.header(1, 1);
        try b.str("general.architecture");
        try b.u32v(8);
        try b.str("qwen3moe");
        try b.oneTensorTail();
        const path = try b.write("cfg_missing.gguf");
        defer alloc.free(@constCast(path));
        var m = try Model.open(alloc, path);
        defer m.deinit();
        try testing.expectError(GgufError.MissingKey, m.config());
    }
}

test "open: missing file is OpenFailed" {
    try expectOpenError(".zig-cache/tmp/ds5-gguf-test/definitely-absent.gguf", GgufError.OpenFailed);
}
