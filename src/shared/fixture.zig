//! DS5T fixture tensor I/O + tolerance comparison (contracts.zig §6, ADR-005).
//!
//! Every kernel workstream loads golden tensors through this module so the
//! pass/fail rule is implemented exactly once. Manifest JSON is parsed with
//! std.json into a dynamic Value; access the frozen schema fields directly.

const std = @import("std");
const contracts = @import("contracts.zig");
const sys = @import("sys.zig");

pub const Error = error{
    BadMagic,
    BadVersion,
    BadDtype,
    SizeMismatch,
} || std.mem.Allocator.Error || anyerror;

/// A tensor loaded from a .ds5t file. `raw` owns the whole file; `data`
/// slices into it at the frozen 64-byte offset.
pub const OwnedTensor = struct {
    desc: contracts.TensorDesc,
    raw: [:0]u8,
    data: []u8,

    pub fn free(self: *OwnedTensor, alloc: std.mem.Allocator) void {
        alloc.free(self.raw);
        self.* = undefined;
    }

    /// Checked view of the data as f32 elements.
    pub fn asF32(self: *const OwnedTensor) []const f32 {
        std.debug.assert(self.desc.dtype == .f32);
        return @alignCast(std.mem.bytesAsSlice(f32, self.data));
    }

    pub fn asI32(self: *const OwnedTensor) []const i32 {
        std.debug.assert(self.desc.dtype == .i32);
        return @alignCast(std.mem.bytesAsSlice(i32, self.data));
    }
};

pub fn loadTensor(alloc: std.mem.Allocator, path: []const u8) !OwnedTensor {
    const raw = try sys.readFileAllocZ(alloc, path);
    errdefer alloc.free(raw);
    if (raw.len < contracts.FIXTURE_DATA_OFFSET) return Error.SizeMismatch;
    var hdr: contracts.FixtureHeader = undefined;
    @memcpy(std.mem.asBytes(&hdr), raw[0..@sizeOf(contracts.FixtureHeader)]);
    if (hdr.magic != contracts.FIXTURE_MAGIC) return Error.BadMagic;
    if (hdr.version != contracts.FIXTURE_VERSION) return Error.BadVersion;
    const dtype = std.enums.fromInt(contracts.Dtype, hdr.dtype) orelse return Error.BadDtype;
    const desc = contracts.TensorDesc{
        .dtype = dtype,
        .n_dims = hdr.n_dims,
        .ne = hdr.ne,
    };
    if (desc.byteSize() != hdr.data_bytes) return Error.SizeMismatch;
    if (raw.len < contracts.FIXTURE_DATA_OFFSET + hdr.data_bytes) return Error.SizeMismatch;
    return .{
        .desc = desc,
        .raw = raw,
        .data = raw[contracts.FIXTURE_DATA_OFFSET..][0..@intCast(hdr.data_bytes)],
    };
}

pub fn writeTensor(alloc: std.mem.Allocator, path: []const u8, desc: contracts.TensorDesc, data: []const u8) !void {
    std.debug.assert(data.len == desc.byteSize());
    var hdr = contracts.FixtureHeader.init(desc);
    var buf = try alloc.alloc(u8, contracts.FIXTURE_DATA_OFFSET + data.len);
    defer alloc.free(buf);
    @memcpy(buf[0..@sizeOf(contracts.FixtureHeader)], std.mem.asBytes(&hdr));
    @memcpy(buf[contracts.FIXTURE_DATA_OFFSET..], data);
    try sys.writeFileTrunc(alloc, path, buf);
}

/// Result of an elementwise tolerance comparison.
pub const CompareResult = struct {
    pass: bool,
    n: usize,
    n_bad: usize,
    first_bad: usize, // index of first failing element (undefined if pass)
    max_abs_diff: f32,
    max_rel_diff: f32,
};

/// Frozen pass rule (ADR-005): |actual - oracle| <= atol + rtol*|oracle|,
/// every element. NaN in either input fails that element.
pub fn compare(oracle: []const f32, actual: []const f32, atol: f32, rtol: f32) CompareResult {
    std.debug.assert(oracle.len == actual.len);
    var r = CompareResult{
        .pass = true,
        .n = oracle.len,
        .n_bad = 0,
        .first_bad = 0,
        .max_abs_diff = 0,
        .max_rel_diff = 0,
    };
    for (oracle, actual, 0..) |o, a, i| {
        const diff = @abs(a - o);
        const rel = if (o != 0) diff / @abs(o) else 0;
        r.max_abs_diff = @max(r.max_abs_diff, diff);
        r.max_rel_diff = @max(r.max_rel_diff, rel);
        const bad = std.math.isNan(diff) or diff > atol + rtol * @abs(o);
        if (bad) {
            if (r.pass) r.first_bad = i;
            r.pass = false;
            r.n_bad += 1;
        }
    }
    return r;
}

/// Test helper: fail with a diagnostic if tensors diverge beyond tolerance.
pub fn expectClose(oracle: []const f32, actual: []const f32, atol: f32, rtol: f32) !void {
    const r = compare(oracle, actual, atol, rtol);
    if (!r.pass) {
        std.debug.print(
            "fixture mismatch: {d}/{d} elements out of tolerance (atol={e} rtol={e}); " ++
                "first bad idx {d}: oracle={e} actual={e}; max_abs={e} max_rel={e}\n",
            .{
                r.n_bad,               r.n,
                atol,                  rtol,
                r.first_bad,           oracle[r.first_bad],
                actual[r.first_bad],   r.max_abs_diff,
                r.max_rel_diff,
            },
        );
        return error.FixtureMismatch;
    }
}

/// Load a fixture directory's manifest.json (frozen schema in ADR-005).
/// Caller keeps the returned Parsed alive while using the Value.
pub fn loadManifest(alloc: std.mem.Allocator, dir: []const u8) !std.json.Parsed(std.json.Value) {
    const path = try std.fmt.allocPrint(alloc, "{s}/manifest.json", .{dir});
    defer alloc.free(path);
    const src = try sys.readFileAllocZ(alloc, path);
    defer alloc.free(src);
    return std.json.parseFromSlice(std.json.Value, alloc, src, .{});
}

test "tensor write/load round-trip with compare" {
    const alloc = std.testing.allocator;
    try sys.mkdirPath(alloc, ".zig-cache/tmp/ds5-fixture-test");
    const path = ".zig-cache/tmp/ds5-fixture-test/t.ds5t";

    const desc = contracts.TensorDesc.init(.f32, &.{ 4, 2 });
    const vals = [_]f32{ 1.0, -2.5, 3.25, 0.0, 7.5, -0.125, 2.0, 9.0 };
    try writeTensor(alloc, path, desc, std.mem.sliceAsBytes(&vals));

    var t = try loadTensor(alloc, path);
    defer t.free(alloc);
    try std.testing.expectEqual(contracts.Dtype.f32, t.desc.dtype);
    try std.testing.expectEqual(@as(u64, 4), t.desc.ne[0]);
    try std.testing.expectEqual(@as(u64, 2), t.desc.ne[1]);
    try expectClose(&vals, t.asF32(), 0, 0);

    var perturbed = vals;
    perturbed[3] += 0.5;
    const r = compare(&vals, &perturbed, 1e-6, 1e-6);
    try std.testing.expect(!r.pass);
    try std.testing.expectEqual(@as(usize, 3), r.first_bad);
    try std.testing.expectEqual(@as(usize, 1), r.n_bad);
}

test "committed synthetic fixture set loads" {
    // Cross-language format check: tests/fixtures/synthetic is generated by
    // tools/make_fixtures.py and committed; this proves the Zig side reads it.
    const alloc = std.testing.allocator;
    var parsed = loadManifest(alloc, "tests/fixtures/synthetic") catch |err| switch (err) {
        error.OpenFailed => return error.SkipZigTest, // fixture set not generated
        else => return err,
    };
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 1), root.get("ds5_fixture_version").?.integer);
    const cfg = root.get("model").?.object.get("config").?.object;
    // Must agree with contracts.SYNTH_TINY (ADR-005 §6).
    try std.testing.expectEqual(@as(i64, contracts.SYNTH_TINY.n_layers), cfg.get("n_layers").?.integer);
    try std.testing.expectEqual(@as(i64, contracts.SYNTH_TINY.n_experts), cfg.get("n_experts").?.integer);
    try std.testing.expectEqual(@as(i64, contracts.SYNTH_TINY.top_k), cfg.get("top_k").?.integer);

    const cases = root.get("cases").?.array.items;
    try std.testing.expect(cases.len >= 20);
    // Load every tensor of the first rmsnorm case and sanity-check shapes.
    const case0 = cases[0].object;
    try std.testing.expectEqualStrings("rmsnorm", case0.get("op").?.string);
    var it = case0.get("tensors").?.object.iterator();
    while (it.next()) |entry| {
        const path = try std.fmt.allocPrint(alloc, "tests/fixtures/synthetic/{s}", .{entry.value_ptr.string});
        defer alloc.free(path);
        var t = try loadTensor(alloc, path);
        defer t.free(alloc);
        try std.testing.expect(t.desc.elems() > 0);
        try std.testing.expectEqual(@as(u64, contracts.SYNTH_TINY.hidden_dim), t.desc.ne[0]);
    }
    // Prompt logits: ne[0] must be vocab_size.
    const p0 = root.get("prompts").?.array.items[0].object;
    const lp = try std.fmt.allocPrint(alloc, "tests/fixtures/synthetic/{s}", .{p0.get("logits").?.string});
    defer alloc.free(lp);
    var logits = try loadTensor(alloc, lp);
    defer logits.free(alloc);
    try std.testing.expectEqual(@as(u64, contracts.SYNTH_TINY.vocab_size), logits.desc.ne[0]);
    try std.testing.expectEqual(contracts.Dtype.f32, logits.desc.dtype);
}

test "manifest json loads" {
    const alloc = std.testing.allocator;
    try sys.mkdirPath(alloc, ".zig-cache/tmp/ds5-fixture-test2");
    try sys.writeFileTrunc(alloc, ".zig-cache/tmp/ds5-fixture-test2/manifest.json",
        \\{"ds5_fixture_version": 1, "cases": [{"op": "rmsnorm", "name": "l0",
        \\ "tolerance": {"atol": 1e-5, "rtol": 1e-4}}]}
    );
    var parsed = try loadManifest(alloc, ".zig-cache/tmp/ds5-fixture-test2");
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases").?.array;
    try std.testing.expectEqual(@as(usize, 1), cases.items.len);
    try std.testing.expectEqualStrings("rmsnorm", cases.items[0].object.get("op").?.string);
}
