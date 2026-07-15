//! T04 — M2a end-to-end forward-pass tests against the synthetic oracle.
//!
//! For each of the 5 manifest prompts: prefill the prompt, greedy-decode 8
//! tokens, and compare (a) logits at EVERY sequence position to the fixture
//! within manifest tolerance and (b) the greedy argmax tokens exactly.
//!
//! Plus the trace-mode hook (the M2c debugging tool): re-run p0's prefill
//! with the per-op trace enabled and compare the residual stream after every
//! layer to the committed l{i}_block.output tensors, reporting the FIRST
//! layer/op whose divergence exceeds tolerance.

const std = @import("std");
const contracts = @import("shared/contracts.zig");
const fixture = @import("shared/fixture.zig");
const gguf = @import("gguf/gguf.zig");
const forward = @import("engine/forward.zig");
const cpu = @import("kernels/cpu/ctx.zig");
const kernels_a = @import("kernels/cpu/kernels_a.zig");
const kernels_b = @import("kernels/cpu/kernels_b.zig");
const kernels_c = @import("kernels/cpu/kernels_c.zig");

const FIXTURE_DIR = "tests/fixtures/synthetic";
const MODEL_PATH = FIXTURE_DIR ++ "/model.gguf";

/// Merged CPU kernel provider namespace (sets A + B + C).
pub const cpu_kernels = struct {
    pub const rmsNorm = kernels_a.rmsNorm;
    pub const rope = kernels_a.rope;
    pub const matmul = kernels_a.matmul;
    pub const kvAppend = kernels_a.kvAppend;
    pub const add = kernels_a.add;
    pub const gqaAttention = kernels_b.gqaAttention;
    pub const routerTopK = kernels_c.routerTopK;
    pub const expertMlpSwiglu = kernels_c.expertMlpSwiglu;
};

pub const CpuEngine = forward.Engine(cpu.CpuCtx, cpu_kernels);

test "merged cpu provider satisfies the frozen kernel api" {
    comptime contracts.assertKernelApi(cpu_kernels, cpu.CpuCtx);
}

fn jsonF32(v: std.json.Value) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

const Rig = struct {
    model: gguf.Model,
    ctx: *cpu.CpuCtx,
    weights: forward.Weights,
    engine: CpuEngine,

    fn init(alloc: std.mem.Allocator) !Rig {
        var model = try gguf.Model.open(alloc, MODEL_PATH);
        errdefer model.deinit();
        const ctx = try cpu.CpuCtx.init(alloc);
        errdefer ctx.deinit();
        var weights = try forward.Weights.fromGguf(alloc, ctx, &model);
        errdefer weights.deinit();
        return .{
            .model = model,
            .ctx = ctx,
            .weights = weights,
            .engine = undefined, // fixed up below (needs a stable *Weights)
        };
    }

    fn deinit(self: *Rig) void {
        self.engine.deinit();
        self.weights.deinit();
        self.ctx.deinit();
        self.model.deinit();
    }
};

fn makeRig(alloc: std.mem.Allocator, max_batch: u32) !*Rig {
    const rig = try alloc.create(Rig);
    errdefer alloc.destroy(rig);
    rig.* = try Rig.init(alloc);
    rig.engine = try CpuEngine.init(alloc, rig.ctx, &rig.weights, max_batch);
    return rig;
}

test "e2e: 5 fixture prompts — logits in tolerance, greedy argmax exact" {
    const alloc = std.testing.allocator;
    var parsed = fixture.loadManifest(alloc, FIXTURE_DIR) catch |err| switch (err) {
        error.OpenFailed => return error.SkipZigTest,
        else => return err,
    };
    defer parsed.deinit();

    const rig = try makeRig(alloc, 64);
    defer {
        rig.deinit();
        alloc.destroy(rig);
    }
    const vocab = rig.weights.config.vocab_size;

    var n_prompts: usize = 0;
    for (parsed.value.object.get("prompts").?.array.items) |p_v| {
        const p = p_v.object;
        const name = p.get("name").?.string;
        const tol = p.get("tolerance").?.object;
        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);

        const tok_json = p.get("token_ids").?.array.items;
        const greedy_json = p.get("greedy_tokens").?.array.items;
        const prompt = try alloc.alloc(u32, tok_json.len);
        defer alloc.free(prompt);
        for (prompt, tok_json) |*t, v| t.* = @intCast(v.integer);

        const n_seq = tok_json.len + greedy_json.len;
        const all_logits = try alloc.alloc(f32, n_seq * vocab);
        defer alloc.free(all_logits);

        // Prefill the prompt: logits for positions 0..P-1.
        rig.engine.reset();
        try rig.engine.forward(prompt, all_logits[0 .. prompt.len * vocab]);

        // Greedy decode: argmax of the last row feeds the next step.
        var greedy_ok = true;
        var next: u32 = forward.argmax(all_logits[(prompt.len - 1) * vocab ..][0..vocab]);
        for (greedy_json, 0..) |want_v, gi| {
            const want: u32 = @intCast(want_v.integer);
            if (next != want) {
                std.debug.print("prompt {s}: greedy step {d} got {d} want {d}\n", .{ name, gi, next, want });
                greedy_ok = false;
            }
            // Feed the ORACLE token so one miss can't cascade (and the logits
            // comparison below stays aligned with the fixture sequence).
            const row = all_logits[(prompt.len + gi) * vocab ..][0..vocab];
            try rig.engine.forward(&.{want}, row);
            next = forward.argmax(row);
        }
        try std.testing.expect(greedy_ok);

        // Full-sequence logits against the oracle.
        const lp = try std.fmt.allocPrint(alloc, FIXTURE_DIR ++ "/{s}", .{p.get("logits").?.string});
        defer alloc.free(lp);
        var oracle = try fixture.loadTensor(alloc, lp);
        defer oracle.free(alloc);
        try std.testing.expectEqual(@as(u64, vocab), oracle.desc.ne[0]);
        try std.testing.expectEqual(@as(u64, n_seq), oracle.desc.ne[1]);

        const r = fixture.compare(oracle.asF32(), all_logits, atol, rtol);
        std.debug.print("prompt {s}: {d} tokens, logits max_abs_diff {e} max_rel_diff {e}\n", .{
            name, n_seq, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), all_logits, atol, rtol);
        n_prompts += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), n_prompts);
}

// ---------------------------------------------------------------------------
// Trace mode: per-layer residual-stream comparison (M2c debugging tool).
// ---------------------------------------------------------------------------

/// Collects the "block" (post-layer residual) trace outputs on the host.
const LayerTraceCollector = struct {
    alloc: std.mem.Allocator,
    /// block_out[layer] = residual stream after that layer, [n, hidden].
    block_out: std.AutoHashMap(i32, []f32),

    fn init(alloc: std.mem.Allocator) LayerTraceCollector {
        return .{ .alloc = alloc, .block_out = std.AutoHashMap(i32, []f32).init(alloc) };
    }

    fn deinit(self: *LayerTraceCollector) void {
        var it = self.block_out.valueIterator();
        while (it.next()) |v| self.alloc.free(v.*);
        self.block_out.deinit();
    }

    fn hook(user: ?*anyopaque, layer: i32, op: []const u8, buf: contracts.Buf, n_elems: usize) void {
        const self: *LayerTraceCollector = @ptrCast(@alignCast(user.?));
        if (!std.mem.eql(u8, op, "block")) return;
        const copy = self.alloc.alloc(f32, n_elems) catch @panic("oom in trace hook");
        @memcpy(copy, cpu.asConstF32(buf)[0..n_elems]);
        self.block_out.put(layer, copy) catch @panic("oom in trace hook");
    }
};

/// The first fixture-checkable point where the traced stream diverges.
const Divergence = struct {
    layer: i32,
    op: []const u8,
    result: fixture.CompareResult,
};

/// Compare collected per-layer residuals against l{i}_block.output fixtures.
/// Returns the FIRST divergent layer/op, or null if all in tolerance.
fn firstDivergence(
    alloc: std.mem.Allocator,
    manifest: std.json.Value,
    collector: *LayerTraceCollector,
) !?Divergence {
    for (manifest.object.get("cases").?.array.items) |case_v| {
        const case = case_v.object;
        if (!std.mem.eql(u8, case.get("op").?.string, "layer")) continue;
        const layer: i32 = @intCast(case.get("params").?.object.get("layer").?.integer);
        const tol = case.get("tolerance").?.object;

        const got = collector.block_out.get(layer) orelse continue;
        const path = try std.fmt.allocPrint(alloc, FIXTURE_DIR ++ "/{s}", .{
            case.get("tensors").?.object.get("output").?.string,
        });
        defer alloc.free(path);
        var oracle = try fixture.loadTensor(alloc, path);
        defer oracle.free(alloc);

        const r = fixture.compare(oracle.asF32(), got, jsonF32(tol.get("atol").?), jsonF32(tol.get("rtol").?));
        std.debug.print("trace layer {d} block: max_abs_diff {e} max_rel_diff {e} ({s})\n", .{
            layer, r.max_abs_diff, r.max_rel_diff, if (r.pass) "ok" else "DIVERGED",
        });
        if (!r.pass) return .{ .layer = layer, .op = "block", .result = r };
    }
    return null;
}

test "trace hook: p0 prefill residual stream matches per-layer fixtures" {
    const alloc = std.testing.allocator;
    var parsed = fixture.loadManifest(alloc, FIXTURE_DIR) catch |err| switch (err) {
        error.OpenFailed => return error.SkipZigTest,
        else => return err,
    };
    defer parsed.deinit();

    // The layer fixtures were captured during p0's 17-token prefill.
    const p0 = parsed.value.object.get("prompts").?.array.items[0].object;
    const tok_json = p0.get("token_ids").?.array.items;
    const prompt = try alloc.alloc(u32, tok_json.len);
    defer alloc.free(prompt);
    for (prompt, tok_json) |*t, v| t.* = @intCast(v.integer);

    const rig = try makeRig(alloc, 64);
    defer {
        rig.deinit();
        alloc.destroy(rig);
    }

    var collector = LayerTraceCollector.init(alloc);
    defer collector.deinit();
    rig.engine.trace_fn = LayerTraceCollector.hook;
    rig.engine.trace_user = &collector;
    try rig.engine.forward(prompt, null);

    // All 4 layers traced; fixtures exist for layers 0, 2, 3.
    try std.testing.expectEqual(@as(u32, 4), collector.block_out.count());

    const div = try firstDivergence(alloc, parsed.value, &collector);
    if (div) |d| {
        std.debug.print("FIRST DIVERGENT layer/op: layer {d} op {s}: {d}/{d} bad, first idx {d}, max_abs {e}\n", .{
            d.layer, d.op, d.result.n_bad, d.result.n, d.result.first_bad, d.result.max_abs_diff,
        });
        return error.TraceDivergence;
    }
}

test "trace hook: detects an injected divergence and names the first layer" {
    // Sanity for the debugging tool itself: corrupt layer 2's collected
    // output and confirm firstDivergence pins layer 2 (0 stays clean).
    const alloc = std.testing.allocator;
    var parsed = fixture.loadManifest(alloc, FIXTURE_DIR) catch |err| switch (err) {
        error.OpenFailed => return error.SkipZigTest,
        else => return err,
    };
    defer parsed.deinit();

    const p0 = parsed.value.object.get("prompts").?.array.items[0].object;
    const tok_json = p0.get("token_ids").?.array.items;
    const prompt = try alloc.alloc(u32, tok_json.len);
    defer alloc.free(prompt);
    for (prompt, tok_json) |*t, v| t.* = @intCast(v.integer);

    const rig = try makeRig(alloc, 64);
    defer {
        rig.deinit();
        alloc.destroy(rig);
    }
    var collector = LayerTraceCollector.init(alloc);
    defer collector.deinit();
    rig.engine.trace_fn = LayerTraceCollector.hook;
    rig.engine.trace_user = &collector;
    try rig.engine.forward(prompt, null);

    for (collector.block_out.get(2).?) |*v| v.* += 1.0; // corrupt layer 2
    const div = (try firstDivergence(alloc, parsed.value, &collector)) orelse
        return error.DivergenceNotDetected;
    try std.testing.expectEqual(@as(i32, 2), div.layer);
}

test "weights bind from the synthetic gguf" {
    const alloc = std.testing.allocator;
    var model = gguf.Model.open(alloc, MODEL_PATH) catch |err| switch (err) {
        contracts.GgufError.OpenFailed => return error.SkipZigTest,
        else => return err,
    };
    defer model.deinit();
    const ctx = try cpu.CpuCtx.init(alloc);
    defer ctx.deinit();
    var w = try forward.Weights.fromGguf(alloc, ctx, &model);
    defer w.deinit();

    try std.testing.expectEqual(@as(u32, 4), w.config.n_layers);
    try std.testing.expectEqual(@as(usize, 4), w.layers.len);
    try std.testing.expectEqual(contracts.Dtype.q8_0, w.token_embd_dtype);
    try std.testing.expectEqual(contracts.Dtype.q8_0, w.layers[0].wq.dtype);
    try std.testing.expectEqual(contracts.Dtype.f32, w.layers[0].router.dtype);
    // Norm buffers wrap the mmap zero-copy: hidden_dim f32s.
    try std.testing.expectEqual(@as(u64, 256 * 4), w.layers[0].attn_norm.len);
}

test {
    _ = @import("engine/forward.zig");
}
