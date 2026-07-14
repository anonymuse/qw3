//! T05 — M2b end-to-end GPU forward-pass tests, mirroring test_forward.zig's
//! CPU e2e/trace coverage but instantiating Engine(metal.Ctx, gpu_kernels).
//!
//! Skips (not fails) when no Metal device is present or fixtures aren't
//! generated, exactly like the CPU tests skip when fixtures are missing.

const std = @import("std");
const contracts = @import("shared/contracts.zig");
const fixture = @import("shared/fixture.zig");
const gguf = @import("gguf/gguf.zig");
const forward = @import("engine/forward.zig");
const metal = @import("metal/metal.zig");
const gpu_kernels = @import("kernels/gpu/kernels.zig");
const cpu_ctx_mod = @import("kernels/cpu/ctx.zig");
const kernels_a = @import("kernels/cpu/kernels_a.zig");
const kernels_b = @import("kernels/cpu/kernels_b.zig");
const kernels_c = @import("kernels/cpu/kernels_c.zig");

// Local copy of test_forward.zig's merged CPU provider namespace, so this
// file's GPU-vs-CPU diff test doesn't need to `@import("test_forward.zig")`
// wholesale (that would pull its entire test tree — and transitively
// kernels_a/b/c.zig's own fixture tests — into `zig build test-gpu`,
// duplicating `zig build test`'s coverage under a different step name).
const cpu_kernels = struct {
    pub const rmsNorm = kernels_a.rmsNorm;
    pub const rope = kernels_a.rope;
    pub const matmul = kernels_a.matmul;
    pub const kvAppend = kernels_a.kvAppend;
    pub const add = kernels_a.add;
    pub const gqaAttention = kernels_b.gqaAttention;
    pub const routerTopK = kernels_c.routerTopK;
    pub const expertMlpSwiglu = kernels_c.expertMlpSwiglu;
};
const CpuEngine = forward.Engine(cpu_ctx_mod.CpuCtx, cpu_kernels);

const FIXTURE_DIR = "tests/fixtures/synthetic";
const MODEL_PATH = FIXTURE_DIR ++ "/model.gguf";

pub const GpuEngine = forward.Engine(metal.Ctx, gpu_kernels);

test "GPU provider satisfies the frozen kernel api" {
    comptime contracts.assertKernelApi(gpu_kernels, metal.Ctx);
}

fn jsonF32(v: std.json.Value) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

const GpuRig = struct {
    model: gguf.Model,
    ctx: *metal.Ctx,
    weights: forward.Weights,
    engine: GpuEngine,

    fn init(alloc: std.mem.Allocator) !GpuRig {
        var model = try gguf.Model.open(alloc, MODEL_PATH);
        errdefer model.deinit();
        const ctx = metal.Ctx.init(alloc) catch |err| switch (err) {
            contracts.KernelError.DeviceFailure => return error.SkipZigTest,
            else => return err,
        };
        errdefer ctx.deinit();
        try gpu_kernels.loadShaders(ctx);
        var weights = try forward.Weights.fromGguf(alloc, ctx, &model);
        errdefer weights.deinit();
        return .{
            .model = model,
            .ctx = ctx,
            .weights = weights,
            .engine = undefined,
        };
    }

    fn deinit(self: *GpuRig) void {
        self.engine.deinit();
        self.weights.deinit();
        self.ctx.deinit();
        self.model.deinit();
    }
};

fn makeGpuRig(alloc: std.mem.Allocator, max_batch: u32) !*GpuRig {
    const rig = try alloc.create(GpuRig);
    errdefer alloc.destroy(rig);
    rig.* = try GpuRig.init(alloc);
    rig.engine = try GpuEngine.init(alloc, rig.ctx, &rig.weights, max_batch);
    return rig;
}

test "GPU e2e: 5 fixture prompts — logits in tolerance, greedy argmax exact" {
    const alloc = std.testing.allocator;
    var parsed = fixture.loadManifest(alloc, FIXTURE_DIR) catch |err| switch (err) {
        error.OpenFailed => return error.SkipZigTest,
        else => return err,
    };
    defer parsed.deinit();

    const rig = makeGpuRig(alloc, 64) catch |err| switch (err) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return err,
    };
    defer {
        rig.deinit();
        alloc.destroy(rig);
    }
    const vocab = rig.weights.config.vocab_size;

    gpu_kernels.beginTiming(alloc);
    defer gpu_kernels.endTiming();

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

        rig.engine.reset();
        try rig.engine.forward(prompt, all_logits[0 .. prompt.len * vocab]);

        var greedy_ok = true;
        var next: u32 = forward.argmax(all_logits[(prompt.len - 1) * vocab ..][0..vocab]);
        for (greedy_json, 0..) |want_v, gi| {
            const want: u32 = @intCast(want_v.integer);
            if (next != want) {
                std.debug.print("GPU prompt {s}: greedy step {d} got {d} want {d}\n", .{ name, gi, next, want });
                greedy_ok = false;
            }
            const row = all_logits[(prompt.len + gi) * vocab ..][0..vocab];
            try rig.engine.forward(&.{want}, row);
            next = forward.argmax(row);
        }
        try std.testing.expect(greedy_ok);

        const lp = try std.fmt.allocPrint(alloc, FIXTURE_DIR ++ "/{s}", .{p.get("logits").?.string});
        defer alloc.free(lp);
        var oracle = try fixture.loadTensor(alloc, lp);
        defer oracle.free(alloc);

        const r = fixture.compare(oracle.asF32(), all_logits, atol, rtol);
        std.debug.print("GPU prompt {s}: {d} tokens, logits max_abs_diff {e} max_rel_diff {e}\n", .{
            name, n_seq, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), all_logits, atol, rtol);
        n_prompts += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), n_prompts);

    std.debug.print("GPU per-layer-boundary elapsed ns (router-sync flushes): ", .{});
    for (gpu_kernels.layerNs()) |ns| std.debug.print("{d} ", .{ns});
    std.debug.print("\n", .{});
}

// ---------------------------------------------------------------------------
// Trace mode: GPU residual stream vs the l{i}_block.output fixtures (same
// oracle the CPU trace test uses), AND a direct GPU-vs-CPU diff per layer.
// ---------------------------------------------------------------------------

const GpuLayerTraceCollector = struct {
    alloc: std.mem.Allocator,
    ctx: *metal.Ctx,
    block_out: std.AutoHashMap(i32, []f32),

    fn init(alloc: std.mem.Allocator, ctx: *metal.Ctx) GpuLayerTraceCollector {
        return .{ .alloc = alloc, .ctx = ctx, .block_out = std.AutoHashMap(i32, []f32).init(alloc) };
    }

    fn deinit(self: *GpuLayerTraceCollector) void {
        var it = self.block_out.valueIterator();
        while (it.next()) |v| self.alloc.free(v.*);
        self.block_out.deinit();
    }

    fn hook(user: ?*anyopaque, layer: i32, op: []const u8, buf: contracts.Buf, n_elems: usize) void {
        const self: *GpuLayerTraceCollector = @ptrCast(@alignCast(user.?));
        if (!std.mem.eql(u8, op, "block")) return;
        const copy = self.alloc.alloc(f32, n_elems) catch @panic("oom in trace hook");
        // download() auto-flushes the pending batch (metal.zig T05 fix), so
        // this always reads the freshly-computed value for this op, at the
        // cost of one submit per traced op (acceptable: trace mode trades
        // batching throughput for per-op observability).
        self.ctx.download(buf, 0, std.mem.sliceAsBytes(copy)) catch @panic("gpu download failed in trace hook");
        self.block_out.put(layer, copy) catch @panic("oom in trace hook");
    }
};

fn firstDivergenceVsFixture(
    alloc: std.mem.Allocator,
    manifest: std.json.Value,
    collector: *GpuLayerTraceCollector,
) !?struct { layer: i32, result: fixture.CompareResult } {
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
        std.debug.print("GPU trace layer {d} block vs fixture: max_abs_diff {e} max_rel_diff {e} ({s})\n", .{
            layer, r.max_abs_diff, r.max_rel_diff, if (r.pass) "ok" else "DIVERGED",
        });
        if (!r.pass) return .{ .layer = layer, .result = r };
    }
    return null;
}

test "GPU trace hook: p0 prefill residual stream matches per-layer fixtures" {
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

    const rig = makeGpuRig(alloc, 64) catch |err| switch (err) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return err,
    };
    defer {
        rig.deinit();
        alloc.destroy(rig);
    }

    var collector = GpuLayerTraceCollector.init(alloc, rig.ctx);
    defer collector.deinit();
    rig.engine.trace_fn = GpuLayerTraceCollector.hook;
    rig.engine.trace_user = &collector;
    // Non-null logits_out throughout this file's tests: metal.Ctx.deinit()
    // asserts no batch is left open, and download() only flushes on an
    // actual host read — always give it one.
    const vocab = rig.weights.config.vocab_size;
    const scratch = try alloc.alloc(f32, prompt.len * vocab);
    defer alloc.free(scratch);
    try rig.engine.forward(prompt, scratch);

    try std.testing.expectEqual(@as(u32, 4), collector.block_out.count());

    const div = try firstDivergenceVsFixture(alloc, parsed.value, &collector);
    if (div) |d| {
        std.debug.print("GPU FIRST DIVERGENT layer: {d}: max_abs {e}\n", .{ d.layer, d.result.max_abs_diff });
        return error.TraceDivergence;
    }
}

test "GPU vs CPU: direct per-layer residual-stream diff on p0 prefill" {
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

    // CPU rig (test_forward.zig's own helpers, re-run here for a live diff
    // rather than trusting the fixture alone).
    var cpu_model = try gguf.Model.open(alloc, MODEL_PATH);
    defer cpu_model.deinit();
    const cpu_ctx = try cpu_ctx_mod.CpuCtx.init(alloc);
    defer cpu_ctx.deinit();
    var cpu_weights = try forward.Weights.fromGguf(alloc, cpu_ctx, &cpu_model);
    defer cpu_weights.deinit();
    var cpu_engine = try CpuEngine.init(alloc, cpu_ctx, &cpu_weights, 64);
    defer cpu_engine.deinit();

    const CpuCollector = struct {
        alloc: std.mem.Allocator,
        block_out: std.AutoHashMap(i32, []f32),
        fn hook(user: ?*anyopaque, layer: i32, op: []const u8, buf: contracts.Buf, n_elems: usize) void {
            const self: *@This() = @ptrCast(@alignCast(user.?));
            if (!std.mem.eql(u8, op, "block")) return;
            const copy = self.alloc.alloc(f32, n_elems) catch @panic("oom");
            @memcpy(copy, cpu_ctx_mod.asConstF32(buf)[0..n_elems]);
            self.block_out.put(layer, copy) catch @panic("oom");
        }
    };
    var cpu_collector = CpuCollector{ .alloc = alloc, .block_out = std.AutoHashMap(i32, []f32).init(alloc) };
    defer {
        var it = cpu_collector.block_out.valueIterator();
        while (it.next()) |v| alloc.free(v.*);
        cpu_collector.block_out.deinit();
    }
    cpu_engine.trace_fn = CpuCollector.hook;
    cpu_engine.trace_user = &cpu_collector;
    try cpu_engine.forward(prompt, null);

    // GPU rig.
    const rig = makeGpuRig(alloc, 64) catch |err| switch (err) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return err,
    };
    defer {
        rig.deinit();
        alloc.destroy(rig);
    }
    var gpu_collector = GpuLayerTraceCollector.init(alloc, rig.ctx);
    defer gpu_collector.deinit();
    rig.engine.trace_fn = GpuLayerTraceCollector.hook;
    rig.engine.trace_user = &gpu_collector;
    const vocab = rig.weights.config.vocab_size;
    const scratch = try alloc.alloc(f32, prompt.len * vocab);
    defer alloc.free(scratch);
    try rig.engine.forward(prompt, scratch);

    // ADR-005 §4 "layer" tolerance applies to CPU-vs-Metal identical
    // dispatches too (PORTING docs' "second gate").
    const tol_atol: f32 = 2e-3;
    const tol_rtol: f32 = 1e-2;
    var worst_abs: f32 = 0;
    var checked: usize = 0;
    var it = cpu_collector.block_out.iterator();
    while (it.next()) |entry| {
        const layer = entry.key_ptr.*;
        const cpu_out = entry.value_ptr.*;
        const gpu_out = gpu_collector.block_out.get(layer) orelse continue;
        const r = fixture.compare(cpu_out, gpu_out, tol_atol, tol_rtol);
        std.debug.print("GPU-vs-CPU trace layer {d}: max_abs_diff {e} max_rel_diff {e} ({s})\n", .{
            layer, r.max_abs_diff, r.max_rel_diff, if (r.pass) "ok" else "DIVERGED",
        });
        worst_abs = @max(worst_abs, r.max_abs_diff);
        checked += 1;
        try std.testing.expect(r.pass);
    }
    try std.testing.expectEqual(@as(usize, 4), checked);
    std.debug.print("GPU-vs-CPU trace: worst max_abs_diff across all layers = {e}\n", .{worst_abs});
}
