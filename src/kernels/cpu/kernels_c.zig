//! Kernel set C (CPU reference): router top-k + fused SwiGLU expert MLP.
//!
//! routerTopK implements the EXACT Qwen3 router semantics frozen in ADR-005 §2
//! (ADR-001 rule 1 — never altered):
//!
//!     logits  = x · Wᵀ                 (f32)
//!     p       = softmax(logits)        over ALL n_experts, f32
//!     top_k   = k largest p, descending; ties → LOWER expert id
//!     weights = norm_topk_prob ? top_p / sum(top_p) : top_p
//!
//! The router is deliberately CPU-only: its outputs are host slices by
//! contract (the scheduler consumes them from CPU memory), the matmul is tiny
//! (n_experts × dim per token), and exactness matters more than speed.
//!
//! expertMlpSwiglu accumulates, per (token, expert, weight) dispatch pair:
//!
//!     out[token] += weight * down( silu(gate(x[token])) ⊙ up(x[token]) )
//!
//! with silu(v) = v / (1 + e^(-v)) in f32. `out` is pre-zeroed by the caller.
//! Expert banks are single 3-D GGUF tensors (gate/up ne = {dim, ffn, E},
//! down ne = {ffn, dim, E}); expert e starts at byte offset e·rowBytes(ne[0])·ne[1].
//!
//! Q8_0 dequant is exactly `f32(f16 scale) * i8 q` per ADR-005 §1; dots
//! accumulate dequantized values elementwise in f32, matching the oracle.

const std = @import("std");
const contracts = @import("../../shared/contracts.zig");
const cpu = @import("ctx.zig");
const fixture = @import("../../shared/fixture.zig");

const CpuCtx = cpu.CpuCtx;
const KernelError = contracts.KernelError;
const Dtype = contracts.Dtype;

// ---------------------------------------------------------------------------
// Row dot products against GGUF-layout weight rows (ne[0] contiguous).
// ---------------------------------------------------------------------------

/// dot(x, row) where `row` is one weight row of x.len elements in `dtype`.
/// Accumulation is f32, elementwise in storage order.
fn dotRow(x: []const f32, row: []const u8, dtype: Dtype) f32 {
    var acc: f32 = 0;
    switch (dtype) {
        .f32 => {
            const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, row));
            for (w, x) |wv, xv| acc += wv * xv;
        },
        .f16 => {
            for (x, 0..) |xv, i| {
                const bits = std.mem.readInt(u16, row[i * 2 ..][0..2], .little);
                acc += @as(f32, @floatCast(@as(f16, @bitCast(bits)))) * xv;
            }
        },
        .q8_0 => {
            // 34-byte blocks: f16 scale + 32 × i8. value = f32(f16 scale) * q.
            const n_blocks = x.len / 32;
            var b: usize = 0;
            while (b < n_blocks) : (b += 1) {
                const blk = row[b * 34 ..][0..34];
                const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk[0..2], .little))));
                const qs: []const i8 = @ptrCast(blk[2..34]);
                for (qs, x[b * 32 ..][0..32]) |q, xv| {
                    acc += (scale * @as(f32, @floatFromInt(q))) * xv;
                }
            }
        },
        else => unreachable, // callers gate dtypes before dispatching here
    }
    return acc;
}

fn siluF32(v: f32) f32 {
    return v / (1.0 + @exp(-v));
}

// ---------------------------------------------------------------------------
// routerTopK — exact semantics, host outputs. Never alter (ADR-001 rule 1).
// ---------------------------------------------------------------------------

pub fn routerTopK(ctx: *CpuCtx, args: contracts.RouterArgs) KernelError!void {
    switch (args.w_dtype) {
        .f32, .f16, .q8_0 => {},
        else => return KernelError.UnsupportedDtype,
    }
    const n_tokens: usize = args.n_tokens;
    const dim: usize = args.dim;
    const n_experts: usize = args.n_experts;
    const top_k: usize = args.top_k;
    if (top_k == 0 or top_k > n_experts) return KernelError.ShapeMismatch;
    if (dim == 0 or dim % args.w_dtype.blockElems() != 0) return KernelError.ShapeMismatch;
    if (args.out_ids.len != n_tokens * top_k) return KernelError.ShapeMismatch;
    if (args.out_weights.len != n_tokens * top_k) return KernelError.ShapeMismatch;
    if (args.x.len < n_tokens * dim * @sizeOf(f32)) return KernelError.ShapeMismatch;
    const row_bytes: usize = @intCast(args.w_dtype.rowBytes(dim));
    if (args.w.len < n_experts * row_bytes) return KernelError.ShapeMismatch;

    const probs = ctx.alloc.alloc(f32, n_experts) catch return KernelError.OutOfMemory;
    defer ctx.alloc.free(probs);
    const taken = ctx.alloc.alloc(bool, n_experts) catch return KernelError.OutOfMemory;
    defer ctx.alloc.free(taken);

    const x = cpu.asConstF32(args.x);
    const w = cpu.constBytes(args.w, 0, args.w.len);

    var t: usize = 0;
    while (t < n_tokens) : (t += 1) {
        const xr = x[t * dim ..][0..dim];

        // logits = x · Wᵀ, f32
        var max_logit: f32 = -std.math.inf(f32);
        for (0..n_experts) |e| {
            const logit = dotRow(xr, w[e * row_bytes ..][0..row_bytes], args.w_dtype);
            probs[e] = logit;
            max_logit = @max(max_logit, logit);
        }

        // softmax over ALL n_experts, f32
        var sum: f32 = 0;
        for (probs) |*p| {
            p.* = @exp(p.* - max_logit);
            sum += p.*;
        }
        const inv_sum = 1.0 / sum;
        for (probs) |*p| p.* *= inv_sum;

        // top-k by probability descending; ties → LOWER expert id. The
        // ascending-id scan with a strict `>` makes the lower id win exactly.
        @memset(taken, false);
        var picked_sum: f32 = 0;
        for (0..top_k) |j| {
            var best: usize = 0;
            var best_p: f32 = -1.0;
            for (0..n_experts) |e| {
                if (!taken[e] and probs[e] > best_p) {
                    best = e;
                    best_p = probs[e];
                }
            }
            taken[best] = true;
            args.out_ids[t * top_k + j] = @intCast(best);
            args.out_weights[t * top_k + j] = best_p;
            picked_sum += best_p;
        }

        // iff norm_topk_prob, divide the k weights by their sum
        if (args.norm_topk_prob) {
            const inv = 1.0 / picked_sum;
            for (args.out_weights[t * top_k ..][0..top_k]) |*ow| ow.* *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// expertMlpSwiglu — fused SwiGLU MoE MLP over a dispatch-pair list.
// ---------------------------------------------------------------------------

pub fn expertMlpSwiglu(ctx: *CpuCtx, args: contracts.ExpertMlpArgs) KernelError!void {
    switch (args.w_dtype) {
        .f32, .q8_0 => {},
        else => return KernelError.UnsupportedDtype,
    }
    const dim: usize = args.dim;
    const ffn_dim: usize = args.ffn_dim;
    const n_tokens: usize = args.n_tokens;
    const block = args.w_dtype.blockElems();
    if (dim == 0 or ffn_dim == 0) return KernelError.ShapeMismatch;
    if (dim % block != 0 or ffn_dim % block != 0) return KernelError.ShapeMismatch;
    if (args.x.len < n_tokens * dim * @sizeOf(f32)) return KernelError.ShapeMismatch;
    if (args.out.len < n_tokens * dim * @sizeOf(f32)) return KernelError.ShapeMismatch;

    // Bank strides (ADR-005 §1): expert e of ne={k,n,E} at e·rowBytes(k)·n.
    const row_h: usize = @intCast(args.w_dtype.rowBytes(dim)); // gate/up rows, k = dim
    const row_f: usize = @intCast(args.w_dtype.rowBytes(ffn_dim)); // down rows, k = ffn_dim
    const gate_stride = row_h * ffn_dim;
    const down_stride = row_f * dim;
    if (args.gate.len < args.n_experts * gate_stride) return KernelError.ShapeMismatch;
    if (args.up.len < args.n_experts * gate_stride) return KernelError.ShapeMismatch;
    if (args.down.len < args.n_experts * down_stride) return KernelError.ShapeMismatch;

    const h = ctx.alloc.alloc(f32, ffn_dim) catch return KernelError.OutOfMemory;
    defer ctx.alloc.free(h);

    const x = cpu.asConstF32(args.x);
    const out = cpu.asF32(args.out);
    const gate = cpu.constBytes(args.gate, 0, args.gate.len);
    const up = cpu.constBytes(args.up, 0, args.up.len);
    const down = cpu.constBytes(args.down, 0, args.down.len);

    for (args.pairs) |pair| {
        if (pair.token >= args.n_tokens or pair.expert >= args.n_experts)
            return KernelError.ShapeMismatch;
        const xr = x[pair.token * dim ..][0..dim];
        const g_e = gate[pair.expert * gate_stride ..][0..gate_stride];
        const u_e = up[pair.expert * gate_stride ..][0..gate_stride];
        const d_e = down[pair.expert * down_stride ..][0..down_stride];

        // h = silu(gate(x)) ⊙ up(x), f32
        for (0..ffn_dim) |j| {
            const gv = dotRow(xr, g_e[j * row_h ..][0..row_h], args.w_dtype);
            const uv = dotRow(xr, u_e[j * row_h ..][0..row_h], args.w_dtype);
            h[j] = siluF32(gv) * uv;
        }

        // out[token] += weight * down(h)
        const or_ = out[pair.token * dim ..][0..dim];
        for (0..dim) |i| {
            or_[i] += pair.weight * dotRow(h, d_e[i * row_f ..][0..row_f], args.w_dtype);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — frozen-signature check, fixture validation, semantics unit tests.
// ---------------------------------------------------------------------------

test "signatures match the frozen kernel api" {
    try std.testing.expect(@TypeOf(routerTopK) ==
        fn (*CpuCtx, contracts.RouterArgs) KernelError!void);
    try std.testing.expect(@TypeOf(expertMlpSwiglu) ==
        fn (*CpuCtx, contracts.ExpertMlpArgs) KernelError!void);
}

const FIXTURE_DIR = "tests/fixtures/synthetic";

fn loadCaseTensor(alloc: std.mem.Allocator, name: []const u8) !fixture.OwnedTensor {
    const path = try std.fmt.allocPrint(alloc, FIXTURE_DIR ++ "/{s}", .{name});
    defer alloc.free(path);
    return fixture.loadTensor(alloc, path);
}

fn jsonF32(v: std.json.Value) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

fn jsonU32(v: std.json.Value) u32 {
    return @intCast(v.integer);
}

fn dtypeFromName(name: []const u8) Dtype {
    if (std.mem.eql(u8, name, "f32")) return .f32;
    if (std.mem.eql(u8, name, "f16")) return .f16;
    if (std.mem.eql(u8, name, "q8_0")) return .q8_0;
    unreachable;
}

test "synthetic router fixtures: expert ids 100% exact, gate weights in tolerance" {
    const alloc = std.testing.allocator;
    var parsed = fixture.loadManifest(alloc, FIXTURE_DIR) catch |err| switch (err) {
        error.OpenFailed => return error.SkipZigTest, // fixture set not generated
        else => return err,
    };
    defer parsed.deinit();
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    var n_cases: usize = 0;
    for (parsed.value.object.get("cases").?.array.items) |case_v| {
        const case = case_v.object;
        if (!std.mem.eql(u8, case.get("op").?.string, "router")) continue;
        n_cases += 1;

        const params = case.get("params").?.object;
        const n_tokens = jsonU32(params.get("n_tokens").?);
        const dim = jsonU32(params.get("dim").?);
        const n_experts = jsonU32(params.get("n_experts").?);
        const top_k = jsonU32(params.get("top_k").?);
        const norm = params.get("norm_topk_prob").?.bool;
        const w_dtype = dtypeFromName(params.get("w_dtype").?.string);
        const tol = case.get("tolerance").?.object;

        const tensors = case.get("tensors").?.object;
        var input = try loadCaseTensor(alloc, tensors.get("input").?.string);
        defer input.free(alloc);
        var weight = try loadCaseTensor(alloc, tensors.get("weight").?.string);
        defer weight.free(alloc);
        var oracle_ids = try loadCaseTensor(alloc, tensors.get("expert_ids").?.string);
        defer oracle_ids.free(alloc);
        var oracle_w = try loadCaseTensor(alloc, tensors.get("gate_weights").?.string);
        defer oracle_w.free(alloc);

        const out_ids = try alloc.alloc(u16, n_tokens * top_k);
        defer alloc.free(out_ids);
        const out_weights = try alloc.alloc(f32, n_tokens * top_k);
        defer alloc.free(out_weights);

        try routerTopK(ctx, .{
            .x = try ctx.bufferFromBytes(input.data),
            .w = try ctx.bufferFromBytes(weight.data),
            .w_dtype = w_dtype,
            .n_tokens = n_tokens,
            .dim = dim,
            .n_experts = n_experts,
            .top_k = top_k,
            .norm_topk_prob = norm,
            .out_ids = out_ids,
            .out_weights = out_weights,
        });

        // Expert ids must match EXACTLY — compared as integers, 100%.
        const want_ids = oracle_ids.asI32();
        var id_mismatches: usize = 0;
        for (want_ids, out_ids) |want, got| {
            if (want != @as(i32, got)) id_mismatches += 1;
        }
        try std.testing.expectEqual(@as(usize, 0), id_mismatches);

        const r = fixture.compare(oracle_w.asF32(), out_weights, jsonF32(tol.get("atol").?), jsonF32(tol.get("rtol").?));
        std.debug.print("router case {s}: ids {d}/{d} exact, gate max_abs_diff {e}\n", .{
            case.get("name").?.string, want_ids.len - id_mismatches, want_ids.len, r.max_abs_diff,
        });
        try fixture.expectClose(oracle_w.asF32(), out_weights, jsonF32(tol.get("atol").?), jsonF32(tol.get("rtol").?));
    }
    try std.testing.expectEqual(@as(usize, 3), n_cases);
}

test "synthetic expert_mlp fixtures: q8_0 banks, output in tolerance" {
    const alloc = std.testing.allocator;
    var parsed = fixture.loadManifest(alloc, FIXTURE_DIR) catch |err| switch (err) {
        error.OpenFailed => return error.SkipZigTest, // fixture set not generated
        else => return err,
    };
    defer parsed.deinit();
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    var n_cases: usize = 0;
    for (parsed.value.object.get("cases").?.array.items) |case_v| {
        const case = case_v.object;
        if (!std.mem.eql(u8, case.get("op").?.string, "expert_mlp")) continue;
        n_cases += 1;

        const params = case.get("params").?.object;
        const n_tokens = jsonU32(params.get("n_tokens").?);
        const dim = jsonU32(params.get("dim").?);
        const ffn_dim = jsonU32(params.get("ffn_dim").?);
        const n_experts = jsonU32(params.get("n_experts").?);
        const top_k = jsonU32(params.get("top_k").?);
        const w_dtype = dtypeFromName(params.get("w_dtype").?.string);
        const tol = case.get("tolerance").?.object;

        const tensors = case.get("tensors").?.object;
        var input = try loadCaseTensor(alloc, tensors.get("input").?.string);
        defer input.free(alloc);
        var gate = try loadCaseTensor(alloc, tensors.get("gate").?.string);
        defer gate.free(alloc);
        var up = try loadCaseTensor(alloc, tensors.get("up").?.string);
        defer up.free(alloc);
        var down = try loadCaseTensor(alloc, tensors.get("down").?.string);
        defer down.free(alloc);
        var ids = try loadCaseTensor(alloc, tensors.get("expert_ids").?.string);
        defer ids.free(alloc);
        var gate_w = try loadCaseTensor(alloc, tensors.get("gate_weights").?.string);
        defer gate_w.free(alloc);
        var oracle_out = try loadCaseTensor(alloc, tensors.get("output").?.string);
        defer oracle_out.free(alloc);

        // Build the dispatch-pair list token-major: for each token, each slot.
        // expert_ids / gate_weights are ne = {top_k, n_tokens} (ne[0] fastest).
        const id_vals = ids.asI32();
        const w_vals = gate_w.asF32();
        const pairs = try alloc.alloc(contracts.PairDispatch, n_tokens * top_k);
        defer alloc.free(pairs);
        for (0..n_tokens) |t| {
            for (0..top_k) |j| {
                pairs[t * top_k + j] = .{
                    .token = @intCast(t),
                    .expert = @intCast(id_vals[t * top_k + j]),
                    .weight = w_vals[t * top_k + j],
                };
            }
        }

        const out_buf = try ctx.createBuffer(n_tokens * dim * @sizeOf(f32)); // pre-zeroed
        try expertMlpSwiglu(ctx, .{
            .x = try ctx.bufferFromBytes(input.data),
            .gate = try ctx.bufferFromBytes(gate.data),
            .up = try ctx.bufferFromBytes(up.data),
            .down = try ctx.bufferFromBytes(down.data),
            .w_dtype = w_dtype,
            .pairs = pairs,
            .out = out_buf,
            .n_tokens = n_tokens,
            .dim = dim,
            .ffn_dim = ffn_dim,
            .n_experts = n_experts,
        });

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle_out.asF32(), cpu.asConstF32(out_buf), atol, rtol);
        std.debug.print("expert_mlp case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle_out.asF32(), cpu.asConstF32(out_buf), atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 3), n_cases);
}

test "router norm_topk_prob=false leaves raw softmax probabilities" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    // dim=4, 4 experts, distinct logits: x = e0-basis, W rows give logits {2,0,1,-1}.
    const x = [_]f32{ 1, 0, 0, 0 };
    const w = [_]f32{
        2, 0, 0, 0,
        0, 9, 9, 9,
        1, 0, 0, 0,
        -1, 0, 0, 0,
    };
    var out_ids: [2]u16 = undefined;
    var out_weights: [2]f32 = undefined;
    try routerTopK(ctx, .{
        .x = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&x)),
        .w = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&w)),
        .w_dtype = .f32,
        .n_tokens = 1,
        .dim = 4,
        .n_experts = 4,
        .top_k = 2,
        .norm_topk_prob = false,
        .out_ids = &out_ids,
        .out_weights = &out_weights,
    });
    try std.testing.expectEqual(@as(u16, 0), out_ids[0]); // logit 2
    try std.testing.expectEqual(@as(u16, 2), out_ids[1]); // logit 1

    // Raw softmax over ALL experts (logits 2, 0, 1, -1), NOT renormalized.
    var denom: f32 = 0;
    for ([_]f32{ 2, 0, 1, -1 }) |l| denom += @exp(l - 2.0);
    try std.testing.expectApproxEqAbs(@exp(@as(f32, 0.0)) / denom, out_weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@exp(@as(f32, -1.0)) / denom, out_weights[1], 1e-6);
    try std.testing.expect(out_weights[0] + out_weights[1] < 1.0);
}

test "router tie-break: equal probabilities pick the LOWER expert id" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    // logits {1, 3, 3, 0}: experts 1 and 2 tie exactly at the top.
    const x = [_]f32{ 1, 0, 0, 0 };
    const w = [_]f32{
        1, 0, 0, 0,
        3, 0, 0, 0,
        3, 0, 0, 0,
        0, 0, 0, 0,
    };
    var out_ids: [3]u16 = undefined;
    var out_weights: [3]f32 = undefined;
    try routerTopK(ctx, .{
        .x = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&x)),
        .w = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&w)),
        .w_dtype = .f32,
        .n_tokens = 1,
        .dim = 4,
        .n_experts = 4,
        .top_k = 3,
        .norm_topk_prob = true,
        .out_ids = &out_ids,
        .out_weights = &out_weights,
    });
    // Descending probability with tie → lower id first: 1, then 2, then 0.
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 0 }, &out_ids);
    try std.testing.expectEqual(out_weights[0], out_weights[1]);
    // norm_topk_prob=true: the three weights sum to 1.
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        out_weights[0] + out_weights[1] + out_weights[2],
        1e-6,
    );
}

test "router f16 and q8_0 weight paths agree with dequantized f32" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    // dim = 32 (one q8_0 block per row), 4 experts. Build q8_0 rows, then
    // dequantize them with the frozen rule to get the f32 reference weights.
    const dim = 32;
    const n_experts = 4;
    var q8: [n_experts * 34]u8 = undefined;
    var wf32: [n_experts * dim]f32 = undefined;
    var wf16: [n_experts * dim]f16 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    for (0..n_experts) |e| {
        const scale: f16 = @floatCast(0.01 + 0.02 * @as(f32, @floatFromInt(e)));
        std.mem.writeInt(u16, q8[e * 34 ..][0..2], @bitCast(scale), .little);
        for (0..dim) |i| {
            const q: i8 = @intCast(rand.intRangeAtMost(i16, -127, 127));
            q8[e * 34 + 2 + i] = @bitCast(q);
            const v: f32 = @as(f32, @floatCast(scale)) * @as(f32, @floatFromInt(q));
            wf32[e * dim + i] = v;
            wf16[e * dim + i] = @floatCast(v); // scale*q fits f16 exactly here
        }
    }
    var x: [dim]f32 = undefined;
    for (&x) |*v| v.* = rand.floatNorm(f32);

    var args = contracts.RouterArgs{
        .x = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&x)),
        .w = undefined,
        .w_dtype = undefined,
        .n_tokens = 1,
        .dim = dim,
        .n_experts = n_experts,
        .top_k = 2,
        .norm_topk_prob = true,
        .out_ids = undefined,
        .out_weights = undefined,
    };

    var ids_f32: [2]u16 = undefined;
    var w_f32: [2]f32 = undefined;
    args.w = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&wf32));
    args.w_dtype = .f32;
    args.out_ids = &ids_f32;
    args.out_weights = &w_f32;
    try routerTopK(ctx, args);

    var ids_q8: [2]u16 = undefined;
    var w_q8: [2]f32 = undefined;
    args.w = try ctx.bufferFromBytes(&q8);
    args.w_dtype = .q8_0;
    args.out_ids = &ids_q8;
    args.out_weights = &w_q8;
    try routerTopK(ctx, args);

    var ids_f16: [2]u16 = undefined;
    var w_f16: [2]f32 = undefined;
    args.w = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&wf16));
    args.w_dtype = .f16;
    args.out_ids = &ids_f16;
    args.out_weights = &w_f16;
    try routerTopK(ctx, args);

    try std.testing.expectEqualSlices(u16, &ids_f32, &ids_q8);
    try std.testing.expectEqualSlices(u16, &ids_f32, &ids_f16);
    for (w_f32, w_q8) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
    for (w_f32, w_f16) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
}

test "expert mlp: empty pairs list leaves pre-zeroed out untouched" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    const x = [_]f32{0} ** 64; // 2 tokens × dim 32
    const bank = [_]u8{0} ** (34 * 32); // 1 expert q8_0, dim=ffn=32
    const out = try ctx.createBuffer(64 * @sizeOf(f32));
    try expertMlpSwiglu(ctx, .{
        .x = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&x)),
        .gate = try ctx.bufferFromBytes(&bank),
        .up = try ctx.bufferFromBytes(&bank),
        .down = try ctx.bufferFromBytes(&bank),
        .w_dtype = .q8_0,
        .pairs = &.{},
        .out = out,
        .n_tokens = 2,
        .dim = 32,
        .ffn_dim = 32,
        .n_experts = 1,
    });
    for (cpu.asConstF32(out)) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "expert mlp: f32 hand case with accumulation over two pairs" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    // dim=2, ffn=2, 2 experts, 1 token, weight applied and accumulated.
    // Expert 0: gate rows {{1,0},{0,1}}, up rows {{2,0},{0,2}}, down identity.
    // Expert 1: gate rows {{0,1},{1,0}}, up = expert0's up, down {{0,1},{1,0}}.
    const x = [_]f32{ 1.0, -2.0 };
    const gate = [_]f32{
        1, 0, 0, 1, // expert 0
        0, 1, 1, 0, // expert 1
    };
    const up = [_]f32{
        2, 0, 0, 2,
        2, 0, 0, 2,
    };
    const down = [_]f32{
        1, 0, 0, 1, // expert 0: identity
        0, 1, 1, 0, // expert 1: swap
    };
    const pairs = [_]contracts.PairDispatch{
        .{ .token = 0, .expert = 0, .weight = 0.5 },
        .{ .token = 0, .expert = 1, .weight = 2.0 },
    };
    const out = try ctx.createBuffer(2 * @sizeOf(f32));
    try expertMlpSwiglu(ctx, .{
        .x = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&x)),
        .gate = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&gate)),
        .up = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&up)),
        .down = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&down)),
        .w_dtype = .f32,
        .pairs = &pairs,
        .out = out,
        .n_tokens = 1,
        .dim = 2,
        .ffn_dim = 2,
        .n_experts = 2,
    });

    // Expert 0: h = {silu(1)*2, silu(-2)*(-4)}; down identity → h.
    // Expert 1: h = {silu(-2)*2, silu(1)*(-4)}; down swaps the two lanes.
    const s1 = siluF32(1.0);
    const s2 = siluF32(-2.0);
    const e0 = [_]f32{ s1 * 2.0, s2 * -4.0 };
    const e1_swapped = [_]f32{ s1 * -4.0, s2 * 2.0 };
    const got = cpu.asConstF32(out);
    try std.testing.expectApproxEqAbs(0.5 * e0[0] + 2.0 * e1_swapped[0], got[0], 1e-6);
    try std.testing.expectApproxEqAbs(0.5 * e0[1] + 2.0 * e1_swapped[1], got[1], 1e-6);
}

test "unsupported dtypes are rejected" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();
    var ids: [1]u16 = undefined;
    var ws: [1]f32 = undefined;
    try std.testing.expectError(KernelError.UnsupportedDtype, routerTopK(ctx, .{
        .x = .{},
        .w = .{},
        .w_dtype = .q4_k,
        .n_tokens = 1,
        .dim = 256,
        .n_experts = 1,
        .top_k = 1,
        .norm_topk_prob = true,
        .out_ids = &ids,
        .out_weights = &ws,
    }));
    try std.testing.expectError(KernelError.UnsupportedDtype, expertMlpSwiglu(ctx, .{
        .x = .{},
        .gate = .{},
        .up = .{},
        .down = .{},
        .w_dtype = .f16,
        .pairs = &.{},
        .out = .{},
        .n_tokens = 1,
        .dim = 32,
        .ffn_dim = 32,
        .n_experts = 1,
    }));
}
