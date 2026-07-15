//! Kernel set A (CPU reference): rmsNorm, rope, matmul (Q8_0/f32 dequant),
//! kvAppend, add. Frozen signatures per contracts.assertKernelApi (ADR-005).
//!
//! Semantics (ADR-005 §1, §6):
//!
//!   rmsNorm  y = x * rsqrt(mean(x²) + eps) * w, rowwise. Serves both the
//!            hidden-dim norms (dim = hidden_dim) and the Qwen3 per-head
//!            Q/K norm (dim = head_dim, n_rows = n_tokens·n_heads).
//!            `out` may alias `x`.
//!   rope     In-place NeoX rotation over the full head_dim: element i pairs
//!            with i + head_dim/2; angle_i = pos · freq_scale / theta^(2i/head_dim)
//!            for i in [0, head_dim/2). x'[i] = x[i]·cos − x[i+h/2]·sin;
//!            x'[i+h/2] = x[i]·sin + x[i+h/2]·cos. Positions come from the
//!            i32 buffer; freq_scale stays 1.0 in v1.
//!   matmul   out[m,n] = x[m,k] · Wᵀ. W is GGUF-layout (ne = {k, n}: n rows of
//!            k contiguous elements) in f32 or Q8_0; other dtypes →
//!            UnsupportedDtype. Q8_0 dequant is exactly f32(f16 scale) · i8 q;
//!            accumulation is f32, elementwise in storage order (matches the
//!            oracle, which compares against dequantized block values).
//!   kvAppend Scatter this step's K/V [n_tokens, n_kv_heads, head_dim] into
//!            the frozen cache layout [n_kv_heads, max_ctx, head_dim] at
//!            positions pos..pos+n_tokens-1.
//!   add      out = x + y elementwise; `out` may alias either input.

const std = @import("std");
const contracts = @import("../../shared/contracts.zig");
const cpu = @import("ctx.zig");
const fixture = @import("../../shared/fixture.zig");

const CpuCtx = cpu.CpuCtx;
const KernelError = contracts.KernelError;
const Dtype = contracts.Dtype;

// ---------------------------------------------------------------------------
// Row dot product against a GGUF-layout weight row (ne[0] contiguous).
// Accumulation is f32, elementwise in storage order.
// ---------------------------------------------------------------------------

fn dotRow(x: []const f32, row: []const u8, dtype: Dtype) f32 {
    var acc: f32 = 0;
    switch (dtype) {
        .f32 => {
            const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, row));
            for (w, x) |wv, xv| acc += wv * xv;
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

// ---------------------------------------------------------------------------
// rmsNorm
// ---------------------------------------------------------------------------

pub fn rmsNorm(ctx: *CpuCtx, args: contracts.RmsNormArgs) KernelError!void {
    _ = ctx;
    const n_rows: usize = args.n_rows;
    const dim: usize = args.dim;
    if (dim == 0) return KernelError.ShapeMismatch;
    if (args.x.len < n_rows * dim * @sizeOf(f32)) return KernelError.ShapeMismatch;
    if (args.out.len < n_rows * dim * @sizeOf(f32)) return KernelError.ShapeMismatch;
    if (args.weight.len < dim * @sizeOf(f32)) return KernelError.ShapeMismatch;

    const x = cpu.asConstF32(args.x);
    const w = cpu.asConstF32(args.weight)[0..dim];
    const out = cpu.asF32(args.out);

    var r: usize = 0;
    while (r < n_rows) : (r += 1) {
        const xr = x[r * dim ..][0..dim];
        const or_ = out[r * dim ..][0..dim];
        var sumsq: f32 = 0;
        for (xr) |v| sumsq += v * v;
        const inv = 1.0 / @sqrt(sumsq / @as(f32, @floatFromInt(dim)) + args.eps);
        // Alias-safe when out == x: element i is read before it is written.
        for (or_, xr, w) |*o, v, wv| o.* = v * inv * wv;
    }
}

// ---------------------------------------------------------------------------
// rope — NeoX pairing, in place, per head.
// ---------------------------------------------------------------------------

pub fn rope(ctx: *CpuCtx, args: contracts.RopeArgs) KernelError!void {
    _ = ctx;
    const n_tokens: usize = args.n_tokens;
    const n_heads: usize = args.n_heads;
    const head_dim: usize = args.head_dim;
    if (head_dim == 0 or head_dim % 2 != 0) return KernelError.ShapeMismatch;
    if (args.x.len < n_tokens * n_heads * head_dim * @sizeOf(f32)) return KernelError.ShapeMismatch;
    if (args.positions.len < n_tokens * @sizeOf(i32)) return KernelError.ShapeMismatch;

    const half = head_dim / 2;
    const x = cpu.asF32(args.x);
    const positions = cpu.asConstI32(args.positions)[0..n_tokens];

    var t: usize = 0;
    while (t < n_tokens) : (t += 1) {
        const pos: f32 = @floatFromInt(positions[t]);
        var h: usize = 0;
        while (h < n_heads) : (h += 1) {
            const head = x[(t * n_heads + h) * head_dim ..][0..head_dim];
            var i: usize = 0;
            while (i < half) : (i += 1) {
                // angle = pos · freq_scale / theta^(2i/head_dim), all f32.
                const expnt = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim));
                const angle = pos * args.freq_scale * std.math.pow(f32, args.theta, -expnt);
                const c = @cos(angle);
                const s = @sin(angle);
                const x0 = head[i];
                const x1 = head[i + half];
                head[i] = x0 * c - x1 * s;
                head[i + half] = x0 * s + x1 * c;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// matmul — out[m,n] = x[m,k] · Wᵀ, W in f32 or Q8_0 GGUF rows.
// ---------------------------------------------------------------------------

pub fn matmul(ctx: *CpuCtx, args: contracts.MatmulArgs) KernelError!void {
    _ = ctx;
    switch (args.w_dtype) {
        .f32, .q8_0 => {},
        else => return KernelError.UnsupportedDtype,
    }
    const m: usize = args.m;
    const n: usize = args.n;
    const k: usize = args.k;
    if (k == 0 or k % args.w_dtype.blockElems() != 0) return KernelError.ShapeMismatch;
    if (args.x.len < m * k * @sizeOf(f32)) return KernelError.ShapeMismatch;
    if (args.out.len < m * n * @sizeOf(f32)) return KernelError.ShapeMismatch;
    const row_bytes: usize = @intCast(args.w_dtype.rowBytes(k));
    if (args.w.len < n * row_bytes) return KernelError.ShapeMismatch;

    const x = cpu.asConstF32(args.x);
    const w = cpu.constBytes(args.w, 0, args.w.len);
    const out = cpu.asF32(args.out);

    var mi: usize = 0;
    while (mi < m) : (mi += 1) {
        const xr = x[mi * k ..][0..k];
        const or_ = out[mi * n ..][0..n];
        for (or_, 0..) |*o, ni| {
            o.* = dotRow(xr, w[ni * row_bytes ..][0..row_bytes], args.w_dtype);
        }
    }
}

// ---------------------------------------------------------------------------
// kvAppend — scatter step K/V into the per-layer cache at position `pos`.
// ---------------------------------------------------------------------------

pub fn kvAppend(ctx: *CpuCtx, args: contracts.KvAppendArgs) KernelError!void {
    _ = ctx;
    const n_tokens: usize = args.n_tokens;
    const n_kv_heads: usize = args.n_kv_heads;
    const head_dim: usize = args.head_dim;
    const max_ctx: usize = args.max_ctx;
    if (head_dim == 0 or n_kv_heads == 0) return KernelError.ShapeMismatch;
    if (args.pos + args.n_tokens > args.max_ctx) return KernelError.ShapeMismatch;

    const new_bytes = n_tokens * n_kv_heads * head_dim * @sizeOf(f32);
    if (args.k_new.len < new_bytes or args.v_new.len < new_bytes) return KernelError.ShapeMismatch;

    const cache_bytes = if (args.kv_dtype == .f32)
        n_kv_heads * max_ctx * head_dim * @sizeOf(f32)
    else if (args.kv_dtype == .f16)
        n_kv_heads * max_ctx * head_dim * @sizeOf(f16)
    else
        return KernelError.UnsupportedDtype;
    if (args.k_cache.len < cache_bytes or args.v_cache.len < cache_bytes) return KernelError.ShapeMismatch;

    const k_new = cpu.asConstF32(args.k_new);
    const v_new = cpu.asConstF32(args.v_new);

    var t: usize = 0;
    while (t < n_tokens) : (t += 1) {
        var h: usize = 0;
        while (h < n_kv_heads) : (h += 1) {
            const src = (t * n_kv_heads + h) * head_dim;
            const dst = (h * max_ctx + (args.pos + t)) * head_dim;

            if (args.kv_dtype == .f32) {
                const k_cache = cpu.asF32(args.k_cache);
                const v_cache = cpu.asF32(args.v_cache);
                @memcpy(k_cache[dst..][0..head_dim], k_new[src..][0..head_dim]);
                @memcpy(v_cache[dst..][0..head_dim], v_new[src..][0..head_dim]);
            } else if (args.kv_dtype == .f16) {
                const k_cache = cpu.asF16(args.k_cache);
                const v_cache = cpu.asF16(args.v_cache);
                var d: usize = 0;
                while (d < head_dim) : (d += 1) {
                    k_cache[dst + d] = @floatCast(k_new[src + d]);
                    v_cache[dst + d] = @floatCast(v_new[src + d]);
                }
            } else {
                return KernelError.UnsupportedDtype;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// add — elementwise residual add.
// ---------------------------------------------------------------------------

pub fn add(ctx: *CpuCtx, args: contracts.AddArgs) KernelError!void {
    _ = ctx;
    const n: usize = @intCast(args.n_elems);
    if (args.x.len < n * @sizeOf(f32)) return KernelError.ShapeMismatch;
    if (args.y.len < n * @sizeOf(f32)) return KernelError.ShapeMismatch;
    if (args.out.len < n * @sizeOf(f32)) return KernelError.ShapeMismatch;
    const x = cpu.asConstF32(args.x)[0..n];
    const y = cpu.asConstF32(args.y)[0..n];
    const out = cpu.asF32(args.out)[0..n];
    for (out, x, y) |*o, xv, yv| o.* = xv + yv;
}

// ---------------------------------------------------------------------------
// Tests — frozen-signature check, fixture validation, semantics unit tests.
// ---------------------------------------------------------------------------

test "signatures match the frozen kernel api" {
    // Kernel set A owns 5 of the 8 assertKernelApi decls; the full assert runs
    // over the merged provider namespace at integration (sets B/C fill the rest).
    try std.testing.expect(@TypeOf(rmsNorm) ==
        fn (*CpuCtx, contracts.RmsNormArgs) KernelError!void);
    try std.testing.expect(@TypeOf(rope) ==
        fn (*CpuCtx, contracts.RopeArgs) KernelError!void);
    try std.testing.expect(@TypeOf(matmul) ==
        fn (*CpuCtx, contracts.MatmulArgs) KernelError!void);
    try std.testing.expect(@TypeOf(kvAppend) ==
        fn (*CpuCtx, contracts.KvAppendArgs) KernelError!void);
    try std.testing.expect(@TypeOf(add) ==
        fn (*CpuCtx, contracts.AddArgs) KernelError!void);
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
    if (std.mem.eql(u8, name, "q8_0")) return .q8_0;
    unreachable;
}

test "synthetic rmsnorm fixtures: hidden-dim and per-head norms in tolerance" {
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
        if (!std.mem.eql(u8, case.get("op").?.string, "rmsnorm")) continue;
        n_cases += 1;

        const params = case.get("params").?.object;
        const n_rows = jsonU32(params.get("n_rows").?);
        const dim = jsonU32(params.get("dim").?);
        const eps = jsonF32(params.get("eps").?);
        const tol = case.get("tolerance").?.object;

        const tensors = case.get("tensors").?.object;
        var input = try loadCaseTensor(alloc, tensors.get("input").?.string);
        defer input.free(alloc);
        var weight = try loadCaseTensor(alloc, tensors.get("weight").?.string);
        defer weight.free(alloc);
        var oracle = try loadCaseTensor(alloc, tensors.get("output").?.string);
        defer oracle.free(alloc);

        try std.testing.expectEqual(@as(u64, dim), input.desc.ne[0]);
        try std.testing.expectEqual(@as(u64, n_rows), input.desc.rows());

        const out = try ctx.createBuffer(@as(u64, n_rows) * dim * @sizeOf(f32));
        try rmsNorm(ctx, .{
            .x = try ctx.bufferFromBytes(input.data),
            .weight = try ctx.bufferFromBytes(weight.data),
            .out = out,
            .n_rows = n_rows,
            .dim = dim,
            .eps = eps,
        });

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle.asF32(), cpu.asConstF32(out), atol, rtol);
        std.debug.print("rmsnorm case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), cpu.asConstF32(out), atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 6), n_cases);
}

test "synthetic rope fixtures: NeoX rotation in place, in tolerance" {
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
        if (!std.mem.eql(u8, case.get("op").?.string, "rope")) continue;
        n_cases += 1;

        const params = case.get("params").?.object;
        const n_tokens = jsonU32(params.get("n_tokens").?);
        const n_heads = jsonU32(params.get("n_heads").?);
        const head_dim = jsonU32(params.get("head_dim").?);
        const theta = jsonF32(params.get("theta").?);
        const freq_scale = jsonF32(params.get("freq_scale").?);
        const tol = case.get("tolerance").?.object;

        const tensors = case.get("tensors").?.object;
        var input = try loadCaseTensor(alloc, tensors.get("input").?.string);
        defer input.free(alloc);
        var positions = try loadCaseTensor(alloc, tensors.get("positions").?.string);
        defer positions.free(alloc);
        var oracle = try loadCaseTensor(alloc, tensors.get("output").?.string);
        defer oracle.free(alloc);

        try std.testing.expectEqual(@as(u64, head_dim), input.desc.ne[0]);
        try std.testing.expectEqual(@as(u64, n_heads), input.desc.ne[1]);
        try std.testing.expectEqual(@as(u64, n_tokens), input.desc.ne[2]);
        try std.testing.expectEqual(contracts.Dtype.i32, positions.desc.dtype);

        // rope is in place: copy the input into a mutable buffer first.
        const x = try ctx.createBuffer(input.data.len);
        try ctx.upload(x, 0, input.data);
        try rope(ctx, .{
            .x = x,
            .positions = try ctx.bufferFromBytes(positions.data),
            .n_tokens = n_tokens,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .theta = theta,
            .freq_scale = freq_scale,
        });

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle.asF32(), cpu.asConstF32(x), atol, rtol);
        std.debug.print("rope case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), cpu.asConstF32(x), atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 3), n_cases);
}

test "synthetic matmul_quant fixtures: q8_0 dequant matmul in tolerance" {
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
        if (!std.mem.eql(u8, case.get("op").?.string, "matmul_quant")) continue;
        n_cases += 1;

        const params = case.get("params").?.object;
        const m = jsonU32(params.get("m").?);
        const n = jsonU32(params.get("n").?);
        const k = jsonU32(params.get("k").?);
        const w_dtype = dtypeFromName(params.get("w_dtype").?.string);
        const tol = case.get("tolerance").?.object;

        const tensors = case.get("tensors").?.object;
        var input = try loadCaseTensor(alloc, tensors.get("input").?.string);
        defer input.free(alloc);
        var weight = try loadCaseTensor(alloc, tensors.get("weight").?.string);
        defer weight.free(alloc);
        var oracle = try loadCaseTensor(alloc, tensors.get("output").?.string);
        defer oracle.free(alloc);

        try std.testing.expectEqual(@as(u64, k), input.desc.ne[0]); // input ne = {k, m}
        try std.testing.expectEqual(@as(u64, m), input.desc.ne[1]);
        try std.testing.expectEqual(@as(u64, k), weight.desc.ne[0]); // weight ne = {k, n}
        try std.testing.expectEqual(@as(u64, n), weight.desc.ne[1]);
        try std.testing.expectEqual(@as(u64, n), oracle.desc.ne[0]); // output ne = {n, m}
        try std.testing.expectEqual(@as(u64, m), oracle.desc.ne[1]);

        const out = try ctx.createBuffer(@as(u64, m) * n * @sizeOf(f32));
        try matmul(ctx, .{
            .x = try ctx.bufferFromBytes(input.data),
            .w = try ctx.bufferFromBytes(weight.data),
            .w_dtype = w_dtype,
            .out = out,
            .m = m,
            .n = n,
            .k = k,
        });

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle.asF32(), cpu.asConstF32(out), atol, rtol);
        std.debug.print("matmul_quant case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), cpu.asConstF32(out), atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 4), n_cases);
}

test "rmsnorm: hand case and out-aliasing-x" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    // Row {3, 4}: mean(x²) = 12.5, rms = sqrt(12.5 + eps). w = {2, 0.5}.
    const eps: f32 = 1e-6;
    const vals = [_]f32{ 3, 4, -1, 1 }; // 2 rows, dim 2
    const w = [_]f32{ 2.0, 0.5 };
    const x = try ctx.createBuffer(vals.len * @sizeOf(f32));
    try ctx.upload(x, 0, std.mem.sliceAsBytes(&vals));

    // out aliases x — the contract explicitly allows it.
    try rmsNorm(ctx, .{
        .x = x,
        .weight = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&w)),
        .out = x,
        .n_rows = 2,
        .dim = 2,
        .eps = eps,
    });

    const got = cpu.asConstF32(x);
    const inv0 = 1.0 / @sqrt(@as(f32, 12.5) + eps);
    const inv1 = 1.0 / @sqrt(@as(f32, 1.0) + eps);
    try std.testing.expectApproxEqAbs(3.0 * inv0 * 2.0, got[0], 1e-6);
    try std.testing.expectApproxEqAbs(4.0 * inv0 * 0.5, got[1], 1e-6);
    try std.testing.expectApproxEqAbs(-1.0 * inv1 * 2.0, got[2], 1e-6);
    try std.testing.expectApproxEqAbs(1.0 * inv1 * 0.5, got[3], 1e-6);
}

test "rope: NeoX pairing hand case, position 0 is identity" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    // 2 tokens, 1 head, head_dim 4 (half = 2). Token 0 at pos 0 (identity),
    // token 1 at pos 3: angle_0 = 3/theta^0 = 3, angle_1 = 3/theta^(1/2).
    const theta: f32 = 100.0;
    const vals = [_]f32{
        1, 2, 3, 4, // token 0
        1, 0, 0, 1, // token 1: pairs (x0=1, x2=0) and (x1=0, x3=1)
    };
    const positions = [_]i32{ 0, 3 };
    const x = try ctx.createBuffer(vals.len * @sizeOf(f32));
    try ctx.upload(x, 0, std.mem.sliceAsBytes(&vals));

    try rope(ctx, .{
        .x = x,
        .positions = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&positions)),
        .n_tokens = 2,
        .n_heads = 1,
        .head_dim = 4,
        .theta = theta,
    });

    const got = cpu.asConstF32(x);
    // pos 0: cos 0 = 1, sin 0 = 0 → unchanged.
    for (vals[0..4], got[0..4]) |want, g| try std.testing.expectApproxEqAbs(want, g, 1e-7);
    // pos 3, pair i=0 (elements 0 and 2), x0=1, x2=0: x0'=cos(3), x2'=sin(3).
    const a0: f32 = 3.0;
    try std.testing.expectApproxEqAbs(@cos(a0), got[4], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(a0), got[6], 1e-6);
    // pair i=1 (elements 1 and 3), x1=0, x3=1: x1'=-sin(a1), x3'=cos(a1).
    const a1: f32 = 3.0 * std.math.pow(f32, theta, -0.5);
    try std.testing.expectApproxEqAbs(-@sin(a1), got[5], 1e-6);
    try std.testing.expectApproxEqAbs(@cos(a1), got[7], 1e-6);
}

test "matmul: f32 weights hand case" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    // x [2,3] · Wᵀ with W ne = {3, 2}: rows w0 = {1,0,2}, w1 = {0,-1,1}.
    const x = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const w = [_]f32{
        1, 0,  2,
        0, -1, 1,
    };
    const out = try ctx.createBuffer(4 * @sizeOf(f32));
    try matmul(ctx, .{
        .x = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&x)),
        .w = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&w)),
        .w_dtype = .f32,
        .out = out,
        .m = 2,
        .n = 2,
        .k = 3,
    });
    const got = cpu.asConstF32(out);
    try std.testing.expectApproxEqAbs(@as(f32, 7), got[0], 1e-6); // 1+0+6
    try std.testing.expectApproxEqAbs(@as(f32, 1), got[1], 1e-6); // 0-2+3
    try std.testing.expectApproxEqAbs(@as(f32, 16), got[2], 1e-6); // 4+0+12
    try std.testing.expectApproxEqAbs(@as(f32, 1), got[3], 1e-6); // 0-5+6
}

test "matmul: q8_0 path agrees with dequantized f32 weights" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    // k = 64 (two q8_0 blocks per row), n = 3 rows. Build q8_0 rows, then
    // dequantize with the frozen rule to get the f32 reference weights.
    const k = 64;
    const n = 3;
    var q8: [n * 2 * 34]u8 = undefined;
    var wf32: [n * k]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    const rand = prng.random();
    for (0..n) |r| {
        for (0..2) |b| {
            const blk = q8[(r * 2 + b) * 34 ..][0..34];
            const scale: f16 = @floatCast(0.005 + 0.01 * @as(f32, @floatFromInt(r + b)));
            std.mem.writeInt(u16, blk[0..2], @bitCast(scale), .little);
            for (0..32) |i| {
                const q: i8 = @intCast(rand.intRangeAtMost(i16, -127, 127));
                blk[2 + i] = @bitCast(q);
                wf32[r * k + b * 32 + i] = @as(f32, @floatCast(scale)) * @as(f32, @floatFromInt(q));
            }
        }
    }
    var x: [2 * k]f32 = undefined;
    for (&x) |*v| v.* = rand.floatNorm(f32);

    const out_q8 = try ctx.createBuffer(2 * n * @sizeOf(f32));
    const out_f32 = try ctx.createBuffer(2 * n * @sizeOf(f32));
    var args = contracts.MatmulArgs{
        .x = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&x)),
        .w = try ctx.bufferFromBytes(&q8),
        .w_dtype = .q8_0,
        .out = out_q8,
        .m = 2,
        .n = n,
        .k = k,
    };
    try matmul(ctx, args);
    args.w = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&wf32));
    args.w_dtype = .f32;
    args.out = out_f32;
    try matmul(ctx, args);

    for (cpu.asConstF32(out_f32), cpu.asConstF32(out_q8)) |a, b|
        try std.testing.expectApproxEqAbs(a, b, 1e-5);
}

test "matmul: unsupported dtypes are rejected" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();
    try std.testing.expectError(KernelError.UnsupportedDtype, matmul(ctx, .{
        .x = .{},
        .w = .{},
        .w_dtype = .q4_k,
        .out = .{},
        .m = 1,
        .n = 1,
        .k = 256,
    }));
}

test "kvAppend: round-trip scatter, prior cache contents untouched" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    const n_kv_heads = 2;
    const head_dim = 4;
    const max_ctx = 8;
    const pos = 3; // 3 tokens already cached
    const n_tokens = 2;

    // Pre-fill the caches with a sentinel pattern to detect stray writes.
    const cache_elems = n_kv_heads * max_ctx * head_dim;
    const k_cache = try ctx.createBuffer(cache_elems * @sizeOf(f32));
    const v_cache = try ctx.createBuffer(cache_elems * @sizeOf(f32));
    for (cpu.asF32(k_cache), 0..) |*v, i| v.* = -@as(f32, @floatFromInt(i));
    for (cpu.asF32(v_cache), 0..) |*v, i| v.* = -1000.0 - @as(f32, @floatFromInt(i));
    var k_before: [cache_elems]f32 = undefined;
    var v_before: [cache_elems]f32 = undefined;
    @memcpy(&k_before, cpu.asConstF32(k_cache));
    @memcpy(&v_before, cpu.asConstF32(v_cache));

    // k_new/v_new [n_tokens, n_kv_heads, head_dim] with recognizable values:
    // k = 100·t + 10·h + d, v = the negation minus 0.5.
    var k_new: [n_tokens * n_kv_heads * head_dim]f32 = undefined;
    var v_new: [n_tokens * n_kv_heads * head_dim]f32 = undefined;
    for (0..n_tokens) |t| for (0..n_kv_heads) |h| for (0..head_dim) |d| {
        const idx = (t * n_kv_heads + h) * head_dim + d;
        k_new[idx] = @floatFromInt(100 * t + 10 * h + d);
        v_new[idx] = -k_new[idx] - 0.5;
    };

    try kvAppend(ctx, .{
        .k_new = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&k_new)),
        .v_new = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&v_new)),
        .k_cache = k_cache,
        .v_cache = v_cache,
        .kv_dtype = .f32,
        .pos = pos,
        .n_tokens = n_tokens,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .max_ctx = max_ctx,
    });

    const kc = cpu.asConstF32(k_cache);
    const vc = cpu.asConstF32(v_cache);
    for (0..n_kv_heads) |h| {
        for (0..max_ctx) |p| {
            for (0..head_dim) |d| {
                const ci = (h * max_ctx + p) * head_dim + d;
                if (p >= pos and p < pos + n_tokens) {
                    const t = p - pos;
                    const si = (t * n_kv_heads + h) * head_dim + d;
                    try std.testing.expectEqual(k_new[si], kc[ci]);
                    try std.testing.expectEqual(v_new[si], vc[ci]);
                } else {
                    // Slots outside pos..pos+n_tokens keep their old values.
                    try std.testing.expectEqual(k_before[ci], kc[ci]);
                    try std.testing.expectEqual(v_before[ci], vc[ci]);
                }
            }
        }
    }
}

test "kvAppend: overflow past max_ctx is rejected" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();
    try std.testing.expectError(KernelError.ShapeMismatch, kvAppend(ctx, .{
        .k_new = .{},
        .v_new = .{},
        .k_cache = .{},
        .v_cache = .{},
        .kv_dtype = .f32,
        .pos = 7,
        .n_tokens = 2,
        .n_kv_heads = 1,
        .head_dim = 4,
        .max_ctx = 8,
    }));
}

test "add: matches scalar loop, aliasing out onto x allowed" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    const n = 97; // deliberately odd-sized
    var xv: [n]f32 = undefined;
    var yv: [n]f32 = undefined;
    for (&xv) |*v| v.* = rand.floatNorm(f32);
    for (&yv) |*v| v.* = rand.floatNorm(f32);
    var want: [n]f32 = undefined;
    for (&want, xv, yv) |*w, a, b| w.* = a + b;

    const x = try ctx.createBuffer(n * @sizeOf(f32));
    try ctx.upload(x, 0, std.mem.sliceAsBytes(&xv));
    const y = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&yv));

    // out aliases x (residual-add usage in the forward recipe).
    try add(ctx, .{ .x = x, .y = y, .out = x, .n_elems = n });
    for (want, cpu.asConstF32(x)) |w, g| try std.testing.expectEqual(w, g);
}
