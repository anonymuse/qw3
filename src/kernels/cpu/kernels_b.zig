//! Kernel set B (CPU reference): causal GQA attention over the KV cache.
//!
//! Frozen semantics (contracts.AttnArgs, ADR-005 §1/§3):
//!
//!   - q is f32 [n_tokens, n_q_heads, head_dim], post-rope post-qknorm.
//!   - K/V caches are f32 [n_kv_heads, max_ctx, head_dim]; only the first
//!     pos + n_tokens positions are valid. All addressing strides by max_ctx —
//!     NEVER by the valid length; fixture caches happen to be packed
//!     (max_ctx == pos + n_tokens) but real caches are not.
//!   - GQA mapping: q head h reads kv head h / (n_q_heads / n_kv_heads).
//!   - Causal: query token t sits at absolute position pos + t and attends to
//!     cache positions 0..pos+t inclusive.
//!   - scores = scale · (q · k), softmax in f32 with rowmax subtraction
//!     (numerical stability), accumulation f32.
//!   - out is f32 [n_tokens, n_q_heads * head_dim] (heads flattened per token).
//!
//! This is the permanent comparator for the Metal kernel
//! (src/kernels/shaders/kernels_b.metal; dispatch story in
//! PORTING-attention.md).

const std = @import("std");
const contracts = @import("../../shared/contracts.zig");
const cpu = @import("ctx.zig");
const fixture = @import("../../shared/fixture.zig");

const CpuCtx = cpu.CpuCtx;
const KernelError = contracts.KernelError;

pub fn gqaAttention(ctx: *CpuCtx, args: contracts.AttnArgs) KernelError!void {
    const hq: usize = args.n_q_heads;
    const hkv: usize = args.n_kv_heads;
    const hd: usize = args.head_dim;
    const n_tokens: usize = args.n_tokens;
    const pos: usize = args.pos;
    const max_ctx: usize = args.max_ctx;

    if (hq == 0 or hkv == 0 or hd == 0) return KernelError.ShapeMismatch;
    if (hq % hkv != 0) return KernelError.ShapeMismatch;
    if (pos + n_tokens > max_ctx) return KernelError.ShapeMismatch;

    const q_bytes = n_tokens * hq * hd * @sizeOf(f32);
    if (args.q.len < q_bytes or args.out.len < q_bytes) return KernelError.ShapeMismatch;

    const cache_bytes = if (args.kv_dtype == .f32)
        hkv * max_ctx * hd * @sizeOf(f32)
    else if (args.kv_dtype == .f16)
        hkv * max_ctx * hd * @sizeOf(f16)
    else
        return KernelError.UnsupportedDtype;
    if (args.k_cache.len < cache_bytes or args.v_cache.len < cache_bytes)
        return KernelError.ShapeMismatch;

    const group = hq / hkv;
    const q = cpu.asConstF32(args.q);
    const out = cpu.asF32(args.out);

    // Longest score row is the last query token's: pos + n_tokens positions.
    const scores = ctx.alloc.alloc(f32, pos + n_tokens) catch return KernelError.OutOfMemory;
    defer ctx.alloc.free(scores);

    for (0..n_tokens) |t| {
        const ctx_len = pos + t + 1;
        for (0..hq) |h| {
            const kh = h / group;
            const qr = q[(t * hq + h) * hd ..][0..hd];

            // Compute scores and attention based on cache dtype.
            if (args.kv_dtype == .f32) {
                const k = cpu.asConstF32(args.k_cache);
                const v = cpu.asConstF32(args.v_cache);
                const k_head = k[kh * max_ctx * hd ..];
                const v_head = v[kh * max_ctx * hd ..];

                var row_max: f32 = -std.math.inf(f32);
                for (0..ctx_len) |p| {
                    var acc: f32 = 0;
                    for (qr, k_head[p * hd ..][0..hd]) |a, b| acc += a * b;
                    const s = acc * args.scale;
                    scores[p] = s;
                    row_max = @max(row_max, s);
                }

                var sum: f32 = 0;
                for (scores[0..ctx_len]) |*s| {
                    s.* = @exp(s.* - row_max);
                    sum += s.*;
                }
                const inv_sum = 1.0 / sum;

                const out_row = out[t * hq * hd + h * hd ..][0..hd];
                @memset(out_row, 0);
                for (0..ctx_len) |p| {
                    const w = scores[p] * inv_sum;
                    for (out_row, v_head[p * hd ..][0..hd]) |*o, vv| o.* += w * vv;
                }
            } else if (args.kv_dtype == .f16) {
                const k = cpu.asConstF16(args.k_cache);
                const v = cpu.asConstF16(args.v_cache);
                const k_head = k[kh * max_ctx * hd ..];
                const v_head = v[kh * max_ctx * hd ..];

                var row_max: f32 = -std.math.inf(f32);
                for (0..ctx_len) |p| {
                    var acc: f32 = 0;
                    for (0..hd) |d| {
                        acc += qr[d] * @as(f32, @floatCast(k_head[p * hd + d]));
                    }
                    const s = acc * args.scale;
                    scores[p] = s;
                    row_max = @max(row_max, s);
                }

                var sum: f32 = 0;
                for (scores[0..ctx_len]) |*s| {
                    s.* = @exp(s.* - row_max);
                    sum += s.*;
                }
                const inv_sum = 1.0 / sum;

                const out_row = out[t * hq * hd + h * hd ..][0..hd];
                @memset(out_row, 0);
                for (0..ctx_len) |p| {
                    const w = scores[p] * inv_sum;
                    for (0..hd) |d| {
                        out_row[d] += w * @as(f32, @floatCast(v_head[p * hd + d]));
                    }
                }
            } else {
                return KernelError.UnsupportedDtype;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — frozen-signature check, fixture validation, semantics unit tests.
// ---------------------------------------------------------------------------

test "signature matches the frozen kernel api" {
    try std.testing.expect(@TypeOf(gqaAttention) ==
        fn (*CpuCtx, contracts.AttnArgs) KernelError!void);
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

test "synthetic attention fixtures: prefill + decode, all layers, in tolerance" {
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
        if (!std.mem.eql(u8, case.get("op").?.string, "attention")) continue;
        n_cases += 1;

        const params = case.get("params").?.object;
        const n_q_heads = jsonU32(params.get("n_q_heads").?);
        const n_kv_heads = jsonU32(params.get("n_kv_heads").?);
        const head_dim = jsonU32(params.get("head_dim").?);
        const scale = jsonF32(params.get("scale").?);
        const pos = jsonU32(params.get("pos").?);
        const n_tokens = jsonU32(params.get("n_tokens").?);
        const max_ctx = jsonU32(params.get("max_ctx").?);
        const kv_dtype: contracts.Dtype = if (params.get("kv_dtype")) |v| @enumFromInt(jsonU32(v)) else .f32;
        const tol = case.get("tolerance").?.object;

        const tensors = case.get("tensors").?.object;
        var q = try loadCaseTensor(alloc, tensors.get("q").?.string);
        defer q.free(alloc);
        var k_cache = try loadCaseTensor(alloc, tensors.get("k_cache").?.string);
        defer k_cache.free(alloc);
        var v_cache = try loadCaseTensor(alloc, tensors.get("v_cache").?.string);
        defer v_cache.free(alloc);
        var oracle_out = try loadCaseTensor(alloc, tensors.get("output").?.string);
        defer oracle_out.free(alloc);

        const out_buf = try ctx.createBuffer(@as(u64, n_tokens) * n_q_heads * head_dim * @sizeOf(f32));
        try gqaAttention(ctx, .{
            .q = try ctx.bufferFromBytes(q.data),
            .k_cache = try ctx.bufferFromBytes(k_cache.data),
            .v_cache = try ctx.bufferFromBytes(v_cache.data),
            .kv_dtype = kv_dtype,
            .out = out_buf,
            .pos = pos,
            .n_tokens = n_tokens,
            .n_q_heads = n_q_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_ctx = max_ctx,
            .scale = scale,
        });

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle_out.asF32(), cpu.asConstF32(out_buf), atol, rtol);
        std.debug.print("attention case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle_out.asF32(), cpu.asConstF32(out_buf), atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 6), n_cases);
}

test "single token at pos=0: output is v[kv_head, 0]; NaN tail proves masking" {
    // n_tokens=1, pos=0: every q head attends only position 0, so softmax is
    // exactly {1.0} and out[h] == v_cache[h/group, 0] regardless of q/k values.
    // max_ctx=3 with NaN at positions 1..2 proves both the causal mask and the
    // max_ctx stride: touching any invalid position poisons the output.
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    const hq = 4;
    const hkv = 2;
    const hd = 4;
    const max_ctx = 3;
    const nan = std.math.nan(f32);

    var q: [hq * hd]f32 = undefined;
    var k: [hkv * max_ctx * hd]f32 = undefined;
    var v: [hkv * max_ctx * hd]f32 = undefined;
    @memset(&k, nan);
    @memset(&v, nan);
    var prng = std.Random.DefaultPrng.init(1);
    const rand = prng.random();
    for (&q) |*x| x.* = rand.floatNorm(f32);
    for (0..hkv) |kh| {
        for (0..hd) |d| {
            k[kh * max_ctx * hd + d] = rand.floatNorm(f32); // position 0 only
            v[kh * max_ctx * hd + d] = @floatFromInt(kh * 100 + d);
        }
    }

    const out = try ctx.createBuffer(hq * hd * @sizeOf(f32));
    try gqaAttention(ctx, .{
        .q = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&q)),
        .k_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&k)),
        .v_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&v)),
        .kv_dtype = .f32,
        .out = out,
        .pos = 0,
        .n_tokens = 1,
        .n_q_heads = hq,
        .n_kv_heads = hkv,
        .head_dim = hd,
        .max_ctx = max_ctx,
        .scale = 0.5,
    });

    const got = cpu.asConstF32(out);
    for (0..hq) |h| {
        const kh = h / (hq / hkv);
        for (0..hd) |d| {
            const want: f32 = @floatFromInt(kh * 100 + d);
            try std.testing.expectApproxEqAbs(want, got[h * hd + d], 1e-6);
        }
    }
}

test "ragged pos mid-sequence: max_ctx stride, causality, packed equivalence" {
    // pos=2, n_tokens=2 in a max_ctx=8 cache whose tail (positions 4..7) is
    // NaN. Results must be bitwise identical to the same valid data packed
    // into a max_ctx=4 cache — proving the kernel strides by max_ctx and never
    // reads past pos+t. Then perturb position 3: token 0 (attends 0..2) must
    // be unchanged, token 1 (attends 0..3) must change.
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    const hq = 4;
    const hkv = 2;
    const hd = 8;
    const pos = 2;
    const n_tokens = 2;
    const valid = pos + n_tokens; // 4
    const big_ctx = 8;
    const nan = std.math.nan(f32);

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    var q: [n_tokens * hq * hd]f32 = undefined;
    for (&q) |*x| x.* = rand.floatNorm(f32);

    var k_packed: [hkv * valid * hd]f32 = undefined;
    var v_packed: [hkv * valid * hd]f32 = undefined;
    for (&k_packed) |*x| x.* = rand.floatNorm(f32);
    for (&v_packed) |*x| x.* = rand.floatNorm(f32);

    var k_big: [hkv * big_ctx * hd]f32 = undefined;
    var v_big: [hkv * big_ctx * hd]f32 = undefined;
    @memset(&k_big, nan);
    @memset(&v_big, nan);
    for (0..hkv) |kh| {
        const n = valid * hd;
        @memcpy(k_big[kh * big_ctx * hd ..][0..n], k_packed[kh * valid * hd ..][0..n]);
        @memcpy(v_big[kh * big_ctx * hd ..][0..n], v_packed[kh * valid * hd ..][0..n]);
    }

    var args = contracts.AttnArgs{
        .q = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&q)),
        .k_cache = undefined,
        .v_cache = undefined,
        .kv_dtype = .f32,
        .out = undefined,
        .pos = pos,
        .n_tokens = n_tokens,
        .n_q_heads = hq,
        .n_kv_heads = hkv,
        .head_dim = hd,
        .max_ctx = undefined,
        .scale = 1.0 / @sqrt(@as(f32, hd)),
    };

    const out_packed = try ctx.createBuffer(n_tokens * hq * hd * @sizeOf(f32));
    args.k_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&k_packed));
    args.v_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&v_packed));
    args.out = out_packed;
    args.max_ctx = valid;
    try gqaAttention(ctx, args);

    const out_big = try ctx.createBuffer(n_tokens * hq * hd * @sizeOf(f32));
    args.k_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&k_big));
    args.v_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&v_big));
    args.out = out_big;
    args.max_ctx = big_ctx;
    try gqaAttention(ctx, args);

    // Identical arithmetic in identical order → bitwise-equal outputs.
    try std.testing.expectEqualSlices(f32, cpu.asConstF32(out_packed), cpu.asConstF32(out_big));

    // Perturb cache position 3 (attended only by query token 1).
    for (0..hkv) |kh| {
        for (0..hd) |d| {
            k_big[(kh * big_ctx + 3) * hd + d] += 1.5;
            v_big[(kh * big_ctx + 3) * hd + d] -= 2.0;
        }
    }
    const out_pert = try ctx.createBuffer(n_tokens * hq * hd * @sizeOf(f32));
    args.out = out_pert;
    try gqaAttention(ctx, args);

    const base = cpu.asConstF32(out_big);
    const pert = cpu.asConstF32(out_pert);
    const row = hq * hd;
    try std.testing.expectEqualSlices(f32, base[0..row], pert[0..row]); // token 0 untouched
    var changed = false;
    for (base[row..], pert[row..]) |a, b| changed = changed or (a != b);
    try std.testing.expect(changed); // token 1 sees position 3
}

test "MHA degenerate (n_q_heads == n_kv_heads): hand-computed softmax mixture" {
    // 2 heads mapping 1:1 onto kv heads, head_dim=2, decode step at pos=1
    // (ctx_len 2). q chosen so scores are (1, -1) for head 0 and (0, 2) for
    // head 1 at scale=1; expected output is the analytic softmax mixture.
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    const hd = 2;
    // Head h of q: [1, 0] for h=0, [0, 1] for h=1.
    const q = [_]f32{ 1, 0, 0, 1 };
    // k_cache [2 heads, 2 positions, 2]: head 0 k = {(1,0), (-1,0)} → scores (1,-1)
    //                                    head 1 k = {(0,0), (0,2)}  → scores (0, 2)
    const k = [_]f32{
        1, 0, -1, 0, // kv head 0
        0, 0, 0,  2, // kv head 1
    };
    const v = [_]f32{
        10, 20, 30,  40, // kv head 0: v0=(10,20) v1=(30,40)
        -1, 1,  -3, 3, // kv head 1: v0=(-1,1) v1=(-3,3)
    };

    const out = try ctx.createBuffer(2 * hd * @sizeOf(f32));
    try gqaAttention(ctx, .{
        .q = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&q)),
        .k_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&k)),
        .v_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&v)),
        .kv_dtype = .f32,
        .out = out,
        .pos = 1,
        .n_tokens = 1,
        .n_q_heads = 2,
        .n_kv_heads = 2,
        .head_dim = hd,
        .max_ctx = 2,
        .scale = 1.0,
    });

    const got = cpu.asConstF32(out);
    // Head 0: p = softmax(1, -1)
    const p0: f32 = @exp(@as(f32, 1.0)) / (@exp(@as(f32, 1.0)) + @exp(@as(f32, -1.0)));
    try std.testing.expectApproxEqAbs(p0 * 10.0 + (1 - p0) * 30.0, got[0], 1e-5);
    try std.testing.expectApproxEqAbs(p0 * 20.0 + (1 - p0) * 40.0, got[1], 1e-5);
    // Head 1: p = softmax(0, 2)
    const p1: f32 = @exp(@as(f32, 0.0)) / (@exp(@as(f32, 0.0)) + @exp(@as(f32, 2.0)));
    try std.testing.expectApproxEqAbs(p1 * -1.0 + (1 - p1) * -3.0, got[2], 1e-5);
    try std.testing.expectApproxEqAbs(p1 * 1.0 + (1 - p1) * 3.0, got[3], 1e-5);
}

test "softmax stability: huge score magnitudes stay finite (rowmax subtraction)" {
    // Scores near ±200: naive exp overflows f32 (max ~exp(88)). With rowmax
    // subtraction the softmax saturates to the argmax position's v row.
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    const hd = 2;
    const q = [_]f32{ 100, 0 };
    const k = [_]f32{ -2, 0, 2, 0 }; // scores: -200, +200
    const v = [_]f32{ 7, -7, 3.5, -3.5 };

    const out = try ctx.createBuffer(hd * @sizeOf(f32));
    try gqaAttention(ctx, .{
        .q = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&q)),
        .k_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&k)),
        .v_cache = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&v)),
        .kv_dtype = .f32,
        .out = out,
        .pos = 1,
        .n_tokens = 1,
        .n_q_heads = 1,
        .n_kv_heads = 1,
        .head_dim = hd,
        .max_ctx = 2,
        .scale = 1.0,
    });
    const got = cpu.asConstF32(out);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), got[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3.5), got[1], 1e-6);
}

test "shape violations are rejected" {
    const alloc = std.testing.allocator;
    const ctx = try CpuCtx.init(alloc);
    defer ctx.deinit();

    var mem = [_]f32{0} ** 64;
    const buf = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&mem));
    var args = contracts.AttnArgs{
        .q = buf,
        .k_cache = buf,
        .v_cache = buf,
        .kv_dtype = .f32,
        .out = buf,
        .pos = 0,
        .n_tokens = 1,
        .n_q_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_ctx = 8,
        .scale = 0.5,
    };

    // pos + n_tokens > max_ctx
    args.pos = 8;
    try std.testing.expectError(KernelError.ShapeMismatch, gqaAttention(ctx, args));
    args.pos = 0;

    // n_q_heads not a multiple of n_kv_heads
    args.n_q_heads = 3;
    args.n_kv_heads = 2;
    try std.testing.expectError(KernelError.ShapeMismatch, gqaAttention(ctx, args));
    args.n_q_heads = 2;

    // cache buffer too small for [n_kv_heads, max_ctx, head_dim]
    args.max_ctx = 64;
    try std.testing.expectError(KernelError.ShapeMismatch, gqaAttention(ctx, args));
}
