//! T05 — M2b: GPU kernel provider. Implements contracts.assertKernelApi over
//! the Metal glue Ctx (src/metal/metal.zig), dispatching the shaders already
//! written for W3/W4/W5 (src/kernels/shaders/kernels_{a,b,c}.metal) exactly
//! per their PORTING-*.md dispatch contracts (params layouts, binding
//! indices, grid geometry). Those shaders were frozen but never wired to a
//! kernel-provider module before this file — that wiring is the whole job
//! here; the shader math itself is untouched.
//!
//! Router (routerTopK): PORTING-moe.md §1 freezes this as CPU-only,
//! permanently — no Metal kernel exists for it. This module's routerTopK
//! reads the ffn-normed activations and router weight back to the host
//! (`Ctx.download`, which auto-flushes any pending batch — see the T05 note
//! on `download` in metal.zig) and runs the identical top-k algorithm the
//! CPU reference (`kernels/cpu/kernels_c.zig`) uses. It is a deliberate
//! reimplementation rather than a call into that CPU function: the CPU
//! version is hard-typed to `*CpuCtx` and its buffer access
//! (`cpu.asConstF32`/`cpu.constBytes`) casts `Buf.handle` directly as a raw
//! host pointer, which is only true for `CpuCtx` — a Metal `Buf.handle` is an
//! `MTLBuffer` object, not a data pointer. Editing kernels_c.zig to
//! genericize it is out of scope ("Forbidden: editing CPU kernels"), so the
//! ~40 lines of softmax/top-k logic here are kept byte-for-byte in step with
//! the CPU version (same f32 op order ⇒ bit-identical router output, which
//! is the frozen requirement — ADR-005 §2 exact expert-id match).
//!
//! Command-buffer batching: every op below opens the current batch lazily
//! (`ensureBatch`) and only *encodes* — nothing calls `ctx.submit()` except
//! routerTopK's forced host-sync and the final `ctx.download(logits)` that
//! `Engine.forward()` does itself (also auto-flushing, per the same T05 fix).
//! Net effect: exactly one submit per layer at the router boundary, well
//! inside the ≤3/layer target — see docs/orchestration/prompts/T05-m2b-gpu-forward.md.
//!
//! KV cache dtype: ADR-005 was amended 2026-07-13 to freeze KV cache to f16,
//! but that amendment only edited the ADR prose — contracts.zig's
//! `KvAppendArgs`/`AttnArgs` docs, the frozen PORTING-kernels-a/b.md dispatch
//! contracts, kernels_a.metal/kernels_b.metal, and the CPU reference kernels
//! all still say/do f32, and the committed fixtures store f32 cache tensors.
//! Implementing f16 on the GPU side only would (a) contradict the "implement
//! EXACTLY these PORTING docs" instruction, (b) diverge CPU vs GPU cache
//! layout in a way the shared Engine (src/engine/forward.zig) doesn't know
//! how to size per-backend, and (c) risk the attention tolerance (rtol 1e-3)
//! on an unreviewed numerics change. This module keeps the KV cache f32,
//! matching the still-frozen PORTING docs and the CPU reference exactly. The
//! f16 amendment needs its own follow-up (contracts.zig + both PORTING docs +
//! both kernels_*.metal + CPU kernels_a/b.zig + fixture regen) before any
//! implementation is safe — flagged separately, not decided unilaterally here.

const std = @import("std");
const contracts = @import("../../shared/contracts.zig");
const metal = @import("../../metal/metal.zig");
const fixture = @import("../../shared/fixture.zig");

const KernelError = contracts.KernelError;
const Dtype = contracts.Dtype;
const Ctx = metal.Ctx;

// ---------------------------------------------------------------------------
// Shader sources. Compiled into the Ctx's library list once via loadShaders();
// callers (tests, `ds5 run --backend metal`) call this right after Ctx.init(),
// mirroring the pattern metal.zig's own doc comment describes for W3-W5.
// ---------------------------------------------------------------------------

const kernels_a_src: [:0]const u8 = @embedFile("../shaders/kernels_a.metal");
const kernels_b_src: [:0]const u8 = @embedFile("../shaders/kernels_b.metal");
const kernels_c_src: [:0]const u8 = @embedFile("../shaders/kernels_c.metal");

pub fn loadShaders(ctx: *Ctx) KernelError!void {
    try ctx.addLibrary(kernels_a_src);
    try ctx.addLibrary(kernels_b_src);
    try ctx.addLibrary(kernels_c_src);
}

// ---------------------------------------------------------------------------
// Params structs — byte-identical to the MSL structs in kernels_{a,b,c}.metal
// (verified with @sizeOf below, matching the PORTING docs' documented sizes).
// ---------------------------------------------------------------------------

const RmsNormParams = extern struct { n_rows: u32, dim: u32, eps: f32 };
const RopeParams = extern struct { n_tokens: u32, n_heads: u32, head_dim: u32, theta: f32, freq_scale: f32 };
const MatmulParams = extern struct { m: u32, n: u32, k: u32 };
const KvAppendParams = extern struct { n_tokens: u32, n_kv_heads: u32, head_dim: u32, pos: u32, max_ctx: u32 };
const GqaAttnParams = extern struct {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    pos: u32,
    n_tokens: u32,
    max_ctx: u32,
    scale: f32,
};
const ExpertMlpParams = extern struct { dim: u32, ffn_dim: u32, n_experts: u32, n_pairs: u32, tile_out: u32 };

comptime {
    std.debug.assert(@sizeOf(RmsNormParams) == 12);
    std.debug.assert(@sizeOf(RopeParams) == 20);
    std.debug.assert(@sizeOf(MatmulParams) == 12);
    std.debug.assert(@sizeOf(KvAppendParams) == 20);
    std.debug.assert(@sizeOf(GqaAttnParams) == 28);
    std.debug.assert(@sizeOf(ExpertMlpParams) == 20);
    std.debug.assert(@sizeOf(contracts.PairDispatch) == 12); // frozen; kernels_c.metal assumes this
}

// Layout constants mirrored from the shaders (must match kernels_{a,b,c}.metal
// and the PORTING docs — see the doc comment at the top of each .metal file).
const RMSNORM_TG: u32 = 256;
const DS5_ATTN_TG: u32 = 128;
const DS5_MAX_HEAD_DIM: u32 = 128;
const DS5_MAX_FFN: u32 = 1536;

// ---------------------------------------------------------------------------
// Dispatch helpers.
// ---------------------------------------------------------------------------

/// Opens the batch lazily so consecutive ops accumulate into one command
/// buffer (the layer-batching this module relies on for the ≤3-submits
/// target). Only routerTopK (and Engine's own final download) ever submits.
fn ensureBatch(ctx: *Ctx) void {
    if (ctx.cmdbuf == null) ctx.begin();
}

/// Threadgroup-count grid: for kernels the PORTING docs dispatch by exact
/// threadgroup count (rmsnorm_f32, gqa_attention_f32, expert_mlp_swiglu_q8 —
/// all threadgroup-barrier kernels). `groups` is used verbatim, no rounding.
fn dispatchGroups(ctx: *Ctx, gx: u32, gy: u32, gz: u32, tgx: u32, tgy: u32, tgz: u32) void {
    ctx.dispatch(.{ .width = gx, .height = gy, .depth = gz }, .{ .width = tgx, .height = tgy, .depth = tgz });
}

/// dispatchThreads-equivalent for the PORTING docs' thread-count grids (rope,
/// matmul, kv_append). Every one of those shaders bounds-checks its global
/// thread id against the real element counts ("Every kernel bounds-checks,
/// so over-provisioned grids are safe" — PORTING-kernels-a.md), so
/// ceil-rounding into dispatchThreadgroups produces identical results to a
/// true `dispatchThreads:` call and reuses the glue's existing dispatch()
/// helper instead of adding a new Objective-C selector.
fn dispatchThreadsCeil(ctx: *Ctx, nx: u32, ny: u32, nz: u32, tgx: u32, tgy: u32, tgz: u32) void {
    const gx = (nx + tgx - 1) / tgx;
    const gy = (ny + tgy - 1) / tgy;
    const gz = (nz + tgz - 1) / tgz;
    dispatchGroups(ctx, gx, gy, gz, tgx, tgy, tgz);
}

// ---------------------------------------------------------------------------
// rmsNorm — rmsnorm_f32, dispatched by threadgroups (PORTING-kernels-a.md).
// ---------------------------------------------------------------------------

pub fn rmsNorm(ctx: *Ctx, args: contracts.RmsNormArgs) KernelError!void {
    if (args.dim == 0) return KernelError.ShapeMismatch;
    ensureBatch(ctx);
    const pso = try ctx.pipeline("rmsnorm_f32");
    ctx.setPipeline(pso);
    const p = RmsNormParams{ .n_rows = args.n_rows, .dim = args.dim, .eps = args.eps };
    ctx.setBytes(0, std.mem.asBytes(&p));
    ctx.setBuf(1, args.x);
    ctx.setBuf(2, args.weight);
    ctx.setBuf(3, args.out);
    dispatchGroups(ctx, args.n_rows, 1, 1, RMSNORM_TG, 1, 1);
}

// ---------------------------------------------------------------------------
// rope — rope_f32, dispatched by (over-provisioned) thread-count grid.
// ---------------------------------------------------------------------------

pub fn rope(ctx: *Ctx, args: contracts.RopeArgs) KernelError!void {
    if (args.head_dim == 0 or args.head_dim % 2 != 0) return KernelError.ShapeMismatch;
    ensureBatch(ctx);
    const pso = try ctx.pipeline("rope_f32");
    ctx.setPipeline(pso);
    const p = RopeParams{
        .n_tokens = args.n_tokens,
        .n_heads = args.n_heads,
        .head_dim = args.head_dim,
        .theta = args.theta,
        .freq_scale = args.freq_scale,
    };
    ctx.setBytes(0, std.mem.asBytes(&p));
    ctx.setBuf(1, args.x);
    ctx.setBuf(2, args.positions);
    const half = args.head_dim / 2;
    dispatchThreadsCeil(ctx, half, args.n_heads, args.n_tokens, @min(half, 64), 1, 1);
}

// ---------------------------------------------------------------------------
// matmul — matmul_q8_0 / matmul_f32, selected by w_dtype (PORTING-kernels-a.md
// glue-side dtype gate; anything else is UnsupportedDtype before encoding).
// ---------------------------------------------------------------------------

pub fn matmul(ctx: *Ctx, args: contracts.MatmulArgs) KernelError!void {
    const name: [:0]const u8 = switch (args.w_dtype) {
        .f32 => "matmul_f32",
        .q8_0 => "matmul_q8_0",
        else => return KernelError.UnsupportedDtype,
    };
    if (args.k == 0 or args.k % args.w_dtype.blockElems() != 0) return KernelError.ShapeMismatch;
    ensureBatch(ctx);
    const pso = try ctx.pipeline(name);
    ctx.setPipeline(pso);
    const p = MatmulParams{ .m = args.m, .n = args.n, .k = args.k };
    ctx.setBytes(0, std.mem.asBytes(&p));
    ctx.setBuf(1, args.x);
    ctx.setBuf(2, args.w);
    ctx.setBuf(3, args.out);
    dispatchThreadsCeil(ctx, args.n, args.m, 1, 256, 1, 1);
}

// ---------------------------------------------------------------------------
// kvAppend — kv_append_f32. Glue-side precondition matches the CPU reference
// exactly (PORTING-kernels-a.md: "the shader has no error channel").
// ---------------------------------------------------------------------------

pub fn kvAppend(ctx: *Ctx, args: contracts.KvAppendArgs) KernelError!void {
    if (args.head_dim == 0 or args.n_kv_heads == 0) return KernelError.ShapeMismatch;
    if (args.pos + args.n_tokens > args.max_ctx) return KernelError.ShapeMismatch;
    ensureBatch(ctx);
    const pso = try ctx.pipeline("kv_append_f32");
    ctx.setPipeline(pso);
    const p = KvAppendParams{
        .n_tokens = args.n_tokens,
        .n_kv_heads = args.n_kv_heads,
        .head_dim = args.head_dim,
        .pos = args.pos,
        .max_ctx = args.max_ctx,
    };
    ctx.setBytes(0, std.mem.asBytes(&p));
    ctx.setBuf(1, args.k_new);
    ctx.setBuf(2, args.v_new);
    ctx.setBuf(3, args.k_cache);
    ctx.setBuf(4, args.v_cache);
    dispatchThreadsCeil(ctx, args.head_dim, args.n_kv_heads, args.n_tokens, 64, 1, 1);
}

// ---------------------------------------------------------------------------
// add — frozen `out = x + y`. metal.zig already implements exactly this
// signature (the `proof_add` kernel, proven against fixtures in
// test_metal.zig); reuse it verbatim rather than re-encode the same
// dispatch, wrapping only to open the batch first.
// ---------------------------------------------------------------------------

pub fn add(ctx: *Ctx, args: contracts.AddArgs) KernelError!void {
    ensureBatch(ctx);
    return metal.add(ctx, args);
}

// ---------------------------------------------------------------------------
// gqaAttention — gqa_attention_f32. Glue-side preconditions mirror
// kernels_b.zig exactly (PORTING-kernels-b.md §"Glue-side preconditions").
// ---------------------------------------------------------------------------

pub fn gqaAttention(ctx: *Ctx, args: contracts.AttnArgs) KernelError!void {
    if (args.n_q_heads == 0 or args.n_kv_heads == 0 or args.head_dim == 0) return KernelError.ShapeMismatch;
    if (args.n_q_heads % args.n_kv_heads != 0) return KernelError.ShapeMismatch;
    if (args.pos + args.n_tokens > args.max_ctx) return KernelError.ShapeMismatch;
    if (args.head_dim > DS5_MAX_HEAD_DIM) return KernelError.ShapeMismatch;
    const q_bytes = @as(u64, args.n_tokens) * args.n_q_heads * args.head_dim * @sizeOf(f32);
    if (args.q.len < q_bytes or args.out.len < q_bytes) return KernelError.ShapeMismatch;
    const cache_bytes = @as(u64, args.n_kv_heads) * args.max_ctx * args.head_dim * @sizeOf(f32);
    if (args.k_cache.len < cache_bytes or args.v_cache.len < cache_bytes) return KernelError.ShapeMismatch;

    ensureBatch(ctx);
    const pso = try ctx.pipeline("gqa_attention_f32");
    ctx.setPipeline(pso);
    const p = GqaAttnParams{
        .n_q_heads = args.n_q_heads,
        .n_kv_heads = args.n_kv_heads,
        .head_dim = args.head_dim,
        .pos = args.pos,
        .n_tokens = args.n_tokens,
        .max_ctx = args.max_ctx,
        .scale = args.scale,
    };
    ctx.setBytes(0, std.mem.asBytes(&p));
    ctx.setBuf(1, args.q);
    ctx.setBuf(2, args.k_cache);
    ctx.setBuf(3, args.v_cache);
    ctx.setBuf(4, args.out);
    dispatchGroups(ctx, args.n_tokens, args.n_q_heads, 1, DS5_ATTN_TG, 1, 1);
}

// ---------------------------------------------------------------------------
// routerTopK — CPU-only per PORTING-moe.md §1; see the module doc comment for
// why this is a parallel implementation rather than a call into
// kernels_c.zig, and how it forces the router-boundary batch flush.
// ---------------------------------------------------------------------------

/// Same dequant rule as kernels_c.zig's private `dotRow` (ADR-005 §1: Q8_0
/// value = f32(f16 scale) · i8 q, f16 promoted to f32 BEFORE the multiply).
/// Kept in lockstep with that function by inspection — see module doc.
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
            const n_blocks = x.len / 32;
            var b: usize = 0;
            while (b < n_blocks) : (b += 1) {
                const blk = row[b * 34 ..][0..34];
                const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk[0..2], .little))));
                const qs: []const i8 = @ptrCast(blk[2..34]);
                for (qs, x[b * 32 ..][0..32]) |q, xv| acc += (scale * @as(f32, @floatFromInt(q))) * xv;
            }
        },
        else => unreachable, // callers gate dtypes before dispatching here
    }
    return acc;
}

pub fn routerTopK(ctx: *Ctx, args: contracts.RouterArgs) KernelError!void {
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

    const x_host = ctx.alloc.alloc(f32, n_tokens * dim) catch return KernelError.OutOfMemory;
    defer ctx.alloc.free(x_host);
    const w_host = ctx.alloc.alloc(u8, n_experts * row_bytes) catch return KernelError.OutOfMemory;
    defer ctx.alloc.free(w_host);

    // Router boundary: download() auto-flushes the pending batch (metal.zig
    // T05 fix), so this is the point every op since the last flush actually
    // executes on the GPU. Record that batch's elapsed time as "this layer's"
    // GPU time (see the module doc comment on what the number covers).
    try ctx.download(args.x, 0, std.mem.sliceAsBytes(x_host));
    recordLayerNs(ctx.gpuElapsedNs());
    try ctx.download(args.w, 0, w_host);

    const probs = ctx.alloc.alloc(f32, n_experts) catch return KernelError.OutOfMemory;
    defer ctx.alloc.free(probs);
    const taken = ctx.alloc.alloc(bool, n_experts) catch return KernelError.OutOfMemory;
    defer ctx.alloc.free(taken);

    var t: usize = 0;
    while (t < n_tokens) : (t += 1) {
        const xr = x_host[t * dim ..][0..dim];

        // logits = x · Wᵀ, f32; softmax over ALL n_experts, f32.
        var max_logit: f32 = -std.math.inf(f32);
        for (0..n_experts) |e| {
            const logit = dotRow(xr, w_host[e * row_bytes ..][0..row_bytes], args.w_dtype);
            probs[e] = logit;
            max_logit = @max(max_logit, logit);
        }
        var sum: f32 = 0;
        for (probs) |*p| {
            p.* = @exp(p.* - max_logit);
            sum += p.*;
        }
        const inv_sum = 1.0 / sum;
        for (probs) |*p| p.* *= inv_sum;

        // top-k by probability descending; ties → LOWER expert id.
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
        if (args.norm_topk_prob) {
            const inv = 1.0 / picked_sum;
            for (args.out_weights[t * top_k ..][0..top_k]) |*ow| ow.* *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// expertMlpSwiglu — expert_mlp_swiglu_q8. Q8_0-only (the only shader that
// exists — PORTING-moe.md); other dtypes are UnsupportedDtype. `out` must
// already be pre-zeroed by the caller (frozen contract; Engine.forward()
// does this before calling).
// ---------------------------------------------------------------------------

pub fn expertMlpSwiglu(ctx: *Ctx, args: contracts.ExpertMlpArgs) KernelError!void {
    if (args.w_dtype != .q8_0) return KernelError.UnsupportedDtype;
    if (args.dim == 0 or args.ffn_dim == 0) return KernelError.ShapeMismatch;
    if (args.ffn_dim > DS5_MAX_FFN) return KernelError.ShapeMismatch; // static threadgroup array bound
    if (args.dim % 32 != 0 or args.ffn_dim % 32 != 0) return KernelError.ShapeMismatch;
    if (args.pairs.len == 0) return; // nothing to accumulate

    ensureBatch(ctx);
    // Transient per-dispatch buffer for the pair list (12 B/pair, small — a
    // handful of KB even at real-model top_k=8 batch sizes). v2 could reuse a
    // persistent upload buffer sized to max_batch*top_k instead of allocating
    // one MTLBuffer per layer; correctness-first for M2b bring-up.
    const pairs_buf = try ctx.bufferFromBytes(std.mem.sliceAsBytes(args.pairs));
    const pso = try ctx.pipeline("expert_mlp_swiglu_q8");
    ctx.setPipeline(pso);
    const p = ExpertMlpParams{
        .dim = args.dim,
        .ffn_dim = args.ffn_dim,
        .n_experts = args.n_experts,
        .n_pairs = @intCast(args.pairs.len),
        .tile_out = args.dim, // v1: one tile covers the whole output row (PORTING-moe.md)
    };
    ctx.setBytes(0, std.mem.asBytes(&p));
    ctx.setBuf(1, args.x);
    ctx.setBuf(2, args.gate);
    ctx.setBuf(3, args.up);
    ctx.setBuf(4, args.down);
    ctx.setBuf(5, pairs_buf);
    ctx.setBuf(6, args.out);
    dispatchGroups(ctx, @intCast(args.pairs.len), 1, 1, 256, 1, 1);
}

// ---------------------------------------------------------------------------
// Per-layer GPU timing capture, for `ds5 run --backend metal` run-metadata.
// Single global buffer: ds5 runs one Engine per process (CLI tool, not a
// server), so this is the simplest correct option, not a design that needs
// to survive concurrent Engines.
// ---------------------------------------------------------------------------

var layer_ns_buf: std.ArrayList(u64) = .empty;
var layer_ns_alloc: ?std.mem.Allocator = null;

/// Call once before a forward() pass whose per-layer GPU time you want to
/// collect (typically once per process, before the whole run).
pub fn beginTiming(alloc: std.mem.Allocator) void {
    if (layer_ns_alloc != null) layer_ns_buf.clearRetainingCapacity();
    layer_ns_alloc = alloc;
}

fn recordLayerNs(ns: u64) void {
    if (layer_ns_alloc) |a| layer_ns_buf.append(a, ns) catch {};
}

/// GPU ns of the batch that ended at each router sync since the last
/// beginTiming()/clearTiming() — index i is layer i's boundary flush. This
/// covers the tail of layer i-1's post-router work (experts + residual add)
/// PLUS layer i's own pre-router work (norm/proj/rope/attn), because nothing
/// forces a submit between "block" (end of a layer) and the next layer's
/// "attn_norm" (deliberately — that's the batching win). It is a real,
/// reproducible GPU-time signal, just not a perfectly isolated per-layer
/// slice; documented here and in the run-metadata JSON.
pub fn layerNs() []const u64 {
    return layer_ns_buf.items;
}

pub fn endTiming() void {
    if (layer_ns_alloc) |a| layer_ns_buf.deinit(a);
    layer_ns_buf = .empty;
    layer_ns_alloc = null;
}

// ---------------------------------------------------------------------------
// Frozen-signature check.
// ---------------------------------------------------------------------------

test "gpu provider satisfies the frozen kernel api" {
    comptime contracts.assertKernelApi(@This(), Ctx);
}

// ---------------------------------------------------------------------------
// Per-op fixture tests: same fixtures/tolerances the CPU kernels use
// (kernels_a/b/c.zig), dispatched through the real Metal shader path. Skips
// (not fails) if no Metal device is present or fixtures aren't generated.
// ---------------------------------------------------------------------------

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

fn makeGpuCtx(alloc: std.mem.Allocator) !*Ctx {
    const ctx = Ctx.init(alloc) catch |err| switch (err) {
        KernelError.DeviceFailure => return error.SkipZigTest, // no Metal device
        else => return err,
    };
    errdefer ctx.deinit();
    try loadShaders(ctx);
    return ctx;
}

fn skippableManifest(alloc: std.mem.Allocator) !std.json.Parsed(std.json.Value) {
    return fixture.loadManifest(alloc, FIXTURE_DIR) catch |err| switch (err) {
        error.OpenFailed => return error.SkipZigTest,
        else => return err,
    };
}

test "GPU rmsnorm fixtures: hidden-dim and per-head norms in tolerance" {
    const alloc = std.testing.allocator;
    var parsed = try skippableManifest(alloc);
    defer parsed.deinit();
    const ctx = try makeGpuCtx(alloc);
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

        const out = try ctx.createBuffer(@as(u64, n_rows) * dim * @sizeOf(f32));
        try rmsNorm(ctx, .{
            .x = try ctx.bufferFromBytes(input.data),
            .weight = try ctx.bufferFromBytes(weight.data),
            .out = out,
            .n_rows = n_rows,
            .dim = dim,
            .eps = eps,
        });
        try ctx.submit();

        const actual = try alloc.alloc(f32, n_rows * dim);
        defer alloc.free(actual);
        try ctx.download(out, 0, std.mem.sliceAsBytes(actual));

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle.asF32(), actual, atol, rtol);
        std.debug.print("GPU rmsnorm case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), actual, atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 6), n_cases);
}

test "GPU rope fixtures: NeoX rotation in place, in tolerance" {
    const alloc = std.testing.allocator;
    var parsed = try skippableManifest(alloc);
    defer parsed.deinit();
    const ctx = try makeGpuCtx(alloc);
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
        try ctx.submit();

        const actual = try alloc.alloc(f32, input.data.len / 4);
        defer alloc.free(actual);
        try ctx.download(x, 0, std.mem.sliceAsBytes(actual));

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle.asF32(), actual, atol, rtol);
        std.debug.print("GPU rope case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), actual, atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 3), n_cases);
}

test "GPU matmul_quant fixtures: q8_0 and f32 dequant matmul in tolerance" {
    const alloc = std.testing.allocator;
    var parsed = try skippableManifest(alloc);
    defer parsed.deinit();
    const ctx = try makeGpuCtx(alloc);
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
        try ctx.submit();

        const actual = try alloc.alloc(f32, @as(usize, m) * n);
        defer alloc.free(actual);
        try ctx.download(out, 0, std.mem.sliceAsBytes(actual));

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle.asF32(), actual, atol, rtol);
        std.debug.print("GPU matmul_quant case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), actual, atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 4), n_cases);
}

test "GPU attention fixtures: prefill + decode, all layers, in tolerance" {
    const alloc = std.testing.allocator;
    var parsed = try skippableManifest(alloc);
    defer parsed.deinit();
    const ctx = try makeGpuCtx(alloc);
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
            .out = out_buf,
            .pos = pos,
            .n_tokens = n_tokens,
            .n_q_heads = n_q_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_ctx = max_ctx,
            .scale = scale,
        });
        try ctx.submit();

        const actual = try alloc.alloc(f32, @as(usize, n_tokens) * n_q_heads * head_dim);
        defer alloc.free(actual);
        try ctx.download(out_buf, 0, std.mem.sliceAsBytes(actual));

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle_out.asF32(), actual, atol, rtol);
        std.debug.print("GPU attention case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle_out.asF32(), actual, atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 6), n_cases);
}

test "GPU router fixtures: expert ids exact, gate weights in tolerance" {
    const alloc = std.testing.allocator;
    var parsed = try skippableManifest(alloc);
    defer parsed.deinit();
    const ctx = try makeGpuCtx(alloc);
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
        const norm_topk_prob = params.get("norm_topk_prob").?.bool;
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
        const out_w = try alloc.alloc(f32, n_tokens * top_k);
        defer alloc.free(out_w);

        try routerTopK(ctx, .{
            .x = try ctx.bufferFromBytes(input.data),
            .w = try ctx.bufferFromBytes(weight.data),
            .w_dtype = .f32,
            .n_tokens = n_tokens,
            .dim = dim,
            .n_experts = n_experts,
            .top_k = top_k,
            .norm_topk_prob = norm_topk_prob,
            .out_ids = out_ids,
            .out_weights = out_w,
        });

        const oracle_ids_i32 = oracle_ids.asI32();
        var ids_exact: usize = 0;
        for (out_ids, oracle_ids_i32) |got, want| {
            if (@as(i32, got) == want) ids_exact += 1;
        }
        std.debug.print("GPU router case {s}: ids {d}/{d} exact\n", .{
            case.get("name").?.string, ids_exact, out_ids.len,
        });
        try std.testing.expectEqual(out_ids.len, ids_exact);

        const atol = jsonF32(tol.get("atol").?);
        try fixture.expectClose(oracle_w.asF32(), out_w, atol, 0);
    }
    try std.testing.expectEqual(@as(usize, 3), n_cases);
}

test "GPU expert_mlp fixtures: q8_0 fused SwiGLU in tolerance" {
    const alloc = std.testing.allocator;
    var parsed = try skippableManifest(alloc);
    defer parsed.deinit();
    const ctx = try makeGpuCtx(alloc);
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
        var expert_ids = try loadCaseTensor(alloc, tensors.get("expert_ids").?.string);
        defer expert_ids.free(alloc);
        var gate_weights = try loadCaseTensor(alloc, tensors.get("gate_weights").?.string);
        defer gate_weights.free(alloc);
        var oracle = try loadCaseTensor(alloc, tensors.get("output").?.string);
        defer oracle.free(alloc);

        const ids = expert_ids.asI32();
        const ws = gate_weights.asF32();
        const pairs = try alloc.alloc(contracts.PairDispatch, n_tokens * top_k);
        defer alloc.free(pairs);
        for (pairs, 0..) |*pr, j| {
            pr.* = .{
                .token = @intCast(j / top_k),
                .expert = @intCast(ids[j]),
                .weight = ws[j],
            };
        }

        const out = try ctx.createBuffer(@as(u64, n_tokens) * dim * @sizeOf(f32));
        try expertMlpSwiglu(ctx, .{
            .x = try ctx.bufferFromBytes(input.data),
            .gate = try ctx.bufferFromBytes(gate.data),
            .up = try ctx.bufferFromBytes(up.data),
            .down = try ctx.bufferFromBytes(down.data),
            .w_dtype = .q8_0,
            .pairs = pairs,
            .out = out,
            .n_tokens = n_tokens,
            .dim = dim,
            .ffn_dim = ffn_dim,
            .n_experts = n_experts,
        });
        try ctx.submit();

        const actual = try alloc.alloc(f32, @as(usize, n_tokens) * dim);
        defer alloc.free(actual);
        try ctx.download(out, 0, std.mem.sliceAsBytes(actual));

        const atol = jsonF32(tol.get("atol").?);
        const rtol = jsonF32(tol.get("rtol").?);
        const r = fixture.compare(oracle.asF32(), actual, atol, rtol);
        std.debug.print("GPU expert_mlp case {s}: max_abs_diff {e} max_rel_diff {e}\n", .{
            case.get("name").?.string, r.max_abs_diff, r.max_rel_diff,
        });
        try fixture.expectClose(oracle.asF32(), actual, atol, rtol);
    }
    try std.testing.expectEqual(@as(usize, 3), n_cases);
}

test "GPU kvAppend: round-trip scatter matches the CPU semantics, bit-exact" {
    const alloc = std.testing.allocator;
    const ctx = try makeGpuCtx(alloc);
    defer ctx.deinit();

    const n_kv_heads = 2;
    const head_dim = 4;
    const max_ctx = 8;
    const pos = 3;
    const n_tokens = 2;

    const cache_elems = n_kv_heads * max_ctx * head_dim;
    const k_cache = try ctx.createBuffer(cache_elems * @sizeOf(f32));
    const v_cache = try ctx.createBuffer(cache_elems * @sizeOf(f32));

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
        .pos = pos,
        .n_tokens = n_tokens,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .max_ctx = max_ctx,
    });
    try ctx.submit();

    var kc: [cache_elems]f32 = undefined;
    var vc: [cache_elems]f32 = undefined;
    try ctx.download(k_cache, 0, std.mem.sliceAsBytes(&kc));
    try ctx.download(v_cache, 0, std.mem.sliceAsBytes(&vc));

    for (0..n_kv_heads) |h| {
        for (pos..pos + n_tokens) |p| {
            for (0..head_dim) |d| {
                const ci = (h * max_ctx + p) * head_dim + d;
                const t = p - pos;
                const si = (t * n_kv_heads + h) * head_dim + d;
                try std.testing.expectEqual(k_new[si], kc[ci]);
                try std.testing.expectEqual(v_new[si], vc[ci]);
            }
        }
    }
}

test "GPU add: matches scalar loop" {
    const alloc = std.testing.allocator;
    const ctx = try makeGpuCtx(alloc);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    const n = 97;
    var xv: [n]f32 = undefined;
    var yv: [n]f32 = undefined;
    for (&xv) |*v| v.* = rand.floatNorm(f32);
    for (&yv) |*v| v.* = rand.floatNorm(f32);
    var want: [n]f32 = undefined;
    for (&want, xv, yv) |*w, a, b| w.* = a + b;

    const x = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&xv));
    const y = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&yv));
    const out = try ctx.createBuffer(n * @sizeOf(f32));
    try add(ctx, .{ .x = x, .y = y, .out = out, .n_elems = n });
    try ctx.submit();

    var got: [n]f32 = undefined;
    try ctx.download(out, 0, std.mem.sliceAsBytes(&got));
    try std.testing.expectEqualSlices(f32, &want, &got);
}

test "GPU batching: consecutive ops before routerTopK share one command buffer" {
    // Not a fixture test — a direct check on the batching claim: rmsNorm then
    // add without an intervening submit() should still leave cmdbuf open
    // (both dispatches landed in the same batch), and routerTopK's forced
    // flush should close it.
    const alloc = std.testing.allocator;
    const ctx = try makeGpuCtx(alloc);
    defer ctx.deinit();

    const dim = 4;
    const x = try ctx.createBuffer(dim * @sizeOf(f32));
    const w = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&[_]f32{ 1, 1, 1, 1 }));
    try ctx.upload(x, 0, std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3, 4 }));
    const h = try ctx.createBuffer(dim * @sizeOf(f32));

    try std.testing.expect(ctx.cmdbuf == null);
    try rmsNorm(ctx, .{ .x = x, .weight = w, .out = h, .n_rows = 1, .dim = dim, .eps = 1e-6 });
    try std.testing.expect(ctx.cmdbuf != null); // batch opened, not yet submitted
    try add(ctx, .{ .x = h, .y = h, .out = h, .n_elems = dim });
    try std.testing.expect(ctx.cmdbuf != null); // still the SAME open batch

    beginTiming(alloc);
    defer endTiming();
    const out_ids = try alloc.alloc(u16, 1);
    defer alloc.free(out_ids);
    const out_w = try alloc.alloc(f32, 1);
    defer alloc.free(out_w);
    try routerTopK(ctx, .{
        .x = h,
        .w = w,
        .w_dtype = .f32,
        .n_tokens = 1,
        .dim = dim,
        .n_experts = 1,
        .top_k = 1,
        .norm_topk_prob = false,
        .out_ids = out_ids,
        .out_weights = out_w,
    });
    try std.testing.expect(ctx.cmdbuf == null); // router boundary flushed it
    try std.testing.expectEqual(@as(usize, 1), layerNs().len);
}
