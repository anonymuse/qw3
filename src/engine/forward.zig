//! T04 — M2a: single-process CPU forward pass (ADR-005 §6 recipe).
//!
//! `Engine` is generic over (Ctx, kernel provider namespace) at comptime — no
//! runtime vtable — so the same wiring drives the CPU reference kernels today
//! and the Metal kernels in M2b. It owns per-layer KV caches and scratch
//! buffers and implements, per token batch:
//!
//!   x = embed[token]                      (row dequant, no scaling)
//!   for each layer i:
//!       h  = rmsNorm(x, attn_norm)
//!       q  = matmul(h, Wq); k = matmul(h, Wk); v = matmul(h, Wv)
//!       q  = rmsNorm per head (q_norm); k = rmsNorm per head (k_norm)
//!       rope(q, pos); rope(k, pos)
//!       kvAppend(k, v -> cache[i])
//!       a  = gqaAttention(q, cache[i])
//!       x  = x + matmul(a, Wo)
//!       h2 = rmsNorm(x, ffn_norm)
//!       ids, w = routerTopK(h2)
//!       x  = x + expertMlpSwiglu(h2, ids, w)
//!   final: logits = matmul(rmsNorm(x, output_norm), output.weight)
//!
//! `Weights` is a thin resolved-tensor table. `Weights.fromGguf` binds a
//! parsed GGUF model (the synthetic model.gguf today, the 30B artifact in
//! M2c) into backend buffers via any Ctx that satisfies the frozen GPU API.
//!
//! The trace hook fires after every op of every layer so M2c debugging can
//! report the FIRST divergent layer/op against per-layer fixture tensors.

const std = @import("std");
const contracts = @import("../shared/contracts.zig");
const gguf = @import("../gguf/gguf.zig");

const Buf = contracts.Buf;
const Dtype = contracts.Dtype;
const ModelConfig = contracts.ModelConfig;

pub const EngineError = contracts.KernelError || contracts.GgufError || error{
    MissingTensor,
    BadShape,
    BatchTooLarge,
    ContextOverflow,
    TokenOutOfRange,
};

/// A weight matrix bound to a backend buffer, kept in artifact dtype.
pub const W = struct {
    buf: Buf,
    dtype: Dtype,
};

/// Resolved weight table for one model. Backend-agnostic: buffers were
/// created by whichever Ctx was passed to `fromGguf`. The token embedding
/// stays as host bytes — the engine dequantizes rows on the CPU and uploads.
pub const Weights = struct {
    config: ModelConfig,
    /// Host bytes of token_embd.weight (row-major, ne = {hidden, vocab}).
    token_embd_data: []const u8,
    token_embd_dtype: Dtype,
    output_norm: Buf,
    output: W, // lm_head, ne = {hidden, vocab}
    layers: []Layer,
    alloc: std.mem.Allocator,

    pub const Layer = struct {
        attn_norm: Buf,
        wq: W,
        wk: W,
        wv: W,
        wo: W,
        q_norm: Buf,
        k_norm: Buf,
        ffn_norm: Buf,
        router: W, // ffn_gate_inp, ne = {hidden, n_experts}
        gate: W, // ffn_gate_exps, ne = {hidden, ffn, E}
        up: W, // ffn_up_exps,   ne = {hidden, ffn, E}
        down: W, // ffn_down_exps, ne = {ffn, hidden, E}
    };

    /// Bind a parsed GGUF model's tensors into Ctx buffers. The Model's mmap
    /// must outlive the returned Weights (buffers wrap it zero-copy on CPU).
    pub fn fromGguf(alloc: std.mem.Allocator, ctx: anytype, model: *const gguf.Model) EngineError!Weights {
        const cfg = try model.config();
        const layers = alloc.alloc(Layer, cfg.n_layers) catch return EngineError.OutOfMemory;
        errdefer alloc.free(layers);

        const embd = model.tensorByName("token_embd.weight") orelse return EngineError.MissingTensor;
        if (embd.desc.ne[0] != cfg.hidden_dim or embd.desc.ne[1] < cfg.vocab_size)
            return EngineError.BadShape;

        const self = Weights{
            .config = cfg,
            .token_embd_data = embd.data,
            .token_embd_dtype = embd.desc.dtype,
            .output_norm = try bindPlain(ctx, model, "output_norm.weight"),
            .output = try bind(ctx, model, "output.weight"),
            .layers = layers,
            .alloc = alloc,
        };

        var name_buf: [64]u8 = undefined;
        for (layers, 0..) |*l, i| {
            l.* = .{
                .attn_norm = try bindLayerPlain(ctx, model, &name_buf, i, "attn_norm"),
                .wq = try bindLayer(ctx, model, &name_buf, i, "attn_q"),
                .wk = try bindLayer(ctx, model, &name_buf, i, "attn_k"),
                .wv = try bindLayer(ctx, model, &name_buf, i, "attn_v"),
                .wo = try bindLayer(ctx, model, &name_buf, i, "attn_output"),
                .q_norm = try bindLayerPlain(ctx, model, &name_buf, i, "attn_q_norm"),
                .k_norm = try bindLayerPlain(ctx, model, &name_buf, i, "attn_k_norm"),
                .ffn_norm = try bindLayerPlain(ctx, model, &name_buf, i, "ffn_norm"),
                .router = try bindLayer(ctx, model, &name_buf, i, "ffn_gate_inp"),
                .gate = try bindLayer(ctx, model, &name_buf, i, "ffn_gate_exps"),
                .up = try bindLayer(ctx, model, &name_buf, i, "ffn_up_exps"),
                .down = try bindLayer(ctx, model, &name_buf, i, "ffn_down_exps"),
            };
        }
        return self;
    }

    pub fn deinit(self: *Weights) void {
        self.alloc.free(self.layers);
        self.* = undefined;
    }

    fn bind(ctx: anytype, model: *const gguf.Model, name: []const u8) EngineError!W {
        const view = model.tensorByName(name) orelse return EngineError.MissingTensor;
        return .{ .buf = try ctx.bufferFromBytes(view.data), .dtype = view.desc.dtype };
    }

    fn bindPlain(ctx: anytype, model: *const gguf.Model, name: []const u8) EngineError!Buf {
        const w = try bind(ctx, model, name);
        if (w.dtype != .f32) return EngineError.BadShape; // norms are always f32
        return w.buf;
    }

    fn layerName(buf: []u8, layer: usize, suffix: []const u8) []const u8 {
        return std.fmt.bufPrint(buf, "blk.{d}.{s}.weight", .{ layer, suffix }) catch unreachable;
    }

    fn bindLayer(ctx: anytype, model: *const gguf.Model, buf: []u8, layer: usize, suffix: []const u8) EngineError!W {
        return bind(ctx, model, layerName(buf, layer, suffix));
    }

    fn bindLayerPlain(ctx: anytype, model: *const gguf.Model, buf: []u8, layer: usize, suffix: []const u8) EngineError!Buf {
        return bindPlain(ctx, model, layerName(buf, layer, suffix));
    }
};

/// Per-op trace callback: fires after each op with the op's f32 output buffer.
/// `layer` is the layer index, or -1 for the pre-layer / final ops
/// ("embed", "output_norm", "logits"). `n_elems` f32 values are valid.
pub const TraceFn = *const fn (user: ?*anyopaque, layer: i32, op: []const u8, buf: Buf, n_elems: usize) void;

pub fn Engine(comptime Ctx: type, comptime K: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        ctx: *Ctx,
        w: *const Weights,
        cfg: ModelConfig,
        max_batch: u32,
        /// Tokens already in the KV caches (absolute position of next token).
        pos: u32 = 0,

        // Per-layer KV caches, frozen layout f32 [n_kv_heads, max_ctx, head_dim].
        k_cache: []Buf,
        v_cache: []Buf,

        // Scratch (device) buffers, sized for max_batch tokens.
        x: Buf, // residual stream [n, hidden]
        h: Buf, // norm output    [n, hidden]
        q: Buf, // [n, n_q_heads, head_dim]
        k: Buf, // [n, n_kv_heads, head_dim]
        v: Buf,
        attn: Buf, // [n, n_q_heads*head_dim]
        proj: Buf, // o_proj / moe output [n, hidden]
        h2: Buf, // ffn norm output [n, hidden]
        logits: Buf, // [n, vocab]
        positions: Buf, // i32 [n]

        // Host scratch.
        embed_row: []f32, // one dequantized embedding row
        zeros: []const u8, // for re-zeroing the MoE accumulator
        pos_host: []i32,
        router_ids: []u16,
        router_weights: []f32,
        pairs: []contracts.PairDispatch,

        // Optional trace hook (set before calling forward).
        trace_fn: ?TraceFn = null,
        trace_user: ?*anyopaque = null,

        pub fn init(alloc: std.mem.Allocator, ctx: *Ctx, w: *const Weights, max_batch: u32) EngineError!Self {
            const cfg = w.config;
            if (max_batch == 0) return EngineError.BadShape;
            const n: u64 = max_batch;
            const hidden: u64 = cfg.hidden_dim;
            const q_dim: u64 = @as(u64, cfg.n_q_heads) * cfg.head_dim;
            const kv_dim: u64 = @as(u64, cfg.n_kv_heads) * cfg.head_dim;
            const f = @sizeOf(f32);

            const k_cache = alloc.alloc(Buf, cfg.n_layers) catch return EngineError.OutOfMemory;
            errdefer alloc.free(k_cache);
            const v_cache = alloc.alloc(Buf, cfg.n_layers) catch return EngineError.OutOfMemory;
            errdefer alloc.free(v_cache);
            const cache_bytes = @as(u64, cfg.n_kv_heads) * cfg.max_ctx * cfg.head_dim * f;
            for (k_cache) |*b| b.* = try ctx.createBuffer(cache_bytes);
            for (v_cache) |*b| b.* = try ctx.createBuffer(cache_bytes);

            const zeros = alloc.alloc(u8, @intCast(n * hidden * f)) catch return EngineError.OutOfMemory;
            @memset(zeros, 0);

            return .{
                .alloc = alloc,
                .ctx = ctx,
                .w = w,
                .cfg = cfg,
                .max_batch = max_batch,
                .k_cache = k_cache,
                .v_cache = v_cache,
                .x = try ctx.createBuffer(n * hidden * f),
                .h = try ctx.createBuffer(n * hidden * f),
                .q = try ctx.createBuffer(n * q_dim * f),
                .k = try ctx.createBuffer(n * kv_dim * f),
                .v = try ctx.createBuffer(n * kv_dim * f),
                .attn = try ctx.createBuffer(n * q_dim * f),
                .proj = try ctx.createBuffer(n * hidden * f),
                .h2 = try ctx.createBuffer(n * hidden * f),
                .logits = try ctx.createBuffer(n * cfg.vocab_size * f),
                .positions = try ctx.createBuffer(n * @sizeOf(i32)),
                .embed_row = alloc.alloc(f32, cfg.hidden_dim) catch return EngineError.OutOfMemory,
                .zeros = zeros,
                .pos_host = alloc.alloc(i32, max_batch) catch return EngineError.OutOfMemory,
                .router_ids = alloc.alloc(u16, max_batch * cfg.top_k) catch return EngineError.OutOfMemory,
                .router_weights = alloc.alloc(f32, max_batch * cfg.top_k) catch return EngineError.OutOfMemory,
                .pairs = alloc.alloc(contracts.PairDispatch, max_batch * cfg.top_k) catch return EngineError.OutOfMemory,
            };
        }

        /// Frees host-side state. Device buffers are owned by the Ctx and are
        /// released with it (CpuCtx frees them in its deinit).
        pub fn deinit(self: *Self) void {
            self.alloc.free(self.k_cache);
            self.alloc.free(self.v_cache);
            self.alloc.free(self.embed_row);
            self.alloc.free(@constCast(self.zeros));
            self.alloc.free(self.pos_host);
            self.alloc.free(self.router_ids);
            self.alloc.free(self.router_weights);
            self.alloc.free(self.pairs);
            self.* = undefined;
        }

        /// Forget cached context (KV contents need not be cleared: attention
        /// only reads positions below `pos`).
        pub fn reset(self: *Self) void {
            self.pos = 0;
        }

        fn traceOp(self: *Self, layer: i32, op: []const u8, buf: Buf, n_elems: usize) void {
            if (self.trace_fn) |f| f(self.trace_user, layer, op, buf, n_elems);
        }

        /// Run one batch of tokens at positions pos..pos+n-1, appending to the
        /// KV caches and advancing `pos`. Writes logits for ALL n tokens
        /// ([n, vocab], f32) into `logits_out` if non-null.
        pub fn forward(self: *Self, tokens: []const u32, logits_out: ?[]f32) EngineError!void {
            const cfg = self.cfg;
            const n: u32 = @intCast(tokens.len);
            if (n == 0 or n > self.max_batch) return EngineError.BatchTooLarge;
            if (self.pos + n > cfg.max_ctx) return EngineError.ContextOverflow;
            const hidden = cfg.hidden_dim;
            const q_dim = cfg.n_q_heads * cfg.head_dim;
            const kv_dim = cfg.n_kv_heads * cfg.head_dim;
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

            // x = embed[token]: dequantize rows on the host, upload.
            for (tokens, 0..) |tok, i| {
                if (tok >= cfg.vocab_size) return EngineError.TokenOutOfRange;
                try self.embedRow(tok);
                try self.ctx.upload(self.x, @as(u64, @intCast(i)) * hidden * @sizeOf(f32), std.mem.sliceAsBytes(self.embed_row));
            }
            self.traceOp(-1, "embed", self.x, n * hidden);

            // Absolute positions for rope.
            for (self.pos_host[0..n], 0..) |*p, i| p.* = @intCast(self.pos + i);
            try self.ctx.upload(self.positions, 0, std.mem.sliceAsBytes(self.pos_host[0..n]));

            for (self.w.layers, 0..) |*l, li| {
                const layer: i32 = @intCast(li);

                // h = rmsNorm(x, attn_norm)
                try K.rmsNorm(self.ctx, .{
                    .x = self.x,
                    .weight = l.attn_norm,
                    .out = self.h,
                    .n_rows = n,
                    .dim = hidden,
                    .eps = cfg.rms_eps,
                });
                self.traceOp(layer, "attn_norm", self.h, n * hidden);

                // q/k/v projections
                try K.matmul(self.ctx, .{ .x = self.h, .w = l.wq.buf, .w_dtype = l.wq.dtype, .out = self.q, .m = n, .n = q_dim, .k = hidden });
                try K.matmul(self.ctx, .{ .x = self.h, .w = l.wk.buf, .w_dtype = l.wk.dtype, .out = self.k, .m = n, .n = kv_dim, .k = hidden });
                try K.matmul(self.ctx, .{ .x = self.h, .w = l.wv.buf, .w_dtype = l.wv.dtype, .out = self.v, .m = n, .n = kv_dim, .k = hidden });
                self.traceOp(layer, "q_proj", self.q, n * q_dim);
                self.traceOp(layer, "k_proj", self.k, n * kv_dim);
                self.traceOp(layer, "v_proj", self.v, n * kv_dim);

                // per-head q/k rmsnorm
                try K.rmsNorm(self.ctx, .{
                    .x = self.q,
                    .weight = l.q_norm,
                    .out = self.q,
                    .n_rows = n * cfg.n_q_heads,
                    .dim = cfg.head_dim,
                    .eps = cfg.rms_eps,
                });
                try K.rmsNorm(self.ctx, .{
                    .x = self.k,
                    .weight = l.k_norm,
                    .out = self.k,
                    .n_rows = n * cfg.n_kv_heads,
                    .dim = cfg.head_dim,
                    .eps = cfg.rms_eps,
                });
                self.traceOp(layer, "q_norm", self.q, n * q_dim);
                self.traceOp(layer, "k_norm", self.k, n * kv_dim);

                // rope(q, pos); rope(k, pos)
                try K.rope(self.ctx, .{
                    .x = self.q,
                    .positions = self.positions,
                    .n_tokens = n,
                    .n_heads = cfg.n_q_heads,
                    .head_dim = cfg.head_dim,
                    .theta = cfg.rope_theta,
                });
                try K.rope(self.ctx, .{
                    .x = self.k,
                    .positions = self.positions,
                    .n_tokens = n,
                    .n_heads = cfg.n_kv_heads,
                    .head_dim = cfg.head_dim,
                    .theta = cfg.rope_theta,
                });
                self.traceOp(layer, "rope_q", self.q, n * q_dim);
                self.traceOp(layer, "rope_k", self.k, n * kv_dim);

                // kvAppend(k, v -> cache[i])
                try K.kvAppend(self.ctx, .{
                    .k_new = self.k,
                    .v_new = self.v,
                    .k_cache = self.k_cache[li],
                    .v_cache = self.v_cache[li],
                    .pos = self.pos,
                    .n_tokens = n,
                    .n_kv_heads = cfg.n_kv_heads,
                    .head_dim = cfg.head_dim,
                    .max_ctx = cfg.max_ctx,
                });

                // a = gqaAttention(q, cache[i])
                try K.gqaAttention(self.ctx, .{
                    .q = self.q,
                    .k_cache = self.k_cache[li],
                    .v_cache = self.v_cache[li],
                    .out = self.attn,
                    .pos = self.pos,
                    .n_tokens = n,
                    .n_q_heads = cfg.n_q_heads,
                    .n_kv_heads = cfg.n_kv_heads,
                    .head_dim = cfg.head_dim,
                    .max_ctx = cfg.max_ctx,
                    .scale = scale,
                });
                self.traceOp(layer, "attn", self.attn, n * q_dim);

                // x = x + matmul(a, Wo)
                try K.matmul(self.ctx, .{ .x = self.attn, .w = l.wo.buf, .w_dtype = l.wo.dtype, .out = self.proj, .m = n, .n = hidden, .k = q_dim });
                try K.add(self.ctx, .{ .x = self.x, .y = self.proj, .out = self.x, .n_elems = @as(u64, n) * hidden });
                self.traceOp(layer, "attn_resid", self.x, n * hidden);

                // h2 = rmsNorm(x, ffn_norm)
                try K.rmsNorm(self.ctx, .{
                    .x = self.x,
                    .weight = l.ffn_norm,
                    .out = self.h2,
                    .n_rows = n,
                    .dim = hidden,
                    .eps = cfg.rms_eps,
                });
                self.traceOp(layer, "ffn_norm", self.h2, n * hidden);

                // ids, w = routerTopK(h2)  (host outputs)
                try K.routerTopK(self.ctx, .{
                    .x = self.h2,
                    .w = l.router.buf,
                    .w_dtype = l.router.dtype,
                    .n_tokens = n,
                    .dim = hidden,
                    .n_experts = cfg.n_experts,
                    .top_k = cfg.top_k,
                    .norm_topk_prob = cfg.norm_topk_prob,
                    .out_ids = self.router_ids[0 .. n * cfg.top_k],
                    .out_weights = self.router_weights[0 .. n * cfg.top_k],
                });

                // x = x + expertMlpSwiglu(h2, ids, w). `proj` doubles as the
                // MoE accumulator; the kernel contract requires it pre-zeroed.
                for (self.pairs[0 .. n * cfg.top_k], 0..) |*p, j| {
                    p.* = .{
                        .token = @intCast(j / cfg.top_k),
                        .expert = self.router_ids[j],
                        .weight = self.router_weights[j],
                    };
                }
                try self.ctx.upload(self.proj, 0, self.zeros[0 .. @as(usize, n) * hidden * @sizeOf(f32)]);
                try K.expertMlpSwiglu(self.ctx, .{
                    .x = self.h2,
                    .gate = l.gate.buf,
                    .up = l.up.buf,
                    .down = l.down.buf,
                    .w_dtype = l.gate.dtype,
                    .pairs = self.pairs[0 .. n * cfg.top_k],
                    .out = self.proj,
                    .n_tokens = n,
                    .dim = hidden,
                    .ffn_dim = cfg.expert_ffn_dim,
                    .n_experts = cfg.n_experts,
                });
                self.traceOp(layer, "experts", self.proj, n * hidden);
                try K.add(self.ctx, .{ .x = self.x, .y = self.proj, .out = self.x, .n_elems = @as(u64, n) * hidden });
                self.traceOp(layer, "block", self.x, n * hidden);
            }

            // final: logits = matmul(rmsNorm(x, output_norm), output.weight)
            try K.rmsNorm(self.ctx, .{
                .x = self.x,
                .weight = self.w.output_norm,
                .out = self.h,
                .n_rows = n,
                .dim = hidden,
                .eps = cfg.rms_eps,
            });
            self.traceOp(-1, "output_norm", self.h, n * hidden);
            try K.matmul(self.ctx, .{
                .x = self.h,
                .w = self.w.output.buf,
                .w_dtype = self.w.output.dtype,
                .out = self.logits,
                .m = n,
                .n = cfg.vocab_size,
                .k = hidden,
            });
            self.traceOp(-1, "logits", self.logits, n * cfg.vocab_size);

            self.pos += n;

            if (logits_out) |out| {
                if (out.len < @as(usize, n) * cfg.vocab_size) return EngineError.BadShape;
                try self.ctx.download(self.logits, 0, std.mem.sliceAsBytes(out[0 .. @as(usize, n) * cfg.vocab_size]));
            }
        }

        /// Dequantize embedding row `tok` into self.embed_row (no scaling).
        fn embedRow(self: *Self, tok: u32) EngineError!void {
            const hidden = self.cfg.hidden_dim;
            const row_bytes: usize = @intCast(self.w.token_embd_dtype.rowBytes(hidden));
            const row = self.w.token_embd_data[@as(usize, tok) * row_bytes ..][0..row_bytes];
            switch (self.w.token_embd_dtype) {
                .f32 => @memcpy(std.mem.sliceAsBytes(self.embed_row), row),
                .q8_0 => gguf.dequantRowQ8_0(row, self.embed_row),
                else => return EngineError.UnsupportedDtype,
            }
        }
    };
}

/// Greedy pick: index of the max logit; ties break to the LOWER token id
/// (strict `>` in an ascending scan), matching the oracle's np.argmax.
pub fn argmax(logits: []const f32) u32 {
    var best: usize = 0;
    for (logits[1..], 1..) |v, i| {
        if (v > logits[best]) best = i;
    }
    return @intCast(best);
}

test "argmax picks first max on ties" {
    try std.testing.expectEqual(@as(u32, 1), argmax(&.{ 0.0, 2.0, 2.0, -1.0 }));
    try std.testing.expectEqual(@as(u32, 0), argmax(&.{5.0}));
}
