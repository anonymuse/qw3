//! DS5 frozen interface contracts (ADR-005).
//!
//! Parallel workstreams (GGUF parser, Metal glue, kernel sets A/B/C) build
//! against the types and signatures in this file. Changing ANYTHING here after
//! the freeze requires an orchestrator decision recorded in ADR-005 — a silent
//! edit that "makes my branch compile" is a contract violation.
//!
//! Contents:
//!   1. Dtype           — GGML/GGUF tensor dtypes + quant block geometry
//!   2. TensorDesc      — shape/layout descriptor (GGUF ne[] convention)
//!   3. ModelConfig     — Qwen3-MoE hyperparameters, sourced from artifact metadata
//!   4. Buf + kernel args — device-buffer views and per-op argument structs
//!   5. Interface asserts — comptime checks impl modules run in their tests
//!   6. Fixture format  — DS5T binary tensor header (manifest schema in ADR-005)

const std = @import("std");

// ---------------------------------------------------------------------------
// 1. Dtypes. Values are GGML type ids so the GGUF parser stores them verbatim.
// ---------------------------------------------------------------------------

pub const Dtype = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,

    /// Elements per quant block (1 for plain scalar types).
    pub fn blockElems(self: Dtype) u32 {
        return switch (self) {
            .f32, .f16, .bf16, .i8, .i16, .i32, .i64, .f64 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_nl => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .iq4_xs => 256,
        };
    }

    /// Bytes per quant block (element size for scalar types).
    pub fn blockBytes(self: Dtype) u32 {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .i8 => 1,
            .i64, .f64 => 8,
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34, // f16 scale + 32 × i8
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .iq2_xxs => 66,
            .iq2_xs => 74,
            .iq2_s => 82,
            .iq3_xxs => 98,
            .iq3_s => 110,
            .iq1_s => 50,
            .iq1_m => 56,
            .iq4_nl => 18,
            .iq4_xs => 136,
        };
    }

    pub fn isQuantized(self: Dtype) bool {
        return self.blockElems() > 1;
    }

    /// Bytes of one row of `elems` elements. Rows are block-aligned: quant
    /// blocks never straddle a row boundary, so `elems` must divide evenly.
    pub fn rowBytes(self: Dtype, elems: u64) u64 {
        std.debug.assert(elems % self.blockElems() == 0);
        return elems / self.blockElems() * self.blockBytes();
    }
};

// ---------------------------------------------------------------------------
// 2. Tensor descriptor. GGUF/GGML convention: ne[0] is the fastest-varying
// (contiguous, in-row) dimension. A 2-D weight of ne = {k, n} is n rows of k
// elements; matmul against it computes x[m,k] · Wᵀ = y[m,n]. Unused dims are 1.
// ---------------------------------------------------------------------------

pub const MAX_DIMS = 4;

pub const TensorDesc = struct {
    dtype: Dtype,
    n_dims: u32,
    ne: [MAX_DIMS]u64,

    pub fn init(dtype: Dtype, dims: []const u64) TensorDesc {
        std.debug.assert(dims.len >= 1 and dims.len <= MAX_DIMS);
        var d = TensorDesc{ .dtype = dtype, .n_dims = @intCast(dims.len), .ne = .{ 1, 1, 1, 1 } };
        for (dims, 0..) |n, i| d.ne[i] = n;
        return d;
    }

    pub fn elems(self: *const TensorDesc) u64 {
        return self.ne[0] * self.ne[1] * self.ne[2] * self.ne[3];
    }

    /// Rows = product of all dims above ne[0].
    pub fn rows(self: *const TensorDesc) u64 {
        return self.ne[1] * self.ne[2] * self.ne[3];
    }

    pub fn byteSize(self: *const TensorDesc) u64 {
        return self.dtype.rowBytes(self.ne[0]) * self.rows();
    }
};

/// A named tensor resolved inside a mapped GGUF file (or fixture). `data`
/// points into the mmap / owned buffer; lifetime is the owner's.
pub const TensorView = struct {
    name: []const u8,
    desc: TensorDesc,
    data: []const u8,
};

// ---------------------------------------------------------------------------
// 3. Model configuration. The engine ALWAYS reads these from artifact metadata
// (GGUF keys / HF config.json); hard-coded constants exist only for the
// synthetic test model, which we control. Expected real-model values are
// tabulated in ADR-005 and verified at parse time (assumption A-05/A-07).
// ---------------------------------------------------------------------------

pub const ModelConfig = struct {
    n_layers: u32,
    hidden_dim: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_experts: u32,
    top_k: u32,
    expert_ffn_dim: u32,
    vocab_size: u32,
    rms_eps: f32,
    rope_theta: f32,
    /// Qwen3 MoE renormalizes the top-k gate weights to sum to 1.
    norm_topk_prob: bool,
    max_ctx: u32,
};

/// The downsized synthetic model emitted by tools/make_fixtures.py --synthetic.
/// MUST stay byte-identical to SYNTH_CONFIG in that script.
pub const SYNTH_TINY = ModelConfig{
    .n_layers = 4,
    .hidden_dim = 256,
    .n_q_heads = 4,
    .n_kv_heads = 2,
    .head_dim = 64,
    .n_experts = 8,
    .top_k = 4,
    .expert_ffn_dim = 128,
    .vocab_size = 512,
    .rms_eps = 1e-6,
    .rope_theta = 1_000_000.0,
    .norm_topk_prob = true,
    .max_ctx = 512,
};

// ---------------------------------------------------------------------------
// 4. Kernel API. Frozen decisions (ADR-005):
//   - Activations are f32 device buffers for M2 bring-up; accumulation is f32.
//   - Weights stay in artifact dtype; kernels dequantize in-shader.
//   - Q8_0 dequant semantics: value = f32(f16 scale) * i8 q. Exactly that.
//   - Router runs with EXACT top-k semantics (ADR-001 rule 1): softmax over all
//     experts in f32, top-k by weight descending, ties broken by LOWER expert
//     id, then renormalize iff norm_topk_prob. Outputs land in host memory.
//   - A kernel provider module passes `assertKernelApi` in its own tests.
// ---------------------------------------------------------------------------

pub const KernelError = error{
    ShapeMismatch,
    UnsupportedDtype,
    DeviceFailure,
    OutOfMemory,
};

/// View into a device buffer. `handle` is the backend object (MTLBuffer* for
/// Metal, host pointer for the CPU reference impl); offset/len are in bytes.
pub const Buf = extern struct {
    handle: ?*anyopaque = null,
    offset: u64 = 0,
    len: u64 = 0,
};

/// y = x / rms(x) * weight, rowwise. Also used for Qwen3 per-head Q/K norm
/// (dim = head_dim, n_rows = n_tokens * n_heads). `out` may alias `x`.
pub const RmsNormArgs = struct {
    x: Buf, // f32 [n_rows, dim]
    weight: Buf, // f32 [dim]
    out: Buf, // f32 [n_rows, dim]
    n_rows: u32,
    dim: u32,
    eps: f32,
};

/// In-place NeoX-style rotation (rotate_half pairing: element i pairs with
/// i + head_dim/2) over the full head_dim. Matches HF Qwen3 / llama.cpp
/// GGML_ROPE_TYPE_NEOX. No frequency scaling in v1 (freq_scale stays 1.0).
pub const RopeArgs = struct {
    x: Buf, // f32 [n_tokens, n_heads, head_dim], rotated in place
    positions: Buf, // i32 [n_tokens] absolute positions
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    theta: f32,
    freq_scale: f32 = 1.0,
};

/// out[m,n] = x[m,k] · Wᵀ. W is a GGUF-layout weight: n rows of k elements in
/// `w_dtype` blocks (desc.ne = {k, n}). No bias — Qwen3 has none.
pub const MatmulArgs = struct {
    x: Buf, // f32 [m, k]
    w: Buf, // w_dtype, n rows × k cols
    w_dtype: Dtype,
    out: Buf, // f32 [m, n]
    m: u32,
    n: u32,
    k: u32,
};

/// Copy this step's rotated K and V into the cache at position `pos`.
/// Frozen KV cache layout: f32 [n_kv_heads, max_ctx, head_dim], K and V in
/// separate buffers, one pair per layer.
pub const KvAppendArgs = struct {
    k_new: Buf, // f32 [n_tokens, n_kv_heads, head_dim]
    v_new: Buf, // f32 [n_tokens, n_kv_heads, head_dim]
    k_cache: Buf,
    v_cache: Buf,
    pos: u32, // tokens already in cache
    n_tokens: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
};

/// Causal GQA attention over the cache. Query tokens occupy absolute positions
/// pos..pos+n_tokens-1; token t attends to cache positions 0..pos+t inclusive.
/// Softmax in f32.
pub const AttnArgs = struct {
    q: Buf, // f32 [n_tokens, n_q_heads, head_dim], post-rope post-qknorm
    k_cache: Buf,
    v_cache: Buf,
    out: Buf, // f32 [n_tokens, n_q_heads * head_dim]
    pos: u32,
    n_tokens: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    scale: f32, // 1/sqrt(head_dim)
};

/// Router with exact Qwen3 semantics. Never altered (ADR-001 rule 1):
/// logits = x·Wᵀ (f32); p = softmax over ALL n_experts; take top_k by p
/// descending, ties → lower expert id; iff norm_topk_prob divide the k weights
/// by their sum. Results are written to HOST slices — the scheduler and the
/// activation-packet path consume them from CPU memory.
pub const RouterArgs = struct {
    x: Buf, // f32 [n_tokens, dim]
    w: Buf, // router weight, n_experts rows × dim (f32/f16/q8_0 per quant matrix)
    w_dtype: Dtype,
    n_tokens: u32,
    dim: u32,
    n_experts: u32,
    top_k: u32,
    norm_topk_prob: bool,
    out_ids: []u16, // host, len n_tokens*top_k
    out_weights: []f32, // host, len n_tokens*top_k
};

/// One (token, expert) execution record. The scheduler builds the list of
/// pairs this node executes locally; layout is frozen for GPU upload.
pub const PairDispatch = extern struct {
    token: u32,
    expert: u32,
    weight: f32,
};

/// Fused SwiGLU expert MLP, accumulating:
///   out[token] += weight * down( silu(gate(x[token])) ⊙ up(x[token]) )
/// Caller zeroes `out` first. Expert banks are single GGUF 3-D tensors:
/// gate/up ne = {hidden_dim, ffn_dim, n_experts}, down ne = {ffn_dim,
/// hidden_dim, n_experts}; expert e lives at byte offset e*rowBytes(ne[0])*ne[1].
pub const ExpertMlpArgs = struct {
    x: Buf, // f32 [n_tokens, dim]
    gate: Buf,
    up: Buf,
    down: Buf,
    w_dtype: Dtype,
    pairs: []const PairDispatch, // host slice; impl uploads as needed
    out: Buf, // f32 [n_tokens, dim], pre-zeroed by caller
    n_tokens: u32,
    dim: u32,
    ffn_dim: u32,
    n_experts: u32,
};

/// out = x + y elementwise (residual add). `out` may alias either input.
pub const AddArgs = struct {
    x: Buf,
    y: Buf,
    out: Buf,
    n_elems: u64,
};

// ---------------------------------------------------------------------------
// 5. Comptime interface asserts. Impl modules prove conformance in their own
// tests:   comptime contracts.assertKernelApi(@This(), Ctx);
// ---------------------------------------------------------------------------

fn assertDecl(comptime M: type, comptime name: []const u8, comptime Sig: type) void {
    if (!@hasDecl(M, name))
        @compileError(@typeName(M) ++ " is missing frozen decl `" ++ name ++ "`");
    const Actual = @TypeOf(@field(M, name));
    if (Actual != Sig)
        @compileError("frozen signature mismatch for `" ++ name ++ "`: expected " ++
            @typeName(Sig) ++ ", got " ++ @typeName(Actual));
}

/// Kernel provider contract: module K exposes these free functions over the
/// glue context type Ctx. All eight ops; Q8_0 support is the M2 minimum for
/// matmul/expertMlpSwiglu, other dtypes may return UnsupportedDtype.
pub fn assertKernelApi(comptime K: type, comptime Ctx: type) void {
    assertDecl(K, "rmsNorm", fn (*Ctx, RmsNormArgs) KernelError!void);
    assertDecl(K, "rope", fn (*Ctx, RopeArgs) KernelError!void);
    assertDecl(K, "matmul", fn (*Ctx, MatmulArgs) KernelError!void);
    assertDecl(K, "kvAppend", fn (*Ctx, KvAppendArgs) KernelError!void);
    assertDecl(K, "gqaAttention", fn (*Ctx, AttnArgs) KernelError!void);
    assertDecl(K, "routerTopK", fn (*Ctx, RouterArgs) KernelError!void);
    assertDecl(K, "expertMlpSwiglu", fn (*Ctx, ExpertMlpArgs) KernelError!void);
    assertDecl(K, "add", fn (*Ctx, AddArgs) KernelError!void);
}

/// Metal glue contract: context lifecycle, buffer management, batch dispatch.
/// `bufferFromBytes` wraps page-aligned mmapped weight bytes without copying
/// (MTLBuffer newBufferWithBytesNoCopy) and may copy as a fallback.
/// begin/submit bracket one command batch; submit blocks until GPU completion;
/// gpuElapsedNs reports the last completed batch for the timing harness.
pub fn assertGpuApi(comptime Ctx: type) void {
    assertDecl(Ctx, "init", fn (std.mem.Allocator) KernelError!*Ctx);
    assertDecl(Ctx, "deinit", fn (*Ctx) void);
    assertDecl(Ctx, "createBuffer", fn (*Ctx, u64) KernelError!Buf);
    assertDecl(Ctx, "bufferFromBytes", fn (*Ctx, []const u8) KernelError!Buf);
    assertDecl(Ctx, "upload", fn (*Ctx, Buf, u64, []const u8) KernelError!void);
    assertDecl(Ctx, "download", fn (*Ctx, Buf, u64, []u8) KernelError!void);
    assertDecl(Ctx, "begin", fn (*Ctx) void);
    assertDecl(Ctx, "submit", fn (*Ctx) KernelError!void);
    assertDecl(Ctx, "gpuElapsedNs", fn (*Ctx) u64);
}

pub const GgufError = error{
    OpenFailed,
    MmapFailed,
    BadMagic,
    UnsupportedVersion,
    Truncated,
    BadMetadata,
    MissingKey,
    OutOfMemory,
};

/// GGUF parser contract: module G exposes a Model type. The file is mmapped;
/// TensorView.data points into the map for zero-copy hand-off to
/// bufferFromBytes. Metadata getters return null on missing key or wrong type.
/// config() derives ModelConfig from qwen3moe metadata keys (ADR-005 table).
pub fn assertGgufApi(comptime G: type) void {
    if (!@hasDecl(G, "Model")) @compileError("gguf module missing `Model` type");
    const M = G.Model;
    assertDecl(M, "open", fn (std.mem.Allocator, []const u8) GgufError!M);
    assertDecl(M, "deinit", fn (*M) void);
    assertDecl(M, "tensorCount", fn (*const M) usize);
    assertDecl(M, "tensorAt", fn (*const M, usize) TensorView);
    assertDecl(M, "tensorByName", fn (*const M, []const u8) ?TensorView);
    assertDecl(M, "metaU32", fn (*const M, []const u8) ?u32);
    assertDecl(M, "metaU64", fn (*const M, []const u8) ?u64);
    assertDecl(M, "metaF32", fn (*const M, []const u8) ?f32);
    assertDecl(M, "metaBool", fn (*const M, []const u8) ?bool);
    assertDecl(M, "metaStr", fn (*const M, []const u8) ?[]const u8);
    assertDecl(M, "config", fn (*const M) GgufError!ModelConfig);
}

// ---------------------------------------------------------------------------
// 6. Fixture tensor file format "DS5T". One tensor per file; all metadata that
// binds tensors into test cases (op, params, tolerances) lives in the fixture
// directory's manifest.json (schema frozen in ADR-005). Data starts at byte 64
// and is raw little-endian, row-major per the TensorDesc convention above.
// Comparison rule: pass iff |actual - oracle| <= atol + rtol*|oracle| for
// every element (plus op-specific exact checks, e.g. router expert ids).
// ---------------------------------------------------------------------------

pub const FIXTURE_MAGIC: u32 = 0x54355344; // "DS5T" little-endian
pub const FIXTURE_VERSION: u32 = 1;
pub const FIXTURE_DATA_OFFSET: usize = 64;

pub const FixtureHeader = extern struct {
    magic: u32,
    version: u32,
    dtype: u32, // Dtype / GGML type id
    n_dims: u32,
    ne: [MAX_DIMS]u64,
    data_bytes: u64,
    reserved: u64 = 0,

    pub fn init(desc: TensorDesc) FixtureHeader {
        return .{
            .magic = FIXTURE_MAGIC,
            .version = FIXTURE_VERSION,
            .dtype = @intFromEnum(desc.dtype),
            .n_dims = desc.n_dims,
            .ne = desc.ne,
            .data_bytes = desc.byteSize(),
        };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "quant block geometry matches ggml" {
    try std.testing.expectEqual(@as(u32, 34), Dtype.q8_0.blockBytes());
    try std.testing.expectEqual(@as(u32, 32), Dtype.q8_0.blockElems());
    try std.testing.expectEqual(@as(u32, 144), Dtype.q4_k.blockBytes());
    try std.testing.expectEqual(@as(u32, 256), Dtype.q4_k.blockElems());
    try std.testing.expectEqual(@as(u32, 66), Dtype.iq2_xxs.blockBytes());
    try std.testing.expectEqual(@as(u32, 74), Dtype.iq2_xs.blockBytes());
    try std.testing.expectEqual(@as(u32, 82), Dtype.iq2_s.blockBytes());
    try std.testing.expectEqual(@as(u32, 2), Dtype.f16.blockBytes());
    try std.testing.expect(Dtype.q8_0.isQuantized());
    try std.testing.expect(!Dtype.f32.isQuantized());
}

test "tensor desc sizes" {
    // Q8_0 weight 2048 cols × 512 rows: 2048/32*34 = 2176 bytes/row.
    const d = TensorDesc.init(.q8_0, &.{ 2048, 512 });
    try std.testing.expectEqual(@as(u64, 2176), Dtype.q8_0.rowBytes(2048));
    try std.testing.expectEqual(@as(u64, 2176 * 512), d.byteSize());
    try std.testing.expectEqual(@as(u64, 512), d.rows());

    // 3-D expert bank: {hidden, ffn, experts}
    const e = TensorDesc.init(.q8_0, &.{ 256, 128, 8 });
    try std.testing.expectEqual(@as(u64, 256 * 128 * 8), e.elems());
    try std.testing.expectEqual(@as(u64, (256 / 32 * 34) * 128 * 8), e.byteSize());
}

test "fixture header is exactly 64 bytes and round-trips" {
    try std.testing.expectEqual(FIXTURE_DATA_OFFSET, @sizeOf(FixtureHeader));
    const desc = TensorDesc.init(.f32, &.{ 64, 3 });
    var hdr = FixtureHeader.init(desc);
    const bytes = std.mem.asBytes(&hdr);
    var back: FixtureHeader = undefined;
    @memcpy(std.mem.asBytes(&back), bytes);
    try std.testing.expectEqual(FIXTURE_MAGIC, back.magic);
    try std.testing.expectEqual(@as(u64, 64 * 3 * 4), back.data_bytes);
    try std.testing.expectEqual(@as(u64, 64), back.ne[0]);
}

test "pair dispatch layout is frozen at 12 bytes" {
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(PairDispatch));
}

// A do-nothing CPU mock proving the frozen signatures are implementable and
// that the asserts compile. Kernel workstreams copy this shape.
const MockCtx = struct {
    fn init(alloc: std.mem.Allocator) KernelError!*MockCtx {
        return alloc.create(MockCtx) catch KernelError.OutOfMemory;
    }
    fn deinit(self: *MockCtx) void {
        _ = self;
    }
    fn createBuffer(self: *MockCtx, len: u64) KernelError!Buf {
        _ = self;
        return .{ .len = len };
    }
    fn bufferFromBytes(self: *MockCtx, bytes: []const u8) KernelError!Buf {
        _ = self;
        return .{ .handle = @constCast(bytes.ptr), .len = bytes.len };
    }
    fn upload(self: *MockCtx, buf: Buf, off: u64, bytes: []const u8) KernelError!void {
        _ = .{ self, buf, off, bytes };
    }
    fn download(self: *MockCtx, buf: Buf, off: u64, out: []u8) KernelError!void {
        _ = .{ self, buf, off, out };
    }
    fn begin(self: *MockCtx) void {
        _ = self;
    }
    fn submit(self: *MockCtx) KernelError!void {
        _ = self;
    }
    fn gpuElapsedNs(self: *MockCtx) u64 {
        _ = self;
        return 0;
    }
};

const mock_kernels = struct {
    fn rmsNorm(ctx: *MockCtx, args: RmsNormArgs) KernelError!void {
        _ = .{ ctx, args };
    }
    fn rope(ctx: *MockCtx, args: RopeArgs) KernelError!void {
        _ = .{ ctx, args };
    }
    fn matmul(ctx: *MockCtx, args: MatmulArgs) KernelError!void {
        _ = .{ ctx, args };
    }
    fn kvAppend(ctx: *MockCtx, args: KvAppendArgs) KernelError!void {
        _ = .{ ctx, args };
    }
    fn gqaAttention(ctx: *MockCtx, args: AttnArgs) KernelError!void {
        _ = .{ ctx, args };
    }
    fn routerTopK(ctx: *MockCtx, args: RouterArgs) KernelError!void {
        _ = .{ ctx, args };
    }
    fn expertMlpSwiglu(ctx: *MockCtx, args: ExpertMlpArgs) KernelError!void {
        _ = .{ ctx, args };
    }
    fn add(ctx: *MockCtx, args: AddArgs) KernelError!void {
        _ = .{ ctx, args };
    }
};

test "frozen interfaces are implementable" {
    comptime assertGpuApi(MockCtx);
    comptime assertKernelApi(mock_kernels, MockCtx);
    const ctx = try MockCtx.init(std.testing.allocator);
    defer std.testing.allocator.destroy(ctx);
    try mock_kernels.add(ctx, .{ .x = .{}, .y = .{}, .out = .{}, .n_elems = 0 });
}
