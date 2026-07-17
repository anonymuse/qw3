//! Metal glue layer (W2): device/queue/pipeline setup, buffer management,
//! dispatch and GPU timing. Implements the frozen GPU API
//! (contracts.assertGpuApi) exactly; kernel workstreams (W3-W5) add their
//! .metal sources via `addLibrary` and drive dispatches with the
//! pipeline/setBuf/setBytes/dispatch1d helpers below.
//!
//! Metal is reached through the Objective-C runtime directly: a single typed
//! `msg` wrapper casts objc_msgSend to the exact C signature per call site
//! (mandatory on aarch64 — objc_msgSend has no stable prototype). Shaders are
//! compiled at RUNTIME from source strings (newLibraryWithSource) so `zig
//! build` needs no Xcode metal toolchain. No MLX/ggml/llama.cpp anywhere
//! (ADR-002); Apple frameworks are the OS and are fair game.
//!
//! Batch model (frozen contract): begin() opens one command buffer + compute
//! encoder; helpers encode any number of dispatches; submit() ends encoding,
//! commits, blocks until GPU completion; gpuElapsedNs() reports the last
//! completed batch from MTLCommandBuffer GPUStartTime/GPUEndTime.

const std = @import("std");
const contracts = @import("../shared/contracts.zig");
const fixture = @import("../shared/fixture.zig");
const sys = @import("../shared/sys.zig");

const KernelError = contracts.KernelError;
const Buf = contracts.Buf;

// ---------------------------------------------------------------------------
// Objective-C runtime + Metal C entry points. libobjc and the Metal/Foundation
// frameworks are linked by build.zig (and by the `zig test` invocation).
// ---------------------------------------------------------------------------

pub const id = ?*anyopaque;
const SEL = ?*anyopaque;

extern "c" fn objc_msgSend() void;
extern "c" fn sel_registerName(name: [*:0]const u8) SEL;
extern "c" fn objc_getClass(name: [*:0]const u8) id;
extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;
/// From Metal.framework. LANDMINE: for CLI processes (no AppKit) this returns
/// nil unless CoreGraphics is ALSO linked — hence the extra framework flag.
extern "c" fn MTLCreateSystemDefaultDevice() id;

/// MTLSize. 24-byte extern struct; passed through objc_msgSend by the normal
/// AAPCS64 aggregate rules (Zig's extern-struct C ABI matches clang's — the
/// proof-kernel test exercises multi-threadgroup grids to prove it).
pub const MTLSize = extern struct { width: usize, height: usize, depth: usize };

/// MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache == 0:
/// unified-memory shared storage, the only mode DS5 uses (Apple Silicon).
const storage_mode_shared: usize = 0;
/// MTLCommandBufferStatus.completed
const status_completed: usize = 4;

fn sel(name: [*:0]const u8) SEL {
    return sel_registerName(name);
}

/// Typed objc_msgSend: builds the exact fn-pointer type from the argument
/// tuple and calls through it. Every argument must already have its final
/// C-ABI type (usize, f32, pointers, extern structs) — no comptime_int.
fn msg(comptime Ret: type, target: id, s: SEL, args: anytype) Ret {
    const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
    const F = switch (fields.len) {
        0 => fn (id, SEL) callconv(.c) Ret,
        1 => fn (id, SEL, fields[0].type) callconv(.c) Ret,
        2 => fn (id, SEL, fields[0].type, fields[1].type) callconv(.c) Ret,
        3 => fn (id, SEL, fields[0].type, fields[1].type, fields[2].type) callconv(.c) Ret,
        4 => fn (id, SEL, fields[0].type, fields[1].type, fields[2].type, fields[3].type) callconv(.c) Ret,
        else => @compileError("msg: unsupported argument count"),
    };
    const f: *const F = @ptrCast(&objc_msgSend);
    return @call(.auto, f, .{ target, s } ++ args);
}

/// Autoreleased NSString from a C string (valid until the enclosing pool pops).
fn nsString(str: [*:0]const u8) id {
    return msg(id, objc_getClass("NSString"), sel("stringWithUTF8String:"), .{str});
}

/// Print an NSError's localizedDescription (diagnostics only).
fn printNsError(what: []const u8, err: id) void {
    if (err == null) {
        std.debug.print("metal: {s} failed (no NSError)\n", .{what});
        return;
    }
    const desc = msg(id, err, sel("localizedDescription"), .{});
    const cstr = msg(?[*:0]const u8, desc, sel("UTF8String"), .{});
    std.debug.print("metal: {s} failed: {s}\n", .{ what, cstr orelse "?" });
}

fn release(obj: id) void {
    if (obj != null) msg(void, obj, sel("release"), .{});
}

/// Embedded proof-kernel source, compiled at Ctx.init.
const proof_src: [:0]const u8 = @embedFile("../kernels/shaders/proof.metal");

// ---------------------------------------------------------------------------
// Ctx — the frozen GPU API (contracts.assertGpuApi) plus dispatch helpers.
// ---------------------------------------------------------------------------

pub const Ctx = struct {
    alloc: std.mem.Allocator,
    device: id,
    queue: id,
    /// Compiled MTLLibrary objects; pipeline() searches newest-first.
    libraries: std.ArrayList(id),
    /// Pipeline cache by function name. Linear scan — a model uses O(10)
    /// distinct pipelines and lookups happen per dispatch, not per element.
    pipelines: std.ArrayList(PipelineEntry),
    /// Every MTLBuffer this Ctx created; released in deinit.
    buffers: std.ArrayList(id),
    /// One reusable shared upload buffer for short-lived dispatch metadata.
    /// It is separate from `buffers` so growing it replaces/releases the old
    /// resource instead of retaining every historical capacity until deinit.
    reusable_upload: Buf = .{},
    /// Bounded host scratch for synchronized CPU-side provider work. The GPU
    /// provider is single-engine/single-threaded, so one slice of each type is
    /// sufficient and retains only the largest capacity requested.
    host_f32_scratch: std.ArrayList(f32),
    host_bool_scratch: std.ArrayList(bool),
    /// In-flight batch state (begin()..submit()).
    pool: ?*anyopaque = null,
    cmdbuf: id = null,
    encoder: id = null,
    last_elapsed_ns: u64 = 0,

    const PipelineEntry = struct { name: []u8, pso: id };

    pub fn init(alloc: std.mem.Allocator) KernelError!*Ctx {
        const self = alloc.create(Ctx) catch return KernelError.OutOfMemory;
        errdefer alloc.destroy(self);
        const device = MTLCreateSystemDefaultDevice();
        if (device == null) return KernelError.DeviceFailure;
        const queue = msg(id, device, sel("newCommandQueue"), .{});
        if (queue == null) {
            release(device);
            return KernelError.DeviceFailure;
        }
        self.* = .{
            .alloc = alloc,
            .device = device,
            .queue = queue,
            .libraries = .empty,
            .pipelines = .empty,
            .buffers = .empty,
            .host_f32_scratch = .empty,
            .host_bool_scratch = .empty,
        };
        errdefer {
            release(queue);
            release(device);
            self.libraries.deinit(alloc);
        }
        try self.addLibrary(proof_src);
        return self;
    }

    pub fn deinit(self: *Ctx) void {
        std.debug.assert(self.cmdbuf == null); // no batch in flight
        release(self.reusable_upload.handle);
        self.host_f32_scratch.deinit(self.alloc);
        self.host_bool_scratch.deinit(self.alloc);
        for (self.buffers.items) |b| release(b);
        self.buffers.deinit(self.alloc);
        for (self.pipelines.items) |p| {
            release(p.pso);
            self.alloc.free(p.name);
        }
        self.pipelines.deinit(self.alloc);
        for (self.libraries.items) |l| release(l);
        self.libraries.deinit(self.alloc);
        release(self.queue);
        release(self.device);
        self.alloc.destroy(self);
    }

    // -- shader/pipeline management ---------------------------------------

    /// Compile a Metal source string into a library. Kernel workstreams call
    /// this once with their @embedFile'd .metal source; every kernel function
    /// in it then resolves through pipeline().
    pub fn addLibrary(self: *Ctx, source: [:0]const u8) KernelError!void {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var err: id = null;
        const lib = msg(id, self.device, sel("newLibraryWithSource:options:error:"), .{
            nsString(source.ptr), @as(id, null), @as(*id, &err),
        });
        if (lib == null) {
            printNsError("newLibraryWithSource", err);
            return KernelError.DeviceFailure;
        }
        self.libraries.append(self.alloc, lib) catch {
            release(lib);
            return KernelError.OutOfMemory;
        };
    }

    /// Get-or-create the compute pipeline for a kernel function name.
    pub fn pipeline(self: *Ctx, name: [:0]const u8) KernelError!id {
        for (self.pipelines.items) |p| {
            if (std.mem.eql(u8, p.name, name)) return p.pso;
        }
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var i = self.libraries.items.len;
        while (i > 0) {
            i -= 1;
            const func = msg(id, self.libraries.items[i], sel("newFunctionWithName:"), .{nsString(name.ptr)});
            if (func == null) continue;
            defer release(func);
            var err: id = null;
            const pso = msg(id, self.device, sel("newComputePipelineStateWithFunction:error:"), .{
                func, @as(*id, &err),
            });
            if (pso == null) {
                printNsError("newComputePipelineStateWithFunction", err);
                return KernelError.DeviceFailure;
            }
            const owned = self.alloc.dupe(u8, name) catch {
                release(pso);
                return KernelError.OutOfMemory;
            };
            self.pipelines.append(self.alloc, .{ .name = owned, .pso = pso }) catch {
                release(pso);
                self.alloc.free(owned);
                return KernelError.OutOfMemory;
            };
            return pso;
        }
        std.debug.print("metal: no kernel function named `{s}` in any library\n", .{name});
        return KernelError.DeviceFailure;
    }

    // -- buffer management (frozen API) ------------------------------------

    fn newSharedBuffer(self: *Ctx, len: u64) KernelError!Buf {
        if (len == 0) return .{};
        const buf = msg(id, self.device, sel("newBufferWithLength:options:"), .{
            @as(usize, @intCast(len)), storage_mode_shared,
        });
        if (buf == null) return KernelError.OutOfMemory;
        // newBufferWithLength contents are undefined; zero for parity with
        // the CPU reference ctx (and ExpertMlp's pre-zeroed accumulators).
        @memset(contentsSlice(buf, 0, len), 0);
        return .{ .handle = buf, .offset = 0, .len = len };
    }

    pub fn createBuffer(self: *Ctx, len: u64) KernelError!Buf {
        const result = try self.newSharedBuffer(len);
        if (result.handle == null) return result;
        const buf = result.handle;
        self.buffers.append(self.alloc, buf) catch {
            release(buf);
            return KernelError.OutOfMemory;
        };
        return result;
    }

    /// Upload dispatch metadata into one context-owned shared buffer. Reuse is
    /// safe only between command buffers: changing shared bytes while an
    /// encoded dispatch can still read them would race/corrupt that dispatch.
    /// The current one-CLI-engine flow naturally calls this immediately after
    /// its router synchronization. Capacity grows to the largest request while
    /// the retained Metal resource count remains exactly one.
    pub fn reusableUpload(self: *Ctx, bytes: []const u8) KernelError!Buf {
        if (bytes.len == 0) return .{};
        if (self.cmdbuf != null) return KernelError.DeviceFailure;
        if (bytes.len > self.reusable_upload.len) {
            const replacement = try self.newSharedBuffer(bytes.len);
            release(self.reusable_upload.handle);
            self.reusable_upload = replacement;
        }
        @memcpy(contentsSlice(self.reusable_upload.handle, 0, bytes.len), bytes);
        return .{ .handle = self.reusable_upload.handle, .offset = 0, .len = bytes.len };
    }

    /// Wrap caller-owned bytes. Page-aligned pointer + page-multiple length
    /// (e.g. mmapped GGUF tensor data) goes zero-copy via
    /// newBufferWithBytesNoCopy; anything else is copied. The caller keeps
    /// zero-copy memory alive/mapped for the Buf's lifetime.
    pub fn bufferFromBytes(self: *Ctx, bytes: []const u8) KernelError!Buf {
        if (bytes.len == 0) return .{};
        const page = std.heap.pageSize();
        const aligned = @intFromPtr(bytes.ptr) % page == 0 and bytes.len % page == 0;
        var buf: id = null;
        if (aligned) {
            buf = msg(id, self.device, sel("newBufferWithBytesNoCopy:length:options:deallocator:"), .{
                @as(*anyopaque, @constCast(bytes.ptr)), bytes.len, storage_mode_shared,
                @as(?*anyopaque, null), // no deallocator block: caller owns
            });
        }
        if (buf == null) {
            buf = msg(id, self.device, sel("newBufferWithBytes:length:options:"), .{
                @as(*const anyopaque, bytes.ptr), bytes.len, storage_mode_shared,
            });
        }
        if (buf == null) return KernelError.OutOfMemory;
        self.buffers.append(self.alloc, buf) catch {
            release(buf);
            return KernelError.OutOfMemory;
        };
        return .{ .handle = buf, .offset = 0, .len = bytes.len };
    }

    pub fn upload(self: *Ctx, buf: Buf, offset: u64, bytes: []const u8) KernelError!void {
        _ = self;
        @memcpy(contentsSlice(buf.handle, buf.offset + offset, bytes.len), bytes);
    }

    /// Complete a pending batch before direct unified-memory host access.
    pub fn synchronizeForHost(self: *Ctx) KernelError!void {
        if (self.cmdbuf != null) try self.submit();
    }

    /// T05: auto-flush a pending batch before reading. Any caller that has
    /// been encoding dispatches without an explicit submit() (the layer-
    /// batching pattern kernel providers use) still gets a coherent read —
    /// download() is the one frozen-API entry point host code uses to pull
    /// results back, so it is the correct place to guarantee freshness rather
    /// than pushing this requirement onto every caller. No-op (and no extra
    /// cost) when no batch is open, so existing begin()/.../submit()/download()
    /// call sites are unaffected.
    pub fn download(self: *Ctx, buf: Buf, offset: u64, out: []u8) KernelError!void {
        try self.synchronizeForHost();
        @memcpy(out, contentsSlice(buf.handle, buf.offset + offset, out.len));
    }

    /// CPU-visible bytes of a Buf region (unified memory). Valid only while
    /// no in-flight GPU work touches the buffer.
    pub fn hostBytes(self: *Ctx, buf: Buf, offset: u64, len: u64) []u8 {
        std.debug.assert(self.cmdbuf == null);
        return contentsSlice(buf.handle, buf.offset + offset, len);
    }

    pub fn hostF32Scratch(self: *Ctx, len: usize) KernelError![]f32 {
        self.host_f32_scratch.resize(self.alloc, len) catch return KernelError.OutOfMemory;
        return self.host_f32_scratch.items;
    }

    pub fn hostBoolScratch(self: *Ctx, len: usize) KernelError![]bool {
        self.host_bool_scratch.resize(self.alloc, len) catch return KernelError.OutOfMemory;
        return self.host_bool_scratch.items;
    }

    /// Diagnostics only: includes the reusable upload resource when allocated.
    pub fn diagnosticBufferCount(self: *const Ctx) usize {
        return self.buffers.items.len + @intFromBool(self.reusable_upload.handle != null);
    }

    fn contentsSlice(handle: id, offset: u64, len: u64) []u8 {
        const base = msg(?[*]u8, handle, sel("contents"), .{});
        return (base.? + offset)[0..@intCast(len)];
    }

    // -- batch encode/submit (frozen API) -----------------------------------

    pub fn begin(self: *Ctx) void {
        std.debug.assert(self.cmdbuf == null);
        // Pool scopes the autoreleased command buffer + encoder; popped in
        // submit() after GPU timing is read.
        self.pool = objc_autoreleasePoolPush();
        self.cmdbuf = msg(id, self.queue, sel("commandBuffer"), .{});
        self.encoder = msg(id, self.cmdbuf, sel("computeCommandEncoder"), .{});
    }

    pub fn submit(self: *Ctx) KernelError!void {
        defer {
            self.cmdbuf = null;
            self.encoder = null;
            if (self.pool) |p| objc_autoreleasePoolPop(p);
            self.pool = null;
        }
        const cb = self.cmdbuf orelse return KernelError.DeviceFailure;
        if (self.encoder == null) return KernelError.DeviceFailure;
        msg(void, self.encoder, sel("endEncoding"), .{});
        msg(void, cb, sel("commit"), .{});
        msg(void, cb, sel("waitUntilCompleted"), .{});
        const start = msg(f64, cb, sel("GPUStartTime"), .{});
        const end = msg(f64, cb, sel("GPUEndTime"), .{});
        self.last_elapsed_ns = if (end > start) @intFromFloat((end - start) * 1e9) else 0;
        const status = msg(usize, cb, sel("status"), .{});
        if (status != status_completed) {
            printNsError("command buffer", msg(id, cb, sel("error"), .{}));
            return KernelError.DeviceFailure;
        }
    }

    pub fn gpuElapsedNs(self: *Ctx) u64 {
        return self.last_elapsed_ns;
    }

    // -- dispatch helpers (between begin() and submit()) ---------------------

    pub fn setPipeline(self: *Ctx, pso: id) void {
        msg(void, self.encoder, sel("setComputePipelineState:"), .{pso});
    }

    /// Bind a device buffer at a [[buffer(index)]] slot, honoring Buf.offset.
    pub fn setBuf(self: *Ctx, index: u32, buf: Buf) void {
        msg(void, self.encoder, sel("setBuffer:offset:atIndex:"), .{
            buf.handle, @as(usize, @intCast(buf.offset)), @as(usize, index),
        });
    }

    /// Bind a small uniform (kernel arg struct) by value — no buffer needed.
    /// Pass std.mem.asBytes(&args_struct); the MSL side declares a matching
    /// `constant T&`. Metal limit: 4KB.
    pub fn setBytes(self: *Ctx, index: u32, bytes: []const u8) void {
        msg(void, self.encoder, sel("setBytes:length:atIndex:"), .{
            @as(*const anyopaque, bytes.ptr), bytes.len, @as(usize, index),
        });
    }

    pub fn dispatch(self: *Ctx, groups: MTLSize, threads_per_group: MTLSize) void {
        msg(void, self.encoder, sel("dispatchThreadgroups:threadsPerThreadgroup:"), .{
            groups, threads_per_group,
        });
    }

    /// 1-D grid covering n_threads with `width`-wide threadgroups. Kernels
    /// bound-check `thread_position_in_grid` against their element count.
    pub fn dispatch1d(self: *Ctx, n_threads: u64, width: u32) void {
        const groups = (n_threads + width - 1) / width;
        self.dispatch(
            .{ .width = @intCast(groups), .height = 1, .depth = 1 },
            .{ .width = width, .height = 1, .depth = 1 },
        );
    }
};

// ---------------------------------------------------------------------------
// Proof kernels: exercise the full glue path. `add` has the exact frozen
// kernel-provider signature (contracts.assertKernelApi's `add` entry) so
// W3-W5 can lift this shape verbatim.
// ---------------------------------------------------------------------------

const ProofScaleAddParams = extern struct { a: f32, n: u32 };

/// Encode out[i] = a*x[i] + y[i] into the current batch.
pub fn proofScaleAdd(ctx: *Ctx, a: f32, x: Buf, y: Buf, out: Buf, n_elems: u64) KernelError!void {
    const pso = try ctx.pipeline("proof_scale_add");
    ctx.setPipeline(pso);
    ctx.setBuf(0, x);
    ctx.setBuf(1, y);
    ctx.setBuf(2, out);
    const params = ProofScaleAddParams{ .a = a, .n = @intCast(n_elems) };
    ctx.setBytes(3, std.mem.asBytes(&params));
    ctx.dispatch1d(n_elems, 256);
}

/// Frozen `add` semantics (out = x + y); signature matches assertKernelApi.
pub fn add(ctx: *Ctx, args: contracts.AddArgs) KernelError!void {
    const pso = try ctx.pipeline("proof_add");
    ctx.setPipeline(pso);
    ctx.setBuf(0, args.x);
    ctx.setBuf(1, args.y);
    ctx.setBuf(2, args.out);
    const n: u32 = @intCast(args.n_elems);
    ctx.setBytes(3, std.mem.asBytes(&n));
    ctx.dispatch1d(args.n_elems, 256);
}

// ---------------------------------------------------------------------------
// Tests. Run from the repo root so tests/fixtures/ paths resolve:
//   zig test src/test_metal.zig -lobjc -framework Metal -framework Foundation \
//       -framework CoreGraphics
// or: zig build test-metal
// ---------------------------------------------------------------------------

test "metal ctx satisfies the frozen gpu api" {
    comptime contracts.assertGpuApi(Ctx);
}

test "add matches the frozen kernel-provider signature" {
    comptime std.debug.assert(@TypeOf(add) == fn (*Ctx, contracts.AddArgs) KernelError!void);
}

test "buffer create/upload/download round-trip" {
    const ctx = try Ctx.init(std.testing.allocator);
    defer ctx.deinit();

    const b = try ctx.createBuffer(16);
    try std.testing.expectEqual(@as(u64, 16), b.len);
    // createBuffer zero-fills.
    var zeros: [16]u8 = undefined;
    try ctx.download(b, 0, &zeros);
    try std.testing.expectEqualSlices(u8, &(.{0} ** 16), &zeros);

    try ctx.upload(b, 4, &.{ 1, 2, 3, 4 });
    var out: [4]u8 = undefined;
    try ctx.download(b, 4, &out);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);
}

test "bufferFromBytes: copy fallback and zero-copy page-aligned path" {
    // No-copy backing memory must outlive the Ctx (the MTLBuffer references
    // it until release in deinit), so allocate it first — defers run LIFO.
    const page = std.heap.pageSize();
    const mem = try std.heap.page_allocator.alloc(u8, page);
    defer std.heap.page_allocator.free(mem);
    @memset(mem, 0xAB);

    const ctx = try Ctx.init(std.testing.allocator);
    defer ctx.deinit();

    // Unaligned small slice → copy path.
    const vals = [_]f32{ 1.5, -2.0, 3.25 };
    const small = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&vals));
    var back: [12]u8 = undefined;
    try ctx.download(small, 0, &back);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&vals), &back);

    // Page-aligned, page-multiple → newBufferWithBytesNoCopy must wrap the
    // SAME memory (contents pointer == source pointer), like mmapped GGUF.
    const big = try ctx.bufferFromBytes(mem);
    const host = ctx.hostBytes(big, 0, big.len);
    try std.testing.expectEqual(@intFromPtr(mem.ptr), @intFromPtr(host.ptr));
    try std.testing.expectEqual(@as(u8, 0xAB), host[page - 1]);
}

test "unknown kernel name is a DeviceFailure, not a crash" {
    const ctx = try Ctx.init(std.testing.allocator);
    defer ctx.deinit();
    try std.testing.expectError(KernelError.DeviceFailure, ctx.pipeline("no_such_kernel"));
}

/// Load two same-shaped f32 fixture tensors as GPU inputs.
fn loadXY(alloc: std.mem.Allocator) !struct { x: fixture.OwnedTensor, y: fixture.OwnedTensor } {
    const x = try fixture.loadTensor(alloc, "tests/fixtures/synthetic/l0_attn_norm.input.ds5t");
    errdefer {
        var xm = x;
        xm.free(alloc);
    }
    const y = try fixture.loadTensor(alloc, "tests/fixtures/synthetic/l0_attn_norm.output.ds5t");
    return .{ .x = x, .y = y };
}

test "proof_scale_add matches CPU oracle on committed fixture tensors" {
    const alloc = std.testing.allocator;
    var t = try loadXY(alloc);
    defer t.x.free(alloc);
    defer t.y.free(alloc);
    const xf = t.x.asF32();
    const yf = t.y.asF32();
    try std.testing.expectEqual(xf.len, yf.len);
    try std.testing.expect(xf.len > 256); // multi-threadgroup grid (MTLSize ABI proof)

    const a: f32 = 1.75;
    const expected = try alloc.alloc(f32, xf.len);
    defer alloc.free(expected);
    for (expected, xf, yf) |*e, xv, yv| e.* = a * xv + yv;

    const ctx = try Ctx.init(alloc);
    defer ctx.deinit();
    const xb = try ctx.bufferFromBytes(t.x.data);
    const yb = try ctx.bufferFromBytes(t.y.data);
    const ob = try ctx.createBuffer(xf.len * 4);

    ctx.begin();
    try proofScaleAdd(ctx, a, xb, yb, ob, xf.len);
    try ctx.submit();

    const actual = try alloc.alloc(f32, xf.len);
    defer alloc.free(actual);
    try ctx.download(ob, 0, std.mem.sliceAsBytes(actual));
    try fixture.expectClose(expected, actual, 1e-6, 0);
}

test "frozen add semantics (out = x + y) on fixture tensors" {
    const alloc = std.testing.allocator;
    var t = try loadXY(alloc);
    defer t.x.free(alloc);
    defer t.y.free(alloc);
    const xf = t.x.asF32();
    const yf = t.y.asF32();

    const expected = try alloc.alloc(f32, xf.len);
    defer alloc.free(expected);
    for (expected, xf, yf) |*e, xv, yv| e.* = xv + yv;

    const ctx = try Ctx.init(alloc);
    defer ctx.deinit();
    const xb = try ctx.bufferFromBytes(t.x.data);
    const yb = try ctx.bufferFromBytes(t.y.data);
    const ob = try ctx.createBuffer(xf.len * 4);

    ctx.begin();
    try add(ctx, .{ .x = xb, .y = yb, .out = ob, .n_elems = xf.len });
    try ctx.submit();

    const actual = try alloc.alloc(f32, xf.len);
    defer alloc.free(actual);
    try ctx.download(ob, 0, std.mem.sliceAsBytes(actual));
    try fixture.expectClose(expected, actual, 1e-6, 0);
}

test "W5 dispatch-contract mechanics: setBytes@0, 2-D grid, barrier, atomic_float" {
    // Proves the exact machinery PORTING-moe.md §2 requires from this glue:
    // a params struct bound with setBytes at index 0, a 2-D threadgroup grid
    // with over-provisioned grid.x, threadgroup memory + barrier, and
    // device atomic_float accumulation — all under runtime MSL compilation.
    const alloc = std.testing.allocator;
    const n: u32 = 1000; // deliberately not a multiple of the 256-wide group
    const tg: u32 = 256;

    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    var rng = std.Random.DefaultPrng.init(42);
    for (x) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

    const Params = extern struct { n: u32, active_x: u32 };
    const params = Params{ .n = n, .active_x = 2 };
    const grid_y: u32 = 2;
    // grid.x = 3 > active_x = 2: the extra column must exit uniformly.
    const n_groups_total = params.active_x * grid_y;

    // CPU oracle: each active group adds the 256-block of x reversed into out.
    const expected = try alloc.alloc(f32, n);
    defer alloc.free(expected);
    @memset(expected, 0);
    for (0..n_groups_total) |_| {
        var base: u32 = 0;
        while (base < n) : (base += tg) {
            for (0..tg) |t| {
                const i = base + t;
                if (i >= n) break;
                const mirror = base + (tg - 1 - @as(u32, @intCast(t)));
                expected[i] += if (mirror < n) x[mirror] else 0;
            }
        }
    }

    const ctx = try Ctx.init(alloc);
    defer ctx.deinit();
    // Host slice → device buffer via the copy path, exactly how the glue
    // uploads ExpertMlpArgs.pairs (frozen 12-byte PairDispatch records).
    const xb = try ctx.bufferFromBytes(std.mem.sliceAsBytes(x));
    const ob = try ctx.createBuffer(@as(u64, n) * 4); // pre-zeroed accumulator

    ctx.begin();
    const pso = try ctx.pipeline("proof_atomic_accum");
    ctx.setPipeline(pso);
    ctx.setBytes(0, std.mem.asBytes(&params));
    ctx.setBuf(1, xb);
    ctx.setBuf(2, ob);
    ctx.dispatch(
        .{ .width = params.active_x + 1, .height = grid_y, .depth = 1 },
        .{ .width = tg, .height = 1, .depth = 1 },
    );
    try ctx.submit();

    const actual = try alloc.alloc(f32, n);
    defer alloc.free(actual);
    try ctx.download(ob, 0, std.mem.sliceAsBytes(actual));
    // Every out[i] receives n_groups_total atomic adds of the SAME value, so
    // the sum is order-independent: exact comparison.
    try fixture.expectClose(expected, actual, 0, 0);
}

test "A-09 microbench: per-command-buffer overhead over 100 dispatches" {
    const alloc = std.testing.allocator;
    var t = try loadXY(alloc);
    defer t.x.free(alloc);
    defer t.y.free(alloc);
    const n = t.x.asF32().len;

    const ctx = try Ctx.init(alloc);
    defer ctx.deinit();
    const xb = try ctx.bufferFromBytes(t.x.data);
    const yb = try ctx.bufferFromBytes(t.y.data);
    const ob = try ctx.createBuffer(n * 4);

    // Warm-up (pipeline compile + first-touch) outside the measurement.
    ctx.begin();
    try proofScaleAdd(ctx, 2.0, xb, yb, ob, n);
    try ctx.submit();

    // 100 one-dispatch command buffers: wall ns/batch is the A-09 number
    // (encode + commit + waitUntilCompleted round trip).
    const iters = 100;
    var gpu_total: u64 = 0;
    const wall_start = sys.monotonicNs();
    for (0..iters) |_| {
        ctx.begin();
        try proofScaleAdd(ctx, 2.0, xb, yb, ob, n);
        try ctx.submit();
        try std.testing.expect(ctx.gpuElapsedNs() > 0);
        gpu_total += ctx.gpuElapsedNs();
    }
    const wall_total = sys.monotonicNs() - wall_start;
    std.debug.print(
        "A-09 command-buffer overhead: {d} ns wall/dispatch, {d} ns gpu/dispatch ({d} iters, n={d})\n",
        .{ wall_total / iters, gpu_total / iters, iters, n },
    );
}
