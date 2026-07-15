//! CPU reference context: implements the frozen GPU API (contracts.assertGpuApi)
//! in host memory. Kernel workstreams validate their math against fixtures
//! through this context before the Metal glue exists; after M2 it stays as the
//! trusted comparator for Metal-vs-CPU trace diffs.
//!
//! Buf.handle points at the first byte of a host allocation; Buf.offset/len
//! are honored exactly as on the GPU side.

const std = @import("std");
const contracts = @import("../../shared/contracts.zig");

const KernelError = contracts.KernelError;
const Buf = contracts.Buf;

pub const CpuCtx = struct {
    alloc: std.mem.Allocator,
    /// Allocations owned by createBuffer (bufferFromBytes wraps foreign memory).
    owned: std.ArrayList([]u8),

    pub fn init(alloc: std.mem.Allocator) KernelError!*CpuCtx {
        const self = alloc.create(CpuCtx) catch return KernelError.OutOfMemory;
        self.* = .{ .alloc = alloc, .owned = .empty };
        return self;
    }

    pub fn deinit(self: *CpuCtx) void {
        for (self.owned.items) |b| self.alloc.free(b);
        self.owned.deinit(self.alloc);
        self.alloc.destroy(self);
    }

    pub fn createBuffer(self: *CpuCtx, len: u64) KernelError!Buf {
        const mem = self.alloc.alloc(u8, @intCast(len)) catch return KernelError.OutOfMemory;
        @memset(mem, 0);
        self.owned.append(self.alloc, mem) catch return KernelError.OutOfMemory;
        return .{ .handle = mem.ptr, .offset = 0, .len = len };
    }

    /// Zero-copy wrap of caller-owned bytes (mirrors newBufferWithBytesNoCopy).
    /// The caller keeps the memory alive for the Buf's lifetime.
    pub fn bufferFromBytes(self: *CpuCtx, bytes: []const u8) KernelError!Buf {
        _ = self;
        return .{ .handle = @constCast(bytes.ptr), .offset = 0, .len = bytes.len };
    }

    pub fn upload(self: *CpuCtx, buf: Buf, offset: u64, bytes: []const u8) KernelError!void {
        _ = self;
        @memcpy(mutBytes(buf, offset, bytes.len), bytes);
    }

    pub fn download(self: *CpuCtx, buf: Buf, offset: u64, out: []u8) KernelError!void {
        _ = self;
        @memcpy(out, constBytes(buf, offset, out.len));
    }

    pub fn begin(self: *CpuCtx) void {
        _ = self;
    }

    pub fn submit(self: *CpuCtx) KernelError!void {
        _ = self;
    }

    pub fn gpuElapsedNs(self: *CpuCtx) u64 {
        _ = self;
        return 0;
    }
};

fn basePtr(buf: Buf, offset: u64, len: u64) [*]u8 {
    const p: [*]u8 = @ptrCast(buf.handle.?);
    std.debug.assert(buf.offset + offset + len <= buf.offset + buf.len);
    return p + buf.offset + offset;
}

/// Kernel-side helpers: view a Buf region as typed host memory.
pub fn mutBytes(buf: Buf, offset: u64, len: u64) []u8 {
    return basePtr(buf, offset, len)[0..@intCast(len)];
}

pub fn constBytes(buf: Buf, offset: u64, len: u64) []const u8 {
    return basePtr(buf, offset, len)[0..@intCast(len)];
}

pub fn asF32(buf: Buf) []f32 {
    std.debug.assert(buf.len % 4 == 0);
    return @alignCast(std.mem.bytesAsSlice(f32, mutBytes(buf, 0, buf.len)));
}

pub fn asConstF32(buf: Buf) []const f32 {
    std.debug.assert(buf.len % 4 == 0);
    return @alignCast(std.mem.bytesAsSlice(f32, constBytes(buf, 0, buf.len)));
}

pub fn asConstI32(buf: Buf) []const i32 {
    std.debug.assert(buf.len % 4 == 0);
    return @alignCast(std.mem.bytesAsSlice(i32, constBytes(buf, 0, buf.len)));
}

pub fn asF16(buf: Buf) []f16 {
    std.debug.assert(buf.len % 2 == 0);
    return @alignCast(std.mem.bytesAsSlice(f16, mutBytes(buf, 0, buf.len)));
}

pub fn asConstF16(buf: Buf) []const f16 {
    std.debug.assert(buf.len % 2 == 0);
    return @alignCast(std.mem.bytesAsSlice(f16, constBytes(buf, 0, buf.len)));
}

test "cpu ctx satisfies the frozen gpu api" {
    comptime contracts.assertGpuApi(CpuCtx);
    const ctx = try CpuCtx.init(std.testing.allocator);
    defer ctx.deinit();

    const b = try ctx.createBuffer(16);
    try ctx.upload(b, 4, &.{ 1, 2, 3, 4 });
    var out: [4]u8 = undefined;
    try ctx.download(b, 4, &out);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);

    const vals = [_]f32{ 1.5, -2.0 };
    const w = try ctx.bufferFromBytes(std.mem.sliceAsBytes(&vals));
    try std.testing.expectEqual(@as(f32, -2.0), asConstF32(w)[1]);
}
