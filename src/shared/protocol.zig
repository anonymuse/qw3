//! DS5 control/benchmark wire protocol.
//!
//! Framing: fixed 16-byte header followed by `payload_len` bytes of payload.
//! All nodes are Apple Silicon (aarch64, little-endian); wire format is native
//! byte order by design (assumption A-12 in docs/assumptions.md).

const std = @import("std");
const version = @import("version.zig");
const sys = @import("sys.zig");

pub const MAGIC: u32 = 0x35534444; // "DDS5" in LE bytes

pub const MsgType = enum(u16) {
    ping = 1,
    pong = 2,
    blob = 3,
    blob_ack = 4,
    health_req = 5,
    health_resp = 6,
    _,
};

pub const FrameHeader = extern struct {
    magic: u32,
    version: u16,
    msg_type: u16,
    payload_len: u64,

    pub fn init(msg_type: MsgType, payload_len: u64) FrameHeader {
        return .{
            .magic = MAGIC,
            .version = version.PROTOCOL_VERSION,
            .msg_type = @intFromEnum(msg_type),
            .payload_len = payload_len,
        };
    }
};

pub const FrameError = error{
    BadMagic,
    ConnectionClosed,
};

/// Maximum payload accepted by a daemon (guards against garbage headers).
pub const MAX_PAYLOAD: u64 = 512 * 1024 * 1024;

pub fn writeFrame(fd: sys.fd_t, msg_type: MsgType, payload: []const u8) !void {
    var hdr = FrameHeader.init(msg_type, payload.len);
    try sys.writeAllFd(fd, std.mem.asBytes(&hdr));
    if (payload.len > 0) try sys.writeAllFd(fd, payload);
}

pub fn readHeader(fd: sys.fd_t) !FrameHeader {
    var hdr: FrameHeader = undefined;
    try sys.readAllFd(fd, std.mem.asBytes(&hdr));
    if (hdr.magic != MAGIC) return FrameError.BadMagic;
    if (hdr.payload_len > MAX_PAYLOAD) return FrameError.BadMagic;
    return hdr;
}

/// Read a frame's payload into `buf` (must be exactly payload-sized).
pub fn readPayload(fd: sys.fd_t, hdr: FrameHeader, buf: []u8) !void {
    std.debug.assert(buf.len == hdr.payload_len);
    try sys.readAllFd(fd, buf);
}

/// Drain a payload we don't need to keep, using a small scratch buffer.
pub fn drainPayload(fd: sys.fd_t, hdr: FrameHeader, scratch: []u8) !void {
    var remaining: u64 = hdr.payload_len;
    while (remaining > 0) {
        const chunk: usize = @intCast(@min(remaining, scratch.len));
        try sys.readAllFd(fd, scratch[0..chunk]);
        remaining -= chunk;
    }
}

test "frame header is exactly 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(FrameHeader));
}

test "frame header init round-trips through bytes" {
    var hdr = FrameHeader.init(.ping, 4096);
    const bytes = std.mem.asBytes(&hdr);
    const back: *const FrameHeader = @ptrCast(@alignCast(bytes.ptr));
    try std.testing.expectEqual(MAGIC, back.magic);
    try std.testing.expectEqual(@intFromEnum(MsgType.ping), back.msg_type);
    try std.testing.expectEqual(@as(u64, 4096), back.payload_len);
}
