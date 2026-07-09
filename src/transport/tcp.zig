//! TCP transport over the Thunderbolt bridge interfaces, on raw libc sockets.
//!
//! macOS exposes no user-space RDMA over Thunderbolt (assumption A-03), so the
//! DS5 transport is TCP with TCP_NODELAY and enlarged socket buffers. Whether
//! that is good enough is exactly what `ds5 bench link` measures.
//!
//! Hosts are IPv4 literals only (see manifest.zig for why).

const std = @import("std");
const c = std.c;
const sys = @import("../shared/sys.zig");

pub const SOCKET_BUF_BYTES: c_int = 4 * 1024 * 1024;

pub const NetError = error{
    InvalidAddress,
    SocketFailed,
    ConnectFailed,
    BindFailed,
    ListenFailed,
    AcceptFailed,
};

pub fn connect(host: []const u8, port: u16) NetError!sys.fd_t {
    const sa = try sockaddrIn(host, port);
    const fd = c.socket(c.AF.INET, c.SOCK.STREAM, c.IPPROTO.TCP);
    if (fd < 0) return NetError.SocketFailed;
    errdefer sys.closeFd(fd);
    if (c.connect(fd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) != 0) return NetError.ConnectFailed;
    tune(fd);
    return fd;
}

pub fn listen(bind_host: []const u8, port: u16) NetError!sys.fd_t {
    const sa = try sockaddrIn(bind_host, port);
    const fd = c.socket(c.AF.INET, c.SOCK.STREAM, c.IPPROTO.TCP);
    if (fd < 0) return NetError.SocketFailed;
    errdefer sys.closeFd(fd);
    const one: c_int = 1;
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, &one, @sizeOf(c_int));
    if (c.bind(fd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) != 0) return NetError.BindFailed;
    if (c.listen(fd, 16) != 0) return NetError.ListenFailed;
    return fd;
}

pub fn accept(listen_fd: sys.fd_t) NetError!sys.fd_t {
    while (true) {
        const fd = c.accept(listen_fd, null, null);
        if (fd < 0) {
            if (std.c._errno().* == @intFromEnum(c.E.INTR)) continue;
            return NetError.AcceptFailed;
        }
        tune(fd);
        return fd;
    }
}

/// Latency- and throughput-oriented socket tuning. Best-effort: the measured
/// numbers are what count, not whether an option was accepted.
fn tune(fd: sys.fd_t) void {
    const one: c_int = 1;
    _ = c.setsockopt(fd, c.IPPROTO.TCP, c.TCP.NODELAY, &one, @sizeOf(c_int));
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.SNDBUF, &SOCKET_BUF_BYTES, @sizeOf(c_int));
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.RCVBUF, &SOCKET_BUF_BYTES, @sizeOf(c_int));
}

fn sockaddrIn(host: []const u8, port: u16) NetError!c.sockaddr.in {
    const octets = try parseIpv4(host);
    return .{
        .len = @sizeOf(c.sockaddr.in),
        .family = c.AF.INET,
        .port = std.mem.nativeToBig(u16, port),
        .addr = @bitCast(octets),
        .zero = @splat(0),
    };
}

pub fn parseIpv4(text: []const u8) NetError![4]u8 {
    var octets: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    var i: usize = 0;
    while (it.next()) |part| : (i += 1) {
        if (i >= 4 or part.len == 0 or part.len > 3) return NetError.InvalidAddress;
        octets[i] = std.fmt.parseInt(u8, part, 10) catch return NetError.InvalidAddress;
    }
    if (i != 4) return NetError.InvalidAddress;
    return octets;
}

test "parseIpv4 accepts dotted quads" {
    try std.testing.expectEqual([4]u8{ 10, 5, 0, 2 }, try parseIpv4("10.5.0.2"));
    try std.testing.expectEqual([4]u8{ 127, 0, 0, 1 }, try parseIpv4("127.0.0.1"));
}

test "parseIpv4 rejects junk" {
    try std.testing.expectError(NetError.InvalidAddress, parseIpv4("10.5.0"));
    try std.testing.expectError(NetError.InvalidAddress, parseIpv4("10.5.0.2.9"));
    try std.testing.expectError(NetError.InvalidAddress, parseIpv4("256.1.1.1"));
    try std.testing.expectError(NetError.InvalidAddress, parseIpv4("host.local"));
}
