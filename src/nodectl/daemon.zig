//! `ds5 node` — the per-node daemon. M0 scope: answer pings (RTT), drain blobs
//! (bandwidth), and report health. One thread per connection; the M0 benchmark
//! runs one connection at a time, so this stays deliberately simple.

const std = @import("std");
const protocol = @import("../shared/protocol.zig");
const tcp = @import("../transport/tcp.zig");
const sys = @import("../shared/sys.zig");
const sysinfo = @import("../shared/sysinfo.zig");
const version = @import("../shared/version.zig");
const out = @import("../shared/out.zig");

pub const Options = struct {
    name: []const u8 = "unnamed",
    bind_host: []const u8 = "0.0.0.0",
    port: u16 = version.DEFAULT_PORT,
};

const Daemon = struct {
    opts: Options,
    started_at: i64,
};

pub fn run(opts: Options) !void {
    var daemon = Daemon{
        .opts = opts,
        .started_at = sys.epochSeconds(),
    };

    const listen_fd = try tcp.listen(opts.bind_host, opts.port);
    defer sys.closeFd(listen_fd);

    var host_buf: [sysinfo.HOSTNAME_MAX]u8 = undefined;
    out.status("ds5 node '{s}' listening on {s}:{d} (host {s}, ds5 {s}, proto {d})\n", .{
        opts.name,
        opts.bind_host,
        opts.port,
        sysinfo.hostname(&host_buf),
        version.DS5_VERSION,
        version.PROTOCOL_VERSION,
    });

    while (true) {
        const conn_fd = tcp.accept(listen_fd) catch |err| {
            out.status("accept failed: {s}\n", .{@errorName(err)});
            continue;
        };
        const t = std.Thread.spawn(.{}, handleConn, .{ &daemon, conn_fd }) catch |err| {
            out.status("thread spawn failed: {s}\n", .{@errorName(err)});
            sys.closeFd(conn_fd);
            continue;
        };
        t.detach();
    }
}

fn handleConn(daemon: *Daemon, fd: sys.fd_t) void {
    defer sys.closeFd(fd);
    serveConn(daemon, fd) catch |err| switch (err) {
        error.ConnectionClosed => {},
        else => out.status("connection error: {s}\n", .{@errorName(err)}),
    };
}

fn serveConn(daemon: *Daemon, fd: sys.fd_t) !void {
    const alloc = std.heap.page_allocator;
    // Scratch for ping echo payloads and blob draining. Pings are small;
    // blobs of any size are drained through this window without allocation.
    const scratch = try alloc.alloc(u8, 1024 * 1024);
    defer alloc.free(scratch);

    while (true) {
        const hdr = try protocol.readHeader(fd);
        switch (@as(protocol.MsgType, @enumFromInt(hdr.msg_type))) {
            .ping => {
                if (hdr.payload_len > scratch.len) return protocol.FrameError.BadMagic;
                const payload = scratch[0..@intCast(hdr.payload_len)];
                try protocol.readPayload(fd, hdr, payload);
                try protocol.writeFrame(fd, .pong, payload);
            },
            .blob => {
                try protocol.drainPayload(fd, hdr, scratch);
                var ack: u64 = hdr.payload_len;
                try protocol.writeFrame(fd, .blob_ack, std.mem.asBytes(&ack));
            },
            .health_req => {
                try protocol.drainPayload(fd, hdr, scratch);
                var host_buf: [sysinfo.HOSTNAME_MAX]u8 = undefined;
                var chip_buf: [128]u8 = undefined;
                var json_buf: [1024]u8 = undefined;
                const json = std.fmt.bufPrint(&json_buf, "{{\"node\":\"{s}\",\"hostname\":\"{s}\",\"chip\":\"{s}\",\"ds5_version\":\"{s}\",\"protocol\":{d},\"mem_total_bytes\":{d},\"uptime_s\":{d}}}", .{
                    daemon.opts.name,
                    sysinfo.hostname(&host_buf),
                    sysinfo.chipBrand(&chip_buf),
                    version.DS5_VERSION,
                    version.PROTOCOL_VERSION,
                    sysinfo.memTotalBytes(),
                    sys.epochSeconds() - daemon.started_at,
                }) catch return protocol.FrameError.BadMagic;
                try protocol.writeFrame(fd, .health_resp, json);
            },
            else => return protocol.FrameError.BadMagic,
        }
    }
}
