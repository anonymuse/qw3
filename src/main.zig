//! ds5 — DS5 distributed inference runtime CLI.
//!
//! M0 subcommands:
//!   ds5 node [--name N] [--port P] [--bind HOST]     run the per-node daemon
//!   ds5 bench link --cluster PATH [--self NAME] ...  measure the mesh
//!   ds5 health --host H [--port P]                   one-shot daemon health check
//!   ds5 version                                      print version

const std = @import("std");
const daemon = @import("nodectl/daemon.zig");
const linkbench = @import("transport/linkbench.zig");
const protocol = @import("shared/protocol.zig");
const sys = @import("shared/sys.zig");
const tcp = @import("transport/tcp.zig");
const version = @import("shared/version.zig");
const out = @import("shared/out.zig");

const usage =
    \\usage:
    \\  ds5 node [--name NAME] [--port PORT] [--bind HOST]
    \\  ds5 bench link --cluster PATH [--self NAME] [--out DIR] [--label S]
    \\                 [--sustained-secs N] [--quick]
    \\  ds5 health --host HOST [--port PORT]
    \\  ds5 version
    \\
;

pub fn main(init: std.process.Init) !void {
    const alloc = init.arena.allocator();
    const args = try init.minimal.args.toSlice(alloc);
    if (args.len < 2) {
        out.print("{s}", .{usage});
        return;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "version")) {
        out.print("ds5 {s} (protocol {d})\n", .{ version.DS5_VERSION, version.PROTOCOL_VERSION });
    } else if (std.mem.eql(u8, cmd, "node")) {
        var opts = daemon.Options{};
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--name")) {
                opts.name = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--port")) {
                opts.port = try std.fmt.parseInt(u16, try requireValue(args, &i), 10);
            } else if (std.mem.eql(u8, args[i], "--bind")) {
                opts.bind_host = try requireValue(args, &i);
            } else return badArg(args[i]);
        }
        try daemon.run(opts);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        if (args.len < 3 or !std.mem.eql(u8, args[2], "link")) {
            out.print("{s}", .{usage});
            return error.UnknownCommand;
        }
        var cluster_path: ?[]const u8 = null;
        var opts_partial = linkbench.Options{ .cluster_path = "" };
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--cluster")) {
                cluster_path = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--self")) {
                opts_partial.self_name = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--out")) {
                opts_partial.out_dir = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--label")) {
                opts_partial.label = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--sustained-secs")) {
                opts_partial.sustained_secs = try std.fmt.parseInt(u64, try requireValue(args, &i), 10);
            } else if (std.mem.eql(u8, args[i], "--quick")) {
                opts_partial.quick = true;
                opts_partial.sustained_secs = @min(opts_partial.sustained_secs, 3);
            } else return badArg(args[i]);
        }
        opts_partial.cluster_path = cluster_path orelse {
            out.print("bench link requires --cluster PATH\n", .{});
            return error.MissingArgument;
        };
        try linkbench.run(alloc, opts_partial);
    } else if (std.mem.eql(u8, cmd, "health")) {
        var host: ?[]const u8 = null;
        var port: u16 = version.DEFAULT_PORT;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--host")) {
                host = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--port")) {
                port = try std.fmt.parseInt(u16, try requireValue(args, &i), 10);
            } else return badArg(args[i]);
        }
        const h = host orelse {
            out.print("health requires --host HOST\n", .{});
            return error.MissingArgument;
        };
        try healthCheck(alloc, h, port);
    } else {
        out.print("{s}", .{usage});
        return error.UnknownCommand;
    }
}

fn healthCheck(alloc: std.mem.Allocator, host: []const u8, port: u16) !void {
    const fd = try tcp.connect(host, port);
    defer sys.closeFd(fd);
    try protocol.writeFrame(fd, .health_req, &.{});
    const hdr = try protocol.readHeader(fd);
    const payload = try alloc.alloc(u8, @intCast(hdr.payload_len));
    try protocol.readPayload(fd, hdr, payload);
    out.print("{s}\n", .{payload});
}

fn requireValue(args: []const [:0]const u8, i: *usize) ![]const u8 {
    i.* += 1;
    if (i.* >= args.len) {
        out.print("missing value for {s}\n", .{args[i.* - 1]});
        return error.MissingArgument;
    }
    return args[i.*];
}

fn badArg(arg: []const u8) anyerror {
    out.print("unknown argument: {s}\n{s}", .{ arg, usage });
    return error.UnknownArgument;
}

test {
    _ = @import("shared/protocol.zig");
    _ = @import("shared/activation_packet.zig");
    _ = @import("shared/checksum.zig");
    _ = @import("shared/manifest.zig");
    _ = @import("shared/stats.zig");
    _ = @import("shared/jsonbuf.zig");
    _ = @import("shared/sysinfo.zig");
    _ = @import("shared/contracts.zig");
    _ = @import("shared/fixture.zig");
}
