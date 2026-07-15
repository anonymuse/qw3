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
const gguf = @import("gguf/gguf.zig");
const forward = @import("engine/forward.zig");
const cpu = @import("kernels/cpu/ctx.zig");
const kernels_a = @import("kernels/cpu/kernels_a.zig");
const kernels_b = @import("kernels/cpu/kernels_b.zig");
const kernels_c = @import("kernels/cpu/kernels_c.zig");

const usage =
    \\usage:
    \\  ds5 node [--name NAME] [--port PORT] [--bind HOST]
    \\  ds5 bench link --cluster PATH [--self NAME] [--out DIR] [--label S]
    \\                 [--sustained-secs N] [--quick]
    \\  ds5 health --host HOST [--port PORT]
    \\  ds5 run --model PATH --prompt-tokens "1,2,3" [--steps N] [--greedy]
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
    } else if (std.mem.eql(u8, cmd, "run")) {
        var model_path: ?[]const u8 = null;
        var prompt_str: ?[]const u8 = null;
        var n_steps: u32 = 8;
        var greedy = false;
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--model")) {
                model_path = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--prompt-tokens")) {
                prompt_str = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--steps")) {
                n_steps = try std.fmt.parseInt(u32, try requireValue(args, &i), 10);
            } else if (std.mem.eql(u8, args[i], "--greedy")) {
                greedy = true;
            } else return badArg(args[i]);
        }
        const model_file = model_path orelse {
            out.print("run requires --model PATH\n", .{});
            return error.MissingArgument;
        };
        const prompt_csv = prompt_str orelse {
            out.print("run requires --prompt-tokens CSV\n", .{});
            return error.MissingArgument;
        };
        try runForward(alloc, model_file, prompt_csv, n_steps, greedy);
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

fn runForward(alloc: std.mem.Allocator, model_path: []const u8, prompt_csv: []const u8, n_steps: u32, greedy: bool) !void {
    // Parse prompt tokens from CSV string: first count, then allocate, then fill.
    var n_toks: usize = 0;
    var it = std.mem.splitSequence(u8, prompt_csv, ",");
    while (it.next()) |tok_str| {
        if (std.mem.trim(u8, tok_str, " \t").len > 0) n_toks += 1;
    }
    if (n_toks == 0) {
        out.print("prompt-tokens cannot be empty\n", .{});
        return;
    }

    const prompt = try alloc.alloc(u32, n_toks);
    defer alloc.free(prompt);
    var idx: usize = 0;
    it = std.mem.splitSequence(u8, prompt_csv, ",");
    while (it.next()) |tok_str| {
        const trimmed = std.mem.trim(u8, tok_str, " \t");
        if (trimmed.len > 0) {
            prompt[idx] = try std.fmt.parseInt(u32, trimmed, 10);
            idx += 1;
        }
    }

    // Load the model.
    var model = gguf.Model.open(alloc, model_path) catch |err| {
        out.print("failed to open {s}: {}\n", .{ model_path, err });
        return err;
    };
    defer model.deinit();

    // Create context and weights.
    const ctx = try cpu.CpuCtx.init(alloc);
    defer ctx.deinit();
    var weights = try forward.Weights.fromGguf(alloc, ctx, &model);
    defer weights.deinit();

    // Merged kernel provider.
    const cpu_kernels = struct {
        pub const rmsNorm = kernels_a.rmsNorm;
        pub const rope = kernels_a.rope;
        pub const matmul = kernels_a.matmul;
        pub const kvAppend = kernels_a.kvAppend;
        pub const add = kernels_a.add;
        pub const gqaAttention = kernels_b.gqaAttention;
        pub const routerTopK = kernels_c.routerTopK;
        pub const expertMlpSwiglu = kernels_c.expertMlpSwiglu;
    };

    const CpuEngine = forward.Engine(cpu.CpuCtx, cpu_kernels);
    var engine = try CpuEngine.init(alloc, ctx, &weights, 64);
    defer engine.deinit();

    // Run prefill.
    const vocab = weights.config.vocab_size;
    const max_seq = n_toks + n_steps;
    var all_logits = try alloc.alloc(f32, @as(u64, max_seq) * vocab);
    defer alloc.free(all_logits);

    try engine.forward(prompt, all_logits[0 .. @as(u64, n_toks) * vocab]);

    // Greedy decode: build the full sequence.
    var seq = try alloc.alloc(u32, max_seq);
    defer alloc.free(seq);
    @memcpy(seq[0..n_toks], prompt);

    var next = forward.argmax(all_logits[(@as(u64, n_toks) - 1) * vocab .. @as(u64, n_toks) * vocab]);
    for (0..n_steps) |step| {
        seq[n_toks + step] = next;
        const row_start = (@as(u64, n_toks) + step) * vocab;
        const row_end = row_start + vocab;
        try engine.forward(&.{next}, all_logits[row_start..row_end]);
        next = forward.argmax(all_logits[row_start..row_end]);
    }

    // Print greedy tokens (the ones added during decode).
    var first = true;
    for (seq[n_toks .. n_toks + n_steps]) |tok| {
        if (!first) out.print(" ", .{});
        out.print("{d}", .{tok});
        first = false;
    }
    out.print("\n", .{});

    if (greedy) {
        var first_g = true;
        for (seq[0 .. n_toks + n_steps]) |tok| {
            if (!first_g) out.print(" ", .{});
            out.print("{d}", .{tok});
            first_g = false;
        }
        out.print("\n", .{});
    }
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
    _ = @import("kernels/cpu/kernels_a.zig");
    _ = @import("kernels/cpu/kernels_b.zig");
    _ = @import("kernels/cpu/kernels_c.zig");
    _ = @import("gguf/gguf.zig");
    _ = @import("test_forward.zig");
}
