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
const contracts = @import("shared/contracts.zig");
const gguf = @import("gguf/gguf.zig");
const forward = @import("engine/forward.zig");
const cpu = @import("kernels/cpu/ctx.zig");
const kernels_a = @import("kernels/cpu/kernels_a.zig");
const kernels_b = @import("kernels/cpu/kernels_b.zig");
const kernels_c = @import("kernels/cpu/kernels_c.zig");
const metal = @import("metal/metal.zig");
const gpu_kernels = @import("kernels/gpu/kernels.zig");
const JsonBuf = @import("shared/jsonbuf.zig").JsonBuf;

const usage =
    \\usage:
    \\  ds5 node [--name NAME] [--port PORT] [--bind HOST]
    \\  ds5 bench link --cluster PATH [--self NAME] [--out DIR] [--label S]
    \\                 [--sustained-secs N] [--quick]
    \\  ds5 health --host HOST [--port PORT]
    \\  ds5 run --model PATH --prompt-tokens "1,2,3" [--steps N] [--greedy]
    \\                [--backend cpu|metal] [--context-capacity N]
    \\                [--kv-dtype f16|f32]
    \\      defaults: context-capacity=prompt tokens + steps, kv-dtype=f32
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
        var backend: []const u8 = "cpu";
        var context_capacity: ?u32 = null;
        var kv_dtype: contracts.Dtype = .f32;
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
            } else if (std.mem.eql(u8, args[i], "--backend")) {
                backend = try requireValue(args, &i);
            } else if (std.mem.eql(u8, args[i], "--context-capacity")) {
                context_capacity = try std.fmt.parseInt(u32, try requireValue(args, &i), 10);
            } else if (std.mem.eql(u8, args[i], "--kv-dtype")) {
                kv_dtype = try parseKvDtype(try requireValue(args, &i));
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
        if (std.mem.eql(u8, backend, "cpu")) {
            try runForward(alloc, model_file, prompt_csv, n_steps, greedy, context_capacity, kv_dtype);
        } else if (std.mem.eql(u8, backend, "metal")) {
            try runForwardGpu(alloc, model_file, prompt_csv, n_steps, greedy, context_capacity, kv_dtype);
        } else {
            out.print("unknown --backend {s} (expected cpu or metal)\n", .{backend});
            return error.UnknownArgument;
        }
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

fn parseKvDtype(value: []const u8) !contracts.Dtype {
    if (std.mem.eql(u8, value, "f16")) return .f16;
    if (std.mem.eql(u8, value, "f32")) return .f32;
    out.print("unknown --kv-dtype {s} (expected f16 or f32)\n", .{value});
    return error.UnknownArgument;
}

fn kvDtypeName(dtype: contracts.Dtype) []const u8 {
    return switch (dtype) {
        .f16 => "f16",
        .f32 => "f32",
        else => "unsupported",
    };
}

/// The run command knows its complete token budget, so its safe default is
/// exactly prompt + decode steps rather than the model's potentially huge
/// advertised maximum. Explicit capacities remain useful for reserved reuse.
fn resolveContextCapacity(requested: ?u32, prompt_len: usize, n_steps: u32, model_max: u32) !u32 {
    const required: u64 = @as(u64, @intCast(prompt_len)) + n_steps;
    if (required > std.math.maxInt(u32)) {
        out.print("prompt tokens + steps exceeds the supported context range\n", .{});
        return error.InvalidContextCapacity;
    }
    const capacity = requested orelse @as(u32, @intCast(required));
    if (capacity == 0) {
        out.print("context-capacity must be at least 1\n", .{});
        return error.InvalidContextCapacity;
    }
    if (capacity > model_max) {
        out.print("context-capacity {d} exceeds model max_ctx {d}\n", .{ capacity, model_max });
        return error.InvalidContextCapacity;
    }
    if (required > capacity) {
        out.print("context-capacity {d} is smaller than prompt tokens + steps ({d})\n", .{ capacity, required });
        return error.InvalidContextCapacity;
    }
    return capacity;
}

fn runForward(alloc: std.mem.Allocator, model_path: []const u8, prompt_csv: []const u8, n_steps: u32, greedy: bool, requested_context_capacity: ?u32, kv_dtype: contracts.Dtype) !void {
    const prompt = try parsePromptTokens(alloc, prompt_csv);
    defer alloc.free(prompt);
    if (prompt.len == 0) {
        out.print("prompt-tokens cannot be empty\n", .{});
        return;
    }
    const n_toks = prompt.len;

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
    const context_capacity = try resolveContextCapacity(requested_context_capacity, n_toks, n_steps, weights.config.max_ctx);
    var engine = try CpuEngine.initWithOptions(alloc, ctx, &weights, .{
        .max_batch = 64,
        .context_capacity = context_capacity,
        .kv_dtype = kv_dtype,
    });
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

fn parsePromptTokens(alloc: std.mem.Allocator, prompt_csv: []const u8) ![]u32 {
    var n_toks: usize = 0;
    var it = std.mem.splitSequence(u8, prompt_csv, ",");
    while (it.next()) |tok_str| {
        if (std.mem.trim(u8, tok_str, " \t").len > 0) n_toks += 1;
    }
    const prompt = try alloc.alloc(u32, n_toks);
    errdefer alloc.free(prompt);
    var idx: usize = 0;
    it = std.mem.splitSequence(u8, prompt_csv, ",");
    while (it.next()) |tok_str| {
        const trimmed = std.mem.trim(u8, tok_str, " \t");
        if (trimmed.len > 0) {
            prompt[idx] = try std.fmt.parseInt(u32, trimmed, 10);
            idx += 1;
        }
    }
    return prompt;
}

/// GPU (`--backend metal`) path: the same prefill + greedy-decode driver as
/// runForward, over Engine(metal.Ctx, gpu_kernels) instead of the CPU
/// provider. Additionally emits a run-metadata JSON (coding standard #4:
/// every DS5 benchmark/run binary emits one) with per-layer GPU elapsed ns.
fn runForwardGpu(alloc: std.mem.Allocator, model_path: []const u8, prompt_csv: []const u8, n_steps: u32, greedy: bool, requested_context_capacity: ?u32, kv_dtype: contracts.Dtype) !void {
    const prompt = try parsePromptTokens(alloc, prompt_csv);
    defer alloc.free(prompt);
    if (prompt.len == 0) {
        out.print("prompt-tokens cannot be empty\n", .{});
        return;
    }
    const n_toks = prompt.len;

    var model = gguf.Model.open(alloc, model_path) catch |err| {
        out.print("failed to open {s}: {}\n", .{ model_path, err });
        return err;
    };
    defer model.deinit();

    const ctx = metal.Ctx.init(alloc) catch |err| {
        out.print("failed to init Metal device (is this Apple Silicon?): {}\n", .{err});
        return err;
    };
    defer ctx.deinit();
    try gpu_kernels.loadShaders(ctx);

    var weights = try forward.Weights.fromGguf(alloc, ctx, &model);
    defer weights.deinit();

    const GpuEngine = forward.Engine(metal.Ctx, gpu_kernels);
    const context_capacity = try resolveContextCapacity(requested_context_capacity, n_toks, n_steps, weights.config.max_ctx);
    var engine = try GpuEngine.initWithOptions(alloc, ctx, &weights, .{
        .max_batch = 64,
        .context_capacity = context_capacity,
        .kv_dtype = kv_dtype,
    });
    defer engine.deinit();

    const vocab = weights.config.vocab_size;
    const max_seq = n_toks + n_steps;
    var all_logits = try alloc.alloc(f32, @as(u64, max_seq) * vocab);
    defer alloc.free(all_logits);

    gpu_kernels.beginTiming(alloc);
    defer gpu_kernels.endTiming();

    const wall_start = sys.monotonicNs();
    try engine.forward(prompt, all_logits[0 .. @as(u64, n_toks) * vocab]);

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
    const wall_ns = sys.monotonicNs() - wall_start;

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

    try writeGpuRunMetadata(alloc, model_path, n_toks, n_steps, context_capacity, kv_dtype, wall_ns);
}

/// Run-metadata JSON (coding standard #4): schema mirrors `ds5 bench link`'s
/// (transport/linkbench.zig) — run_id/schema_version/backend fields plus the
/// GPU-specific per-layer timing this deliverable requires. `gpu_ns_per_layer
/// _boundary[i]` is the elapsed ns of the command-buffer batch that ended at
/// layer i's router-sync flush; see kernels/gpu/kernels.zig's `layerNs` doc
/// comment for exactly what that batch spans (not a perfectly isolated
/// per-layer slice, by design — isolating it would cost the batching this
/// deliverable also requires).
fn writeGpuRunMetadata(alloc: std.mem.Allocator, model_path: []const u8, n_prompt_toks: usize, n_steps: u32, context_capacity: u32, kv_dtype: contracts.Dtype, wall_ns: u64) !void {
    var jb = JsonBuf.init(alloc);
    const epoch = sys.epochSeconds();
    try jb.print("{{\"run_id\":\"run-{d}\",\"benchmark\":\"run\",\"schema_version\":1,", .{epoch});
    try jb.print("\"epoch_seconds\":{d},\"backend\":\"metal\",", .{epoch});
    try jb.raw("\"ds5_version\":");
    try jb.str(version.DS5_VERSION);
    try jb.raw(",\"model\":");
    try jb.str(model_path);
    try jb.print(",\"context_capacity\":{d},\"kv_dtype\":", .{context_capacity});
    try jb.str(kvDtypeName(kv_dtype));
    try jb.print(",\"n_prompt_tokens\":{d},\"n_decode_steps\":{d},\"wall_ns\":{d},", .{ n_prompt_toks, n_steps, wall_ns });
    try jb.raw("\"gpu_ns_per_layer_boundary\":[");
    for (gpu_kernels.layerNs(), 0..) |ns, i| {
        if (i != 0) try jb.raw(",");
        try jb.print("{d}", .{ns});
    }
    try jb.raw("]}");

    try sys.mkdirPath(alloc, "bench/results");
    var name_buf: [512]u8 = undefined;
    const file_name = try std.fmt.bufPrint(&name_buf, "bench/results/run-{d}.json", .{epoch});
    try sys.writeFileTrunc(alloc, file_name, jb.items());
    out.print("run-metadata written to {s}\n", .{file_name});
}

// No `test {}` block here: `zig build test` uses src/test_cpu.zig as its
// root instead of this file, specifically so this file's Metal import (for
// `--backend metal`, above) never becomes part of that device-independent
// step's module graph. See test_cpu.zig's doc comment.
