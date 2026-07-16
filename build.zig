const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    // T05: main.zig's `ds5 run --backend metal` imports the Metal glue, so
    // the exe needs these frameworks linked. Safe to link unconditionally on
    // macOS (only exercised at runtime if the user actually passes
    // --backend metal); see src/test_cpu.zig for why `zig build test` does
    // NOT use this same root module.
    root.linkSystemLibrary("objc", .{});
    root.linkFramework("Metal", .{});
    root.linkFramework("Foundation", .{});
    root.linkFramework("CoreGraphics", .{});

    const exe = b.addExecutable(.{
        .name = "ds5",
        .root_module = root,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the ds5 binary");
    run_step.dependOn(&run_cmd.step);

    // CPU-only, device-independent: a SEPARATE root from main.zig (T05) so
    // main.zig's Metal import never becomes part of this step's module
    // graph — Zig's test collector runs every `test` block reachable from a
    // root, and metal.zig's tests need a real GPU. See test_cpu.zig.
    const test_root = b.createModule(.{
        .root_source_file = b.path("src/test_cpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tests = b.addTest(.{ .root_module = test_root });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // W2: Metal glue tests (requires a Metal device; run on Apple Silicon).
    // CoreGraphics is needed for MTLCreateSystemDefaultDevice in CLI processes.
    const metal_test_mod = b.createModule(.{
        .root_source_file = b.path("src/test_metal.zig"),
        .target = target,
        .optimize = optimize,
    });
    metal_test_mod.linkSystemLibrary("objc", .{});
    metal_test_mod.linkFramework("Metal", .{});
    metal_test_mod.linkFramework("Foundation", .{});
    metal_test_mod.linkFramework("CoreGraphics", .{});
    const metal_tests = b.addTest(.{ .root_module = metal_test_mod });
    const run_metal_tests = b.addRunArtifact(metal_tests);
    run_metal_tests.setCwd(b.path(".")); // fixture paths are repo-root relative
    const test_metal_step = b.step("test-metal", "Run Metal glue tests (needs GPU)");
    test_metal_step.dependOn(&run_metal_tests.step);

    // T05: GPU kernel-provider + end-to-end forward-pass tests (needs a
    // Metal device). Separate step so `zig build test` (CPU) stays
    // device-independent, per the integration playbook.
    const gpu_test_mod = b.createModule(.{
        .root_source_file = b.path("src/test_gpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    gpu_test_mod.linkSystemLibrary("objc", .{});
    gpu_test_mod.linkFramework("Metal", .{});
    gpu_test_mod.linkFramework("Foundation", .{});
    gpu_test_mod.linkFramework("CoreGraphics", .{});
    const gpu_tests = b.addTest(.{ .root_module = gpu_test_mod });
    const run_gpu_tests = b.addRunArtifact(gpu_tests);
    run_gpu_tests.setCwd(b.path(".")); // fixture paths are repo-root relative
    const test_gpu_step = b.step("test-gpu", "Run GPU kernel-provider + e2e forward tests (needs GPU)");
    test_gpu_step.dependOn(&run_gpu_tests.step);
}
