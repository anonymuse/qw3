const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "ds5",
        .root_module = root,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the ds5 binary");
    run_step.dependOn(&run_cmd.step);

    const tests = b.addTest(.{ .root_module = root });
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
}
