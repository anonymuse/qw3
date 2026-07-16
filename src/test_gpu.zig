//! Test root for T05 (M2b GPU forward pass). Exists so `zig test` gets a
//! module root at src/, letting kernels/gpu/kernels.zig reach ../../shared
//! and ../../metal. Run from the repo root (fixture paths are relative):
//!
//!   zig test src/test_gpu.zig -lobjc -framework Metal -framework Foundation \
//!       -framework CoreGraphics
//!
//! or: zig build test-gpu

test {
    _ = @import("kernels/gpu/kernels.zig");
    _ = @import("test_gpu_forward.zig");
}
