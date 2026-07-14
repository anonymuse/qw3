//! Standalone test root for kernel set A (rmsNorm, rope, matmul, kvAppend, add).
//! Run from the repo root so fixture paths resolve:
//!     zig test src/test_kernels_a.zig

test {
    _ = @import("kernels/cpu/kernels_a.zig");
}
