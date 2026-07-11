//! Standalone test root for kernel set B (causal GQA attention).
//! Run from the repo root so fixture paths resolve:
//!     zig test src/test_kernels_b.zig

test {
    _ = @import("kernels/cpu/kernels_b.zig");
}
