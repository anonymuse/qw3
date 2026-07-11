//! Standalone test root for kernel set C (router top-k + expert SwiGLU MLP).
//! Run from the repo root so fixture paths resolve:
//!     zig test src/test_kernels_c.zig

test {
    _ = @import("kernels/cpu/kernels_c.zig");
}
