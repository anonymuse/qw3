//! Test root for the Metal glue layer (W2). Exists so `zig test` gets a
//! module root at src/, letting metal.zig reach ../shared/*. Run from the
//! repo root (fixture paths are relative):
//!
//!   zig test src/test_metal.zig -lobjc -framework Metal -framework Foundation \
//!       -framework CoreGraphics
//!
//! or: zig build test-metal

test {
    _ = @import("metal/metal.zig");
}
