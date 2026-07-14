//! CPU-only test root for `zig build test`. T05 gives `src/main.zig` a
//! `--backend metal` path that imports the Metal glue (src/metal/metal.zig).
//! Zig's test collection walks the WHOLE reachable module graph of a test
//! root and runs every `test` block it finds there — including metal.zig's
//! own (device-dependent) tests — so if `zig build test` kept using
//! src/main.zig as its root, it would start requiring a GPU purely because
//! main.zig imports metal.zig, regardless of which CLI subcommand actually
//! runs. That would violate the standing rule that `zig build test` stays
//! device-independent (metal/GPU coverage lives only in `test-metal` and
//! `test-gpu`).
//!
//! This file is main.zig's old `test { ... }` block, unchanged, moved to its
//! own root so it never touches metal.zig. `src/main.zig` keeps its CLI
//! entry point but is no longer the `test` step's root module (see build.zig).

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
