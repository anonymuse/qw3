#!/bin/sh
# Standalone test entry for W1 (GGUF parser).
#
# `zig test src/gguf/gguf.zig` cannot work directly: gguf.zig imports
# ../shared/*.zig, and Zig refuses file imports above the root module's
# directory ("import of file outside module path"). Real integration is one
# line in main.zig's test block:
#
#     _ = @import("gguf/gguf.zig");
#
# Until that lands, this script stages a mirror of the needed sources under
# .zig-cache with a root file that sits above both gguf/ and shared/, and runs
# `zig test` from the repo root (the fixture paths in the tests are
# CWD-relative, e.g. tests/fixtures/synthetic/model.gguf).
set -eu
cd "$(dirname "$0")/../.." # repo root
stage=.zig-cache/w1-gguf-test-root
rm -rf "$stage"
mkdir -p "$stage/gguf" "$stage/shared"
cp src/gguf/gguf.zig "$stage/gguf/"
cp src/shared/contracts.zig src/shared/sys.zig src/shared/fixture.zig "$stage/shared/"
printf 'test {\n    _ = @import("gguf/gguf.zig");\n}\n' >"$stage/root.zig"
exec zig test "$stage/root.zig"
