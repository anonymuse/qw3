// W2 proof kernels: trivial elementwise ops validating the Metal glue layer
// (pipeline creation, buffer binding, uniform structs, grid dispatch, timing).
// Math kernels (W3-W5) live in their own .metal files and reuse the same
// dispatch plumbing in src/metal/metal.zig.

#include <metal_stdlib>
using namespace metal;

struct ProofScaleAddParams {
    float a;
    uint n;
};

// out[i] = a*x[i] + y[i]
kernel void proof_scale_add(device const float *x [[buffer(0)]],
                            device const float *y [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            constant ProofScaleAddParams &p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < p.n) {
        out[gid] = p.a * x[gid] + y[gid];
    }
}

// W5 dispatch-contract mechanics proof (PORTING-moe.md §2): params struct via
// setBytes at buffer(0), 2-D threadgroup grid, threadgroup memory + barrier,
// device atomic_float accumulation (MSL 2.4 under runtime compilation), and
// uniform whole-group early exit for over-provisioned grid.x.
//
// Each ACTIVE threadgroup (tgid.x < active_x, any tgid.y) stages a 256-wide
// block of x into threadgroup memory, barriers, then atomically accumulates
// the block REVERSED into out — so out[base+t] sums x[base+255-t] once per
// active group. The reversal makes the barrier load-bearing: thread t adds a
// value written by thread 255-t.
struct ProofAtomicParams {
    uint n;        // element count (need not be a threadgroup multiple)
    uint active_x; // groups with tgid.x >= active_x exit before any barrier
};

#define PROOF_TG 256

kernel void proof_atomic_accum(constant ProofAtomicParams &p [[buffer(0)]],
                               device const float *x [[buffer(1)]],
                               device atomic_float *out [[buffer(2)]],
                               uint2 tgid [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    if (tgid.x >= p.active_x) {
        return; // whole group exits uniformly — no barrier divergence
    }
    threadgroup float sh[PROOF_TG];
    for (uint base = 0; base < p.n; base += PROOF_TG) { // uniform trip count
        uint i = base + tid;
        sh[tid] = (i < p.n) ? x[i] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i < p.n) {
            atomic_fetch_add_explicit(&out[i], sh[PROOF_TG - 1 - tid],
                                      memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// out[i] = x[i] + y[i]  (contracts.AddArgs residual-add semantics)
kernel void proof_add(device const float *x [[buffer(0)]],
                      device const float *y [[buffer(1)]],
                      device float *out [[buffer(2)]],
                      constant uint &n [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        out[gid] = x[gid] + y[gid];
    }
}
