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
