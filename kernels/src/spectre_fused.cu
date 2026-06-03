// kernels/src/spectre_fused.cu
// Fuses: anchor interp → modReLU → positional phase → V_fft scale → modReLU
// All in one kernel, avoiding 4 separate launches in SpectreHead.forward()
//
// Grid:  (B, num_groups) blocks
// Block: min(F_half, 1024) threads — one thread per frequency bin

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cooperative_groups.h>
#include "binding.h"

namespace cg = cooperative_groups;

// ─── modReLU in-place ────────────────────────────────────────────────────────
// z ← relu(|z| + bias) / sqrt(|z|² + eps²)  · z
__device__ __forceinline__ cuFloatComplex modrelu(
    cuFloatComplex z, float bias, float eps)
{
    float mag = hypotf(z.x, z.y);
    float mag_stable = sqrtf(mag * mag + eps * eps);
    float scale = fmaxf(mag + bias, 0.0f) / mag_stable;
    return make_cuFloatComplex(z.x * scale, z.y * scale);
}

// ─── Main fused kernel ───────────────────────────────────────────────────────
// gate_anchors : (B, G, B_buckets, 2) float  [re, im interleaved]
// gate_out     : (B, G, F_half) complex out
// V_fft        : (B, F_half, d_g*G) complex in/out  [modified in-place]
// modrelu_bias : (F_half * G,) float
// t            : current timestep (for positional phase)
// N            : FFT size
// B_buckets    : anchor count before interpolation
// F_half       : n_fft//2 + 1
// G            : num_groups
// d_g          : head_dim / G
extern "C" __global__ void spectre_fused_gate_phase_scale(
    const float* __restrict__  gate_anchors,  // (B, G, B_k, 2)
    float2*      __restrict__  V_fft,         // (B, F_half, d) complex in/out
    const float* __restrict__  modrelu_bias,  // (F_half*G,)
    float2*      __restrict__  gate_debug,    // (B, G, F_half) optional debug
    int B, int G, int B_k, int F_half, int d_g,
    int t, int N,
    float eps
) {
    // Each block handles one (batch, group) pair
    int b = blockIdx.x;
    int g = blockIdx.y;
    int f = threadIdx.x;  // frequency bin

    if (b >= B || g >= G || f >= F_half) return;

    // ── Step 1: Linear interpolation of anchors to F_half ──────────────────
    // anchor positions uniformly spaced in [0, F_half-1]
    float pos = (float)f * (float)(B_k - 1) / (float)(F_half - 1);
    int   lo  = min((int)pos, B_k - 2);
    float t_interp = pos - (float)lo;

    // Load anchor re/im for group g, batch b
    int base = ((b * G + g) * B_k + lo) * 2;
    float re_lo = gate_anchors[base + 0];
    float im_lo = gate_anchors[base + 1];
    float re_hi = gate_anchors[base + 2];
    float im_hi = gate_anchors[base + 3];

    // Linear interpolation (upgrade to Catmull-Rom for cubic quality)
    cuFloatComplex gate = make_cuFloatComplex(
        re_lo + t_interp * (re_hi - re_lo),
        im_lo + t_interp * (im_hi - im_lo)
    );

    // ── Step 2: modReLU ─────────────────────────────────────────────────────
    float bias = modrelu_bias[g * F_half + f];
    gate = modrelu(gate, bias, eps);

    // ── Step 3: Positional phase injection ──────────────────────────────────
    // g_k ← g_k * exp(j * 2π * k * t / N)
    float angle = 2.0f * M_PIf * (float)f * (float)t / (float)N;
    float cos_a, sin_a;
    __sincosf(angle, &sin_a, &cos_a);
    cuFloatComplex phase = make_cuFloatComplex(cos_a, sin_a);
    gate = cuCmulf(gate, phase);

    // Optional debug dump
    if (gate_debug) {
        gate_debug[b * G * F_half + g * F_half + f] = make_float2(gate.x, gate.y);
    }

    // ── Step 4: Scale V_fft[:, f, g*d_g:(g+1)*d_g] by gate ────────────────
    int d = G * d_g;
    int v_base = b * F_half * d + f * d + g * d_g;
    for (int c = 0; c < d_g; ++c) {
        float2 v = V_fft[v_base + c];
        cuFloatComplex vc = make_cuFloatComplex(v.x, v.y);
        cuFloatComplex out = cuCmulf(gate, vc);
        V_fft[v_base + c] = make_float2(out.x, out.y);
    }
}