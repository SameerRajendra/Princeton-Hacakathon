// kernels/src/spectre_decode.cu
// Fuses the entire decode_step() of PrefixFFTCache into one kernel:
//  (a) evict old token from prefix_fft
//  (b) add new token twiddle
//  (c) update ring buffer + sum_q
//  (d) apply gate (pre-computed by gate kernel above)
//  (e) compute pruned iRFFT at position pos = t % N
//
// This eliminates 5 separate Python/CUDA round-trips per decode step.
// Called with grid=(F_half,) threads for one (head, batch=1) decode step.

#include <cuComplex.h>

extern "C" __global__ void spectre_decode_fused(
    float2*       __restrict__  prefix_fft,    // (F_half, d) complex  [in/out]
    const float*  __restrict__  v_new,         // (d,) new value token
    const float*  __restrict__  v_old,         // (d,) evicted value token
    const float2* __restrict__  gate,          // (F_half, d) complex  [pre-computed]
    float*        __restrict__  v_out,         // (d,) output
    int F_half, int d, int t, int N,
    bool do_evict   // true when t >= N
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F_half) return;

    // Precompute twiddle factors
    float omega = -2.0f * M_PIf / (float)N;

    // ── (a) Evict old token ──────────────────────────────────────────────────
    if (do_evict) {
        int j_old = (t - N) % N;
        float angle_old = omega * (float)f * (float)j_old;
        float cos_old, sin_old;
        __sincosf(angle_old, &sin_old, &cos_old);  // note: exp(-j*ω)

        for (int c = 0; c < d; ++c) {
            float2 p = prefix_fft[f * d + c];
            // prefix_fft[f,c] -= exp(-j*2π*f*j_old/N) * v_old[c]
            p.x -= cos_old * v_old[c];
            p.y -= sin_old * v_old[c];
            prefix_fft[f * d + c] = p;
        }
    }

    // ── (b) Add new token ────────────────────────────────────────────────────
    float angle_new = omega * (float)f * (float)t;
    float cos_new, sin_new;
    __sincosf(angle_new, &sin_new, &cos_new);

    for (int c = 0; c < d; ++c) {
        float2 p = prefix_fft[f * d + c];
        p.x += cos_new * v_new[c];
        p.y += sin_new * v_new[c];
        prefix_fft[f * d + c] = p;
    }

    __syncthreads();  // ensure prefix_fft updated before gate apply

    // ── (c) Apply gate and accumulate iRFFT at pos = t % N ──────────────────
    // pruned iRFFT: only compute output at position pos
    // contrib[f] = (gate[f,c] * prefix_fft[f,c]).real * cos_phase
    //            - (gate[f,c] * prefix_fft[f,c]).imag * sin_phase
    //            (then doubled for f in [1, F_half-1) except Nyquist)
    int pos = t % N;
    float phase_angle = 2.0f * M_PIf * (float)f * (float)pos / (float)N;
    float cos_p, sin_p;
    __sincosf(phase_angle, &sin_p, &cos_p);

    // Accumulate per-frequency contribution into v_out via atomicAdd
    float scale = (f == 0 || (N % 2 == 0 && f == F_half - 1)) ? 1.0f : 2.0f;
    scale /= (float)N;

    for (int c = 0; c < d; ++c) {
        float2 pfft = prefix_fft[f * d + c];
        float2 g    = gate[f * d + c];

        // Complex multiply: (g.x + j*g.y) * (pfft.x + j*pfft.y)
        float mixed_re = g.x * pfft.x - g.y * pfft.y;
        float mixed_im = g.x * pfft.y + g.y * pfft.x;

        // Real part of mixed * exp(j * phase)
        float contrib = (mixed_re * cos_p - mixed_im * sin_p) * scale;
        atomicAdd(&v_out[c], contrib);
    }
}