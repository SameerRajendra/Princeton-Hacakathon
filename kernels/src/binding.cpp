// kernels/src/binding.cpp
#include <torch/extension.h>

// Forward declarations
void launch_spectre_fused_gate_phase_scale(
    torch::Tensor gate_anchors,
    torch::Tensor V_fft,
    torch::Tensor modrelu_bias,
    torch::Tensor gate_debug,
    int t, int N, float eps
);

void launch_spectre_decode_fused(
    torch::Tensor prefix_fft,
    torch::Tensor v_new,
    torch::Tensor v_old,
    torch::Tensor gate,
    torch::Tensor v_out,
    int t, int N, bool do_evict
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spectre_fused_gate_phase_scale",
          &launch_spectre_fused_gate_phase_scale,
          "Fused gate interp + modReLU + phase + V_fft scale");
    m.def("spectre_decode_fused",
          &launch_spectre_decode_fused,
          "Fused prefix-FFT evict + update + gate + pruned iRFFT");
}