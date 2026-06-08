#include <torch/extension.h>


torch::Tensor w4a16_linear_cuda(
    torch::Tensor x,
    torch::Tensor wq,
    torch::Tensor scales,
    torch::Tensor zeros);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "w4a16_linear",
      &w4a16_linear_cuda,
      "FOCAL W4A16 small-M linear CUDA kernel");
}
