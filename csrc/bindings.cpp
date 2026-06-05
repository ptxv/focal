#include <torch/extension.h>


torch::Tensor decode_attn_contig_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor seq_lens,
    double sm_scale);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_attn_contig", &decode_attn_contig_cuda, "contiguous GQA decode attention");
}
