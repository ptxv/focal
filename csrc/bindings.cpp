#include <torch/extension.h>


torch::Tensor decode_attn_contig_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor seq_lens,
    double sm_scale);

torch::Tensor decode_attn_contig_out_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor seq_lens,
    torch::Tensor out,
    double sm_scale);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_attn_contig", &decode_attn_contig_cuda, "contiguous GQA decode attention");
    m.def(
        "decode_attn_contig_out",
        &decode_attn_contig_out_cuda,
        "contiguous GQA decode attention with preallocated output");
}
