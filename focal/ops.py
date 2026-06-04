from ._extension import cuda_extension_available, require_cuda_extension
from ._validation import default_sm_scale, validate_decode_attn_cuda_inputs
from .reference import ref_decode_attn_contig


def decode_attn_contig(q, k, v, seq_lens, sm_scale=None):
    if not getattr(q, "is_cuda", False):
        return ref_decode_attn_contig(q, k, v, seq_lens, sm_scale)

    # Public CUDA calls validate lengths to fail clearly instead of returning NaNs.
    validate_decode_attn_cuda_inputs(q, k, v, seq_lens)
    ext = require_cuda_extension()
    return ext.decode_attn_contig(q, k, v, seq_lens, default_sm_scale(q.shape[-1], sm_scale))


def _decode_attn_contig_out(q, k, v, seq_lens, out, sm_scale=None):
    if not getattr(q, "is_cuda", False):
        out.copy_(ref_decode_attn_contig(q, k, v, seq_lens, sm_scale))
        return out

    # Benchmarks validate correctness before timing; skip length sync inside timed loops.
    validate_decode_attn_cuda_inputs(q, k, v, seq_lens, out, check_lengths=False)
    ext = require_cuda_extension()
    return ext.decode_attn_contig_out(q, k, v, seq_lens, out, default_sm_scale(q.shape[-1], sm_scale))
