from .cuda_extension import require_cuda_extension
from .validation import default_sm_scale, validate_decode_attn_cuda_inputs


def decode_attn_contig(q, k, v, seq_lens, sm_scale=None):
    validate_decode_attn_cuda_inputs(q, k, v, seq_lens)
    ext = require_cuda_extension()
    return ext.decode_attn_contig(q, k, v, seq_lens, default_sm_scale(q.shape[-1], sm_scale))
