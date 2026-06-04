import torch

from ._validation import default_sm_scale, validate_decode_attn_inputs


def ref_decode_attn_contig(q, k, v, seq_lens, sm_scale=None):
    B, Hq, Hkv, _, D = validate_decode_attn_inputs(q, k, v, seq_lens)
    group_size = Hq // Hkv
    scale = default_sm_scale(D, sm_scale)

    # Accumulate in fp32 to match the CUDA kernel contract.
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    # The reference is intentionally simple, so ragged lengths drive Python loops.
    seq_lens_cpu = seq_lens.to(dtype=torch.int64).detach().cpu()
    out = torch.empty((B, Hq, D), device=q.device, dtype=torch.float32)

    for b in range(B):
        seq_len = int(seq_lens_cpu[b])
        for hq in range(Hq):
            if seq_len == 0:
                out[b, hq].zero_()
                continue

            hkv = hq // group_size
            scores = torch.matmul(k_f[b, hkv, :seq_len], q_f[b, hq]) * scale
            weights = torch.softmax(scores, dim=0)
            out[b, hq] = torch.matmul(weights, v_f[b, hkv, :seq_len])

    return out.to(dtype=q.dtype)
