import math

import torch


CUDA_HEAD_DIM = 128

INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def default_sm_scale(D, sm_scale):
    # Keep scale normalization shared so Python and CUDA dispatch agree.
    scale = (1.0 / math.sqrt(int(D))) if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale):
        raise ValueError("sm_scale must be finite")
    return scale


def check_seq_lens_values(seq_lens, L):
    # Length values guard K/V bounds; on CUDA this is a deliberate sync.
    seq_lens_i64 = seq_lens.to(dtype=torch.int64)
    if bool((seq_lens_i64 < 0).any().item()):
        raise ValueError("seq_lens values must be non-negative")
    if bool((seq_lens_i64 > L).any().item()):
        raise ValueError(f"seq_lens values must be <= L={L}")


def validate_decode_attn_inputs(q, k, v, seq_lens, *, check_lengths=True):
    if not torch.is_tensor(q):
        raise TypeError("q must be a torch.Tensor")
    if not torch.is_tensor(k):
        raise TypeError("k must be a torch.Tensor")
    if not torch.is_tensor(v):
        raise TypeError("v must be a torch.Tensor")
    if not torch.is_tensor(seq_lens):
        raise TypeError("seq_lens must be a torch.Tensor")

    if q.dim() != 3:
        raise ValueError(f"q must have rank 3 [B, Hq, D], got rank {q.dim()}")
    if k.dim() != 4:
        raise ValueError(f"k must have rank 4 [B, Hkv, L, D], got rank {k.dim()}")
    if v.dim() != 4:
        raise ValueError(f"v must have rank 4 [B, Hkv, L, D], got rank {v.dim()}")

    if not q.is_floating_point():
        raise TypeError("q must be a floating-point tensor")
    if not k.is_floating_point():
        raise TypeError("k must be a floating-point tensor")
    if not v.is_floating_point():
        raise TypeError("v must be a floating-point tensor")

    B, Hq, D = q.shape
    Bk, Hkv, L, Dk = k.shape
    Bv, Hkv_v, Lv, Dv = v.shape

    if Bk != B or Bv != B:
        raise ValueError("q, k, and v batch dimensions must match")
    if Hq <= 0:
        raise ValueError("Hq must be positive")
    if Hkv <= 0:
        raise ValueError("Hkv must be positive")
    if D <= 0:
        raise ValueError("D must be positive")
    if Hkv_v != Hkv:
        raise ValueError("k and v Hkv dimensions must match")
    if Lv != L:
        raise ValueError("k and v sequence dimensions must match")
    if Dk != D or Dv != D:
        raise ValueError("q, k, and v head dimensions must match")
    if Hq % Hkv != 0:
        raise ValueError(f"Hq must be divisible by Hkv, got Hq={Hq}, Hkv={Hkv}")

    if seq_lens.dim() != 1 or seq_lens.shape[0] != B:
        raise ValueError(f"seq_lens must have shape [B], got {tuple(seq_lens.shape)}")
    if seq_lens.dtype not in INTEGER_DTYPES:
        raise TypeError("seq_lens must have an integer dtype")

    devices = {q.device, k.device, v.device, seq_lens.device}
    if len(devices) != 1:
        raise ValueError("q, k, v, and seq_lens must be on the same device")

    if check_lengths:
        check_seq_lens_values(seq_lens, L)

    return B, Hq, Hkv, L, D


def validate_decode_attn_cuda_inputs(q, k, v, seq_lens, out=None, *, check_lengths=True):
    B, Hq, Hkv, L, D = validate_decode_attn_inputs(
        q, k, v, seq_lens, check_lengths=False
    )

    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor for decode_attn_contig CUDA kernel")
    if not k.is_cuda or not v.is_cuda or not seq_lens.is_cuda:
        raise ValueError("k, v, and seq_lens must be CUDA tensors for decode_attn_contig CUDA kernel")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous() or not seq_lens.is_contiguous():
        raise ValueError("q, k, v, and seq_lens must be contiguous for decode_attn_contig CUDA kernel")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError("q, k, and v must be fp16 for decode_attn_contig CUDA kernel v1")
    if seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must be int32 for decode_attn_contig CUDA kernel v1")
    if D != CUDA_HEAD_DIM:
        raise ValueError(f"D must equal {CUDA_HEAD_DIM} for decode_attn_contig CUDA kernel v1")

    if out is not None:
        if not torch.is_tensor(out):
            raise TypeError("out must be a torch.Tensor")
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.dtype != torch.float16:
            raise TypeError("out must be fp16")
        if tuple(out.shape) != (B, Hq, D):
            raise ValueError(f"out must have shape {(B, Hq, D)}, got {tuple(out.shape)}")

    if check_lengths:
        check_seq_lens_values(seq_lens, L)

    return B, Hq, Hkv, L, D
