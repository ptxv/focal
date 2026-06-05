import math

import torch


# One-kernel repo for now: keep Python API, oracle, and contract in one file.
CONTIGUOUS_GQA_DECODE_ATTENTION_HEAD_DIM = 128

INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}

CUDA_EXTENSION_MODULE = None
CUDA_EXTENSION_ERROR = None


def load_cuda_extension():
    global CUDA_EXTENSION_MODULE, CUDA_EXTENSION_ERROR

    # Import can fail before local build; cache that so availability checks stay cheap.
    if CUDA_EXTENSION_MODULE is not None:
        return CUDA_EXTENSION_MODULE
    if CUDA_EXTENSION_ERROR is not None:
        return None

    try:
        from . import contiguous_gqa_decode_attention_cuda
    except ImportError as exc:
        CUDA_EXTENSION_ERROR = exc
        return None

    CUDA_EXTENSION_MODULE = contiguous_gqa_decode_attention_cuda
    return CUDA_EXTENSION_MODULE


def cuda_extension_available():
    return load_cuda_extension() is not None


def require_cuda_extension():
    cuda_extension = load_cuda_extension()
    if cuda_extension is not None:
        return cuda_extension

    detail = f" ({CUDA_EXTENSION_ERROR})" if CUDA_EXTENSION_ERROR is not None else ""
    raise RuntimeError(
        "focal CUDA extension is unavailable. Build it with: "
        "python -m pip install -e . --no-build-isolation"
        f"{detail}"
    )


def default_softmax_scale(head_dim, softmax_scale):
    scale = (1.0 / math.sqrt(int(head_dim))) if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError(f"softmax_scale must be finite, got {softmax_scale}")
    return scale


def check_sequence_lengths_values(sequence_lengths, max_sequence_length):
    # This sync is intentional: bad lengths would read past the KV cache.
    sequence_lengths_int64 = sequence_lengths.to(dtype=torch.int64)
    if bool((sequence_lengths_int64 < 0).any().item()):
        raise ValueError("sequence_lengths values must be non-negative")
    if bool((sequence_lengths_int64 > max_sequence_length).any().item()):
        raise ValueError(f"sequence_lengths values must be <= max_sequence_length={max_sequence_length}")


def validate_contiguous_gqa_decode_attention_inputs(
    query,
    key_cache,
    value_cache,
    sequence_lengths,
    *,
    check_lengths=True,
):
    # This is the tensor contract shared by PyTorch oracle and CUDA candidate.
    if not torch.is_tensor(query):
        raise TypeError("query must be a torch.Tensor")
    if not torch.is_tensor(key_cache):
        raise TypeError("key_cache must be a torch.Tensor")
    if not torch.is_tensor(value_cache):
        raise TypeError("value_cache must be a torch.Tensor")
    if not torch.is_tensor(sequence_lengths):
        raise TypeError("sequence_lengths must be a torch.Tensor")

    if query.dim() != 3:
        raise ValueError(f"query must have rank 3 [batch, query_heads, head_dim], got rank {query.dim()}")
    if key_cache.dim() != 4:
        raise ValueError(
            f"key_cache must have rank 4 [batch, key_value_heads, max_sequence_length, head_dim], "
            f"got rank {key_cache.dim()}"
        )
    if value_cache.dim() != 4:
        raise ValueError(
            f"value_cache must have rank 4 [batch, key_value_heads, max_sequence_length, head_dim], "
            f"got rank {value_cache.dim()}"
        )

    if not query.is_floating_point():
        raise TypeError(f"query must be a floating-point tensor, got dtype={query.dtype}")
    if not key_cache.is_floating_point():
        raise TypeError(f"key_cache must be a floating-point tensor, got dtype={key_cache.dtype}")
    if not value_cache.is_floating_point():
        raise TypeError(f"value_cache must be a floating-point tensor, got dtype={value_cache.dtype}")

    batch_size, query_heads, head_dim = query.shape
    key_batch_size, key_value_heads, max_sequence_length, key_head_dim = key_cache.shape
    value_batch_size, value_key_value_heads, value_max_sequence_length, value_head_dim = value_cache.shape

    if key_batch_size != batch_size or value_batch_size != batch_size:
        raise ValueError(
            f"batch dimensions must match, got query={batch_size}, "
            f"key_cache={key_batch_size}, value_cache={value_batch_size}"
        )
    if query_heads <= 0:
        raise ValueError(f"query_heads must be positive, got {query_heads}")
    if key_value_heads <= 0:
        raise ValueError(f"key_value_heads must be positive, got {key_value_heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if value_key_value_heads != key_value_heads:
        raise ValueError(
            f"key_cache and value_cache key_value_heads must match, "
            f"got key_cache={key_value_heads}, value_cache={value_key_value_heads}"
        )
    if value_max_sequence_length != max_sequence_length:
        raise ValueError(
            f"key_cache and value_cache max_sequence_length must match, "
            f"got key_cache={max_sequence_length}, value_cache={value_max_sequence_length}"
        )
    if key_head_dim != head_dim or value_head_dim != head_dim:
        raise ValueError(
            f"head_dim values must match, got query={head_dim}, "
            f"key_cache={key_head_dim}, value_cache={value_head_dim}"
        )
    if query_heads % key_value_heads != 0:
        raise ValueError(
            f"query_heads must be divisible by key_value_heads, "
            f"got query_heads={query_heads}, key_value_heads={key_value_heads}"
        )

    if sequence_lengths.dim() != 1 or sequence_lengths.shape[0] != batch_size:
        raise ValueError(f"sequence_lengths must have shape [{batch_size}], got {tuple(sequence_lengths.shape)}")
    if sequence_lengths.dtype not in INTEGER_DTYPES:
        raise TypeError(f"sequence_lengths must have an integer dtype, got dtype={sequence_lengths.dtype}")

    devices = {query.device, key_cache.device, value_cache.device, sequence_lengths.device}
    if len(devices) != 1:
        raise ValueError(
            f"query, key_cache, value_cache, and sequence_lengths must be on the same device, got {devices}"
        )

    if check_lengths:
        check_sequence_lengths_values(sequence_lengths, max_sequence_length)

    return batch_size, query_heads, key_value_heads, max_sequence_length, head_dim


def validate_contiguous_gqa_decode_attention_cuda_inputs(
    query,
    key_cache,
    value_cache,
    sequence_lengths,
    *,
    check_lengths=True,
):
    batch_size, query_heads, key_value_heads, max_sequence_length, head_dim = (
        validate_contiguous_gqa_decode_attention_inputs(
            query,
            key_cache,
            value_cache,
            sequence_lengths,
            check_lengths=False,
        )
    )

    kernel_name = "contiguous_gqa_decode_attention"
    # CUDA v1 is intentionally narrow: fp16, contiguous, head_dim=128.
    if not query.is_cuda:
        raise ValueError(f"query must be a CUDA tensor for {kernel_name}, got device={query.device}")
    if not key_cache.is_cuda or not value_cache.is_cuda or not sequence_lengths.is_cuda:
        raise ValueError(
            f"key_cache, value_cache, and sequence_lengths must be CUDA tensors for {kernel_name}, "
            f"got devices key_cache={key_cache.device}, value_cache={value_cache.device}, "
            f"sequence_lengths={sequence_lengths.device}"
        )
    if (
        not query.is_contiguous()
        or not key_cache.is_contiguous()
        or not value_cache.is_contiguous()
        or not sequence_lengths.is_contiguous()
    ):
        raise ValueError(f"query, key_cache, value_cache, and sequence_lengths must be contiguous for {kernel_name}")
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise TypeError(
            f"query, key_cache, and value_cache must be fp16 for {kernel_name} v1, "
            f"got query={query.dtype}, key_cache={key_cache.dtype}, value_cache={value_cache.dtype}"
        )
    if sequence_lengths.dtype != torch.int32:
        raise TypeError(f"sequence_lengths must be int32 for {kernel_name} v1, got dtype={sequence_lengths.dtype}")
    if head_dim != CONTIGUOUS_GQA_DECODE_ATTENTION_HEAD_DIM:
        raise ValueError(
            f"head_dim must equal {CONTIGUOUS_GQA_DECODE_ATTENTION_HEAD_DIM} for {kernel_name} v1, "
            f"got {head_dim}"
        )

    if check_lengths:
        check_sequence_lengths_values(sequence_lengths, max_sequence_length)

    return batch_size, query_heads, key_value_heads, max_sequence_length, head_dim


def contiguous_gqa_decode_attention(
    query,
    key_cache,
    value_cache,
    sequence_lengths,
    softmax_scale=None,
):
    # Public CUDA path. no CPU fallback, so kernel failures stay obvious.
    validate_contiguous_gqa_decode_attention_cuda_inputs(
        query,
        key_cache,
        value_cache,
        sequence_lengths,
    )
    cuda_extension = require_cuda_extension()
    return cuda_extension.contiguous_gqa_decode_attention(
        query,
        key_cache,
        value_cache,
        sequence_lengths,
        default_softmax_scale(query.shape[-1], softmax_scale),
    )


def pytorch_contiguous_gqa_decode_attention(
    query,
    key_cache,
    value_cache,
    sequence_lengths,
    softmax_scale=None,
):
    # This is the oracle for CUDA changes. It optimizes for readable math, not speed.
    (
        batch_size,
        query_heads,
        key_value_heads,
        max_sequence_length,
        head_dim,
    ) = validate_contiguous_gqa_decode_attention_inputs(
        query,
        key_cache,
        value_cache,
        sequence_lengths,
    )
    query_heads_per_key_value_head = query_heads // key_value_heads
    scale = default_softmax_scale(head_dim, softmax_scale)

    query_float = query.float()
    key_cache_float = key_cache.float()
    value_cache_float = value_cache.float()
    sequence_lengths_host = sequence_lengths.to(dtype=torch.int64).detach().cpu()
    output = torch.empty((batch_size, query_heads, head_dim), device=query.device, dtype=torch.float32)

    for batch_index in range(batch_size):
        sequence_length = int(sequence_lengths_host[batch_index])
        for query_head_index in range(query_heads):
            if sequence_length == 0:
                output[batch_index, query_head_index].zero_()
                continue

            key_value_head_index = query_head_index // query_heads_per_key_value_head
            scores = torch.matmul(
                key_cache_float[batch_index, key_value_head_index, :sequence_length],
                query_float[batch_index, query_head_index],
            ) * scale
            weights = torch.softmax(scores, dim=0)
            output[batch_index, query_head_index] = torch.matmul(
                weights,
                value_cache_float[batch_index, key_value_head_index, :sequence_length],
            )

    return output.to(dtype=query.dtype)
