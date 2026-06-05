import pytest
import torch

import focal


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def require_extension_or_skip():
    if not focal.cuda_extension_available():
        pytest.skip("focal CUDA extension is not built")


@pytest.mark.parametrize(
    "batch_size,query_heads,key_value_heads,head_dim,max_sequence_length,sequence_lengths,query_key_scale",
    [
        (1, 4, 2, 128, 8, [8], 0.1),
        (1, 4, 2, 128, 8, [5], 1.0),
        (2, 8, 2, 128, 33, [33, 17], 1.0),
        (1, 4, 2, 128, 1, [1], 3.0),
    ],
)
def test_cuda_contiguous_gqa_decode_attention_matches_pytorch(
    batch_size,
    query_heads,
    key_value_heads,
    head_dim,
    max_sequence_length,
    sequence_lengths,
    query_key_scale,
):
    require_extension_or_skip()
    torch.manual_seed(2024)
    query = torch.randn((batch_size, query_heads, head_dim), device="cuda", dtype=torch.float16) * query_key_scale
    key_cache = (
        torch.randn(
            (batch_size, key_value_heads, max_sequence_length, head_dim),
            device="cuda",
            dtype=torch.float16,
        )
        * query_key_scale
    )
    value_cache = torch.randn(
        (batch_size, key_value_heads, max_sequence_length, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    sequence_lengths_t = torch.tensor(sequence_lengths, device="cuda", dtype=torch.int32)

    got = focal.contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths_t)
    expected = focal.pytorch_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths_t)
    torch.cuda.synchronize()

    assert torch.isfinite(got).all()

    difference = (got.float() - expected.float()).abs()
    max_absolute_error = difference.max().item()
    root_mean_square_error = torch.sqrt(torch.mean(difference * difference)).item()

    assert max_absolute_error <= 5e-2, (
        f"max_absolute_error={max_absolute_error:.6g}, root_mean_square_error={root_mean_square_error:.6g}, "
        f"shape={(batch_size, query_heads, key_value_heads, head_dim, max_sequence_length)}"
    )
    assert root_mean_square_error <= 5e-3, (
        f"max_absolute_error={max_absolute_error:.6g}, root_mean_square_error={root_mean_square_error:.6g}, "
        f"shape={(batch_size, query_heads, key_value_heads, head_dim, max_sequence_length)}"
    )


@pytest.mark.parametrize(
    "case,error",
    [
        ("bad_D", "head_dim must equal 128"),
        ("bad_q_dtype", "query, key_cache, and value_cache must be fp16"),
        ("bad_sequence_lengths_dtype", "sequence_lengths must be int32"),
        ("bad_grouping", "divisible"),
        ("bad_sequence_lengths_value", "<= max_sequence_length=5"),
        ("non_contiguous", "contiguous"),
    ],
)
def test_cuda_contiguous_gqa_decode_attention_rejects_invalid_contracts(case, error):
    require_extension_or_skip()
    query = torch.empty((1, 4, 128), device="cuda", dtype=torch.float16)
    key_cache = torch.empty((1, 2, 5, 128), device="cuda", dtype=torch.float16)
    value_cache = torch.empty((1, 2, 5, 128), device="cuda", dtype=torch.float16)
    sequence_lengths = torch.tensor([5], device="cuda", dtype=torch.int32)

    if case == "bad_D":
        query = torch.empty((1, 4, 64), device="cuda", dtype=torch.float16)
        key_cache = torch.empty((1, 2, 5, 64), device="cuda", dtype=torch.float16)
        value_cache = torch.empty((1, 2, 5, 64), device="cuda", dtype=torch.float16)
    elif case == "bad_q_dtype":
        query = query.float()
    elif case == "bad_sequence_lengths_dtype":
        sequence_lengths = sequence_lengths.to(torch.int64)
    elif case == "bad_grouping":
        query = torch.empty((1, 3, 128), device="cuda", dtype=torch.float16)
    elif case == "bad_sequence_lengths_value":
        sequence_lengths = torch.tensor([6], device="cuda", dtype=torch.int32)
    elif case == "non_contiguous":
        query = torch.empty((1, 128, 4), device="cuda", dtype=torch.float16).transpose(1, 2)

    with pytest.raises((TypeError, ValueError, RuntimeError), match=error):
        focal.contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths)
