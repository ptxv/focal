import math

import pytest
import torch

import focal
from focal import pytorch_contiguous_gqa_decode_attention


def scalar_math_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths, softmax_scale=None):
    batch_size, query_heads, head_dim = query.shape
    key_value_heads = key_cache.shape[1]
    query_heads_per_key_value_head = query_heads // key_value_heads
    scale = (1.0 / math.sqrt(head_dim)) if softmax_scale is None else float(softmax_scale)
    output = torch.empty((batch_size, query_heads, head_dim), dtype=torch.float32, device=query.device)
    query_float = query.float()
    key_cache_float = key_cache.float()
    value_cache_float = value_cache.float()

    for batch_index in range(batch_size):
        sequence_length = int(sequence_lengths[batch_index])
        for query_head_index in range(query_heads):
            if sequence_length == 0:
                output[batch_index, query_head_index].zero_()
                continue
            key_value_head_index = query_head_index // query_heads_per_key_value_head
            logits = []
            for token_index in range(sequence_length):
                score = torch.zeros((), dtype=torch.float32, device=query.device)
                for head_dim_index in range(head_dim):
                    score = (
                        score
                        + query_float[batch_index, query_head_index, head_dim_index]
                        * key_cache_float[batch_index, key_value_head_index, token_index, head_dim_index]
                    )
                logits.append(score * scale)
            logits = torch.stack(logits)
            probs = torch.softmax(logits, dim=0)
            for head_dim_index in range(head_dim):
                output_component = torch.zeros((), dtype=torch.float32, device=query.device)
                for token_index in range(sequence_length):
                    output_component = (
                        output_component
                        + probs[token_index]
                        * value_cache_float[batch_index, key_value_head_index, token_index, head_dim_index]
                    )
                output[batch_index, query_head_index, head_dim_index] = output_component

    return output.to(query.dtype)


def test_public_imports():
    assert callable(focal.pytorch_contiguous_gqa_decode_attention)
    assert callable(focal.contiguous_gqa_decode_attention)
    assert isinstance(focal.cuda_extension_available(), bool)


def test_missing_extension_error_is_clear_when_unbuilt():
    if focal.cuda_extension_available():
        pytest.skip("focal CUDA extension is built")

    with pytest.raises(RuntimeError, match="pip install -e"):
        focal.require_cuda_extension()


@pytest.mark.parametrize(
    "shape,sequence_lengths",
    [
        ((1, 4, 2, 8, 5), [5]),
        ((2, 8, 2, 16, 7), [7, 3]),
    ],
)
def test_pytorch_contiguous_gqa_decode_attention_matches_expected(shape, sequence_lengths):
    batch_size, query_heads, key_value_heads, head_dim, max_sequence_length = shape
    torch.manual_seed(123)
    query = torch.randn((batch_size, query_heads, head_dim), dtype=torch.float32)
    key_cache = torch.randn((batch_size, key_value_heads, max_sequence_length, head_dim), dtype=torch.float32)
    value_cache = torch.randn((batch_size, key_value_heads, max_sequence_length, head_dim), dtype=torch.float32)
    sequence_lengths_t = torch.tensor(sequence_lengths, dtype=torch.int64)

    got = pytorch_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths_t)
    expected = scalar_math_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths_t)

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_pytorch_contiguous_gqa_decode_attention_returns_query_dtype():
    query = torch.randn((1, 4, 8), dtype=torch.float16)
    key_cache = torch.randn((1, 2, 5, 8), dtype=torch.float16)
    value_cache = torch.randn((1, 2, 5, 8), dtype=torch.float16)
    sequence_lengths = torch.tensor([4], dtype=torch.int32)

    output = pytorch_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths)

    assert output.dtype == query.dtype
    assert output.shape == query.shape


def test_pytorch_contiguous_gqa_decode_attention_allows_zero_length():
    query = torch.randn((1, 4, 8), dtype=torch.float32)
    key_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    value_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    sequence_lengths = torch.tensor([0], dtype=torch.int32)

    output = pytorch_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths)

    assert torch.count_nonzero(output) == 0


def test_contiguous_gqa_decode_attention_rejects_non_cuda_tensors():
    query = torch.randn((1, 4, 8), dtype=torch.float32)
    key_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    value_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    sequence_lengths = torch.tensor([3], dtype=torch.int32)

    with pytest.raises(ValueError, match="CUDA tensor"):
        focal.contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths)


def test_pytorch_contiguous_gqa_decode_attention_reports_invalid_lengths():
    query = torch.randn((1, 4, 8), dtype=torch.float32)
    key_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    value_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    sequence_lengths = torch.tensor([6], dtype=torch.int32)

    with pytest.raises(ValueError, match="<= max_sequence_length=5"):
        focal.pytorch_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths)


def test_contiguous_gqa_decode_attention_reports_non_tensor_query():
    key_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    value_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    sequence_lengths = torch.tensor([5], dtype=torch.int32)

    with pytest.raises(TypeError, match="query must be a torch.Tensor"):
        focal.contiguous_gqa_decode_attention(None, key_cache, value_cache, sequence_lengths)


def test_pytorch_contiguous_gqa_decode_attention_rejects_nonfinite_scale():
    query = torch.randn((1, 4, 8), dtype=torch.float32)
    key_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    value_cache = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    sequence_lengths = torch.tensor([5], dtype=torch.int32)

    with pytest.raises(ValueError, match="softmax_scale must be finite"):
        pytorch_contiguous_gqa_decode_attention(
            query,
            key_cache,
            value_cache,
            sequence_lengths,
            softmax_scale=float("inf"),
        )


@pytest.mark.parametrize(
    "query,key_cache,value_cache,sequence_lengths,error",
    [
        (
            torch.randn((4, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.tensor([5]),
            "rank 3",
        ),
        (
            torch.randn((1, 4, 8)),
            torch.randn((2, 5, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.tensor([5]),
            "rank 4",
        ),
        (
            torch.randn((1, 3, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.tensor([5]),
            "divisible",
        ),
        (
            torch.randn((1, 0, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.tensor([5]),
            "query_heads must be positive",
        ),
        (
            torch.randn((1, 4, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.tensor([[5]]),
            "shape",
        ),
        (
            torch.randn((1, 4, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.randn((1, 2, 5, 8)),
            torch.tensor([6]),
            "<= max_sequence_length",
        ),
        (
            torch.randn((1, 4, 0)),
            torch.randn((1, 2, 5, 0)),
            torch.randn((1, 2, 5, 0)),
            torch.tensor([5]),
            "head_dim must be positive",
        ),
    ],
)
def test_pytorch_contiguous_gqa_decode_attention_contract_errors(
    query,
    key_cache,
    value_cache,
    sequence_lengths,
    error,
):
    with pytest.raises((TypeError, ValueError), match=error):
        pytorch_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths)
