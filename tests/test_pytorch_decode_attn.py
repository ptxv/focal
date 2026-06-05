import math

import pytest
import torch

from focal import pytorch_decode_attn_contig


def expected_decode(q, k, v, seq_lens, sm_scale=None):
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    group_size = Hq // Hkv
    scale = (1.0 / math.sqrt(D)) if sm_scale is None else float(sm_scale)
    out = torch.empty((B, Hq, D), dtype=torch.float32, device=q.device)
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    for b in range(B):
        seq_len = int(seq_lens[b])
        for hq in range(Hq):
            if seq_len == 0:
                out[b, hq].zero_()
                continue
            hkv = hq // group_size
            logits = []
            for t in range(seq_len):
                score = torch.zeros((), dtype=torch.float32, device=q.device)
                for d in range(D):
                    score = score + q_f[b, hq, d] * k_f[b, hkv, t, d]
                logits.append(score * scale)
            logits = torch.stack(logits)
            probs = torch.softmax(logits, dim=0)
            for d in range(D):
                value = torch.zeros((), dtype=torch.float32, device=q.device)
                for t in range(seq_len):
                    value = value + probs[t] * v_f[b, hkv, t, d]
                out[b, hq, d] = value

    return out.to(q.dtype)


@pytest.mark.parametrize(
    "shape,seq_lens",
    [
        ((1, 4, 2, 8, 5), [5]),
        ((2, 8, 2, 16, 7), [7, 3]),
    ],
)
def test_pytorch_decode_attn_matches_expected(shape, seq_lens):
    B, Hq, Hkv, D, L = shape
    torch.manual_seed(123)
    q = torch.randn((B, Hq, D), dtype=torch.float32)
    k = torch.randn((B, Hkv, L, D), dtype=torch.float32)
    v = torch.randn((B, Hkv, L, D), dtype=torch.float32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)

    got = pytorch_decode_attn_contig(q, k, v, seq_lens_t)
    expected = expected_decode(q, k, v, seq_lens_t)

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_pytorch_decode_attn_returns_q_dtype():
    q = torch.randn((1, 4, 8), dtype=torch.float16)
    k = torch.randn((1, 2, 5, 8), dtype=torch.float16)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float16)
    seq_lens = torch.tensor([4], dtype=torch.int32)

    out = pytorch_decode_attn_contig(q, k, v, seq_lens)

    assert out.dtype == q.dtype
    assert out.shape == q.shape


def test_pytorch_decode_attn_allows_zero_length():
    q = torch.randn((1, 4, 8), dtype=torch.float32)
    k = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    seq_lens = torch.tensor([0], dtype=torch.int32)

    out = pytorch_decode_attn_contig(q, k, v, seq_lens)

    assert torch.count_nonzero(out) == 0


def test_pytorch_decode_attn_rejects_nonfinite_scale():
    q = torch.randn((1, 4, 8), dtype=torch.float32)
    k = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    seq_lens = torch.tensor([5], dtype=torch.int32)

    with pytest.raises(ValueError, match="sm_scale must be finite"):
        pytorch_decode_attn_contig(q, k, v, seq_lens, sm_scale=float("inf"))


@pytest.mark.parametrize(
    "q,k,v,seq_lens,error",
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
            "Hq must be positive",
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
            "<= L",
        ),
        (
            torch.randn((1, 4, 0)),
            torch.randn((1, 2, 5, 0)),
            torch.randn((1, 2, 5, 0)),
            torch.tensor([5]),
            "D must be positive",
        ),
    ],
)
def test_pytorch_decode_attn_validation_errors(q, k, v, seq_lens, error):
    with pytest.raises((TypeError, ValueError), match=error):
        pytorch_decode_attn_contig(q, k, v, seq_lens)
