import pytest
import torch

import focal
from focal.ops import cuda_extension_available


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _require_extension():
    if not cuda_extension_available():
        pytest.skip("focal CUDA extension is not built")


@pytest.mark.parametrize(
    "B,Hq,Hkv,D,L,seq_lens,qk_scale",
    [
        (1, 4, 2, 128, 8, [8], 0.1),
        (1, 4, 2, 128, 8, [5], 1.0),
        (2, 8, 2, 128, 33, [33, 17], 1.0),
        (1, 4, 2, 128, 1, [1], 3.0),
    ],
)
def test_cuda_decode_attn_contig_matches_reference(B, Hq, Hkv, D, L, seq_lens, qk_scale):
    _require_extension()
    torch.manual_seed(2024)
    q = torch.randn((B, Hq, D), device="cuda", dtype=torch.float16) * qk_scale
    k = torch.randn((B, Hkv, L, D), device="cuda", dtype=torch.float16) * qk_scale
    v = torch.randn((B, Hkv, L, D), device="cuda", dtype=torch.float16)
    seq_lens_t = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

    got = focal.decode_attn_contig(q, k, v, seq_lens_t)
    expected = focal.ref_decode_attn_contig(q, k, v, seq_lens_t)
    torch.cuda.synchronize()

    assert torch.isfinite(got).all()

    diff = (got.float() - expected.float()).abs()
    max_abs_error = diff.max().item()
    rms_error = torch.sqrt(torch.mean(diff * diff)).item()

    assert max_abs_error <= 5e-2, (
        f"max_abs_error={max_abs_error:.6g}, rms_error={rms_error:.6g}, "
        f"shape={(B, Hq, Hkv, D, L)}"
    )
    assert rms_error <= 5e-3, (
        f"max_abs_error={max_abs_error:.6g}, rms_error={rms_error:.6g}, "
        f"shape={(B, Hq, Hkv, D, L)}"
    )


@pytest.mark.parametrize(
    "case,error",
    [
        ("bad_D", "D must equal 128"),
        ("bad_q_dtype", "q must be fp16"),
        ("bad_seq_lens_dtype", "seq_lens must be int32"),
        ("bad_grouping", "divisible"),
        ("bad_seq_lens_value", "<= L=5"),
        ("non_contiguous", "contiguous"),
    ],
)
def test_cuda_decode_attn_contig_rejects_invalid_contracts(case, error):
    _require_extension()
    q = torch.empty((1, 4, 128), device="cuda", dtype=torch.float16)
    k = torch.empty((1, 2, 5, 128), device="cuda", dtype=torch.float16)
    v = torch.empty((1, 2, 5, 128), device="cuda", dtype=torch.float16)
    seq_lens = torch.tensor([5], device="cuda", dtype=torch.int32)

    if case == "bad_D":
        q = torch.empty((1, 4, 64), device="cuda", dtype=torch.float16)
        k = torch.empty((1, 2, 5, 64), device="cuda", dtype=torch.float16)
        v = torch.empty((1, 2, 5, 64), device="cuda", dtype=torch.float16)
    elif case == "bad_q_dtype":
        q = q.float()
    elif case == "bad_seq_lens_dtype":
        seq_lens = seq_lens.to(torch.int64)
    elif case == "bad_grouping":
        q = torch.empty((1, 3, 128), device="cuda", dtype=torch.float16)
    elif case == "bad_seq_lens_value":
        seq_lens = torch.tensor([6], device="cuda", dtype=torch.int32)
    elif case == "non_contiguous":
        q = torch.empty((1, 128, 4), device="cuda", dtype=torch.float16).transpose(1, 2)

    with pytest.raises((TypeError, ValueError, RuntimeError), match=error):
        focal.decode_attn_contig(q, k, v, seq_lens)
