import pytest
import torch

import focal
from focal._extension import require_cuda_extension
from focal.ops import cuda_extension_available


def test_public_imports_without_cuda_extension():
    assert callable(focal.ref_decode_attn_contig)
    assert callable(focal.decode_attn_contig)
    assert isinstance(cuda_extension_available(), bool)


def test_missing_extension_error_is_clear_when_unbuilt():
    if cuda_extension_available():
        pytest.skip("focal CUDA extension is built")

    with pytest.raises(RuntimeError, match="FOCAL_BUILD_CUDA=1"):
        require_cuda_extension()


def test_decode_attn_contig_cpu_falls_back_to_reference():
    torch.manual_seed(11)
    q = torch.randn((1, 4, 8), dtype=torch.float32)
    k = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    seq_lens = torch.tensor([3], dtype=torch.int32)

    got = focal.decode_attn_contig(q, k, v, seq_lens)
    expected = focal.ref_decode_attn_contig(q, k, v, seq_lens)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_decode_attn_contig_cpu_reports_invalid_lengths():
    q = torch.randn((1, 4, 8), dtype=torch.float32)
    k = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    seq_lens = torch.tensor([6], dtype=torch.int32)

    with pytest.raises(ValueError, match="<= L=5"):
        focal.decode_attn_contig(q, k, v, seq_lens)


def test_decode_attn_contig_reports_non_tensor_q():
    k = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    seq_lens = torch.tensor([5], dtype=torch.int32)

    with pytest.raises(TypeError, match="q must be a torch.Tensor"):
        focal.decode_attn_contig(None, k, v, seq_lens)
