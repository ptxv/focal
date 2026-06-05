import pytest
import torch

import focal
from focal.cuda_extension import cuda_extension_available, require_cuda_extension


def test_public_imports():
    assert callable(focal.pytorch_decode_attn_contig)
    assert callable(focal.decode_attn_contig)
    assert isinstance(cuda_extension_available(), bool)


def test_missing_extension_error_is_clear_when_unbuilt():
    if cuda_extension_available():
        pytest.skip("focal CUDA extension is built")

    with pytest.raises(RuntimeError, match="pip install -e"):
        require_cuda_extension()


def test_decode_attn_contig_rejects_cpu_tensors():
    q = torch.randn((1, 4, 8), dtype=torch.float32)
    k = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    seq_lens = torch.tensor([3], dtype=torch.int32)

    with pytest.raises(ValueError, match="CUDA tensor"):
        focal.decode_attn_contig(q, k, v, seq_lens)


def test_pytorch_decode_attn_reports_invalid_lengths():
    q = torch.randn((1, 4, 8), dtype=torch.float32)
    k = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    seq_lens = torch.tensor([6], dtype=torch.int32)

    with pytest.raises(ValueError, match="<= L=5"):
        focal.pytorch_decode_attn_contig(q, k, v, seq_lens)


def test_decode_attn_contig_reports_non_tensor_q():
    k = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    v = torch.randn((1, 2, 5, 8), dtype=torch.float32)
    seq_lens = torch.tensor([5], dtype=torch.int32)

    with pytest.raises(TypeError, match="q must be a torch.Tensor"):
        focal.decode_attn_contig(None, k, v, seq_lens)
