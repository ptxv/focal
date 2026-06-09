from importlib import import_module

import torch


SUPPORTED_M = (1, 2, 4, 8)
SUPPORTED_K = (4096, 8192)
SUPPORTED_N = (4096, 8192, 11008)
SUPPORTED_EXTRA_SHAPES = (
    (1, 11008, 4096),
    (2, 11008, 4096),
    (4, 11008, 4096),
    (8, 11008, 4096),
)
extension = None


def w4a16_linear(
    x: torch.Tensor,
    wq: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
) -> torch.Tensor:
    # Import only when the CUDA kernel is called.
    global extension
    if extension is None:
        extension = import_module("focal_w4a16._C")
    return extension.w4a16_linear(x, wq, scales, zeros)


def require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def pack_int4_weight(w_q_uint8: torch.Tensor) -> torch.Tensor:
    require_cuda_contiguous("w_q_uint8", w_q_uint8)
    if w_q_uint8.dtype != torch.uint8:
        raise ValueError("w_q_uint8 must have dtype torch.uint8")
    if w_q_uint8.ndim != 2:
        raise ValueError("w_q_uint8 must have shape [N, K]")

    n, k = w_q_uint8.shape
    if k % 8 != 0:
        raise ValueError("K must be divisible by 8")

    # Packing stays on CUDA tensors.
    q = w_q_uint8.view(n, k // 8, 8)
    packed = torch.zeros((n, k // 8), device=w_q_uint8.device, dtype=torch.int32)
    for i in range(8):
        packed.bitwise_or_((q[:, :, i].to(torch.int32) & 0xF) << (4 * i))
    return packed.contiguous()


def dequant_ref(
    wq: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    K: int,
) -> torch.Tensor:
    require_cuda_contiguous("wq", wq)
    require_cuda_contiguous("scales", scales)
    require_cuda_contiguous("zeros", zeros)
    if wq.dtype != torch.int32:
        raise ValueError("wq must have dtype torch.int32")
    if scales.dtype != torch.float16:
        raise ValueError("scales must have dtype torch.float16")
    if zeros.dtype != torch.float16:
        raise ValueError("zeros must have dtype torch.float16")
    if wq.ndim != 2:
        raise ValueError("wq must have shape [N, K / 8]")
    if K % 128 != 0:
        raise ValueError("K must be divisible by 128")

    n, k_words = wq.shape
    if k_words * 8 != K:
        raise ValueError("wq shape does not match K")
    if scales.shape != (n, K // 128):
        raise ValueError("scales must have shape [N, K / 128]")
    if zeros.shape != (n, K // 128):
        raise ValueError("zeros must have shape [N, K / 128]")

    # Repeat group scales along K.
    shifts = torch.arange(8, device=wq.device, dtype=torch.int32) * 4
    q = ((wq.view(n, k_words, 1) >> shifts.view(1, 1, 8)) & 0xF).to(torch.float32)
    q = q.reshape(n, K)
    scales_f = scales.to(torch.float32).repeat_interleave(128, dim=1)
    zeros_f = zeros.to(torch.float32).repeat_interleave(128, dim=1)
    return (q - zeros_f) * scales_f


def random_case(
    M: int,
    K: int,
    N: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if M not in SUPPORTED_M:
        raise ValueError(f"unsupported M: {M}")
    if K not in SUPPORTED_K and (M, K, N) not in SUPPORTED_EXTRA_SHAPES:
        raise ValueError(f"unsupported shape: M={M} K={K} N={N}")
    if N not in SUPPORTED_N:
        raise ValueError(f"unsupported N: {N}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16, generator=gen)
    q = torch.randint(0, 16, (N, K), device="cuda", dtype=torch.uint8, generator=gen)
    wq = pack_int4_weight(q)
    # Random scales exercise signed dequant offsets.
    scales = 0.005 + 0.045 * torch.rand(
        (N, K // 128), device="cuda", dtype=torch.float16, generator=gen
    )
    zeros = 7.0 + torch.rand(
        (N, K // 128), device="cuda", dtype=torch.float16, generator=gen
    )
    return x.contiguous(), wq, scales.contiguous(), zeros.contiguous()


__all__ = [
    "SUPPORTED_K",
    "SUPPORTED_M",
    "SUPPORTED_N",
    "SUPPORTED_EXTRA_SHAPES",
    "dequant_ref",
    "pack_int4_weight",
    "random_case",
    "w4a16_linear",
]
