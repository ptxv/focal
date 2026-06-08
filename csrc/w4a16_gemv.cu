#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>


namespace {

constexpr int N_TILE = 16;
constexpr int THREADS = 256;
constexpr int COL_THREADS = THREADS / N_TILE;


template <int M>
__global__ void w4a16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ wq,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    __nv_bfloat16* __restrict__ y,
    int K,
    int N) {
  const int tid = threadIdx.x;
  const int col_local = tid / COL_THREADS;
  const int lane = tid & (COL_THREADS - 1);
  const int n = blockIdx.x * N_TILE + col_local;
  const int words_per_row = K >> 3;
  const int groups_per_row = K >> 7;

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    acc[m] = 0.0f;
  }

  const int w_base = n * words_per_row;
  const int group_base = n * groups_per_row;

  // More CTAs keep small-N H100 launches from underfilling the SMs.
  for (int group = 0; group < groups_per_row; ++group) {
    const float scale = __half2float(scales[group_base + group]);
    const float zero = __half2float(zeros[group_base + group]);
    const int k_begin = (group << 7) + lane;
    const int k_end = (group + 1) << 7;

    for (int k = k_begin; k < k_end; k += COL_THREADS) {
      const uint32_t word = static_cast<uint32_t>(wq[w_base + (k >> 3)]);
      const int q = static_cast<int>((word >> (4 * (k & 7))) & 0xF);
      const float w = (static_cast<float>(q) - zero) * scale;

#pragma unroll
      for (int m = 0; m < M; ++m) {
        acc[m] = fmaf(__bfloat162float(x[m * K + k]), w, acc[m]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M; ++m) {
    acc[m] += __shfl_down_sync(0xffffffff, acc[m], 8, COL_THREADS);
    acc[m] += __shfl_down_sync(0xffffffff, acc[m], 4, COL_THREADS);
    acc[m] += __shfl_down_sync(0xffffffff, acc[m], 2, COL_THREADS);
    acc[m] += __shfl_down_sync(0xffffffff, acc[m], 1, COL_THREADS);
  }

  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      y[m * N + n] = __float2bfloat16(acc[m]);
    }
  }
}


void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
}


}  // namespace


torch::Tensor w4a16_linear_cuda(
    torch::Tensor x,
    torch::Tensor wq,
    torch::Tensor scales,
    torch::Tensor zeros) {
  check_tensor(x, "x", at::kBFloat16);
  check_tensor(wq, "wq", at::kInt);
  check_tensor(scales, "scales", at::kHalf);
  check_tensor(zeros, "zeros", at::kHalf);

  TORCH_CHECK(x.dim() == 2, "x must have shape [M, K]");
  TORCH_CHECK(wq.dim() == 2, "wq must have shape [N, K / 8]");
  TORCH_CHECK(scales.dim() == 2, "scales must have shape [N, K / 128]");
  TORCH_CHECK(zeros.dim() == 2, "zeros must have shape [N, K / 128]");

  TORCH_CHECK(wq.device() == x.device(), "wq must be on the same device as x");
  TORCH_CHECK(scales.device() == x.device(), "scales must be on the same device as x");
  TORCH_CHECK(zeros.device() == x.device(), "zeros must be on the same device as x");

  const int64_t M64 = x.size(0);
  const int64_t K64 = x.size(1);
  const int64_t N64 = wq.size(0);

  TORCH_CHECK(M64 == 1 || M64 == 2 || M64 == 4 || M64 == 8, "unsupported M");
  TORCH_CHECK(K64 == 4096 || K64 == 8192, "unsupported K");
  TORCH_CHECK(N64 == 4096 || N64 == 8192 || N64 == 11008, "unsupported N");
  TORCH_CHECK(K64 % 128 == 0, "K must be divisible by 128");
  TORCH_CHECK(N64 % 64 == 0, "N must be divisible by 64");
  TORCH_CHECK(wq.size(1) == K64 / 8, "wq must have shape [N, K / 8]");
  TORCH_CHECK(scales.size(0) == N64 && scales.size(1) == K64 / 128,
              "scales must have shape [N, K / 128]");
  TORCH_CHECK(zeros.size(0) == N64 && zeros.size(1) == K64 / 128,
              "zeros must have shape [N, K / 128]");

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto y = torch::empty({M64, N64}, x.options());

  const int M = static_cast<int>(M64);
  const int K = static_cast<int>(K64);
  const int N = static_cast<int>(N64);

  const auto* x_ptr =
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* wq_ptr = wq.data_ptr<int32_t>();
  const auto* scales_ptr =
      reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  const auto* zeros_ptr =
      reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>());
  auto* y_ptr = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());

  const dim3 grid(N / N_TILE);
  const dim3 block(THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (M) {
    case 1:
      w4a16_gemv_kernel<1><<<grid, block, 0, stream>>>(
          x_ptr, wq_ptr, scales_ptr, zeros_ptr, y_ptr, K, N);
      break;
    case 2:
      w4a16_gemv_kernel<2><<<grid, block, 0, stream>>>(
          x_ptr, wq_ptr, scales_ptr, zeros_ptr, y_ptr, K, N);
      break;
    case 4:
      w4a16_gemv_kernel<4><<<grid, block, 0, stream>>>(
          x_ptr, wq_ptr, scales_ptr, zeros_ptr, y_ptr, K, N);
      break;
    case 8:
      w4a16_gemv_kernel<8><<<grid, block, 0, stream>>>(
          x_ptr, wq_ptr, scales_ptr, zeros_ptr, y_ptr, K, N);
      break;
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
