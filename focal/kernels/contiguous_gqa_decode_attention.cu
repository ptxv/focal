#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>


namespace {

constexpr int kHeadDimension = 128;
constexpr int kThreadsPerBlock = 128;
constexpr const char* kKernelName = "contiguous_gqa_decode_attention";

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_contiguous_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_fp16_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be fp16");
}

__device__ __forceinline__ float warp_reduce_sum(float partial_sum) {
    // Score dot-product reduction starts as one head_dim component per lane.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, offset);
    }
    return partial_sum;
}

__device__ __forceinline__ float block_reduce_sum(float partial_sum) {
    __shared__ float warp_sums[4];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    partial_sum = warp_reduce_sum(partial_sum);
    if (lane == 0) {
        warp_sums[warp] = partial_sum;
    }
    __syncthreads();

    // Four warp sums exactly match the fixed 128-thread block.
    partial_sum = threadIdx.x < 4 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        partial_sum = warp_reduce_sum(partial_sum);
    }
    return partial_sum;
}

__global__ void contiguous_gqa_decode_attention_kernel(
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int32_t* __restrict__ sequence_lengths,
    __half* __restrict__ output,
    int query_heads,
    int key_value_heads,
    int max_sequence_length,
    float softmax_scale) {
    // Layouts are contiguous fp16:
    // query [batch, query_heads, 128]
    // key/value cache [batch, key_value_heads, max_sequence_length, 128]
    const int head_dim_index = threadIdx.x;
    const int output_vector_index = blockIdx.x;
    const int batch_index = output_vector_index / query_heads;
    const int query_head_index = output_vector_index - batch_index * query_heads;
    const int query_heads_per_key_value_head = query_heads / key_value_heads;
    const int key_value_head_index = query_head_index / query_heads_per_key_value_head;
    const int sequence_length = sequence_lengths[batch_index];

    // One block owns output[batch, query_head, :]. no atomics, no cross-block merge.
    const long long query_offset =
        (static_cast<long long>(batch_index) * query_heads + query_head_index) * kHeadDimension;
    const long long key_value_offset =
        (static_cast<long long>(batch_index) * key_value_heads + key_value_head_index) *
        max_sequence_length *
        kHeadDimension;
    const long long output_offset = query_offset;

    const bool invalid_sequence_length = sequence_length < 0 || sequence_length > max_sequence_length;
    if (invalid_sequence_length || sequence_length == 0) {
        // Python validation rejects invalid lengths before launch; this keeps direct native calls bounded without a host sync.
        output[output_offset + head_dim_index] = __float2half(invalid_sequence_length ? CUDART_NAN_F : 0.0f);
        return;
    }

    __shared__ float running_max_shared;
    __shared__ float normalizer_shared;
    __shared__ float previous_weight_scale_shared;
    __shared__ float current_weight_shared;

    // Online softmax state is scalar per output vector, so thread 0 owns it.
    if (head_dim_index == 0) {
        running_max_shared = -INFINITY;
        normalizer_shared = 0.0f;
    }
    __syncthreads();

    const float query_component = __half2float(query[query_offset + head_dim_index]);
    float output_accumulator = 0.0f;

    for (int token_index = 0; token_index < sequence_length; ++token_index) {
        const long long key_value_index =
            key_value_offset + (static_cast<long long>(token_index) * kHeadDimension + head_dim_index);
        const float score_sum =
            block_reduce_sum(query_component * __half2float(key_cache[key_value_index]));

        if (head_dim_index == 0) {
            const float score = score_sum * softmax_scale;
            if (token_index == 0) {
                previous_weight_scale_shared = 0.0f;
                current_weight_shared = 1.0f;
                normalizer_shared = 1.0f;
                running_max_shared = score;
            } else {
                const float next_running_max = fmaxf(running_max_shared, score);
                // Rescale the running sum when the max changes to keep softmax stable.
                previous_weight_scale_shared = __expf(running_max_shared - next_running_max);
                current_weight_shared = __expf(score - next_running_max);
                normalizer_shared = normalizer_shared * previous_weight_scale_shared + current_weight_shared;
                running_max_shared = next_running_max;
            }
        }
        __syncthreads();

        // Online softmax update: rescale old partial output, then add current value.
        output_accumulator =
            output_accumulator * previous_weight_scale_shared +
            current_weight_shared * __half2float(value_cache[key_value_index]);
    }

    output[output_offset + head_dim_index] = __float2half(output_accumulator / normalizer_shared);
}

void check_contiguous_gqa_decode_attention_contract(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& sequence_lengths) {
    // Native binding can be called directly, so C++ repeats the public contract.
    check_cuda_tensor(query, "query");
    check_cuda_tensor(key_cache, "key_cache");
    check_cuda_tensor(value_cache, "value_cache");
    check_cuda_tensor(sequence_lengths, "sequence_lengths");
    check_contiguous_tensor(query, "query");
    check_contiguous_tensor(key_cache, "key_cache");
    check_contiguous_tensor(value_cache, "value_cache");
    check_contiguous_tensor(sequence_lengths, "sequence_lengths");
    check_fp16_tensor(query, "query");
    check_fp16_tensor(key_cache, "key_cache");
    check_fp16_tensor(value_cache, "value_cache");
    TORCH_CHECK(sequence_lengths.scalar_type() == at::kInt, "sequence_lengths must be int32");

    TORCH_CHECK(query.dim() == 3, "query must have rank 3 [batch, query_heads, head_dim]");
    TORCH_CHECK(key_cache.dim() == 4, "key_cache must have rank 4 [batch, key_value_heads, max_sequence_length, head_dim]");
    TORCH_CHECK(value_cache.dim() == 4, "value_cache must have rank 4 [batch, key_value_heads, max_sequence_length, head_dim]");
    TORCH_CHECK(sequence_lengths.dim() == 1, "sequence_lengths must have rank 1 [batch]");

    const auto batch_size = query.size(0);
    const auto query_heads = query.size(1);
    const auto head_dim = query.size(2);
    const auto key_value_heads = key_cache.size(1);
    const auto max_sequence_length = key_cache.size(2);

    TORCH_CHECK(query_heads > 0, "query_heads must be positive");
    TORCH_CHECK(key_value_heads > 0, "key_value_heads must be positive");
    TORCH_CHECK(head_dim == kHeadDimension, "head_dim must equal 128 for ", kKernelName, " v1");
    TORCH_CHECK(key_cache.device() == query.device(), "key_cache must be on the same CUDA device as query");
    TORCH_CHECK(value_cache.device() == query.device(), "value_cache must be on the same CUDA device as query");
    TORCH_CHECK(sequence_lengths.device() == query.device(), "sequence_lengths must be on the same CUDA device as query");
    TORCH_CHECK(key_cache.size(0) == batch_size, "key_cache batch dimension must match query");
    TORCH_CHECK(value_cache.size(0) == batch_size, "value_cache batch dimension must match query");
    TORCH_CHECK(value_cache.size(1) == key_value_heads, "key_cache and value_cache key_value_heads must match");
    TORCH_CHECK(value_cache.size(2) == max_sequence_length, "key_cache and value_cache max_sequence_length must match");
    TORCH_CHECK(key_cache.size(3) == head_dim, "key_cache head_dim must match query");
    TORCH_CHECK(value_cache.size(3) == head_dim, "value_cache head_dim must match query");
    TORCH_CHECK(sequence_lengths.size(0) == batch_size, "sequence_lengths must have shape [batch]");
    TORCH_CHECK(query_heads % key_value_heads == 0, "query_heads must be divisible by key_value_heads");
}

void check_int_size(const char* name, int64_t dimension_size) {
    TORCH_CHECK(dimension_size >= 0, name, " must be non-negative");
    TORCH_CHECK(dimension_size <= std::numeric_limits<int>::max(), name, " must fit in int32 for ", kKernelName, " v1");
}

void launch_contiguous_gqa_decode_attention(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& sequence_lengths,
    torch::Tensor& output,
    double softmax_scale) {
    check_contiguous_gqa_decode_attention_contract(query, key_cache, value_cache, sequence_lengths);
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    // Device guard fixes device; current stream preserves caller scheduling.
    c10::cuda::CUDAGuard device_guard(query.device());

    check_int_size("batch_size", query.size(0));
    check_int_size("query_heads", query.size(1));
    check_int_size("key_value_heads", key_cache.size(1));
    check_int_size("max_sequence_length", key_cache.size(2));
    TORCH_CHECK(
        query.size(0) * query.size(1) <= std::numeric_limits<int>::max(),
        "batch_size * query_heads must fit in int32 grid dimension for ", kKernelName, " v1");

    const int batch_size = static_cast<int>(query.size(0));
    const int query_heads = static_cast<int>(query.size(1));
    const int key_value_heads = static_cast<int>(key_cache.size(1));
    const int max_sequence_length = static_cast<int>(key_cache.size(2));

    if (batch_size == 0) {
        return;
    }

    const __half* query_ptr = reinterpret_cast<const __half*>(query.data_ptr<at::Half>());
    const __half* key_cache_ptr = reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>());
    const __half* value_cache_ptr = reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>());
    const int32_t* sequence_lengths_ptr = sequence_lengths.data_ptr<int32_t>();
    __half* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

    const dim3 grid(batch_size * query_heads);
    const dim3 block(kThreadsPerBlock);
    const auto stream = at::cuda::getCurrentCUDAStream();
    contiguous_gqa_decode_attention_kernel<<<grid, block, 0, stream>>>(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        sequence_lengths_ptr,
        output_ptr,
        query_heads,
        key_value_heads,
        max_sequence_length,
        static_cast<float>(softmax_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace


torch::Tensor contiguous_gqa_decode_attention_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor sequence_lengths,
    double softmax_scale) {
    auto output = torch::empty_like(query);
    launch_contiguous_gqa_decode_attention(query, key_cache, value_cache, sequence_lengths, output, softmax_scale);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "contiguous_gqa_decode_attention",
        &contiguous_gqa_decode_attention_cuda,
        "contiguous GQA decode attention");
}
