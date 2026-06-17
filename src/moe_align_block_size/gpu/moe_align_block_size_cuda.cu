#include "kernel.cuh"

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
template <typename scalar_t>
void moe_align_block_size(int32_t *sorted_token_ids,
                          int32_t *experts_ids,
                          int32_t *num_tokens_post_pad,
                          int32_t *expert_map,
                          const scalar_t *topk_ids,
                          int64_t num_experts,
                          int64_t block_size,
                          int64_t num_topk_ids,
                          int32_t sorted_token_ids_size_0, int32_t topk_ids_size_1,
                          cudaStream_t stream)
{
    int64_t padded_num_experts =
        ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int experts_per_warp = WARP_SIZE;
    int threads = 1024;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // BlockScan uses 1024 threads and assigns one thread per expert.
    RUNTIME_CHECK(padded_num_experts < 1024,
                  "padded_num_experts must be less than 1024");

    bool has_expert_map = (expert_map != nullptr);

    // calc needed amount of shared mem for `cumsum` tensors
    bool small_batch_expert_mode =
        (num_topk_ids < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode)
    {
        const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
        const int32_t shared_mem_size =
            ((threads + 1) * num_experts + (num_experts + 1)) *
            sizeof(int32_t);

        // threadIdx.x >= fill_threads: counting experts and aligning
        // threadIdx.x < fill_threads: filling sorted_token_ids
        constexpr int32_t fill_threads = 256;
        auto small_batch_expert_kernel =
            vllm::moe::moe_align_block_size_small_batch_expert_kernel<
                scalar_t, fill_threads>;

        small_batch_expert_kernel<<<1, fill_threads + threads,
                                    shared_mem_size, stream>>>(
            topk_ids,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            expert_map, num_experts, block_size,
            num_topk_ids, sorted_token_ids_size_0, topk_ids_size_1,
            has_expert_map);
    }
    else
    {
        int32_t *cumsum_buffer;
        cudaMalloc(&cumsum_buffer, (num_experts + 1) * sizeof(int32_t));
        auto align_kernel = vllm::moe::moe_align_block_size_kernel<scalar_t>;

        size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
        size_t shared_mem_size =
            num_warps * experts_per_warp * sizeof(int32_t);

        // launch two threadblocks
        // blockIdx.x == 0: counting experts and aligning
        // blockIdx.x == 1: filling sorted_token_ids
        align_kernel<<<2, threads, shared_mem_size, stream>>>(
            topk_ids,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            expert_map, num_experts, padded_num_experts,
            experts_per_warp, block_size, num_topk_ids,
            cumsum_buffer, sorted_token_ids_size_0,
            topk_ids_size_1, has_expert_map);

        const int block_threads = std::min(256, (int)threads);
        const int num_blocks =
            (num_topk_ids + block_threads - 1) / block_threads;
        const int max_blocks = 65535;
        const int actual_blocks = std::min(num_blocks, max_blocks);
        dim3 gridDims(1, actual_blocks);

        auto sort_kernel =
            vllm::moe::count_and_sort_expert_tokens_kernel<scalar_t>;
        sort_kernel<<<gridDims, block_threads, 0, stream>>>(
            topk_ids,
            sorted_token_ids,
            cumsum_buffer, expert_map,
            num_topk_ids, num_experts, sorted_token_ids_size_0,
            topk_ids_size_1, has_expert_map);
        cudaFree(cumsum_buffer);
    };
}

enum DataType
{
    DT_INT32 = 0,
    DT_INT64 = 1,
};

extern "C" void moe_align_block_size_nv(void *sorted_token_ids,
                                        void *experts_ids,
                                        void *num_tokens_post_pad,
                                        void *expert_map,
                                        const void *topk_ids,
                                        int32_t num_experts,
                                        int32_t block_size,
                                        int32_t num_topk_ids,
                                        int32_t sorted_token_ids_size_0, int32_t topk_ids_size_1,
                                        int dataType)
{
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }

    switch (static_cast<DataType>(dataType))
    {
    case DT_INT32:
        moe_align_block_size<int32_t>(
            (int32_t *)sorted_token_ids,
            (int32_t *)experts_ids,
            (int32_t *)num_tokens_post_pad,
            (int32_t *)expert_map,
            (const int32_t *)topk_ids,
            static_cast<int64_t>(num_experts),
            static_cast<int64_t>(block_size),
            static_cast<int64_t>(num_topk_ids),
            sorted_token_ids_size_0, topk_ids_size_1,
            stream);
        break;
    case DT_INT64:
        moe_align_block_size<int64_t>(
            (int32_t *)sorted_token_ids,
            (int32_t *)experts_ids,
            (int32_t *)num_tokens_post_pad,
            (int32_t *)expert_map,
            (const int64_t *)topk_ids,
            static_cast<int64_t>(num_experts),
            static_cast<int64_t>(block_size),
            static_cast<int64_t>(num_topk_ids),
            sorted_token_ids_size_0, topk_ids_size_1,
            stream);
        break;
    default:
        printf("Unsupported datatype\n");
        break;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}
