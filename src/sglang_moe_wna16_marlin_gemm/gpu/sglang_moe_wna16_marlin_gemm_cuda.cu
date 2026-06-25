#if defined(ENABLE_NVIDIA_API) && defined(ENABLE_TVM_API)

#include "kernel.cuh"

template <typename scalar_t>
void sglang_moe_wna16_marlin_gemm(
    void *c,
    const void *a,
    const void *b_q_weight,
    void *b_bias,
    void *b_scales,
    void *global_scales,
    void *b_zeros,
    void *g_idx,
    void *perm,
    void *sorted_token_ids,
    void *expert_ids,
    void *num_tokens_post_padded,
    void *topk_weights,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    bool is_ep,
    int64_t b_q_type_id,
    int size_m,
    int size_n,
    int size_k,
    bool has_act_order,
    bool has_bias,
    bool is_k_full,
    bool has_zp,
    int num_groups,
    int group_size,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    int sorted_token_ids_size_0,
    int b_q_weight_size_1,
    int b_q_weight_size_2,
    int c_size_0,
    cudaStream_t stream)
{
    using namespace host;

    ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
    int pack_factor = 32 / b_q_type.size_bits();

    if (moe_block_size != 8)
    {
        if (moe_block_size % 16 != 0)
        {
            printf("unsupported moe_block_size=%d\n", moe_block_size);
            return;
        }
        if (moe_block_size < 16 || moe_block_size > 64)
        {
            printf("unsupported moe_block_size=%d\n", moe_block_size);
            return;
        }
    }
    if (size_k % device::marlin::tile_size != 0 || (size_k / device::marlin::tile_size) != b_q_weight_size_1 || b_q_weight_size_2 % device::marlin::tile_size != 0)
    {
        return;
    }
    int actual_size_n = (b_q_weight_size_2 / device::marlin::tile_size) * pack_factor;
    if (actual_size_n != size_n)
    {
        printf("size_n =%d, actual_size_n = %d\n", size_n, actual_size_n);
        return;
    }
    if (c_size_0 != size_m * top_k)
    {
        printf("Shape mismatch: c.size(0) = %d, top_k * size_m = %d\n", c_size_0, size_m * top_k);
        return;
    }

    // Verify device and strides

    // thread_k, thread_n, sms
    int thread_k = -1;
    int thread_n = -1;
    int sms = -1;

    int device_id = 0;

    RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id));

    // Verify global_scale (Optional unwrap done in Python)
    // int64_t global_scale_size = global_scale.size(0);
    if (global_scales != nullptr)
    {
        RuntimeCheck(b_q_type == kFE2M1f && group_size == 16, "global_scales can only be used for nvfp4 format.");
    }
    else
    {
        RuntimeCheck(
            !(b_q_type == kFE2M1f && group_size == 16), "the global_scales parameter must be passed for nvfp4 format.");
    }

    // b_zeros Optional unwrap + has_zp derivation: SKIP (done in Python)

    // Verify b_q_type vs has_zp
    if (has_zp)
    {

        RuntimeCheck(
            b_q_type == kU4 || b_q_type == kU8, "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());
    }
    else
    {
        RuntimeCheck(
            b_q_type == kU4B8 || b_q_type == kU8B128 || b_q_type == kFE4M3fn || b_q_type == kFE2M1f,
            "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or "
            "float4_e2m1f when "
            "has_zp = False. Got = ",
            b_q_type.str());
    }

    if (has_zp && is_zp_float)
    {
        RuntimeCheck(
            std::is_same<scalar_t, fp16_t>::value,
            "Computation type must be float16 (half) when using float zero "
            "points.");
    }

    if (b_q_type == kFE2M1f)
    {
        RuntimeCheck(
            group_size == 16 || group_size == 32,
            "float4_e2m1f only supports group_size == 16 (NVFP4) or group_size == 32 (MXFP4). Got group_size = ",
            group_size);
        RuntimeCheck(
            group_size != 32 || std::is_same<scalar_t, nv_bfloat16>::value,
            "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 activations.");
    }

    // Verify b_zeros
    // if (has_zp)
    // {
    //     // b_zeros.ndim = 3

    //     if (is_zp_float)
    //     {
    //         RuntimeCheck(b_zeros_size_2 == size_n, "b_zeros dim 2 = ", b_zeros_size_2, " is not size_n = ", size_n);
    //         RuntimeCheck(
    //             num_groups == b_zeros_size_1, "b_zeros dim 1 = ", b_zeros_size_1, " is not num_groups = ", num_groups);
    //         RuntimeCheck(num_groups != -1, "num_groups must be != -1");
    //     }
    //     else
    //     {
    //         RuntimeCheck(
    //             b_zeros_size_1 == num_groups, "b_zeros dim 1 = ", b_zeros_size_1, " is not num_groups = ", num_groups);
    //         RuntimeCheck(
    //             b_zeros_size_2 == size_n / pack_factor,
    //             "b_zeros dim 2 = ",
    //             b_zeros_size_2,
    //             " is not size_n / pack_factor = ",
    //             size_n / pack_factor);
    //     }
    // }

    // Alloc C tmp buffer that is going to be used for the global reduce
    if (size_n % device::marlin::min_thread_n != 0)
    {
        return;
    }
    int c_tmp_bytes = 0;
    if (use_fp32_reduce && !use_atomic_add)
    {
        // max num of threadblocks is sms * 4
        int max_c_tmp_size = min(
            size_n * sorted_token_ids_size_0,
            sms * 4 * moe_block_size * device::marlin::max_thread_n);
        if (moe_block_size == 8)
        {
            max_c_tmp_size *= 2;
        }

        c_tmp_bytes = max_c_tmp_size * sizeof(float);
    }

    int a_tmp_bytes = 0;
    if (has_act_order)
    {
        a_tmp_bytes = size_m * top_k * size_k * sizeof(scalar_t);
        if (is_k_full)
        {
            RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");
            RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k,
                         ", is not divisible by num_groups = ", num_groups);
            group_size = size_k / num_groups;
        }
        else
        {
            group_size = 0;
        }
    }
    else
    {

        if (num_groups > 1)
        {
            // RuntimeCheck(
            //     size_k % num_groups == 0, "size_k = ", size_k,
            //     ", is not divisible by b_scales.size(1) = ", b_scales_size_1);
            group_size = size_k / num_groups;
        }
        else
        {
            group_size = -1;
        }
    }

    int64_t max_n_tiles = size_n / device::marlin::min_thread_n;
    int64_t min_workspace_size = std::min(max_n_tiles * (sorted_token_ids_size_0 / moe_block_size), static_cast<int64_t>(sms) * 4);
    const int total_bytes = c_tmp_bytes + a_tmp_bytes + min_workspace_size;
    void *total_buffer;
    if (total_bytes > 0)
    {
        cudaMalloc(&total_buffer, total_bytes);
    }

    void *workspace = nullptr;
    void *c_tmp = nullptr;
    void *a_tmp = nullptr;

    uint8_t *ptr = reinterpret_cast<uint8_t *>(total_buffer);
    if (use_fp32_reduce && !use_atomic_add)
    {
        c_tmp = reinterpret_cast<float *>(ptr);
        ptr += c_tmp_bytes;
    }
    if (has_act_order && a_tmp_bytes > 0)
    {
        a_tmp = ptr;
        ptr += a_tmp_bytes;
    }
    if (min_workspace_size > 0)
    {
        workspace = ptr;
        cudaMemset(workspace, 0, min_workspace_size);
        ptr += min_workspace_size;
    }

    // Early return for zero-size M (moved after all validation)
    if (size_m == 0)
    {
        return;
    }

    device::marlin_moe::marlin_mm<scalar_t>(
        a,
        b_q_weight,
        c,
        c_tmp,
        b_bias,
        b_scales,
        global_scales,
        b_zeros,
        g_idx,
        perm,
        a_tmp,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        size_m,
        size_n,
        size_k,
        workspace,
        b_q_type,
        has_bias,
        has_act_order,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        device_id,
        stream,
        thread_k,
        thread_n,
        sms,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float);
    cudaFree(total_buffer);
}

enum DataType
{
    DT_FLOAT16 = 0,
    DT_BFLOAT16 = 1,
};

extern "C" void sglang_moe_wna16_marlin_gemm_nv(
    void *c,
    const void *a,
    const void *b_q_weight,
    void *b_bias,
    void *b_scales,
    void *global_scales,
    void *b_zeros,
    void *g_idx,
    void *perm,
    void *sorted_token_ids,
    void *expert_ids,
    void *num_tokens_post_padded,
    void *topk_weights,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    bool is_ep,
    int64_t b_q_type_id,
    int size_m,
    int size_n,
    int size_k,
    bool has_act_order,
    bool has_bias,
    bool is_k_full,
    bool has_zp,
    int num_groups,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    int sorted_token_ids_size_0,
    int b_q_weight_size_1,
    int b_q_weight_size_2,
    int c_size_0,
    int dataType)
{

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }
    int group_size = 0;
    if (g_idx != nullptr && perm != nullptr)
    {
        if (is_k_full)
        {
            group_size = size_k / num_groups;
        }
        else
        {
            group_size = 0;
        }
    }
    else
    {
        if (num_groups > 1)
        {
            group_size = size_k / num_groups;
        }
        else
        {
            group_size = -1;
        }
    }
    switch (static_cast<DataType>(dataType))
    {
    case DT_FLOAT16:
        sglang_moe_wna16_marlin_gemm<__half>(
            c,
            a,
            b_q_weight,
            b_bias,
            b_scales,
            global_scales,
            b_zeros,
            g_idx,
            perm,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size,
            top_k,
            mul_topk_weights,
            is_ep,
            b_q_type_id,
            size_m,
            size_n,
            size_k,
            has_act_order,
            has_bias,
            is_k_full,
            has_zp,
            num_groups,
            group_size,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
            sorted_token_ids_size_0,
            b_q_weight_size_1,
            b_q_weight_size_2,
            c_size_0,
            stream);
        break;
    case DT_BFLOAT16:
        sglang_moe_wna16_marlin_gemm<__nv_bfloat16>(
            c,
            a,
            b_q_weight,
            b_bias,
            b_scales,
            global_scales,
            b_zeros,
            g_idx,
            perm,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size,
            top_k,
            mul_topk_weights,
            is_ep,
            b_q_type_id,
            size_m,
            size_n,
            size_k,
            has_act_order,
            has_bias,
            is_k_full,
            has_zp,
            num_groups,
            group_size,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
            sorted_token_ids_size_0,
            b_q_weight_size_1,
            b_q_weight_size_2,
            c_size_0,
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

#endif
