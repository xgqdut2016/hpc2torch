#include "common_kernel_kunlun.h"

// Sum(x^2) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ inline Tcompute sumSquared(__shared_ptr__ const Tdata *data_ptr, size_t count)
{
    Tcompute ss = 0;

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE)
    {
        Tdata xi = data_ptr[i];
        ss += to<Tcompute>(xi) * to<Tcompute>(xi);
    }

    __shared__ Tcompute temp_storage;
    if (core_id() == 0)
    {
        temp_storage = to<Tcompute>(0.f);
    }
    sync_cluster();

    atomicAdd(&temp_storage, ss);
    sync_cluster();

    return temp_storage;
}

// Sum(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ inline Tcompute sum(__shared_ptr__ const Tdata *data_ptr, size_t count)
{
    Tcompute ss = 0;

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE)
    {
        Tdata xi = data_ptr[i];
        ss += to<Tcompute>(xi);
    }

    __shared__ Tcompute temp_storage;
    if (core_id() == 0)
    {
        temp_storage = to<Tcompute>(0.f);
    }
    sync_cluster();

    atomicAdd(&temp_storage, ss);
    sync_cluster();

    return temp_storage;
}

// Max(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ inline Tdata max(__shared_ptr__ const Tdata *data_ptr, size_t count)
{
    Tdata max_val = data_ptr[0];

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE)
    {
        Tdata xi = data_ptr[i];
        max_val = fmax(max_val, to<Tdata>(xi));
    }

    __shared__ Tdata temp_storage;
    if (core_id() == 0)
    {
        temp_storage = data_ptr[0];
    }
    sync_cluster();

    atomicMax(&temp_storage, max_val);
    sync_cluster();

    return temp_storage;
}
template <unsigned int CLUSTER_SIZE, unsigned int BLOCK_SIZE, typename Tval, typename Tcompute>
__device__ inline Tcompute softmaxSum(__global_ptr__ const Tval *probs,
                                      Tval max_value,
                                      __shared_ptr__ Tval *x_sm,
                                      __shared_ptr__ Tval *y_sm,
                                      float temperature,
                                      int voc,
                                      __global_ptr__ Tcompute *sum_global)
{

    int sm_size = SM_SIZE / sizeof(Tval);
    int all_sm_size = cluster_num() * sm_size;
    int sm_remain = voc % all_sm_size;
    int sm_repeat = (voc - sm_remain) / all_sm_size;
    int sm_remain_cluster = sm_remain % cluster_num();
    int sm_step_easy = (sm_remain - sm_remain_cluster) / cluster_num();
    int sm_step_hard = sm_step_easy + 1;
    int sm_step = (cluster_id() < sm_remain_cluster ? sm_step_hard : sm_step_easy);
    int sm_ind_start = (cluster_id() < sm_remain_cluster ? cluster_id() * sm_step_hard : sm_remain_cluster * sm_step_hard + (cluster_id() - sm_remain_cluster) * sm_step_easy);

    __shared__ Tcompute sum_;
    if (core_id() == 0)
    {
        sum_ = to<Tcompute>(0.f);
    }
    sync_cluster();

    //__global_ptr__ Tval const *probs_ = probs;

    for (int r = 0; r < sm_repeat + (sm_step > 0 ? 1 : 0); r++)
    {
        int read_len = (r < sm_repeat ? sm_size : sm_step);
        int start = (r < sm_repeat ? r * all_sm_size + cluster_id() * sm_size : sm_repeat * all_sm_size + sm_ind_start);
        if (core_id() == 0)
        {
            GM2SM(probs + start, x_sm, read_len * sizeof(Tval));
        }
        sync_cluster();

        for (int index = core_id(); index < read_len; index += BLOCK_SIZE)
        {
            if constexpr (std::is_same_v<Tval, half>)
            {
                y_sm[index] = __float2half(exp((__half2float(x_sm[index]) - to<float>(max_value)) / temperature));
            }
            // else if constexpr (std::is_same_v<Tval, bfloat16_t>)
            // {
            //     y_sm[index] = __float2bfloat16(exp((__bfloat162float(x_sm[index]) - to<float>(max_value)) / temperature));
            // }
            else if constexpr (std::is_same_v<Tval, float>)
            {
                y_sm[index] = exp((x_sm[index] - max_value) / temperature);
            }
        }
        sync_cluster();

        Tcompute sum_0 = sum<BLOCK_SIZE, Tval, Tcompute>(y_sm, read_len);

        if (core_id() == 0)
        {
            sum_ = sum_ + sum_0;
        }
        sync_cluster();
    }

    __global_ptr__ Tcompute *sum_global_ = sum_global;
    if (core_id() == 0)
    {
        SM2GM(&sum_, sum_global_ + cluster_id(), sizeof(Tcompute));
    }
    sync_cluster();

    __shared__ Tcompute all_sum;
    __shared__ Tcompute z_sm[CLUSTER_SIZE];
    if (core_id() == 0)
    {
        GM2SM(sum_global_, z_sm, cluster_num() * sizeof(Tcompute));
    }
    sync_cluster();

    Tcompute all_sum_0 = sum<BLOCK_SIZE, Tcompute, Tcompute>(z_sm, cluster_num());
    if (core_id() == 0)
    {
        all_sum = all_sum_0;
    }
    sync_cluster();

    return all_sum;
}