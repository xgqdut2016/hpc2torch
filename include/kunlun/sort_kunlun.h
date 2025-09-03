#include "common_kernel_kunlun.h"

template <typename Tval>
__device__ inline void swap(__local__ Tval &a, __local__ Tval &b)
{
    __local__ Tval tmp = a;
    a = b;
    b = tmp;
}

template <typename Tval, typename Tidx>
__device__ inline void findTopk(
    __global_ptr__ Tval *values,
    __global_ptr__ Tidx *indices,
    int size,
    int topk)
{
    __local__ Tval values_a;
    __local__ Tval values_b;
    __local__ Tidx indices_a;
    __local__ Tidx indices_b;
    for (int i = 0; i < topk; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            GM2LM(values + i, &values_a, sizeof(Tval));
            GM2LM(values + j, &values_b, sizeof(Tval));
            GM2LM(indices + i, &indices_a, sizeof(Tidx));
            GM2LM(indices + j, &indices_b, sizeof(Tidx));
            if constexpr (std::is_same_v<Tval, float>)
            {
                if (values_a < values_b)
                {
                    swap(values_a, values_b);
                    swap(indices_a, indices_b);
                }
            }
            else if constexpr (std::is_same_v<Tval, half>)
            {
                if (__half2float(values_a) < __half2float(values_b))
                {
                    swap(values_a, values_b);
                    swap(indices_a, indices_b);
                }
            }

            // else if constexpr (std::is_same_v<Tval, bfloat16_t>)
            // {
            //     if (__bfloat162float(values_a) < __bfloat162float(values_b))
            //     {
            //         swap(values_a, values_b);
            //         swap(indices_a, indices_b);
            //     }
            // }

            LM2GM(&values_a, values + i, sizeof(Tval));
            LM2GM(&values_b, values + j, sizeof(Tval));
            LM2GM(&indices_a, indices + i, sizeof(Tidx));
            LM2GM(&indices_b, indices + j, sizeof(Tidx));
        }
    }
}

template <typename Tval, typename Tidx>
__device__ inline void findTopkLocal(
    __local__ Tval *values,
    __local__ Tidx *result,
    int size,
    int topk)
{
    for (int i = 0; i < topk; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            if constexpr (std::is_same_v<Tval, float>)
            {
                if (values[i] < values[j])
                {
                    swap(values[i], values[j]);
                    swap(result[i], result[j]);
                }
            }
            else if constexpr (std::is_same_v<Tval, half>)
            {
                if (__half2float(values[i]) < __half2float(values[j]))
                {
                    swap(values[i], values[j]);
                    swap(result[i], result[j]);
                }
            }

            // else if constexpr (std::is_same_v<Tval, bfloat16_t>)
            // {
            //     if (__bfloat162float(values[i]) < __bfloat162float(values[j]))
            //     {
            //         swap(values[i], values[j]);
            //         swap(result[i], result[j]);
            //     }
            // }
        }
    }
}

template <typename Tval, typename Tidx>
__device__ inline void findTopOne(
    __global_ptr__ Tval *values,
    __global_ptr__ Tidx *indices,
    int size)
{
    __local__ Tval values_a = (Tval)(-INFINITY);
    __local__ Tval values_b;
    __local__ Tidx indices_a = 0;
    __local__ Tidx indices_b;
    for (int j = 0; j < size; ++j)
    {
        GM2LM(values + j, &values_b, sizeof(Tval));
        GM2LM(indices + j, &indices_b, sizeof(Tidx));
        if constexpr (std::is_same_v<Tval, float>)
        {
            if (values_a < values_b)
            {
                values_a = values_b;
                indices_a = indices_b;
            }
        }
        else if constexpr (std::is_same_v<Tval, half>)
        {
            if (__half2float(values_a) < __half2float(values_b))
            {
                values_a = values_b;
                indices_a = indices_b;
            }
        }

        // else if constexpr (std::is_same_v<Tval, bfloat16_t>)
        // {
        //     if (__bfloat162float(values_a) < __bfloat162float(values_b))
        //     {
        //         values_a = values_b;
        //         indices_a = indices_b;
        //     }
        // }

        LM2GM(&values_a, values, sizeof(Tval)); // 把最大值存储在0号位置
        LM2GM(&indices_a, indices, sizeof(Tidx));
    }
}

template <typename Tval, typename Tidx>
__device__ inline void findTopOneLocal(
    __local__ Tval *values,
    __local__ Tidx *result,
    int size)
{
    __local__ Tval values_a = (Tval)(-INFINITY);
    __local__ Tidx indices_a = 0;
    for (int j = 0; j < size; ++j)
    {
        if constexpr (std::is_same_v<Tval, float>)
        {
            if (values_a < values[j])
            {
                values_a = values[j];
                indices_a = result[j];
            }
        }
        else if constexpr (std::is_same_v<Tval, half>)
        {
            if (__half2float(values_a) < __half2float(values[j]))
            {
                values_a = values[j];
                indices_a = result[j];
            }
        }

        // else if constexpr (std::is_same_v<Tval, bfloat16_t>)
        // {
        //     if (__bfloat162float(values_a) < __bfloat162float(values[j]))
        //     {
        //         values_a = values[j];
        //         indices_a = result[j];
        //     }
        // }
    }
    values[0] = values_a;
    result[0] = indices_a;
}
template <typename Tval, typename Tidx>
__device__ inline void TopkKernel(__global_ptr__ Tval *values,
                                  __global_ptr__ Tidx *indices,
                                  __global_ptr__ Tidx *indices_global, // 长度为cluster_num() * core_num() * topk
                                  __global_ptr__ Tval *values_global,  // 把长度为voc的values的前topk元素集中倒values_global
                                  __local__ Tval *values_local,
                                  __local__ Tidx *indices_local,
                                  int voc,
                                  int topk,
                                  int buf_size)
{
    int cid = core_id();
    if (cid >= core_num())
    {
        return;
    }
    int thread_id = core_num() * cluster_id() + cid;
    int nthreads = core_num() * cluster_num();

    // 每个coreId分配step个元素
    int remain = voc % nthreads;
    int step_easy = (voc - remain) / nthreads;
    int step_hard = step_easy + 1;
    int step = (thread_id < remain ? step_hard : step_easy);
    int ind_start = (thread_id < remain ? thread_id * step_hard : remain * step_hard + (thread_id - remain) * step_easy);
    for (int index = ind_start; index < ind_start + step; index++)
    {
        indices[index] = index;
    }

    for (int i = 0; i < 2 * buf_size; i++)
    {
        values_local[i] = (Tval)(-INFINITY);
        indices_local[i] = 0;
    }

    int remainTask = step % buf_size;
    int repeat = (step - remainTask) / buf_size;
    if (topk >= step_easy)
    {
        if (thread_id == 0)
        {
            findTopk(values, indices, voc, topk);
        }
        sync_cluster();
        for (int index = thread_id; index < topk; index += nthreads)
        {
            GM2LM(values + index, values_local, sizeof(Tval));
            GM2LM(indices + index, indices_local, sizeof(Tidx));
            LM2GM(values_local, values_global + index, sizeof(Tval));
            LM2GM(indices_local, indices_global + index, sizeof(Tidx));
        }
        sync_cluster();
    }
    else
    { // topk < step_easy
        if (buf_size > step_easy)
        { // buf_size >= step_hard > step_easy > topk
            GM2LM(values + ind_start, values_local, step * sizeof(Tval));
            GM2LM(indices + ind_start, indices_local, step * sizeof(Tidx));
            findTopkLocal(values_local, indices_local, step, topk);
            LM2GM(values_local, values_global + thread_id * topk, topk * sizeof(Tval)); // values_global前面nthreads * topk存储不同core的topk元素
            LM2GM(indices_local, indices_global + thread_id * topk, topk * sizeof(Tidx));
        }
        else
        { // buf_size <= step_easy
            if (topk > buf_size)
            { // step_easy > topk > buf_size

                findTopk(&values[ind_start], &indices[ind_start], step, topk);

                for (int r = 0; r < topk / buf_size + (topk % buf_size > 0 ? 1 : 0); r++)
                {
                    int read_len = (r < topk / buf_size ? buf_size : topk % buf_size);
                    GM2LM(values + ind_start + r * buf_size, values_local, read_len * sizeof(Tval));
                    GM2LM(indices + ind_start + r * buf_size, indices_local, read_len * sizeof(Tidx));
                    LM2GM(values_local, values_global + thread_id * topk + r * buf_size, read_len * sizeof(Tval));
                    LM2GM(indices_local, indices_global + thread_id * topk + r * buf_size, read_len * sizeof(Tidx));
                }
            }
            else
            { // step_easy >= buf_size >= topk

                for (int r = 0; r < repeat; r++)
                {
                    GM2LM(values + ind_start + r * buf_size, values_local, buf_size * sizeof(Tval));
                    GM2LM(indices + ind_start + r * buf_size, indices_local, buf_size * sizeof(Tidx));
                    findTopkLocal(values_local, indices_local, buf_size + topk, topk); // 每次循环把上次的前topk也加入对比
                    for (int i = buf_size; i < buf_size + topk; i++)
                    { // 把上一轮循环的topk加载到后半部分
                        values_local[i] = values_local[i - buf_size];
                        indices_local[i] = indices_local[i - buf_size];
                    }
                }
                if (remainTask)
                {
                    // 此时repeat一定大于0，且values_local[buf_size:buf_size + topk]存储上次的前topk数据
                    for (int i = 0; i < topk; i++)
                    {
                        values_local[i] = values_local[i + buf_size];
                        indices_local[i] = indices_local[i + buf_size];
                    }
                    GM2LM(values + ind_start + repeat * buf_size, values_local + topk, remainTask * sizeof(Tval));
                    GM2LM(indices + ind_start + repeat * buf_size, indices_local + topk, remainTask * sizeof(Tidx));
                    findTopkLocal(values_local, indices_local, remainTask + topk, topk);
                }
                LM2GM(values_local, values_global + thread_id * topk, topk * sizeof(Tval));
                LM2GM(indices_local, indices_global + thread_id * topk, topk * sizeof(Tidx));
            }
        }
        if (thread_id == 0)
        {
            findTopk(values_global, indices_global, nthreads * topk, topk);
        }
    }
}
template <typename Tval, typename Tidx>
__device__ inline void TopOneKernel(__global_ptr__ Tidx *result,
                                    __global_ptr__ Tval *values,
                                    __global_ptr__ Tidx *indices,
                                    __global_ptr__ Tidx *indices_global,
                                    __global_ptr__ Tval *values_global,
                                    __local__ Tval *values_local,
                                    __local__ Tidx *indices_local,
                                    int voc,
                                    int buf_size)
{
    int cid = core_id();
    if (cid >= core_num())
    {
        return;
    }
    int thread_id = core_num() * cluster_id() + cid;
    int nthreads = core_num() * cluster_num();

    // 每个coreId分配step个元素
    int remain = voc % nthreads;
    int step_easy = (voc - remain) / nthreads;
    int step_hard = step_easy + 1;
    int step = (thread_id < remain ? step_hard : step_easy);
    int ind_start = (thread_id < remain ? thread_id * step_hard : remain * step_hard + (thread_id - remain) * step_easy);
    for (int index = ind_start; index < ind_start + step; index++)
    {
        indices[index] = index;
    }

    for (int i = 0; i < 2 * buf_size; i++)
    {
        values_local[i] = (Tval)(-INFINITY);
        indices_local[i] = 0;
    }

    int remainTask = step % buf_size;
    int repeat = (step - remainTask) / buf_size;
    if (buf_size > step_easy)
    { // buf_size >= step_hard > step_easy
        GM2LM(values + ind_start, values_local, step * sizeof(Tval));
        GM2LM(indices + ind_start, indices_local, step * sizeof(Tidx));
        findTopOneLocal(values_local, indices_local, step);
        LM2GM(values_local, values_global + thread_id, sizeof(Tval));
        LM2GM(indices_local, indices_global + thread_id, sizeof(Tidx));
    }
    else
    { // buf_size <= step_easy
        for (int r = 0; r < repeat; r++)
        {
            GM2LM(values + ind_start + r * buf_size, values_local, buf_size * sizeof(Tval));
            GM2LM(indices + ind_start + r * buf_size, indices_local, buf_size * sizeof(Tidx));
            findTopOneLocal(values_local, indices_local, buf_size + 1);
            values_local[buf_size] = values_local[0];
            indices_local[buf_size] = indices_local[0];
        }
        if (remainTask)
        {
            GM2LM(values + ind_start + repeat * buf_size, values_local, remainTask * sizeof(Tval));
            GM2LM(indices + ind_start + repeat * buf_size, indices_local, remainTask * sizeof(Tidx));
            // 此时repeat一定大于0
            values_local[remainTask] = values_local[buf_size];
            indices_local[remainTask] = indices_local[buf_size];
            findTopOneLocal(values_local, indices_local, remainTask + 1);
        }
        LM2GM(values_local, values_global + thread_id, sizeof(Tval));
        LM2GM(indices_local, indices_global + thread_id, sizeof(Tidx));
    }
    if (thread_id == 0)
    {
        findTopOne(values_global, indices_global, nthreads);
        result[0] = indices_global[0];
    }
}