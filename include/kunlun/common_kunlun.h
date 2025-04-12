#ifndef __INFINIOP_COMMON_KUNLUN_H__
#define __INFINIOP_COMMON_KUNLUN_H__

// This header file will only be include by .xpu file
#include "kunlun_type.h"
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_io.h"
#include "xpu/kernel/xtdk_math.h"
#include "xpu/kernel/xtdk_simd.h"
#include "xpu/runtime.h"
#include <stdio.h>
#include <utility>
#if !defined(__xpu__) || defined(__xpu_on_host__)
#include_next <assert.h>
#else
#define assert(x)
#endif

inline __device__ kunlun_ptrdiff_t indexToReducedOffset(
    kunlun_ptrdiff_t flat_index,
    kunlun_size_t ndim,
    _global_ptr_ kunlun_ptrdiff_t *broadcasted_strides,
    _global_ptr_ kunlun_ptrdiff_t *target_strides)
{
    kunlun_ptrdiff_t res = 0;

    __local__ kunlun_ptrdiff_t a[8];
    __local__ kunlun_ptrdiff_t b[8];

    for (kunlun_size_t i = 0; i < ndim; ++i)
    {
        GM2LM(broadcasted_strides + i, a + i, 1 * sizeof(kunlun_ptrdiff_t));
        GM2LM(target_strides + i, b + i, 1 * sizeof(kunlun_ptrdiff_t));
        res += flat_index / a[i] * b[i];
        flat_index %= a[i];
        mfence();
    }
    return res;
}

inline __device__ kunlun_ptrdiff_t indexToOffset(
    kunlun_ptrdiff_t flat_index,
    kunlun_size_t ndim,
    _global_ptr_ kunlun_size_t *shape,
    _global_ptr_ kunlun_ptrdiff_t *strides)
{
    kunlun_ptrdiff_t res = 0;

    __local__ kunlun_ptrdiff_t b[8];
    __local__ kunlun_size_t c[8];
    // printf("%d\n", flat_index);

    for (int i = ndim - 1; i >= 0; i--)
    {
        GM2LM(shape + i, c + i, 1 * sizeof(kunlun_size_t));
        GM2LM(strides + i, b + i, 1 * sizeof(kunlun_ptrdiff_t));
        // printf("%d ", flat_index % c[i]);
        res += (flat_index % c[i]) * b[i];
        flat_index /= c[i];
        mfence();
    }
    // printf("\n");
    // printf("res:%d\n", res);
    // printf("shape\n");
    // for (int i = ndim - 1; i >= 0; i--)
    // {
    //     GM2LM(shape + i, c + i, 1 * sizeof(kunlun_size_t));
    //     GM2LM(strides + i, b + i, 1 * sizeof(kunlun_ptrdiff_t));
    //     printf("%d ", c[i]);
    // }
    // printf("\n");
    // printf("strides\n");
    // for (int i = ndim - 1; i >= 0; i--)
    // {
    //     GM2LM(shape + i, c + i, 1 * sizeof(kunlun_size_t));
    //     GM2LM(strides + i, b + i, 1 * sizeof(kunlun_ptrdiff_t));
    //     printf("%d ", b[i]);
    // }
    // printf("\n");
    // printf("------------------\n");
    return res;
}

inline __device__ kunlun_ptrdiff_t getPaddedSize(
    kunlun_size_t ndim,
    _global_ptr_ kunlun_size_t *shape,
    _global_ptr_ kunlun_ptrdiff_t *pads)
{
    kunlun_ptrdiff_t total_size = 1;

    __local__ kunlun_size_t c[8];
    __local__ kunlun_ptrdiff_t d[8];
    for (kunlun_size_t i = 0; i < ndim; ++i)
    {
        GM2LM(shape + i, c + i, 1 * sizeof(kunlun_size_t));
        GM2LM(pads + i, d + i, 1 * sizeof(kunlun_ptrdiff_t));

        total_size *= c[i] + (i < 2 ? 0 : 2 * d[i - 2]);
        mfence();
    }
    return total_size;
}
inline void broadcast_shapes(const kunlun_size_t *a_shape, int a_dims,
                             const kunlun_size_t *b_shape, int b_dims,
                             const kunlun_size_t *c_shape, int c_dims,
                             const kunlun_ptrdiff_t *a_strides, const kunlun_ptrdiff_t *b_strides,
                             kunlun_ptrdiff_t *new_a_strides, kunlun_ptrdiff_t *new_b_strides,
                             kunlun_size_t *new_a_shape, kunlun_size_t *new_b_shape)
{
    int offset_a = c_dims - a_dims;
    int offset_b = c_dims - b_dims;

    for (int i = 0; i < c_dims; ++i)
    {
        new_a_shape[i] = (i - offset_a >= 0) ? a_shape[i - offset_a] : 1;
        new_b_shape[i] = (i - offset_b >= 0) ? b_shape[i - offset_b] : 1;

        new_a_strides[i] = 1;
        new_b_strides[i] = 1;
        // 验证是否可广播（可略去，假设总是合法）
        if ((new_a_shape[i] != c_shape[i] && new_a_shape[i] != 1) || (new_b_shape[i] != c_shape[i] && new_b_shape[i] != 1))
        {
            printf("Shapes cannot be broadcast at dimension %d!\n", i);
            return;
        }
    }
    for (int i = c_dims - 2; i >= 0; i--)
    {

        new_a_strides[i] = new_a_shape[i + 1] * new_a_strides[i + 1];
        new_b_strides[i] = new_b_shape[i + 1] * new_b_strides[i + 1];
    }
}
#endif
