#pragma once
#include <iostream>
#include <sstream>
#include <cstdlib>

template <typename... Args>
inline void RuntimeCheckImpl(bool cond,
                             const char *file,
                             int line,
                             Args &&...args)
{
    if (!cond)
    {
        std::ostringstream oss;
        (oss << ... << args);

        std::cerr << "RuntimeCheck failed at "
                  << file << ":" << line << "\n"
                  << "  Reason: " << oss.str() << std::endl;
        std::abort();
    }
}

#define RUNTIME_CHECK(cond, ...) RuntimeCheckImpl((cond), __FILE__, __LINE__, __VA_ARGS__)

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return max(a, b);
    }
};
template <typename T>
struct MinOp
{
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return a < b ? a : b;
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width>
__inline__ __device__ T WarpAllReduce(T val)
{
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
