#include "acl/acl.h"
// #include "aclnnop/level2/aclnn_avgpool2d.h"
// #include "aclnnop/level2/aclnn_max_pool.h"
#include "aclnnop/aclnn_avgpool2d.h"
// #include "aclnnop/aclnn_avgpool3d.h"
#include "aclnnop/aclnn_max_pool.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

struct PoolingMode
{
    enum Mode
    {
        Avg,
        Max,
        Count, ///< Number of unary operation types (marker for counting purposes).
    };

    static const size_t numPoolingMode = Count;
};
template <typename T>
void poolingAclnnDevice(void *input, void *output,
                        int *windows, int *pads, int *strides, int *dilations,
                        int *x_shape, int *y_shape,
                        PoolingMode::Mode mode,
                        bool ceil_mode, int nDim,
                        aclrtStream &stream)
{
    aclDataType dataType;
    if (sizeof(T) == 2)
    {
        dataType = aclDataType::ACL_FLOAT16;
    }
    else if (sizeof(T) == 4)
    {
        dataType = aclDataType::ACL_FLOAT;
    }
    aclFormat format;
    if (nDim == 3)
    {
        format = aclFormat::ACL_FORMAT_NCL;
    }
    else if (nDim == 4)
    {
        format = aclFormat::ACL_FORMAT_NCHW;
    }
    // else if (nDim == 5)
    // {
    //     format = aclFormat::ACL_FORMAT_ND; // 5维向量用ND
    // }

    std::vector<int64_t> inputDim(nDim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(nDim, 1); // 初始化为1
    std::vector<int64_t> outputDim(nDim);
    std::vector<int64_t> outputStride(nDim, 1);

    for (int i = nDim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(x_shape[i]);
        outputDim[i] = int64_t(y_shape[i]);
        if (i < nDim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    std::vector<int64_t> ksize(nDim - 2);
    std::vector<int64_t> stride(nDim - 2);
    std::vector<int64_t> pad(nDim - 2);
    std::vector<int64_t> dilation(nDim - 2);
    for (int i = 0; i < nDim - 2; i++)
    {
        ksize[i] = int64_t(windows[i]);
        stride[i] = int64_t(strides[i]);
        pad[i] = int64_t(pads[i]);
        dilation[i] = int64_t(dilations[i]);
    }
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);

    aclIntArray *kernelSizeArray = aclCreateIntArray(ksize.data(), ksize.size());
    aclIntArray *stridesArray = aclCreateIntArray(stride.data(), stride.size());
    aclIntArray *paddingsArray = aclCreateIntArray(pad.data(), pad.size());
    aclIntArray *dilationsArray =
        aclCreateIntArray(dilation.data(), dilation.size()); // max专属
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    int8_t cubeMathType = 0;
    int64_t divisorOverride = 0;
    bool countIncludePad = true;
    if (mode == PoolingMode::Avg)
    {
        if (nDim == 3 || nDim == 4)
        {
            auto ret = aclnnAvgPool2dGetWorkspaceSize(
                inputTensor, kernelSizeArray, stridesArray, paddingsArray, ceil_mode, countIncludePad,
                divisorOverride, cubeMathType, outputTensor, &workspaceSize,
                &executor);
            if (ret != ACL_SUCCESS)
            {
                printf("aclnnAvgPool2dGetWorkspaceSize failed. ERROR: %d\n", ret);
            }
            void *workspaceAddr = nullptr;
            if (workspaceSize > 0)
            {
                ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

                if (ret != ACL_SUCCESS)
                {
                    printf("allocate workspace failed. ERROR: %d\n", ret);
                }
            }
            ret = aclnnAvgPool2d(workspaceAddr, workspaceSize, executor,
                                 stream);
            if (ret != ACL_SUCCESS)
            {
                printf("avg pool failed. ERROR: %d\n", ret);
            }
        }
        // else if (nDim == 5)
        // {
        //     auto ret = aclnnAvgPool3dGetWorkspaceSize(
        //         inputTensor, kernelSizeArray, stridesArray, paddingsArray, ceil_mode, countIncludePad,
        //         divisorOverride, outputTensor, &workspaceSize,
        //         &executor);
        //     if (ret != ACL_SUCCESS)
        //     {
        //         printf("aclnnAvgPool3dGetWorkspaceSize failed. ERROR: %d\n", ret);
        //     }
        //     void *workspaceAddr = nullptr;
        //     if (workspaceSize > 0)
        //     {
        //         ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        //         if (ret != ACL_SUCCESS)
        //         {
        //             printf("allocate workspace failed. ERROR: %d\n", ret);
        //         }
        //     }
        //     ret = aclnnAvgPool3d(workspaceAddr, workspaceSize, executor,
        //                          stream);
        // }
        }
    else if (mode == PoolingMode::Max)
    {
        const int64_t autoPad = 0;

        auto ret = aclnnMaxPoolGetWorkspaceSize(
            inputTensor, kernelSizeArray, stridesArray, autoPad, paddingsArray, dilationsArray, ceil_mode,
            outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
        {
            printf("aclnnMaxPoolGetWorkspaceSize failed. ERROR: %d\n", ret);
        }
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

            if (ret != ACL_SUCCESS)
            {
                printf("allocate workspace failed. ERROR: %d\n", ret);
            }
        }
        ret = aclnnMaxPool(workspaceAddr, workspaceSize, executor,
                           stream);
        if (ret != ACL_SUCCESS)
        {
            printf("max pool failed. ERROR: %d\n", ret);
        }
    }
    aclDestroyTensor(inputTensor);
    aclDestroyIntArray(kernelSizeArray);
    aclDestroyIntArray(stridesArray);
    aclDestroyIntArray(paddingsArray);
    aclDestroyIntArray(dilationsArray);
    aclDestroyTensor(outputTensor);
}

template <typename T>
void poolingAclnn(void *input, void *output,
                  int *windows, int *pads, int *strides, int *dilations,
                  int *x_shape, int *y_shape,
                  PoolingMode::Mode mode,
                  bool ceil_mode, int nDim)
{
    // static int count = 0;
    // printf("count is %d \n", count);
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    poolingAclnnDevice<T>(input, output,
                          windows, pads, strides, dilations,
                          x_shape, y_shape,
                          mode,
                          ceil_mode, nDim, stream);
    Finalize(deviceId, stream);
}
extern "C" void MaxPooling_aclnn(void *input, void *output,
                                 int *windows, int *pads, int *strides, int *dilations,
                                 int *x_shape, int *y_shape,
                                 int nDim,
                                 int byteSize)
{
    bool ceil_mode = false;
    if (byteSize == 2)
    {
        poolingAclnn<uint16_t>(input, output,
                               windows, pads, strides, dilations,
                               x_shape, y_shape,
                               PoolingMode::Max,
                               ceil_mode, nDim);
    }
    else if (byteSize == 4)
    {
        poolingAclnn<float>(input, output,
                            windows, pads, strides, dilations,
                            x_shape, y_shape,
                            PoolingMode::Max,
                            ceil_mode, nDim);
    }
}
extern "C" void AvgPooling_aclnn(void *input, void *output,
                                 int *windows, int *pads, int *strides, int *dilations,
                                 int *x_shape, int *y_shape,
                                 int nDim,
                                 int byteSize)
{
    bool ceil_mode = false;
    if (byteSize == 2)
    {
        poolingAclnn<uint16_t>(input, output,
                               windows, pads, strides, dilations,
                               x_shape, y_shape,
                               PoolingMode::Avg,
                               ceil_mode, nDim);
    }
    else if (byteSize == 4)
    {
        poolingAclnn<float>(input, output,
                            windows, pads, strides, dilations,
                            x_shape, y_shape,
                            PoolingMode::Avg,
                            ceil_mode, nDim);
    }
}
