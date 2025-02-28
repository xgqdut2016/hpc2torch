#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void layernormAclnnDevice(void *source, void *weight, void *bias, void *destination,
                          int *shape, int ndim, int axis, float eps,
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
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    std::vector<int64_t> normShape(ndim - axis);
    for (int i = 0; i < ndim - axis; i++)
    {
        normShape[i] = int64_t(shape[i + axis]);
    }
    std::vector<int64_t> inputDim(ndim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(ndim, 1); // 初始化为1
    std::vector<int64_t> weightDim(ndim - axis);
    std::vector<int64_t> weightStride(ndim - axis, 1);
    std::vector<int64_t> biasDim(ndim - axis);
    std::vector<int64_t> biasStride(ndim - axis, 1);
    std::vector<int64_t> outputDim(ndim);
    std::vector<int64_t> outputStride(ndim, 1);

    for (int i = ndim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(shape[i]);
        outputDim[i] = int64_t(shape[i]);
        if (i < ndim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    for (int i = ndim - axis - 1; i >= 0; i--)
    {
        weightDim[i] = normShape[i];
        biasDim[i] = normShape[i];
        if (i < ndim - axis - 1)
        {
            weightStride[i] = weightDim[i + 1] * weightStride[i + 1];
            biasStride[i] = biasDim[i + 1] * biasStride[i + 1];
        }
    }

    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), source); // const aclTensor *inputTensor
    auto weightTensor =
        aclCreateTensor(weightDim.data(), weightDim.size(), dataType,
                        weightStride.data(), 0, format,
                        weightDim.data(), weightDim.size(), weight);
    // aclTensor *biasTensor = NULL;//如果bias传入nullptr，就初始化biasTensor = NULL
    auto biasTensor =
        aclCreateTensor(biasDim.data(), biasDim.size(), dataType,
                        biasStride.data(), 0, format,
                        biasDim.data(), biasDim.size(), bias);
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), destination);

    auto *normArray =
        aclCreateIntArray(normShape.data(), normShape.size());
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto ret = aclnnLayerNormGetWorkspaceSize(
        inputTensor, normArray, weightTensor, biasTensor, eps, outputTensor,
        NULL, NULL, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnLayerNorm(workspaceAddr, workspaceSize, executor,
                         stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnLayerNorm failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(biasTensor);
    aclDestroyTensor(weightTensor);
    aclDestroyIntArray(normArray);
    aclDestroyTensor(outputTensor);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void layernormAclnn(void *source, void *weight, void *bias, void *destination,
                    int *shape, int ndim, int axis, float eps)
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

    layernormAclnnDevice<T>(source, weight, bias, destination,
                            shape, ndim, axis, eps, stream);
    Finalize(deviceId, stream);
}
extern "C" void layernorm_aclnn(void *source, void *weight, void *bias, void *destination,
                                int *shape, int ndim, int axis, float eps, int byteSize)
{
    if (byteSize == 4)
    {
        layernormAclnn<float>(source, weight, bias, destination,
                              shape, ndim, axis, eps);
    }
    else if (byteSize == 2)
    {
        layernormAclnn<uint16_t>(source, weight, bias, destination,
                                 shape, ndim, axis, eps);
    }
}
