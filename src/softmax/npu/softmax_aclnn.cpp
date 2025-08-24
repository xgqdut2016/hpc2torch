#include "acl/acl.h"
#include "aclnnop/aclnn_softmax.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void softmaxAclnnDevice(void *input, void *output, int ndim, int axis, int *shape,
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
    std::vector<int64_t> inputDim(ndim); // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(ndim, 1);
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
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor

    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto ret = aclnnSoftmaxGetWorkspaceSize(inputTensor, int64_t(axis), outputTensor,
                                            &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnSoftmax(workspaceAddr, workspaceSize, executor,
                       stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnSoftmax failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
}
template <typename T>
void softmaxAclnn(void *input, void *output, int ndim, int axis, int *shape)
{
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    softmaxAclnnDevice<T>(input, output, ndim, axis, shape, stream);
    Finalize(deviceId, stream);
}

extern "C" void softmax_aclnn(void *input, void *output, int ndim, int axis, int *shape, int byteSize)
{
    if (byteSize == 4)
    {
        softmaxAclnn<float>(input, output, ndim, axis, shape);
    }
    else if (byteSize == 2)
    {
        softmaxAclnn<uint16_t>(input, output, ndim, axis, shape);
    }
}
