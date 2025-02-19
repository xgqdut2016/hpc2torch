#include "acl/acl.h"
// #include "aclnnop/level2/aclnn_expand.h"
#include "aclnnop/aclnn_expand.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void expandAclnnDevice(void *input, void *output, int *inputShape, int *outputShape, int nDim, aclrtStream &stream)
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

    std::vector<int64_t> inputDim(nDim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(nDim, 1); // 初始化为1
    std::vector<int64_t> outputDim(nDim);
    std::vector<int64_t> outputStride(nDim, 1);

    for (int i = nDim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(inputShape[i]);
        outputDim[i] = int64_t(outputShape[i]);

        if (i < nDim - 1)
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
    // aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);只能接受int64_t的数组

    aclIntArray *expandSize = aclCreateIntArray(outputDim.data(), outputDim.size()); // output的形状根据input和expandSize的形状进行broadcast

    auto ret = aclnnExpandGetWorkspaceSize(inputTensor, expandSize, outputTensor, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnExpandGetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnExpand(workspaceAddr, workspaceSize, executor,
                      stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnExpand failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);

    aclDestroyIntArray(expandSize);

    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void expandAclnn(void *input, void *output, int *inputShape, int *outputShape, int nDim)
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

    expandAclnnDevice<T>(input, output, inputShape, outputShape, nDim, stream);
    Finalize(deviceId, stream);
}
extern "C" void expand_aclnn(void *input, void *output, int *inputShape, int *outputShape, int nDim, int byteSize)
{
    if (byteSize == 4)
    {
        expandAclnn<float>(input, output, inputShape, outputShape, nDim);
    }
    else if (byteSize == 2)
    {
        expandAclnn<uint16_t>(input, output, inputShape, outputShape, nDim);
    }
}
