#include "acl/acl.h"
#include "aclnnop/aclnn_clamp.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void clipAclnnDevice(void *input, void *output,
                     int *shape, int nDim,
                     float minValue, float maxValue, aclrtStream &stream)
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
        inputDim[i] = int64_t(shape[i]);
        outputDim[i] = int64_t(shape[i]);

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
    aclScalar *max = nullptr;
    aclScalar *min = nullptr;
    max = aclCreateScalar(&maxValue, aclDataType::ACL_FLOAT); // 如果input数据类型是f16，min,max也要使用f32，否则结果报错
    min = aclCreateScalar(&minValue, aclDataType::ACL_FLOAT);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto ret = aclnnClampGetWorkspaceSize(inputTensor, min, max, outputTensor, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnClampGetWorkspaceSize failed. ERROR: %d\n", ret);
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

    ret = aclnnClamp(workspaceAddr, workspaceSize, executor, stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnClamp failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);

    aclDestroyScalar(max);
    aclDestroyScalar(min);

    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void clipAclnn(void *input, void *output,
               int *shape, int nDim,
               float minValue, float maxValue)
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

    clipAclnnDevice<T>(input, output,
                       shape, nDim,
                       minValue, maxValue, stream);
    Finalize(deviceId, stream);
}
extern "C" void clip_aclnn(void *input, void *output,
                           int *shape, int nDim,
                           float minValue, float maxValue, int byteSize)
{
    if (byteSize == 4)
    {
        clipAclnn<float>(input, output,
                         shape, nDim,
                         minValue, maxValue);
    }
    else if (byteSize == 2)
    {

        clipAclnn<uint16_t>(input, output,
                            shape, nDim,
                            minValue, maxValue);
    }
}
