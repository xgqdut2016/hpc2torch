import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def maxReduce(data, axes):
    if not axes:  # 如果 axes 是空的，计算所有维度的最大值
        return torch.max(data)
    
    # 如果 axes 是非空的，沿着指定的维度进行 max 操作
    # 为了保持原始维度信息，我们使用 keepdim=True 来保留被 "压缩" 维度的大小
    output = data
    for axis in axes:
        output = output.max(dim=axis, keepdim=True)[0]  # [0] 获取最大值（max 返回的是值和索引）
    return output
def meanReduce(data, axes):
    if not axes:  # 如果 axes 是空的，计算所有维度的平均值
        return torch.mean(data)
    
    output = data
    for axis in axes:
        output = output.mean(dim=axis, keepdim=True)  # 沿着指定的维度求平均值
    
    return output
def minReduce(data, axes):
    if not axes:  # 如果 axes 是空的，计算所有维度的最大值
        return torch.min(data)
    
    # 如果 axes 是非空的，沿着指定的维度进行 max 操作
    # 为了保持原始维度信息，我们使用 keepdim=True 来保留被 "压缩" 维度的大小
    output = data
    for axis in axes:
        output = output.min(dim=axis, keepdim=True)[0]  # [0] 获取最大值（max 返回的是值和索引）
    return output
def prodReduce(data, axes):
    if not axes:  # 如果 axes 是空的，计算所有维度的乘积
        return torch.prod(data)
    
    output = data
    for axis in axes:
        output = output.prod(dim=axis, keepdim=True)  # 沿着指定的维度求乘积
    
    return output
def sumReduce(data, axes):
    if not axes:  # 如果 axes 是空的，计算所有维度的和
        return torch.sum(data)
    
    output = data
    for axis in axes:
        output = output.sum(dim=axis, keepdim=True)  # 沿着指定的维度求和
    
    return output
def test(inputShape, axes, device):
    # 昇腾的min,prod只支持针对固定某一个维度规约
    operators = ["Max", "Mean", "Min", "Prod", "Sum"]
    operator = operators[1]
    byteSize = 2
    
    if byteSize == 2:
        tensor_dtype = torch.float16
    elif byteSize == 4:
        tensor_dtype = torch.float32
    print(
        f"Testing {operator} reduce on {device} with inputShape:{inputShape}, axes:{axes}, dtype:{tensor_dtype}"
    )
    
    
    a = torch.rand(inputShape, dtype=tensor_dtype).to(device)
    ndim = len(inputShape)
    if axes:
        outputShape = list(inputShape)
        for i in axes:
            outputShape[i] = 1
    else:
        outputShape = [1 for i in range(ndim)]
    c = torch.rand(outputShape, dtype=tensor_dtype).to(device)
    
    
    aData = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    cData = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    axes_ptr = np.array(axes, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    aShape = np.array(inputShape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cShape = np.array(outputShape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    if operator == "Max":
        if device == "mlu":
            torch_reduce_time = performance.BangProfile((maxReduce, (a, axes)))  # 可以替换为pRelu
            lib.maxReduce_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.BangProfile((lib.maxReduce_cnnl, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))
        elif device == "npu":
            torch_reduce_time = performance.AscendProfile((maxReduce, (a, axes)))  # 可以替换为pRelu
            lib.maxReduce_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.AscendProfile((lib.maxReduce_aclnn, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))                            
        performance.logBenchmark(torch_reduce_time, custom_reduce_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = maxReduce(a, axes).to('cpu').numpy().flatten()
    elif operator == "Min":
        if device == "mlu":
            torch_reduce_time = performance.BangProfile((minReduce, (a, axes)))  # 可以替换为pRelu
            lib.minReduce_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.BangProfile((lib.minReduce_cnnl, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))
        elif device == "npu":
            torch_reduce_time = performance.AscendProfile((minReduce, (a, axes)))  # 可以替换为pRelu
            lib.minReduce_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.AscendProfile((lib.minReduce_aclnn, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))  
        performance.logBenchmark(torch_reduce_time, custom_reduce_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = minReduce(a, axes).to('cpu').numpy().flatten()
    elif operator == "Mean":
        if device == "mlu":
            torch_reduce_time = performance.BangProfile((meanReduce, (a, axes)))  # 可以替换为pRelu
            lib.meanReduce_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.BangProfile((lib.meanReduce_cnnl, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))
        elif device == "npu":
            torch_reduce_time = performance.AscendProfile((meanReduce, (a, axes)))  # 可以替换为pRelu
            lib.meanReduce_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.AscendProfile((lib.meanReduce_aclnn, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))  
        performance.logBenchmark(torch_reduce_time, custom_reduce_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = meanReduce(a, axes).to('cpu').numpy().flatten()
    elif operator == "Prod":
        if device == "mlu":
            torch_reduce_time = performance.BangProfile((prodReduce, (a, axes)))  # 可以替换为pRelu
            lib.prodReduce_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.BangProfile((lib.prodReduce_cnnl, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))
        elif device == "npu":
            torch_reduce_time = performance.AscendProfile((prodReduce, (a, axes)))  # 可以替换为pRelu
            lib.prodReduce_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.AscendProfile((lib.prodReduce_aclnn, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))  
        performance.logBenchmark(torch_reduce_time, custom_reduce_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = prodReduce(a, axes).to('cpu').numpy().flatten()
    elif operator == "Sum":
        if device == "mlu":
            torch_reduce_time = performance.BangProfile((sumReduce, (a, axes)))  # 可以替换为pRelu
            lib.sumReduce_cnnl.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.BangProfile((lib.sumReduce_cnnl, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))
        elif device == "npu":
            torch_reduce_time = performance.AscendProfile((sumReduce, (a, axes)))  # 可以替换为pRelu
            lib.sumReduce_aclnn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_reduce_time = \
            performance.AscendProfile((lib.sumReduce_aclnn, (aData, axes_ptr, cData, aShape, cShape,
                                    ndim, len(axes), byteSize)))  
        performance.logBenchmark(torch_reduce_time, custom_reduce_time)
        # 将结果转换回 PyTorch 张量以进行比较
        tmpa = sumReduce(a, axes).to('cpu').numpy().flatten()
    tmpb = c.to('cpu').numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test reduce on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # inputShape, axes
        ((1000, ), ()),
        ((1000, 3), (0, )),
        ((200, 40, 3), (1, 2)),
        ((20, 3, 4, 5), ()),
        ((20, 3, 4, 5), (1, 3)), #如果axes遇到了负数需要特殊处理
        
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for inputShape, axes in test_cases:
    test(inputShape, axes, args.device)