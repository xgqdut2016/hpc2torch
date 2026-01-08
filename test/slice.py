import torch
import ctypes
import numpy as np
from functools import partial
import argparse

from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)
def sliceFunction(data, begin, end, stride):#暂时不支持axis参数
    ndim = data.ndimension()
    if len(begin) != ndim or len(end) != ndim or len(stride) != ndim:
        raise ValueError("begin, end, and stride must have the same length as the number of dimensions of data.")
    
    # 创建一个切片列表
    slices = []
    for i in range(ndim):
        slices.append(slice(begin[i].item(), end[i].item(), stride[i].item()))
    
    # 使用切片列表切片数据
    out = data[tuple(slices)]
    
    return out
def inferShape(data, begin, end, stride):
    ndim = data.ndimension()
    outputShape = []
    for i in range(ndim):
        # 计算每一维切片后的大小
        size = (end[i] - begin[i] + stride[i] - 1) / stride[i]
        
        # 向下取整并转换为整数
        outputShape.append(int(size) if size > 0 else 0)
    return outputShape
def test(shape, beginValue, endValue, strideValue, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing slice on {device} with shape:{shape}, dtype:{test_dtype}"
    )
    
    ndim = len(shape)
    input = torch.rand(shape, device=device, dtype=test_dtype, requires_grad=False)
    #特别注意,begin,end, stride不要放到device上
    begin = torch.tensor(beginValue, dtype=torch.int32, requires_grad=False)
    end = torch.tensor(endValue, dtype=torch.int32, requires_grad=False)
    stride = torch.tensor(strideValue, dtype=torch.int32, requires_grad=False)
    
    yShape = inferShape(input, begin, end, stride)
    output = torch.zeros(yShape, device=device, dtype=test_dtype, requires_grad=False)

    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    begin_ptr = ctypes.cast(begin.data_ptr(), ctypes.POINTER(ctypes.c_int))
    end_ptr = ctypes.cast(end.data_ptr(), ctypes.POINTER(ctypes.c_int))
    stride_ptr = ctypes.cast(stride.data_ptr(), ctypes.POINTER(ctypes.c_int))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    x_shape = np.array(shape, dtype=np.int32)
    inputShape = x_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    y_shape = np.array(yShape, dtype=np.int32)
    outputShape = y_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if device == "mlu":
        torch_slice_time = performance.BangProfile((sliceFunction, (input, begin, end, stride)))  # 以毫秒为单位
        lib.slice_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_slice_time = \
        performance.BangProfile((lib.slice_cnnl, (input_ptr, output_ptr, 
        inputShape, outputShape, begin_ptr, end_ptr, stride_ptr, 
        ndim, len(yShape), len(beginValue), byteSize)))
    elif device == "npu":
        torch_slice_time = performance.AscendProfile((sliceFunction, (input, begin, end, stride)))  # 以毫秒为单位
        lib.slice_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_slice_time = \
        performance.AscendProfile((lib.slice_aclnn, (input_ptr, output_ptr, 
        inputShape, outputShape, begin_ptr, end_ptr, stride_ptr, 
        ndim, len(yShape), len(beginValue), byteSize)))
    
    performance.logBenchmark(torch_slice_time, custom_slice_time)
    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = sliceFunction(input, begin, end, stride).to('cpu').numpy().flatten()
    tmpb = output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test slice on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # shape, begin, end, stride
        ((700, 10), (1, 0), (500, 6), (2, 1)),
        ((700, 10, 24), (1, 0, 0), (50, 6, 20), (2, 1, 3)),
        ((700, 10, 24, 6), (1, 0, 0, 3), (50, 6, 20, 5), (2, 1, 3, 1)),     
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for shape, begin, end, stride in test_cases:
    test(shape, begin, end, stride, args.device)
