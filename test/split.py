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
# split_size_or_sections表示每一个块的大小，并不是说切多少块
def splitFunction(tensors, split_size_or_sections, axis):
    return torch.split(tensors, split_size_or_sections, dim=axis)

def inferShape(xShape, split_size_or_sections, axis):
    outShape = []
    if isinstance(split_size_or_sections, int): # 如果是 int 类型
        remain = xShape[axis] % split_size_or_sections
        repeat = (xShape[axis] - remain) / split_size_or_sections

        for i in range((int)(repeat)):
            outShape.append(list(xShape))
            outShape[i][axis] = split_size_or_sections
        if remain:
            outShape.append(list(xShape))
            outShape[(int)(repeat)][axis] = remain
    elif isinstance(split_size_or_sections, list):# 如果是 list 类型
        assert sum(split_size_or_sections) == xShape[axis]
        for i in range(len(split_size_or_sections)):
            outShape.append(list(xShape))
            outShape[i][axis] = split_size_or_sections[i]
    return outShape

# 将 inputs 转换为 ctype 的二维指针
def tensor_list_to_ctype_void(inputs):
    # 创建一个 list，用于存储每个 tensor 的指针
    tensor_pointers = []

    for tensor in inputs:
        # 获取 tensor 数据的指针
        ptr = tensor.data_ptr()
        # 将指针转换为 ctypes.c_void_p 类型
        tensor_pointers.append(ctypes.c_void_p(ptr))
    
    # 创建一个 ctypes 数组类型，指向 c_void_p 的指针
    pointer_array_type = ctypes.POINTER(ctypes.c_void_p) * len(tensor_pointers)
    
    # 返回指向这个数组的指针
    return pointer_array_type(*tensor_pointers)

def test(xShape, axis, split_size_or_sections, device):
    ndim = len(xShape)
    byteSize = 2
    tensor_dtype = torch.float16
    if byteSize == 4:
        tensor_dtype = torch.float32
    print(
        f"Testing split on {device} with xShape:{xShape}, axis:{axis}, split_size_or_sections:{split_size_or_sections}, dtype:{tensor_dtype}"
    )

    input = torch.rand(xShape, dtype = tensor_dtype, device = device)
    outShape = inferShape(xShape, split_size_or_sections, axis)
    num_outputs = len(outShape)
    outputs = [torch.zeros(shape, dtype=tensor_dtype, device = device) for shape in outShape]

    #下面获取outputs的二维指针地址
    output_ptrs = [ctypes.cast(output_tensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) for output_tensor in outputs]
    output_ptr_array = (ctypes.POINTER(ctypes.c_void_p) * len(output_ptrs))(*output_ptrs)
    output_ptrs_ctypes = ctypes.cast(output_ptr_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
    #下面获取outputshape的二维指针地址
    yShape_ctypes = []
    for shape in outShape:
        shape_array = (ctypes.c_int * len(shape))(*shape)  # 创建形状数组
        shape_ptr = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int))  # 创建指向该数组的指针
        yShape_ctypes.append(shape_ptr)

    # 将 xShape_ctypes 转换为 ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
    yShape_array = (ctypes.POINTER(ctypes.c_int) * len(yShape_ctypes))(*yShape_ctypes)
    yShape_ctypes = ctypes.cast(yShape_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))

    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    inputShape = np.array(xShape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if device == "mlu":
        torch_split_time = performance.BangProfile((splitFunction, (input, split_size_or_sections, axis)))  # 以毫秒为单位
        lib.split_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_split_time = \
        performance.BangProfile((lib.split_cnnl, (input_ptr, output_ptrs_ctypes, inputShape, yShape_ctypes, num_outputs, axis, ndim, byteSize)))
    elif device == "npu":
        torch_split_time = performance.AscendProfile((splitFunction, (input, split_size_or_sections, axis)))  # 以毫秒为单位
        lib.split_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_split_time = \
        performance.AscendProfile((lib.split_aclnn, (input_ptr, output_ptrs_ctypes, inputShape, yShape_ctypes, num_outputs, axis, ndim, byteSize)))
    performance.logBenchmark(torch_split_time, custom_split_time)
    # 将结果转换回 PyTorch 张量以进行比较
    atol = 0
    rtol = 0
    for i in range(num_outputs):
        tmpa = splitFunction(input, split_size_or_sections, axis)[i].to('cpu').numpy().flatten()
        tmpb = outputs[i].to('cpu').numpy().flatten()
        atoltmp = max(abs(tmpa - tmpb))
        rtoltmp = atol / (max(abs(tmpb)) + 1e-8)
        rtol = max(rtol, rtoltmp)
        atol = max(atol, atoltmp)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test split on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        #xShape, axis, split_size_or_sections
        ((10,), 0, 3),
        ((10,), 0, [2, 2, 6]),
        ((10, 90), 0, [2, 2, 6]),
        ((10, 90), 1, 7),
        ((10, 90, 3), 0, [1, 5, 4]),
        ((10, 90, 3), 1, [10, 50, 30]),
        ((10, 90, 3), 2, 2),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for xShape, axis, split_size_or_sections in test_cases:
    test(xShape, axis, split_size_or_sections, args.device)
