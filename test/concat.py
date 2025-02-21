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
def concatFunction(tensors, dim=0):
    return torch.cat(tensors, dim=dim)
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

def test(c_shape, axis, input_shapes, device):
    num_inputs = len(input_shapes)
    ndim = len(c_shape)
    byteSize = 2
    tensor_dtype = torch.float16
    if byteSize == 4:
        tensor_dtype = torch.float32
    print(
        f"Testing concat on {device} with xShape:{input_shapes}, yShape:{c_shape}, dtype:{tensor_dtype}"
    )

    inputs = [torch.rand(shape, dtype=tensor_dtype).to(device) for shape in input_shapes]
    c = torch.zeros(c_shape, dtype=tensor_dtype).to(device)
    #下面获取inputs的二维指针地址
    input_ptrs = [ctypes.cast(input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) for input_tensor in inputs]
    input_ptr_array = (ctypes.POINTER(ctypes.c_void_p) * len(input_ptrs))(*input_ptrs)
    input_ptrs_ctypes = ctypes.cast(input_ptr_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
    #下面获取inputshape的二维指针地址
    input_shapes_ctypes = []
    for shape in input_shapes:
        shape_array = (ctypes.c_int * len(shape))(*shape)  # 创建形状数组
        shape_ptr = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int))  # 创建指向该数组的指针
        input_shapes_ctypes.append(shape_ptr)

    # 将 input_shapes_ctypes 转换为 ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
    input_shapes_array = (ctypes.POINTER(ctypes.c_int) * len(input_shapes_ctypes))(*input_shapes_ctypes)
    input_shapes_ctypes = ctypes.cast(input_shapes_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))

    output_ptr = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    
    
    outputShape = np.array(c_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if device == "mlu":
        torch_concat_time = performance.BangProfile((concatFunction, (inputs, axis)))  # 以毫秒为单位
        lib.concat_cnnl.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_concat_time = \
        performance.BangProfile((lib.concat_cnnl, (input_ptrs_ctypes, output_ptr, input_shapes_ctypes, outputShape, num_inputs, axis, ndim, byteSize)))
    
    performance.logBenchmark(torch_concat_time, custom_concat_time)
    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = concatFunction(inputs, axis).to('cpu').numpy().flatten()
    tmpb = c.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test concat on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        #c_shape, axis, input_shapes

        ((6,), 0, [(2,), (4,)]),  

        ((6, 3), 0, [(2, 3), (4, 3)]),  
        ((3, 6), 1, [(3, 2), (3, 4)]),  
        ((3, 7), 1, [(3, 2), (3, 4), (3, 1)]), 
        ((3, 3, 10), 2, [(3, 3, 4), (3, 3, 6)]),  
        ((4, 3, 6), 0, [(3, 3, 6), (1, 3, 6)]),  
        ((2, 6, 3), 1, [(2, 3, 3), (2, 3, 3)]),  
        ((2, 3, 6), 2, [(2, 3, 3), (2, 3, 3)]),  
        ((4, 3, 5, 6), 0, [(1, 3, 5, 6), (3, 3, 5, 6)]),  
        ((2, 5, 5, 6), 1, [(2, 3, 5, 6), (2, 2, 5, 6)]),  
        ((2, 3, 5, 6), 2, [(2, 3, 2, 6), (2, 3, 3, 6)]),  
        ((2, 3, 5, 6), 3, [(2, 3, 5, 3), (2, 3, 5, 3)]),  
        ((2, 3, 5, 15), 3, [(2, 3, 5, 3), (2, 3, 5, 3), (2, 3, 5, 9)]),  
        ((4, 2, 3, 4, 5), 0, [(1, 2, 3, 4, 5), (3, 2, 3, 4, 5)]),  
        ((2, 4, 3, 2, 5), 1, [(2, 2, 3, 2, 5), (2, 2, 3, 2, 5)]),  
        ((1, 2, 4, 4, 5), 2, [(1, 2, 2, 4, 5), (1, 2, 2, 4, 5)]),  
        ((1, 2, 3, 8, 5), 3, [(1, 2, 3, 4, 5), (1, 2, 3, 4, 5)]), 
        ((1, 2, 3, 4, 5), 4, [(1, 2, 3, 4, 3), (1, 2, 3, 4, 2)]),  
        ((4, 14, 3, 4, 5), 1, [(4, 3, 3, 4, 5), (4, 5, 3, 4, 5), (4, 6, 3, 4, 5)]),        
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for c_shape, axis, input_shapes in test_cases:
    test(c_shape, axis, input_shapes, args.device)
