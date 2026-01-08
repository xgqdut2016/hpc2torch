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

def swiglu(x, y):
    return torch.sigmoid(y) * y * x

def test(c_shape, a_shape, b_shape, device):
    torch_device = "kunlun"
    if device == "kunlun":
        torch_device = "cuda"
    else:
        torch_device = device
    byteSize = 4
    
    if byteSize == 2:
        tensor_dtype = torch.float16
    elif byteSize == 4:
        tensor_dtype = torch.float32
    print(
        f"Testing swiglu on {device} with aShape:{a_shape}, bShape:{b_shape}, cShape:{c_shape}, dtype:{tensor_dtype}"
    )
    
    a = torch.rand(a_shape, dtype=tensor_dtype).to(torch_device)
    b = torch.rand(b_shape, dtype=tensor_dtype).to(torch_device)
    c = torch.zeros(c_shape, dtype=tensor_dtype).to(torch_device)
    
    ndim = len(c_shape)
    
    aData = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    bData = ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    cData = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    aShape = np.array(a_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    bShape = np.array(b_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cShape = np.array(c_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    a_strides = np.array(list(a.stride()), dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    b_strides = np.array(list(b.stride()), dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    c_strides = np.array(list(c.stride()), dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    c_data_size = 1
    for i in range(ndim):
        c_data_size *= c_shape[i]

    continugous = ctypes.c_bool(True)
    broadcast = ctypes.c_bool(False)
    if len(a_shape) != ndim or len(b_shape) != ndim:
        broadcast = ctypes.c_bool(True)
        continugous = ctypes.c_bool(False)
    else:
        for i in range(ndim):
            if a_shape[i] != c_shape[i] or b_shape[i] != c_shape[i]:
                broadcast = ctypes.c_bool(True)
                continugous = ctypes.c_bool(False)
                break
    
    if device == "kunlun":
        torch_swiglu_time = performance.KunlunProfile((swiglu, (a, b)))  # 可以替换为mul, swiglu
        lib.swiglu_kunlun.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int
        ]
        custom_swiglu_time = \
        performance.KunlunProfile((lib.swiglu_kunlun, (c_data_size, ndim, len(a_shape), len(b_shape), continugous, broadcast,
         cShape, c_strides, aShape, a_strides, bShape, b_strides, cData, aData, bData, 
                                byteSize)))
             
        performance.logBenchmark(torch_swiglu_time, custom_swiglu_time)
    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = swiglu(a, b).to('cpu').numpy().flatten()
    tmpb = c.to('cpu').numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test swiglu on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu', 'kunlun'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # c_shape, a_shape, b_shape
        ((1, 3), (1,), (1, 3)),
        ((1, 3), (1, 3), (1, 3)),
        ((2, 4, 3), (2, 1, 3), (4, 3)),
        ((2, 3, 4, 5), (2, 3, 4, 5), (5,)),

        ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1)),
        ((3, 20, 33), (3, 20, 33), (3, 20, 33)),
        ((32, 3, 112, 112), (32, 3, 112, 112), (32, 3, 112, 112)),

        
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
if args.device == "kunlun":
    import torch_xmlir
# 执行过滤后的测试用例
for c_shape, a_shape, b_shape in test_cases:
    test(c_shape, a_shape, b_shape, args.device)
        