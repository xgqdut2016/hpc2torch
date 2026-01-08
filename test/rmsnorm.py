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

def rms_norm(x, w, eps):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return w * hidden_states.to(input_dtype)

def test(test_shape, w_shape, dtype, w_dtype, torch_device):
    print(f"Testing RMS_Norm on {torch_device} with shape:{test_shape} w_shape:{w_shape}"
        f" dtype:{dtype} w_dtype:{w_dtype}")
    assert test_shape[-1] == w_shape[0]
    ndim = len(test_shape)
    y = torch.zeros(test_shape, dtype=dtype).to(torch_device)
    x = torch.rand(test_shape, dtype=dtype).to(torch_device)
    w = torch.ones(w_shape, dtype=w_dtype).to(torch_device)

    eps = 1e-5
    stride_x = list(x.stride())
    stride_y = list(y.stride())
    input_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    scale_ptr = ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    shapeData = np.array(test_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    strideXData = np.array(stride_x, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    strideYData = np.array(stride_y, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    byteT = 2
    byteTw = 2
    if (dtype == torch.float32):
        byteT = 4
    if (w_dtype == torch.float32):
        byteTw = 4

    
    if torch_device == "mlu":
        torch_RMSNorm_time = performance.BangProfile((rms_norm, (x, w, eps)))  # 以毫秒为单位
        lib.RMSNorm_bang.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_RMSNorm_time = \
        performance.BangProfile((lib.RMSNorm_bang, (output_ptr, input_ptr, scale_ptr, shapeData, strideYData, strideXData, eps, ndim, byteT, byteTw)))
        
    performance.logBenchmark(torch_RMSNorm_time, custom_RMSNorm_time)

    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = rms_norm(x, w, eps).to('cpu').detach().numpy().flatten()
    
    tmpb = y.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test RMSNorm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # test_shape, w_shape
        ((2050, ), (2050,)),
        ((20500, ), (20500,)),
        ((16, 2048), (2048,)),
        ((5, 4096), (4096,)),
        ((5, 99, 1000), (1000,)),   
]

if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for test_shape, w_shape in test_cases:
    test(test_shape, w_shape, torch.float16, torch.float16, args.device)
    test(test_shape, w_shape, torch.float16, torch.float32, args.device)
    test(test_shape, w_shape, torch.float32, torch.float32, args.device)
