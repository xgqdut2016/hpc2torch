import torch
import ctypes
import torch.nn as nn
from functools import partial
import argparse

import performance
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

def test(y_shape, x_shape, w_shape, dtype, w_dtype, torch_device):
    print(f"Testing RMS_Norm on {torch_device} with y_shape:{y_shape} x_shape:{x_shape} w_shape:{w_shape}"
        f" dtype:{dtype} w_dtype:{w_dtype}")
    assert len(x_shape) == 2 and len(y_shape) == 2
    y = torch.zeros(y_shape, dtype=dtype).to(torch_device)
    x = torch.rand(x_shape, dtype=dtype).to(torch_device)
    w = torch.ones(w_shape, dtype=w_dtype).to(torch_device)

    eps = 1e-5
    stride_x = x_shape[1]
    stride_y = y_shape[1]
    n = x_shape[0]
    d = x_shape[1]
    input_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    scale_ptr = ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
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
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_RMSNorm_time = \
        performance.BangProfile((lib.RMSNorm_bang, (input_ptr, scale_ptr, output_ptr, stride_y, stride_x, eps, n, d, byteT, byteTw)))
    performance.logBenchmark(torch_RMSNorm_time, custom_RMSNorm_time)

    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = rms_norm(x, w, eps).to('cpu').detach().numpy().flatten()
    
    tmpb = y.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test RMSNorm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # test_shape, axis, test_dtype, eps, device
        ((16, 2048), (16, 2048), (2048,), torch.float16, torch.float16),
        ((16, 2048), (16, 2048), (2048,), torch.float32, torch.float32),
        ((16, 2048), (16, 2048), (2048,), torch.float16, torch.float32),
         
]

if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for y_shape, x_shape, w_shape, dtype, w_dtype, in test_cases:
    test(y_shape, x_shape, w_shape, dtype, w_dtype, args.device)