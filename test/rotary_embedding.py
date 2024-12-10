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

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotary_embedding(t, pos, theta, torch_device):
    dh = t.shape[2]
    freqs = (1.0 / (theta ** (torch.arange(0, dh, 2)[: (dh // 2)].float() / dh))).to(
        torch_device
    )
    
    freqs = torch.outer(pos, freqs)#寒武纪不支持这种运算，如果在寒武纪机器测试，只能提前把数据移动到cpu
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    t_ = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, t_)
    t_out = torch.view_as_real(t_ * freqs_cis).flatten(2).to(t.dtype)
    return t_out

def sin_cos_table(max_seq_len, dim, torch_device, theta):
    pos = torch.arange(
        0, max_seq_len, dtype=torch.float32, device=torch.device(torch_device)
    )
    freqs = (1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))).to(
        torch_device
    )
    # (a0, a1, a2) -> (a0, a0, a1, a1, a2, a2)
    freqs = torch.repeat_interleave(freqs, repeats=2)
    angles = torch.outer(pos, freqs)
    return torch.sin(angles), torch.cos(angles)
def test(test_shape, torch_device, test_dtype=torch.float16):
    print(
        f"Testing Rotary Positional Embedding on {torch_device} with shape:{test_shape} and dtype:{test_dtype}"
    )
    ndim = len(test_shape)
    t = torch.rand(test_shape, dtype=test_dtype)
    
    pos = torch.arange(0, t.shape[0])
    theta = 1e4
    
    pos = pos.to(torch.int64)
    pos = pos.to(torch_device)
    t = t.to(torch_device)

    
    # 2x table length for test
    sin_table, cos_table = sin_cos_table(t.shape[0] * 2, t.shape[2], t.device, theta)

    t_ptr = ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    pos_ptr = ctypes.cast(pos.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    sin_ptr = ctypes.cast(sin_table.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    cos_ptr = ctypes.cast(cos_table.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    stride_0 = t.stride()[0]
    stride_1 = t.stride()[1]
    nt = test_shape[0]
    nh = test_shape[1]
    dimsize = test_shape[2]

    import numpy as np
    np_array = np.array(test_shape, dtype=np.int32)
    shape = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    strides = (ctypes.c_int * ndim)(*(t.stride()))
    total_seq_len = sin_table.shape[0]

    if test_dtype == torch.float16:
        
        if device == "mlu":
            torch_RoPE_time = performance.BangProfile((rotary_embedding, (t.to("cpu"), pos.to("cpu"), theta, "cpu")))  
            
            lib.RoPE_cnnl_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int
            ]
            custom_RoPE_time = \
            performance.BangProfile((lib.RoPE_cnnl_f16, (t_ptr, pos_ptr, sin_ptr, cos_ptr, shape, strides, total_seq_len)))
            '''
            lib.RoPE_bang_f16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]
            custom_RoPE_time = \
            performance.BangProfile((lib.RoPE_bang_f16, (t_ptr, pos_ptr, sin_ptr, cos_ptr, stride_0, stride_1, nt, nh, dimsize)))
            '''
            
    performance.logBenchmark(torch_RoPE_time, custom_RoPE_time)
    tmpa = rotary_embedding(t.to("cpu"), pos.to("cpu"), theta, "cpu").detach().numpy().flatten()
    
    tmpb = t.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
    
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test rotary_embedding on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        ((1, 32, 128), "mlu", torch.float16),
        ((1, 32, 64), "mlu", torch.float16),
        
        ((4, 1, 32), "mlu", torch.float16),
        ((1, 32, 128), "mlu", torch.float16),
        
        ((3, 32, 128), "mlu", torch.float16),
    ]
filtered_test_cases = [
    (test_shape, device, test_dtype)
    for test_shape, device, test_dtype in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for test_shape, device, test_dtype in filtered_test_cases:
    test(test_shape, device, test_dtype)