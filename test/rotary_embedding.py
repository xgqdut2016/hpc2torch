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

def rotary_embedding(t, sin, cos, torch_device):
    dh = t.shape[2]
    assert dh % 2 == 0, "Embedding dimension must be even."
    t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
    t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]
    cos = cos.unsqueeze(1)  # [seq_len, 1, dh // 2]
    sin = sin.unsqueeze(1)  # [seq_len, 1, dh // 2]

    t_out_even = t_even * cos - t_odd * sin
    t_out_odd = t_even * sin + t_odd * cos

    t_out = torch.empty_like(t)
    t_out[..., 0::2] = t_out_even
    t_out[..., 1::2] = t_out_odd

    return t_out.to(torch_device)


def sin_cos_table(pos, dim, torch_device, theta):
    assert dim % 2 == 0, "Embedding dimension must be even."
    freqs = (1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))).to(
        torch_device
    )
    angles = torch.outer(pos, freqs)
    return torch.sin(angles), torch.cos(angles)

def test(test_shape, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing Rotary Positional Embedding on {device} with shape:{test_shape} and dtype:{test_dtype}"
    )
    torch_device = device
    if device == "kunlun":
        torch_device = "cuda"
    ndim = len(test_shape)
    theta = 1e4

    t = torch.rand(test_shape, dtype=test_dtype).to(torch_device)
    output = torch.zeros(test_shape, dtype=test_dtype).to(torch_device)
    pos = (
        torch.arange(0, t.shape[0], dtype=torch.int32).to(torch_device)
    )

    sin_table, cos_table = sin_cos_table(pos, t.shape[2], t.device, theta)
    if device == "kunlun":
        sin_table = sin_table.to(test_dtype);cos_table = cos_table.to(test_dtype)
    t_ptr = ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    pos_ptr = ctypes.cast(pos.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    sin_ptr = ctypes.cast(sin_table.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    cos_ptr = ctypes.cast(cos_table.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    nt = test_shape[0]
    nh = test_shape[1]
    dimsize = test_shape[2]

    import numpy as np
    np_array = np.array(test_shape, dtype=np.int32)
    shape = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    x_strides = (ctypes.c_int * ndim)(*(t.stride()))
    y_strides = (ctypes.c_int * ndim)(*(output.stride()))
    total_seq_len = sin_table.shape[0]

    if device == "mlu":
        torch_RoPE_time = performance.BangProfile((rotary_embedding, (t, sin_table, cos_table, torch_device)))
        
        lib.RoPE_bang.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int
        ]
        custom_RoPE_time = \
        performance.BangProfile((lib.RoPE_bang, (output_ptr, t_ptr, pos_ptr, sin_ptr, cos_ptr, 
            nt, nh, dimsize, x_strides, y_strides, byteSize)))
    elif device == "kunlun":
        torch_RoPE_time = performance.KunlunProfile((rotary_embedding, (t, sin_table, cos_table, torch_device)))
        lib.rope_kunlun.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int
            ]
        custom_RoPE_time = \
            performance.KunlunProfile((lib.rope_kunlun, (output_ptr, t_ptr, pos_ptr, sin_ptr, cos_ptr, 
            nt, nh, dimsize, x_strides, y_strides, byteSize)))
    performance.logBenchmark(torch_RoPE_time, custom_RoPE_time)

    tmpa = output.to("cpu").detach().numpy().flatten()
    
    tmpb = rotary_embedding(t, sin_table, cos_table, torch_device).to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
    
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test rotary_embedding on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'kunlun'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        ((1, 32, 128)),
        ((1, 32, 64)), 
        
        ((4, 1, 32)),
        ((4, 1, 2050)),
        ((3, 32, 128)),
    ]

if args.device == 'mlu':
    import torch_mlu
if args.device == "kunlun":
    import torch_xmlir

for test_shape in test_cases:
    test(test_shape, args.device)
