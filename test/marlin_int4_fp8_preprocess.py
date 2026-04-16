import torch
import ctypes
import numpy as np
import numpy
from functools import partial
import argparse
import itertools
import warnings

from utils import performance
from utils.scalar_type import scalar_types, ScalarType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def marlin_int4_fp8_preprocess_torch(qweight_unpacked, qzeros_unpacked, group_size):
    repeated_zp = qzeros_unpacked.repeat_interleave(group_size, 0)
    torch_res = qweight_unpacked - repeated_zp
    torch_res[torch_res < 0] = 15 - qweight_unpacked[torch_res < 0]
    torch_res = torch_res[:, ::2] * 16 + torch_res[:, 1::2]
    torch_res = torch_res.to(torch.int8).view(torch.int32)
    return torch_res

def test(size_k, size_n, group_size, device):
    print(
        f"Testing marlin int4 fp8 preprocess on {device} with size_k-size_n-group_size:[{size_k, size_n, group_size}]"
    )
    qweight_unpacked = torch.randint(
        0, 16, size=(size_k, size_n), dtype=torch.int32, device=device
    )
    qzeros_unpacked = torch.randint(
        0, 16, size=(size_k // group_size, size_n), dtype=torch.int32, device=device
    )
    qweight_packed = qweight_unpacked[:, ::2] * 16 + qweight_unpacked[:, 1::2]
    qweight_packed = qweight_packed.to(torch.int8).view(torch.int32)
    qzeros_packed = qzeros_unpacked[:, ::2] * 16 + qzeros_unpacked[:, 1::2]
    qzeros_packed = qzeros_packed.to(torch.int8).view(torch.int32)
    
    ans = marlin_int4_fp8_preprocess_torch(qweight_unpacked, qzeros_unpacked, group_size)
    K, N = qweight_packed.shape
    num_groups = qzeros_packed.shape[0]
    
    output = torch.zeros_like(ans).to(torch.int32)
    
    qweight_ptr = ctypes.cast(qweight_packed.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    qzeros_ptr = ctypes.cast(qzeros_packed.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    if device == "cuda":
        torch_marlin_int4_fp8_preprocess_time = performance.CudaProfile((marlin_int4_fp8_preprocess_torch, (qweight_unpacked, qzeros_unpacked, group_size)))  # 以毫秒为单位
        lib.marlin_int4_fp8_preprocess_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_marlin_int4_fp8_preprocess_time = \
        performance.CudaProfile((lib.marlin_int4_fp8_preprocess_nv, (output_ptr, qweight_ptr, qzeros_ptr, K, N, num_groups)))
    
    performance.logBenchmark(torch_marlin_int4_fp8_preprocess_time, custom_marlin_int4_fp8_preprocess_time)
    
    assert (output == ans).all()

def marlin_int4_fp8_preprocess_torch_without_zp(qweight_unpacked):
    torch_res = torch.where(
        qweight_unpacked >= 8, qweight_unpacked - 8, 15 - qweight_unpacked
    )
    torch_res = torch_res[:, ::2] * 16 + torch_res[:, 1::2]
    torch_res = torch_res.to(torch.int8).view(torch.int32)

    return torch_res
    
def test_without_zp(size_k, size_n, group_size, device):
    print(
        f"Testing marlin int4 fp8 preprocess without zp on {device} with size_k-size_n-group_size:[{size_k, size_n, group_size}]"
    )
    qweight_unpacked = torch.randint(
        0, 16, size=(size_k, size_n), dtype=torch.int32, device=device
    )
    
    qweight_packed = qweight_unpacked[:, ::2] * 16 + qweight_unpacked[:, 1::2]
    qweight_packed = qweight_packed.to(torch.int8).view(torch.int32)
    
    
    ans = marlin_int4_fp8_preprocess_torch_without_zp(qweight_unpacked)
    K, N = qweight_packed.shape
    num_groups = -1
    
    output = torch.zeros_like(ans).to(torch.int32)
    
    qweight_ptr = ctypes.cast(qweight_packed.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    qzeros_ptr = None
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    if device == "cuda":
        torch_marlin_int4_fp8_preprocess_time = performance.CudaProfile((marlin_int4_fp8_preprocess_torch_without_zp, (qweight_unpacked, )))  # 以毫秒为单位
        lib.marlin_int4_fp8_preprocess_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_marlin_int4_fp8_preprocess_time = \
        performance.CudaProfile((lib.marlin_int4_fp8_preprocess_nv, (output_ptr, qweight_ptr, qzeros_ptr, K, N, num_groups)))
    
    performance.logBenchmark(torch_marlin_int4_fp8_preprocess_time, custom_marlin_int4_fp8_preprocess_time)
    
    assert (output == ans).all()
    
parser = argparse.ArgumentParser(description="Test marlin_int4_fp8_preprocess on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
    (2048, 2048, 128),
    (1024, 2048, 128),
    (1024, 512, 128),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for size_k, size_n, group_size in test_cases:
    test(size_k, size_n, group_size, args.device)
    test_without_zp(size_k, size_n, group_size, args.device)
    