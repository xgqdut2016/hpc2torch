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

def random_sample_torch(data, random_val, topp, topk, voc, temperature):
    if topp > 0 and topk > 1:
        sorted_vals, sorted_indices = torch.sort(data, descending=True)
        scaled_vals = (sorted_vals - sorted_vals[0]) / temperature
        try:
            probs = torch.softmax(scaled_vals, dim=0)
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e):
                scaled_vals = scaled_vals.to(torch.float32)
                probs = torch.softmax(scaled_vals, dim=0)
            else:
                raise
        cum_probs = torch.cumsum(probs, dim=0)

        k_index = min(topk, voc) - 1
        threshold = min(cum_probs[k_index], topp) * random_val

        try:
            idx = torch.searchsorted(cum_probs, threshold)
        except Exception:
            # Fallback for manual search if torch.searchsorted is not supported
            indices = (cum_probs >= threshold).nonzero(as_tuple=True)[0]
            idx = (
                indices[0]
                if indices.numel() > 0
                else torch.tensor(len(cum_probs) - 1, device=cum_probs.device)
            )
        return sorted_indices[idx]

    return torch.argmax(data)

def test(device, voc, random_val, topp, topk, temperature):
    byteSize = 2
    x_dtype = torch.float16
    if byteSize == 4:
        x_dtype = torch.float32
    print(
        f"Testing RandomSample on {device} with voc:{voc} , topk:{topk}, topp:{topp}, random_val:{random_val}, temperature:{temperature}, dtype:{x_dtype}"
    )
    if device == "kunlun":
        torch_device = "cuda"
    else:
        torch_device = device
    data = torch.arange(voc).float() * 0.0001
    
    _perm = torch.randperm(voc)
    
    data = data[_perm].to(x_dtype).to(torch_device)
    
    if(torch_device == 'mlu' or torch_device == 'npu'):
        
        indices = torch.zeros([1], dtype = torch.int64, device = torch_device)
    elif device == "kunlun":
        indices = torch.zeros([1], dtype = torch.int32, device = torch_device)
    else:
        
        indices = torch.zeros([1], dtype = torch.uint64, device = torch_device)
    
    ans = random_sample_torch(data, random_val, topp, topk, voc, temperature)
    
    probs_ptr = ctypes.cast(data.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    result_ptr = ctypes.cast(indices.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    if device == "sdaa":
        torch_randomSample_time = performance.TecoProfile((random_sample_torch, (data, random_val, topp, topk, voc, temperature)))
        lib.randomSample_teco.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int
        ]
        custom_randomSample_time = \
        performance.TecoProfile((lib.randomSample_teco, (result_ptr, 
                                probs_ptr, 
                                random_val,
                                topp,
                                voc,
                                topk,
                                temperature, byteSize)))
    elif device == "kunlun":
        torch_randomSample_time = performance.KunlunProfile((random_sample_torch, (data, random_val, topp, topk, voc, temperature)))
        lib.randomSample_kunlun.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int
        ]
        custom_randomSample_time = \
        performance.KunlunProfile((lib.randomSample_kunlun, (result_ptr, 
                                probs_ptr, 
                                random_val,
                                topp,
                                voc,
                                topk,
                                temperature, byteSize)))
    performance.logBenchmark(torch_randomSample_time, custom_randomSample_time)
    
    ans = ans.to("cpu")
    index = indices[0].to("cpu").to(ans.dtype)
    print(ans, index)
    assert index == ans or data[ans] == data[index]

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test randomSample on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', "sdaa", "kunlun"], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # voc, random_val, topp, topk, temperature
        (512, 0.8, 0.8, 3, 0.5),
        (4096, 0.05, 0.9, 5, 1.0),
        (16384, 0.15, 0.85, 10, 2.0),
        (512, 0.08, 0, 3, 0.5),
        (4096, 0.5, 0.9, 1, 1.0),
        (16384, 0.15, 0, 1, 2.0),
        (16384, 0.15, 0, 1, 2.0),
        (32000, 0.08, 0.8, 50, 1.0),
        (32000, 0.08, 1.0, 25, 1.0),
        # (119696, 0.01, 1.0, 100, 1.0),
    ]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'sdaa':
    import torch_sdaa
if args.device == 'kunlun':
    import torch_xmlir
# 执行过滤后的测试用例
for voc, random_val, topp, topk, temperature in test_cases:
    test(args.device, voc, random_val, topp, topk, temperature)
