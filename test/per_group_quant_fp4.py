import torch
import ctypes
import numpy as np
import torch.nn as nn
from functools import partial
import argparse

from utils import performance

# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

group_size = 16
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def cast_from_fp4(x, m, n):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([E2M1_TO_FLOAT32[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out.to(x.device)


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")
    

def per_group_quant_fp4_torch(x, global_scale):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // group_size, group_size))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def recover_swizzled_scales(scale, m, n):
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // group_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]

    
def test(
    x_shape,
    device
    ):
    
    dataType = 0
    test_dtype = torch.float16
    if (dataType == 0):
        test_dtype = torch.float16
    elif (dataType == 1):
        test_dtype = torch.bfloat16
    atol = 1e-3; rtol = 1e-5
    if test_dtype == torch.float16:
        atol = 1e-3; rtol = 1e-5
    elif test_dtype == torch.bfloat16:
        atol = 1.6e-2; rtol = 1e-5
    print(
        f"Testing Per Group Quant Fp4 on {device} with x_shape:{x_shape}, dtype:{test_dtype}"
    )
    
    x = torch.rand(x_shape, dtype=test_dtype, device=device)
    M, N = x_shape

    input_global_scale = torch.zeros(1, dtype=torch.float32, device=device)
    input_global_scale[0] = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(x.flatten(), dim=-1)
    ).to(torch.float32)
    output = torch.zeros((M, N // 2), dtype=torch.uint8, device=device)

    rounded_m = ((M + 128 - 1) // 128) * 128
    scale_n = N // group_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4

    output_scale = torch.zeros((rounded_m, rounded_n // 4), dtype=torch.int32, device=device)
    
    out_ref, scale_ref = per_group_quant_fp4_torch(x, input_global_scale)
    
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    input_global_scale_ptr = ctypes.cast(input_global_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_scale_ptr = ctypes.cast(output_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    
    if device == "cuda":
        torch_quant_linear_time = performance.CudaProfile((per_group_quant_fp4_torch, (x, input_global_scale))) 
        lib.PerGroupQuantF4_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        
        custom_quant_linear_time = \
        performance.CudaProfile((lib.PerGroupQuantF4_nv, (
                      output_ptr, output_scale_ptr, x_ptr, input_global_scale_ptr,
                      M, N, dataType)))
        
    performance.logBenchmark(torch_quant_linear_time, custom_quant_linear_time)
    # 将结果转换回 PyTorch 张量以进行比较
    
    scale_ans = recover_swizzled_scales(output_scale.view(torch.float8_e4m3fn), M, N)
    out_ans = cast_from_fp4(output, M, N)

    tmpa_out = out_ans.float().detach().to('cpu').numpy().flatten()
    tmpb_out = out_ref.float().to('cpu').detach().numpy().flatten()
    tmpa_scale = scale_ans.float().detach().to('cpu').numpy().flatten()
    tmpb_scale = scale_ref.float().to('cpu').detach().numpy().flatten()
    
    atol_out = max(abs(tmpa_out - tmpb_out))
    rtol_out = atol_out / (max(abs(tmpb_out)) + 1e-8)
    print("output absolute error:%.4e"%(atol_out))
    print("output relative error:%.4e"%(rtol_out))

    atol_scale = max(abs(tmpa_scale - tmpb_scale))
    rtol_scale = atol_scale / (max(abs(tmpb_scale)) + 1e-8)
    print("scale absolute error:%.4e"%(atol_scale))
    print("scale relative error:%.4e"%(rtol_scale))

    assert (torch.allclose(out_ans, out_ref, atol=atol, rtol=rtol) and 
                torch.allclose(scale_ans, scale_ref, atol=atol, rtol=rtol))


# 解析命令行参数
parser = argparse.ArgumentParser(description="Test per group quant fp4 on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
    # x_shape
    (128, 64),
    (128, 128),
    (256, 64),
    (256, 128),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for x_shape in test_cases:
    test(x_shape, args.device)
