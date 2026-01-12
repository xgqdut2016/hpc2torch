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
def quantWeights(w: torch.Tensor, symmetric, axis):
    """
    对权重矩阵 w ∈ [K, N] 做 per-channel (按列) 量化。
    返回:
      w_packed: int8 量化权重，形状 [K, N]
      w_scale:  每列的scale，形状 [1, N]，dtype与w相同
      w_zero:   每列的zero point，形状 [1, N]，dtype与w相同
    """
    assert w.dim() == 2, "w must be [K, N]"
    if symmetric:
        # 对称量化：zero=0, 只用最大绝对值
        w_abs_max = torch.max(w.abs(), dim=axis, keepdim=True)[0]

        # 避免除 0
        w_scale = w_abs_max / 127.0
        w_scale = torch.clamp(w_scale, min=1e-8)

        # 计算量化值 q = round(w / scale)
        w_q = torch.round(w / w_scale)

        # 限制到 [-128, 127]
        w_q = torch.clamp(w_q, -128, 127)

        # 转 int8
        w_packed = w_q.to(torch.int8)

        # 对称量化 zero 固定为 0
        w_zero = None

        return w_packed, w_scale.to(w.dtype), w_zero
    else:
        # 计算每列的最小值和最大值
        w_min = w.min(dim=axis, keepdim=True)[0]
        w_max = w.max(dim=axis, keepdim=True)[0]

        # 避免除以零
        w_scale = (w_max - w_min) / 255.0
        w_scale = torch.clamp(w_scale, min=1e-8)

        # 计算zero point
        w_zero = -w_min / w_scale - 128.0

        # 计算量化值
        w_q = torch.round(w / w_scale + w_zero)

        # 限制范围[-128, 127]
        w_q = torch.clamp(w_q, -128, 127)

        # 转为int8
        w_packed = w_q.to(torch.int8)

        return w_packed, w_scale.to(w.dtype), w_zero.to(w.dtype)
def linearFunction(c, bias, x, w, alpha, beta):
    if bias is not None:
        ans = (
            alpha * torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)
            + beta * c
            + bias
        )
    else:
        ans = (
            alpha * torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)
            + beta * c
        )
    return ans
def computeQuant(
        device,
        x, 
        symmetric
):
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    byteSize = 4
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        byteSize = 2
    elif x.dtype == torch.float32:
        byteSize = 4
    M, K = x.shape
    x_packed = torch.zeros((M, K), device=device, dtype=torch.int8, requires_grad=False)
    x_packed_ptr = ctypes.cast(x_packed.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    x_scale = torch.rand((M, 1), device=device, dtype=x.dtype, requires_grad=False)
    x_scale_ptr = ctypes.cast(x_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    x_zero = None
    x_zero_ptr = None
    if symmetric == False:
        x_zero = torch.rand((M, 1), device=device, dtype=x.dtype, requires_grad=False)
        x_zero_ptr = ctypes.cast(x_zero.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    if device == "cuda":
        lib.PerChannelQuantI8_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        lib.PerChannelQuantI8_nv(x_packed_ptr, x_scale_ptr, x_zero_ptr, x_ptr, M, K, byteSize)
    return x_packed, x_scale, x_zero
def test(
    x_shape,
    w_shape,
    symmetric,
    bias_exit,
    y_shape,
    alpha,
    beta,
    device
    ):
    byteSize = 4
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    atol = 1e-3; rtol = 5e-2
    if test_dtype == torch.float16 or test_dtype == torch.bfloat16:
        atol = 1e-3; rtol = 5e-2
    elif test_dtype == torch.float32:
        atol = 3e-5; rtol = 5e-3
    print(
        f"Testing Quant Linear on {device} with x_shape:{x_shape}, w_shape:{w_shape}, symmetric:{symmetric}, bias:{bias_exit}, alpha:{alpha}, beta:{beta}, dtype:{test_dtype}"
    )
    M, K = x_shape
    N = w_shape[1]
    if bias_exit:
        bias = torch.rand((N, ), device=device, dtype=test_dtype, requires_grad=False)
        bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    else:
        bias = None
        bias_ptr = None
    x = torch.rand(x_shape, device=device, dtype=test_dtype, requires_grad=False)
    w = torch.rand(w_shape, device=device, dtype=test_dtype, requires_grad=False)
    y = torch.rand(y_shape, device=device, dtype=test_dtype, requires_grad=False)
    output = torch.zeros(y_shape, device=device, dtype=test_dtype, requires_grad=False)

    
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    x_packed, x_scale, x_zero = computeQuant(
            device,
            x, 
            symmetric
    )
    w_packed, w_scale, w_zero = computeQuant(
            device,
            w.t().contiguous(), 
            symmetric
    )
    
    w_packed_t = w_packed.t().contiguous()
    w_scale_t = w_scale.t().contiguous()
    # x_p, x_s, x_z = quantWeights(x, symmetric, 1)
    # w_p, w_s, w_z = quantWeights(w, symmetric, 0)
    # print(x_p - x_packed)
    # print(w_p - w_packed_t)
    
    x_packed_ptr = ctypes.cast(x_packed.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    x_scale_ptr = ctypes.cast(x_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    w_packed_ptr = ctypes.cast(w_packed_t.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    w_scale_ptr = ctypes.cast(w_scale_t.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    if symmetric == False:
        w_zero_t = w_zero.t().contiguous()
        x_zero_ptr = ctypes.cast(x_zero.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        w_zero_ptr = ctypes.cast(w_zero_t.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    else:
        x_zero_ptr = None
        w_zero_ptr = None
    if device == "cuda":
        torch_quant_linear_time = performance.CudaProfile((linearFunction, (y, bias, x, w, alpha, beta))) 
        lib.linear_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]
        
        custom_quant_linear_time = \
        performance.CudaProfile((lib.linear_nv, (output_ptr, y_ptr, bias_ptr, 
                      x_packed_ptr, x_scale_ptr, x_zero_ptr, 
                      w_packed_ptr, w_scale_ptr, w_zero_ptr, 
                      M, K, N, alpha, beta, byteSize)))
        
    performance.logBenchmark(torch_quant_linear_time, custom_quant_linear_time)
    # 将结果转换回 PyTorch 张量以进行比较
    assert torch.allclose(linearFunction(y, bias, x, w, alpha, beta), output, atol=atol, rtol=rtol)
    tmpa = linearFunction(y, bias, x, w, alpha, beta).to('cpu').detach().numpy().flatten()
    
    tmpb = output.to('cpu').detach().numpy().flatten()
    # print(x_packed)
    # print(w_packed.t().contiguous())
    
    # print(output, output.dtype)
    # print(tmpa)
    # print(tmpb)
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test layernorm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # x_shape, w_shape, symmetric, bias_exit, y_shape, alpha, beta
        ((8, 8), (8, 8), True, True, (8, 8), 1.0, 0.0),
        ((128, 512), (512, 1024), True, False, (128, 1024), 1.0, 0.0),
        ((128, 128), (128, 128), False, True, (128, 128), 2.0, 1.0),
        ((256, 1024), (1024, 2048), True, False, (256, 2048), 1.0, 1.0),
        ((256, 2048), (2048, 1024), False, True, (256, 1024), 1.5, 2.5),
        ((1024, 2048), (2048, 4096), True, False, (1024, 4096), 1.0, 0.0),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for x_shape, w_shape, symmetric, bias_exit, y_shape, alpha, beta in test_cases:
    test(x_shape, w_shape, symmetric, bias_exit, y_shape, alpha, beta, args.device)