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

def permute_rows(
    q_w: torch.Tensor,
    w_ref: torch.Tensor,
    group_size: int,
    test_perm: Optional[torch.Tensor] = None,
):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = torch.zeros((k_size,), dtype=torch.int32)
    for i in range(k_size):
        g_idx[i] = i // group_size

    # Simulate act_order by doing a random permutation on K
    rand_perm = test_perm if test_perm is not None else torch.randperm(k_size)

    g_idx = g_idx[rand_perm].contiguous()
    q_w = q_w[rand_perm, :].contiguous()
    w_ref = w_ref[rand_perm, :].contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )

# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py
def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: Optional[int],
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    assert (
        quant_type.is_integer()
    ), "Floating point quantization may work but has not been tested"
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided "
        "(-1 group_size is channelwise)"
    )

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [groupsize, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values

    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.Tensor([1.0]).to(w.device)  # unscaled case
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = (
                torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
            )
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

    # Quantize
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )

SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

def gptq_quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: Optional[torch.Tensor] = None,
):
    size_k, _ = w.shape

    assert w.is_floating_point(), "w must be float"
    assert (
        quant_type in SUPPORTED_GPTQ_QUANT_TYPES
    ), f"Unsupported gptq type = {quant_type}"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)

    # Apply act_order
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        assert (
            group_size < size_k
        ), "For act_order, groupsize = {} must be less than size_k = {}".format(
            group_size, size_k
        )

        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size, test_perm)

    return w_ref, w_q, w_s, g_idx, rand_perm


def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    orig_device = q_w.device

    sort_indices = torch.argsort(g_idx).to(dtype=torch.int32)  # Sort based on g_idx

    g_idx = g_idx[sort_indices].contiguous()
    q_w = q_w[sort_indices, :].contiguous()

    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )
    
GPTQ_MARLIN_TILE = 16
    
def marlin_permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w

def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits

def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed


def get_weight_perm(num_bits: int):
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm

def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single

def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s

def marlin_quantize(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: Optional[torch.Tensor] = None,
):
    size_k, size_n = w.shape
    num_bits = quant_type.size_bits

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        w, quant_type, group_size, act_order, test_perm
    )

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Reformat to marlin
    weight_perm = get_weight_perm(num_bits)
    marlin_q_w = marlin_weights(q_w, size_k, size_n, num_bits, weight_perm)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size)

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list

def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res

def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp

def awq_marlin_quantize(w: torch.Tensor, quant_type: ScalarType, group_size: int):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Detect num groups
    assert size_k % group_size == 0
    num_groups = size_k // group_size

    # Quantize with zp
    w_ref, q_w, s, zp = quantize_weights(w, quant_type, group_size, zero_points=True)

    # Reformat to marlin
    weight_perm = get_weight_perm(quant_type.size_bits)
    marlin_q_w = marlin_weights(q_w, size_k, size_n, quant_type.size_bits, weight_perm)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size)
    marlin_zp = marlin_zero_points(zp, num_groups, size_n, quant_type.size_bits)

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, marlin_zp]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list

def stack_and_dev(tensors: list[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)

def _setup_moe_weights(e, n, k, quant_type, group_size, act_order, dtype):
    """Set up quantized MoE weights for a single gate (e experts, output n, input k)."""
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    w = torch.randn((e, n, k), device="cuda", dtype=dtype) / 20

    w_ref_l = []
    qweight_l = []
    scales_l = []
    zeros_l = []
    g_idx_l = []
    sort_indices_l = []

    for i in range(e):
        if has_zp:
            w_ref, qweight, scales, zeros = awq_marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size
            )
            w_ref_l.append(w_ref.T)
            qweight_l.append(qweight)
            scales_l.append(scales)
            zeros_l.append(zeros)
        else:
            test_perm = torch.randperm(k)
            w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size, act_order, test_perm
            )
            w_ref_l.append(w_ref.T)
            qweight_l.append(qweight)
            scales_l.append(scales)
            g_idx_l.append(g_idx)
            sort_indices_l.append(sort_indices)

    w_ref = stack_and_dev(w_ref_l)
    qweight = stack_and_dev(qweight_l).contiguous()
    scales = stack_and_dev(scales_l)
    g_idx = stack_and_dev(g_idx_l) if g_idx_l else None
    sort_indices = stack_and_dev(sort_indices_l) if sort_indices_l else None
    zeros = stack_and_dev(zeros_l) if zeros_l else None

    return w_ref, qweight, scales, zeros, g_idx, sort_indices

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y

def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    
    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    
    sorted_ids_ptr = ctypes.cast(sorted_ids.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    expert_ids_ptr = ctypes.cast(expert_ids.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    num_tokens_post_pad_ptr = ctypes.cast(num_tokens_post_pad.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    expert_map_ptr = None
    topk_ids_ptr = ctypes.cast(topk_ids.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    num_topk_ids = topk_ids.numel()
    sorted_token_ids_size_0 = sorted_ids.shape[0]
    topk_ids_size_1 = topk_ids.shape[1]
    
    dataType = 0 
    if topk_ids.dtype == torch.int32:
        dataType = 0 
    elif topk_ids.dtype == torch.int64:
        dataType = 1
    if topk_ids.device == "cuda":
        lib.moe_align_block_size_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
    lib.moe_align_block_size_nv(sorted_ids_ptr,
                                        expert_ids_ptr,
                                        num_tokens_post_pad_ptr,
                                        expert_map_ptr,
                                        topk_ids_ptr,
                                        num_experts,
                                        block_size,
                                        num_topk_ids,
                                        sorted_token_ids_size_0, 
                                        topk_ids_size_1,
                                        dataType)
    
    return sorted_ids, expert_ids, num_tokens_post_pad

def _get_scalar_type(num_bits: int, has_zp: bool):
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def vllm_moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
):
    device = a.device
    dataType = 0
    if a.dtype == torch.float16:
        dataType = 0
    elif a.dtype == torch.bfloat16:
        dataType = 1
    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.empty((size_m * top_k, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # Determine activation ordering
    has_act_order = (
        g_idx_or_none is not None
        and perm_or_none is not None
        and g_idx_or_none.numel() > 0
        and perm_or_none.numel() > 0
        and g_idx_or_none.size(-1) > 0
        and perm_or_none.size(-1) > 0
    )


    
    c_ptr = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    a_ptr = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_q_weight_ptr = ctypes.cast(b_q_weight.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    b_bias_ptr = ctypes.cast(b_bias_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if b_bias_or_none is not None else None
    b_scales_ptr = ctypes.cast(b_scales.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    global_scale_ptr = ctypes.cast(global_scale_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if global_scale_or_none is not None else None
    b_zeros_ptr = ctypes.cast(b_zeros_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if b_zeros_or_none is not None else None
    g_idx_ptr = ctypes.cast(g_idx_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if g_idx_or_none is not None else None
    perm_ptr = ctypes.cast(perm_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if perm_or_none is not None else None
    
    sorted_token_ids_ptr = ctypes.cast(sorted_token_ids.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    expert_ids_ptr = ctypes.cast(expert_ids.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    num_tokens_post_padded_ptr = ctypes.cast(num_tokens_post_padded.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    topk_weights_ptr = ctypes.cast(topk_weights.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    if device == "cuda":
        lib.vllm_moe_wna16_marlin_gemm_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), # c
            ctypes.POINTER(ctypes.c_void_p), # a 
            ctypes.POINTER(ctypes.c_void_p), # b_q_weight
            ctypes.POINTER(ctypes.c_void_p), # b_bias
            ctypes.POINTER(ctypes.c_void_p), # b_scales
            ctypes.POINTER(ctypes.c_void_p), # a_scales
            ctypes.POINTER(ctypes.c_void_p), # global_scale
            ctypes.POINTER(ctypes.c_void_p), # b_zeros
            ctypes.POINTER(ctypes.c_void_p), # g_idx
            ctypes.POINTER(ctypes.c_void_p), # perm
            ctypes.POINTER(ctypes.c_void_p), # sorted_token_ids
            ctypes.POINTER(ctypes.c_void_p), # expert_ids
            ctypes.POINTER(ctypes.c_void_p), # num_tokens_past_padded
            ctypes.POINTER(ctypes.c_void_p), # topk_weights
            ctypes.c_int, # moe_block_size
            ctypes.c_int, # top_k
            ctypes.c_bool, # mul_topk_weights
            ctypes.c_int64, # b_type_id,
            ctypes.c_int, # size_m
            ctypes.c_int, # size_n 
            ctypes.c_int, # size_k 
            ctypes.c_bool, # is_k_full
            ctypes.c_bool, # use_atomic_add
            ctypes.c_bool, # use_fp32_reduce
            ctypes.c_bool, # is_zp_float
            ctypes.c_int, # sorted_token_ids_size_0
            ctypes.c_int, # b_q_weight_size_0
            ctypes.c_int, # b_q_weight_size_1
            ctypes.c_int, # b_q_weight_size_2
            ctypes.c_int, # b_scales_size_1
            ctypes.c_int, # b_scales_size_2
            ctypes.c_int # dataType
        ]
    
    lib.vllm_moe_wna16_marlin_gemm_nv(
        c_ptr,
        a_ptr,
        b_q_weight_ptr,
        b_bias_ptr,
        b_scales_ptr,
        None,
        global_scale_ptr,
        b_zeros_ptr,
        g_idx_ptr,
        perm_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        topk_weights_ptr,
        moe_block_size,
        top_k,
        mul_topk_weights,
        ctypes.c_int64(b_q_type.id),
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        sorted_token_ids.shape[0],
        b_q_weight.shape[0],
        b_q_weight.shape[1],
        b_q_weight.shape[2],
        b_scales.shape[1],
        b_scales.shape[2],
        dataType
    )
    
    return c

def sglang_moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    device = a.device
    dataType = 0
    if a.dtype == torch.float16:
        dataType = 0
    elif a.dtype == torch.bfloat16:
        dataType = 1
    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.empty((size_m * top_k, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # Determine activation ordering
    has_act_order = (
        g_idx_or_none is not None
        and perm_or_none is not None
        and g_idx_or_none.numel() > 0
        and perm_or_none.numel() > 0
        and g_idx_or_none.size(-1) > 0
        and perm_or_none.size(-1) > 0
    )

    # Determine has_zp
    has_zp = b_zeros_or_none is not None and b_zeros_or_none.numel() > 0

    # Determine has_bias
    has_bias = b_bias_or_none is not None

    # Derive num_groups and group_size from b_scales
    num_groups = b_scales.size(1)

    
    c_ptr = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    a_ptr = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_q_weight_ptr = ctypes.cast(b_q_weight.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    b_bias_ptr = ctypes.cast(b_bias_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if b_bias_or_none is not None else None
    b_scales_ptr = ctypes.cast(b_scales.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    global_scale_ptr = ctypes.cast(global_scale_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if global_scale_or_none is not None else None
    b_zeros_ptr = ctypes.cast(b_zeros_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if b_zeros_or_none is not None else None
    g_idx_ptr = ctypes.cast(g_idx_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if g_idx_or_none is not None else None
    perm_ptr = ctypes.cast(perm_or_none.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if perm_or_none is not None else None
    
    sorted_token_ids_ptr = ctypes.cast(sorted_token_ids.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    expert_ids_ptr = ctypes.cast(expert_ids.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    num_tokens_post_padded_ptr = ctypes.cast(num_tokens_post_padded.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    topk_weights_ptr = ctypes.cast(topk_weights.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    if device == "cuda":
        lib.sglang_moe_wna16_marlin_gemm_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), # c
            ctypes.POINTER(ctypes.c_void_p), # a 
            ctypes.POINTER(ctypes.c_void_p), # b_q_weight
            ctypes.POINTER(ctypes.c_void_p), # b_bias
            ctypes.POINTER(ctypes.c_void_p), # b_scales
            ctypes.POINTER(ctypes.c_void_p), # global_scale
            ctypes.POINTER(ctypes.c_void_p), # b_zeros
            ctypes.POINTER(ctypes.c_void_p), # g_idx
            ctypes.POINTER(ctypes.c_void_p), # perm
            ctypes.POINTER(ctypes.c_void_p), # sorted_token_ids
            ctypes.POINTER(ctypes.c_void_p), # export_ids
            ctypes.POINTER(ctypes.c_void_p), # num_tokens_past_padded
            ctypes.POINTER(ctypes.c_void_p), # topk_weights
            ctypes.c_int, # moe_block_size
            ctypes.c_int, # top_k
            ctypes.c_bool, # mul_topk_weights
            ctypes.c_bool, # is_ep
            ctypes.c_int64, # b_type_id
            ctypes.c_int, # size_m
            ctypes.c_int, # size_n 
            ctypes.c_int, # size_k 
            ctypes.c_bool, # has_act_order
            ctypes.c_bool, # has_bias,
            ctypes.c_bool, # is_k_full
            ctypes.c_bool, # has_zp
            ctypes.c_int, # num_groups
            ctypes.c_bool, # use_atomic_add
            ctypes.c_bool, # use_fp32_reduce
            ctypes.c_bool, # is_zp_float
            ctypes.c_int, # sorted_token_ids_size_0
            ctypes.c_int, # b_q_weight_size_1
            ctypes.c_int, # b_q_weight_size_2
            ctypes.c_int, # c_size_0
            ctypes.c_int # dataType
        ]
    lib.sglang_moe_wna16_marlin_gemm_nv(
        c_ptr,
        a_ptr,
        b_q_weight_ptr,
        b_bias_ptr,
        b_scales_ptr,
        global_scale_ptr,
        b_zeros_ptr,
        g_idx_ptr,
        perm_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        topk_weights_ptr,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        ctypes.c_int64(b_q_type.id),
        size_m,
        size_n,
        size_k,
        has_act_order,
        has_bias,
        is_k_full,
        has_zp,
        num_groups,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        sorted_token_ids.shape[0],
        b_q_weight.shape[1],
        b_q_weight.shape[2],
        c.shape[0],
        dataType
    )

    return c


def test_moe_wna16_marlin_gemm(
    m, n, k, e, topk, dtype, group_size, act_order, quant_type, device
):
    print(
        f"Testing moe_wna16_marlin_gemm on {device} with m:{m}, n:{n}, k:{k}, topk:{topk}, dtype:{dtype}, group_size:{group_size}, act_order:{act_order}, quant_type:{quant_type}, device:{device}"
    )
    torch.manual_seed(0)

    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    a = torch.randn((m, k), device=device, dtype=dtype) / 10

    # Set up quantized weights for first gemm (gate_up: output 2*n, input k)
    w_ref1, qweight1, scales1, zeros1, g_idx1, sort_indices1 = _setup_moe_weights(
        e, 2 * n, k, quant_type, group_size, act_order, dtype
    )

    # Compute block_size_m
    for block_size_m in [8, 16, 32, 48, 64]:
        if m * topk / e / block_size_m < 0.9:
            break

    # Align tokens
    score = torch.randn((m, e), device=device, dtype=dtype)
    score_softmax = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, e
    )

    # Workspace
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    max_workspace_size = (max(2 * n, k) // 64) * (
        sorted_token_ids.size(0) // block_size_m
    )
    max_workspace_size = min(max_workspace_size, sms * 4)
    

    use_atomic_add = (
        dtype == torch.half or torch.cuda.get_device_capability(device)[0] >= 9
    )

    scalar_type = _get_scalar_type(4, has_zp)

    moe_block_size = block_size_m
    size_m = m
    size_n = 2 * n
    size_k = k
    is_ep = False
    mul_topk_weights = False
    is_k_full= True
    use_atomic_add=use_atomic_add
    use_fp32_reduce=True
    is_zp_float=False
    c_sglang = torch.zeros((size_m * topk, size_n), dtype=dtype, device=device)
    c_vllm = torch.zeros((size_m * topk, size_n), dtype=dtype, device=device)
    #print("start", c_sglang.flatten()[7:17], c_vllm.flatten()[7:17], c_sglang.flatten()[7:17] - c_vllm.flatten()[7:17])
    c_sglang = sglang_moe_wna16_marlin_gemm(
        a,
        c_sglang,
        qweight1,
        None, # b_bias
        scales1,
        None, # global_scale
        zeros1,
        g_idx1,
        sort_indices1,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        topk,
        mul_topk_weights,
        is_ep,
        scalar_type,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    torch.cuda.synchronize()

    # --- Check bitwise equality with AOT kernel ---
    c_vllm = vllm_moe_wna16_marlin_gemm(
        a,
        c_sglang,
        qweight1,
        None, # b_bias
        scales1,
        None, # global_scale
        zeros1,
        g_idx1,
        sort_indices1,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        topk,
        mul_topk_weights,
        scalar_type,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )
    torch.cuda.synchronize()
    #print(c_sglang.flatten()[7:17], c_vllm.flatten()[7:17], c_sglang.flatten()[7:17] - c_vllm.flatten()[7:17])
    torch.testing.assert_close(c_sglang, c_vllm, rtol=0, atol=0)
    vllm_moe_wna16_marlin_gemm_time = performance.CudaProfile((vllm_moe_wna16_marlin_gemm, (
        a,
        c_sglang,
        qweight1,
        None, # b_bias
        scales1,
        None, # global_scale
        zeros1,
        g_idx1,
        sort_indices1,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        topk,
        mul_topk_weights,
        scalar_type,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    ))) 
    sglang_moe_wna16_marlin_gemm_time = performance.CudaProfile((sglang_moe_wna16_marlin_gemm, (
        a,
        c_sglang,
        qweight1,
        None, # b_bias
        scales1,
        None, # global_scale
        zeros1,
        g_idx1,
        sort_indices1,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        topk,
        mul_topk_weights,
        is_ep,
        scalar_type,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    ))) 
    performance.logBenchmark(sglang_moe_wna16_marlin_gemm_time, vllm_moe_wna16_marlin_gemm_time)

def generate_test_cases():
    m_list = [1, 123]
    n_list = [128, 1024]
    k_list = [256]
    e_list = [4]
    topk_list = [2]
    dtype_list = [torch.float16, torch.bfloat16]
    group_size_list = [128]
    act_order_list = [False, True]
    quant_type_list = [scalar_types.uint4, scalar_types.uint4b8]

    all_combinations = itertools.product(
        m_list,
        n_list,
        k_list,
        e_list,
        topk_list,
        dtype_list,
        group_size_list,
        act_order_list,
        quant_type_list,
    )

    def is_valid(m, n, k, e, topk, dtype, group_size, act_order, quant_type):
        has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]
        if act_order:
            if group_size == -1 or group_size == k:
                return False
            if has_zp:
                return False
        if group_size > 0 and k % group_size != 0:
            return False
        return True

    return [case for case in all_combinations if is_valid(*case)]


test_cases = generate_test_cases()

parser = argparse.ArgumentParser(description="Test marlin_int4_fp8_preprocess on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for m, n, k, e, topk, dtype, group_size, act_order, quant_type in test_cases:
    test_moe_wna16_marlin_gemm(m, n, k, e, topk, dtype, group_size, act_order, quant_type, args.device)
    
