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

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y

def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            sorted_token_ids[current_pos : current_pos + num_expert_tokens] = (
                expert_tokens
            )

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size

            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[current_block : current_block + expert_blocks_needed] = (
                expert_id_new
            )

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    num_tokens_post_pad = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=topk_ids.device
    )

    return sorted_token_ids, expert_ids, num_tokens_post_pad

def _group_tokens_by_expert(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
) -> dict:
    num_blocks = valid_length // block_size
    expert_tokens: dict[int, list[int]] = {}

    for block_idx in range(num_blocks):
        expert_id = expert_ids[block_idx].item()
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, valid_length)

        block_tokens = sorted_ids[block_start:block_end]
        valid_tokens = block_tokens[block_tokens < total_tokens]

        if expert_id not in expert_tokens:
            expert_tokens[expert_id] = []
        expert_tokens[expert_id].extend(valid_tokens.tolist())
    return expert_tokens

def _verify_expert_level_sorting(
    actual_sorted_ids: torch.Tensor,
    golden_sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
):
    """
    Verify that actual_sorted_ids follows the correct expert-level sorting.
    The kerne limplementation may or may not preserve original token order
    in topk_ids in the final sorted_ids however this does not impact quality.
    """
    # Group tokens by expert from the golden implementation
    golden_expert_tokens = _group_tokens_by_expert(
        golden_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    actual_expert_tokens = _group_tokens_by_expert(
        actual_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    assert set(golden_expert_tokens.keys()) == set(actual_expert_tokens.keys()), (
        f"Expert IDs mismatch: golden={set(golden_expert_tokens.keys())}, "
        f"actual={set(actual_expert_tokens.keys())}"
    )

    for expert_id in golden_expert_tokens:
        golden_tokens = torch.tensor(
            golden_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        actual_tokens = torch.tensor(
            actual_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        assert torch.equal(
            torch.sort(golden_tokens)[0], torch.sort(actual_tokens)[0]
        ), (
            f"Expert {expert_id} token mismatch: "
            f"golden={golden_expert_tokens[expert_id]}, "
            f"actual={actual_expert_tokens[expert_id]}"
        )

def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
        should be padded to a multiple of block_size,
    - ignore_invalid_experts: A flag indicating whether to ignore invalid
        experts. When False, all expert_ids in topk_ids will participate in
        counting and ranking, but invalid experts in expert_ids will be marked
        as -1. When True, all invalid expert_ids in topk_ids will be ignored
        and will not participate in counting or ranking, and there will be no
        -1 in expert_ids.

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
    
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
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
    expert_map_ptr = ctypes.cast(expert_map.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if ignore_invalid_experts else None
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

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]
    
    return sorted_ids, expert_ids, num_tokens_post_pad

def test_moe_align_block_size(
    m: int, topk: int, num_experts: int, block_size: int, pad_sorted_ids: bool, device
):
    print(
        f"Testing Moe align block size on {device} with m:{m}, topk:{topk}, block_size:{block_size}, pad_sorted_ids:{pad_sorted_ids}, device:{device}"
    )
    """Test moe_align_block_size without expert mapping"""
    topk_ids = torch.zeros((m, topk), device=device, dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device=device)[:topk]
        topk_ids[i] = experts
    
    actual_sorted_ids, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        pad_sorted_ids=pad_sorted_ids,
    )
    
    golden_sorted_ids, golden_expert_ids, golden_num_tokens = (
        torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            pad_sorted_ids=pad_sorted_ids,
        )
    )
    
    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)

    # For sorted_token_ids, verify block-level correctness rather than exact
    # order Tokens within each expert's blocks can be in any order, but expert
    # regions must be correct
    _verify_expert_level_sorting(
        actual_sorted_ids,
        golden_sorted_ids,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        m * topk,
    )
    
    total_tokens = m * topk
    assert actual_num_tokens.item() % block_size == 0, (
        "num_tokens_post_pad should be divisible by block_size"
    )
    assert actual_num_tokens.item() >= total_tokens, (
        "num_tokens_post_pad should be at least total_tokens"
    )
    valid_tokens = actual_sorted_ids[actual_sorted_ids < total_tokens]
    assert len(valid_tokens) == total_tokens, (
        f"Should have exactly {total_tokens} valid tokens, got {len(valid_tokens)}"
    )
    actual_num_blocks = cdiv(int(actual_num_tokens.item()), block_size)
    assert (actual_expert_ids[:actual_num_blocks] >= 0).all() and (
        actual_expert_ids[:actual_num_blocks] < num_experts
    ).all(), "expert_ids should contain valid expert indices"
    
    custom_moe_align_block_size_time = performance.CudaProfile((moe_align_block_size, (
        topk_ids,
        block_size,
        num_experts,
        None,
        pad_sorted_ids,
    ))) 
    torch_moe_align_block_size_time = performance.CudaProfile((torch_moe_align_block_size, (
            topk_ids,
            block_size,
            num_experts,
            None,
            pad_sorted_ids,
        ))) 
    performance.logBenchmark(torch_moe_align_block_size_time, custom_moe_align_block_size_time)

def test_moe_align_block_size_with_expert_map(
    m: int, topk: int, num_experts: int, block_size: int, device
):
    print(
        f"Testing Moe align block size with expert_map on {device} with m:{m}, topk:{topk}, block_size:{block_size}, pad_sorted_ids:{pad_sorted_ids}, device:{device}"
    )
    """Test moe_align_block_size with expert mapping (EP scenario)"""
    topk_ids = torch.zeros((m, topk), device=device, dtype=torch.int32)
    for i in range(m):
        experts = torch.randperm(num_experts, device=device)[:topk]
        topk_ids[i] = experts

    expert_map = torch.full((num_experts,), -1, device=device, dtype=torch.int32)
    local_experts = list(range(0, num_experts, 2))
    for i, expert_id in enumerate(local_experts):
        expert_map[expert_id] = i

    actual_sorted_ids, actual_expert_ids, actual_num_tokens = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
        expert_map=expert_map,
        ignore_invalid_experts=True,
    )
    golden_sorted_ids, golden_expert_ids, golden_num_tokens = (
        torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_size,
            num_experts=num_experts,
            expert_map=expert_map,
        )
    )

    torch.testing.assert_close(actual_num_tokens, golden_num_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual_expert_ids, golden_expert_ids, atol=0, rtol=0)
    _verify_expert_level_sorting(
        actual_sorted_ids,
        golden_sorted_ids,
        actual_expert_ids,
        block_size,
        actual_num_tokens.item(),
        m * topk,
    )
    ignore_invalid_experts = True
    custom_moe_align_block_size_time = performance.CudaProfile((moe_align_block_size, (
        topk_ids,
        block_size,
        num_experts,
        expert_map,
        False,
        ignore_invalid_experts,
    ))) 
    torch_moe_align_block_size_time = performance.CudaProfile((torch_moe_align_block_size, (
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            False,
        ))) 
    performance.logBenchmark(torch_moe_align_block_size_time, custom_moe_align_block_size_time)
    
    
parser = argparse.ArgumentParser(description="Test moe_align_block_size on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()  

# NUM_TOKENS = [1, 3, 256, 2256, 4096]
# TOP_KS = [1, 2, 16, 32]
# NUM_EXPERTS = [32, 160, 256, 257]
# BLOCK_SIZES = [32, 128]
NUM_TOKENS = [3, 4096]
TOP_KS = [1, 32]
NUM_EXPERTS = [256, 257]
BLOCK_SIZES = [32, 128]
PAD_SORTES_IDS = [False, True]

def to_iter(x):
    return x if isinstance(x, (list, tuple)) else (x,)

test_cases = list(itertools.product(
    NUM_TOKENS,
    TOP_KS,
    NUM_EXPERTS,
    BLOCK_SIZES,
    PAD_SORTES_IDS,
))


if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for m, topk, num_experts, block_size, pad_sorted_ids in test_cases:
    test_moe_align_block_size(m, topk, num_experts, block_size, pad_sorted_ids, args.device)
    test_moe_align_block_size_with_expert_map(m, topk, num_experts, block_size, args.device)
