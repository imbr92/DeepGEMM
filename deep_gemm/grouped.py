from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from . import _C


@dataclass(frozen=True)
class BF16GroupedAlignmentPolicy:
    expected_tokens_per_group: float
    max_tokens_per_group: Optional[int]
    alignment: int


def get_bf16_grouped_alignment(total_tokens: int,
                               num_groups: int,
                               max_tokens_per_group: Optional[int] = None) -> BF16GroupedAlignmentPolicy:
    """Choose the measured SM100 GLM backward alignment without mutable global state."""
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    if max_tokens_per_group is not None and max_tokens_per_group < 0:
        raise ValueError(
            f"max_tokens_per_group must be non-negative, got {max_tokens_per_group}")

    expected = total_tokens / num_groups
    effective = max(expected, max_tokens_per_group or 0)
    alignment = 32 if effective <= 32 else 224
    return BF16GroupedAlignmentPolicy(expected, max_tokens_per_group, alignment)


def k_grouped_bf16_wgrad_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks_cpu: Optional[Sequence[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    *,
    route_tokens: int,
    compiled_dims: str = "mn",
    use_psum_layout: bool = True,
    alignment: Optional[int] = None,
    max_tokens_per_group: Optional[int] = None,
) -> BF16GroupedAlignmentPolicy:
    """Run FP32-accumulating BF16 wgrad with a shape-local alignment policy."""
    if d.dtype != torch.float32:
        raise ValueError(f"wgrad output must be FP32, got {d.dtype}")
    policy = get_bf16_grouped_alignment(
        route_tokens, d.shape[0], max_tokens_per_group=max_tokens_per_group)
    selected_alignment = policy.alignment if alignment is None else alignment
    _C.k_grouped_bf16_gemm_tn_contiguous(
        a, b, d, ks_cpu, grouped_layout, c,
        compiled_dims, use_psum_layout, selected_alignment)
    return BF16GroupedAlignmentPolicy(
        policy.expected_tokens_per_group,
        policy.max_tokens_per_group,
        selected_alignment,
    )
