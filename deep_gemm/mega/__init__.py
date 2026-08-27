import torch
import types
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


_WORKSPACE_ALIGNMENT = 16


@dataclass(frozen=True)
class MegaMoECapability:
    supported: bool
    reason: str = ''

    def require(self) -> None:
        if not self.supported:
            raise ValueError(self.reason)


@dataclass(frozen=True)
class MegaMoEWorkspaceSpec:
    num_ranks: int
    num_experts: int
    requested_max_tokens_per_rank: int
    num_max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_shared_experts: int
    mma_type: str
    activation: str
    token_alignment: int
    num_ring_tokens: int
    num_sf_ring_tokens: int
    num_bytes: int


def get_mega_moe_capability(num_ranks: int,
                            num_experts: int,
                            num_max_tokens_per_rank: int,
                            num_topk: int,
                            hidden: int,
                            intermediate_hidden: int,
                            num_shared_experts: int = 0,
                            mma_type: str = 'fp8xfp4',
                            activation: str = 'swiglu',
                            device: Optional[Union[torch.device, str, int]] = None) -> MegaMoECapability:
    if num_ranks <= 0:
        return MegaMoECapability(False, f'num_ranks must be positive, got {num_ranks}')
    if num_experts <= 0 or num_experts % num_ranks != 0:
        return MegaMoECapability(
            False, f'num_experts ({num_experts}) must be positive and divisible by num_ranks ({num_ranks})')
    if num_max_tokens_per_rank < 0:
        return MegaMoECapability(
            False, f'num_max_tokens_per_rank must be non-negative, got {num_max_tokens_per_rank}')
    if num_topk <= 0 or num_topk > num_experts:
        return MegaMoECapability(False, f'num_topk must be in [1, {num_experts}], got {num_topk}')
    if hidden <= 0 or intermediate_hidden <= 0:
        return MegaMoECapability(
            False, f'hidden sizes must be positive, got hidden={hidden}, intermediate_hidden={intermediate_hidden}')
    if num_shared_experts < 0:
        return MegaMoECapability(False, f'num_shared_experts must be non-negative, got {num_shared_experts}')
    if mma_type not in ('bf16xbf16', 'fp8xfp4'):
        return MegaMoECapability(False, f'unsupported mma_type {mma_type!r}')
    if activation != 'swiglu':
        return MegaMoECapability(False, f'unsupported activation {activation!r}; only swiglu is supported')
    if mma_type == 'fp8xfp4' and (
            hidden % 128 != 0 or intermediate_hidden % 128 != 0 or
            (num_shared_experts > 0 and intermediate_hidden * num_shared_experts % 128 != 0)):
        return MegaMoECapability(
            False, 'fp8xfp4 requires hidden, intermediate, and shared intermediate sizes divisible by 128')

    if device is not None:
        device = torch.device(device)
        if device.type != 'cuda':
            return MegaMoECapability(False, f'Mega-MoE requires CUDA, got {device}')
        if not torch.cuda.is_available():
            return MegaMoECapability(False, 'Mega-MoE requires an available CUDA device')
        if torch.cuda.get_device_capability(device)[0] != 10:
            return MegaMoECapability(
                False, f'Mega-MoE requires an SM100-family GPU, got capability {torch.cuda.get_device_capability(device)}')
    return MegaMoECapability(True)


def get_mega_moe_workspace_spec(num_ranks: int,
                                num_experts: int,
                                num_max_tokens_per_rank: int,
                                num_topk: int,
                                hidden: int,
                                intermediate_hidden: int,
                                num_shared_experts: int = 0,
                                mma_type: str = 'fp8xfp4',
                                activation: str = 'swiglu') -> MegaMoEWorkspaceSpec:
    capability = get_mega_moe_capability(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_shared_experts, mma_type, activation)
    capability.require()
    token_alignment = _C.get_token_alignment_for_mega_moe()
    aligned_tokens = align(num_max_tokens_per_rank, token_alignment)
    num_bytes, num_ring_tokens, num_sf_ring_tokens = _C.get_symm_buffer_metadata_for_mega_moe(
        num_ranks, num_experts, aligned_tokens, num_topk, hidden, intermediate_hidden,
        mma_type, activation, num_shared_experts)
    return MegaMoEWorkspaceSpec(
        num_ranks=num_ranks, num_experts=num_experts,
        requested_max_tokens_per_rank=num_max_tokens_per_rank,
        num_max_tokens_per_rank=aligned_tokens, num_topk=num_topk,
        hidden=hidden, intermediate_hidden=intermediate_hidden,
        num_shared_experts=num_shared_experts, mma_type=mma_type, activation=activation,
        token_alignment=token_alignment, num_ring_tokens=num_ring_tokens,
        num_sf_ring_tokens=num_sf_ring_tokens, num_bytes=num_bytes)


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 buffer: Optional[torch.Tensor] = None,
                 buffer_ptrs: Optional[Sequence[int]] = None):
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        self.spec = get_mega_moe_workspace_spec(
            group.size(), num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, num_shared_experts, mma_type, activation)
        assert self.spec.num_max_tokens_per_rank == num_max_tokens_per_rank

        # Allocate a symmetric buffer, or wrap a caller-owned slice of a
        # symmetric arena. The kernel only consumes the local tensor and the
        # process-group-ordered peer pointers; it does not depend on the
        # allocator that produced them.
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation, num_shared_experts
        )
        assert num_bytes == self.spec.num_bytes
        if (buffer is None) != (buffer_ptrs is None):
            raise ValueError('buffer and buffer_ptrs must be provided together')

        self.owns_buffer = buffer is None
        if buffer is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(self.spec.num_bytes, dtype=torch.int8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
        else:
            if not buffer.is_cuda or not buffer.is_contiguous() or buffer.element_size() != 1:
                raise ValueError('caller-owned Mega-MoE buffer must be a contiguous one-byte CUDA tensor')
            if buffer.nbytes < self.spec.num_bytes:
                raise ValueError(
                    f'caller-owned Mega-MoE buffer has {buffer.nbytes} bytes, '
                    f'but {self.spec.num_bytes} bytes are required')
            pointers = [int(pointer) for pointer in buffer_ptrs]
            if len(pointers) != group.size():
                raise ValueError(
                    f'expected {group.size()} process-group-ordered buffer pointers, got {len(pointers)}')
            if any(pointer <= 0 or pointer % _WORKSPACE_ALIGNMENT for pointer in pointers):
                raise ValueError(
                    f'caller-owned Mega-MoE buffer pointers must be positive and '
                    f'{_WORKSPACE_ALIGNMENT}-byte aligned')
            if pointers[group.rank()] != buffer.data_ptr():
                raise ValueError(
                    'caller-owned Mega-MoE local buffer pointer does not match buffer_ptrs[group.rank()]')
            self.buffer = buffer.flatten()[:self.spec.num_bytes]
            self.handle = types.SimpleNamespace(buffer_ptrs=pointers)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)
        assert self.l1_acts.shape[0] == self.spec.num_ring_tokens
        has_sf_ring = self.l1_acts_sf is not None and self.l1_acts_sf.numel() > 0
        assert has_sf_ring == (self.spec.num_sf_ring_tokens > 0)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu',
                                 buffer: Optional[torch.Tensor] = None,
                                 buffer_ptrs: Optional[Sequence[int]] = None) -> SymmBuffer:
    """Allocate a workspace or wrap a caller-owned symmetric-arena slice.

    When ``buffer`` is supplied, ``buffer_ptrs`` must contain the base address
    of the corresponding slice on every process-group rank, in process-group
    rank order. The caller retains ownership of the backing allocation.
    """
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation,
        buffer=buffer, buffer_ptrs=buffer_ptrs
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math
    )
