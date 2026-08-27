import pytest
import torch

import deep_gemm


GLM_CONFIG = dict(
    num_ranks=8,
    num_experts=256,
    num_max_tokens_per_rank=512,
    num_topk=8,
    hidden=6144,
    intermediate_hidden=2048,
    num_shared_experts=1,
    mma_type="bf16xbf16",
)


class _SingleProcessGroup:
    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def barrier(self) -> None:
        pass


SMALL_CONFIG = dict(
    num_experts=1,
    num_max_tokens_per_rank=8,
    num_topk=1,
    hidden=256,
    intermediate_hidden=128,
    mma_type="bf16xbf16",
)


def test_glm_bf16_capability() -> None:
    capability = deep_gemm.get_mega_moe_capability(**GLM_CONFIG)
    assert capability.supported, capability.reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_glm_bf16_workspace_spec() -> None:
    spec = deep_gemm.get_mega_moe_workspace_spec(**GLM_CONFIG)
    assert spec.requested_max_tokens_per_rank == 512
    assert spec.num_max_tokens_per_rank >= spec.requested_max_tokens_per_rank
    assert spec.num_max_tokens_per_rank % spec.token_alignment == 0
    assert spec.num_ring_tokens > 0
    assert spec.num_sf_ring_tokens == 0
    assert spec.num_bytes > 0


@pytest.mark.parametrize(
    ("update", "reason_fragment"),
    [
        ({"num_ranks": 0}, "num_ranks"),
        ({"num_experts": 255}, "divisible"),
        ({"num_topk": 0}, "num_topk"),
        ({"activation": "gelu"}, "swiglu"),
        ({"mma_type": "tf32"}, "mma_type"),
    ],
)
def test_capability_rejections(update: dict, reason_fragment: str) -> None:
    config = GLM_CONFIG | update
    capability = deep_gemm.get_mega_moe_capability(**config)
    assert not capability.supported
    assert reason_fragment in capability.reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_caller_owned_workspace() -> None:
    group = _SingleProcessGroup()
    spec = deep_gemm.get_mega_moe_workspace_spec(num_ranks=1, **SMALL_CONFIG)
    arena_slice = torch.empty(spec.num_bytes + 4096, dtype=torch.uint8, device="cuda")

    workspace = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        **SMALL_CONFIG,
        buffer=arena_slice,
        buffer_ptrs=[arena_slice.data_ptr()],
    )

    assert not workspace.owns_buffer
    assert workspace.buffer.data_ptr() == arena_slice.data_ptr()
    assert workspace.buffer.nbytes == spec.num_bytes
    assert workspace.handle.buffer_ptrs == [arena_slice.data_ptr()]
    assert workspace.x.data_ptr() >= workspace.buffer.data_ptr()
    assert workspace.x.data_ptr() + workspace.x.nbytes <= workspace.buffer.data_ptr() + workspace.buffer.nbytes

    workspace.destroy()
    arena_slice.fill_(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_caller_owned_workspace_rejects_mismatched_pointer() -> None:
    group = _SingleProcessGroup()
    spec = deep_gemm.get_mega_moe_workspace_spec(num_ranks=1, **SMALL_CONFIG)
    arena_slice = torch.empty(spec.num_bytes, dtype=torch.uint8, device="cuda")

    with pytest.raises(ValueError, match="local buffer pointer"):
        deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            **SMALL_CONFIG,
            buffer=arena_slice,
            buffer_ptrs=[arena_slice.data_ptr() + 16],
        )
