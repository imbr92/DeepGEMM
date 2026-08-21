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
