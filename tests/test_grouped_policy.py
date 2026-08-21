import pytest

import deep_gemm


@pytest.mark.parametrize(
    ("total_tokens", "num_groups", "max_tokens", "expected_alignment"),
    [
        (8, 32, None, 32),
        (64, 32, None, 32),
        (512, 32, None, 32),
        (1024, 32, None, 32),
        (4096, 32, None, 224),
        (64, 32, 64, 224),
    ],
)
def test_glm_bf16_grouped_alignment_policy(
    total_tokens: int,
    num_groups: int,
    max_tokens: int | None,
    expected_alignment: int,
) -> None:
    policy = deep_gemm.get_bf16_grouped_alignment(
        total_tokens, num_groups, max_tokens_per_group=max_tokens)
    assert policy.alignment == expected_alignment


def test_glm_bf16_grouped_alignment_policy_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        deep_gemm.get_bf16_grouped_alignment(-1, 32)
    with pytest.raises(ValueError):
        deep_gemm.get_bf16_grouped_alignment(1, 0)
