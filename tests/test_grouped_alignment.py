import pytest
import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from generators import (
    MajorTypeAB,
    generate_k_grouped_contiguous_psum,
    generate_m_grouped_contiguous,
)


@pytest.mark.parametrize("alignment", [32, 224])
def test_m_grouped_bf16_alignment_is_per_call(alignment: int) -> None:
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        _, a, b, grouped_layout, d, reference = generate_m_grouped_contiguous(
            4, 64, 128, 256,
            MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            use_bf16=True, use_psum_layout=False,
        )
        deep_gemm.set_mk_alignment_for_contiguous_layout(224 if alignment == 32 else 32)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            a, b, d, grouped_layout, alignment=alignment)
        assert calc_diff(d, reference) < 1e-5
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@pytest.mark.parametrize("alignment", [32, 224])
def test_k_grouped_bf16_alignment_is_per_call(alignment: int) -> None:
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        total_k, a, b, c, d, reference, grouped_layout, _ = (
            generate_k_grouped_contiguous_psum(
                4, 128, 128,
                MajorTypeAB.MNMajor, MajorTypeAB.MNMajor,
                [1, 17, 33, 65], alignment,
                use_bf16=True, gran_k=alignment,
            )
        )
        assert total_k % alignment == 0
        deep_gemm.set_mk_alignment_for_contiguous_layout(224 if alignment == 32 else 32)
        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
            a, b, d, None, grouped_layout, c,
            use_psum_layout=True, alignment=alignment)
        assert calc_diff(d, reference) < 1e-5
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@pytest.mark.parametrize("alignment", [32, 224])
def test_bf16_wgrad_without_accumulation_buffer(alignment: int) -> None:
    real_ks = [1, 17, 33, 65]
    _, a, b, _, _, _, grouped_layout, _ = generate_k_grouped_contiguous_psum(
        4,
        128,
        128,
        MajorTypeAB.MNMajor,
        MajorTypeAB.MNMajor,
        real_ks,
        alignment,
        use_bf16=True,
        gran_k=alignment,
    )
    out = torch.empty((4, 128, 128), dtype=torch.bfloat16, device="cuda")
    deep_gemm.k_grouped_bf16_wgrad_tn_contiguous(
        a,
        b,
        out,
        None,
        grouped_layout,
        None,
        route_tokens=sum(real_ks),
        use_psum_layout=True,
        alignment=alignment,
        max_tokens_per_group=max(real_ks),
    )

    expected = torch.empty_like(out)
    for group, real_k in enumerate(real_ks):
        end = grouped_layout[group].item()
        expected[group] = a[end - real_k : end].T @ b[end - real_k : end]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

@pytest.mark.parametrize(
    ("real_ks", "layout_alignment", "expected_alignment"),
    [([1, 8, 16, 32], 32, 32), ([65, 65, 65, 65], 224, 224)],
)
def test_fp32_wgrad_wrapper_uses_shape_policy(
    real_ks: list[int],
    layout_alignment: int,
    expected_alignment: int,
) -> None:
    _, a, b, c, d, reference, grouped_layout, _ = (
        generate_k_grouped_contiguous_psum(
            4, 128, 128,
            MajorTypeAB.MNMajor, MajorTypeAB.MNMajor,
            real_ks, layout_alignment,
            use_bf16=True, gran_k=layout_alignment,
        )
    )
    policy = deep_gemm.k_grouped_bf16_wgrad_tn_contiguous(
        a, b, d, None, grouped_layout, c,
        route_tokens=sum(real_ks), max_tokens_per_group=max(real_ks))
    assert policy.alignment == expected_alignment
    assert calc_diff(d, reference) < 1e-5
