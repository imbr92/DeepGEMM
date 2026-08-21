import pytest
import torch

from test_mega_moe import make_test_routes


@pytest.mark.parametrize("tokens", [1, 8, 64, 512])
@pytest.mark.parametrize("skew_factor", [1, 2, 4, 8])
def test_glm_route_load_and_uniqueness(tokens: int, skew_factor: int) -> None:
    num_ranks, num_experts, num_topk = 8, 256, 8
    all_routes = [
        make_test_routes(
            tokens, num_topk, num_experts, num_ranks, rank,
            "balanced" if skew_factor == 1 else "skew",
            skew_factor, 0, torch.device("cpu"),
        )
        for rank in range(num_ranks)
    ]
    routes = torch.cat(all_routes)
    local_rank = routes.div(num_experts // num_ranks, rounding_mode="floor")
    rank_loads = torch.bincount(local_rank.flatten(), minlength=num_ranks)
    average = rank_loads.float().mean()

    assert torch.all(torch.sort(routes, dim=1).values.diff(dim=1) != 0)
    assert rank_loads.sum().item() == num_ranks * tokens * num_topk
    assert rank_loads[0].item() / average.item() == pytest.approx(skew_factor, abs=0.05)

