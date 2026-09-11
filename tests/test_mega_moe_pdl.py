import pytest
import torch

import deep_gemm
from test_mega_moe_spec import _SingleProcessGroup


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("global_pdl", [False, True])
def test_single_rank_pdl_replay(global_pdl):
    from cuda.bindings import runtime

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 required")
    torch.manual_seed(17)
    config = dict(num_experts=4, num_max_tokens_per_rank=8, num_topk=2,
                  hidden=256, intermediate_hidden=128, mma_type="bf16xbf16")
    spec = deep_gemm.get_mega_moe_workspace_spec(num_ranks=1, **config)
    arena = torch.zeros(spec.num_bytes, dtype=torch.uint8, device="cuda")
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        _SingleProcessGroup(), **config, buffer=arena, buffer_ptrs=[arena.data_ptr()])
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(8, 4, device="cuda").topk(2).indices
    weights = torch.rand(8, 2, device="cuda")
    l1 = torch.randn(4, 256, 256, device="cuda", dtype=torch.bfloat16) * 0.05
    l2 = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)
    out = torch.empty_like(x)
    prior = deep_gemm.get_pdl()
    deep_gemm.set_pdl(global_pdl)

    def run(pdl=False):
        buffer.x[:8].copy_(x)
        buffer.topk_idx[:8].copy_(ids)
        torch.mul(weights, 1.0, out=buffer.topk_weights[:8])
        deep_gemm.bf16_mega_moe(out, l1, l2, buffer, enable_pdl=pdl)

    try:
        run()
        run(True)
        torch.cuda.synchronize()
        for enabled in (False, True):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                run(enabled)
            handle = runtime.cudaGraph_t(graph.raw_cuda_graph())
            status, _, _, _, count = runtime.cudaGraphGetEdges(handle)
            assert int(status) == 0
            status, _, _, edges, _ = runtime.cudaGraphGetEdges(handle, count)
            assert int(status) == 0
            assert sum(int(e.type) == 1 for e in edges) == int(enabled), [
                (int(e.type), int(e.from_port), int(e.to_port)) for e in edges
            ]
            for _ in range(5):
                x.normal_()
                weights.uniform_(0.1, 1.0)
                ids.copy_(torch.rand(8, 4, device="cuda").topk(2).indices)
                graph.replay()
                actual = out.clone()
                run()
                torch.testing.assert_close(actual, out, rtol=0, atol=0)
            assert deep_gemm.get_pdl() == global_pdl
    finally:
        deep_gemm.set_pdl(prior)
        buffer.destroy()
