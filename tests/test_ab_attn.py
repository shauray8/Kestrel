import math

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("AB attention requires CUDA")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("N2 extension targets sm_120a")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def ext(cuda_device):
    from kestrel import ops

    return ops.n2()


@pytest.fixture(scope="module")
def make_plan(cuda_device):
    from kestrel.ab import make_plan

    return make_plan


def inputs(m, n, h, device, seed=123):
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(torch.randn(length, h, 128, device=device, dtype=torch.float16,
                             generator=generator) for length in (m, n, n))


def dense_mask(q, k, selection_block):
    h, rows = q.shape[1], math.ceil(q.shape[0] / 128)
    blocks = math.ceil(k.shape[0] / selection_block)
    indices = torch.arange(blocks, device=q.device, dtype=torch.int32)
    indices = indices.expand(h, rows, blocks).contiguous()
    return indices, torch.zeros(indices.shape, device=q.device, dtype=torch.float32)


def random_mask(q, k, selection_block, slots, weighted):
    h, rows = q.shape[1], math.ceil(q.shape[0] / 128)
    blocks = math.ceil(k.shape[0] / selection_block)
    slots = min(slots, blocks)
    generator = torch.Generator().manual_seed(319)
    indices = torch.full((h, rows, slots), -1, dtype=torch.int32)
    bias = torch.zeros((h, rows, slots), dtype=torch.float32)
    for head in range(h):
        for row in range(rows):
            count = max(1, slots - (head + row) % 3)
            chosen = torch.cat((torch.randperm(blocks - 1, generator=generator)[:count - 1],
                                torch.tensor([blocks - 1]))).sort().values
            indices[head, row, :count] = chosen.to(torch.int32)
            if weighted:
                bias[head, row, :count] = torch.rand(count, generator=generator) * 6
    return indices.to(q.device), bias.to(q.device)


def masked_reference(q, k, v, indices, log_bias, selection_block, query_rows):
    rows = torch.tensor(query_rows, device=q.device, dtype=torch.long)
    result = torch.empty(len(query_rows), q.shape[1], 128, device=q.device, dtype=torch.float32)
    indices, log_bias = indices.cpu(), log_bias.cpu()
    with sdpa_kernel(SDPBackend.MATH):
        for head in range(q.shape[1]):
            kh, vh = (x[:, head].float()[None, None] for x in (k, v))
            for block in sorted({row // 128 for row in query_rows}):
                positions = [i for i, row in enumerate(query_rows) if row // 128 == block]
                pos = torch.tensor(positions, device=q.device)
                mask = torch.full((k.shape[0],), -torch.inf, device=q.device)
                for index, bias in zip(indices[head, block].tolist(), log_bias[head, block].tolist()):
                    if index >= 0:
                        start = index * selection_block
                        mask[start:min(start + selection_block, k.shape[0])] = bias * math.log(2)
                qh = q[rows[pos], head].float()[None, None]
                result[pos, head] = F.scaled_dot_product_attention(
                    qh, kh, vh, attn_mask=mask[None, None, None], dropout_p=0.0)[0, 0]
    return result


def assert_n2_close(actual, reference):
    actual = actual.float()
    assert torch.isfinite(actual).all()
    difference = actual - reference
    relative_l2 = difference.norm() / reference.norm().clamp_min(1e-12)
    assert relative_l2.item() < 0.03, f"relative L2 error: {relative_l2.item():.6f}"
    assert difference.abs().max().item() < 0.04


@pytest.mark.parametrize("selection_block", [4, 64])
@pytest.mark.parametrize("m,n,h", [(1, 1, 1), (127, 67, 2), (129, 131, 3),
                                  (257, 1027, 2), (3120, 28080, 12)])
def test_all_ones_bitwise(ext, cuda_device, selection_block, m, n, h):
    q, k, v = inputs(m, n, h, cuda_device)
    indices, bias = dense_mask(q, k, selection_block)
    expected = ext.n2_attention(q, k, v)
    actual = ext.n2_attention_sparse(q, k, v, indices, bias, selection_block)
    assert actual.shape == q.shape
    assert actual.dtype == torch.float16
    assert actual.device == q.device
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("selection_block", [4, 64])
@pytest.mark.parametrize("weighted", [False, True], ids=["masked", "sip_bias"])
@pytest.mark.parametrize("m,n,h,slots", [(1, 9, 1, 1), (129, 267, 2, 5),
                                        (259, 1027, 3, 17), (3120, 28080, 12, 19)])
def test_sparse_float_sdpa(ext, cuda_device, selection_block, weighted, m, n, h, slots):
    q, k, v = inputs(m, n, h, cuda_device, seed=287)
    indices, bias = random_mask(q, k, selection_block, slots, weighted)
    actual = ext.n2_attention_sparse(q, k, v, indices, bias, selection_block)
    query_rows = list(range(m)) if m < 1000 else [0, 127, 128, 1535, 3071, 3119]
    reference = masked_reference(q, k, v, indices, bias, selection_block, query_rows)
    assert_n2_close(actual[query_rows], reference)


@pytest.mark.parametrize("selection_block", [4, 64])
def test_default_selection_block(ext, cuda_device, selection_block):
    q, k, v = inputs(129, 257, 2, cuda_device)
    indices, bias = random_mask(q, k, selection_block, 3, True)
    explicit = ext.n2_attention_sparse(q, k, v, indices, bias, selection_block)
    if selection_block == 4:
        assert torch.equal(ext.n2_attention_sparse(q, k, v, indices, bias), explicit)
    else:
        reference = masked_reference(q, k, v, indices, bias, selection_block, [0, 128])
        assert_n2_close(explicit[[0, 128]], reference)


def check_plan(plan, q, k, selection_block, expected_slots):
    rows, blocks = math.ceil(q.shape[0] / 128), math.ceil(k.shape[0] / selection_block)
    assert plan.indices.shape == (q.shape[1], rows, expected_slots)
    assert plan.log_bias.shape == plan.indices.shape
    assert plan.indices.dtype == torch.int32
    assert plan.log_bias.dtype == torch.float32
    assert plan.indices.device == plan.log_bias.device == q.device
    assert plan.indices.is_contiguous() and plan.log_bias.is_contiguous()
    assert ((plan.indices >= 0) & (plan.indices < blocks)).all()
    assert (plan.indices[..., 1:] > plan.indices[..., :-1]).all()
    assert plan.kept == q.shape[1] * rows * expected_slots
    assert plan.total == q.shape[1] * rows * blocks
    assert torch.isfinite(plan.log_bias).all()
    assert (plan.log_bias >= 0).all()
    probabilities = torch.exp2(-plan.log_bias)
    assert ((probabilities > 0) & (probabilities <= 1)).all()


@pytest.mark.parametrize("selection_block", [4, 64])
@pytest.mark.parametrize("sparsity", [0.5, 0.75, 0.9, 1.0])
@pytest.mark.parametrize("sink,current", [(0, 0), (5, 0), (0, 6), (5, 6), (129, 130), (1000, 130)])
def test_plan_mandatory_and_exact_budget(make_plan, cuda_device, selection_block, sparsity, sink, current):
    q, k, _ = inputs(129, 1027, 2, cuda_device)
    blocks = math.ceil(k.shape[0] / selection_block)
    starts = torch.arange(blocks, device=cuda_device) * selection_block
    mandatory = (starts < sink) | ((starts + selection_block > k.shape[0] - current) & (current > 0))
    mandatory_ids = mandatory.nonzero().flatten().to(torch.int32)
    tile = 64 // selection_block
    requested = max(1, math.ceil((1 - sparsity) * blocks))
    expected_slots = min(blocks, max(mandatory_ids.numel(), math.ceil(requested / tile) * tile))
    plan = make_plan(q, k, sparsity=sparsity, sink_tokens=sink, current_tokens=current,
                     selection_block=selection_block,
                     generator=torch.Generator(device=cuda_device).manual_seed(71))
    check_plan(plan, q, k, selection_block, expected_slots)
    for head in range(q.shape[1]):
        for row in range(plan.indices.shape[1]):
            positions = torch.searchsorted(plan.indices[head, row], mandatory_ids)
            assert (positions < expected_slots).all()
            assert torch.equal(plan.indices[head, row, positions], mandatory_ids)
            assert (plan.log_bias[head, row, positions] == 0).all()


@pytest.mark.parametrize("selection_block", [4, 64])
def test_plan_dense_all_ones(make_plan, ext, cuda_device, selection_block):
    q, k, v = inputs(129, 1027, 2, cuda_device)
    plan = make_plan(q, k, sparsity=0.0, sink_tokens=5, current_tokens=6,
                     selection_block=selection_block)
    indices, bias = dense_mask(q, k, selection_block)
    check_plan(plan, q, k, selection_block, indices.shape[-1])
    assert torch.equal(plan.indices, indices)
    assert torch.equal(plan.log_bias, bias)
    assert torch.equal(ext.n2_attention_sparse(q, k, v, plan.indices, plan.log_bias, selection_block),
                       ext.n2_attention(q, k, v))


@pytest.mark.parametrize("selection_block", [4, 64])
def test_plan_seeded_determinism(make_plan, cuda_device, selection_block):
    q, k, _ = inputs(259, 2051, 3, cuda_device)
    kwargs = dict(sparsity=0.5, sink_tokens=5, current_tokens=6, selection_block=selection_block)
    first = make_plan(q, k, **kwargs, generator=torch.Generator(device=cuda_device).manual_seed(939))
    torch.rand(100, device=cuda_device)
    second = make_plan(q, k, **kwargs, generator=torch.Generator(device=cuda_device).manual_seed(939))
    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.log_bias, second.log_bias)
    assert first.kept == second.kept and first.total == second.total


@pytest.mark.parametrize("selection_block", [4, 64])
def test_plan_default_current_suffix(make_plan, cuda_device, selection_block):
    q, k, _ = inputs(129, 1027, 2, cuda_device)
    kwargs = dict(sparsity=0.9, sink_tokens=0, selection_block=selection_block)
    default = make_plan(q, k, **kwargs, generator=torch.Generator(device=cuda_device).manual_seed(51))
    explicit = make_plan(q, k, **kwargs, current_tokens=q.shape[0],
                         generator=torch.Generator(device=cuda_device).manual_seed(51))
    assert torch.equal(default.indices, explicit.indices)
    assert torch.equal(default.log_bias, explicit.log_bias)


@pytest.fixture
def hook_setup(monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace
    from lingbot import ab_hook

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sink_size = 3
            self.ab = None

        def forward(self, q, k, v):
            output = self.ab.attend(q, k, v)
            return q if output is None else output

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.cross_attn = Attention(), Attention()

        def forward(self, q, k, v):
            self.self_attn(q, k, v)
            return self.cross_attn(q, k, v)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

        def forward(self, q, k, v, *, current_start, frame_seqlen):
            for block in self.blocks:
                block(q, k, v)
            return q

    class Extension:
        def __init__(self):
            self.calls = []

        def n2_attention(self, q, k, v):
            self.calls.append(("dense",))
            return q.clone()

        def n2_attention_sparse(self, q, k, v, indices, log_bias, selection_block):
            self.calls.append(("sparse", indices, log_bias, selection_block))
            return q.clone()

    def setup(**environment):
        defaults = dict(GRAPHS="0", AB_SPARSITY="0.9", AB_DENSE_STEPS="1",
                        AB_DENSE_BLOCKS="1", AB_PLAN_CYCLE="4", AB_CLUSTERS="8",
                        AB_SEED="42", AB_KEEP_SINK="1", AB_KEEP_CURRENT="1")
        for name, value in (defaults | environment).items():
            monkeypatch.setenv(name, value)
        builds = []

        def select(q, k, **kwargs):
            rows = math.ceil(q.shape[0] / 128)
            indices = torch.zeros(q.shape[1], rows, 1, dtype=torch.int32)
            plan = SimpleNamespace(indices=indices, log_bias=torch.zeros_like(indices, dtype=torch.float32),
                                   kept=q.shape[1] * rows, total=q.shape[1] * rows * math.ceil(k.shape[0] / 4))
            builds.append((kwargs, plan))
            return plan

        monkeypatch.setattr(ab_hook, "make_plan", select)
        model, extension, paths = Model(), Extension(), defaultdict(int)
        hook = ab_hook.ABAttention(model, extension, paths)
        for block in model.blocks:
            block.self_attn.ab = block.cross_attn.ab = hook
        q = torch.zeros(1, 129, 1, 128)
        k, v = (torch.zeros(1, 1027, 1, 128) for _ in range(2))

        def run(start=0):
            return model(q, k, v, current_start=start, frame_seqlen=2)

        return SimpleNamespace(hook=hook, model=model, ext=extension, paths=paths, builds=builds, run=run)

    return setup


def test_hook_schedule_rebuild_reuse_and_cross_attention(hook_setup):
    state = hook_setup()
    state.run()
    assert [call[0] for call in state.ext.calls] == ["dense", "dense"]
    assert not state.builds and state.hook.layer is None
    state.run()
    first = state.builds[0][1]
    assert len(state.builds) == 1
    for _ in range(3):
        state.run()
        assert state.ext.calls[-1][1] is first.indices
    assert state.paths["ab_plan_reuses"] == 3
    state.run()
    assert len(state.builds) == 2
    assert state.ext.calls[-1][1] is state.builds[-1][1].indices
    assert state.ext.calls[-1][1] is not first.indices
    assert state.paths["attn_ab_dense"] == 7
    assert state.paths["attn_ab"] == 5
    assert state.paths["ab_plan_builds"] == 2
    assert len(state.ext.calls) == 12
    assert state.hook.layer is None
    assert state.hook.attend(torch.empty(0), torch.empty(0), torch.empty(0)) is None
    for kwargs, _ in state.builds:
        assert kwargs["sink_tokens"] == 6
        assert kwargs["current_tokens"] == 129
        assert kwargs["clusters"] == 8
    assert state.builds[0][0]["generator"].initial_seed() != state.builds[1][0]["generator"].initial_seed()


def test_hook_chunk_change_and_video_reset(hook_setup):
    state = hook_setup()
    state.run()
    state.run()
    old = state.builds[-1][1]
    state.run(start=129)
    assert state.hook.step == 0 and not state.hook.plans
    assert [call[0] for call in state.ext.calls[-2:]] == ["dense", "dense"]
    state.run(start=129)
    assert len(state.builds) == 2
    assert state.builds[-1][1] is not old
    state.hook.reset()
    assert state.hook.start is None and state.hook.step == -1 and not state.hook.plans
    state.run(start=129)
    assert state.hook.step == 0 and not state.hook.plans
    assert [call[0] for call in state.ext.calls[-2:]] == ["dense", "dense"]


@pytest.mark.parametrize("keep_sink,keep_current", [(1, 1), (1, 0), (0, 1), (0, 0)])
def test_hook_optional_mandatory_regions(hook_setup, keep_sink, keep_current):
    state = hook_setup(AB_KEEP_SINK=str(keep_sink), AB_KEEP_CURRENT=str(keep_current))
    state.run()
    state.run()
    kwargs, _ = state.builds[0]
    assert kwargs["sink_tokens"] == 6 * keep_sink
    assert kwargs["current_tokens"] == 129 * keep_current


@pytest.mark.parametrize("environment", [dict(AB_SPARSITY="0"), dict(AB_DENSE_BLOCKS="2"),
                                        dict(AB_DENSE_STEPS="3")])
def test_hook_preserves_dense_schedule(hook_setup, environment):
    state = hook_setup(**environment)
    for _ in range(3):
        state.run()
    assert not state.builds
    assert all(call[0] == "dense" for call in state.ext.calls)
    assert state.paths["ab_blocks_kept"] == state.paths["ab_blocks_total"]


def test_hook_rejects_cuda_graph_schedule(hook_setup):
    with pytest.raises(ValueError, match="GRAPHS"):
        hook_setup(GRAPHS="1")
