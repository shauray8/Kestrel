import os
import torch
from kestrel.ab import make_plan

class ABAttention:
    def __init__(self, model, ext, paths):
        if os.environ.get("GRAPHS") == "1":
            raise ValueError("ATTN=ab requires GRAPHS unset or GRAPHS=0; sampled plans change between chunks")
        self.ext, self.paths = ext, paths
        self.sparsity = float(os.environ.get("AB_SPARSITY", "0.9"))
        self.dense_steps = int(os.environ.get("AB_DENSE_STEPS", "1"))
        self.dense_blocks = int(os.environ.get("AB_DENSE_BLOCKS", "1"))
        self.cycle = int(os.environ.get("AB_PLAN_CYCLE", "4"))
        self.clusters = int(os.environ.get("AB_CLUSTERS", "8"))
        self.seed = int(os.environ.get("AB_SEED", "42"))
        self.keep_sink = os.environ.get("AB_KEEP_SINK", "1") == "1"
        self.keep_current = os.environ.get("AB_KEEP_CURRENT", "1") == "1"
        if not 0 <= self.sparsity <= 1 or min(self.dense_steps, self.dense_blocks) < 0 or self.cycle < 1 or not 1 <= self.clusters <= 128:
            raise ValueError("invalid AB sparsity, dense schedule, plan cycle, or cluster count")
        self.reset()
        model.register_forward_pre_hook(self.begin, with_kwargs=True)
        for layer, block in enumerate(model.blocks):
            block.self_attn.register_forward_pre_hook(self.enter(layer), with_kwargs=True)
            block.self_attn.register_forward_hook(self.leave, always_call=True)

    def reset(self):
        self.start, self.step, self.layer, self.frame_tokens = None, -1, None, None
        self.sink_frames, self.plans = 0, {}

    def begin(self, model, args, kwargs):
        start = kwargs["current_start"]
        if start != self.start:
            self.start, self.step, self.plans = start, 0, {}
        else:
            self.step += 1
        self.frame_tokens = kwargs["frame_seqlen"]

    def enter(self, layer):
        def hook(module, args, kwargs):
            self.layer, self.sink_frames = layer, module.sink_size
        return hook

    def leave(self, module, args, output):
        self.layer = None

    def attend(self, q, k, v):
        if self.layer is None:
            return None
        q, k, v = (t[0].half().contiguous() for t in (q, k, v))
        total = q.shape[1] * ((q.shape[0] + 127) // 128) * ((k.shape[0] + 3) // 4)
        self.paths["ab_blocks_total"] += total
        if self.step < self.dense_steps or self.layer < self.dense_blocks or self.sparsity == 0:
            self.paths["attn_ab_dense"] += 1
            self.paths["ab_blocks_kept"] += total
            return self.ext.n2_attention(q, k, v)[None]
        anchor = (self.step - self.dense_steps) // self.cycle
        key = (self.layer, anchor, q.shape, k.shape, q.device)
        plan = self.plans.get(key)
        if plan is None:
            generator = torch.Generator(device=q.device).manual_seed(self.seed + self.layer * 1_000_003 + anchor * 97)
            plan = make_plan(q, k, sparsity=self.sparsity, sink_tokens=self.sink_frames * self.frame_tokens if self.keep_sink else 0,
                             current_tokens=q.shape[0] if self.keep_current else 0, clusters=self.clusters, generator=generator)
            self.plans = {key_: value for key_, value in self.plans.items() if key_[0] != self.layer}
            self.plans[key] = plan
            self.paths["ab_plan_builds"] += 1
        else:
            self.paths["ab_plan_reuses"] += 1
        self.paths["attn_ab"] += 1
        self.paths["ab_blocks_kept"] += plan.kept
        return self.ext.n2_attention_sparse(q, k, v, plan.indices, plan.log_bias, 4)[None]
