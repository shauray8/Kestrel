import math

import torch


def _ns(g, steps):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tr = x.size(0) > x.size(1)
    if tr:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tr else x


torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
ns = torch.compile(_ns, dynamic=False)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, momentum=0.95, weight_decay=0.1, nesterov=True, ns_steps=5, compiled=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, ns_steps=ns_steps))
        self.fn = ns if compiled else _ns

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, m, wd = g["lr"], g["momentum"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                gr = p.grad.reshape(p.size(0), -1)
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(gr)
                buf = st["buf"]
                buf.mul_(m).add_(gr)
                u = gr.add(buf, alpha=m) if g["nesterov"] else buf
                o = self.fn(u, g["ns_steps"])
                scale = 0.2 * math.sqrt(max(o.shape))
                p.mul_(1 - lr * wd)
                p.add_(o.view_as(p).to(p.dtype), alpha=-lr * scale)
