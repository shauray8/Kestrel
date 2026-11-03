import json
from types import SimpleNamespace as NS

import torch

from kestrel import vm
from lingbot.vm_block import block_parts
from lingbot.vm_schedule import cache_plan


D, FF, TARGET = 1536, 8960, 16.0
BF, E4 = torch.bfloat16, torch.float8_e4m3fn


def block(g, normalized):
    def linear(n=D, k=D):
        w = torch.randn(n, k, device="cuda", generator=g) / k ** 0.5
        sw = (w.abs().amax(1) / TARGET).clamp_min(1e-12)
        return NS(wq=(w / sw[:, None]).to(E4).contiguous(), sw=sw,
                  bias_bf16=(torch.randn(n, device="cuda", generator=g) * 0.01).to(BF), amax=1.0)
    def attention():
        return NS(q=linear(), k=linear(), v=linear(), o=linear(), eps=1e-6,
                  norm_q=NS(weight=torch.ones(D, device="cuda", dtype=BF)),
                  norm_k=NS(weight=torch.ones(D, device="cuda", dtype=BF)))
    norm3 = torch.nn.LayerNorm(D, eps=1e-5, device="cuda", dtype=BF) if normalized else torch.nn.Identity()
    if normalized:
        with torch.no_grad():
            norm3.weight.copy_(1 + torch.randn(D, device="cuda", generator=g) * 0.1)
            norm3.bias.copy_(torch.randn(D, device="cuda", generator=g) * 0.01)
    return NS(self_attn=attention(), cross_attn=attention(), norm1=NS(eps=1e-6), norm2=NS(eps=1e-6),
              ffn=NS(l0=linear(FF), l2=linear(D, FF)),
              norm3=norm3,
              modulation=(torch.randn(1, 6, D, device="cuda", generator=g) * 0.1).to(BF))


def case(M, broadcast, camera, rolling, first):
    g = torch.Generator(device="cuda").manual_seed(1895)
    R, L, capacity = (1 if broadcast else M), 512, 4 * M
    x0 = torch.randn(M, D, device="cuda", generator=g)
    e = torch.randn(R, 6 * D, device="cuda", generator=g) * 0.1
    ev = torch.empty_like(e)
    context = torch.randn(L, D, device="cuda", dtype=BF, generator=g)
    buffers = [torch.empty(M, D, device="cuda", dtype=BF) for _ in range(7)]
    buffers += [torch.empty(M, D, device="cuda", dtype=E4), torch.empty(M, FF, device="cuda", dtype=E4)]
    theta = torch.arange(M, device="cuda")[:, None] / (10000 ** (torch.arange(64, device="cuda") / 64))
    cs = torch.stack((theta.cos(), theta.sin()), -1).flatten(1).float().contiguous()
    start = capacity if rolling else 0
    p = cache_plan(start, M, capacity if rolling else 0, capacity if rolling else 0, capacity, M, capacity)
    caches, crosses, saved, executions = [], [], [], []
    blocks = [block(g, normalized) for normalized in (False, True)]
    for blk in blocks:
        kv = {name: torch.randn(capacity, D, device="cuda", dtype=BF, generator=g) for name in ("k", "v")}
        kv.update(global_end_index=torch.zeros(1, device="cuda", dtype=torch.int64),
                  local_end_index=torch.zeros(1, device="cuda", dtype=torch.int64))
        cross = {name: torch.randn(L, D, device="cuda", dtype=BF, generator=g) for name in ("k", "v")}
        cross["is_init"] = torch.tensor(int(not first), device="cuda", dtype=torch.int32)
        saved.append(({k: v.clone() for k, v in kv.items()}, {k: v.clone() for k, v in cross.items()}))
        caches.append(kv); crosses.append(cross)
        mp, sq, so, cps = block_parts(blk, x0, e, ev, context, cross, TARGET, buffers)
        ap, ao = vm.self_attention_program(*buffers[:3], cs, kv["k"], kv["v"], p.lo, p.local_end, p.write_offset,
                                           p.advance, blk.self_attn.norm_q.weight.view(1, -1),
                                           blk.self_attn.norm_k.weight.view(1, -1), build=False)
        ap.keep = [t for t in ap.keep if t is not ao]
        ap.bind("out", buffers[3])
        before = [("mod.", mp), ("sq.", sq), ("sa.", ap), ("so.", so)]
        after = cps[0 if first else 1]
        sequential = ([vm.Program.compose([part]) for part in before], [vm.Program.compose([part]) for part in after])
        merged = ([vm.Program.compose(before, build=False), vm.Program.compose(after, build=False)] if camera else
                  [vm.Program.compose(before + after, build=False)])
        merged[-1].cache_commit(kv["global_end_index"], kv["local_end_index"], cross["is_init"],
                                p.global_end, p.local_end, first)
        executions.append((sequential, [program.build() for program in merged]))

    def launch(program, x, kv, cross):
        bindings = {"mod.e": e, "sq.residual": x, "sa.cs": cs, "sa.kc": kv["k"], "sa.vc": kv["v"],
                    "sa.ak": kv["k"][p.lo:p.local_end], "sa.av": kv["v"][p.lo:p.local_end], "so.residual": x,
                    "cq.qx" if first else "cq.x": x, "cq.kx": context, "cq.ky": cross["k"], "cq.vy": cross["v"],
                    "ca.k": cross["k"], "ca.v": cross["v"], "ff.x": x,
                    "global_index": kv["global_end_index"].view(1, 1), "local_index": kv["local_end_index"].view(1, 1),
                    "initialized": cross["is_init"].view(1, 1)}
        residual_name = "cq.qresidual" if first else "cq.residual"
        if residual_name in program.slots:
            del bindings["cq.qx" if first else "cq.x"]
            bindings[residual_name] = x
        for name, tensor in bindings.items():
            if name in program.slots:
                program.bind(name, tensor)
        program.run()

    statuses = []
    outputs = []
    for merged in (False, True):
        for kv, cross, (ks, xs) in zip(caches, crosses, saved):
            for name in kv:
                kv[name].copy_(ks[name])
            for name in cross:
                cross[name].copy_(xs[name])
        x = x0.clone()
        for i, (sequential, programs) in enumerate(executions):
            before, after = (programs[:1], programs[1:]) if merged and camera else ((programs, []) if merged else sequential)
            for program in before:
                launch(program, x, caches[i], crosses[i]); statuses.append(program)
            if camera:
                x = (x * 1.01 + 0.001).contiguous()
            for program in after:
                launch(program, x, caches[i], crosses[i]); statuses.append(program)
            if not merged:
                caches[i]["global_end_index"].fill_(p.global_end)
                caches[i]["local_end_index"].fill_(p.local_end)
                if first:
                    crosses[i]["is_init"].fill_(1)
            assert torch.equal(ev, e + blocks[i].modulation.view(1, -1))
        outputs.append([x.clone(), *(t.clone() for kv in caches for t in kv.values()),
                        *(t.clone() for cross in crosses for t in cross.values())])
    torch.cuda.synchronize()
    states = [program.status() for program in statuses]
    assert all(s["err"] == "ok" and s["done"] == s["n"] for s in states), states
    assert all(torch.equal(a, b) for a, b in zip(*outputs))
    assert all(len(program.tt) <= vm.MAXT for _, programs in executions for program in programs)
    print(json.dumps({"test": "vm_multiblock", "M": M, "broadcast": broadcast, "camera": camera,
                      "rolling": rolling, "first": first, "blocks": 2, "pass": True}), flush=True)


if __name__ == "__main__":
    for args in ((128, True, False, False, True), (128, False, True, True, False),
                 (3120, True, True, False, True), (3120, False, False, True, False)):
        case(*args)
