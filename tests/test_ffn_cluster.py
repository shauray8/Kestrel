import argparse
import json

import torch

from kestrel import ops, vm

E4, BF, D, FF = torch.float8_e4m3fn, torch.bfloat16, 1536, 8960


def inputs(M, seed=0, bias=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    sa, s_mid = 0.125, 0.25
    x8 = (torch.randn(M, D, device="cuda", generator=g) / sa).to(E4)
    weights = []
    for n, k in ((FF, D), (D, FF)):
        w = torch.randn(n, k, device="cuda", generator=g) * k ** -0.5
        sw = (w.abs().amax(1) / 16).clamp(min=1e-12)
        w8 = (w / sw[:, None]).to(E4).contiguous()
        b = (torch.randn(n, device="cuda", generator=g) * 0.1).to(BF) if bias else None
        weights.extend((w8, sw, b))
    residual = torch.randn(M, D, device="cuda", generator=g)
    R = M // 1560 if M % 1560 == 0 else 1
    gate = torch.randn(R, D, device="cuda", generator=g) * 0.1
    return x8, sa, s_mid, weights, residual, gate


def program(x8, sa, s_mid, W, y, gate=None):
    h8 = torch.empty(x8.size(0), FF, dtype=E4, device=x8.device)
    if gate is None and W[2] is not None and W[5] is not None:
        return vm.ffn_program(x8, sa, *W[:3], h8, s_mid, *W[3:], y)
    p = vm.Program()
    tx, th, ty = p.tensor(x8, sa), p.tensor(h8, s_mid), p.tensor(y)
    tw0, tw2 = p.tensor(W[0], vscale=W[1]), p.tensor(W[3], vscale=W[4])
    b0 = -1 if W[2] is None else p.tensor(W[2].view(1, -1))
    b2 = -1 if W[5] is None else p.tensor(W[5].view(1, -1))
    tg = -1 if gate is None else p.tensor(gate)
    up = p.gemm(tx, tw0, th, b0, gelu=True)
    mid = [p.join(row) for row in up]
    p.gemm(th, tw2, ty, b2, gate=tg, after=lambda m, n: [mid[m]])
    return p.build()


def equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def case(M, gated=False, bias=True):
    x8, sa, s_mid, W, residual, gate = inputs(M, M + int(gated), bias)
    gate = gate if gated else None
    yv = residual.clone() if gated else torch.empty(M, D, device="cuda", dtype=BF)
    yc = yv.clone()
    p = program(x8, sa, s_mid, W, yv, gate)
    p.run()
    h = ops.n1().n1_linear_f8in_gelu_f8(x8, *W[:3], sa, s_mid)
    yn = ops.n1().n1_linear_f8in(h, *W[3:], s_mid)
    if gated:
        yr = residual.clone()
        ops.fused_mod().gated_residual(yr, yn, gate)
        yn = yr
    run = lambda: ops.ffn_cluster().run(x8, sa, *W[:3], s_mid, *W[3:], yc, gate)
    run()
    torch.cuda.synchronize()
    status = p.status()
    assert status["err"] == "ok" and status["done"] == status["n"], status
    assert equal(yc, yv), (M, gated, bias, "cluster_vs_vm")
    assert equal(yc, yn), (M, gated, bias, "cluster_vs_n1")
    yc.copy_(residual) if gated else yc.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    yc.copy_(residual) if gated else yc.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert equal(yc, yv), (M, gated, bias, "graph_vs_vm")
    print(json.dumps({"M": M, "gated": gated, "bias": bias, "bitwise_vm": True, "bitwise_n1": True, "bitwise_graph": True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    ops.ffn_cluster(), ops.vm(), ops.n1(), ops.fused_mod()
    if args.build_only:
        return
    for M in (3120, 4680, 129):
        for gated in (False, True):
            case(M, gated)
    for gated in (False, True):
        case(129, gated, bias=False)


if __name__ == "__main__":
    main()
