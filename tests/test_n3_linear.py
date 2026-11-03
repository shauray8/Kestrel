import argparse
import itertools
import json

import torch

from kestrel import ops


def same(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    n3, n1 = ops.n3(), ops.n1()
    mod = ops.fused_mod()
    if args.build_only:
        return
    torch.manual_seed(13)
    for m, (n, k), biased in itertools.product((3120, 4680, 129, 512), ((1536, 1536), (8960, 1536), (1536, 8960)), (False, True)):
        x = (torch.randn(m, k, device="cuda") * 1.3).bfloat16()
        w = torch.randn(n, k, device="cuda") * 0.03
        sw = w.abs().amax(1).float() / 448.0
        wq = (w / sw[:, None]).to(torch.float8_e4m3fn).contiguous()
        bias = (torch.randn(n, device="cuda") * 0.1).bfloat16() if biased else None
        sa, mid = 0.375, 0.625
        xq = (x.float() * float(torch.tensor(1.0 / sa, dtype=torch.float32))).to(torch.float8_e4m3fn)
        calls = (
            ("linear", lambda e: getattr(e, "n3_linear" if e is n3 else "n1_linear")(x, wq, sw, bias, sa)),
            ("gelu", lambda e: getattr(e, "n3_linear_gelu_f8" if e is n3 else "n1_linear_gelu_f8")(x, wq, sw, bias, sa, mid)),
            ("f8in", lambda e: getattr(e, "n3_linear_f8in" if e is n3 else "n1_linear_f8in")(xq, wq, sw, bias, sa)),
            ("f8in_gelu", lambda e: getattr(e, "n3_linear_f8in_gelu_f8" if e is n3 else "n1_linear_f8in_gelu_f8")(xq, wq, sw, bias, sa, mid)),
        )
        for name, call in calls:
            expected = call(n1)
            same(call(n3), expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = call(n3)
            graph.replay()
            same(captured, expected)
            print(json.dumps({"shape": [m, n, k], "bias": biased, "op": name, "bitwise": True, "graph": True}), flush=True)
        residual = torch.randn(m, n, device="cuda")
        gate = torch.randn(1, n, device="cuda")
        grouped_gate = torch.randn(3 if m % 3 == 0 else 2, n * 2, device="cuda")[:, :n]
        for index, op, activation in ((0, n3.n3_linear_residual, x), (2, n3.n3_linear_f8in_residual, xq)):
            for g in (None, gate, grouped_gate):
                expected = residual.clone()
                y = calls[index][1](n1)
                if n % 256 == 0 and n // 256 in (4, 6, 8, 12, 16, 20):
                    mod.gated_residual(expected, y, g)
                else:
                    expected += y.float() if g is None else y.float() * g.repeat_interleave(m // g.shape[0], 0)
                actual = residual.clone()
                op(activation, wq, sw, bias, sa, actual, g)
                same(actual, expected)
                graph = torch.cuda.CUDAGraph()
                actual.copy_(residual)
                with torch.cuda.graph(graph):
                    op(activation, wq, sw, bias, sa, actual, g)
                actual.copy_(residual)
                graph.replay()
                same(actual, expected)
                print(json.dumps({"shape": [m, n, k], "bias": biased, "op": "residual_bf16" if index == 0 else "residual_f8",
                                  "gate_rows": None if g is None else g.shape[0], "bitwise": True, "graph": True}), flush=True)
    for m, biased in itertools.product((3120, 129, 512), (False, True)):
        x = (torch.randn(m, 1536, device="cuda") * 1.3).bfloat16()
        parts = []
        for _ in range(3):
            w = torch.randn(1536, 1536, device="cuda") * 0.03
            sw = w.abs().amax(1).float() / 448.0
            parts.append(((w / sw[:, None]).to(torch.float8_e4m3fn).contiguous(), sw,
                          (torch.randn(1536, device="cuda") * 0.1).bfloat16() if biased else None))
        cat = [torch.cat([p[i] for p in parts]).contiguous() if parts[0][i] is not None else None for i in range(3)]
        planes = n3.n3_linear_split(x, cat[0], cat[1], cat[2], 0.375, 1536)
        assert planes.shape == (3, m, 1536)
        for i, (w, sw, b) in enumerate(parts):
            same(planes[i], n3.n3_linear(x, w, sw, b, 0.375))
        print(json.dumps({"shape": [m, 4608, 1536], "bias": biased, "op": "split", "bitwise": True}), flush=True)


if __name__ == "__main__":
    main()
