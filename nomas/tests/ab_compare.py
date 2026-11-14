import torch, sys
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for k in a: print(k, "bitexact" if torch.equal(a[k], b[k]) else f"DIFF maxabs={(a[k].float()-b[k].float()).abs().max().item():.3g}")
