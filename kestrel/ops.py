import os
from functools import cache

from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_FLAGS = ["-O3", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a"]

@cache
def _ext(name):
    return load(f"kestrel_{name}", [os.path.join(_CSRC, f"{name}.cu")], extra_cuda_cflags=_FLAGS, verbose=False)

def n1():
    return _ext("n1_linear")

def n2():
    return _ext("n2_attn")
