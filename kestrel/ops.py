import os, sys
from functools import cache

from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_FLAGS = ["-O3", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a"]
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

@cache
def _ext(name):
    mode = int(os.environ.get("VM_ATTN_PERF", "0")) if name == "vm" else 0
    if mode not in (0, 48, 96):
        raise ValueError("VM_ATTN_PERF must be 0, 48, or 96")
    flags = _FLAGS + ([f"-DVM_ATTN_PERF={mode}"] if mode else [])
    return load(f"kestrel_{name}" + (f"_attn{mode}" if mode else ""), [os.path.join(_CSRC, f"{name}.cu")],
                extra_cuda_cflags=flags, verbose=False)

def n1():
    return _ext("n1_linear")

def n2():
    return _ext("n2_attn")

def nvfp4():
    return _ext("nvfp4_quant")

def fused_mod():
    return _ext("fused_mod")

def rope_kv():
    return _ext("rope_kv")

def vm():
    return _ext("vm")
