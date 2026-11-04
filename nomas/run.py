import importlib, os, sys

ROOT = os.environ.get("NOMAS_ROOT", "/workspace/Hy-Nomas")
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_INDUCTOR_FX_GRAPH_CACHE", "1")
sys.argv = [os.path.join(ROOT, "hymm/inference.py")] + sys.argv[1:]
inference = importlib.import_module("hymm.inference")
for flags, mod in ((("NOMAS_PIPE",), "pipe"), (("NOMAS_ATTN", "NOMAS_DEC_ATTN"), "attn"), (("NOMAS_GEMM", "NOMAS_DEC_GEMM"), "gemm"), (("NOMAS_DEC_GEMM", "NOMAS_DEC_FP4"), "dec4"), (("NOMAS_GLUE",), "glue"), (("NOMAS_GRAPH",), "graph"), (("NOMAS_GPU_ARCHIVE",), "history")):
    if mod == "dec4" and not (os.environ.get("NOMAS_DEC_GEMM") == "nvfp4" or os.environ.get("NOMAS_DEC_FP4") == "1"):
        continue
    if any(os.environ.get(f) for f in flags):
        decoder_gemm = os.environ.get("NOMAS_DEC_GEMM")
        if mod == "gemm" and decoder_gemm == "nvfp4":
            os.environ.pop("NOMAS_DEC_GEMM")
        try:
            importlib.import_module(f"nomas.{mod}").install()
        finally:
            if mod == "gemm" and decoder_gemm == "nvfp4":
                os.environ["NOMAS_DEC_GEMM"] = decoder_gemm
if os.environ.get("NOMAS_SYNC_DEBUG"):
    import torch, warnings
    warnings.simplefilter("default")
    _main = inference.main
    def _wrapped():
        torch.cuda.set_sync_debug_mode(1)
        _main()
    inference.main = _wrapped
inference.main()
