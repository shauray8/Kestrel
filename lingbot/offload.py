import contextlib
import torch


@contextlib.contextmanager
def dit_on_cpu(cls):
    orig = cls._configure_model

    def configure(self, *a, **kw):
        model = orig(self, *a, **kw)
        model.to = lambda *aa, **kk: model
        return model
    cls._configure_model = configure
    try:
        yield
    finally:
        cls._configure_model = orig


def _tensors(m):
    return [t for _, t in m.named_parameters(recurse=True)] + [t for _, t in m.named_buffers(recurse=True)]


class BlockOffload:
    def __init__(self, model, device="cuda"):
        self.model, self.device, self.hooks = model, torch.device(device), []
        try:
            del model.to
        except AttributeError:
            pass
        blocks = set(id(t) for b in model.blocks for t in _tensors(b))
        for t in _tensors(model):
            if id(t) not in blocks:
                t.data = t.data.to(self.device)
        self.copy = torch.cuda.Stream()
        self.blocks = list(model.blocks)
        rt = torch.cuda.cudart()
        for b in self.blocks:
            b._off = []
            for t in _tensors(b):
                c = t.data.contiguous()
                if c.numel():
                    rt.cudaHostRegister(c.data_ptr(), c.numel() * c.element_size(), 0)
                t.data = c
                b._off.append((t, c))
        for i, b in enumerate(self.blocks):
            self.hooks.append(b.register_forward_pre_hook(lambda m, a, i=i: self.load(i)))
            self.hooks.append(b.register_forward_hook(lambda m, a, o, i=i: self.drop(i)))
        self.ready = {}

    def fetch(self, i):
        if i in self.ready or i >= len(self.blocks):
            return
        with torch.cuda.stream(self.copy):
            g = [c.to(self.device, non_blocking=True) for _, c in self.blocks[i]._off]
            ev = torch.cuda.Event()
            ev.record(self.copy)
        self.ready[i] = (g, ev)

    def load(self, i):
        self.fetch(i)
        g, ev = self.ready.pop(i)
        torch.cuda.current_stream().wait_event(ev)
        for (t, _), x in zip(self.blocks[i]._off, g):
            x.record_stream(torch.cuda.current_stream())
            t.data = x
        self.fetch(i + 1 if i + 1 < len(self.blocks) else 0)

    def drop(self, i):
        for t, c in self.blocks[i]._off:
            t.data = c

    def resident(self):
        for h in self.hooks:
            h.remove()
        self.hooks, self.ready = [], {}
        rt = torch.cuda.cudart()
        for b in self.blocks:
            for t, c in b._off:
                t.data = c.to(self.device) if t.data.numel() else torch.empty(0, device=self.device, dtype=c.dtype)
        torch.cuda.synchronize()
        for b in self.blocks:
            for _, c in b._off:
                if c.numel():
                    rt.cudaHostUnregister(c.data_ptr())
            b._off = []
        torch.cuda.empty_cache()


def lean_loader(mod):
    import glob, json, os
    from safetensors.torch import load_file
    orig = mod.load_dit_model

    def load(model_cls, checkpoint_dir, subfolder, config, torch_dtype, extra=None):
        dit_dir = mod._resolve_dit_dir(checkpoint_dir, subfolder)
        if os.path.isfile(os.path.join(dit_dir, "config.json")):
            return orig(model_cls, checkpoint_dir, subfolder, config, torch_dtype, extra)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
        try:
            model = model_cls(**mod._dit_kwargs_from_config(config, extra or {}))
        finally:
            torch.set_default_dtype(prev)
        keys = set(model.state_dict())
        seen = set()
        for f in sorted(glob.glob(os.path.join(dit_dir, "*.safetensors"))):
            sd = load_file(f)
            seen |= set(sd)
            model.load_state_dict({k: v.to(torch_dtype) if v.is_floating_point() else v for k, v in sd.items()}, strict=False)
            del sd
        missing = keys - seen
        if missing:
            raise RuntimeError(f"lean_loader missing keys: {sorted(missing)[:8]}")
        return model.to(dtype=torch_dtype)
    mod.load_dit_model = load
