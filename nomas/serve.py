import base64, hashlib, http.server, urllib.parse, importlib, json, os, sys, threading, time, traceback, uuid
from pathlib import Path

ROOT = os.environ.get("NOMAS_ROOT", "/workspace/Hy-Nomas")
PORT = int(os.environ.get("NOMAS_SERVE_PORT", "7860"))
OUT = Path(os.environ.get("NOMAS_SERVE_OUT", "/workspace/serve_out"))
EXAMPLES = [p for p in os.environ.get("NOMAS_SERVE_EXAMPLES", "/workspace/nomas/three.json:/workspace/nomas/prompts_weirdcore.json").split(":") if p]
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["NOMAS_PIPE"] = "1"
os.environ["NOMAS_GRAPH"] = ""
sys.argv = [os.path.join(ROOT, "hymm/inference.py")] + sys.argv[1:]
inference = importlib.import_module("hymm.inference")
for flags, mod in ((("NOMAS_PIPE",), "pipe"), (("NOMAS_ATTN", "NOMAS_DEC_ATTN"), "attn"), (("NOMAS_GEMM", "NOMAS_DEC_GEMM"), "gemm"),
                   (("NOMAS_GLUE",), "glue"), (("NOMAS_GPU_ARCHIVE",), "history")):
    if any(os.environ.get(f) for f in flags):
        importlib.import_module(f"nomas.{mod}").install()

import numpy as np
import torch
from nomas import pipe as npipe

STATE = {"phase": "starting", "log": [], "history": []}
LOCK = threading.Lock()


def note(msg):
    STATE["log"] = (STATE["log"] + [f"{time.strftime('%H:%M:%S')} {msg}"])[-200:]
    print(msg, flush=True)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


args = inference.parse_args()
inference._validate_inference_config(args)
if isinstance(args.inference.video_size, int) or len(args.inference.video_size) == 1:
    v = args.inference.video_size if isinstance(args.inference.video_size, int) else args.inference.video_size[0]
    args.inference.video_size = [v, v]
H, W = int(args.inference.video_size[0]), int(args.inference.video_size[1])
FPS = int(args.inference.video_fps)
torch.set_grad_enabled(False)
device = torch.device("cuda:0")
from hymm.pipeline.causal_ar import CPTCausalInferencePipeline

note("loading models (CPU staging)")
t = time.perf_counter()
P = CPTCausalInferencePipeline(args=args, device=device, dit_model_path=args.dit_weight)
note(f"models loaded in {time.perf_counter() - t:.1f} s")
P._nomas_text_cache, P._nomas_text_ready = {}, True
CPU = torch.device("cpu")


def _mlps():
    return [m for model in (P.transformer, P.vae) for m in model.modules() if hasattr(m, "_nomas_gu")]


def world_to(dev):
    for m in _mlps():
        w, s, n = m._nomas_gu
        w, s = w.to(dev), s.to(dev)
        m._nomas_gu = (w, s, n)
        g, u = m.gate_proj, m.up_proj
        g._buffers["wq"], u._buffers["wq"], g._buffers["sw"], u._buffers["sw"] = w[:n], w[n:], s[:n], s[n:]
        if hasattr(m, "_nomas_cublas"):
            m._nomas_cublas = tuple(x.to(dev) for x in m._nomas_cublas)
    P.transformer.to(dev)
    P.vae.to(dev)
    if dev.type == "cpu":
        torch.cuda.empty_cache()


def _world_on_gpu():
    return next(iter(P.transformer.buffers()), next(P.transformer.parameters())).is_cuda


def encode_text(prompts, negative):
    keys = list(dict.fromkeys([npipe._key(p) for p in prompts] + [npipe._key("")]))
    if float(args.inference.cfg_scale_video) > 1 or float(getattr(args.inference, "camera_cfg_scale", 1)) > 1:
        keys.append(npipe._key(negative or "", is_uncond=True))
    need = [k for k in keys if k not in P._nomas_text_cache]
    if not need:
        return 0.0, 0
    t0 = sync()
    back = _world_on_gpu()
    if back:
        world_to(CPU)
    enc = P.text_encoder
    npipe._remove_offload(enc)
    enc.to(device=device)
    enc.device = device
    for prompt, data_type, is_uncond in need:
        states, mask = P._nomas_original_encode(prompt, data_type, is_uncond)
        P._nomas_text_cache[npipe._key(prompt, data_type, is_uncond)] = (states.detach().cpu(), None if mask is None else mask.detach().cpu())
    enc.to(device=CPU)
    enc.device = CPU
    torch.cuda.empty_cache()
    if back:
        world_to(device)
    return sync() - t0, len(need)


_IMG = {}
_ENCODER = [None]


def encode_image(data, num_frames):
    key = (hashlib.sha256(data).hexdigest(), num_frames)
    if key in _IMG:
        return _IMG[key], 0.0
    from hymm.models.vae import load_vae
    from hymm.pipeline.reference_image import encode_reference_image
    t0 = sync()
    path = OUT / f"ref_{key[0][:16]}.png"
    path.write_bytes(data)
    world_to(CPU)
    try:
        if _ENCODER[0] is None:
            _ENCODER[0] = load_vae(vae_path=str(args.vae.encoder_checkpoint), vae_precision="bf16", device=device)[0]
        else:
            _ENCODER[0].to(device)
        latent = encode_reference_image(_ENCODER[0], path, height=H, width=W, num_frames=num_frames, device=device, dtype=torch.bfloat16)
        latent = latent.detach().cpu()
        _ENCODER[0].to(CPU)
        torch.cuda.empty_cache()
    finally:
        world_to(device)
    _IMG[key] = latent
    return latent, sync() - t0


def _camera(kind, n):
    base = np.load("/workspace/nomas/cameras/follow.npz")
    if kind == "follow":
        return "/workspace/nomas/cameras/follow.npz"
    path = OUT / f"cam_{kind}_{n}.npz"
    if path.exists():
        return str(path)
    K = np.repeat(base["intrinsics"][:1], n, 0).astype(np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], n, 0)
    for i in range(n):
        f = i / max(n - 1, 1)
        yaw, x, z = 0.0, 0.0, 0.0
        if kind == "forward": z = 2.0 * f
        elif kind == "backward": z = -2.0 * f
        elif kind == "strafe_left": x = -1.5 * f
        elif kind == "strafe_right": x = 1.5 * f
        elif kind == "turn_left": yaw = -0.9 * f
        elif kind == "turn_right": yaw = 0.9 * f
        elif kind == "forward_turn_left": z, yaw = 1.5 * f, -0.6 * f
        elif kind == "forward_turn_right": z, yaw = 1.5 * f, 0.6 * f
        c, s = np.cos(yaw), np.sin(yaw)
        c2w[i, :3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        c2w[i, :3, 3] = [x, 0, z]
    np.savez(path, c2ws=c2w, intrinsics=K)
    return str(path)


CAMERAS = ["follow", "static", "forward", "backward", "strafe_left", "strafe_right", "turn_left", "turn_right", "forward_turn_left", "forward_turn_right"]


def _examples():
    rk = getattr(args.inference, "keep_keywords", None) or getattr(getattr(args, "dataset", object()), "json_prompt_keep_keys", None)
    out = []
    for path in EXAMPLES:
        if not os.path.exists(path):
            continue
        try:
            entries = inference._load_prompt_entries(path, desc_field_first=args.inference.desc_field_first, desc_field_rest=args.inference.desc_field_rest,
                max_segments=args.inference.max_segments, keep_keywords=rk, fps=FPS)
            names = [e.get("test_id", str(i)) for i, e in enumerate(json.load(open(path)))]
        except Exception as exc:
            note(f"example load failed {path}: {exc}")
            continue
        for name, entry in zip(names, entries):
            segs = [s.strip() for s in entry[0].split(inference.PROMPT_SEPARATOR) if s.strip()]
            out.append({"name": name, "segments": segs})
    return out


def reset_world():
    if hasattr(P, "_finish_interactive_decode_phase"):
        try:
            P._finish_interactive_decode_phase()
        except Exception:
            pass
    kv = getattr(P, "kv_cache", None)
    if kv is not None:
        for layer in (kv if isinstance(kv, (list, tuple)) else []):
            if isinstance(layer, dict):
                for key in [k for k in layer if str(k).startswith("swa_")]:
                    layer.pop(key, None)
        if hasattr(kv, "reset") and hasattr(kv, "cache_masks"):
            kv.reset()
    P.kv_cache = None
    if hasattr(P, "cache_mask_list"):
        P.cache_mask_list = None
    P.cache_valid_len, P.global_frame_counter = 0, 0
    for name, value in (("segment_token_lengths", None), ("_interactive_cached_action_tails", []), ("_interactive_cached_seg_video_ranges", []),
                        ("_interactive_segment_count", 0), ("_interactive_segment_cache_tokens", []), ("_interactive_segment_latent_lengths", [])):
        if hasattr(P, name):
            if value is None:
                getattr(P, name).clear()
            else:
                setattr(P, name, value)
    for fn in ("_reset_interactive_noise_state", "_reset_interactive_pixelumm_decode_state"):
        if hasattr(P, fn):
            getattr(P, fn)()
    torch.cuda.empty_cache()


def generate(req):
    segs = [s.strip() for s in req.get("segments", []) if s and s.strip()]
    if not segs:
        raise ValueError("enter at least one segment prompt")
    segs = segs[:4]
    seconds = float(req.get("seconds_per_segment", 5))
    num_frames = int(round(seconds * FPS)) * len(segs)
    num_frames -= num_frames % 4
    bounds = [0] + [int(round(num_frames * (i + 1) / len(segs))) for i in range(len(segs))]
    steps = int(req.get("steps", args.inference.infer_steps_video))
    seed = int(req.get("seed", 42))
    cam = _camera(req.get("camera", "follow"), num_frames)
    timings = {}
    t_req = sync()
    timings["text_encode_s"], timings["new_text_encodes"] = encode_text(segs, args.inference.neg_prompt)
    ref = None
    if req.get("image_b64"):
        ref, timings["image_encode_s"] = encode_image(base64.b64decode(req["image_b64"].split(",")[-1]), num_frames)
    reset_world()
    marks = []
    t0 = sync()
    with torch.inference_mode():
        output = P.inference(prompts=segs, num_frames=num_frames, height=H, width=W, guidance_scale=args.inference.cfg_scale_video,
            camera_guidance_scale=args.inference.camera_cfg_scale, num_inference_steps=steps, flow_shift=args.inference.infer_flow_shift_video,
            seed=seed, negative_prompt=args.inference.neg_prompt, segment_frame_boundaries=bounds, camera_trajectory=cam,
            cfg_schedule=getattr(args.inference, "cfg_schedule", None), cfg_skip_steps=getattr(args.inference, "cfg_skip_steps", 0),
            reference_latents=ref, segment_callback=lambda *a, **k: marks.append(time.perf_counter()))
        t1 = sync()
        frames = inference.video_to_uint8_numpy(output[0], chunk_frames=args.inference.output_chunk_frames)
        del output
    t2 = sync()
    rid = uuid.uuid4().hex[:10]
    import imageio
    imageio.mimwrite(OUT / f"{rid}.mp4", frames, fps=FPS, quality=8, macro_block_size=1)
    t3 = time.perf_counter()
    clip = len(frames) / FPS
    timings.update(world_model_s=t1 - t0, to_uint8_s=t2 - t1, mp4_write_s=t3 - t2, request_total_s=t3 - t_req,
                   clip_seconds=clip, realtime_factor=clip / (t1 - t0), frames=len(frames))
    if marks:
        timings["segment_marks_s"] = [m - t0 for m in marks]
    rec = {"id": rid, "video": f"/video/{rid}.mp4", "timings": timings, "camera": req.get("camera", "follow"), "steps": steps,
           "seed": seed, "segments": [s[:200] for s in segs], "image": bool(ref is not None), "time": time.strftime("%H:%M:%S")}
    STATE["history"] = ([rec] + STATE["history"])[:30]
    return rec


PLAY = {"run": False, "frames": [], "base": 0, "keys": set(), "sticky": set(), "kv": 0, "shown": 0, "stats": [], "error": None, "thread": None}
KEYS = ("w", "a", "s", "d", "up", "down", "left", "right")


def _key_traj(keys, n, gain):
    from hymm.models.camera.geometry.keyboard_mouse_to_extrinsics import CameraConfig, FrameInput, generate_trajectory
    base = np.load("/workspace/nomas/cameras/follow.npz")
    K = base["intrinsics"][0]
    cfg = CameraConfig(fx=float(K[0, 0]), fy=float(K[1, 1]))
    fi = [FrameInput(**{k: (k in keys) for k in KEYS}) for _ in range(n)]
    c2w, intr = generate_trajectory(fi, cfg)
    path = OUT / f"play_{uuid.uuid4().hex[:8]}.npz"
    c2w = c2w.cpu().numpy() if hasattr(c2w, "cpu") else np.asarray(c2w)
    intr = intr.cpu().numpy() if hasattr(intr, "cpu") else np.asarray(intr)
    np.savez(path, c2ws=c2w.astype(np.float32), extrinsics=c2w.astype(np.float32), intrinsics=intr.astype(np.float32))
    return str(path)


def _play_loop(cfg):
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(2)
    n = int(round(cfg["seconds"] * FPS)); n -= n % 4
    prompt = first_prompt = cfg["prompt"]
    nxt = cfg.get("prompt_next")
    if not nxt:
        try:
            d = json.loads(first_prompt)
            d["segments"] = [{"description_change": "The camera keeps exploring the same place; the scene stays exactly consistent with no new location and no cut, preserving the same layout, lighting, colors, sky and atmosphere as before."}]
            nxt = json.dumps(d, ensure_ascii=False)
        except Exception:
            nxt = first_prompt + " The scene stays exactly consistent, with the same lighting, colors and atmosphere."
    prompts_all = [first_prompt, nxt]
    est = cfg["seconds"]
    seg = 0
    seg_total = 0
    from omegaconf import open_dict
    saved = (bool(getattr(args.inference, "seg_ar_segment_vae_decode", False)), bool(getattr(args.vae, "stable_prefix_streaming", False)))
    stream = bool(cfg.get("stream_decode", True))
    try:
        with open_dict(args):
            args.inference.seg_ar_segment_vae_decode = stream
            args.vae.stable_prefix_streaming = stream
        with LOCK:
            STATE["phase"] = "playing"
            t_enc, _ = encode_text(prompts_all, args.inference.neg_prompt)
            note(f"play: text encode {t_enc:.2f}s")
            reset_world()
            while PLAY["run"]:
                ahead = (PLAY["base"] + len(PLAY["frames"]) - PLAY["shown"]) / FPS
                if seg > 0 and not cfg.get("script") and ahead > est * 1.15 + 0.15:
                    time.sleep(0.01)
                    continue
                keys = set(PLAY["keys"]) | set(PLAY["sticky"])
                if cfg.get("script"):
                    keys = set(cfg["script"][min(seg_total, len(cfg["script"]) - 1)])
                    if seg_total >= len(cfg["script"]):
                        PLAY["run"] = False
                        break
                PLAY["sticky"] = set()
                kv = PLAY["kv"]
                prompt = first_prompt if seg == 0 else nxt
                t0 = sync()
                cam = _key_traj(keys, n, cfg["gain"])
                with torch.inference_mode():
                    out = P.inference(prompts=[prompt], num_frames=n, height=H, width=W, guidance_scale=args.inference.cfg_scale_video,
                        camera_guidance_scale=args.inference.camera_cfg_scale, num_inference_steps=cfg["steps"],
                        flow_shift=args.inference.infer_flow_shift_video, seed=cfg["seed"], negative_prompt=args.inference.neg_prompt,
                        segment_frame_boundaries=[0, n], camera_trajectory=cam, preserve_kv_cache=True,
                        trajectory_translation_scale=cfg["move"], trajectory_rotation_scale=cfg["turn"],
                        cfg_schedule=getattr(args.inference, "cfg_schedule", None), cfg_skip_steps=getattr(args.inference, "cfg_skip_steps", 0))
                    frames = inference.video_to_uint8_numpy(out[0], chunk_frames=args.inference.output_chunk_frames)
                    del out
                t1 = sync()
                md = getattr(P, "_last_pixelumm_streaming_metadata", None) or {}
                prov = int(md.get("provisional_pixel_frames", 0) or 0) if stream else 0
                if prov:
                    frames = frames[:max(0, len(frames) - prov)]
                os.remove(cam)
                jpg = list(pool.map(lambda f: base64.b64encode(cv2.imencode(".jpg", f[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode(), frames))
                PLAY["frames"].extend((kv, j) for j in jpg)
                drop = max(0, len(PLAY["frames"]) - 24 * 60)
                if drop:
                    PLAY["frames"] = PLAY["frames"][drop:]
                    PLAY["base"] += drop
                gen = t1 - t0
                est = 0.7 * est + 0.3 * gen if seg else gen
                mem = torch.cuda.memory_allocated() / 2**30
                PLAY["stats"] = (PLAY["stats"] + [{"seg": seg, "gen_s": gen, "clip_s": len(frames) / FPS, "rt": len(frames) / FPS / gen,
                                                   "keys": sorted(keys), "gib": mem, "buffer_s": ahead}])[-50:]
                note(f"play seg {seg}: {gen:.2f}s for {len(frames) / FPS:.2f}s video ({len(frames) / FPS / gen:.2f}x RT) keys={sorted(keys)} mem={mem:.1f}GiB")
                seg += 1
                seg_total += 1
                if seg >= cfg["max_segments"]:
                    note("play: max segments reached, restarting world")
                    reset_world()
                    seg = 0
    except Exception as exc:
        traceback.print_exc()
        PLAY["error"] = repr(exc)
        note("play error: " + repr(exc))
    finally:
        with open_dict(args):
            args.inference.seg_ar_segment_vae_decode, args.vae.stable_prefix_streaming = saved
        PLAY["run"] = False
        reset_world()
        STATE["phase"] = "ready"


def play_start(req):
    play_stop()
    arch = req.get("archive", "sink")
    os.environ["NOMAS_ARCHIVE_GPU_SEGMENTS"] = str(int(req.get("recent", 3)))
    os.environ["NOMAS_ARCHIVE_SINK"] = str(int(req.get("sink", 1)))
    os.environ["NOMAS_ARCHIVE_DROP"] = "0" if arch == "full" else "1"
    if arch == "full":
        os.environ["NOMAS_ARCHIVE_GPU_SEGMENTS"] = "0"
    os.environ["NOMAS_SINK_FRAMES"] = str(int(req.get("sink_frames", 4)))
    if arch == "recent":
        os.environ["NOMAS_ARCHIVE_SINK"] = "0"
    cfg = {"prompt": req.get("prompt") or (EXAMPLE_LIST[0]["segments"][0] if EXAMPLE_LIST else "A wide open landscape."),
           "seconds": float(req.get("seconds", 3.0)), "steps": int(req.get("steps", 4)), "seed": int(req.get("seed", 42)),
           "move": float(req.get("move", 0.6)), "stream_decode": bool(req.get("stream_decode", True)), "turn": float(req.get("turn", 1.5)), "gain": 1.0, "max_segments": int(req.get("max_segments", 90)), "script": req.get("script"), "prompt_next": req.get("prompt_next")}
    PLAY.update(run=True, frames=[], base=0, keys=set(), sticky=set(), kv=0, shown=0, stats=[], error=None)
    PLAY["thread"] = threading.Thread(target=_play_loop, args=(cfg,), daemon=True)
    PLAY["thread"].start()
    return {"ok": True, "frames_per_segment": int(round(cfg["seconds"] * FPS))}


def play_stop():
    PLAY["run"] = False
    t = PLAY.get("thread")
    if t is not None and t.is_alive():
        t.join(timeout=60)
    return {"ok": True}


PLAYPAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Hy-Nomas · play</title>
<style>body{font:14px system-ui,sans-serif;margin:0;background:#0c0c0c;color:#ddd}#wrap{display:grid;grid-template-columns:1fr 360px;gap:14px;padding:14px}
canvas{width:100%;background:#000;border:2px solid #222;outline:none}canvas:focus{border-color:#2d5bd7}textarea{width:100%;height:150px;background:#181818;color:#eee;border:1px solid #333;font:11px monospace}
button,select,input{background:#1b1b1b;color:#eee;border:1px solid #444;padding:6px;margin:2px 0}button.go{background:#2d5bd7;border:0;font-weight:600;padding:9px 14px}
.k{display:inline-block;min-width:34px;padding:6px 8px;margin:2px;border:1px solid #444;border-radius:4px;text-align:center;background:#181818}.k.on{background:#2d5bd7;color:#fff}
.big{font-size:20px;font-weight:700}.ok{color:#5c5}.bad{color:#e66}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid #222;padding:2px 4px;text-align:left}label{display:block;color:#999;margin-top:6px}</style></head>
<body><div id="wrap"><div><canvas id="cv" width="832" height="480" tabindex="0"></canvas>
<div><span class="k" id="k_w">W</span><span class="k" id="k_a">A</span><span class="k" id="k_s">S</span><span class="k" id="k_d">D</span>
&nbsp; look: <span class="k" id="k_left">←</span><span class="k" id="k_up">↑</span><span class="k" id="k_down">↓</span><span class="k" id="k_right">→</span>
&nbsp; <span id="hud"></span></div>
<p class="big" id="lat"></p><table id="segs"></table></div>
<div><a href="/" style="color:#7af">← clip mode</a><label>World prompt</label><select id="ex"><option value="">(custom)</option></select><textarea id="prompt"></textarea>
<label>Segment length</label><select id="secs"><option value="3">3 s</option><option value="2">2 s</option><option value="5">5 s</option></select>
<label>Steps</label><select id="steps"><option>4</option><option>3</option></select>
<label>Move gain / turn gain</label><input id="move" value="0.6" size="4"> <input id="turn" value="1.5" size="4">
<label>Seed</label><input id="seed" value="42" size="6"><p><button class="go" onclick="start()">Start world</button> <button onclick="stop()">Stop</button></p>
<p style="color:#888">Click the video, then hold W A S D to move and arrow keys to look. Input is applied at the next segment boundary.</p><div id="log" style="font:11px monospace;color:#777;white-space:pre-wrap"></div></div></div>
<script>
const KEYMAP={KeyW:'w',KeyA:'a',KeyS:'s',KeyD:'d',ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right'};
let held=new Set(),kv=0,buf=[],next=0,shown=0,playing=false,pressT={},lats=[],info;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
function send(){fetch('/api/play/keys',{method:'POST',body:JSON.stringify({keys:[...held],shown:shown,kv:kv})})}
function upd(){for(const k of Object.values(KEYMAP))document.getElementById('k_'+k).className='k'+(held.has(k)?' on':'')}
addEventListener('keydown',e=>{const k=KEYMAP[e.code];if(!k)return;e.preventDefault();if(held.has(k))return;held.add(k);kv++;pressT[kv]=performance.now();upd();send()});
addEventListener('keyup',e=>{const k=KEYMAP[e.code];if(!k)return;e.preventDefault();held.delete(k);kv++;pressT[kv]=performance.now();upd();send()});
async function load(){info=await (await fetch('/api/info')).json();const ex=document.getElementById('ex');info.examples.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.text=e.name;ex.appendChild(o)});
ex.onchange=()=>{const e=info.examples[ex.value];if(e)document.getElementById('prompt').value=e.segments[0]};ex.value=0;ex.onchange()}
async function start(){buf=[];next=0;shown=0;lats=[];const g=id=>document.getElementById(id).value;const body={prompt:g('prompt'),seconds:+g('secs'),steps:+g('steps'),seed:+g('seed'),move:+g('move'),turn:+g('turn')};
await fetch('/api/play/start',{method:'POST',body:JSON.stringify(body)});playing=true;cv.focus();pull()}
async function stop(){playing=false;await fetch('/api/play/stop',{method:'POST'})}
async function pull(){if(!playing)return;try{const r=await (await fetch('/api/play/frames?since='+next)).json();
for(const [v,j] of r.frames){const im=new Image();im.src='data:image/jpeg;base64,'+j;buf.push([v,im])}next=r.next;
document.getElementById('segs').innerHTML='<tr><th>seg</th><th>gen s</th><th>video s</th><th>×RT</th><th>keys</th><th>GiB</th></tr>'+r.stats.slice(-8).reverse().map(s=>'<tr><td>'+s.seg+'</td><td>'+s.gen_s.toFixed(2)+'</td><td>'+s.clip_s.toFixed(2)+'</td><td class="'+(s.rt>=1?'ok':'bad')+'">'+s.rt.toFixed(2)+'</td><td>'+s.keys.join('+')+'</td><td>'+s.gib.toFixed(1)+'</td></tr>').join('');
document.getElementById('log').textContent=(r.error?'ERROR '+r.error+'\n':'')+r.log.slice(-6).join('\n');if(r.error)playing=false}catch(e){}setTimeout(pull,150)}
let last=0,stall=0;function draw(t){requestAnimationFrame(draw);if(!playing||t-last<1000/24-2)return;last=t;if(!buf.length){stall++;document.getElementById('hud').innerHTML='<span class="bad">buffering…</span>';return}
const [v,im]=buf.shift();if(im.complete)ctx.drawImage(im,0,0,832,480);shown++;
for(const k in pressT){if(+k<=v){lats.push(performance.now()-pressT[k]);delete pressT[k]}}
if(shown%12==0)send();const l=lats.slice(-10);const m=l.length?l.reduce((a,b)=>a+b)/l.length/1000:0;
document.getElementById('hud').innerHTML='buffer '+(buf.length/24).toFixed(2)+' s · stalls '+stall;document.getElementById('lat').innerHTML=l.length?'input → screen: <span class="'+(m<2?'ok':'bad')+'">'+m.toFixed(2)+' s</span> (last '+(l[l.length-1]/1000).toFixed(2)+' s)':''}
requestAnimationFrame(draw);load()</script></body></html>"""


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Hy-Nomas on one 5090</title>
<style>body{font:14px system-ui,sans-serif;margin:0;background:#111;color:#ddd}main{display:grid;grid-template-columns:420px 1fr;gap:16px;padding:16px}
h1{font-size:18px;margin:12px 16px}textarea{width:100%;height:110px;background:#1b1b1b;color:#eee;border:1px solid #333;font:12px monospace}
select,input,button{background:#1b1b1b;color:#eee;border:1px solid #444;padding:6px;margin:2px 0}button{background:#2d5bd7;border:0;padding:10px 16px;font-weight:600;cursor:pointer}
button:disabled{background:#444}label{display:block;margin-top:8px;color:#aaa}video{width:100%;background:#000}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #2a2a2a;padding:4px 6px;text-align:left}.big{font-size:22px;font-weight:700}.ok{color:#5c5}.bad{color:#e66}#log{font:11px monospace;color:#888;white-space:pre-wrap;max-height:160px;overflow:auto}</style></head>
<body><h1>Hy-Nomas world model · one RTX 5090 · Kestrel stack <span id="phase" style="font-size:12px;color:#888"></span></h1><main><div>
<label>Example prompt</label><select id="ex"><option value="">(custom)</option></select>
<div id="segs"></div><button onclick="addSeg()" style="background:#333">+ segment</button>
<label>Seconds per segment</label><select id="secs"><option>5</option><option>2.5</option></select>
<label>Camera / action</label><select id="cam"></select>
<label>Steps</label><select id="steps"><option>4</option><option>3</option><option>6</option><option>8</option></select>
<label>Seed</label><input id="seed" value="42">
<label>Reference image (optional)</label><input type="file" id="img" accept="image/*">
<p><button id="go" onclick="go()">Generate</button></p><div id="err" class="bad"></div><div id="log"></div></div>
<div><video id="vid" controls autoplay loop></video><div id="res"></div><h3>History</h3><table id="hist"></table></div></main>
<script>
let info={};function seg(v){const t=document.createElement('textarea');t.value=v||'';document.getElementById('segs').appendChild(t)}
function addSeg(){seg('')}
async function load(){info=await (await fetch('/api/info')).json();const ex=document.getElementById('ex');info.examples.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.text=e.name+' ('+e.segments.length+' seg)';ex.appendChild(o)});
info.cameras.forEach(c=>{const o=document.createElement('option');o.text=c;document.getElementById('cam').appendChild(o)});
ex.onchange=()=>{document.getElementById('segs').innerHTML='';const e=info.examples[ex.value];(e?e.segments:['','']).forEach(seg)};
if(info.examples.length){ex.value=0;ex.onchange()}else{seg('');seg('')}poll()}
async function poll(){const s=await (await fetch('/api/state')).json();document.getElementById('phase').textContent=s.phase;document.getElementById('log').textContent=s.log.slice(-12).join('\n');hist(s.history);setTimeout(poll,2000)}
function fmt(x){return typeof x=='number'?x.toFixed(3):JSON.stringify(x)}
function show(r){document.getElementById('vid').src=r.video;const t=r.timings;const rt=t.realtime_factor;
document.getElementById('res').innerHTML='<p class="big '+(rt>=1?'ok':'bad')+'">'+t.clip_seconds.toFixed(1)+' s of video in '+t.world_model_s.toFixed(2)+' s · '+rt.toFixed(2)+'× real time</p><table>'+Object.entries(t).map(([k,v])=>'<tr><td>'+k+'</td><td>'+fmt(v)+'</td></tr>').join('')+'</table>'}
function hist(h){document.getElementById('hist').innerHTML='<tr><th>time</th><th>camera</th><th>steps</th><th>world model s</th><th>× RT</th><th>encode s</th><th></th></tr>'+h.map(r=>'<tr><td>'+r.time+'</td><td>'+r.camera+'</td><td>'+r.steps+'</td><td>'+r.timings.world_model_s.toFixed(2)+'</td><td>'+r.timings.realtime_factor.toFixed(2)+'</td><td>'+(r.timings.text_encode_s||0).toFixed(2)+'</td><td><a href="'+r.video+'" style="color:#7af">mp4</a></td></tr>').join('')}
async function go(){const b=document.getElementById('go');b.disabled=true;document.getElementById('err').textContent='';const t0=performance.now();
const body={segments:[...document.querySelectorAll('#segs textarea')].map(t=>t.value),camera:cam.value,steps:+steps.value,seed:+seed.value,seconds_per_segment:+secs.value};
const f=document.getElementById('img').files[0];if(f){body.image_b64=await new Promise(r=>{const fr=new FileReader();fr.onload=()=>r(fr.result);fr.readAsDataURL(f)})}
b.textContent='Generating…';try{const r=await (await fetch('/api/generate',{method:'POST',body:JSON.stringify(body)})).json();if(r.error)throw r.error;r.timings.browser_roundtrip_s=(performance.now()-t0)/1000;show(r)}catch(e){document.getElementById('err').textContent=e}b.disabled=false;b.textContent='Generate'}
load()</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.replace('<h1>', '<h1><a href="/play" style="color:#7af;font-size:14px;float:right">▶ play mode (WASD)</a>'), "text/html; charset=utf-8")
        if self.path == "/play":
            return self._send(200, PLAYPAGE, "text/html; charset=utf-8")
        if self.path.startswith("/api/play/frames"):
            since = int(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("since", ["0"])[0])
            start = max(since, PLAY["base"]) - PLAY["base"]
            fr = PLAY["frames"][start:start + 96]
            return self._send(200, json.dumps({"frames": fr, "next": PLAY["base"] + start + len(fr), "stats": PLAY["stats"][-8:],
                                               "error": PLAY["error"], "log": STATE["log"][-6:]}))
        if self.path == "/api/info":
            return self._send(200, json.dumps({"examples": EXAMPLE_LIST, "cameras": CAMERAS, "size": [H, W], "fps": FPS}))
        if self.path == "/api/state":
            return self._send(200, json.dumps({k: STATE[k] for k in ("phase", "log", "history")}))
        if self.path.startswith("/video/"):
            f = OUT / os.path.basename(self.path)
            if f.exists():
                return self._send(200, f.read_bytes(), "video/mp4")
        self._send(404, '{"error":"not found"}')

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
        if self.path.startswith("/api/play/"):
            req = json.loads(body)
            if self.path == "/api/play/keys":
                keys = {k for k in req.get("keys", []) if k in KEYS}
                PLAY["kv"] = max(PLAY["kv"], int(req.get("kv", 0)))
                PLAY["sticky"] |= keys
                PLAY["keys"] = keys
                PLAY["shown"] = max(PLAY["shown"], int(req.get("shown", 0)))
                return self._send(200, '{"ok":true}')
            if self.path == "/api/play/start":
                if STATE["phase"] not in ("ready", "playing"):
                    return self._send(503, json.dumps({"error": f"server {STATE['phase']}"}))
                return self._send(200, json.dumps(play_start(req)))
            if self.path == "/api/play/stop":
                return self._send(200, json.dumps(play_stop()))
        self.rfile_body = body
        if self.path != "/api/generate":
            return self._send(404, '{"error":"not found"}')
        req = json.loads(self.rfile_body)
        if STATE["phase"] != "ready" and not STATE["phase"].startswith("busy"):
            return self._send(503, json.dumps({"error": f"server {STATE['phase']}"}))
        with LOCK:
            STATE["phase"] = "busy"
            try:
                rec = generate(req)
                note(f"done {rec['id']}: world model {rec['timings']['world_model_s']:.2f}s, {rec['timings']['realtime_factor']:.2f}x RT")
                self._send(200, json.dumps(rec))
            except Exception as exc:
                note("error: " + repr(exc))
                traceback.print_exc()
                self._send(500, json.dumps({"error": repr(exc)}))
            finally:
                STATE["phase"] = "ready"


OUT.mkdir(parents=True, exist_ok=True)
EXAMPLE_LIST = _examples()
note(f"{len(EXAMPLE_LIST)} examples loaded")
server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
note(f"serving on :{PORT}")
STATE["phase"] = "warming up"
try:
    warm = EXAMPLE_LIST[0]["segments"] if EXAMPLE_LIST else ["A quiet empty office corridor.", "The camera keeps moving down the corridor."]
    world_to(device)
    for i in range(2):
        rec = generate({"segments": warm, "camera": "follow", "seed": 42})
        note(f"warmup {i + 1}: world model {rec['timings']['world_model_s']:.2f}s ({rec['timings']['realtime_factor']:.2f}x RT), text encode {rec['timings']['text_encode_s']:.2f}s")
    STATE["phase"] = "ready"
except Exception as exc:
    traceback.print_exc()
    STATE["phase"] = "warmup failed: " + repr(exc)
threading.Event().wait()
