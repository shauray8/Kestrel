import queue, threading
from dataclasses import dataclass


@dataclass
class StreamChunk:
    chunk_id: int
    latent: object
    pixels: object = None


@dataclass
class StreamResult:
    video: object


@dataclass
class _StreamFailure:
    error: BaseException
    traceback: object


class _StreamClosed(Exception):
    pass


def _count(stats, key):
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _cache_update(kw):
    t = kw.get("t")
    return t is not None and bool((t == 0).all())


def generate(pipe, *args, on_chunk=None, decode_chunk=None, stats=None, **kwargs):
    if on_chunk is None and decode_chunk is None and stats is None:
        return pipe.generate(*args, **kwargs)
    if getattr(pipe, "infer_mode", "causal_fast") != "causal_fast":
        raise ValueError("streaming chunk observation requires causal_fast")
    original = pipe.model.forward
    chunk_id = 0

    def forward(*a, **kw):
        nonlocal chunk_id
        _count(stats, "forwards")
        out = original(*a, **kw)
        if _cache_update(kw):
            _count(stats, "cache_updates")
            latent = kw["x"][0]
            pixels = decode_chunk(latent, chunk_id) if decode_chunk is not None else None
            if decode_chunk is not None:
                _count(stats, "decoded_chunks")
            chunk = StreamChunk(chunk_id, latent, pixels)
            chunk_id += 1
            _count(stats, "chunks")
            if on_chunk is not None:
                on_chunk(chunk)
        return out

    pipe.model.forward = forward
    try:
        video = pipe.generate(*args, **kwargs)
        _count(stats, "completed")
        return video
    finally:
        pipe.model.forward = original


def iter_generate(pipe, *args, decode_chunk=None, stats=None, **kwargs):
    events = queue.Queue()
    stopped = threading.Event()

    def on_chunk(chunk):
        if stopped.is_set():
            raise _StreamClosed()
        events.put(chunk)

    def run():
        try:
            video = generate(pipe, *args, on_chunk=on_chunk, decode_chunk=decode_chunk, stats=stats, **kwargs)
            events.put(StreamResult(video))
        except _StreamClosed:
            pass
        except BaseException as exc:
            events.put(_StreamFailure(exc, exc.__traceback__))

    thread = threading.Thread(target=run)
    thread.start()
    try:
        while True:
            event = events.get()
            if isinstance(event, _StreamFailure):
                thread.join()
                raise event.error.with_traceback(event.traceback)
            yield event
            if isinstance(event, StreamResult):
                thread.join()
                return
    finally:
        stopped.set()
