"""The long jobs, written as plain functions of a Job (progress, cancel, log)."""

import time

import numpy as np

from . import direction, engine
from .prompts import looks_refused
from .store import model_shape


class Cancelled(Exception):
    pass


class Job:
    """What a job function can call. The GUI's runner fills these in with signals."""
    def progress(self, done: int, total: int, text: str) -> None: ...
    def log(self, text: str) -> None: ...
    def stream(self, pane: str, kind: str, chunk: str) -> None: ...
    def row(self, data: dict) -> None: ...
    def cancelled(self) -> bool: return False

    def check(self):
        if self.cancelled():
            raise Cancelled()


def _wait_file(path, seconds=60.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.2)
    return path.exists()


def capture(job: Job, store, server: engine.Server, prompts: list[str]) -> np.ndarray:
    """The capture of each prompt under the current ablation; cached ones are reused."""
    ablation = store.ablation()
    out, todo = [], []
    for p in prompts:
        path = store.capture_path(p, ablation)
        out.append(path)
        if not path.exists():
            todo.append((p, path))
    if len(prompts) - len(todo):
        job.log(f"{len(prompts) - len(todo)} of {len(prompts)} prompts already captured, reusing them")
    times = []
    for i, (prompt, path) in enumerate(todo):
        job.check()
        eta = ""
        if times or store.state.seconds_per_prompt:
            per = np.mean(times) if times else store.state.seconds_per_prompt
            eta = f", about {_fmt((len(todo) - i) * per)} left"
        job.progress(i, len(todo), f"Capturing {i + 1} of {len(todo)}{eta}")
        engine.write_control(store.control, ablation, capture=path)
        t0 = time.time()
        engine.chat(server.port, prompt, 1, store.state.thinking, cancelled=job.cancelled)
        if not _wait_file(path):
            raise RuntimeError("The engine answered but wrote no capture. Is it the patched engine? "
                               "Check the engine log tab.")
        times.append(time.time() - t0)
        store.state.seconds_per_prompt = float(np.mean(times))
        store.save()
    engine.write_control(store.control, ablation)
    job.progress(len(todo), len(todo), "Captured")
    hidden = model_shape(store.state.model)[0]
    return np.stack([direction.load_capture(p, hidden) for p in out])


def find_direction(job: Job, store, server) -> direction.LayerStats:
    s = store.state
    job.log(f"Capturing {len(s.refused)} refused and {len(s.normal)} normal prompts")
    refused = capture(job, store, server, s.refused)
    normal = capture(job, store, server, s.normal)
    stats = direction.layer_stats(refused, normal)
    best = stats.best()
    job.log(f"Best layer {best}: separation {stats.score[best]:.2f}")
    return stats


def ask(job: Job, store, server, prompt: str, max_tokens: int, which: list[str]) -> dict:
    """Answer one prompt with the ablation off ('original') and/or on ('ablated')."""
    replies = {}
    for pane in which:
        job.check()
        engine.write_control(store.control, store.ablation(on=pane == "ablated"))
        job.progress(0, 0, f"Generating ({pane})... the first token waits for the prompt to be read")
        reply = engine.chat(server.port, prompt, max_tokens, store.state.thinking,
                            on_text=lambda kind, chunk, pane=pane: job.stream(pane, kind, chunk),
                            cancelled=job.cancelled)
        replies[pane] = reply
        if reply.tokens:
            store.state.seconds_per_token = reply.seconds / reply.tokens
            store.save()
    engine.write_control(store.control, store.ablation())
    return replies


def refusal_check(job: Job, store, server, max_tokens: int, original: bool) -> dict:
    """Run every refused prompt; count the answers that still read like refusals."""
    s = store.state
    modes = (["original"] if original else []) + ["ablated"]
    counts = {m: 0 for m in modes}
    total = len(s.refused) * len(modes)
    n = 0
    for i, prompt in enumerate(s.refused):
        result = {"index": i}
        for mode in modes:
            job.check()
            per = s.seconds_per_token * max_tokens + s.seconds_per_prompt if s.seconds_per_token else 0
            eta = f", about {_fmt((total - n) * per)} left" if per else ""
            job.progress(n, total, f"Prompt {i + 1} of {len(s.refused)} ({mode}){eta}")
            engine.write_control(store.control, store.ablation(on=mode == "ablated"))
            reply = engine.chat(server.port, prompt, max_tokens, s.thinking, cancelled=job.cancelled)
            refused = looks_refused(reply.text or reply.reasoning)
            counts[mode] += refused
            result[mode] = (refused, (reply.text or reply.reasoning).strip())
            n += 1
        job.row(result)
    engine.write_control(store.control, store.ablation())
    job.progress(total, total, "Done")
    return counts


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)} s"
    if seconds < 5400:
        return f"{int(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"
