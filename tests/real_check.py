"""Check the patched engine on the real model, through the app's own code paths.

Needs the app's engine running (Start engine in the app, or engine.Server(...).start()) and
the same XDG dirs it was started with:

    XDG_DATA_HOME=... XDG_STATE_HOME=... python3 tests/real_check.py

Checks, on one prompt: a capture is written and sane; the live switch removes a direction from
the input of every layer; switching off gives the original back (stale state dropped); a layer
range leaves the inputs before it untouched; generation streams.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k3_abliterator import engine, paths  # noqa: E402
from k3_abliterator.direction import load_capture  # noqa: E402

PROMPT = "Explain why the sky is blue."
HIDDEN = 7168
work = paths.data_dir() / "real-check"
work.mkdir(exist_ok=True)
ctl = paths.data_dir() / "control.txt"
dirs_file = work / "dir.f32"
results = []


def capture(name, ablation):
    out = work / f"{name}.f32"
    out.unlink(missing_ok=True)
    engine.write_control(ctl, ablation, capture=out)
    t0 = time.time()
    engine.chat(engine.PORT, PROMPT, 1, False)
    for _ in range(100):
        if out.exists():
            break
        time.sleep(0.2)
    took = time.time() - t0
    rows = load_capture(out, HIDDEN)
    print(f"  {name}: {rows.shape[0]} rows in {took:.0f} s", flush=True)
    return rows


def rel_proj(rows, d):
    """|row·d| / |row| per row."""
    return np.abs(rows @ d) / (np.linalg.norm(rows, axis=1) + 1e-12)


def check(label, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}", flush=True)


print("1. capture with the ablation off", flush=True)
base = capture("off", engine.Ablation())
check("capture shape and values", base.shape == (94, HIDDEN) and np.isfinite(base).all()
      and (np.linalg.norm(base, axis=1) > 0).all(),
      f"shape {base.shape}, finite {np.isfinite(base).all()}, row norms {np.linalg.norm(base, axis=1)[[0, 45, 93]].round(2)}")

d = base[45] / np.linalg.norm(base[45])
d.astype(np.float32).tofile(dirs_file)
before = rel_proj(base, d)

print("2. capture with that direction removed from all layers and the embedding", flush=True)
full = capture("on", engine.Ablation(str(dirs_file), 1.0, 0, 92, True))
after = rel_proj(full, d)
check("direction gone from every layer input", after.max() < 1e-3,
      f"|cos| before: median {np.median(before):.3f}, max {before.max():.3f}; after: max {after.max():.2e}")
check("hidden state otherwise still alive", (np.linalg.norm(full, axis=1) > 0.1 * np.linalg.norm(base, axis=1)).all(),
      f"row 93 norm {np.linalg.norm(base[93]):.2f} -> {np.linalg.norm(full[93]):.2f}")

print("3. off again", flush=True)
again = capture("off-again", engine.Ablation())
diff = np.abs(again - base).max() / np.abs(base).max()
check("switching off restores the original", diff < 1e-4, f"max relative difference {diff:.2e}")

print("4. layers 50-92 only, no embedding", flush=True)
part = capture("range", engine.Ablation(str(dirs_file), 1.0, 50, 92, False))
same = np.abs(part[:51] - base[:51]).max() / np.abs(base).max()
check("inputs of layers 0-50 untouched", same < 1e-4, f"max relative difference {same:.2e}")
# Layers 50+ stop writing the direction, but what the embedding and layers 0-49 wrote is still
# in the blocks that later layers mix, as in any residual stream: less of it, not none.
late, late0 = rel_proj(part[51:], d).mean(), before[51:].mean()
check("inputs of layers 51+ carry less of it", late < 0.5 * late0, f"mean |cos| {late0:.3f} -> {late:.3f}")

print("5. generation, ablation off", flush=True)
engine.write_control(ctl, engine.Ablation())
reply = engine.chat(engine.PORT, PROMPT, 24, False)
check("generation streams", bool(reply.text.strip()),
      f"{reply.tokens} tokens in {reply.seconds:.0f} s: {reply.text.strip()[:120]!r}")

print(f"\n{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
