"""The refusal direction: difference of means between refused and normal prompts, per layer.

A capture is the engine's layer-input mix at the last prompt token: n_layers + 1 rows of
hidden-size f32 (row L is what layer L reads; the last row is what the head reads).
"""

from dataclasses import dataclass

import numpy as np


def load_capture(path, hidden: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0 or data.size % hidden:
        raise ValueError(f"{path}: {data.size} floats is not a whole number of {hidden}-wide rows")
    return data.reshape(-1, hidden)


@dataclass
class LayerStats:
    dirs: np.ndarray        # [rows, hidden], unit length
    score: np.ndarray       # [rows], held-out separation (Cohen's d) along each row's direction
    n_refused: int
    n_normal: int

    def best(self) -> int:
        """Highest score in the middle of the stack: the first and last layers separate
        prompts by surface wording, and removing a direction there hurts the most."""
        rows = len(self.score)
        lo, hi = int(rows * 0.2), max(int(rows * 0.8), int(rows * 0.2) + 1)
        window = np.nan_to_num(self.score[lo:hi], nan=-np.inf)
        return lo + int(np.argmax(window))


def _cohen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row effect size between projections a [n, rows] and b [m, rows]."""
    pooled = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2) + 1e-9
    return (a.mean(axis=0) - b.mean(axis=0)) / pooled


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def layer_stats(refused: np.ndarray, normal: np.ndarray) -> LayerStats:
    """refused [n, rows, hidden], normal [m, rows, hidden] -> per-row direction and score.

    The score is measured on prompts the direction wasn't computed from (two halves, each
    scored with the other's direction), so a direction that only memorised the examples
    doesn't look good. With fewer than 4 prompts a side it falls back to in-sample.
    """
    refused = refused.astype(np.float64)
    normal = normal.astype(np.float64)
    dirs = _unit(refused.mean(axis=0) - normal.mean(axis=0))
    if min(len(refused), len(normal)) >= 4:
        scores = []
        for keep in (0, 1):
            r_fit, r_eval = refused[keep::2], refused[1 - keep::2]
            n_fit, n_eval = normal[keep::2], normal[1 - keep::2]
            d = _unit(r_fit.mean(axis=0) - n_fit.mean(axis=0))
            scores.append(_cohen(np.einsum("nrh,rh->nr", r_eval, d), np.einsum("nrh,rh->nr", n_eval, d)))
        score = np.mean(scores, axis=0)
    else:
        score = _cohen(np.einsum("nrh,rh->nr", refused, dirs), np.einsum("nrh,rh->nr", normal, dirs))
    return LayerStats(dirs.astype(np.float32), score, len(refused), len(normal))


def stack(rows: list[np.ndarray]) -> np.ndarray:
    """The active directions as one orthonormal [k, hidden] array (what the engine and the bake use)."""
    out = []
    for r in rows:
        v = r.astype(np.float64).copy()
        for q in out:
            v -= (v @ q) * q
        n = np.linalg.norm(v)
        if n > 1e-6:
            out.append(v / n)
    if not out:
        return np.zeros((0, rows[0].size if rows else 0), dtype=np.float32)
    return np.array(out, dtype=np.float32)
