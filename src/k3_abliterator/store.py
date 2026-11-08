"""A run: the prompts, the directions found so far and the cached captures, kept between sessions."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import direction, paths
from .engine import Ablation
from .prompts import DEFAULT_NORMAL


@dataclass
class Round:
    file: str           # one direction, [hidden] f32
    layer: int          # capture row it came from
    score: float
    n_refused: int
    n_normal: int
    created: float = field(default_factory=time.time)


@dataclass
class State:
    model: str = str(paths.DEFAULT_MODEL)
    refused: list[str] = field(default_factory=list)    # never saved: entered each session
    normal: list[str] = field(default_factory=lambda: list(DEFAULT_NORMAL))
    rounds: list[Round] = field(default_factory=list)
    scale: float = 1.0
    lo: int = 0
    hi: int = 92
    embed: bool = True
    thinking: bool = False
    seconds_per_prompt: float = 0.0     # measured capture time, for estimates
    seconds_per_token: float = 0.0      # measured generation time


def model_shape(model_dir) -> tuple[int, int]:
    """(hidden size, layers) from the model's config.json."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    return int(text["hidden_size"]), int(text["num_hidden_layers"])


class Store:
    def __init__(self, name: str = "default"):
        self.dir = paths.data_dir() / "runs" / name
        self.captures = self.dir / "captures"
        self.dirs_dir = self.dir / "directions"
        for d in (self.captures, self.dirs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.control = self.dir / "control.txt"
        self.active = self.dir / "active.f32"
        self.state = self._load()

    def _load(self) -> State:
        try:
            raw = json.loads((self.dir / "state.json").read_text())
        except (OSError, ValueError):
            return State()
        rounds = [Round(**r) for r in raw.pop("rounds", [])]
        stale = raw.pop("refused", None)
        known = State.__dataclass_fields__
        state = State(rounds=rounds, **{k: v for k, v in raw.items() if k in known})
        if stale is not None:           # written by 0.1.1 or older: drop it from disk now
            self.state = state
            self.save()
        return state

    def save(self) -> None:
        """Everything but the refused prompts, which only ever live in memory."""
        data = asdict(self.state)
        del data["refused"]
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.dir / "state.json")

    # ---- directions

    def add_round(self, vec: np.ndarray, layer: int, score: float, n_refused: int, n_normal: int) -> Round:
        path = self.dirs_dir / f"round-{len(self.state.rounds) + 1}-{int(time.time())}.f32"
        vec.astype(np.float32).tofile(path)
        r = Round(str(path), layer, float(score), n_refused, n_normal)
        self.state.rounds.append(r)
        self.write_active()
        self.save()
        return r

    def remove_round(self, i: int) -> None:
        del self.state.rounds[i]
        self.write_active()
        self.save()

    def directions(self) -> np.ndarray:
        rows = [np.fromfile(r.file, dtype=np.float32) for r in self.state.rounds]
        return direction.stack(rows)

    def write_active(self) -> None:
        stacked = self.directions()
        if len(stacked):
            tmp = self.active.with_suffix(".tmp")
            stacked.tofile(tmp)
            tmp.replace(self.active)
        else:
            self.active.unlink(missing_ok=True)

    def ablation(self, on: bool = True) -> Ablation:
        s = self.state
        if not on or not s.rounds:
            return Ablation()
        if not self.active.exists():
            self.write_active()
        return Ablation(str(self.active), s.scale, s.lo, s.hi, s.embed)

    def capture_path(self, prompt: str, ablation: Ablation) -> Path:
        key = hashlib.sha1(f"{prompt}\0{ablation.signature()}\0{self.state.thinking}\0{self.state.model}".encode())
        return self.captures / f"{key.hexdigest()}.f32"
