"""Write the active ablation into the weights: a new model folder colibri (or transformers) loads.

Only the tensors that write into the hidden state change: embed_tokens, and per layer
o_proj, the routed experts' up-projection back to full width, the shared experts' down_proj
and the dense layer's down_proj. They're BF16 (the MXFP4 experts are never touched) and
come to about 32 GB. The new folder symlinks every original file, adds one shard with the
changed tensors, and points model.safetensors.index.json at it; colibri resolves a tensor
present in two shards through that index.

    W [hidden, in]   ->  W - s·Rᵀ(R W)       (R: orthonormal directions [k, hidden])
    E [vocab, hidden] -> E - s·(E Rᵀ) R
This is exactly what the engine's live switch does to those matrices' outputs.
"""

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHARD = "model-k3abl-00001-of-00001.safetensors"
INDEX = "model.safetensors.index.json"
SKIP = {INDEX, ".coli_analysis.json", ".cache", "abliteration.json", "directions.f32"}
WRITERS = ("self_attn.o_proj.weight", "mlp.down_proj.weight",
           "block_sparse_moe.routed_expert_up_proj.weight",
           "block_sparse_moe.shared_experts.down_proj.weight")


@dataclass
class Target:
    name: str
    shard: str
    shape: tuple
    embedding: bool

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * 2


def read_header(path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)   # round to nearest even


def _prefix(weight_map: dict) -> str:
    for name in weight_map:
        if name.endswith("embed_tokens.weight") and "vision" not in name:
            return name[: -len("embed_tokens.weight")]
    raise ValueError("no embed_tokens.weight in the index")


def plan(src: Path, lo: int, hi: int, embed: bool) -> list[Target]:
    """The tensors a bake of layers lo..hi (and the embedding) rewrites."""
    weight_map = json.loads((src / INDEX).read_text())["weight_map"]
    pfx = _prefix(weight_map)
    names = [f"{pfx}embed_tokens.weight"] if embed else []
    for layer in range(lo, hi + 1):
        names += [f"{pfx}layers.{layer}.{w}" for w in WRITERS if f"{pfx}layers.{layer}.{w}" in weight_map]
    headers, targets = {}, []
    for name in names:
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = read_header(src / shard)[0]
        info = headers[shard][name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name} is {info['dtype']}, only BF16 writers can be baked")
        targets.append(Target(name, shard, tuple(info["shape"]), name.endswith("embed_tokens.weight")))
    return targets


def _ablate(w: np.ndarray, dirs: np.ndarray, scale: float, embedding: bool) -> np.ndarray:
    if embedding:                       # rows live in hidden space
        return w - scale * (w @ dirs.T) @ dirs
    return w - scale * dirs.T @ (dirs @ w)  # columns live in hidden space


def bake(src: Path, out: Path, dirs: np.ndarray, scale: float, lo: int, hi: int, embed: bool,
         progress=lambda done, total, name: None, cancelled=lambda: False) -> list[Target]:
    """Make `out` from `src` with the ablation baked in. `out` must not exist yet."""
    src, out = Path(src), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; pick a new folder")
    dirs = np.asarray(dirs, dtype=np.float32)
    targets = plan(src, lo, hi, embed)
    need = sum(t.nbytes for t in targets)
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    if need + (2 << 30) > free:
        raise OSError(f"needs {need / 2**30:.1f} GB, only {free / 2**30:.1f} GB free on {out.parent}")
    for t in targets:
        rows = t.shape[1] if t.embedding else t.shape[0]
        if rows != dirs.shape[1]:
            raise ValueError(f"{t.name}: hidden size {rows} != direction size {dirs.shape[1]}")

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    header, offset = {}, 0
    for t in targets:
        header[t.name] = {"dtype": "BF16", "shape": list(t.shape), "data_offsets": [offset, offset + t.nbytes]}
        offset += t.nbytes
    header["__metadata__"] = {"format": "pt", "k3_abliterator": "baked directional ablation"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    done = 0
    with open(tmp / SHARD, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for t in targets:
            if cancelled():
                raise InterruptedError("cancelled")
            progress(done, need, t.name)
            shard_header, start = read_header(src / t.shard)
            a, b = shard_header[t.name]["data_offsets"]
            raw = np.memmap(src / t.shard, dtype=np.uint16, mode="r", offset=start + a, shape=t.shape)
            step = 16384 if t.embedding else t.shape[0]
            if t.embedding:
                for r in range(0, t.shape[0], step):
                    f.write(f32_to_bf16(_ablate(bf16_to_f32(raw[r:r + step]), dirs, scale, True)).tobytes())
            else:
                f.write(f32_to_bf16(_ablate(bf16_to_f32(raw[:]), dirs, scale, False)).tobytes())
            del raw
            done += t.nbytes
    progress(done, need, "")

    index = json.loads((src / INDEX).read_text())
    for t in targets:
        index["weight_map"][t.name] = SHARD
    (tmp / INDEX).write_text(json.dumps(index, indent=2))
    dirs.tofile(tmp / "directions.f32")
    (tmp / "abliteration.json").write_text(json.dumps({
        "source": str(src), "scale": scale, "layers": [lo, hi], "embed": embed,
        "directions": int(dirs.shape[0]), "tensors": [t.name for t in targets]}, indent=2))
    for entry in src.iterdir():
        if entry.name not in SKIP and not (tmp / entry.name).exists():
            os.symlink(entry.resolve(), tmp / entry.name)
    tmp.rename(out)
    return targets
