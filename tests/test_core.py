"""Unit tests that need no model and no engine:  python3 -m pytest tests/"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    # Never touch the real runs, logs or report queue.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


from k3_abliterator import bake, direction, engine, prompts  # noqa: E402


def test_layer_stats_finds_planted_direction():
    rng = np.random.default_rng(0)
    rows, hidden, n = 6, 64, 24
    planted = rng.normal(size=hidden)
    planted /= np.linalg.norm(planted)
    normal = rng.normal(size=(n, rows, hidden))
    refused = rng.normal(size=(n, rows, hidden))
    refused[:, 3] += 4 * planted          # only row 3 carries it
    stats = direction.layer_stats(refused.astype(np.float32), normal.astype(np.float32))
    assert int(np.argmax(stats.score)) == 3
    assert stats.score[3] > 2 and abs(stats.score[0]) < 1.5
    assert abs(float(stats.dirs[3] @ planted)) > 0.8   # 24 noisy samples in 64 dims: ~0.87 expected


def test_best_prefers_middle_layers():
    s = direction.LayerStats(np.zeros((10, 4)), np.array([9, 1, 2, 5, 3, 2, 1, 1, 1, 9.0]), 1, 1)
    assert s.best() == 3


def test_stack_is_orthonormal():
    a = np.array([1, 0, 0, 0], np.float32)
    b = np.array([1, 1, 0, 0], np.float32)
    out = direction.stack([a, b, a])       # the repeat adds nothing
    assert out.shape == (2, 4)
    assert np.allclose(out @ out.T, np.eye(2), atol=1e-6)
    assert direction.stack([]).shape[0] == 0


def test_bf16_round_trip():
    x = np.array([1.0, -2.5, 3.14159, 1e-3, 65504.0], np.float32)
    back = bake.bf16_to_f32(bake.f32_to_bf16(x))
    assert np.allclose(back, x, rtol=1 / 128)
    assert bake.bf16_to_f32(bake.f32_to_bf16(np.array([1.0], np.float32)))[0] == 1.0


def _write_shard(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, arr in tensors.items():
        raw = bake.f32_to_bf16(arr).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


def _read(path, name):
    hdr, start = bake.read_header(path)
    a, b = hdr[name]["data_offsets"]
    raw = np.frombuffer(path.read_bytes()[start + a:start + b], dtype=np.uint16).reshape(hdr[name]["shape"])
    return bake.bf16_to_f32(raw)


def test_bake_matches_live_ablation(tmp_path):
    rng = np.random.default_rng(1)
    D, V, I = 16, 40, 24
    pfx = "language_model.model."
    src = tmp_path / "model"
    src.mkdir()
    t1 = {f"{pfx}embed_tokens.weight": rng.normal(size=(V, D)).astype(np.float32),
          f"{pfx}layers.0.self_attn.o_proj.weight": rng.normal(size=(D, I)).astype(np.float32),
          f"{pfx}layers.0.mlp.down_proj.weight": rng.normal(size=(D, I)).astype(np.float32)}
    t2 = {f"{pfx}layers.1.self_attn.o_proj.weight": rng.normal(size=(D, I)).astype(np.float32),
          f"{pfx}layers.1.block_sparse_moe.routed_expert_up_proj.weight": rng.normal(size=(D, 8)).astype(np.float32),
          f"{pfx}layers.1.block_sparse_moe.shared_experts.down_proj.weight": rng.normal(size=(D, I)).astype(np.float32),
          f"{pfx}layers.1.input_layernorm.weight": rng.normal(size=(D,)).astype(np.float32)}
    _write_shard(src / "model-00001-of-00002.safetensors", t1)
    _write_shard(src / "model-00002-of-00002.safetensors", t2)
    wm = {**{k: "model-00001-of-00002.safetensors" for k in t1}, **{k: "model-00002-of-00002.safetensors" for k in t2}}
    (src / bake.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (src / "config.json").write_text("{}")
    dirs = direction.stack([rng.normal(size=D).astype(np.float32), rng.normal(size=D).astype(np.float32)])

    out = tmp_path / "baked"
    targets = bake.bake(src, out, dirs, 1.0, 1, 1, True)
    names = {t.name for t in targets}
    assert f"{pfx}layers.0.self_attn.o_proj.weight" not in names      # outside 1..1
    assert f"{pfx}layers.1.input_layernorm.weight" not in names       # not a writer
    assert len(names) == 4
    index = json.loads((out / bake.INDEX).read_text())["weight_map"]
    assert index[f"{pfx}layers.1.self_attn.o_proj.weight"] == bake.SHARD
    assert index[f"{pfx}layers.0.self_attn.o_proj.weight"] == "model-00001-of-00002.safetensors"
    assert (out / "config.json").is_symlink() and (out / "model-00002-of-00002.safetensors").is_symlink()

    # The baked matrix's output equals the live switch's x - s Σ (x·r) r on the original output.
    name = f"{pfx}layers.1.self_attn.o_proj.weight"
    w0, w1 = _read(src / wm[name], name), _read(out / bake.SHARD, name)
    x = rng.normal(size=I).astype(np.float32)
    y = w0 @ x
    live = y - dirs.T @ (dirs @ y)
    assert np.allclose(w1 @ x, live, atol=0.05 * np.abs(live).max())
    assert np.abs(dirs @ (w1 @ x)).max() < 0.05 * np.abs(y).max()
    # Embedding rows lose the directions too.
    e1 = _read(out / bake.SHARD, f"{pfx}embed_tokens.weight")
    assert np.abs(e1 @ dirs.T).max() < 0.05 * np.abs(e1).max()

    with pytest.raises(FileExistsError):
        bake.bake(src, out, dirs, 1.0, 1, 1, True)


def test_control_file(tmp_path):
    ctl = tmp_path / "ctl.txt"
    engine.write_control(ctl, engine.Ablation())
    assert ctl.read_text().strip() == ""
    d = tmp_path / "d.f32"
    np.ones(4, np.float32).tofile(d)
    engine.write_control(ctl, engine.Ablation(str(d), 0.8, 10, 60, False), capture=tmp_path / "c.f32")
    lines = ctl.read_text().splitlines()
    assert f"direction={d}" in lines and "scale=0.8" in lines and "layers=10-60" in lines
    assert "embed=0" in lines and f"capture={tmp_path / 'c.f32'}" in lines


def test_refusal_keywords():
    assert prompts.looks_refused("I'm sorry, but I can't help with that.")
    assert prompts.looks_refused("I cannot provide instructions for")
    assert not prompts.looks_refused("Sure! Here's a simple tomato sauce: first, ...")


def test_store_round_trip():
    from k3_abliterator.store import Store
    s = Store("t")
    s.state.refused = ["a", "b"]
    s.add_round(np.arange(8, dtype=np.float32), 40, 3.2, 2, 2)
    again = Store("t")
    assert again.state.refused == ["a", "b"] and again.state.rounds[0].layer == 40
    abl = again.ablation()
    assert abl.directions and np.fromfile(abl.directions, np.float32).shape == (8,)
    assert again.ablation(on=False).directions is None
    again.remove_round(0)
    assert not again.active.exists() and again.ablation().directions is None
