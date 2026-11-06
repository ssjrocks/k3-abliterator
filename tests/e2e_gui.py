"""End-to-end GUI test against a fake engine (no model, no colibri):

    QT_QPA_PLATFORM=offscreen python3 tests/e2e_gui.py build/e2e

The fake speaks the gateway's API and follows the real control file: captures are random rows
plus a planted "refusal" direction for prompts containing REFUSE, and the ablation (direction,
scale, layer range) is applied to them the way the engine applies it. Replies refuse while the
planted component survives. Screenshots of each tab land in the output folder.
"""

import faulthandler
import json
import os
import sys
import threading
import traceback
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "build/e2e").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["XDG_DATA_HOME"] = str(OUT / "data")
os.environ["XDG_STATE_HOME"] = str(OUT / "state")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
faulthandler.dump_traceback_later(180, exit=True)
sys.excepthook = lambda *a: (traceback.print_exception(*a), os._exit(1))

from k3_abliterator import appreports, bake, engine  # noqa: E402

appreports.report_error = lambda *a, **k: None          # test failures must not reach the VPS
appreports.send_report = lambda *a, **k: None
appreports.init = lambda *a, **k: None

HIDDEN, LAYERS, PORT = 64, 6, 18611
rng = np.random.default_rng(7)
PLANTED = rng.normal(size=HIDDEN).astype(np.float32)
PLANTED /= np.linalg.norm(PLANTED)
CTL = OUT / "data/k3-abliterator/runs/default/control.txt"


def read_ctl():
    cfg = {}
    if CTL.exists():
        for line in CTL.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k] = v
    return cfg


def hidden_rows(prompt, cfg):
    seed = zlib.crc32(prompt.encode())   # hash() changes every run
    rows = np.random.default_rng(seed).normal(size=(LAYERS + 1, HIDDEN)).astype(np.float32)
    if "REFUSE" in prompt:
        rows[2:] += 10 * PLANTED
    if cfg.get("direction"):
        dirs = np.fromfile(cfg["direction"], dtype=np.float32).reshape(-1, HIDDEN)
        lo, hi = map(int, cfg.get("layers", f"0-{LAYERS - 1}").split("-"))
        s = float(cfg.get("scale", 1))
        for row in range(lo + 1, min(hi + 1, LAYERS) + 1):   # rows after an ablated writer
            rows[row] -= s * (rows[row] @ dirs.T) @ dirs
    return rows


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        cfg = read_ctl()
        rows = hidden_rows(prompt, cfg)
        if cfg.get("capture"):
            rows.tofile(cfg["capture"])
        refuses = float(rows[-1] @ PLANTED) > 5
        text = "I'm sorry, but I can't help with that." if refuses else "Sure, here is an answer to that."
        words = text.split(" ")[: body["max_tokens"]]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i, w in enumerate(words):
            chunk = {"choices": [{"delta": {"content": (" " if i else "") + w}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": len(words)}})}\n\n'.encode())
        self.wfile.write(b"data: [DONE]\n\n")


def fake_model():
    """A tiny model folder the bake can work on."""
    d = OUT / "model"
    if d.exists():
        return d
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"text_config": {"hidden_size": HIDDEN, "num_hidden_layers": LAYERS}}))
    import struct
    pfx = "language_model.model."
    tensors = {f"{pfx}embed_tokens.weight": (100, HIDDEN)}
    for i in range(LAYERS):
        tensors[f"{pfx}layers.{i}.self_attn.o_proj.weight"] = (HIDDEN, 32)
    header, off, blobs = {}, 0, []
    for name, shape in tensors.items():
        raw = bake.f32_to_bf16(rng.normal(size=shape).astype(np.float32)).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    (d / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))
    (d / bake.INDEX).write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    return d


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    model = fake_model()

    engine.Server.running = lambda self: False
    engine.Server.healthy = lambda self: False
    engine.Server.memory_gb = lambda self: 44.0
    engine.Server.stop = lambda self: None
    engine.other_servers = lambda: []

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from k3_abliterator import gui, updater
    updater.due = lambda: False
    errors = []
    QMessageBox.critical = staticmethod(lambda _p, title, msg: errors.append(f"{title}: {msg}"))
    QMessageBox.warning = staticmethod(lambda _p, title, msg: errors.append(f"{title}: {msg}"))
    QMessageBox.information = staticmethod(lambda _p, title, msg: errors.append(f"{title}: {msg}"))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)

    app = QApplication(sys.argv)
    win = gui.Window()
    win.resize(1180, 820)
    win.model_edit.setText(str(model))
    win._model_changed()
    assert win.store.state.model == str(model), errors
    engine.Server.running = lambda self: True
    engine.Server.healthy = lambda self: True
    win.server.port = PORT
    win.lo_spin.setMaximum(LAYERS - 1)
    win.hi_spin.setMaximum(LAYERS - 1)
    win.hi_spin.setValue(LAYERS - 1)
    win.show()

    refused = [f"REFUSE request number {i} about a forbidden thing" for i in range(12)]
    normal = [f"ordinary question {i} about cooking" for i in range(12)]
    win.refused_edit.setPlainText("\n".join(refused))
    win.normal_edit.setPlainText("\n".join(normal))
    win._save_prompts()
    win._poll_engine()

    steps = []
    first_score = []

    def wait_job(then):
        def poll():
            if win.job and win.job.isRunning():
                QTimer.singleShot(50, poll)
            else:
                QTimer.singleShot(50, then)
        QTimer.singleShot(50, poll)

    def shot(name, tab):
        win.tabs.setCurrentIndex(tab)
        app.processEvents()
        win.grab().save(str(OUT / f"{name}.png"))

    def step1():
        shot("1-prompts", 0)
        win.tabs.setCurrentIndex(1)
        win._find()
        wait_job(step2)

    def step2():
        assert win.stats is not None, errors
        best = win.picked_layer
        steps.append(f"found: best row {best}, score {win.stats.score[best]:.2f}")
        first_score.append(float(win.stats.score[best]))
        assert best >= 2, f"picked row {best}, the plant starts at row 2"
        shot("2-directions-found", 1)
        win._add_direction()
        assert len(win.store.state.rounds) == 1
        win.tabs.setCurrentIndex(2)
        win.check_original.setChecked(True)
        win._refusal_check()
        wait_job(step3)

    def step3():
        text = win.check_summary.text()
        steps.append(f"refusal check: {text}")
        assert "original: 12 of 12 refused" in text and "ablated: 0 of 12 refused" in text, text
        win.test_prompt.setPlainText(refused[0])
        win._ask(["original", "ablated"])
        wait_job(step4)

    def step4():
        o, a = win.out["original"].toPlainText(), win.out["ablated"].toPlainText()
        steps.append(f"ask both: original={o.splitlines()[0]!r} ablated={a.splitlines()[0]!r}")
        assert "sorry" in o and "Sure" in a
        shot("3-test", 2)
        # Half strength leaves the refusal: the control file follows the slider.
        win.scale_slider.setValue(20)
        win._ask(["ablated"])
        wait_job(step5)

    def step5():
        a = win.out["ablated"].toPlainText()
        steps.append(f"at 20%: {a.splitlines()[0]!r}")
        assert "sorry" in a
        win.scale_slider.setValue(100)
        # A second round after the first: captures under the ablation, nothing left to find.
        win.tabs.setCurrentIndex(1)
        win._find()
        wait_job(step6)

    def step6():
        best = win.picked_layer
        steps.append(f"second round: best score {win.stats.score[best]:.2f} (first round {first_score[0]:.2f})")
        # What's left is the part of the plant the first estimate missed: clearly weaker.
        assert win.stats.score[best] < 0.7 * first_score[0], "the second round should find less"
        win.tabs.setCurrentIndex(3)
        win.bake_out.setText(str(OUT / "baked"))
        win._bake_plan()
        steps.append(f"bake plan: {win.bake_info.text()}")
        if (OUT / "baked").exists():
            import shutil
            shutil.rmtree(OUT / "baked")
        win._bake()
        wait_job(step7)

    def step7():
        assert (OUT / "baked" / bake.SHARD).exists(), errors
        steps.append(f"baked: {win.bake_result.text().splitlines()[0]}")
        shot("4-bake", 3)
        win.close()
        app.quit()

    QTimer.singleShot(300, step1)
    app.exec()
    httpd.shutdown()
    for s in steps:
        print(s)
    if errors:
        print("dialogs:", *errors, sep="\n  ")
    assert len(steps) == 7, f"stopped after {len(steps)} steps"
    print("e2e: all steps passed; screenshots in", OUT)


if __name__ == "__main__":
    main()
