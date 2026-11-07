"""The colibri side: the patched engine, its server, the control file and chat requests.

`coli serve` runs the engine this app builds (COLI_ENGINE), with K3_ABL_CTL naming a control
file the engine re-reads at the start of every request. Writing that file before a request is
how the live switch, the strength and the capture work. The stock ~/colibri/c/kimi_k3 is never
rebuilt or touched.
"""

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

LOG = logging.getLogger(__name__)

HERE = Path(__file__).parent / "engine"
UNIT = "k3-abliterator-engine"
PORT = 8011
RAM_CAP = "60G"        # measured: the stock engine peaks at 44.6 GB on this model with a 64k context
CONTEXT = 8192


@dataclass
class Ablation:
    """What the engine applies to the next request."""
    directions: str | None = None    # path of [k, hidden] f32, None = off
    scale: float = 1.0
    lo: int = 0
    hi: int = 92
    embed: bool = True

    def signature(self) -> str:
        if not self.directions:
            return "off"
        digest = hashlib.sha1(Path(self.directions).read_bytes()).hexdigest()[:12]
        return f"{digest}-s{self.scale:.3f}-{self.lo}-{self.hi}-e{int(self.embed)}"


def write_control(path: Path, ablation: Ablation, capture: Path | None = None) -> None:
    lines = []
    if ablation.directions and ablation.scale:
        lines += [f"direction={ablation.directions}", f"scale={ablation.scale}",
                  f"layers={ablation.lo}-{ablation.hi}", f"embed={int(ablation.embed)}"]
    if capture:
        lines.append(f"capture={capture}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- build

def engine_binary() -> Path:
    return paths.data_dir() / "engine" / "kimi_k3"


def _stamp() -> str:
    head = subprocess.run(["git", "-C", str(paths.COLIBRI), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    h = hashlib.sha1(head.encode())
    for name in ("kimi_k3.patch", "k3_dirabl.h", "build.sh"):
        h.update((HERE / name).read_bytes())
    for name in ("kimi_k3.c", ".build-config"):
        p = paths.COLIBRI / "c" / name
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


def ensure_engine() -> Path:
    """Build the patched engine when it's missing or colibri / the patch changed. Takes seconds."""
    out = engine_binary()
    stamp_file = out.with_suffix(".stamp")
    stamp = _stamp()
    if out.exists() and stamp_file.exists() and stamp_file.read_text() == stamp:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("building the engine into %s", out)
    env = dict(os.environ, COLIBRI_DIR=str(paths.COLIBRI))
    r = subprocess.run(["bash", str(HERE / "build.sh"), str(out)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Building the engine failed:\n{(r.stderr or r.stdout)[-3000:]}")
    stamp_file.write_text(stamp)
    return out


# ---------------------------------------------------------------- server

def other_servers() -> list[tuple[int, str]]:
    """Kimi K3 engines not run by this app: (pid of their coli launcher, its command line)."""
    found = []
    ours = _unit_pids()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            exe = os.readlink(proc / "exe")
            if os.path.basename(exe) != "kimi_k3" or int(proc.name) in ours:
                continue
            ppid = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[1])
            cmd = (Path("/proc") / str(ppid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            found.append((ppid, cmd.strip()))
        except (OSError, ValueError, IndexError):
            continue
    return found


def stop_pid(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)


def _systemctl(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def _unit_pids() -> set[int]:
    r = _systemctl("show", UNIT, "-p", "MainPID", "--value")
    pids = set()
    try:
        main = int(r.stdout.strip() or 0)
    except ValueError:
        main = 0
    if main:
        pids.add(main)
        children = Path(f"/proc/{main}/task/{main}/children")
        if children.exists():
            pids |= {int(p) for p in children.read_text().split()}
    return pids


@dataclass
class Server:
    model: Path
    control: Path
    port: int = PORT
    log: Path = field(default_factory=paths.engine_log)

    def command(self, engine: Path) -> list[str]:
        coli = paths.COLIBRI / "c" / "coli"
        env = [f"COLI_ENGINE={engine}", f"K3_ABL_CTL={self.control}", "KV_SLOTS=1"]
        serve = [str(coli), "serve", "--model", str(self.model), "--host", "127.0.0.1",
                 "--port", str(self.port), "--ctx", str(CONTEXT), "--gpu", "auto", "--auto-tier"]
        heavy = shutil.which("heavy") or str(Path.home() / ".local/bin/heavy")
        if Path(heavy).exists():
            # Own cgroup with a RAM cap: if the engine overruns, only it is killed.
            return [heavy, "-m", RAM_CAP, "-n", UNIT, "-o", str(self.log), "--", "env", *env, *serve]
        return ["systemd-run", "--user", f"--unit={UNIT}", "--collect", "-p", "Nice=10",
                "-p", f"StandardOutput=append:{self.log}", "-p", f"StandardError=append:{self.log}",
                "--", "env", *env, *serve]

    def start(self) -> None:
        if self.running():
            return
        engine = ensure_engine()
        write_control(self.control, Ablation())
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a") as f:
            f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} starting {self.model}\n")
        _systemctl("reset-failed", UNIT)
        r = subprocess.run(self.command(engine), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Couldn't start the engine:\n{(r.stderr or r.stdout)[-2000:]}")

    def stop(self) -> None:
        _systemctl("stop", UNIT)

    def running(self) -> bool:
        return _systemctl("is-active", UNIT).stdout.strip() in ("active", "activating")

    def control_matches(self) -> bool:
        """False when the running engine reads another control file (started by another copy or
        run): the live switch would then silently do nothing."""
        for pid in _unit_pids():
            try:
                env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            except OSError:
                continue
            for item in env:
                if item.startswith(b"K3_ABL_CTL="):
                    return item[len(b"K3_ABL_CTL="):].decode() == str(self.control)
        return True

    def memory_gb(self) -> float | None:
        r = _systemctl("show", UNIT, "-p", "MemoryCurrent", "--value")
        try:
            return int(r.stdout.strip()) / 2**30
        except ValueError:
            return None

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                return r.status == 200
        except (OSError, urllib.error.URLError):
            return False


# ---------------------------------------------------------------- requests

@dataclass
class Reply:
    text: str = ""
    reasoning: str = ""
    tokens: int = 0
    seconds: float = 0.0
    finish: str = ""


_model_ids: dict[int, str] = {}


def model_id(port: int) -> str:
    """The id the gateway serves (it rejects any other)."""
    if port not in _model_ids:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
            _model_ids[port] = json.load(r)["data"][0]["id"]
    return _model_ids[port]


def chat(port: int, prompt: str, max_tokens: int, thinking: bool,
         on_text=lambda kind, chunk: None, cancelled=lambda: False) -> Reply:
    """One user turn, streamed. on_text("text" | "reasoning", chunk) as it arrives."""
    body = json.dumps({
        "model": model_id(port), "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "enable_thinking": thinking, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    reply, t0 = Reply(), time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"engine said {e.code}: {e.read().decode(errors='replace')[:500]}") from None
    with resp:
        for raw in resp:
            if cancelled():
                break           # closing the connection makes the gateway cancel the request
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"engine error: {event['error'].get('message', event['error'])}")
            if event.get("usage"):
                reply.tokens = event["usage"].get("completion_tokens", reply.tokens)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("reasoning_content"):
                    reply.reasoning += delta["reasoning_content"]
                    on_text("reasoning", delta["reasoning_content"])
                if delta.get("content"):
                    reply.text += delta["content"]
                    on_text("text", delta["content"])
                if choice.get("finish_reason"):
                    reply.finish = choice["finish_reason"]
    reply.seconds = time.time() - t0
    return reply
