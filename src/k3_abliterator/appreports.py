# Shared crash and problem reporter for the ssjrocks desktop (Python) apps.
#
# Source of truth: ~/dev/app-reports/python/appreports.py (github.com/ssjrocks/app-reports).
# Copy it into an app unchanged; don't edit the per-app copies.
#
# Same server and report format as android/AppReports.kt (schema 1):
#  - init(app_id, app_name, version, state_dir, log_file): installs hooks for uncaught exceptions
#    (main thread and other threads) and sends any queued reports in the background.
#  - report_error(exc, note=None): send a caught, non-fatal exception (once per run per place).
#  - send_report(note, extra, on_done): the "Report a problem" button; sends the recent log now.
#  - attach(name, provider): include extra text (e.g. another log) in every report.
#
# Reports are queued as JSON files in <state_dir>/app-reports/pending and posted gzipped to
# https://ssjrocks.com/api/reports/v1/report. Stdlib only.

import gzip
import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

LIB_VERSION = 1
ENDPOINT = "https://ssjrocks.com/api/reports/v1/report"
MAX_TEXT = 512 * 1024
MAX_PENDING = 20
PENDING_MAX_AGE = 7 * 24 * 3600
MAX_ERRORS_PER_RUN = 10

LOG = logging.getLogger("appreports")

_cfg = {}
_attachments = {}
_sent_errors = set()
_lock = threading.Lock()
_started = time.monotonic()


def init(app_id, app_name, version, state_dir, log_file=None, version_code=None):
    """Start reporting. app_id is a dotted id like io.github.ssjrocks.TalkpodManager."""
    _cfg.update(
        app=app_id,
        name=app_name,
        version=version,
        version_code=version_code,
        dir=Path(state_dir) / "app-reports",
        log_file=Path(log_file) if log_file else None,
    )
    (_cfg["dir"] / "pending").mkdir(parents=True, exist_ok=True)

    prev_hook = sys.excepthook

    def hook(etype, value, tb):
        if not issubclass(etype, KeyboardInterrupt):
            _capture("crash", value, threading.current_thread().name)
        prev_hook(etype, value, tb)

    sys.excepthook = hook

    prev_thread_hook = threading.excepthook

    def thread_hook(args):
        if args.exc_value is not None and not issubclass(args.exc_type, SystemExit):
            _capture("crash", args.exc_value, args.thread.name if args.thread else "?")
        prev_thread_hook(args)

    threading.excepthook = thread_hook
    threading.Thread(target=flush, name="app-reports", daemon=True).start()


def attach(name, provider):
    """provider() -> str or None, called when a report is built."""
    _attachments[name] = provider


def report_error(exc, note=None):
    """Send a caught exception. Repeats of the same place are sent once per run."""
    if not _cfg:
        return
    tb = traceback.extract_tb(exc.__traceback__)
    key = f"{type(exc).__qualname__}@{tb[-1].filename}:{tb[-1].lineno}" if tb else type(exc).__qualname__
    with _lock:
        if len(_sent_errors) >= MAX_ERRORS_PER_RUN or key in _sent_errors:
            return
        _sent_errors.add(key)
    r = _base("error")
    r["thread"] = threading.current_thread().name
    r["exception"] = _exception(exc)
    if note:
        r["note"] = note
    _save(r)
    threading.Thread(target=flush, name="app-reports", daemon=True).start()


def send_report(note=None, extra=None, on_done=None):
    """Send a problem report now (in the background). on_done(bool) runs on that thread:
    True once the server has it, False if it's queued for the next start."""
    if not _cfg:
        if on_done:
            on_done(False)
        return

    def run():
        ok = False
        try:
            r = _base("report")
            if note:
                r["note"] = note
            if extra:
                r["extra"] = extra
            path = _save(r)
            flush()
            ok = not path.exists()
        except Exception:
            LOG.exception("problem report failed")
        if on_done:
            on_done(ok)

    threading.Thread(target=run, name="app-reports", daemon=True).start()


# --- report building ---------------------------------------------------------------------------


def _capture(kind, exc, thread_name):
    try:
        r = _base(kind)
        r["thread"] = thread_name
        r["exception"] = _exception(exc)
        _save(r)
        # GUI toolkits often keep running after an uncaught exception in a callback; if the
        # process is exiting instead, the saved report goes out on the next start.
        threading.Thread(target=flush, name="app-reports", daemon=True).start()
    except Exception:
        pass  # never make a crash worse


def _exception(exc):
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    # Java-style "at" lines as well, so the server can group repeats of the same crash.
    frames = [f"\tat {Path(f.filename).stem}.{f.name}({Path(f.filename).name}:{f.lineno})"
              for f in reversed(traceback.extract_tb(exc.__traceback__))]
    stack = "".join(lines) + "\n" + f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}\n" + "\n".join(frames)
    return {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "stack": stack[:MAX_TEXT],
    }


def _read(path, limit=MAX_TEXT):
    try:
        data = Path(path).read_bytes()
        return data[-limit:].decode("utf-8", "replace")
    except OSError:
        return None


def _dmi(name):
    try:
        return Path(f"/sys/class/dmi/id/{name}").read_text().strip() or None
    except OSError:
        return None


def _os_release():
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    return platform.platform()


def _install_id():
    p = _cfg["dir"] / "install-id"
    try:
        return p.read_text().strip()
    except OSError:
        i = str(uuid.uuid4())
        p.write_text(i)
        return i


def _iso(t=None):
    return datetime.fromtimestamp(t or time.time(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _base(kind):
    r = {
        "schema": 1,
        "lib": LIB_VERSION,
        "app": _cfg["app"],
        "appName": _cfg["name"],
        "version": _cfg["version"],
        "versionCode": _cfg["version_code"],
        "debuggable": False,
        "kind": kind,
        "happenedAt": _iso(),
        "installId": _install_id(),
        "process": f"python {os.getpid()}",
        "uptimeSec": int(time.monotonic() - _started),
        "device": {
            "manufacturer": _dmi("sys_vendor") or "PC",
            "model": _dmi("product_name") or platform.machine(),
            "os": _os_release(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "arch": platform.machine(),
        },
    }
    if _cfg.get("log_file"):
        r["log"] = _read(_cfg["log_file"])
    if _attachments:
        a = {}
        for name, provider in list(_attachments.items()):
            try:
                text = provider()
            except Exception as e:
                text = f"(attachment failed: {e})"
            if text:
                a[name] = text[-MAX_TEXT:]
        if a:
            r["attachments"] = a
    return r


# --- queue and upload --------------------------------------------------------------------------


def _pending():
    return _cfg["dir"] / "pending"


def _save(report):
    d = _pending()
    path = d / f"{int(time.time() * 1000)}-{report['kind']}-{uuid.uuid4().hex[:8]}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report))
    tmp.replace(path)
    for old in sorted(d.glob("*.json"))[:-MAX_PENDING]:
        old.unlink(missing_ok=True)
    return path


def flush():
    """Send queued reports, oldest first; stop at the first network/server failure."""
    if not _cfg:
        return
    with _lock:
        for f in sorted(_pending().glob("*.json")):
            try:
                age = time.time() - int(f.name.split("-", 1)[0]) / 1000
            except ValueError:
                age = 0
            if age > PENDING_MAX_AGE:
                f.unlink(missing_ok=True)
                continue
            try:
                code = _post(f.read_bytes())
            except Exception as e:
                LOG.info("report upload failed, will retry: %s", e)
                return
            if 200 <= code < 300:
                f.unlink(missing_ok=True)
            elif code in (408, 429) or code >= 500:
                return
            else:
                LOG.warning("report rejected (%s), dropping %s", code, f.name)
                f.unlink(missing_ok=True)


def _post(body):
    req = urllib.request.Request(
        ENDPOINT,
        data=gzip.compress(body),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "User-Agent": f"AppReports/{LIB_VERSION} {_cfg['app']}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
