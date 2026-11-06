"""K3 Abliterator: start logging and problem reports, launch the window."""

import logging
import logging.handlers
import sys
import traceback

from k3_abliterator import APP_ID, APP_NAME, __version__, appreports, paths


def _log_exception(exc_type, exc, tb) -> None:
    # Without a hook, PyQt6 aborts the process on an exception in a slot.
    traceback.print_exception(exc_type, exc, tb)
    logging.getLogger("k3_abliterator").error("uncaught", exc_info=(exc_type, exc, tb))


def _engine_log_tail():
    try:
        with open(paths.engine_log(), "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 30_000))
            lines = f.read().decode(errors="replace").splitlines()[1:]
        # The engine's own lines ("[K3] ...", "[ABL] ...", timings), never request bodies.
        return "\n".join(ln for ln in lines if ln.startswith("[") or "error" in ln.lower())[-12_000:]
    except OSError:
        return None


def main():
    log_file = paths.log_file()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.handlers.RotatingFileHandler(log_file, maxBytes=512 * 1024, backupCount=1)],
    )
    logging.info("%s %s starting", APP_NAME, __version__)
    sys.excepthook = _log_exception
    appreports.init(APP_ID, APP_NAME, __version__, paths.state_dir(), log_file=log_file)
    appreports.attach("engine.log", _engine_log_tail)
    from k3_abliterator.gui import run_gui

    run_gui()


if __name__ == "__main__":
    main()
