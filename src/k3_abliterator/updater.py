"""Updates from the project's GitHub releases.

Each release carries the wheel and k3-abliterator.json ({"version", "wheel",
"sha256"}). The wheel must come from github.com, match the checksum, and is
installed with pip into the venv scripts/install.sh made. Running from a source
checkout, the app only reports that an update exists.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from k3_abliterator import REPO, __version__, paths

LOG = logging.getLogger(__name__)

API = f"https://api.github.com/repos/{REPO}/releases/latest"
INFO_ASSET = "k3-abliterator.json"
CHECK_EVERY = 12 * 3600
USER_AGENT = f"k3-abliterator/{__version__}"


@dataclass
class Update:
    version: str
    wheel_url: str
    wheel_name: str
    sha256: str


def parse_version(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.lstrip("v").split("."):
        digits = "".join(c for c in part if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read()


def _trusted(url: str) -> bool:
    return url.startswith(f"https://github.com/{REPO}/releases/download/")


def check() -> Update | None:
    """The latest release if it's newer than this version, else None."""
    release = json.loads(_get(API))
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    info_url = assets.get(INFO_ASSET)
    if not info_url or not _trusted(info_url):
        return None
    info = json.loads(_get(info_url))
    if parse_version(info["version"]) <= parse_version(__version__):
        return None
    wheel_url = assets.get(info["wheel"])
    if not wheel_url or not _trusted(wheel_url):
        return None
    return Update(info["version"], wheel_url, info["wheel"], info["sha256"].lower())


def _state_dir() -> Path:
    return paths.state_dir()


def due() -> bool:
    try:
        last = json.loads((_state_dir() / "update.json").read_text())["checked_at"]
    except (OSError, ValueError, KeyError):
        last = 0
    return time.time() - last > CHECK_EVERY


def mark_checked() -> None:
    (_state_dir() / "update.json").write_text(json.dumps({"checked_at": time.time()}))


def installed_in_venv() -> bool:
    """True when running from the venv scripts/install.sh made."""
    return (Path(sys.prefix) / "k3-abliterator-venv").exists()


def install(update: Update) -> None:
    """Download, verify and pip-install the update. Raises on any failure."""
    if not installed_in_venv():
        raise RuntimeError(
            f"This copy runs from source. Update it with git, or get "
            f"{update.version} from https://github.com/{REPO}/releases"
        )
    folder = _state_dir() / "updates"
    folder.mkdir(exist_ok=True)
    path = folder / update.wheel_name
    data = _get(update.wheel_url)
    if hashlib.sha256(data).hexdigest() != update.sha256:
        raise RuntimeError("The download didn't match its checksum; nothing was installed.")
    path.write_bytes(data)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", str(path)],
        capture_output=True,
        text=True,
    )
    path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"pip couldn't install the update:\n{result.stderr[-2000:]}")
    LOG.info("installed update %s", update.version)


def restart() -> None:
    os.execv(sys.executable, [sys.executable, "-m", "k3_abliterator"])
