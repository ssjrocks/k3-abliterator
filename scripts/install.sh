#!/usr/bin/env bash
# Installs K3 Abliterator for the current user: a Python venv in
# ~/.local/share/k3-abliterator/venv, the desktop entry and the icon.
#
#   From a checkout:  scripts/install.sh            installs that checkout
#   Anywhere:         curl -fsSL https://raw.githubusercontent.com/ssjrocks/k3-abliterator/main/scripts/install.sh | bash
#   Remove:           scripts/install.sh --uninstall   (keeps runs and captures in ~/.local/share/k3-abliterator/runs)
set -euo pipefail

APP_ID=io.github.ssjrocks.K3Abliterator
REPO=ssjrocks/k3-abliterator
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
ROOT="$DATA/k3-abliterator"
VENV="$ROOT/venv"
DESKTOP="$DATA/applications/$APP_ID.desktop"
ICON="$DATA/icons/hicolor/scalable/apps/$APP_ID.svg"

if [[ "${1:-}" == "--uninstall" ]]; then
  rm -rf "$VENV" "$ROOT/engine"
  rm -f "$DESKTOP" "$ICON"
  update-desktop-database -q "$DATA/applications" 2>/dev/null || true
  echo "K3 Abliterator removed (runs kept in $ROOT/runs)."
  exit 0
fi

command -v uv >/dev/null || { echo "uv not found: see https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
[[ -x "${COLIBRI_DIR:-$HOME/colibri}/c/coli" ]] || echo "note: no colibri at ${COLIBRI_DIR:-$HOME/colibri}; the app needs it to run the model" >&2

src=""
if [[ -f "${BASH_SOURCE[0]:-}" ]]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  [[ -f "$here/../pyproject.toml" ]] && src="$(cd "$here/.." && pwd)"
fi

uv venv -q --clear --seed --python 3.12 "$VENV"
touch "$VENV/k3-abliterator-venv"   # tells the in-app updater it may update this venv

if [[ -n "$src" ]]; then
  echo "Installing from $src"
  "$VENV/bin/pip" install -q "$src"
else
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  read -r wheel_url sha < <("$VENV/bin/python" - "$REPO" <<'EOF'
import json, sys, urllib.request
repo = sys.argv[1]
rel = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{repo}/releases/latest"))
assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
info = json.load(urllib.request.urlopen(assets["k3-abliterator.json"]))
print(assets[info["wheel"]], info["sha256"])
EOF
)
  echo "Downloading $wheel_url"
  curl -fsSL -o "$tmp/$(basename "$wheel_url")" "$wheel_url"
  echo "$sha  $tmp/$(basename "$wheel_url")" | sha256sum -c --quiet
  "$VENV/bin/pip" install -q "$tmp/$(basename "$wheel_url")"
fi

mkdir -p "$(dirname "$DESKTOP")" "$(dirname "$ICON")"
"$VENV/bin/python" - "$ICON" <<'EOF'
import sys
from importlib.resources import files
open(sys.argv[1], "wb").write((files("k3_abliterator") / "data/io.github.ssjrocks.K3Abliterator.svg").read_bytes())
EOF
cat > "$DESKTOP" <<EOF
[Desktop Entry]
Version=1.1
Type=Application
Name=K3 Abliterator
Comment=Remove the refusals a Kimi K3 model still makes
Exec=$VENV/bin/python -m k3_abliterator
Icon=$APP_ID
Terminal=false
Categories=Development;Science;
StartupWMClass=$APP_ID
EOF
update-desktop-database -q "$DATA/applications" 2>/dev/null || true
gtk-update-icon-cache -q "$DATA/icons/hicolor" 2>/dev/null || true
echo "Installed. Open K3 Abliterator from the app menu."
