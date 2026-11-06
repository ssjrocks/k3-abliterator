#!/usr/bin/env bash
# Build the Kimi K3 engine with directional ablation, out of colibri's tree.
# The stock ~/colibri/c/kimi_k3 is never touched: `coli serve` runs this one
# through COLI_ENGINE. Uses the flags colibri's own build recorded.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
COLI="${COLIBRI_DIR:-$HOME/colibri}/c"
OUT="${1:?usage: build.sh OUTPUT}"
WORK="$(dirname "$OUT")/engine-src"

[ -f "$COLI/kimi_k3.c" ] || { echo "no $COLI/kimi_k3.c (set COLIBRI_DIR)" >&2; exit 1; }
[ -f "$COLI/.build-config" ] || { echo "build colibri once first (make -C $COLI kimi_k3)" >&2; exit 1; }
IFS='|' read -r CC CFLAGS LDFLAGS REST < "$COLI/.build-config"
CUDA_OBJ=""
case "$CFLAGS" in *-DCOLI_CUDA*) CUDA_OBJ="$COLI/backend_cuda.o";; esac
case "$CFLAGS" in *-DCOLI_VULKAN*) echo "Vulkan builds aren't supported (the dense chain bypasses the hook)" >&2; exit 1;; esac

mkdir -p "$WORK"
cp "$COLI/kimi_k3.c" "$WORK/kimi_k3.c"
if ! patch -s -d "$WORK" -p0 --no-backup-if-mismatch kimi_k3.c < "$HERE/kimi_k3.patch"; then
    echo "kimi_k3.patch no longer applies to this colibri version; the hooks need re-placing" >&2
    exit 1
fi
CFLAGS="${CFLAGS//-MMD/}"; CFLAGS="${CFLAGS//-MP/}"
# shellcheck disable=SC2086
nice -n 10 "$CC" $CFLAGS -I"$HERE" -I"$COLI" "$WORK/kimi_k3.c" $CUDA_OBJ -o "$OUT" $LDFLAGS
echo "built $OUT (colibri $(git -C "$COLI/.." rev-parse --short HEAD 2>/dev/null || echo '?'))"
