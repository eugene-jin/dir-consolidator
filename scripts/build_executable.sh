#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${VENV_BIN:-$ROOT_DIR/.venv/bin}"
PYINSTALLER="$VENV_BIN/pyinstaller"
APP_ENTRY="$ROOT_DIR/consolidator/app.py"
DIST_ROOT="$ROOT_DIR/dist"
BUILD_ROOT="$ROOT_DIR/build/pyinstaller"
ARCH="${ARCH:-$(uname -m)}"
DIST_DIR="$DIST_ROOT/macos-$ARCH"
BUILD_DIR="$BUILD_ROOT/$ARCH"

if [[ ! -x "$PYINSTALLER" ]]; then
  echo "PyInstaller not found at $PYINSTALLER. Activate .venv and run: pip install pyinstaller" >&2
  exit 1
fi

rm -rf "$DIST_DIR" "$BUILD_DIR"
mkdir -p "$DIST_DIR"

PYTHONPATH="$ROOT_DIR" "$PYINSTALLER" \
  --clean \
  --noconfirm \
  --onefile \
  --name dir-consolidator \
  --distpath "$DIST_DIR" \
  --workpath "$BUILD_DIR" \
  "$APP_ENTRY"

echo "Executable created: $DIST_DIR/dir-consolidator"

if [[ "${ARCH}" == "arm64" && -f "$DIST_ROOT/macos-x86_64/dir-consolidator" ]]; then
  UNIVERSAL_PATH="$DIST_ROOT/dir-consolidator-universal"
  lipo -create \
    "$DIST_ROOT/macos-arm64/dir-consolidator" \
    "$DIST_ROOT/macos-x86_64/dir-consolidator" \
    -output "$UNIVERSAL_PATH"
  echo "Universal binary refreshed: $UNIVERSAL_PATH"
fi
