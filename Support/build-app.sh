#!/usr/bin/env bash
set -euo pipefail
export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${TMPDIR:-/tmp}/lalia-clang-module-cache}"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
swift build --disable-sandbox -c release
app=dist/Lalia.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp Support/Info.plist "$app/Contents/Info.plist"
cp .build/release/Lalia .build/release/LaliaSpeechSmoke "$app/Contents/MacOS/"
codesign --force --deep --sign - --timestamp=none "$app"
