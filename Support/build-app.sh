#!/usr/bin/env bash
set -euo pipefail
export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${TMPDIR:-/tmp}/lalia-clang-module-cache}"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

version=${LALIA_VERSION:-0.0.0}
identity=${LALIA_CODESIGN_IDENTITY:--}
distribution=${LALIA_DISTRIBUTION:-0}
semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'

if [[ ! $version =~ $semver_re ]]; then
  echo "invalid Lalia version: $version" >&2
  exit 2
fi
case "$distribution" in
  0|1) ;;
  *) echo "LALIA_DISTRIBUTION must be 0 or 1" >&2; exit 2 ;;
esac

swift build --disable-sandbox -c release
app=dist/Lalia.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp Support/Info.plist "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $version" "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $version" "$app/Contents/Info.plist"
cp .build/release/Lalia "$app/Contents/MacOS/"

timestamp=(--timestamp)
if [[ $identity == - ]]; then
  timestamp=(--timestamp=none)
fi

if [[ $distribution -eq 0 ]]; then
  cp .build/release/LaliaSpeechSmoke "$app/Contents/MacOS/"
  codesign --force --sign "$identity" "${timestamp[@]}" "$app/Contents/MacOS/LaliaSpeechSmoke"
  codesign --force --sign "$identity" "${timestamp[@]}" "$app"
else
  codesign --force --sign "$identity" "${timestamp[@]}" --options runtime \
    --entitlements Support/Lalia.entitlements "$app"
fi
