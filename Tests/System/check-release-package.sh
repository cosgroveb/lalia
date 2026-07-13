#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

if LALIA_VERSION=banana Support/build-app.sh >/dev/null 2>&1; then
  echo 'build-app accepted an invalid version' >&2
  exit 1
fi

LALIA_CODESIGN_IDENTITY=- Support/build-dmg.sh 1.2.3 >/dev/null
dmg=dist/Lalia-1.2.3.dmg
[[ -f $dmg ]]
hdiutil verify "$dmg" >/dev/null
codesign --verify "$dmg"

mount=$(mktemp -d "${TMPDIR:-/tmp}/lalia-release-check.XXXXXX")
mounted=0
cleanup() {
  if [[ $mounted -eq 1 ]]; then
    hdiutil detach "$mount" >/dev/null
  fi
  rmdir "$mount"
}
trap cleanup EXIT
hdiutil attach -readonly -nobrowse -mountpoint "$mount" "$dmg" >/dev/null
mounted=1

app="$mount/Lalia.app"
plist="$app/Contents/Info.plist"
[[ -d $app ]]
[[ -L $mount/Applications ]]
[[ ! -e $app/Contents/MacOS/LaliaSpeechSmoke ]]
[[ $(/usr/libexec/PlistBuddy -c 'Print :CFBundlePackageType' "$plist") == APPL ]]
[[ $(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$plist") == 1.2.3 ]]
[[ $(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$plist") == 1.2.3 ]]
codesign --verify --deep --strict "$app"

entitlements=$(mktemp "${TMPDIR:-/tmp}/lalia-entitlements.XXXXXX.plist")
codesign --display --entitlements :- "$app" >"$entitlements" 2>/dev/null
/usr/libexec/PlistBuddy -c 'Print :com.apple.security.device.audio-input' "$entitlements" | rg -qx 'true|1'
rm "$entitlements"

hdiutil detach "$mount" >/dev/null
mounted=0
rmdir "$mount"
trap - EXIT
