#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
Support/build-app.sh
app=dist/Lalia.app
plist="$app/Contents/Info.plist"
for pair in 'CFBundleIdentifier com.bcosgrove.Lalia' 'CFBundleExecutable Lalia' 'LSMinimumSystemVersion 26.0'; do
  key=${pair%% *}; expected=${pair#* }
  actual=$(/usr/libexec/PlistBuddy -c "Print :$key" "$plist")
  [[ $actual == "$expected" ]]
done
/usr/libexec/PlistBuddy -c 'Print :LSUIElement' "$plist" | rg -qx 'true|1'
/usr/libexec/PlistBuddy -c 'Print :NSMicrophoneUsageDescription' "$plist" >/dev/null
/usr/libexec/PlistBuddy -c 'Print :NSSpeechRecognitionUsageDescription' "$plist" >/dev/null
codesign --verify --deep --strict "$app"
fixture=$(mktemp /tmp/lalia-speech-fixture.XXXXXX.wav)
trap 'rm -f "$fixture"' EXIT
/usr/bin/say -o "${fixture%.wav}.aiff" 'Lalia speech check'
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 "${fixture%.wav}.aiff" -o "$fixture"
rm -f "${fixture%.wav}.aiff"
set +e
"$app/Contents/MacOS/LaliaSpeechSmoke" "$fixture"
status=$?
set -e
if [[ $status -eq 77 ]]; then
  echo 'SKIP: real Speech prerequisites unavailable'
  exit 0
fi
exit "$status"
