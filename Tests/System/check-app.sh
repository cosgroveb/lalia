#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
Support/build-app.sh
app=dist/Lalia.app
plist="$app/Contents/Info.plist"
for pair in \
  'CFBundleIdentifier com.bcosgrove.Lalia' \
  'CFBundleExecutable Lalia' \
  'CFBundlePackageType APPL' \
  'CFBundleShortVersionString 0.0.0' \
  'CFBundleVersion 0.0.0' \
  'LSMinimumSystemVersion 26.0'; do
  key=${pair%% *}; expected=${pair#* }
  actual=$(/usr/libexec/PlistBuddy -c "Print :$key" "$plist")
  [[ $actual == "$expected" ]]
done
/usr/libexec/PlistBuddy -c 'Print :LSUIElement' "$plist" | rg -qx 'true|1'
/usr/libexec/PlistBuddy -c 'Print :NSMicrophoneUsageDescription' "$plist" >/dev/null
/usr/libexec/PlistBuddy -c 'Print :NSAudioCaptureUsageDescription' "$plist" >/dev/null
/usr/libexec/PlistBuddy -c 'Print :NSAppTransportSecurity:NSAllowsLocalNetworking' "$plist" | rg -qx 'true|1'
if /usr/libexec/PlistBuddy -c 'Print :NSSpeechRecognitionUsageDescription' "$plist" >/dev/null 2>&1; then
  echo 'obsolete Speech permission remains' >&2
  exit 1
fi
codesign --verify --deep --strict "$app"
python3 Tests/System/check-http.py "$app/Contents/MacOS/Lalia"
if [[ ${LALIA_LIVE_ASR:-0} == 1 ]]; then
  : "${LALIA_ASR_FIXTURE:?set LALIA_ASR_FIXTURE to a speech recording for the live check}"
  "$app/Contents/MacOS/Lalia" --transcribe-file "$LALIA_ASR_FIXTURE"
else
  echo 'Live ASR not run. Set LALIA_LIVE_ASR=1 and LALIA_ASR_FIXTURE to request it.'
fi
