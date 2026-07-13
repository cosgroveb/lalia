#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'usage: Support/build-dmg.sh VERSION' >&2
  exit 2
fi

version=$1
semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
if [[ ! $version =~ $semver_re ]]; then
  echo "invalid Lalia version: $version" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
identity=${LALIA_CODESIGN_IDENTITY:--}
LALIA_VERSION="$version" LALIA_CODESIGN_IDENTITY="$identity" \
  LALIA_DISTRIBUTION=1 Support/build-app.sh

staging=$(mktemp -d "${TMPDIR:-/tmp}/lalia-dmg.XXXXXX")
cleanup() { rm -rf "$staging"; }
trap cleanup EXIT
cp -R dist/Lalia.app "$staging/"
ln -s /Applications "$staging/Applications"

dmg="dist/Lalia-$version.dmg"
if [[ -e $dmg ]]; then
  rm "$dmg"
fi
hdiutil create -volname Lalia -srcfolder "$staging" -format UDZO "$dmg" >/dev/null

timestamp=(--timestamp)
if [[ $identity == - ]]; then
  timestamp=(--timestamp=none)
fi
codesign --force --sign "$identity" "${timestamp[@]}" "$dmg"
hdiutil verify "$dmg" >/dev/null
printf '%s\n' "$dmg"
