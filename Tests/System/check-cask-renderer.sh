#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/lalia-cask-check.XXXXXX")
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT
cask="$tmp/Casks/lalia.rb"
sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

if Support/render-homebrew-cask.rb v1.2.3 "$sha" "$cask" >/dev/null 2>&1; then
  echo 'renderer accepted a prefixed version' >&2
  exit 1
fi
if Support/render-homebrew-cask.rb 1.2.3 short "$cask" >/dev/null 2>&1; then
  echo 'renderer accepted an invalid checksum' >&2
  exit 1
fi

Support/render-homebrew-cask.rb 1.2.3 "$sha" "$cask"
ruby -c "$cask" | rg -Fqx 'Syntax OK'
rg -Fqx 'cask "lalia" do' "$cask"
rg -Fqx '  version "1.2.3"' "$cask"
rg -Fqx "  sha256 \"$sha\"" "$cask"
rg -Fqx '  url "https://github.com/cosgroveb/lalia/releases/download/v#{version}/Lalia-#{version}.dmg"' "$cask"
rg -Fqx '  desc "Menu-bar voice dictation"' "$cask"
rg -Fqx '  depends_on arch: :arm64' "$cask"
rg -Fqx '  depends_on macos: :tahoe' "$cask"
rg -Fqx '  app "Lalia.app"' "$cask"
