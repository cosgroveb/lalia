# Lalia

Native macOS menu-bar voice dictation. Hold Command+Shift+D, speak, then
release to paste an on-device Speech transcript into the focused app.

Requires Apple silicon, macOS 26 or newer, and microphone, Speech Recognition,
and Accessibility permission.

## Install

```sh
brew install --cask cosgroveb/tap/lalia
open -a Lalia
```

Open the menu-bar item and choose `Enable Dictation` on first launch. macOS
prompts for the three required permissions and downloads compatible Speech
assets when needed.

## Use

Focus a text field, hold Command+Shift+D, speak, and release. Lalia records
while the shortcut is held, transcribes after release, and pastes one result.

`Copy Last Transcript` remains available from the menu-bar item when paste
injection fails. Clipboard restoration covers eagerly readable pasteboard
data; macOS does not expose every lazy or promised representation.

Lalia has no backend, account, history, settings screen, microphone chooser,
or configurable shortcut.

## Development

Build and launch the development app:

```sh
Support/build-app.sh
open dist/Lalia.app
```

Run deterministic tests and the signed bundle/Speech probe:

```sh
swift test
shellcheck Support/*.sh Tests/System/*.sh
Tests/System/check-app.sh
Tests/System/check-release-package.sh
Tests/System/check-cask-renderer.sh
```

The Speech probe may skip only when its documented hardware, permission, or
asset prerequisites are unavailable.
