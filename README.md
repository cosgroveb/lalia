# Lalia

Native macOS 26 menu-bar voice dictation. Hold Command+Shift+D, speak, then
release to paste a local Speech transcript into the focused app.

Build and launch only through the bundled app path:

```sh
Support/build-app.sh
open dist/Lalia.app
```

Requires macOS 26+, microphone, Speech Recognition, and Accessibility grants,
plus installed compatible Speech assets. No backend, worker, settings, hotkey
configuration, microphone chooser, or toggle mode is provided.

Manual acceptance: grant permissions; focus a text editor; hold and release
Command+Shift+D while speaking; confirm one paste and clipboard restoration
after 150 ms. Deny each permission and confirm the focused app is unchanged.
Press during recording/transcription must not start another dictation.

Clipboard restoration includes only eagerly readable pasteboard data; lazy or
promised representations are a macOS limitation.
