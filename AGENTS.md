# Lalia

- Install with `brew install --cask cosgroveb/tap/lalia`. The tap repository is
  `cosgroveb/homebrew-tap`. Reuse it rather than creating another tap.
- For packaging changes, validate the rendered formula and cask inside a
  temporary tap using `.github/workflows/release.yml` as the command reference.
  Preserve its tool, trust, and authentication prerequisites. Ruby syntax and
  renderer tests do not establish Homebrew acceptance. Complete applicable
  pre-release checks before tagging.
- Scope `LALIA_CODESIGN_IDENTITY` to the distribution build after certificate
  import. Pre-import packaging tests need the default ad-hoc signing.
- `Tests/System/check-app.sh` checks packaging and HTTP behavior. Its opt-in
  ASR check transcribes a file. Verify gestures, microphone capture, and paste
  injection through the running app before claiming those paths work. A probe
  in another process does not establish the app's event-delivery behavior.
- In pasteboard tests, compare restoration against the representations AppKit
  exposes after writing the fixture. AppKit can add representations beyond
  those requested. Follow `Tests/LaliaTests/PasteInjectorTests.swift`.
