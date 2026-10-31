# Lalia

Lalia records dictation and sessions from the macOS menu bar and transcribes
them with local MOSS.
Requires Apple silicon and macOS 26 or newer.

## Install

```sh
brew install --cask cosgroveb/tap/lalia
lalia-runtime setup
open -a Lalia
```

Setup downloads the transcription model, starts its isolated runtime, and
configures Lalia. Use `Check Runtime Setup` in the menu if transcription fails.

## Use

Choose `Enable Dictation` and `Enable Gestures` in the menu, then grant the
requested macOS permissions:

| Permission | Used for |
| --- | --- |
| Microphone | Dictation and session recording |
| Input Monitoring | Right Option gestures |
| Accessibility | Pasting dictation |
| Screen & System Audio Recording | Recording Mac audio |

- Dictate: focus a text field, hold Right Option, speak after Recording appears,
  then release to paste. Use `Copy Last Transcript` if pasting fails.
- Record a session: double-tap Right Option to start recording microphone and Mac
  audio. Double-tap again to save and transcribe. The menu has the same controls.
  Sessions stop at 60 minutes. Use headphones to reduce playback leaking into
  the microphone.

Speaker labels are experimental. MOSS can omit speech or invent text and speaker
labels. Review transcripts against the saved audio.

Open `Recording History` to play audio, copy or export transcripts, correct
speaker labels, or retry failed transcription. You can record while earlier
audio transcribes. Queued dictation saves text without pasting, so copy it from
History. Retranscription preserves the current transcript until a replacement
succeeds, with `Restore Previous Transcript` available afterward.

Lalia saves audio under `~/Library/Application Support/Lalia/Sessions/` until
you delete it. `Reveal Recordings` opens a recording's directory. Sleep or
closing the lid can interrupt capture. After wake, choose `Enable Gestures` to
restore the listener.

## Configuration

Lalia reads `~/.config/lalia/config.json`. New installations use:

```json
{
  "baseURL": "http://127.0.0.1:8002/v1",
  "summaryBaseURL": "http://127.0.0.1:8000/v1",
  "transcriptionModel": "MOSS-Transcribe-Diarize",
  "language": "en",
  "summaryModel": "Qwen3.6-35B-A3B-oQ8-mtp"
}
```

Both endpoints must use loopback addresses. Setup preserves custom endpoints
and summary settings. `Generate Notes` uses a separate local chat server, which
you must provide. Setup does not install a notes model. Speaker corrections
mark existing notes stale until you generate them again.

## Development

Build the app with Swift 6.2 or newer:

```sh
Support/build-app.sh
open dist/Lalia.app
```

Run tests and bundle checks without recording or live inference:

```sh
swift test
shellcheck Support/*.sh Tests/System/*.sh
Tests/System/check-app.sh
Tests/System/check-release-package.sh
Tests/System/check-cask-renderer.sh
```

These checks do not establish capture permissions or transcription accuracy.
To test live inference against your configured runtime:

```sh
dist/Lalia.app/Contents/MacOS/Lalia --transcribe-file /path/to/speech.wav
dist/Lalia.app/Contents/MacOS/Lalia --transcribe-session-file /path/to/meeting.wav
```
