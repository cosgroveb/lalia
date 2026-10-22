# Lalia

Native macOS menu-bar dictation and session recording. Hold Command+Shift+D
and release to paste a local oMLX/Cohere transcript. Start a session to save
separate microphone and Mac audio tracks, then transcribe them after Stop.

Requires Apple silicon and macOS 26 or newer. Dictation needs Microphone and
Accessibility permission. Sessions need Microphone and Screen & System Audio
Recording permission. Accessibility is only for pasting.

## Install and configure

The published cask may predate session support. Build this revision locally
until a release includes these changes:

```sh
Support/build-app.sh
open dist/Lalia.app
```

Run oMLX locally with `cohere-transcribe-03-2026` installed and authorized
through Hugging Face. Lalia reads `~/.config/lalia/config.json`. A missing file
uses these defaults, while malformed settings produce an error:

```json
{
  "baseURL": "http://127.0.0.1:8000/v1",
  "transcriptionModel": "cohere-transcribe-03-2026",
  "language": "en",
  "summaryModel": "Qwen3.6-35B-A3B-oQ8-mtp"
}
```

Only loopback endpoints are supported. Lalia disables redirects and system
proxies for its HTTP client. Recording works while oMLX is offline. Retry
transcription from history once the server and model are ready.

## Use

Choose `Enable Dictation`, focus a text field, hold Command+Shift+D, speak,
and release. `Copy Last Transcript` remains available if pasting fails.
Clipboard restoration covers eagerly readable pasteboard data. macOS does
not expose every lazy or promised representation.

Choose `Start Session` to record microphone and Mac audio, then `Stop Session`
to save and transcribe. Lalia excludes its own playback. Use headphones to
reduce Mac playback leaking into the microphone.

`Recording History` shows saved audio, status, errors, and selectable text.
`Retry Transcription` processes unfinished jobs and preserves completed
responses. `Export TXT` exports source-labelled text with approximate times.
`Generate Notes` makes a separate Qwen request from a complete transcript.
Notes do not replace the transcript.

Lalia stores recordings and atomic manifests under
`~/Library/Application Support/Lalia/Sessions/`. Directories have mode 0700
and files 0600. `Reveal Recordings` opens the session directory. Audio remains
until you remove it. Failed dictation and cancelled processing retain audio.
Quit closes active recordings before exiting. On reopening, Lalia attempts to
recover unfinished audio and reports files it cannot recover.

Sessions use lossless native PCM CAF files, rotating every minute. Dictation
uses 16 kHz mono PCM WAV. Transcription derives bounded 16 kHz mono WAV jobs
of at most ten minutes, split near low-energy audio at the boundary. Lalia
skips digital silence and preserves quiet audio. Ambient noise can still
produce hallucinated text. Times describe approximate source ranges, not
words or named speakers. Recovery does not guarantee the final partial file
survives power loss.

## Development

Run deterministic tests and bundle checks without recording or live inference:

```sh
swift test
shellcheck Support/*.sh Tests/System/*.sh
Tests/System/check-app.sh
Tests/System/check-release-package.sh
Tests/System/check-cask-renderer.sh
```

The app check runs a local HTTP fixture against the signed executable. It
verifies multipart uploads, ten-minute limits, and redirect rejection. It does
not measure transcription accuracy or establish macOS capture permissions.

Run live inference on an existing recording with the same client and chunk
preparation as the app:

```sh
dist/Lalia.app/Contents/MacOS/Lalia --transcribe-file /path/to/speech.wav
```

The probe accepts `--config /path/to/config.json` after the audio path. Its
derived files are temporary, and it leaves the input recording intact. To
include live inference in the app check, set `LALIA_LIVE_ASR=1` and
`LALIA_ASR_FIXTURE=/path/to/speech.wav`. A requested live check fails if the
server, model, or inference is unavailable. Signed microphone/system capture
and real ASR quality need separate acceptance checks.
