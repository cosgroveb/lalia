# Lalia

Native macOS menu-bar dictation and session recording. Hold Command+Shift+D
and release to paste a local oMLX/Cohere transcript. Press Command+Shift+R to
toggle recording of microphone and Mac audio, then add speaker labels through
local MOSS after Stop. Recordings stop after 90 minutes.

Speaker-label support on this branch is experimental and has not been deployed.
Two 60-minute tests with MLX's English default prompt ended after 16,384
generated tokens, with the last parsed speech at 33m29s. This matches
[MOSS's open early-stop defect](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/issues/26).
One diagnostic with the exact official instruction completed the same hour
with consistent speaker labels. The actual API path then passed a 17-minute
baseline but failed at 90 minutes: native EOS after 17,127 tokens, with the
last parsed speech at 36m49s. Deployment is on hold. Neither the successful
hour-long diagnostic nor response validation establishes reliable long audio.
The 90-minute limit bounds capture, not verified transcription capability.
Keep the working installation for regular meetings until long recordings pass
qualification. Real-device and coding-model contention checks remain deferred.

Requires Apple silicon and macOS 26 or newer. Dictation needs Microphone and
Accessibility permission. Sessions need Microphone and Screen & System Audio
Recording permission. Accessibility is only for pasting.

## Install and configure

Build this experimental revision for development checks:

```sh
Support/build-app.sh
open dist/Lalia.app
```

Run oMLX locally with `cohere-transcribe-03-2026` for dictation and
`MOSS-Transcribe-Diarize` for recordings. Cohere requires Hugging Face access.
MOSS is public. Session analysis requires the pinned oMLX 0.7.2 / mlx-audio
0.5.7 runtime with normal JSON token usage, explicit prompt forwarding and a
250 MB audio upload limit.
Brian's `omlx-install-transcription` installs this compatibility patch and the
pinned MOSS model. The installer refuses unknown runtime sources. Lalia rejects
missing or inconsistent usage. New runtimes still require qualification.

Lalia reads `~/.config/lalia/config.json`. A missing file uses these defaults.
Older configurations can omit `sessionTranscriptionModel`. Malformed settings
produce an error:

```json
{
  "baseURL": "http://127.0.0.1:8000/v1",
  "transcriptionModel": "cohere-transcribe-03-2026",
  "sessionTranscriptionModel": "MOSS-Transcribe-Diarize",
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

Press Command+Shift+R once to start recording microphone and Mac audio, then
again to stop, save, and transcribe. The menu and history window offer the same
`Start Recording` / `Stop Recording` button. Recording continues when you
release the shortcut. Lalia shows elapsed and remaining time, warns with five
minutes remaining, then saves and transcribes at 90 minutes. It does not start
another recording. Holding the shortcut does not toggle repeatedly. Lalia
ignores the toggle while starting, saving, or dictating. Use
headphones to reduce Mac playback leaking into the microphone.

You can start another recording or dictate while earlier audio is transcribing.
Lalia processes transcription and notes requests one at a time and shows each
recording's queued or running status in History. Cancel an individual request
there, or use `Cancel Current Processing` in the menu. Cancellation leaves an
active recording running. Quitting saves active capture and cancels queued work,
which you can retry from History after reopening.

Dictation queued behind another request saves its text without pasting or
changing the clipboard. Use `Copy Transcript` in History or `Copy Last Transcript`
in the menu. Immediate dictation still pastes when it finishes, unless you have
started another capture in the meantime.

Lalia keeps the Mac and display awake during session recording because display
sleep interrupts ScreenCaptureKit audio capture. Stopping, capture failure, or
quitting releases that hold. Explicit sleep or closing the lid can still
interrupt recording. Saved audio remains available for retry. Session analysis
keeps the system awake until it completes, fails, or is cancelled.

`Recording History` shows saved audio, status, errors, and selectable text.
Speaker names apply throughout one recording. Open `Speakers` to rename or
merge labels. Use a passage's speaker menu to reassign it, mark it unassigned,
or create a speaker. `Play from Here` plays the saved mix from that passage.
Names do not carry between recordings. Sound events remain separate from
dialogue, and model labels can be wrong.

`Export TXT` and `Generate Notes` use the corrected transcript. Notes use Qwen
and remain separate. Later corrections mark existing notes stale until you
generate them again. `Add Speaker Labels` analyzes older recordings within the
limit. Longer recordings and their existing transcripts remain readable.

`Retry Transcription` retries a failed analysis with its saved request settings.
`Reprocess Speaker Labels` starts a replacement with current settings. Lalia
keeps the current transcript and corrections during either operation. An
accepted replacement keeps the previous result for `Restore Previous
Transcript`. Older source-labelled transcripts remain available until a
speaker result succeeds. Cancelling abandons the result, but oMLX may continue
processing. An explicit retry can wait behind that request.

Lalia stores recordings and atomic manifests under
`~/Library/Application Support/Lalia/Sessions/`. Directories have mode 0700
and files 0600. `Reveal Recordings` opens the session directory. Audio remains
until you remove it. Failed dictation and cancelled processing retain audio.
Quit closes active recordings before exiting. On reopening, Lalia attempts to
recover unfinished audio and reports files it cannot recover.

Sessions retain separate lossless PCM CAF tracks, rotating every minute. Lalia
aligns their offsets into one 16 kHz mono WAV for one MOSS request. It preserves
gaps and reduces gain only when the sum would clip. Dictation keeps its ten-minute
WAV jobs and Cohere routing. Both skip exact digital silence and preserve quiet
audio. Ambient noise can still produce hallucinated text. Timestamps and speaker
assignments need review. Recovery cannot guarantee that the final partial file
survives power loss.

Lalia checks token usage, context limits, timed passages, and raw response
coverage before replacing a transcript. A rejected response stays with the
saved audio for diagnosis. These checks reject exhausted budgets and malformed
responses. A well-formed early stop can still omit speech. First writes to
older manifests retain a private schema-1 backup before upgrading to schema 2.

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
verifies dictation/session model routing, Unicode speakers, multipart limits,
token-usage rejection, and redirect rejection. It does
not measure transcription accuracy or establish macOS capture permissions.

Run live inference on an existing recording with the same client and chunk
preparation as the app:

```sh
dist/Lalia.app/Contents/MacOS/Lalia --transcribe-file /path/to/speech.wav
dist/Lalia.app/Contents/MacOS/Lalia --transcribe-session-file /path/to/meeting.wav
```

The probe accepts `--config /path/to/config.json` after the audio path. Its
derived files are temporary, and it leaves the input recording intact. To
include live inference in the app check, set `LALIA_LIVE_ASR=1` and
`LALIA_ASR_FIXTURE=/path/to/speech.wav`. Add `LALIA_SESSION_ASR=1` for MOSS.
A requested live check fails if the
server, model, or inference is unavailable. Signed microphone/system capture
and real ASR quality need separate acceptance checks.
