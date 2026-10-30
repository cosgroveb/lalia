# Lalia

Native macOS menu-bar dictation and session recording. Hold either Option key
to record dictation, then release to paste a local MOSS transcript. Double-tap
the same Option key to start or stop recording microphone and Mac audio, then
add speaker labels through local MOSS after Stop. Recordings stop after 60 minutes.

Speaker labels remain experimental. Both tested hour-long recordings completed,
but MOSS can omit speech or add incorrect text and speaker labels. Review the
transcript against the saved audio. Failed analysis retains audio and the
previous transcript for retry. The 60-minute capture limit does not guarantee
transcription accuracy.

Requires Apple silicon and macOS 26 or newer. Gestures need Input Monitoring.
Dictation needs Microphone and Accessibility permission. Sessions need
Microphone and Screen & System Audio Recording permission. Accessibility is
only for pasting.

## Install and configure

Install the signed, notarized app and its isolated transcription runtime:

```sh
brew install --cask cosgroveb/tap/lalia
lalia-runtime setup
```

Setup downloads the public MOSS checkpoint, starts the local runtime, verifies
transcription, then configures Lalia. The runtime owns its Python dependencies
and API compatibility patches. It does not modify a separate oMLX installation.
Use `Check Runtime Setup` in Lalia's menu to check server/model availability,
or `Copy Setup Command` to get the installation command. Recordings remain
available for Retry if the server is offline or incompatible.

The runtime uses loopback port 8002 and private state under
`~/.local/share/lalia/runtime`. Its pinned oMLX 0.7.2 / mlx-audio 0.6.2 build
includes the MOSS frequency correction at
`67408c4910edd83935f5618780e51c403ded2676`, prompt forwarding, token usage,
and a 250 MB upload limit. Models download during setup, not Homebrew installation.
Notes use a separate local chat server. The runtime does not install a notes model.

Lalia reads `~/.config/lalia/config.json`. New installations default to:

```json
{
  "baseURL": "http://127.0.0.1:8002/v1",
  "summaryBaseURL": "http://127.0.0.1:8000/v1",
  "transcriptionModel": "MOSS-Transcribe-Diarize",
  "language": "en",
  "summaryModel": "Qwen3.6-35B-A3B-oQ8-mtp"
}
```

Existing configurations without `summaryBaseURL` retain their existing endpoint
for notes. Setup preserves custom endpoints and summary settings. Older
`sessionTranscriptionModel` values and the retired Cohere identifier migrate as
before. Existing compatible loopback servers remain supported. Both URLs must
use loopback addresses. Lalia disables HTTP redirects and system proxies.

Dictation requests cap output at 8,192 tokens, uploads at 20 MB, and requests at
30 minutes. Each dictation chunk is at most ten minutes. Pasted text omits
speaker labels, timestamps and sound events. Missing or inconsistent token
usage fails analysis rather than accepting an incomplete response.

## Use

Choose `Enable Dictation` and `Enable Gestures`, focus a text field, hold either
Option key, speak after Recording appears, then release. `Copy Last Transcript`
remains available if pasting fails. Option-letter, modifier, mouse and scroll
chords cancel capture without transcription or paste; cancelled audio remains
available in History for explicit retry or deletion.
Holding for 250 ms starts dictation. A same-side second press within 300 ms
toggles session recording when released before the hold threshold.

If Input Monitoring is lost, Secure Event Input starts, or the Mac sleeps,
Lalia saves active audio without automatic transcription or analysis. After
wake or when Secure Event Input ends, choose `Enable Gestures` to restore the
listener; gestures rearm only after all keys are released. Clipboard restoration
covers eagerly readable pasteboard data. macOS does not expose every lazy or
promised representation.

Double-tap the same Option key to start recording microphone and Mac audio,
then double-tap it again to stop, save, and transcribe. The menu and history
window offer the same `Start Recording` / `Stop Recording` button. Recording
continues when you release the key. Lalia shows elapsed and remaining time,
warns with five minutes remaining, then saves and transcribes at 60 minutes. It does not start
another recording. Holding the key does not toggle repeatedly. Lalia ignores
the toggle while starting, saving, or dictating. Use
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
WAV jobs and MOSS routing. Both skip exact digital silence and preserve quiet
audio. Ambient noise can still produce hallucinated text. Timestamps and speaker
assignments need review. Recovery cannot guarantee that the final partial file
survives power loss.

Lalia checks token usage, context limits, timed passages, and raw response
coverage before replacing a transcript. End timestamps may exceed the audio
by at most 100 ms for rounding. Lalia retains the original timestamp and text. A rejected response stays with the
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
