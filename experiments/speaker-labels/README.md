# Speaker-label experiment

Compare local oMLX speech models on annotated meeting audio before changing
Lalia's transcription backend. These scripts leave the app and production
oMLX configuration unchanged. Keep audio, responses and generated manifests
outside the repository, particularly when using a retained Lalia recording.

The October 2026 run uses oMLX 0.7.2, mlx-audio 0.5.7 and MLX 0.32.4 on a
128 GiB M5 Max. `download_models.py` pins offline VibeVoice-ASR BF16 and
MOSS-Transcribe-Diarize weights. It also downloads VibeVoice's missing Qwen2.5
tokenizer and MOSS's chat template. Cohere uses the existing local installation.

## Run

Run from this directory in an initialized login shell. Create an isolated
scoring environment with Python 3.11 and install `requirements.txt`. Use the
oMLX environment's Python for model downloads.

```sh
umask 077
state="$HOME/.local/state/lalia-speaker-eval"
mkdir -p "$state/models"
python3.11 -m venv "$state/scoring-venv"
"$state/scoring-venv/bin/pip" install -r requirements.txt
"$(brew --prefix omlx)/libexec/bin/python" download_models.py "$state/models"
ln -s "$HOME/.omlx/models/cohere-transcribe-03-2026" "$state/models/cohere-transcribe-03-2026"
python3 prepare_public.py --output "$state/data/public"
"$state/scoring-venv/bin/python" run.py --state "$state" \
  --manifest "$state/data/public/manifest.json"
```

Check each preparation script's `--help` for optional local session inputs.
The public manifest contains AMI ES2004a, a 17-minute four-speaker headset
mix, and a three-minute excerpt selected before inference. The excerpt has
substantial speech from all four speakers. Source checksums, reference
revisions and CC-BY-4.0 attribution accompany the generated audio. The
overnight matrix also includes a repeated excerpt, 20 seconds of digital
silence, and short/full versions of one saved recording.

`run.py` starts a separate loopback oMLX process on port18001, one model at a
time. It refuses an occupied port, limits each request to40 minutes and each
server to64 GiB sampled physical footprint, and stops its server on exit.
Use `--deadline` with a Unix timestamp for an overall cutoff. Existing result
files are skipped, including failures. Preserve failed attempts elsewhere
before rerunning them.

## Qualify capped recordings

Use an isolated copy of oMLX 0.7.2 with the JSON usage patch from the dotfiles
installer. Keep production oMLX running. Point `--omlx` at a launcher that
sets `PYTHONPATH` to that copy before invoking the installed oMLX Python.
The patcher accepts `--site-packages` and `--backup-dir` for isolated copies.
Those options never restart Homebrew.

Prepare repeated public fixtures with `prepare_public.py --output "$state/data/public"
--stress-minutes 60 90`. Each fixture ends with the source meeting's tail and
retains shifted references. These test capacity and continuity, using repeated
audio rather than independent accuracy samples.

Supply `--prompt-file /path/to/official-moss-prompt.txt` for both baseline and
long requests. Use the exact `SessionTranscriptionProfile.officialPrompt`
from `Sources/Lalia/SpeakerTranscript.swift`: 222 UTF-8 bytes, no added newline,
SHA-256 `5696a90762c9a738598bdf62530952cb9058f0bd2da8dccc23d988b68990e22a`.
The runner sends literal multipart text and records the prompt, source path
and hash. The patched signature adapter must forward it to MOSS. The English
backend default stopped early in two hour-long tests.

Run the full 17-minute meeting first with `--qualify --max-tokens 49152` and
that prompt file.
Run each long fixture with the same options plus `--baseline` pointing to
that fresh result's `result.json`. Qualification requires actual token usage,
EOS below the requested ceiling, complete raw/structured coverage, valid
overlapping timestamps, four global speakers, final speech and no more than
five percentage points of cpWER or collared DER regression. It reports
matched-passage timestamp error near the beginning, middle and end and checks
speaker assignment across repeated sections. An exhausted response permits
one diagnostic at 57344 tokens. Do not retry with a larger ceiling for other
failures or add production escalation.

Stress, repeated and hour-long fixtures require a completed, accepted short
MOSS baseline with finite cpWER and collared DER and identical recorded prompt
text and hash. Missing, invalid or archived baselines without prompt metadata
stop the runner before it starts children. Short calibration may omit one.

The runner records swap before and after each request. A cancelled client can
leave the server decoding, so the runner stops and waits for its owned server
on failure or exit. Production Lalia can only abandon the result. Verify the
isolated port closes before starting another run.
At 25% charge on battery power, the runner stops the whole batch and waits for
its client, server and awake assertion. AC power does not trigger this cutoff.

Results contain raw local responses, aggregate scores and memory samples.
`summary.json` and `REPORT.md` contain aggregates only. HTTP timing excludes
server startup, which has its own field. Physical footprint is sampled every
0.5 seconds and can miss short peaks. Cold/warm describes whether that server
has already processed a request, not the operating system's file cache.

`report_at.py` keeps the machine awake and queues a follow-up in an existing
Codex task at an explicit time. The overnight run launches it through a
temporary LaunchAgent. Remove that agent after the report. It does not prevent
lid-close sleep or a shutdown.

## Interpret

WER lowercases text, removes punctuation and normalizes Unicode. cpWER matches
whole speaker transcripts under the best label permutation. DER includes
overlap and reports zero collar plus a ±250 ms boundary tolerance. Plain WER
depends on word order during overlapping speech. Sentence-level model times
and manual speech-turn boundaries also differ, so inspect cpWER and DER
together. Known non-speech event tags are excluded from hypothesis speech.
Reference speech under those events still counts as missed speech. Scoring
tests cover label permutation, merged speakers, non-speech events, missing
output, invalid times, overlap, silence and privacy.

HTTP completion does not establish usable speaker output. Check schema
diagnostics, missing speech and reference coverage. Private recordings have
no reference, so their results establish completion and output structure,
not accuracy. Digital-silence hallucinations describe model behavior. Lalia
already skips exact digital silence before inference.

One headset-mix meeting cannot establish accuracy on ambient MacBook audio.
Training-set overlap is unknown. Scores describe these installed model/runtime
combinations, not a general model ranking. Deployment still requires long-request
and real two-track qualification.

The October 26 actual-API test with the exact official prompt passed its
17-minute baseline (12.44% cpWER, 15.32% collared DER). Its 90-minute request
failed: native EOS after 17,127 output tokens despite a 49,152 ceiling, last
parsed speech at 36m49s, 65.70% cpWER and 66.22% collared DER. It took 476.03s
at 23.94 GiB peak footprint, with unchanged swap. A separate 60-minute prompt
diagnostic succeeded, but does not qualify the 90-minute path. Deployment,
real-device testing and coding-model contention checks remain on hold.
