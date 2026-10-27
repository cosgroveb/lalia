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

## Short-window diagnostic

Run `window_matching.py` first to test CPU speaker matching on disjoint AMI
reference clips. It does not test diarization or unknown speakers. Use Python
3.13.11 in an isolated environment with `numpy==2.4.4`,
`onnxruntime==1.24.3`, and `kaldi-native-fbank==1.22.3`. The measured transitive
versions were flatbuffers 25.12.19, mpmath 1.3.0, packaging 26.3, protobuf
7.36.2, and sympy 1.14.0. No upstream realtime framework is required.

Download `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` from
`csukuangfj/speaker-embedding-models` at revision
`0743f301363dec56491a490f6d6cbc9d67f9a3bf`. Both scripts require SHA-256
`f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11`.
From the repository root in an initialized login shell:

```sh
matching="$HOME/.local/state/lalia-speaker-qualification/window-matching-20261027"
manifest="$HOME/.local/state/lalia-speaker-eval-20261025/data/public/manifest.json"
uv venv --python python3.13 "$matching/venv"
uv pip install --python "$matching/venv/bin/python" --only-binary :all: \
  numpy==2.4.4 onnxruntime==1.24.3 kaldi-native-fbank==1.22.3 \
  flatbuffers==25.12.19 mpmath==1.3.0 packaging==26.3 protobuf==7.36.2 sympy==1.14.0
curl --fail --location --output "$matching/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
  https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/0743f301363dec56491a490f6d6cbc9d67f9a3bf/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
"$matching/venv/bin/python" -B experiments/speaker-labels/window_matching.py \
  --manifest "$manifest" --model "$matching/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
  --output /private/path/to/fresh-matching-results --select-only
```

Review the frozen selection, then repeat without `--select-only`. Completed
results cannot be overwritten. Only the public `ami-ES2004a-full` clip is used.

`short_windows.py` then tests three fixed windows, [0,390], [330,750], and
[690,end] seconds, through the existing isolated patched API runtime. Its
paths name the saved October 26 baseline, prompt, launcher and pins under
`~/.local/state/lalia-speaker-qualification/official-api-path`. Preparation
verifies those artifacts and extracts sample-exact PCM. It requires those
prior local qualification artifacts, not the default installed oMLX launcher.
Choose a fresh output directory for each attempt. An attempted run cannot
resume or overwrite its evidence.

```sh
scoring="$HOME/.local/state/lalia-speaker-eval-20261025/scoring-venv/bin/python"
windows="$HOME/.local/state/lalia-speaker-qualification/official-api-path/short-windows-reproduction"
"$scoring" -B experiments/speaker-labels/short_windows.py --state "$windows" --prepare
"$scoring" -B experiments/speaker-labels/short_windows.py --state "$windows" --run
```

The run uses port 18001, one server and serial requests with the same literal
prompt and 49,152-token ceiling. Limits are 300 seconds per request, 600 seconds
overall, 16 GiB sampled child footprint, and 25% charge on battery. It stops
and waits for owned children on exit. Audio, raw replies and provenance stay
in private state.

Matching uses only predicted single-label spans, trims 150 ms, and takes the
first three seconds from up to two longest distinct usable spans per label.
First-window normalized means remain frozen. Later labels must meet cosine
0.55 and map uniquely. Rejection or collision stops the experiment. Seams use
the nearest midpoint of a shared speech-free gap at least 0.2 seconds long.
The join keeps whole speech segments and accounts for redundant copies.
Non-speech events keep their full intervals under midpoint ownership.

Aggregate results compare cpWER and strict/collared DER against the saved
full-file diagnostic baseline. The regression limit is five percentage points
for cpWER and collared DER. Reference-position alignment counts seam deletions
and insertions within 30 seconds of each cut, with overlap/repeated-word
limitations. Insertions are not proven duplicates. No threshold or boundary
tuning follows this evaluation. These checks do not qualify 90-minute input,
new speakers, ambient microphones, or deployment.

The October 27 Stage A probe correctly accepted all 15 held-out oracle clips.
One of 45 impostor pair scores exceeded 0.55, so the threshold alone does not
establish unknown-speaker rejection. Stage B's first 390-second request
completed in 10.10 seconds, with final predicted speech ending at 384.42
seconds. Enrollment stopped before matching because S03's longest clean span
left 2.98 seconds after trimming, below the fixed three-second requirement.
No later window, matching or seam scoring ran. Request completion and the last
timestamp do not establish full speech coverage or merged accuracy.
