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
together. Scoring tests cover label permutation, merged speakers, missing
output, invalid times, overlap, silence and privacy.

HTTP completion does not establish usable speaker output. Check schema
diagnostics, missing speech and reference coverage. Private recordings have
no reference, so their results establish completion and output structure,
not accuracy. Digital-silence hallucinations describe model behavior. Lalia
already skips exact digital silence before inference.

One headset-mix meeting cannot establish accuracy on ambient MacBook audio.
Training-set overlap is unknown. Scores describe these installed model/runtime
combinations, not a general model ranking. Lalia still needs persistent speaker
IDs, overlap handling and a longer token budget before these responses can
become a session feature.
