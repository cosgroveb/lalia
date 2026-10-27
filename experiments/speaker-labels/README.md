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

For the controlled two-second follow-up, preserve the original three-second
selection and shorten only its end frames. Use the same runtime and model:

```sh
"$matching/venv/bin/python" -B experiments/speaker-labels/window_matching.py \
  --manifest "$manifest" --model "$matching/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
  --output /private/path/to/fresh-two-second-results --clip-seconds 2 --select-only
```

Compare the frozen starts, labels, roles and input hashes with the original
protocol before repeating without `--select-only`. The default remains three
seconds. Both durations require all 15 held-out clips correctly accepted at
0.55, with no rejects or wrong identities, before proceeding to an ASR run.

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

The controlled two-second follow-up retained all 23 original clip starts and
roles. It correctly accepted 13 of 15 held-out clips, rejected one and wrongly
accepted one as another speaker. It failed the frozen acceptance rule. No
two-second ASR follow-up ran. Wall time was 0.514 seconds and peak RSS was
251.91 MiB. The 0.5-second footprint sampler caught only startup (196,752
bytes), so its sampled value missed the worker's peak.

## Tests

From the repository root in an initialized login shell, use the matching
environment for its feature-extraction tests and the scoring environment for
the remaining suites. Neither command runs model inference.

```sh
matching="$HOME/.local/state/lalia-speaker-qualification/window-matching-20261027"
scoring="$HOME/.local/state/lalia-speaker-eval-20261025/scoring-venv/bin/python"
"$matching/venv/bin/python" -B -m unittest discover \
  -s experiments/speaker-labels -p test_window_matching.py
PYTHONPATH=experiments/speaker-labels "$scoring" -B -m unittest \
  test_score test_run test_qualification test_short_windows
```

## Frozen-prefix RoPE diagnostic

Compare next-token scores at the observed 90-minute early EOS with
`eos_rope.py`. Preparation hashes the public audio, saved token prefix,
model, isolated runtime, operator binaries, and production files. It does
not load model weights. Run preparation with the oMLX Python environment:

```sh
/opt/homebrew/opt/omlx/libexec/bin/python -B experiments/speaker-labels/eos_rope.py --prepare
```

Review the driver and the resulting
`~/.local/state/lalia-speaker-qualification/eos-rope-20261027/protocol.json`
before running `--run` with the same command. The run builds audio/prompt
embeddings once, appends the saved 17,127 generated tokens, and consumes
all 88,814 inputs. The final query occupies position 88,813. Each pass uses
2,048-token chunks followed by one final token.

The installed sampler (including normalization in the original dtype) must
select EOS 151645. Raw-logit argmax and FP32 probabilities are separate
diagnostics. Otherwise the probe records
`native_control_not_reproduced` and stops, since chunked replay can change
numerics. After releasing its KV cache, the reference pass reuses the
embeddings with Torch CPU FP32 RoPE frequencies and trigonometry. Evaluated
products preserve Torch eager BF16 rounding. Tiny tests verify exact
rotation equality on MLX CPU and GPU. This compares one operator, not full
Transformers backend parity or transcription accuracy.

The supervisor enforces 300 seconds per pass (including reference-table
preparation), 600 seconds overall, a sampled 32 GiB child footprint, and a
25% battery cutoff. It stops and waits for its worker and awake process.
`result.json` records numeric token scores, timing, sampled footprint and
the final frozen-file check. No generated text or second live KV cache is
needed. An attempted state directory cannot run again automatically.

Run both probe test suites without loading model weights:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /opt/homebrew/opt/omlx/libexec/bin/python -B \
  -m unittest discover -s experiments/speaker-labels -p 'test_eos_*.py'
```

The October 27 attempt stopped at the 32 GiB guard after 10.93 seconds.
The sampled peak was 34.25 GiB. Audio/prompt setup took 6.80 seconds, and
native replay consumed 22,528 of 88,814 inputs. No final native score or
reference comparison exists. The supervisor reaped both children and
verified the frozen production/input files unchanged. It did not retry.

That attempt cleared MLX's allocator cache after each pass. Installed
generation clears it after each prefill chunk. That difference may explain
the diagnostic's memory growth. This failed replay establishes neither MOSS's
required memory nor a connection between RoPE and premature EOS. The state
directory retains the measured source snapshot, protocol, memory samples,
and cleanup verification.

### Allocator follow-up

`eos_allocator.py` isolates per-chunk allocator clearing with two fresh
native workers. Both rebuild and hash the full frozen input embeddings,
then consume exactly eight 2,048-token chunks (16,384 inputs). Worker A
keeps the existing allocation policy. Worker B adds `mx.clear_cache()`
after each chunk. Neither computes final logits or generates tokens.

Each worker records synchronized MLX active/cache/peak bytes, physical
footprint, and logical/allocated KV bytes. A terminal clear records the
same quantities before and after, with no subsequent model computation.
The final hidden-state digests must match. Worker A saves and validates
the full BF16 embeddings after measurements for possible later reuse.

Prepare `eos-allocator-20261027` with the command below, then review its
driver and protocol before replacing `--prepare` with `--run`. Either
worker failing, reaching a guard, or producing different inputs/hidden
states stops the comparison. Limits are 30 seconds per worker, 60 seconds
overall, 32 GiB sampled physical footprint, and 25% battery.

```sh
/opt/homebrew/opt/omlx/libexec/bin/python -B experiments/speaker-labels/eos_allocator.py --prepare
```

The October 27 allocator A/B completed in 19.66 seconds with identical
input and final-hidden digests. A's terminal clear reduced physical
footprint from 27.40 to 9.10 GiB and cached allocation from 18.74 GiB to
zero. Active allocation stayed at 3.62 GiB, including 1.75 GiB of KV
storage. B peaked at 19.82 GiB physical footprint and ended at 9.98 GiB.
The measured prefixes took 0.995 seconds (A) and 1.104 seconds (B).

This establishes allocator retention at 16,384 inputs. It does not test
the long-prefix EOS decision. Both workers exited successfully, cleanup
reaped all children, and all 1,504 frozen files remained unchanged. The
state directory includes the validated BF16 embedding artifact for a
separately reviewed follow-up.

The corrected replay adds one allocator clear after every evaluated chunk.
`eos-rope-cleared-20261027` retains the same re-encoded inputs, 2,048-token
schedule, native EOS requirement, conditional reference pass, and resource
limits. It does not reuse the saved embeddings. This command prepared its
state for independent review:

```sh
/opt/homebrew/opt/omlx/libexec/bin/python -B experiments/speaker-labels/eos_rope.py \
  --state "$HOME/.local/state/lalia-speaker-qualification/eos-rope-cleared-20261027" --prepare
```

Completed protocols and results remain unchanged. Their source snapshots
retain the measured code. Historical verification maps original source
paths through those manifests, so later edits cannot substitute current
code for a completed experiment.

The corrected October 27 replay completed in 65.05 seconds at a sampled
peak physical footprint of 25.69 GiB. Native replay consumed all 88,814
inputs in 25.97 seconds and selected EOS 151645. EOS scored 24.875 versus
24.75 for continuation 42474, a 0.125 margin (53.12% versus 46.88%).

Reference RoPE took 30.34 seconds and selected continuation 42474 through
an exact BF16 tie: both tokens scored 24.875, about 50% each. EOS's score
did not change. The continuation rose one BF16 step, and the lower token
ID won the tie. This establishes sensitivity to RoPE operator numerics at
the frozen prefix. It does not establish a robust fix or completed
transcription. The full input digest matched the earlier attempt, cleanup
reaped both children, and all 1,484 frozen files remained unchanged.

## CUDA decoder comparison

`cuda/` scores the same saved embeddings with the official Qwen3 decoder
on Spark's GB10. It loads unchanged BF16 decoder weights and the tied head,
uses the same 2,048-token schedule, and validates every layer's cache length.
It does not encode audio or generate a transcript. The default loader matches
the published MOSS runner's BF16 cast, including its rotary-frequency buffer.
`score --fp32-rope` recomputes that buffer from the pinned default initializer
on CPU after the cast. It verifies unchanged scaling and the original BF16
round-back hash. Weights and all other computation remain unchanged. Use a
fresh root for each comparison. Full native logits accompany the summary.

The isolated ARM64 image uses Torch 2.14.1+cu130 and Transformers 5.17.0.
Build from `cuda/`, tagging it `lalia-moss-cuda-probe:20261027-r4` for the
supervisor. The Dockerfile preserves the measured package-install order.
Its cuSPARSELt repair changes the pinned wheel's invalid platform metadata,
regenerates RECORD and verifies every other payload hash. Library bytes stay
unchanged. GCC and libc headers let Torch compile its Triton driver helper.
The final dependency check must pass.

These are dated, fixture-specific diagnostics. On Spark, prepare a fresh
root with `probe/` containing the Python scripts and `inputs/` containing
`manifest.json` plus its three named files. Use the public model and saved
AMI embeddings from the allocator experiment. The scorer checks their
hashes and seven installed Transformers source hashes. Do not substitute
private audio or embeddings. Results are written under `results/<mode>`.

Run `python3 ROOT/probe/supervise.py MODE --root ROOT` on Spark. Modes are
`smoke` (small attention/GEMM checks), `shapes` (long-context attention),
`decoder` (tiny complete CPU/CUDA decoder), and `score` (the frozen prefix).
Review actual results between modes. A failed check is evidence, not
permission to continue automatically.

The supervisor requires Docker GPU access, an active `vllm-vllm.service`
and permission to stop/start it. Only `score` pauses that service. It limits
the container to 300 seconds plus five seconds to stop, 24 GiB with no swap,
and a 24 GiB CUDA allocator. The scorer requires 32 GiB available initially
and checks an 8 GiB reserve after each chunk. Cleanup confirms the owned
container is absent, reaps its client, then restores the service and checks
health. Read `lifecycle.json` as well as the model result. Restoration can
take several minutes. No production image or configuration is upgraded.

The October 27 kernel checks passed their fixed CPU tolerances, including
attention shapes up to 88,814 keys. The tiny complete decoder did not:
fused CUDA had 2/128 logit errors outside tolerance, and CUDA math had 5/128.
All three argmaxes agreed. Relative L2 differences from CPU were about
0.45%. Those failures remain failures. The subsequent full score was
reviewed as a runtime-specific observation with numerical parity unresolved.

That full score consumed all 88,814 inputs in 51.67 seconds, with 11.75 GiB
peak CUDA allocation. Continuation 42474 scored 28.625 versus EOS at 21.125,
a −7.5 EOS margin. This differs from native MLX's +0.125 margin and the
FP32-reference RoPE tie. It narrows the decoder investigation but does not
isolate a cause: the official loader rounds frequencies to BF16, and other
operators differ across runtimes. Peak allocation is not physical footprint,
and decoder scoring time is not full transcription time. The container and
client exited, and the original coding service recovered with healthy HTTP.

The frequency-only CUDA control completed in 52.57 seconds with the same
11.75 GiB peak allocation. Recomputed FP32 frequencies changed the EOS margin
from −7.5 to zero: EOS and continuation both scored 24.875. These two scores
match the local FP32-reference RoPE result. This isolates the effect of
frequency rounding on that CUDA decision. Other logits and operators are not
established equivalent, and neither run generated continued transcription.

Run the CUDA diagnostic CPU tests without GPU inference:

Download [configuration_moss_transcribe_diarize.py](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/resolve/704aa4a9c304e8520be88901e0d1960158ef5b15/configuration_moss_transcribe_diarize.py)
and [modeling_moss_transcribe_diarize.py](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/resolve/704aa4a9c304e8520be88901e0d1960158ef5b15/modeling_moss_transcribe_diarize.py)
into one directory, keeping their filenames. Set `MOSS_OFFICIAL_SOURCE` to
that directory.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MOSS_OFFICIAL_SOURCE=/path/to/pinned/moss/source \
  /opt/homebrew/opt/omlx/libexec/bin/python -B -m unittest discover \
  -s experiments/speaker-labels/cuda -p 'test_*.py'
```

The environment needs Torch, Transformers 5.17.0, safetensors, wheel 0.48.0
and packaging 26.0. Tests verify both upstream file hashes before importing
them. Without `MOSS_OFFICIAL_SOURCE`, only that equivalence test is
skipped. Acceptance used the fixture: all 42 tests passed with no skips.
