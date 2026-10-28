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
the [published MOSS runner's BF16 cast](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/79c675f0670cc2f824bcbcdd98263704f40f1f7e/moss_transcribe_diarize/app/model_runner.py#L178-L192),
including its rotary-frequency buffer.
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

## Match the published rotary frequencies

`eos_rope.py --prepare --bf16-frequencies` rounds the reference inverse
frequencies through BF16 before its existing FP32 trigonometry.
`--prepare --native-bf16-frequencies` instead uses a MOSS-local wrapper over
`mx.fast.rope`, with FP32 denominators derived from those same BF16 values.
Both verify the published runner's frequency hash. The flags are mutually
exclusive and prepare-only. Omitting them preserves the FP32 reference.

The native wrapper replaces only this model's attention RoPE objects during
the reference pass, then restores them. It changes neither shared Qwen code
nor model weights. MLX's native trig and rotation differ from the reference's
BF16 intermediates, so matching frequencies does not establish exact parity.

Prepare a fresh native-candidate replay from the repository root:

```sh
state="$HOME/.local/state/lalia-speaker-qualification/eos-native-reproduction"
/opt/homebrew/opt/omlx/libexec/bin/python -B experiments/speaker-labels/eos_rope.py \
  --prepare --native-bf16-frequencies --state "$state"
```

Review the frozen protocol before running the same command with `--run` in
place of both preparation flags. Selection comes from the hashed protocol.
Native EOS must reproduce before either alternative runs. The existing
32 GiB footprint, 300-second pass, 600-second total and 25% battery guards
remain. Full native-dtype logits and their hashes accompany both summaries.
Historical runs retain their measured source snapshots and protocol formats.

`native_rope_basis.py --out /private/path/to/fresh-basis-results` checks eight
small GPU cases without loading weights. It covers single/chunk inputs,
FP32/BF16 and positions 0/88813 with model-like strides. The fixed 0.02 bound
detects gross errors, not exact parity. The October 27 cases passed: long
BF16 basis outputs matched the reference, while maximum FP32 error was
0.000183. Using inverse frequencies as denominators produced errors near 2.

Measured conditional scores at the same 88,814-input prefix:

| Decoder and rotary calculation | EOS | Continuation 42474 | EOS minus continuation |
|---|---:|---:|---:|
| Native MLX | 24.875 | 24.750 | +0.125 |
| MLX with FP32 reference frequencies | 24.875 | 24.875 | 0 |
| CUDA with published BF16 frequencies | 21.125 | 28.625 | −7.5 |
| CUDA with recomputed FP32 frequencies | 24.875 | 24.875 | 0 |
| MLX reference with BF16 frequencies | 21.125 | 28.5 | −7.375 |
| Native MLX with BF16-derived denominators | 21.0 | 28.625 | −7.625 |

The BF16-reference run took 68.67 seconds at 24.75 GiB sampled footprint.
The native candidate run took 64.08 seconds at 25.64 GiB. Its original and
candidate passes took 27.11 and 27.98 seconds. These are conditional replay
timings, not full transcription performance. Both native controls reproduced
the original scores. The later native vector matched the earlier one byte
for byte. Input/prompt and saved-logit hashes matched, frozen files stayed
unchanged and owned processes exited.

Frequency precision explains a large change in this stopping decision across
both runtimes. The actual API test below still failed long-audio completion.
No production change is qualified.

Run `test_eos_*.py` as above, then include the native wrapper tests:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /opt/homebrew/opt/omlx/libexec/bin/python -B \
  -m unittest discover -s experiments/speaker-labels -p 'test_native_rope.py'
```

All 28 relevant tests passed: 17 RoPE, eight allocator and three native tests.

### Actual API qualification

The frequency change extended the 90-minute fixture's transcription from
36m49s to 62m52s, but it still stopped early. The short baseline passed.

A separate launcher called the original MOSS post-load hook before installing
the measured native-frequency wrapper on that model's 28 layers. It reused
the frozen runtime, official prompt, 49,152-token cap and 4,096-token API
prefill. Seven focused tests and independent setup review passed.

| Audio / frequencies | Accepted | Last speech | HTTP seconds | Peak GiB | cpWER | Collared DER |
|---|---|---:|---:|---:|---:|---:|
| 17m29 / original | Yes | 1048.98s | 68.64 | 8.36 | 12.44% | 15.32% |
| 17m29 / BF16-derived | Yes | 1048.94s | 65.14 | 8.25 | 12.17% | 13.78% |
| 90m / original | No | 2208.71s | 476.03 | 23.94 | 65.70% | 66.22% |
| 90m / BF16-derived | No | 3771.62s | 857.00 | 26.13 | 42.76% | 43.42% |

The candidate baseline returned 8,693 output tokens, native EOS and 293
reconciled segments across four speakers. The long request returned 29,448
output tokens plus observed EOS 151645, below the output ceiling. Its 933
segments leave 19 unmatched trailing characters and the final 27m08s
untranscribed. Validation rejected `Unmatched raw transcript span`.

Speaker mappings agree across the first three complete repeated sections and
the partial fourth. The absent final sections fail whole-recording continuity.
This is missing coverage, not evidence of identity drift in the returned
sections. Timestamp matching found beginning and middle passages, with no
final speech available to check. Swap did not grow.

All 1,495 frozen files matched after both requests. Owned processes exited,
the experimental port closed and production oMLX remained healthy. Evidence,
protocol, source archive and cleanup records are under
`~/.local/state/lalia-speaker-qualification/official-api-path/native-bf16-api-20261027`.
Production app, runtime and configuration remain unchanged. This repeated
fixture adds capacity evidence, not independent long-meeting accuracy evidence.
The frequency change alone is not a qualified truncation fix.

### Exact-token reproduction

Capture generator IDs and the score vector paired with native EOS before
replaying a failed prefix. Re-encoding the candidate's returned text yields
29,451 tokens, versus 29,448 generated, so text cannot reconstruct that prefix.
The observer also saves prompt embeddings and generated-token embeddings,
excluding terminal EOS. Trace serialization adds work to the HTTP request.

These diagnostics require the frozen local API-qualification state above.
Prepare a fresh trace directory with the oMLX Python:

```sh
python prepare_trace.py --state "$HOME/.local/state/lalia-speaker-qualification/exact-traces/repeat"
```

Run a selected command from its `protocol.json` explicitly, one at a time.
The commands carry `LALIA_EXACT_TRACE_STATE` into the isolated launcher.
Each phase checks its frozen dependencies and records whether text, output
count and native EOS match the historical response.

`exact_prefix.py --prepare --state <fresh-replay> --trace <completed-trace>`
requires a complete candidate reproduction. `--run` replays its 101,135
embeddings in 4,096-token chunks with a final singleton, then compares the
normalized scores with captured EOS scores. This differs from serial API
generation. A changed selection blocks `--export-cuda <fresh-bundle>`.
Exported bundles require `probe/supervise.py score --root <bundle> --exact-prefix`.
The shared CUDA scorer retains the earlier 88,814-input profile as its default.

October 27 repeats reproduced both historical failures. A fresh unchanged
control matched its earlier text, segments, language and usage, stopping after
17,127 output tokens at 36m49s. Two fresh candidate processes each matched the
earlier 29,448-token response ending at 62m52s. Their exact token IDs, prompt
embeddings, combined embeddings and EOS-paired score vectors were byte-equal.
This establishes repeatability on this fixture, not general determinism or
reliable long-recording transcription.

The original repeat took 406.61s at 23.78 GiB sampled physical footprint.
Instrumented candidate requests took 859.21s and 854.03s at 26.19 and 26.14
GiB. Neither grew swap. Trace timings include capture and serialization.
All 1,508 trace pins and cleanup checks passed.

At the candidate's later stop, the same 101,135 inputs produced these native
BF16 scores after identical MLX normalization:

| Execution | EOS log-probability | Best continuation (3821) | EOS margin | Selected token |
|---|---:|---:|---:|---:|
| Captured serial MLX | -0.625 | -0.750 | +0.125 | 151645 |
| Batched MLX | -0.625 | -0.875 | +0.250 | 151645 |
| Batched CUDA logits | -0.750 | -0.750 | 0 | 3821 |

MLX replay took 42.70s at 23.85 GiB sampled physical footprint. CUDA scoring
took 64.32s with 14.08 GiB peak CUDA allocation, a different memory measure.
Both used 4,096-token chunks and a final singleton. The CUDA tie survives
normalization. These near-tie differences do not identify a defective kernel.
Strict tiny CPU/CUDA decoder parity remains failed. The full official CUDA
comparison below subsequently completed this fixture.

Exact traces and replay evidence live under
`~/.local/state/lalia-speaker-qualification/exact-trace-20261027` and
`exact-prefix-20261027`. Spark's owned container and processes exited, and its
original coding service returned healthy with the same image. Production
Lalia and local oMLX remain unchanged.

### Residual rotary-operator comparison

`residual_rope.py` compares the native BF16-frequency wrapper with existing
`ReferenceRoPE(..., bf16_frequencies=True)` at the saved 101,135-input prefix.
It uses the historical runtime, fresh workers/KV caches and the same replay
schedule. Both historical raw and normalized vector hashes must reproduce
before the candidate runs. Neither worker generates a token.

Prepare a fresh state, then review its frozen protocol before replacing
`--prepare` with `--run` in the frozen `source/residual_rope.py` command:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 nice -n 10 \
  /opt/homebrew/opt/omlx/libexec/bin/python -B experiments/speaker-labels/residual_rope.py \
  --state /path/to/new-residual-trial --prepare
```

Limits are 300 seconds per arm, 600 overall and 32 GiB combined task-child
physical footprint. Useful work stops at 555 seconds to reserve postflight
time. Cancellation, guard failures or changed pins invalidate the comparison,
even if a worker exits zero. Cleanup reaps owned groups and verifies production
health. Completed states cannot be retried.

The October 28 comparison passed its baseline gate but rejected the hypothesis:

| Operator | Raw EOS / continuation 3821 | EOS margin | Wall seconds | Peak physical GiB |
|---|---|---:|---:|---:|
| Native BF16-frequency wrapper | 19.5 / 19.25 | +0.25 | 42.35 | 24.18 |
| BF16 reference | 19.625 / 19.25 | +0.375 | 49.91 | 23.70 |

EOS 151645 remained the unique maximum. Normalized margins agree. Both arms
consumed all 101,135 inputs with 28 BF16 caches, and the full comparison took
97.44 seconds. All 1,554 pins matched, production stayed healthy, owned workers
exited and swap did not grow.

This whole-operator replacement is insufficient at the residual prefix. It
changes phase reconstruction, CPU-versus-Metal trigonometry and BF16 rounding
together. The negative result does not rule out every rotary difference or
identify defective native math. No serial continuation or transcript was
generated. Full vectors and receipts remain under
`~/.local/state/lalia-speaker-qualification/residual-operator-20261028`.

### Official full CUDA comparison

Run the official audio frontend and generation helper to distinguish a
conditional decoder score from completed transcription. `cuda/official/prepare.py`
freezes the pinned public 90-minute WAV, model metadata, official helper,
scoring references and runner sources into a new directory outside the repo:

```sh
python cuda/official/prepare.py /path/to/new-state \
  --inputs /path/to/pinned-inputs --audio /path/to/public90.wav \
  --helper /path/to/inference_utils.py --references /path/to/public-manifest.json
```

`inputs-manifest.json` and `provenance.json` identify the required asset hashes
and revisions. Run the frozen `stage.sh` with a fresh Spark directory and the
existing remote weight-shard path. Reference transcripts stay local. Build
the isolated image with `build-image.sh`, then run `preflight.sh` and
`python3 -B probe/official_supervise.py kernel --root "$PWD"` there. These
preflights load the WAV and exercise the CUDA kernels without pausing services
or loading the model.

The full `python3 -B probe/official_supervise.py run --root "$PWD"` requires
matching source/image receipts and successful preflight cleanup. It pauses
only the main coding service and restores its original image and health
afterward. Limits are 64 GiB without swap and 10,800 seconds for the model.
GB10 admission uses host `MemAvailable`, with a 72 GiB admission threshold
and an 8 GiB running floor. The read-only container writes Triton/Torch
compiler caches into its executable 2 GiB temporary filesystem.

Retrieve the results and run the frozen `score_result.py` locally against
`results/e2e/model`. Native generated IDs establish EOS versus token-cap
termination. Missing output, failed lifecycle cleanup or changed pins are
failures even if a process exits zero. Never retry a failed run automatically.

The observer forwards the existing streamer's tensors unchanged and saves
selected CPU IDs every 128 tokens. Checkpoints remain incomplete lower bounds.
Only equality with the returned generation sequence establishes final ID
evidence. Checkpoint copying and writes add unmeasured CPU/I/O overhead, so
timing is not an uninstrumented performance benchmark.

The October 28 run hit its original 2,400-second deadline. Its last checkpoint
recorded 17,024 tokens at 2,391.74s and an 11.53 GiB CUDA allocation peak.
Both are lower bounds. That run saved no final IDs or transcript and cannot
establish premature EOS or accuracy. Cleanup restored the original coding
service and reaped owned processes. The extended three-hour bound allows
slower decoding than the observed final interval of 6.92 tokens/s. It does
not guarantee completion.

The extended run reached native EOS after 43,187 generated IDs, including
the terminal EOS, below the 49,152 cap. The observer's IDs exactly matched
the final returned sequence. It produced 1,345 valid passages, four speakers
and no unmatched raw text. Final speech ended at 89m58.31s, versus the last
reference word at 89m59.685s. Public scoring gave 14.70% cpWER, 15.38% DER
with a 250ms boundary tolerance, and 91.54% reference-speech coverage.

The worker took 6,882.39s (114m42s), including setup and observer overhead,
with 14.38 GiB peak CUDA allocation. This is not total physical footprint
and is not directly comparable with the Mac footprint measurements.
The host supervisor then failed its result read with `PermissionError`:
the container had created the model directory as root with mode 0700.
Read-only recovery independently verified the final IDs, response and scores.
The failed supervisor check remains recorded. Cleanup restored the original
coding-service image and HTTP 200, left sidecars unchanged and reaped every
owned process. Both source inventories and all 289 protected hashes matched.

The launcher now runs the container as the host UID/GID. A separate no-model
check reproduced the old permission failure, then verified host-readable
outputs with directory mode 0700 and file mode 0600. The existing tiny CUDA
kernel check also passed under that identity. All three check containers
exited, and production image, PID and health stayed unchanged. The full
transcription was not rerun for this ownership-only correction.

The same public 90-minute fixture now has these measured endpoints:

| Inference path | Last speech | Text tokens, excluding EOS |
|---|---:|---:|
| Original MLX | 36m49s | 17,127 |
| MLX with BF16-derived frequencies | 62m52s | 29,448 |
| Official CUDA | 89m58s | 43,186 |

The published model can complete this fixture through the official path.
The residual Mac cutoff remains to be isolated among the differing inference
paths. This result does not identify a defective kernel or qualify the
frequency patch as a complete fix. The MLX rows used the earlier frozen
runtime replacement. The exact-fork 60-minute matrix below is a separate test.
Full reference evidence is under
`/tmp/unified-coordinator-lalia-speaker-labels/official-cuda-e2e-20261028-r4`.

### Qualify a 60-minute limit

Use `limit60.py` to test the exact fork package through isolated oMLX before
changing Lalia's recording limit. Preparation freezes fork commit `67408c4`,
the existing HTTP prompt/usage patches, scoring code and public inputs. The
constructor audit requires the MOSS-local RoPE implementation and its measured
BF16-derived frequencies. It does not apply the earlier post-load replacement.

```sh
python limit60.py prepare --state /path/to/new-trial \
  --fork /path/to/mlx-audio --audio /path/to/IN1014.Mix-Headset.wav \
  --rttm /path/to/IN1014.rttm
```

Use the existing scoring environment and the [official IN1014 WAV](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IN1014/audio/IN1014.Mix-Headset.wav)
with its [pinned RTTM](https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/67c2d539286e89f68952d5dcf83912bd9f01dfae/only_words/rttms/train/IN1014.rttm).
The prepared manifest retains AMI attribution and the training-partition
caveat. MOSS training overlap is unknown. Existing local ES2004a assets and
the accepted historical short result are required too.

Run `source/limit60.py run --state /path/to/new-trial --phase <phase>` from
the frozen trial, in order: `a-control`, `a-hour`, `b-control`, `b-hour`.
Each phase uses a cold process and stops the matrix on failure. A tests the
existing repeated hour after a short regression control. B tests IN1014's
continuous final hour after its matched short control. Both retain the same
final five minutes as their controls.

Acceptance combines native EOS and token audits with full-reference scoring,
matched-tail accuracy/coverage and speaker consistency across the recording.
Read `evaluation.json` and `cleanup.json` before advancing. Control requests
have a 600-second bound, hour requests 2,400 seconds, with the existing 64 GiB
footprint and 25% battery guards. Record power before each request.

Passing would support an alpha claim for these two headset-mix inputs. It
would not establish universal 60-minute completion or ambient-audio accuracy.
The fork also contains upstream changes relative to the earlier runtime, so
these tests cannot isolate rotary-frequency rounding as the sole cause.

October 28 exact-fork results:

| Input | Accepted | Text tokens | HTTP seconds | Peak physical GiB | cpWER | DER, 250ms |
|---|---|---:|---:|---:|---:|---:|
| ES2004a 17m29 control | Yes | 8,693 | 65.18 | 8.25 | 12.17% | 13.78% |
| ES2004a repeated 60m | Yes | 31,806 | 668.63 | 20.80 | 12.40% | 12.88% |
| IN1014 17m29 control | Yes | 10,299 | 78.57 | 8.62 | 12.62% | 11.61% |
| IN1014 continuous 60m | No | 30,965 | 722.59 | 20.64 | 12.88% | 9.87% |

All four requests reached native EOS without swap growth. Both hour requests
reached the annotated ending. The repeated case kept its speaker assignments
across all blocks, and the independent case kept the four reference-speaker
assignments across all quarters. The short ES2004a response exactly matched
the earlier runtime-replacement response in text, segments, language and usage.

The independent hour failed with `Invalid segment timestamps`: its final
S05 passage contains 32 normalized words at 3593.81–3600.01s. The literal
`[3600.01]` exceeds input duration by 10ms. The extra label also fails the
four-speaker gate. No threshold or output was changed to accept it. That
passage follows the last annotated word at 3593.570625s. The source tail has
nonzero audio, so missing annotations do not establish silence or hallucination.
Its identical short-control tail generated one later word, not that 32-word
passage. Different input context and encoder alignment prevent isolating a
decoder cause.

Matched-tail checks stayed within their frozen deltas. The independent hour's
tail cpWER was 15.45%, DER 13.27% and reference speech coverage 95.27%, versus
12.76%, 12.70% and 94.26% in its control. This distinguishes the final output
contract failure from the earlier premature-cutoff failure. The formal
60-minute qualification remains failed.

Both hour outputs exceed the earlier 90-minute candidate's 29,448 text tokens,
which excludes a fixed 29k output ceiling on these runs. Their shorter input
contexts and the fork's other upstream changes remain confounders. These
observations do not establish a causal sequence-length threshold.

All 3,163 pins matched after the matrix. Nineteen recorded task PIDs exited,
port 18001 closed and production returned HTTP 200. Memory values are sampled
physical peaks. Nice 10 was applied during the repeated hour and inherited by
both IN1014 phases to preserve CPU responsiveness. Evidence and the unchanged
rejection are under
`~/.local/state/lalia-speaker-qualification/limit60-20261028-r2`.
