"""Freeze and run four explicit, serial qualification requests against the exact fork."""

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tarfile
import urllib.request
import wave

import prepare_public as public
from limit60_runtime import PHASES
from limit60_score import compare_tail, native_finish, one_audit, quarter_continuity, tail_metrics
from run import save


SOURCE = Path(__file__).resolve().parent
QUALIFICATION = Path.home() / ".local/state/lalia-speaker-qualification"
API = QUALIFICATION / "official-api-path"
FORK_COMMIT = "67408c4910edd83935f5618780e51c403ded2676"
PYTHON = Path("/opt/homebrew/opt/omlx/libexec/bin/python")
SCORING_PYTHON = Path.home() / ".local/state/lalia-speaker-eval-20261025/scoring-venv/bin/python"
MODEL = "MOSS-Transcribe-Diarize"
CLIPS = {"a-control": "ami-ES2004a-full", "a-hour": "ami-ES2004a-stress-60m",
         "b-control": "ami-IN1014-control", "b-hour": "ami-IN1014-hour"}
RTTM_SHA = "be6aaff13df15469a9d153e7fb6c0e8fffd5e7afea8e00727431364b94975b3a"
AUDIO_SHA = "ea8c65624b0a4ea4a1fad83fddbc34fcc2c0ff6912dc570e68699ce92da5b00a"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(hashes):
    for name, expected in hashes.items():
        if sha256(name) != expected:
            raise ValueError(f"Frozen file changed: {name}")


def crop_wave(source, destination, start, end):
    with wave.open(str(source)) as audio, wave.open(str(destination), "wb") as target:
        if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getnframes()) != (1, 16000, 2, 59085910):
            raise ValueError("IN1014 WAV header differs")
        if not 0 <= start < end <= audio.getnframes():
            raise ValueError("Invalid sample crop")
        target.setparams(audio.getparams())
        audio.setpos(start)
        remaining = end - start
        while remaining:
            count = min(remaining, 16000 * 30)
            data = audio.readframes(count)
            if len(data) != count * 2:
                raise ValueError("Truncated WAV")
            target.writeframes(data)
            remaining -= count
        return (end - start) / audio.getframerate()


def snapshot_fork(fork, runtime):
    revision = subprocess.run(["git", "rev-parse", FORK_COMMIT + "^{commit}"], cwd=fork,
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if revision != FORK_COMMIT:
        raise ValueError("Fork revision differs")
    archive = subprocess.run(["git", "archive", FORK_COMMIT, "mlx_audio"], cwd=fork,
                             check=True, capture_output=True, timeout=30).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source:
            path = runtime / item.name
            if not path.resolve().is_relative_to((runtime / "mlx_audio").resolve()):
                raise ValueError("Unexpected fork archive path")
            if item.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif item.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(item) as stream:
                    path.write_bytes(stream.read())
            else:
                raise ValueError("Fork snapshot must contain regular files only")


def environment(state):
    return {**os.environ, "LALIA_LIMIT60_STATE": str(state), "PYTHONPATH": str(state / "runtime"),
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}


def prepare(state, fork, audio, rttm):
    if state.is_relative_to(SOURCE.parents[1]):
        raise ValueError("Keep run state outside the repository")
    if state.exists():
        raise ValueError("State already exists. No automatic regeneration")
    previous_path = API / "native-bf16-api-20261027/protocol.json"
    if sha256(previous_path) != "a48c761b408eb2ffc484df99a343b139fb2ff86e609f4f730f06531424865dd9":
        raise ValueError("Historical candidate protocol differs")
    previous = json.loads(previous_path.read_text())
    verify(previous["hashes"])
    if sha256(rttm) != RTTM_SHA or sha256(audio) != AUDIO_SHA or audio.stat().st_size != 118171864:
        raise ValueError("IN1014 input size or RTTM differs")
    state.mkdir(parents=True, mode=0o700)
    data = state / "data"
    data.mkdir()
    shutil.copyfile(audio, data / "IN1014.Mix-Headset.wav")
    shutil.copyfile(rttm, data / "IN1014.rttm")
    archive = data / "ami_public_manual_1.6.2.zip"
    shutil.copyfile(QUALIFICATION / "data/public/ami_public_manual_1.6.2.zip", archive)
    original_meeting = public.MEETING
    try:
        public.MEETING = "IN1014"
        words, excluded = public.words_from_archive(archive)
    finally:
        public.MEETING = original_meeting
    turns = public.turns_from_rttm(data / "IN1014.rttm")
    existing = json.loads((QUALIFICATION / "data/public/manifest.json").read_text())
    clips = [clip for clip in existing["clips"] if clip["id"] in (CLIPS["a-control"], CLIPS["a-hour"])]
    for suffix, first_frame, word_count in (("control", 42296235, 3800), ("hour", 1485910, 10821)):
        path = data / f"ami-IN1014-{suffix}.wav"
        duration = crop_wave(data / "IN1014.Mix-Headset.wav", path, first_frame, 59085910)
        start, end = first_frame / 16000, 59085910 / 16000
        refs, diarization = public.crop(words, start, end), public.crop(turns, start, end)
        if (len(refs) != word_count or len({word["speaker_id"] for word in refs}) != 4
                or {word["speaker_id"] for word in refs} != {turn["speaker_id"] for turn in diarization}
                or any(not 0 <= item["start"] < item["end"] <= duration for item in refs + diarization)):
            raise ValueError("IN1014 reference bounds, counts or speakers differ")
        clips.append({"id": f"ami-IN1014-{suffix}", "audio_path": str(path), "duration": duration,
                      "source_start_frame": first_frame, "source_end_frame": 59085910,
                      "source_start": start, "source_end": end, "private": False,
                      "reference_segments": refs, "reference_turns": diarization,
                      "boundary_words_clipped": sum(word["start"] < start < word["end"] for word in words),
                      "words_per_speaker": dict(Counter(word["speaker_id"] for word in refs))})
    manifest = {"license": existing["license"], "attribution": existing["attribution"],
                "reference_policy": existing["reference_policy"], "excluded_IN1014_entries": excluded,
                "caveats": ["ES2004a hour repeats one meeting. IN1014 is continuous independent content.",
                            "IN1014 is in the RTTM setup training partition. MOSS training overlap is unknown.",
                            "Headset mix does not qualify ambient MacBook audio.",
                            "Each IN1014 crop clips one initial word, retained in scoring."], "clips": clips}
    save(state / "manifest.json", manifest)
    runtime = state / "runtime"
    runtime.mkdir()
    snapshot_fork(fork, runtime)
    shutil.copytree(API / "runtime/omlx", runtime / "omlx", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    code = state / "source"
    code.mkdir()
    for name in ("limit60.py", "limit60_runtime.py", "limit60_score.py", "test_limit60.py",
                 "run.py", "score.py", "qualification.py", "prepare_public.py"):
        shutil.copyfile(SOURCE / name, code / name)
    shutil.copyfile(API / "official-prompt.txt", state / "prompt.txt")
    launcher = state / "omlx-limit60"
    launcher.write_text(f'#!/bin/sh\nexec {PYTHON} -B "$LALIA_LIMIT60_STATE/source/limit60_runtime.py" "$@"\n')
    launcher.chmod(0o700)
    result = subprocess.run([str(PYTHON), "-B", str(code / "limit60_runtime.py"), "--check-imports"],
                            env=environment(state), check=False, capture_output=True, text=True, timeout=60)
    (state / "import-preflight.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Import preflight exited {result.returncode}. See {state / 'import-preflight.log'}")
    imports = json.loads(result.stdout)
    if imports["installed_versions"] != previous["versions"]:
        raise ValueError("Pinned installed dependencies changed")
    save(state / "import-preflight.json", imports)
    historical = API / "native-bf16-api-20261027/baseline/results" / MODEL / "ami-ES2004a-full/result.json"
    commands = {}
    for phase in PHASES:
        models = state / phase / "models"
        models.mkdir(parents=True)
        (models / MODEL).symlink_to((API / "baseline/models" / MODEL).resolve(strict=True), target_is_directory=True)
        command = [str(SCORING_PYTHON), "-B", str(code / "run.py"), "--omlx", str(launcher),
                   "--manifest", str(state / "manifest.json"), "--models", MODEL,
                   "--prompt-file", str(state / "prompt.txt"), "--qualify", "--max-tokens", "49152",
                   "--port", "18001", "--timeout", "600" if phase.endswith("control") else "2400",
                   "--memory-gib", "64", "--min-battery-percent", "25", "--state", str(state / phase),
                   "--clips", CLIPS[phase]]
        if phase == "a-control":
            command.extend(["--baseline", str(historical)])
        elif phase.endswith("hour"):
            control = phase[0] + "-control"
            command.extend(["--baseline", str(state / control / "results" / MODEL / CLIPS[control] / "result.json")])
        commands[phase] = command
    hashes = {**previous["hashes"], str(previous_path): sha256(previous_path), str(historical): sha256(historical)}
    for path in state.rglob("*"):
        if path.is_file() and not path.is_symlink():
            hashes[str(path)] = sha256(path)
    protocol = {"fork_commit": FORK_COMMIT, "model_revision": previous["model_revision"],
                "installed_dependency_versions": previous["versions"],
                "runtime_scope": "Complete fork package overrides installed mlx-audio, not one-variable RoPE A/B",
                "sources": {"IN1014_audio": "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IN1014/audio/IN1014.Mix-Headset.wav",
                            "audio_sha256": sha256(audio), "rttm_sha256": RTTM_SHA,
                            "RTTM": f"https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/{public.SETUP_REVISION}/only_words/rttms/train/IN1014.rttm"},
                "acceptance": {"existing_qualification": True, "a_control_historical_baseline": str(historical),
                               "tail_seconds": 300, "tail_max_regression": 0.05,
                               "b_hour_quarters_must_match_global_speakers": True,
                               "native_eos": 151645, "stop_on_first_failure": True},
                "limits": {"short_seconds": 600, "hour_seconds": 2400, "footprint_gib": 64, "min_battery_percent": 25},
                "commands": commands, "hashes": hashes,
                "power_at_prepare": subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True).stdout}
    save(state / "protocol.json", protocol)
    (state / "protocol.sha256").write_text(sha256(state / "protocol.json") + "\n")
    return {"state": str(state), "protocol_sha256": sha256(state / "protocol.json"),
            "pins": len(hashes), "imports_only": True, "model_loaded": False, "inference_run": False}


def cleanup_evidence(phase):
    processes = {}
    for name in ("runner-process.json", "server-process.json", "awake-process.json"):
        path = phase / name
        if path.exists():
            pid = json.loads(path.read_text())["pid"]
            try:
                os.kill(pid, 0)
                processes[name] = {"pid": pid, "gone": False}
            except ProcessLookupError:
                processes[name] = {"pid": pid, "gone": True}
    with socket.socket() as connection:
        connection.settimeout(1)
        closed = connection.connect_ex(("127.0.0.1", 18001)) != 0
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        production_status = response.status
    return {"processes": processes, "port_closed": closed, "production_http_status": production_status,
            "complete": closed and bool(processes) and all(row["gone"] for row in processes.values())
            and production_status == 200}


def run_phase(state, phase):
    protocol_path = state / "protocol.json"
    if sha256(protocol_path) != (state / "protocol.sha256").read_text().strip():
        raise ValueError("Protocol changed")
    protocol = json.loads(protocol_path.read_text())
    verify(protocol["hashes"])
    for earlier in PHASES[:PHASES.index(phase)]:
        if json.loads((state / earlier / "evaluation.json").read_text())["accepted"] is not True:
            raise ValueError("Previous phase did not pass. No later request allowed")
    directory = state / phase
    with (directory / "started.json").open("x") as marker:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
            production_status = response.status
        json.dump({"power": subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True).stdout,
                   "production_http_status": production_status}, marker)
    with (directory / "runner.log").open("w") as log:
        interrupted = None

        def interrupt(number, _):
            nonlocal interrupted
            if interrupted is None:
                interrupted = number

        handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
        try:
            for number in handlers:
                signal.signal(number, interrupt)
            process = subprocess.Popen(protocol["commands"][phase], env=environment(state), stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while interrupted is None:
                try:
                    process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if interrupted is not None:
                process.send_signal(interrupted)
                process.wait()
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
    cleanup = cleanup_evidence(directory)
    save(directory / "cleanup.json", cleanup)
    evaluation = {"accepted": False, "failures": [], "runner_returncode": process.returncode, "cleanup": cleanup}
    try:
        verify(protocol["hashes"])
        if interrupted is not None:
            raise ValueError(f"Runner interrupted by {signal.Signals(interrupted).name}")
        if process.returncode or not cleanup["complete"]:
            raise ValueError("Runner or cleanup failed")
        result_dir = directory / "results" / MODEL / CLIPS[phase]
        result = json.loads((result_dir / "result.json").read_text())
        response = json.loads((result_dir / "response.json").read_text())
        rows = [json.loads(line.removeprefix("LALIA_LIMIT60_AUDIT "))
                for line in (directory / "server" / MODEL / "server.log").read_text().splitlines()
                if line.startswith("LALIA_LIMIT60_AUDIT ")]
        if one_audit(rows, "imports")["imports"] != json.loads((state / "import-preflight.json").read_text())["modules"]:
            raise ValueError("Runtime imports differ from preflight")
        evaluation["native_finish"] = native_finish(rows, response)
        if result["status"] != "complete" or result["qualification"]["accepted"] is not True:
            raise ValueError("Existing completion or quality qualification rejected the request")
        clips = {clip["id"]: clip for clip in json.loads((state / "manifest.json").read_text())["clips"]}
        clip = clips[CLIPS[phase]]
        evaluation["reference_words_per_minute"] = len(clip["reference_segments"]) / (clip["duration"] / 60)
        evaluation["tail"] = tail_metrics(clip, response)
        if phase.endswith("hour"):
            control = json.loads((state / (phase[0] + "-control") / "evaluation.json").read_text())
            comparison = compare_tail(evaluation["tail"]["metrics"], control["tail"]["metrics"])
            evaluation["tail_comparison"] = comparison
            evaluation["failures"].extend(comparison["failures"])
        if phase == "b-hour":
            continuity = quarter_continuity(clip, response)
            evaluation["continuity"] = continuity
            if any(not row["global_assignment_agrees"] for row in continuity["quarters"]):
                evaluation["failures"].append("Speaker assignment changed across quarters")
        evaluation["accepted"] = not evaluation["failures"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        evaluation["failures"].append(str(error))
    save(directory / "evaluation.json", evaluation)
    return evaluation


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--fork", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--rttm", type=Path)
    parser.add_argument("--phase", choices=PHASES)
    args = parser.parse_args()
    state = args.state.expanduser().resolve()
    if args.mode == "prepare":
        if None in (args.fork, args.audio, args.rttm):
            parser.error("prepare requires --fork, --audio and --rttm")
        print(json.dumps(prepare(state, args.fork, args.audio, args.rttm)))
    else:
        if args.phase is None:
            parser.error("run requires --phase")
        result = run_phase(state, args.phase)
        print(json.dumps({"phase": args.phase, "accepted": result["accepted"], "failures": result["failures"]}))
        if not result["accepted"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
