"""Freeze independent exact-token captures without starting model inference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from trace_runtime import BASE, PHASES, TRACE_STATES, save, sha256


def prepare(state):
    state = state.resolve()
    if state.parent != TRACE_STATES.resolve():
        raise ValueError("Require a fresh state under exact-traces")
    if state.exists():
        raise ValueError("State already exists. No automatic rerun.")
    previous = BASE / "native-bf16-api-20261027"
    if sha256(previous / "protocol.json") != "a48c761b408eb2ffc484df99a343b139fb2ff86e609f4f730f06531424865dd9":
        raise ValueError("Previous protocol differs")
    inherited = json.loads((previous / "protocol.json").read_text())
    hashes = dict(inherited["hashes"])
    for name, expected in hashes.items():
        if sha256(name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    source = Path(__file__).resolve().parent
    paths = [previous / "protocol.json", previous / "protocol.sha256"]
    paths.extend(source / name for name in ("trace_runtime.py", "prepare_trace.py", "test_trace_runtime.py",
                                           "omlx-exact-trace", "eos_rope.py", "short_windows.py", "window_matching.py"))
    expected = {}
    commands = {}
    model = BASE / "baseline/models/MOSS-Transcribe-Diarize"
    for phase, candidate in PHASES.items():
        historical = previous if candidate else BASE
        result_dir = historical / "stress90/results/MOSS-Transcribe-Diarize/ami-ES2004a-stress-90m"
        response_path = result_dir / "response.json"
        response = json.loads(response_path.read_text())
        log_path = historical / "stress90/server/MOSS-Transcribe-Diarize/server.log"
        endings = [json.loads(line.removeprefix("LALIA_API_AUDIT ")) for line in log_path.read_text().splitlines()
                   if line.startswith("LALIA_API_AUDIT ") and "last_token_id" in line]
        if (len(endings) != 1 or endings[0]["last_token_id"] != 151645
                or endings[0]["yielded_tokens"] != response["usage"]["output_tokens"] + 1):
            raise ValueError("Historical terminal token audit differs")
        expected[phase] = {"text_sha256": hashlib.sha256(response["text"].encode()).hexdigest(),
                           "output_tokens": response["usage"]["output_tokens"], "eos_id": 151645}
        baseline = historical / "baseline/results/MOSS-Transcribe-Diarize/ami-ES2004a-full/result.json"
        paths.extend([response_path, result_dir / "result.json", baseline, log_path])
        command = list(inherited["commands"]["stress90"])
        for flag, value in (("--state", str(state / phase)), ("--omlx", str(source / "omlx-exact-trace")),
                            ("--baseline", str(baseline))):
            command[command.index(flag) + 1] = value
        commands[phase] = ["env", f"LALIA_EXACT_TRACE_STATE={state}", *command]
    hashes.update({str(path): sha256(path) for path in paths})
    protocol = {"scope": "Serial exact-token reproductions, each started explicitly by the coordinator",
                "phases": PHASES, "expected": expected, "commands": commands,
                "model_revision": inherited["model_revision"], "versions": inherited["versions"],
                "prompt_tokens": 71687, "embedding_width": 1024,
                "max_tokens": 49152, "prefill_tokens": 4096, "eos_override": False,
                "limits": inherited["limits"], "hashes": hashes,
                "power_at_prepare": subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True).stdout}
    state.mkdir(parents=True, mode=0o700)
    for phase in PHASES:
        directory = state / phase / "models"
        directory.mkdir(parents=True, mode=0o700)
        (directory / "MOSS-Transcribe-Diarize").symlink_to(model.resolve(strict=True), target_is_directory=True)
    save(state / "protocol.json", protocol)
    (state / "protocol.sha256").write_text(sha256(state / "protocol.json") + "\n")
    return {"state": str(state), "pins": len(hashes), "protocol_sha256": sha256(state / "protocol.json"),
            "inference_run": False}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True,
                        help="Fresh run directory under ~/.local/state/lalia-speaker-qualification/exact-traces")
    args = parser.parse_args()
    print(json.dumps(prepare(args.state)))


if __name__ == "__main__":
    main()
