"""Freeze a fresh local run from pinned assets; no download, build, or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parent
WEIGHTS = "model/model-00000-of-00001.safetensors"
BUILD_FILES = ("Dockerfile", ".dockerignore", "requirements.lock", "requirements.in",
               "constraints.txt", "verify_environment.py", "build-image.sh", "preflight.sh")
PROBE_FILES = ("official_transcribe.py", "official_preflight.py", "official_supervise.py", "official_kernel_preflight.py",
               "supervise.py", "smoke_numeric.py")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root, names, output):
    (root / output).write_text("".join(f"{sha256(root / name)}  {name}\n" for name in sorted(names)))


def prepare(state, inputs, audio, helper, references):
    state = state.resolve()
    if state.is_relative_to(SOURCE.parents[3]):
        raise ValueError("Keep run state and assets outside the repository")
    if state.exists():
        raise ValueError("State already exists. No automatic regeneration")
    manifest = json.loads((SOURCE / "inputs-manifest.json").read_text())
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    assets = {name: audio if name == "public90.wav" else inputs / name
              for name in manifest["files"] if name != WEIGHTS}
    expected = {**{f"inputs/{name}": digest for name, digest in manifest["files"].items() if name != WEIGHTS},
                "probe/inference_utils.py": manifest["helper_sha256"],
                "scoring/public-manifest.json": provenance["reference_manifest_sha256"]}
    files = {f"inputs/{name}": path for name, path in assets.items()}
    files.update({"probe/inference_utils.py": helper, "scoring/public-manifest.json": references})
    for name, path in files.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Pinned input differs: {name}")
    files.update({name: SOURCE / name for name in BUILD_FILES})
    files.update({f"probe/{name}": SOURCE.parent / name for name in PROBE_FILES})
    files.update({f"scoring/{name}": SOURCE.parents[1] / name for name in ("score.py", "qualification.py")})
    files.update({name: SOURCE / name for name in ("score_result.py", "stage.sh", "prepare.py", "provenance.json")})
    files["inputs/manifest.json"] = SOURCE / "inputs-manifest.json"
    state.mkdir(parents=True, mode=0o700)
    for name, path in files.items():
        destination = state / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if name in expected and sha256(destination) != expected[name]:
            raise ValueError(f"Pinned input changed during preparation: {name}")
    scoring_names = ("score_result.py", "scoring/score.py", "scoring/qualification.py", "scoring/public-manifest.json")
    (state / "scoring-manifest.json").write_text(json.dumps({
        "files": {name: sha256(state / name) for name in scoring_names},
        "reference_assets": "Word segments and diarization turns are embedded; no external reference file is read."
    }, indent=2) + "\n")
    remote = list(BUILD_FILES) + ["provenance.json"]
    remote += [name for name in files if name.startswith(("probe/", "inputs/"))]
    inventory(state, remote, "source.sha256")
    inventory(state, [*files, "scoring-manifest.json", "source.sha256"], "local-source.sha256")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path, help="New state directory outside the repository")
    parser.add_argument("--inputs", type=Path, required=True, help="Pinned model/ directory and prompt.txt")
    parser.add_argument("--audio", type=Path, required=True, help="Public 90-minute AMI WAV")
    parser.add_argument("--helper", type=Path, required=True, help="Official inference_utils.py from provenance.json")
    parser.add_argument("--references", type=Path, required=True, help="Frozen local public reference manifest")
    args = parser.parse_args()
    print(prepare(args.state, args.inputs, args.audio, args.helper, args.references))


if __name__ == "__main__":
    main()
