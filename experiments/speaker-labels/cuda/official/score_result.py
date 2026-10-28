"""Score the saved public response locally, retaining native finish evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_scoring_inputs(root):
    manifest_path = root / "scoring-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {"score_result.py", "scoring/score.py", "scoring/qualification.py", "scoring/public-manifest.json"}
    if set(manifest["files"]) != required:
        raise ValueError("Frozen scoring inventory is incomplete or unexpected")
    for name, digest in manifest["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen scoring input changed: {name}")
    return {"reference_manifest_sha256": manifest["files"]["scoring/public-manifest.json"],
            "scoring_manifest_sha256": sha256(manifest_path), "scoring_files": manifest["files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    provenance = verify_scoring_inputs(root)
    sys.path.insert(0, str(root / "scoring"))
    from score import score
    from qualification import completion, reference_timing

    clip = next(c for c in json.loads((root / "scoring/public-manifest.json").read_text())["clips"]
                if c["id"] == "ami-ES2004a-stress-90m")
    response = json.loads((args.results / "response.json").read_text())
    native = json.loads((args.results / "result.json").read_text())
    result = {**provenance, "native": native, "metrics": score(clip, response, expect_speakers=True),
              "reference_timing": reference_timing(clip, response)}
    try:
        result["raw_structure"] = completion(response, clip["duration"], 49152)
        # qualification.completion infers EOS from a sub-limit count. Use actual IDs instead.
        result["raw_structure"]["eos"] = native["eos_observed"]
    except ValueError as error:
        result["raw_structure_error"] = str(error)
    path = args.results / "public-score.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "native_finish_reason": native["native_finish_reason"],
                      "reference_manifest_sha256": provenance["reference_manifest_sha256"],
                      "last_speech_end": result["metrics"]["schema"]["last_speech_end"],
                      "reference_last_word_end": result["metrics"]["schema"]["reference_last_word_end"]}))


if __name__ == "__main__":
    main()
