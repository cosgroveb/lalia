"""Download pinned experiment weights without changing the normal oMLX model directory."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "VibeVoice-ASR-bf16": (
        "mlx-community/VibeVoice-ASR-bf16",
        "12076ff8cb141fcb672abc9f8957b08aab5ecf94",
    ),
    "MOSS-Transcribe-Diarize": (
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "704aa4a9c304e8520be88901e0d1960158ef5b15",
    ),
}


def download_tokenizer(directory):
    # This conversion omits the tokenizer used by mlx-audio's Qwen2.5 fallback.
    snapshot_download(
        "Qwen/Qwen2.5-7B", revision="d149729398750b98c0af14eb82c78cfe92750796",
        local_dir=directory / "VibeVoice-ASR-bf16",
        allow_patterns=["tokenizer*.json", "vocab.json", "merges.txt"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if "VibeVoice-ASR-bf16" in args.models:
        download_tokenizer(args.directory)
    for name, (repo, revision) in MODELS.items():
        if name not in args.models:
            continue
        print(f"Downloading {repo} at {revision}", flush=True)
        snapshot_download(
            repo, revision=revision, local_dir=args.directory / name,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
            max_workers=4,
        )
        print(f"Ready: {name}", flush=True)
    (args.directory / "revisions.json").write_text(json.dumps(MODELS, indent=2) + "\n")


if __name__ == "__main__":
    main()
