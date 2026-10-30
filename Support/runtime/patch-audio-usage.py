#!/usr/bin/env python3
"""Expose STT token counts and named prompt hooks in the pinned oMLX runtime."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


VERSIONS = {"omlx": "0.7.2", "mlx-audio": "0.6.2"}
ORIGINALS = {
    "omlx/api/audio_models.py": "6b536079069f0881ae247f2805fed9b25b4f311c8b07bba3b67ef8bcd6f6e518",
    "omlx/api/audio_routes.py": "69cab91bcc0bb1ee18063344da7a915a53383aa2a86a3a6efb1883a6cf0c26bf",
    "omlx/engine/stt.py": "49b4f08ab94d0b7fd11dcef4c81c34f19af897e2bdb96085c25800e7e7c1279e",
    "mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py": "91a44abfbd1640aa1b3fa724013ec60c2ecfb89d2eda09801117e058ff7a55c2",
    "mlx_audio/lm/generate.py": "69bd45c9f1445b23220e99857febd08cf96289a35f34e8fb20c40e953558d758",
}
REPLACEMENTS = {
    "omlx/api/audio_models.py": [(
        "    segments: Optional[List[dict]] = None\n",
        "    segments: Optional[List[dict]] = None\n    usage: Optional[dict] = None\n",
    )],
    "omlx/api/audio_routes.py": [(
        '@router.post("/v1/audio/transcriptions", response_model=AudioTranscriptionResponse)',
        '@router.post("/v1/audio/transcriptions", response_model=AudioTranscriptionResponse, response_model_exclude_unset=True)',
    ), (
        "        segments=segments,\n    )\n",
        '        segments=segments,\n        **({"usage": result["usage"]} if result.get("usage") is not None else {}),\n    )\n',
    )],
    "omlx/engine/stt.py": [(
        '    if "initial_prompt" in params:\n        return {"initial_prompt": prompt}\n',
        '    if "initial_prompt" in params:\n        return {"initial_prompt": prompt}\n'
        '    if "prompt" in params:\n        return {"prompt": prompt}\n',
    ), (
        '    soft prior (~224-token window). Backends with neither hook ignore the\n',
        '    soft prior (~224-token window). MOSS exposes a named ``prompt`` hook.\n'
        '    Backends with none of these hooks ignore the\n',
    ), (
        '                return {\n                    "text": result.text or "",',
        '''                usage = None
                counts = [getattr(result, name, None) for name in (
                    "prompt_tokens", "generation_tokens", "total_tokens"
                )]
                if (all(type(value) is int for value in counts)
                        and counts[0] > 0 and counts[1] >= 0
                        and counts[2] == counts[0] + counts[1]):
                    usage = {
                        "type": "tokens", "input_tokens": counts[0],
                        "output_tokens": counts[1], "total_tokens": counts[2],
                    }

                return {
                    **({"usage": usage} if usage is not None else {}),
                    "text": result.text or "",''',
    )],
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patched(original, name):
    text = original.decode()
    for before, after in REPLACEMENTS[name]:
        if text.count(before) != 1:
            raise ValueError(f"Unexpected patch context: {name}")
        text = text.replace(before, after)
    compile(text, name, "exec")
    return text.encode()


def validate_imports(root):
    code = """
from omlx.api.audio_models import AudioTranscriptionResponse
from omlx.api.audio_routes import router
from omlx.engine.stt import STTEngine
from mlx_audio.stt.models.moss_transcribe_diarize import Model
response = AudioTranscriptionResponse(text='नमस्ते 🫠', segments=[], usage={
    'type': 'tokens', 'input_tokens': 2, 'output_tokens': 0, 'total_tokens': 2})
assert response.model_dump()['usage']['total_tokens'] == 2
assert response.text == 'नमस्ते 🫠'
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"})


def apply(root, backup, rollback=False, check=False):
    if backup.resolve().is_relative_to(root.resolve()):
        raise ValueError("Keep original runtime backups outside the package directory")
    for name, version in VERSIONS.items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise ValueError(f"Unsupported {name} {actual}, expected {version}")
    contents = {name: (root / name).read_bytes() for name in ORIGINALS}
    originals = {}
    targets = {}
    for name, checksum in ORIGINALS.items():
        current = contents[name]
        if name not in REPLACEMENTS:
            if digest(current) != checksum:
                raise ValueError(f"Refusing unknown runtime source: {name}")
            continue
        if digest(current) == checksum:
            original = current
        else:
            saved = backup / name
            if not saved.is_file() or digest(saved.read_bytes()) != checksum:
                raise ValueError(f"Refusing unknown source or missing original backup: {name}")
            original = saved.read_bytes()
            if current != patched(original, name):
                raise ValueError(f"Refusing unknown patched source: {name}")
        originals[name] = original
        targets[name] = original if rollback else patched(original, name)
    changed = [name for name in targets if contents[name] != targets[name]]
    if check:
        print(json.dumps({"changed_files": changed, "patched": not changed and not rollback}))
        return bool(changed)
    if not changed:
        print("oMLX audio usage already " + ("restored" if rollback else "patched"))
        return False
    backup.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, original in originals.items():
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if saved.exists() and saved.read_bytes() != original:
            raise ValueError(f"Refusing to replace an unexpected backup: {saved}")
        if not saved.exists():
            saved.write_bytes(original)
            saved.chmod(0o600)
    manifest = {name: {"original": digest(originals[name]), "patched": digest(patched(originals[name], name))}
                for name in originals}
    (backup / "manifest.json").write_text(json.dumps({"versions": VERSIONS, "files": manifest}, indent=2) + "\n")
    # Validate an entire staged package before replacing any installed source.
    with tempfile.TemporaryDirectory(prefix="omlx-usage-") as temporary:
        stage = Path(temporary)
        shutil.copytree(root / "omlx", stage / "omlx", ignore=shutil.ignore_patterns("__pycache__"))
        for name, content in targets.items():
            (stage / name).write_bytes(content)
        if not rollback:
            validate_imports(stage)
        activated = []
        try:
            for name in changed:
                target = root / name
                if target.read_bytes() != contents[name]:
                    raise ValueError(f"Runtime changed during patch: {name}")
                staged = target.with_suffix(".py.lalia-usage")
                staged.write_bytes(targets[name])
                staged.chmod(target.stat().st_mode & 0o777)
                staged.replace(target)
                activated.append(name)
        except BaseException:
            for name in reversed(activated):
                target = root / name
                if target.read_bytes() == targets[name]:
                    target.write_bytes(contents[name])
            raise
    print("oMLX audio usage " + ("restored" if rollback else "patched"))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True,
                        help="Site-packages directory inside the private Lalia runtime")
    parser.add_argument("--backup-dir", type=Path,
                        default=Path.home() / ".local/share/lalia/runtime/patch-backup")
    parser.add_argument("--rollback", action="store_true", help="Restore recognized originals")
    parser.add_argument("--check", action="store_true", help="Validate source without changing files")
    args = parser.parse_args()
    os.umask(0o077)
    apply(args.site_packages.resolve(), args.backup_dir, args.rollback, args.check)


if __name__ == "__main__":
    main()
