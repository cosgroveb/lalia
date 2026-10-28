"""Verify staged inputs and the official WAV loader without loading model weights."""
import importlib.metadata
import json
import os
from pathlib import Path

from official_transcribe import sha256, verify_inputs

inputs = Path("/inputs")
manifest, prompt = verify_inputs(inputs)
from transformers import AutoProcessor
from inference_utils import load_audio_item

processor = AutoProcessor.from_pretrained(inputs / "model", trust_remote_code=True,
                                         local_files_only=True, fix_mistral_regex=True)
audio = load_audio_item(str(inputs / "public90.wav"), processor.feature_extractor.sampling_rate)
if audio.shape != (86400000,) or str(audio.dtype) != "float32":
    raise ValueError("Official WAV loader did not return the expected public waveform")
print(json.dumps({"status": "ready", "audio_shape": list(audio.shape), "audio_dtype": str(audio.dtype),
                  "image": os.environ["FRONTEND_IMAGE_ID"], "manifest_sha256": sha256(inputs / "manifest.json"),
                  "model_revision": manifest["model_revision"],
                  "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "librosa", "soundfile", "soxr", "numba")}}))
