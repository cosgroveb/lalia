#!/usr/bin/env python3
"""One official MOSS audio-to-transcript run, retaining native token evidence."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import struct
import time
import wave

GIB = 1024**3
MAX_NEW_TOKENS = 49152
CONTEXT = 131072
PROMPT_SHA = "5696a90762c9a738598bdf62530952cb9058f0bd2da8dccc23d988b68990e22a"
ROPE_SHA = "fb39dde72a24bd6b8b3573c5e6375d3f163568c8649e0e97f9fd27ef52f0a493"
SEGMENT = re.compile(r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\]"
                     r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]", re.DOTALL)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def resource_preflight(cuda, out):
    free, total = cuda.mem_get_info()
    host = {line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemAvailable:", "MemFree:"))}
    resources = {"device": cuda.get_device_name(0),
                 "host_mem_available_bytes": host["MemAvailable"], "host_mem_free_bytes": host["MemFree"],
                 "cuda_free_bytes": free, "cuda_total_bytes": total,
                 "cgroup_memory_max_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text()),
                 "cgroup_swap_max_bytes": int(Path("/sys/fs/cgroup/memory.swap.max").read_text())}
    write_json(out / "resource-preflight.json", resources)
    if resources["device"] != "NVIDIA GB10":
        raise ValueError(f"Expected NVIDIA GB10 UMA: {json.dumps(resources)}")
    # GB10 cudaMemGetInfo excludes reclaimable host cache. Swap is not admission capacity.
    if host["MemAvailable"] < 72 * GIB:
        raise MemoryError(f"Need 72 GiB host MemAvailable before loading: {json.dumps(resources)}")


def verify_inputs(inputs):
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(inputs / name) != digest:
            raise ValueError(f"Input hash mismatch: {name}")
    if sha256(Path(__file__).with_name("inference_utils.py")) != manifest["helper_sha256"]:
        raise ValueError("Official inference helper hash differs")
    prompt = (inputs / "prompt.txt").read_bytes()
    if len(prompt) != 222 or hashlib.sha256(prompt).hexdigest() != PROMPT_SHA:
        raise ValueError("Prompt bytes differ")
    with wave.open(str(inputs / "public90.wav")) as audio:
        if (audio.getnchannels(), audio.getframerate(), audio.getnframes()) != (1, 16000, 86400000):
            raise ValueError("Expected mono 16 kHz 90-minute public fixture")
    verify_checkpoint_layout(inputs / "model")
    return manifest, prompt.decode("utf-8")


def verify_checkpoint_layout(model):
    shard = "model-00000-of-00001.safetensors"
    with (model / shard).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 4 * 1024**2:
            raise ValueError("Unexpected checkpoint header size")
        header = json.loads(stream.read(length))
    weights = {key: value for key, value in header.items() if key != "__metadata__"}
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    if set(weights) != set(index) or set(index.values()) != {shard}:
        raise ValueError("Checkpoint tensor keys differ from the official index")
    expected = {"model.whisper_encoder.conv1.weight": [1024, 80, 3],
                "model.whisper_encoder.conv2.weight": [1024, 1024, 3]}
    for key, shape in expected.items():
        if weights[key]["shape"] != shape:
            raise ValueError(f"Expected Torch Conv1d checkpoint layout: {key}")


def finish_evidence(ids, eos_id, limit=MAX_NEW_TOKENS):
    if not ids or len(ids) > limit or eos_id != 151645:
        raise ValueError("Invalid generated IDs or changed native EOS")
    positions = [index for index, token in enumerate(ids) if token == eos_id]
    if positions and positions != [len(ids) - 1]:
        raise ValueError("Native EOS precedes final generated token")
    eos = ids[-1] == eos_id
    return {"generated_tokens": len(ids), "last_token_id": ids[-1], "eos_token_id": eos_id,
            "eos_observed": eos, "native_finish_reason": "eos" if eos else "length" if len(ids) == limit else "unknown",
            "token_limit_reached": len(ids) == limit, "max_new_tokens": limit}


class TokenObserver:
    def __init__(self, streamer, prompt, path):
        self.streamer, self.prompt, self.path = streamer, prompt, path
        self.ids = []
        self.seen_prompt = self.ended = False

    def put(self, value):
        if self.ended or (self.ids and self.ids[-1] == 151645):
            raise ValueError("Streamer received tokens after EOS or end")
        if value.device.type != "cpu":
            raise ValueError("Expected already-selected CPU streamer tokens")
        tokens = value.reshape(-1).tolist()
        if not self.seen_prompt:
            if tuple(value.shape) != (1, len(self.prompt)) or tokens != self.prompt:
                raise ValueError("Streamed prompt differs from generation input")
            self.seen_prompt = True
        else:
            if tuple(value.shape) != (1,) or len(tokens) != 1 or type(tokens[0]) is not int:
                raise ValueError("Expected one selected token per greedy generation step")
            if len(self.ids) == MAX_NEW_TOKENS:
                raise ValueError("Streamed IDs exceed the token ceiling")
            self.ids.extend(tokens)
            if len(self.ids) % 128 == 0 or tokens[0] == 151645:
                self.checkpoint()
        return self.streamer.put(value)

    def end(self):
        result = self.streamer.end()
        self.ended = True
        self.checkpoint()
        return result

    def checkpoint(self):
        # Only the returned generation sequence can establish a final result.
        write_json(self.path, {"status": "incomplete", "generated_tokens_lower_bound": len(self.ids),
                               "generated_ids": self.ids})

    def verify(self, ids):
        if not self.seen_prompt or not self.ended or self.ids != ids:
            raise ValueError("Returned generation differs from ended streamed IDs")


def response_from_text(text, prompt_len, generated_tokens):
    matches = list(SEGMENT.finditer(text))
    segments = [{"start": float(m["start"]), "end": float(m["end"]),
                 "speaker_id": m["speaker"], "text": m["text"].strip()} for m in matches]
    cursor, unmatched = 0, 0
    for match in matches:
        unmatched += len(text[cursor:match.start()].strip())
        cursor = match.end()
    unmatched += len(text[cursor:].strip())
    return {"text": text, "segments": segments,
            "usage": {"type": "tokens", "input_tokens": prompt_len,
                      "output_tokens": generated_tokens, "total_tokens": prompt_len + generated_tokens}}, unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("/inputs"))
    parser.add_argument("--out", type=Path, default=Path("/results/model"))
    args = parser.parse_args()
    args.out.mkdir(mode=0o700)
    began = time.monotonic()
    manifest, prompt = verify_inputs(args.inputs)
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from inference_utils import build_transcription_messages, generate_transcription

    if importlib.metadata.version("torch") != "2.14.1+cu130" or importlib.metadata.version("transformers") != "5.17.0":
        raise ValueError("Pinned Torch/Transformers versions differ")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one CUDA device")
    resource_preflight(torch.cuda, args.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(64 * GIB / torch.cuda.get_device_properties(0).total_memory)
    write_json(args.out / "started.json", {"model_revision": manifest["model_revision"],
               "helper_revision": manifest["helper_revision"], "prompt_sha256": PROMPT_SHA,
               "audio_sha256": manifest["files"]["public90.wav"], "max_new_tokens": MAX_NEW_TOKENS,
               "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "librosa", "soundfile", "soxr")}})
    model = AutoModelForCausalLM.from_pretrained(
        args.inputs / "model", trust_remote_code=True, local_files_only=True,
        dtype="auto", attn_implementation="sdpa",
    ).to(dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.inputs / "model", trust_remote_code=True,
                                              local_files_only=True, fix_mistral_regex=True)
    rotary = model.model.language_model.rotary_emb
    rope_hash = hashlib.sha256(rotary.inv_freq.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if rotary.inv_freq.dtype != torch.bfloat16 or rope_hash != ROPE_SHA:
        raise ValueError("Published BF16 rotary frequency cast differs")
    original_generate = model.generate
    observed = {}

    def record_generate(**kwargs):
        if observed:
            raise RuntimeError("A second generation attempt is forbidden")
        config = kwargs["generation_config"]
        if config.eos_token_id != 151645 or config.do_sample or config.max_new_tokens != MAX_NEW_TOKENS:
            raise ValueError("Native greedy generation configuration differs")
        input_ids = kwargs["input_ids"][0].tolist()
        if len(input_ids) + MAX_NEW_TOKENS > CONTEXT:
            raise ValueError("Input plus generation allowance exceeds context")
        observed["input_tokens"] = len(input_ids)
        write_json(args.out / "input-ids.json", input_ids)
        write_json(args.out / "generation-config.json", config.to_dict())
        observer = TokenObserver(kwargs["streamer"], input_ids, args.out / "partial-generated-ids.json")
        kwargs["streamer"] = observer
        outputs = original_generate(**kwargs)
        ids = outputs[0, len(input_ids):].tolist()
        observer.verify(ids)
        write_json(args.out / "generated-ids.json", ids)
        observed.update(finish_evidence(ids, config.eos_token_id))
        (args.out / "decoded-with-special-tokens.txt").write_text(
            processor.tokenizer.decode(ids, skip_special_tokens=False))
        return outputs

    model.generate = record_generate

    def progress(count):
        if count % 128:
            return
        row = {"generated_tokens": count, "seconds": time.monotonic() - began,
               "available_bytes": available_bytes(), "cuda_allocated": torch.cuda.memory_allocated(),
               "cuda_peak": torch.cuda.max_memory_allocated()}
        write_json(args.out / "progress.json", row)
        if row["available_bytes"] < 8 * GIB:
            raise MemoryError("MemAvailable below 8 GiB")

    result = generate_transcription(
        model, processor, build_transcription_messages(args.inputs / "public90.wav", prompt),
        max_length=CONTEXT, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        device=torch.device("cuda:0"), dtype=torch.bfloat16, token_callback=progress,
    )
    torch.cuda.synchronize()
    write_json(args.out / "official-result.json", result)
    (args.out / "raw-transcript.txt").write_text(result["text"])
    if result["prompt_len"] != observed["input_tokens"] or result["generated_tokens"] != observed["generated_tokens"]:
        raise ValueError("Official helper counts differ from actual IDs")
    response, unmatched = response_from_text(result["text"], result["prompt_len"], result["generated_tokens"])
    write_json(args.out / "response.json", response)
    report = {"status": "generated", **observed, "unmatched_raw_characters": unmatched,
              "last_segment_end": max((s["end"] for s in response["segments"]), default=None),
              "segment_count": len(response["segments"]), "rope_dtype": str(rotary.inv_freq.dtype),
              "rope_sha256": rope_hash, "seconds": time.monotonic() - began,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              "generated_ids_sha256": sha256(args.out / "generated-ids.json"),
              "response_sha256": sha256(args.out / "response.json")}
    write_json(args.out / "result.json", report)
    print(json.dumps(report))
    if report["native_finish_reason"] == "unknown":
        raise RuntimeError("Generation returned without native EOS or token limit")


if __name__ == "__main__":
    main()
