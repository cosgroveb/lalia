"""Bound private-runtime transcription without treating quiet audio as silence."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import tempfile
import threading
import time


class TranscriptionControl:
    def __init__(self, duration, max_tokens=None, timeout=None):
        if not math.isfinite(duration) or not 0 < duration <= 5400:
            raise ValueError("Audio duration must be between zero and 90 minutes.")
        self.max_tokens = min(49152, 256 + math.ceil(32 * duration))
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValueError("Invalid transcription token budget.")
            self.max_tokens = min(self.max_tokens, max_tokens)
        seconds = min(2400, 30 + 2 * duration)
        if timeout is not None:
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Invalid transcription deadline.")
            seconds = min(seconds, timeout)
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.cancelled = threading.Event()
        self.duration = duration
        self.stop_reason = None
        self.generation_state = {}

    def limit_duration(self, duration):
        limits = TranscriptionControl(duration, self.max_tokens)
        self.duration = duration
        self.max_tokens = limits.max_tokens
        self.deadline = min(self.deadline, self.started + min(2400, 30 + 2 * duration))
        self.check()

    def check(self):
        if self.cancelled.is_set():
            self.stop_reason = "cancelled"
            raise RuntimeError("Transcription cancelled. Audio is retained. Retry when ready.")
        if time.monotonic() >= self.deadline:
            self.stop_reason = "timeout"
            raise TimeoutError("Transcription exceeded its time limit. Audio is retained. Retry when ready.")

    def record_failure(self, audio_path, model, tokens, tokenizer, reason, error):
        """Retain decoder evidence locally, including after client disconnect."""
        temporary = None
        try:
            diagnostic = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": model, "duration_seconds": self.duration,
                "stop_reason": self.stop_reason or reason, "error": str(error),
                "maximum_tokens": self.max_tokens, "output_tokens": len(tokens),
                "token_ids": tokens, "text": None,
                "elapsed_seconds": time.monotonic() - self.started,
            }
            try:
                diagnostic["text"] = tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception as decode_error:
                diagnostic["decode_error"] = str(decode_error)
            with open(audio_path, "rb") as audio:
                diagnostic["audio_sha256"] = hashlib.file_digest(audio, "sha256").hexdigest()
            directory = Path.home() / ".local/share/lalia/runtime/diagnostics"
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix="transcription-", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                json.dump(diagnostic, output, ensure_ascii=False)
                output.write("\n")
            saved = temporary.with_suffix(".json")
            temporary.replace(saved)
            temporary = None
            logging.getLogger(__name__).warning("Transcription diagnostic saved to %s", saved)
            return saved
        except Exception as save_error:
            logging.getLogger(__name__).error("Could not save transcription diagnostic: %s", save_error)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logging.getLogger(__name__).error("Could not remove incomplete diagnostic: %s", cleanup_error)


async def finish_worker(worker, control):
    """Keep resources owned until the executor has actually stopped."""
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        control.cancelled.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled():
            worker.exception()
        raise


async def guarded_transcription(request, engine, path, kwargs, timeout):
    # Queueing and decode count against the deadline. The worker tightens both
    # limits from decoded sample count before calling the model.
    control = TranscriptionControl(5400, kwargs.get("max_tokens"), timeout)
    kwargs = {**kwargs, "max_tokens": control.max_tokens, "_lalia_control": control}

    async def watch_disconnect():
        while True:
            if await request.is_disconnected():
                control.cancelled.set()
                return
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(watch_disconnect())
    try:
        return await engine.transcribe(path, **kwargs)
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
