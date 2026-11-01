"""Bound private-runtime transcription without treating quiet audio as silence."""

import asyncio
import math
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

    def limit_duration(self, duration):
        limits = TranscriptionControl(duration, self.max_tokens)
        self.max_tokens = limits.max_tokens
        self.deadline = min(self.deadline, self.started + min(2400, 30 + 2 * duration))
        self.check()

    def check(self):
        if self.cancelled.is_set():
            raise RuntimeError("Transcription cancelled. Audio is retained. Retry when ready.")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Transcription exceeded its time limit. Audio is retained. Retry when ready.")


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
