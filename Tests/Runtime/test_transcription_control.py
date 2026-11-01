import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import threading
import time
import unittest

SPEC = importlib.util.spec_from_file_location("control", Path(__file__).parents[2] / "Support/runtime/transcription_control.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class BudgetTests(unittest.TestCase):
    def test_short_and_long_budgets_preserve_lower_explicit_limits(self):
        self.assertEqual(control.TranscriptionControl(1.85875).max_tokens, 316)
        self.assertEqual(control.TranscriptionControl(3600).max_tokens, 49152)
        self.assertEqual(control.TranscriptionControl(5400).max_tokens, 49152)
        self.assertEqual(control.TranscriptionControl(2, 128).max_tokens, 128)
        self.assertEqual(control.TranscriptionControl(2, 49152).max_tokens, 320)
        for duration in (float("nan"), float("inf"), 0, -1, 5401):
            with self.assertRaises(ValueError):
                control.TranscriptionControl(duration)

    def test_deadline_and_cancellation_are_failures(self):
        deadline = control.TranscriptionControl(2, timeout=0.001)
        time.sleep(0.003)
        with self.assertRaises(TimeoutError):
            deadline.check()
        cancelled = control.TranscriptionControl(2)
        cancelled.cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            cancelled.check()


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_waits_for_worker_cleanup_before_next_job(self):
        events = []
        started = threading.Event()
        guard = control.TranscriptionControl(2)
        def generate():
            started.set()
            try:
                while True:
                    guard.check()
                    time.sleep(0.001)
            finally:
                time.sleep(0.01)
                events.append("cleaned")
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(executor, generate)
            task = asyncio.create_task(control.finish_worker(future, guard))
            await asyncio.to_thread(started.wait)
            next_job = loop.run_in_executor(executor, lambda: events.append("next"))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(future.done())
            await next_job
        self.assertEqual(events, ["cleaned", "next"])

    async def test_timeout_stops_worker_and_next_job_succeeds(self):
        guard = control.TranscriptionControl(2, timeout=0.01)
        def generate():
            while True:
                guard.check()
                time.sleep(0.001)
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop = asyncio.get_running_loop()
            with self.assertRaises(TimeoutError):
                await control.finish_worker(loop.run_in_executor(executor, generate), guard)
            self.assertEqual(await loop.run_in_executor(executor, lambda: "valid"), "valid")

    async def test_cancelled_queued_work_never_calls_model(self):
        gate = threading.Event()
        invoked = []
        guard = control.TranscriptionControl(2)
        def generate():
            guard.check()
            invoked.append(True)
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop = asyncio.get_running_loop()
            blocking = loop.run_in_executor(executor, gate.wait)
            worker = loop.run_in_executor(executor, generate)
            task = asyncio.create_task(control.finish_worker(worker, guard))
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            gate.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await blocking
        self.assertEqual(invoked, [])

    async def test_disconnect_stops_worker_and_awaits_cleanup(self):
        from types import SimpleNamespace
        disconnected = asyncio.Event()
        cleaned = threading.Event()
        async def is_disconnected():
            return disconnected.is_set()
        with ThreadPoolExecutor(max_workers=1) as executor:
            async def transcribe(path, **kwargs):
                guard = kwargs['_lalia_control']
                def generate():
                    try:
                        while True:
                            guard.check()
                            time.sleep(0.001)
                    finally:
                        cleaned.set()
                return await control.finish_worker(asyncio.get_running_loop().run_in_executor(executor, generate), guard)
            task = asyncio.create_task(control.guarded_transcription(
                SimpleNamespace(is_disconnected=is_disconnected), SimpleNamespace(transcribe=transcribe), 'unused', {}, None))
            await asyncio.sleep(0.01)
            disconnected.set()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                await asyncio.wait_for(task, timeout=1)
            self.assertTrue(cleaned.is_set())

    async def test_deadline_response_does_not_wait_for_queued_request(self):
        guard = control.TranscriptionControl(2, timeout=0.01)
        release_next = threading.Event()
        next_started = threading.Event()
        def generate():
            while True:
                guard.check()
                time.sleep(0.001)
        def next_request():
            next_started.set()
            release_next.wait()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop = asyncio.get_running_loop()
            worker = loop.run_in_executor(executor, generate)
            next_job = loop.run_in_executor(executor, next_request)
            try:
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(control.finish_worker(worker, guard), timeout=0.5)
                await asyncio.wait_for(asyncio.to_thread(next_started.wait), timeout=0.5)
                self.assertFalse(next_job.done())
            finally:
                release_next.set()
                await next_job
