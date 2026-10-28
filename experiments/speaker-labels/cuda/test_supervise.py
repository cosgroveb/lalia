from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import supervise


CID = "a" * 64


class FakeDocker:
    def __init__(self, outcome="ok", cleanup_failure=False, collision=False, create_failure=False,
                 pause_failure=False, restore_failure=False, health_interrupt=False,
                 stop_not_found=False):
        self.outcome, self.cleanup_failure = outcome, cleanup_failure
        self.collision, self.create_failure = collision, create_failure
        self.pause_failure, self.restore_failure = pause_failure, restore_failure
        self.health_interrupt, self.stop_not_found = health_interrupt, stop_not_found
        self.exists = False
        self.active = True
        self.calls = []
        self.waits = 0
        self.metrics = 'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'

    def call(self, args, timeout):
        assert timeout > 0
        self.calls.append(args)
        if args[:2] == ["docker", "ps"]:
            owned = any(arg.startswith("label=") for arg in args)
            return CID if self.exists or (self.collision and not owned) else ""
        if args[:2] == ["docker", "create"]:
            self.exists = True
            Path(args[args.index("--cidfile") + 1]).write_text(CID)
            if self.create_failure:
                raise RuntimeError("create response lost")
            return CID
        if args[:3] == ["docker", "image", "inspect"]:
            return supervise.EXACT_PREFIX_IMAGE
        if args[:2] == ["docker", "inspect"]:
            return supervise.PRODUCTION_IMAGE
        if args[:2] == ["docker", "rm"]:
            self.exists = False
            return CID
        if args[:2] == ["docker", "stop"]:
            if self.stop_not_found:
                self.exists = False
                raise RuntimeError("No such container")
            if self.cleanup_failure:
                raise RuntimeError("stop failed")
            self.exists = False
            return CID
        if args[:2] == ["docker", "wait"]:
            if not self.exists:
                raise RuntimeError("No such container")
            return "0"
        if args[:3] == ["sudo", "-n", "systemctl"]:
            if args[3] == "start" and self.restore_failure:
                raise RuntimeError("start failed")
            self.active = args[3] == "start"
            if args[3] == "stop" and self.pause_failure:
                raise RuntimeError("stop service response lost")
            return ""
        if args[0] == "curl":
            if self.health_interrupt:
                self.health_interrupt = False
                supervise.interrupted(signal.SIGTERM, None)
            if not self.active:
                raise RuntimeError("service unavailable")
            if args[-1].endswith("/metrics"):
                return self.metrics
            return ""
        if args[:2] == ["systemctl", "is-active"]:
            return ""
        raise AssertionError(args)

    def popen(self, args, **kwargs):
        self.calls.append(args)
        return self

    def wait(self, timeout):
        self.waits += 1
        if self.waits == 1:
            if self.outcome == "signal":
                supervise.interrupted(signal.SIGTERM, None)
            if self.outcome == "failure":
                return 7
            self.exists = False
        return 0


class ImmediateHealthSupervisor(supervise.Supervisor):
    def health(self, seconds):
        self.call(["curl", "http://127.0.0.1:8000/health"], 5)
        self.events.append("service_health_ok")


class SupervisorTests(unittest.TestCase):
    def test_hidden_source_inventory_rejects_extra_or_changed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe"
            probe.mkdir()
            source = probe / "worker.py"
            source.write_text("original")
            (root / "source-hashes.json").write_text(json.dumps({source.name: hashlib.sha256(source.read_bytes()).hexdigest()}))
            runner = supervise.Supervisor(root, "hidden-kernel")
            runner.source_hash()
            extra = probe / "unexpected.py"
            extra.write_text("extra")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                runner.source_hash()
            extra.unlink()
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source differs"):
                runner.source_hash()

    def test_idle_gauges_require_both_families_and_every_series_zero(self):
        valid = '# HELP ignored\nvllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0.0\n'
        self.assertEqual(supervise.idle_metrics(valid)["vllm:num_requests_running"], [0.0])
        for text in ("", valid.splitlines()[1], valid + 'vllm:num_requests_waiting{engine="1"} 1\n',
                     valid.replace("0.0", "NaN"), valid.replace("0.0", "Inf"),
                     valid.replace("0.0", "bad"), valid + "vllm:num_requests_running{broken\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                supervise.idle_metrics(text)

    def test_idle_gauges_validate_label_syntax_and_unique_keys(self):
        waiting = "\nvllm:num_requests_waiting 0\n"
        for labels in ('{broken}', '{engine=0}', '{engine="0",engine="1"}',
                       '{engine="0" engine="1"}', '{engine="0",,}', '{1engine="0"}',
                       r'{engine="bad\t"}', '{engine="unterminated}', '{= "0"}'):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                supervise.idle_metrics("vllm:num_requests_running" + labels + " 0" + waiting)
        for labels in ('{}', '{ }', '{engine="0", model_name="qwen3-coder-next"}',
                       r'{engine="quote\" slash\\ newline\n {braces},equals=",}',
                       '{ engine = "0" , model_name = "qwen" }'):
            with self.subTest(labels=labels):
                self.assertEqual(supervise.idle_metrics("vllm:num_requests_running" + labels + " 0" + waiting)
                                 ["vllm:num_requests_running"], [0.0])

    def test_hidden_idle_refusal_never_pauses_restarts_or_starts_container(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(supervise.Supervisor, "source_hash", return_value="f" * 64), \
                patch.object(supervise.Supervisor, "verify_hidden_kernel"), \
                patch.object(supervise.Supervisor, "verify_boundary_inputs"):
            for index, sample in enumerate((' 1', '{broken} 0', '{engine=0} 0', '{engine="0",engine="1"} 0')):
                with self.subTest(sample=sample):
                    fake = FakeDocker()
                    fake.metrics = "vllm:num_requests_running" + sample + "\nvllm:num_requests_waiting 0\n"
                    runner = ImmediateHealthSupervisor(Path(temporary) / str(index), "score", fake.call, fake.popen,
                                                       exact_prefix=True, hidden_boundary=True)
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(runner.run(), 1)
                    self.assertFalse(runner.pause_requested)
                    self.assertTrue(fake.active)
                    self.assertFalse(fake.exists)
                    self.assertFalse(any(call[0] == "sudo" or call[:2] == ["docker", "start"] for call in fake.calls))
                    self.assertIn("unstarted_owned_container_removed", runner.events)

    def test_hidden_prepare_keeps_archived_image_uid_and_cache_environment(self):
        fake = FakeDocker()
        with tempfile.TemporaryDirectory() as temporary, patch.object(supervise.Supervisor, "source_hash", return_value="f" * 64):
            runner = ImmediateHealthSupervisor(Path(temporary), "hidden-kernel", fake.call, fake.popen)
            runner.prepare()
            create = next(call for call in fake.calls if call[:2] == ["docker", "create"])
            self.assertIn(supervise.EXACT_PREFIX_IMAGE, create)
            self.assertEqual(create[create.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
            self.assertIn("/tmp:rw,nosuid,exec,size=2g", create)
            self.assertFalse("--read-only" in create)
            for value in supervise.CACHE_ENV:
                self.assertIn(value, create)
            self.assertIn("60s", create)
            self.assertEqual(create[create.index("--memory") + 1], "4g")
            self.assertIn("mkdir -p /tmp/torch-kernels", create[-1])
            self.assertFalse(any(call[0] == "sudo" for call in fake.calls))

    def test_hidden_preflight_failure_stops_before_container_creation_or_pause(self):
        fake = FakeDocker()
        with tempfile.TemporaryDirectory() as temporary, patch.object(supervise.Supervisor, "source_hash", return_value="f" * 64), \
                patch.object(supervise.Supervisor, "verify_hidden_kernel", side_effect=ValueError("preflight failed")):
            runner = ImmediateHealthSupervisor(Path(temporary), "score", fake.call, fake.popen,
                                               exact_prefix=True, hidden_boundary=True)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(), 1)
            self.assertFalse(any(call[0] == "sudo" or call[:2] == ["docker", "create"] for call in fake.calls))
            self.assertTrue(fake.active)

    def test_hidden_kernel_receipt_requires_cache_execution_and_private_host_output(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(supervise.Supervisor, "source_hash", return_value="f" * 64):
            root = Path(temporary)
            private = root / "private-output"
            private.mkdir(mode=0o700)
            output = private / "check.json"
            output.write_text(json.dumps({"uid": os.getuid(), "gid": os.getgid()}))
            output.chmod(0o600)
            receipt = {"status": "ready", "image": supervise.EXACT_PREFIX_IMAGE,
                       "source_inventory_sha256": "f" * 64, "outer_product_max_error": 0,
                       "cuda_peak_allocated_bytes": 11776, "root_read_only": False,
                       "tmp_mount": "0 0 0:1 / /tmp rw - tmpfs tmpfs rw",
                       "caches": {name: {"path": location, "files": {"compiled": {}}}
                                  for name, location in (value.split("=", 1) for value in supervise.CACHE_ENV[2:])},
                       "ownership": {"uid": os.getuid(), "gid": os.getgid(), "directory_mode": 0o700,
                                     "file_uid": os.getuid(), "file_gid": os.getgid(), "file_mode": 0o600}}
            runner = supervise.Supervisor(root, "hidden-kernel")
            runner.require_hidden_kernel(receipt, root)
            for field, value in (("image", "wrong"), ("source_inventory_sha256", "wrong"),
                                 ("outer_product_max_error", 1), ("root_read_only", True),
                                 ("tmp_mount", "0 0 0:1 / /tmp rw,noexec - tmpfs tmpfs rw")):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.require_hidden_kernel(dict(receipt, **{field: value}), root)
            output.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "private modes"):
                runner.require_hidden_kernel(receipt, root)

    def run_case(self, fake, mode="score", cid_collision=False, real_health=False, fp32_rope=False, exact_prefix=False):
        with tempfile.TemporaryDirectory() as directory:
            cls = supervise.Supervisor if real_health else ImmediateHealthSupervisor
            runner = cls(Path(directory), mode, fake.call, fake.popen, sleep=lambda seconds: None,
                         fp32_rope=fp32_rope, exact_prefix=exact_prefix)
            if cid_collision:
                runner.work.mkdir(parents=True)
                runner.cidfile.write_text("foreign")
            output = io.StringIO()
            with redirect_stdout(output):
                code = runner.run()
            return code, json.loads(output.getvalue()), fake.calls

    def assert_cleanup_order(self, report):
        events = report["events"]
        self.assertLess(events.index("owned_container_absent"), events.index("docker_attach_reaped"))
        self.assertLess(events.index("docker_attach_reaped"), events.index("service_start_completed"))
        self.assertEqual(events[-1], "service_health_ok")

    def test_success_restores_and_verifies_health(self):
        code, report, calls = self.run_case(FakeDocker())
        self.assertEqual(code, 0)
        self.assert_cleanup_order(report)

    def test_exact_prefix_selects_pinned_image_and_preserves_bounds(self):
        for enabled in (False, True):
            code, report, calls = self.run_case(FakeDocker(), exact_prefix=enabled)
            self.assertEqual(code, 0)
            self.assertEqual(report["exact_prefix"], enabled)
            create = next(call for call in calls if call[:2] == ["docker", "create"])
            image = supervise.EXACT_PREFIX_IMAGE if enabled else supervise.IMAGE
            self.assertIn(image, create)
            self.assertEqual("--exact-prefix" in create, enabled)
            self.assertEqual(create[create.index("--memory") + 1], "24g")
            self.assertIn("300s", create)
            self.assert_cleanup_order(report)
        with self.assertRaisesRegex(ValueError, "requires score mode"):
            supervise.Supervisor(Path("/unused"), "smoke", exact_prefix=True)

    def test_fp32_rope_is_explicit_and_preserves_score_bounds(self):
        for enabled in (False, True):
            code, report, calls = self.run_case(FakeDocker(), fp32_rope=enabled)
            self.assertEqual(code, 0)
            self.assertEqual(report["fp32_rope"], enabled)
            create = next(call for call in calls if call[:2] == ["docker", "create"])
            self.assertEqual("--fp32-rope" in create, enabled)
            self.assertEqual(create[create.index("--memory") + 1], "24g")
            self.assertIn("300s", create)
            self.assert_cleanup_order(report)

    def test_fp32_rope_rejects_non_score_mode_before_any_command(self):
        with self.assertRaisesRegex(ValueError, "requires score mode"):
            supervise.Supervisor(Path("/unused"), "decoder", fp32_rope=True)

    def test_decoder_is_bounded_and_never_pauses_service(self):
        code, report, calls = self.run_case(FakeDocker(), mode="decoder")
        self.assertEqual(code, 0)
        self.assertFalse(any(call[0] == "sudo" for call in calls))
        create = next(call for call in calls if call[:2] == ["docker", "create"])
        self.assertEqual(create[create.index("--memory") + 1], "4g")
        self.assertIn("100s", create)
        self.assertEqual(create[-1], "/probe/decoder_smoke.py")
        self.assertEqual(report["events"][-1], "service_health_ok")

    def test_worker_failure_stops_waits_reaps_then_restores(self):
        code, report, calls = self.run_case(FakeDocker(outcome="failure"))
        self.assertEqual(code, 1)
        self.assertIn("exited 7", report["original_error"])
        self.assertEqual(report["cleanup_errors"], [])
        self.assertTrue(any(call[:2] == ["docker", "stop"] for call in calls))
        self.assertTrue(any(call[:2] == ["docker", "wait"] for call in calls))
        self.assert_cleanup_order(report)

    def test_sigterm_stops_waits_and_restores(self):
        code, report, calls = self.run_case(FakeDocker(outcome="signal"))
        self.assertEqual(code, 1)
        self.assertIn("Received signal 15", report["original_error"])
        self.assert_cleanup_order(report)

    def test_cleanup_failure_preserves_original_and_prevents_overlap(self):
        code, report, calls = self.run_case(FakeDocker(outcome="failure", cleanup_failure=True))
        self.assertEqual(code, 1)
        self.assertIn("exited 7", report["original_error"])
        self.assertTrue(any("stop failed" in error for error in report["cleanup_errors"]))
        self.assertTrue(any("restoration withheld" in error for error in report["cleanup_errors"]))
        self.assertTrue(any("final health" in error for error in report["cleanup_errors"]))
        self.assertFalse(any(call[:4] == ["sudo", "-n", "systemctl", "start"] for call in calls))

    def test_collision_does_not_pause_or_stop_foreign_container(self):
        for cid in (False, True):
            with self.subTest(cid=cid):
                code, report, calls = self.run_case(FakeDocker(collision=not cid), cid_collision=cid)
                self.assertEqual(code, 1)
                self.assertEqual(report["events"][-1], "service_health_ok")
                self.assertFalse(any(call[0] == "sudo" or call[:2] == ["docker", "stop"] for call in calls))

    def test_create_failure_cleans_owned_container_before_any_pause(self):
        code, report, calls = self.run_case(FakeDocker(create_failure=True))
        self.assertEqual(code, 1)
        self.assertIn("create response lost", report["original_error"])
        self.assertTrue(any(call[:2] == ["docker", "stop"] for call in calls))
        self.assertFalse(any(call[0] == "sudo" for call in calls))
        self.assertEqual(report["events"][-1], "service_health_ok")

    def test_smoke_is_tracked_and_never_pauses_service(self):
        code, report, calls = self.run_case(FakeDocker(), mode="smoke")
        self.assertEqual(code, 0)
        self.assertFalse(any(call[0] == "sudo" for call in calls))
        self.assertTrue(any(call[:2] == ["docker", "create"] and "--cidfile" in call for call in calls))
        self.assertEqual(report["events"][-1], "service_health_ok")

    def test_partial_pause_failure_still_cleans_and_restores(self):
        code, report, calls = self.run_case(FakeDocker(pause_failure=True))
        self.assertEqual(code, 1)
        self.assertIn("stop service response lost", report["original_error"])
        self.assertTrue(any(call[:2] == ["docker", "wait"] for call in calls))
        self.assertIn("service_start_completed", report["events"])
        self.assertEqual(report["events"][-1], "service_health_ok")

    def test_restore_failure_keeps_original_and_final_health_error(self):
        code, report, calls = self.run_case(FakeDocker(outcome="failure", restore_failure=True))
        self.assertEqual(code, 1)
        self.assertIn("exited 7", report["original_error"])
        self.assertTrue(any("service restoration" in error for error in report["cleanup_errors"]))
        self.assertTrue(any("final health" in error for error in report["cleanup_errors"]))
        self.assertEqual(calls[-1][0], "curl")

    def test_pre_pause_health_interruption_never_pauses(self):
        code, report, calls = self.run_case(FakeDocker(health_interrupt=True), real_health=True)
        self.assertEqual(code, 1)
        self.assertIn("Received signal 15", report["original_error"])
        self.assertFalse(any(call[0] == "sudo" or call[:2] == ["docker", "create"] for call in calls))
        self.assertEqual(report["events"][-1], "service_health_ok")

    def test_list_then_auto_remove_before_stop_restores(self):
        code, report, calls = self.run_case(FakeDocker(outcome="failure", stop_not_found=True))
        self.assertEqual(code, 1)
        self.assertIn("exited 7", report["original_error"])
        self.assertEqual(report["cleanup_errors"], [])
        self.assertIn("auto_removed_before_stop", report["events"])
        self.assert_cleanup_order(report)

    def test_fatal_log_rejects_zero_exit_and_restores_service(self):
        class FatalDocker(FakeDocker):
            def popen(self, args, **kwargs):
                kwargs["stdout"].write("FATAL: kernel architecture mismatch\n")
                kwargs["stdout"].flush()
                return super().popen(args, **kwargs)
        code, report, calls = self.run_case(FatalDocker())
        self.assertEqual(code, 1)
        self.assertIn("Fatal runtime diagnostic", report["original_error"])
        self.assertEqual(report["cleanup_errors"], [])
        self.assert_cleanup_order(report)


if __name__ == "__main__":
    unittest.main()
