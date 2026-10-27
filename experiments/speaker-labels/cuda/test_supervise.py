from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest

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
    def run_case(self, fake, mode="score", cid_collision=False, real_health=False, fp32_rope=False):
        with tempfile.TemporaryDirectory() as directory:
            cls = supervise.Supervisor if real_health else ImmediateHealthSupervisor
            runner = cls(Path(directory), mode, fake.call, fake.popen, sleep=lambda seconds: None,
                         fp32_rope=fp32_rope)
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
