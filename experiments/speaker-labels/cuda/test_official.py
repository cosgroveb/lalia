import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import official_supervise as supervise
import official_transcribe as transcribe
from official_transcribe import finish_evidence, response_from_text, verify_checkpoint_layout

from official import prepare, score_result, verify_environment


class ResourceAdmissionTests(unittest.TestCase):
    def run_preload(self, root, available, free=60, device="NVIDIA GB10"):
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.device_count.return_value = 1
        cuda.get_device_name.return_value = device
        cuda.mem_get_info.return_value = (free * transcribe.GIB, 120 * transcribe.GIB)
        cuda.get_device_properties.return_value = SimpleNamespace(total_memory=120 * transcribe.GIB)
        torch = SimpleNamespace(cuda=cuda, set_num_threads=Mock())
        loader = Mock(side_effect=RuntimeError("model load reached"))
        modules = {"torch": torch,
                   "transformers": SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
                                                   AutoProcessor=Mock()),
                   "inference_utils": SimpleNamespace(build_transcription_messages=Mock(),
                                                       generate_transcription=Mock())}
        files = {"/proc/meminfo": f"MemAvailable: {available * 1024**2} kB\nMemFree: {free * 1024**2} kB\n",
                 "/sys/fs/cgroup/memory.max": str(64 * transcribe.GIB),
                 "/sys/fs/cgroup/memory.swap.max": "0"}
        manifest = {"model_revision": "model", "helper_revision": "helper", "files": {"public90.wav": "audio"}}
        versions = {"torch": "2.14.1+cu130", "transformers": "5.17.0"}
        with patch.dict(sys.modules, modules), \
                patch.object(sys, "argv", ["official_transcribe.py", "--out", str(root / "model")]), \
                patch.object(transcribe, "verify_inputs", return_value=(manifest, "prompt")), \
                patch.object(transcribe.importlib.metadata, "version", side_effect=lambda name: versions.get(name, "test")), \
                patch.object(Path, "read_text", autospec=True, side_effect=lambda path: files[str(path)]):
            try:
                transcribe.main()
            except (MemoryError, RuntimeError, ValueError) as error:
                return error, cuda, loader
        self.fail("Expected a resource rejection or the model-load sentinel")

    def test_gb10_uses_reclaimable_host_memory_and_keeps_allocator_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error, cuda, loader = self.run_preload(root, available=80)
            self.assertEqual(str(error), "model load reached")
            loader.assert_called_once()
            cuda.set_per_process_memory_fraction.assert_called_once_with(64 / 120)
            resource = json.loads((root / "model/resource-preflight.json").read_text())
            self.assertEqual(resource["host_mem_available_bytes"], 80 * transcribe.GIB)
            self.assertEqual(resource["host_mem_free_bytes"], 60 * transcribe.GIB)
            self.assertEqual(resource["cuda_free_bytes"], 60 * transcribe.GIB)
            self.assertEqual(resource["cuda_total_bytes"], 120 * transcribe.GIB)
            self.assertEqual(resource["device"], "NVIDIA GB10")
            self.assertEqual(resource["cgroup_memory_max_bytes"], 64 * transcribe.GIB)
            self.assertEqual(resource["cgroup_swap_max_bytes"], 0)

    def test_low_host_memory_records_actual_values_before_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error, cuda, loader = self.run_preload(root, available=71)
            self.assertIsInstance(error, MemoryError)
            loader.assert_not_called()
            cuda.set_per_process_memory_fraction.assert_not_called()
            resource = json.loads((root / "model/resource-preflight.json").read_text())
            self.assertEqual(resource["host_mem_available_bytes"], 71 * transcribe.GIB)
            for value in (71 * transcribe.GIB, 60 * transcribe.GIB, 120 * transcribe.GIB):
                self.assertIn(str(value), str(error))

    def test_non_gb10_is_rejected_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error, cuda, loader = self.run_preload(root, available=80, free=80, device="NVIDIA H100")
            self.assertIsInstance(error, ValueError)
            self.assertIn("NVIDIA H100", str(error))
            loader.assert_not_called()
            cuda.set_per_process_memory_fraction.assert_not_called()
            self.assertEqual(json.loads((root / "model/resource-preflight.json").read_text())["device"],
                             "NVIDIA H100")


class EvidenceTests(unittest.TestCase):
    def test_native_eos_and_token_limit_are_distinct(self):
        self.assertEqual(finish_evidence([7, 151645], 151645)["native_finish_reason"], "eos")
        self.assertFalse(finish_evidence([7, 151645], 151645)["token_limit_reached"])
        self.assertEqual(finish_evidence([7, 8], 151645, 2)["native_finish_reason"], "length")
        self.assertEqual(finish_evidence([7], 151645)["native_finish_reason"], "unknown")
        boundary = finish_evidence([7, 151645], 151645, 2)
        self.assertTrue(boundary["eos_observed"])
        self.assertTrue(boundary["token_limit_reached"])

    def test_invalid_native_sequences_fail(self):
        for ids, eos in (([], 151645), ([151645, 7], 151645), ([7], 151643)):
            with self.subTest(ids=ids, eos=eos), self.assertRaises(ValueError):
                finish_evidence(ids, eos)

    def test_adapter_preserves_text_and_exposes_unmatched_tail(self):
        text = "[0.0][S01] hello [1.2]\n[1.2][S02] unfinished"
        response, unmatched = response_from_text(text, 100, 20)
        self.assertEqual(response["text"], text)
        self.assertEqual(response["segments"], [{"start": 0.0, "end": 1.2, "speaker_id": "S01", "text": "hello"}])
        self.assertGreater(unmatched, 0)
        self.assertEqual(response["usage"]["total_tokens"], 120)

    def test_checkpoint_requires_complete_keys_and_torch_convolutions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = "model-00000-of-00001.safetensors"
            weights = {"model.whisper_encoder.conv1.weight": {"shape": [1024, 80, 3]},
                       "model.whisper_encoder.conv2.weight": {"shape": [1024, 1024, 3]}}
            index = {"weight_map": dict.fromkeys(weights, shard)}
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            def write_header():
                data = json.dumps(weights).encode()
                (root / shard).write_bytes(struct.pack("<Q", len(data)) + data)
            write_header()
            verify_checkpoint_layout(root)
            weights["model.whisper_encoder.conv1.weight"]["shape"] = [1024, 3, 80]
            write_header()
            with self.assertRaisesRegex(ValueError, "Torch Conv1d"):
                verify_checkpoint_layout(root)
            weights.pop("model.whisper_encoder.conv1.weight")
            write_header()
            with self.assertRaisesRegex(ValueError, "tensor keys"):
                verify_checkpoint_layout(root)


class TokenObserverTests(unittest.TestCase):
    def value(self, tokens, prompt=False):
        return SimpleNamespace(device=SimpleNamespace(type="cpu"),
                               shape=(1, len(tokens)) if prompt else (len(tokens),),
                               reshape=lambda *args: SimpleNamespace(tolist=lambda: list(tokens)))

    def test_forwards_identical_values_and_excludes_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial-generated-ids.json"
            delegate = Mock()
            observer = transcribe.TokenObserver(delegate, [10, 11], path)
            values = [self.value([10, 11], prompt=True), self.value([7]), self.value([151645])]
            for value in values:
                self.assertIs(observer.put(value), delegate.put.return_value)
            self.assertIs(observer.end(), delegate.end.return_value)
            for call, value in zip(delegate.put.call_args_list, values):
                self.assertIs(call.args[0], value)
            delegate.end.assert_called_once_with()
            observer.verify([7, 151645])
            checkpoint = json.loads(path.read_text())
            self.assertEqual(checkpoint["generated_ids"], [7, 151645])
            self.assertEqual(checkpoint["generated_tokens_lower_bound"], 2)
            self.assertEqual(checkpoint["status"], "incomplete")

    def test_interrupted_checkpoint_is_a_bounded_prefix_not_native_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial-generated-ids.json"
            observer = transcribe.TokenObserver(Mock(), [10], path)
            observer.put(self.value([10], prompt=True))
            for token in range(131):
                observer.put(self.value([token]))
            checkpoint = json.loads(path.read_text())
            self.assertEqual(checkpoint["generated_ids"], list(range(128)))
            self.assertEqual(checkpoint["generated_tokens_lower_bound"], 128)
            self.assertEqual(checkpoint["status"], "incomplete")
            self.assertNotIn("native_finish_reason", checkpoint)
            with self.assertRaisesRegex(ValueError, "streamed IDs"):
                observer.verify(list(range(131)))

    def test_completion_requires_ended_matching_ids_and_respects_cap(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(transcribe, "MAX_NEW_TOKENS", 2):
            observer = transcribe.TokenObserver(Mock(), [10], Path(directory) / "partial.json")
            observer.put(self.value([10], prompt=True))
            observer.put(self.value([7]))
            observer.put(self.value([8]))
            with self.assertRaisesRegex(ValueError, "token ceiling"):
                observer.put(self.value([9]))
            observer.end()
            observer.verify([7, 8])
            with self.assertRaisesRegex(ValueError, "streamed IDs"):
                observer.verify([7, 9])
            self.assertEqual(finish_evidence(observer.ids, 151645, 2)["native_finish_reason"], "length")

    def test_rejects_changed_prompt_device_shape_and_tokens_after_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = transcribe.TokenObserver(Mock(), [10], Path(directory) / "partial.json")
            with self.assertRaisesRegex(ValueError, "prompt"):
                observer.put(self.value([11], prompt=True))
            prompt = self.value([10], prompt=True)
            prompt.device.type = "cuda"
            with self.assertRaisesRegex(ValueError, "CPU"):
                observer.put(prompt)
            observer.put(self.value([10], prompt=True))
            with self.assertRaisesRegex(ValueError, "one selected token"):
                observer.put(self.value([7, 8]))
            with self.assertRaisesRegex(ValueError, "one selected token"):
                observer.put(self.value([1.5]))
            observer.put(self.value([151645]))
            with self.assertRaisesRegex(ValueError, "after EOS"):
                observer.put(self.value([7]))


class SupervisionTests(unittest.TestCase):
    def preflight_root(self, root):
        (root / "inputs").mkdir()
        (root / "inputs/manifest.json").write_text("{}")
        (root / "source.sha256").write_text(
            supervise.sha256(root / "inputs/manifest.json") + "  inputs/manifest.json\n")
        image = "sha256:" + "a" * 64
        (root / "image.id").write_text(image)
        (root / "frontend-preflight.json").write_text(json.dumps({"status": "ready", "image": image,
            "manifest_sha256": supervise.sha256(root / "inputs/manifest.json")}))
        return {"status": "ready", "image": image, "root_read_only": True,
                "tmp_mount": " /tmp rw - tmpfs tmpfs rw",
                "source_inventory_sha256": supervise.sha256(root / "source.sha256"),
                "caches": {variable: {"path": location, "files": {"kernel": {}}}
                           for variable, location in (value.split("=", 1) for value in supervise.CACHE_ENV[2:])}}

    def watchdog_root(self, root):
        work = root / "results/e2e"
        work.mkdir(parents=True)
        (work / "watchdog-owner.json").write_text(json.dumps({"owner": "x", "cid": "a" * 64,
            "pid": 1234, "pid_identity": "old"}))
        (work / "pause-requested").touch()
        return work

    def test_watchdog_allows_full_normal_run_and_restoration_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = self.watchdog_root(root)
            clock = [0]
            def sleep(seconds):
                clock[0] += seconds
                if clock[0] == 11495:
                    (work / "supervisor-cleanup-finished").touch()
            with patch.object(supervise, "pid_identity", return_value="old"), \
                    patch.object(supervise.time, "monotonic", side_effect=lambda: clock[0]), \
                    patch.object(supervise.time, "sleep", side_effect=sleep), \
                    patch.object(supervise.Supervisor, "cleanup") as cleanup:
                self.assertEqual(supervise.watchdog(root), 0)
            self.assertEqual(clock[0], 11495)
            cleanup.assert_not_called()
            self.assertFalse((work / "watchdog-recovering").exists())

    def test_watchdog_recovers_at_hard_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = self.watchdog_root(root)
            clock = [0]
            def sleep(seconds):
                clock[0] += seconds
            def cleanup(runner, errors):
                self.assertEqual(clock[0], 11700)
                self.assertTrue((work / "watchdog-recovering").exists())
                clock[0] += 535
            with patch.object(supervise, "pid_identity", return_value="old"), \
                    patch.object(supervise.time, "monotonic", side_effect=lambda: clock[0]), \
                    patch.object(supervise.time, "sleep", side_effect=sleep), \
                    patch.object(supervise.Supervisor, "cleanup", cleanup):
                self.assertEqual(supervise.watchdog(root), 0)
            self.assertEqual(clock[0], 12235)

    def test_cleanup_waits_for_recovery_and_does_not_report_unreaped_success(self):
        for recovery_seconds, exit_code in ((535, 0), (535, 1), (601, 0)):
            with self.subTest(seconds=recovery_seconds, code=exit_code), tempfile.TemporaryDirectory() as directory:
                runner = supervise.Supervisor(Path(directory))
                runner.work.mkdir(parents=True)
                (runner.work / "watchdog-recovering").touch()
                clock = [0]
                def wait(timeout):
                    self.assertEqual(timeout, 600)
                    clock[0] += min(timeout, recovery_seconds)
                    if recovery_seconds > timeout:
                        raise subprocess.TimeoutExpired("watchdog", timeout)
                    return exit_code
                runner.watchdog = SimpleNamespace(wait=wait)
                runner.watchdog_log = io.StringIO()
                errors = []
                with patch.object(supervise.base.Supervisor, "cleanup"):
                    runner.cleanup(errors)
                self.assertEqual(clock[0], min(600, recovery_seconds))
                self.assertEqual("host_watchdog_reaped" in runner.events, recovery_seconds <= 600)
                self.assertEqual(bool(errors), recovery_seconds > 600 or exit_code != 0)

    def test_extended_attach_leaves_service_and_recovery_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = supervise.Supervisor(root, call=Mock(return_value=""))
            (runner.work / "model").mkdir(parents=True)
            ids = [7, 151645]
            (runner.work / "model/generated-ids.json").write_text(json.dumps(ids))
            (runner.work / "model/result.json").write_text(json.dumps({
                "status": "generated", **finish_evidence(ids, 151645)}))
            attachment = SimpleNamespace(wait=Mock(return_value=0))
            watchdog = SimpleNamespace(poll=Mock(return_value=None))
            def popen(args, **kwargs):
                if "watchdog" in args:
                    (runner.work / "watchdog-ready").touch()
                    return watchdog
                return attachment
            runner.popen = popen
            with patch.object(runner, "prepare"), patch.object(supervise, "pid_identity", return_value="same"), \
                    patch.object(supervise, "available_bytes", return_value=80 * transcribe.GIB):
                try:
                    runner.execute()
                finally:
                    runner.log.close()
                    runner.watchdog_log.close()
            attachment.wait.assert_called_once_with(timeout=10820)
            self.assertIn("native_result_verified", runner.events)
            # Arm + service stop + snapshots + attach + worst normal restoration.
            self.assertLess(10 + 100 + 30 + 10820 + 535, supervise.WATCHDOG_SECONDS)
            self.assertGreaterEqual(supervise.WATCHDOG_JOIN_SECONDS, 535)

    def test_prepare_keeps_bounds_and_refuses_active_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self.preflight_root(root)
            image = receipt["image"]
            (root / "kernel-preflight.json").write_text(json.dumps(receipt))
            (root / "results/kernel").mkdir(parents=True)
            lifecycle = {"original_error": None, "cleanup_errors": [], "events": ["kernel_preflight_verified",
                         "owned_container_absent", "docker_attach_reaped", "service_health_ok"]}
            (root / "results/kernel/lifecycle.json").write_text(json.dumps(lifecycle))
            calls = []
            def command(args, timeout):
                calls.append(args)
                if args[:3] == ["docker", "image", "inspect"]:
                    return image
                if args[:2] == ["docker", "inspect"]:
                    return supervise.PRODUCTION_IMAGE
                if args[:2] == ["docker", "create"]:
                    runner.cidfile.write_text("b" * 64)
                if args[:2] == ["docker", "ps"] and any(x.startswith("label=") for x in args):
                    return "b" * 64 if runner.cidfile.exists() else ""
                return ""
            runner = supervise.Supervisor(root, call=command)
            runner.prepare()
            create = next(c for c in calls if c[:2] == ["docker", "create"])
            self.assertEqual(create[create.index("--memory") + 1], "64g")
            self.assertEqual(create[create.index("--memory-swap") + 1], "64g")
            self.assertIn("10800s", create)
            self.assertIn("--init", create)
            self.assertIn("--rm", create)
            self.assertEqual(create[create.index("--user") + 1],
                             f"{supervise.os.getuid()}:{supervise.os.getgid()}")
            self.assertEqual(runner.work.stat().st_mode & 0o777, 0o700)
            self.assertIn("/probe/official_transcribe.py", create[-1])
            self.assertIn('mkdir -p "$PYTORCH_KERNEL_CACHE_PATH" && exec python3', create[-1])
            for value in supervise.CACHE_ENV:
                self.assertIn(value, create)
            self.assertEqual(json.loads((runner.work / "image.json").read_text())["parent"],
                             supervise.PARENT_IMAGE)
            self.assertFalse(any(c[:3] == ["sudo", "-n", "systemctl"] for c in calls))
            with self.assertRaisesRegex(ValueError, "collision"):
                runner.prepare()

    def test_kernel_preflight_runs_bounded_without_inputs_or_service_pause(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = self.preflight_root(root)
                calls = []
                active = [False]
                def command(args, timeout):
                    calls.append(args)
                    if args[:3] == ["docker", "image", "inspect"]:
                        return receipt["image"]
                    if args[:2] == ["docker", "inspect"]:
                        return supervise.PRODUCTION_IMAGE
                    if args[:2] == ["docker", "create"]:
                        runner.cidfile.write_text("b" * 64)
                        active[0] = True
                    if args[:2] == ["docker", "ps"]:
                        return "b" * 64 if active[0] else ""
                    return ""
                def popen(args, stdout, stderr):
                    stdout.write(json.dumps(receipt))
                    active[0] = False
                    return SimpleNamespace(wait=Mock(return_value=exit_code))
                runner = supervise.Supervisor(root, call=command, popen=popen, kernel=True)
                with patch("builtins.print"):
                    self.assertEqual(runner.run(), exit_code)
                create = next(c for c in calls if c[:2] == ["docker", "create"])
                for flag in ("--memory", "--memory-swap"):
                    self.assertEqual(create[create.index(flag) + 1], "4g")
                for value in ("--init", "--rm", "--read-only", "60s", "/tmp:rw,nosuid,exec,size=2g",
                              *supervise.CACHE_ENV):
                    self.assertIn(value, create)
                self.assertEqual(create[create.index("--network") + 1], "none")
                self.assertEqual(create[create.index("--user") + 1],
                                 f"{supervise.os.getuid()}:{supervise.os.getgid()}")
                self.assertFalse(any("/inputs:ro" in value for value in create))
                self.assertFalse(any(c[:3] == ["sudo", "-n", "systemctl"] for c in calls))
                self.assertEqual((root / "kernel-preflight.json").exists(), exit_code == 0)
                lifecycle = json.loads((runner.work / "lifecycle.json").read_text())
                self.assertEqual(lifecycle["cleanup_errors"], [])
                self.assertIn("owned_container_absent", lifecycle["events"])
                self.assertIn("docker_attach_reaped", lifecycle["events"])

    def test_full_run_rejects_missing_stale_or_unclean_kernel_preflight_before_pause(self):
        for fault in ("missing", "image", "source", "cache", "mount", "cleanup"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = self.preflight_root(root)
                image = receipt["image"]
                if fault == "image":
                    receipt["image"] = "wrong"
                if fault == "source":
                    receipt["source_inventory_sha256"] = "wrong"
                if fault == "cache":
                    receipt["caches"]["TRITON_CACHE_DIR"]["path"] = "/root/.triton"
                if fault == "mount":
                    receipt["tmp_mount"] = " /tmp rw,noexec - tmpfs tmpfs rw"
                if fault != "missing":
                    (root / "kernel-preflight.json").write_text(json.dumps(receipt))
                if fault == "cleanup":
                    (root / "results/kernel").mkdir(parents=True)
                    (root / "results/kernel/lifecycle.json").write_text(json.dumps({
                        "original_error": None, "cleanup_errors": ["container still running"], "events": []}))
                calls = []
                def command(args, timeout):
                    calls.append(args)
                    return image if args[:3] == ["docker", "image", "inspect"] else ""
                runner = supervise.Supervisor(root, call=command)
                with self.assertRaises((FileNotFoundError, ValueError)):
                    runner.execute()
                self.assertFalse(runner.pause_requested)
                self.assertFalse(any(c[:2] == ["docker", "create"] for c in calls))
                self.assertFalse(any(c[:3] == ["sudo", "-n", "systemctl"] for c in calls))

    def test_changed_source_fails_before_service_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.sha256").write_text("0" * 64 + "  worker.py\n")
            (root / "worker.py").write_text("changed")
            calls = []
            def command(args, timeout):
                calls.append(args)
                return ""
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                supervise.Supervisor(root, call=command).prepare()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][:2], ["docker", "ps"])

    def test_watchdog_restores_only_after_owned_container_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "results/e2e"
            work.mkdir(parents=True)
            (work / "watchdog-owner.json").write_text(json.dumps({"owner": "x", "cid": "a" * 64,
                "pid": 987654321, "pid_identity": "old"}))
            (work / "pause-requested").touch()
            def cleanup(runner, errors):
                self.assertEqual(runner.owner, "x")
                self.assertTrue(runner.pause_requested)
                runner.events.extend(["owned_container_absent", "service_start_completed"])
            with patch.object(supervise.Supervisor, "cleanup", cleanup):
                self.assertEqual(supervise.watchdog(root), 0)
            result = json.loads((work / "watchdog-recovery.json").read_text())
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["events"], ["owned_container_absent", "service_start_completed"])


class FrozenInputTests(unittest.TestCase):
    def test_workers_match_current_provenance(self):
        provenance = json.loads((prepare.SOURCE / "provenance.json").read_text())
        self.assertEqual(prepare.sha256(prepare.SOURCE.parent / "official_transcribe.py"),
                         provenance["current_worker_sha256"])
        self.assertEqual(prepare.sha256(prepare.SOURCE.parent / "official_kernel_preflight.py"),
                         provenance["current_kernel_preflight_sha256"])

    def test_preparation_snapshots_helpers_keeps_references_local_and_refuses_reuse(self):
        for relative in (True, False):
            with self.subTest(relative=relative):
                self.check_prepared_snapshot(relative)

    def check_prepared_snapshot(self, relative):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "repo/experiments/speaker-labels/cuda/official"
            shutil.copytree(prepare.SOURCE, source)
            for name in prepare.PROBE_FILES:
                shutil.copyfile(prepare.SOURCE.parent / name, source.parent / name)
            for name in ("score.py", "qualification.py"):
                shutil.copyfile(prepare.SOURCE.parents[1] / name, source.parents[1] / name)
            inputs = root / "assets"
            (inputs / "model").mkdir(parents=True)
            assets = {"model/config.json": "{}", "prompt.txt": "prompt", "public90.wav": "wave"}
            for name, content in assets.items():
                (inputs / name).write_text(content)
            prompt_target = root / "pinned-prompt.txt"
            (inputs / "prompt.txt").rename(prompt_target)
            (inputs / "prompt.txt").symlink_to("../pinned-prompt.txt" if relative else prompt_target)
            helper, references = root / "helper.py", root / "references.json"
            helper_target = root / "pinned-helper.py"
            helper_target.write_text("# pinned upstream helper\n")
            helper.symlink_to(helper_target.name if relative else helper_target)
            references.write_text('{"clips": []}')
            manifest = {"files": {name: prepare.sha256(inputs / name) for name in assets},
                        "helper_sha256": prepare.sha256(helper)}
            manifest["files"][prepare.WEIGHTS] = "remote-weight-hash"
            (source / "inputs-manifest.json").write_text(json.dumps(manifest))
            (source / "provenance.json").write_text(json.dumps({
                "reference_manifest_sha256": prepare.sha256(references)}))
            state = root / "state"
            args = (state, inputs, inputs / "public90.wav", helper, references)
            with patch.object(prepare, "SOURCE", source):
                prepare.prepare(*args)
                inventory = (state / "local-source.sha256").read_bytes()
                with self.assertRaisesRegex(ValueError, "already exists"):
                    prepare.prepare(*args)
                self.assertEqual((state / "local-source.sha256").read_bytes(), inventory)
                references.write_text("changed")
                with self.assertRaisesRegex(ValueError, "Pinned input differs"):
                    prepare.prepare(root / "bad-state", *args[1:])
                self.assertFalse((root / "bad-state").exists())
            score_result.verify_scoring_inputs(state)
            for target in (helper_target, prompt_target):
                target.write_text("changed after freezing")
            frozen = {"probe/inference_utils.py": "# pinned upstream helper\n", "inputs/prompt.txt": "prompt"}
            for name, content in frozen.items():
                self.assertFalse((state / name).is_symlink())
                self.assertEqual((state / name).read_text(), content)
            helper_target.unlink()
            prompt_target.unlink()
            for name, content in frozen.items():
                self.assertEqual((state / name).read_text(), content)
            remote_inventory = (state / "source.sha256").read_text()
            self.assertNotIn("scoring/", remote_inventory)
            self.assertNotIn(prepare.WEIGHTS, remote_inventory)
            self.assertFalse((state / "inputs" / prepare.WEIGHTS).exists())
            for line in inventory.decode().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(prepare.sha256(state / name), digest)
            self.assertEqual((state / "probe/supervise.py").read_bytes(),
                             (source.parent / "supervise.py").read_bytes())
            (source.parent / "supervise.py").write_text("changed after freezing")
            self.assertNotEqual((state / "probe/supervise.py").read_bytes(),
                                (source.parent / "supervise.py").read_bytes())

    def test_parent_source_and_binary_changes_fail_without_version_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [Path(name) for name in ("numeric/source.py", "numeric/kernel.so",
                     "numeric/__pycache__/source.pyc", "numeric-1.dist-info/RECORD")]
            for entry in files:
                path = root / entry
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original")
            distribution = SimpleNamespace(metadata={"Name": "numeric"}, version="1.0",
                                           files=files, locate_file=lambda entry: root / entry)
            with patch.object(verify_environment.importlib.metadata, "distributions", return_value=[distribution]):
                original = verify_environment.snapshot()
                for entry in files[2:]:
                    (root / entry).write_bytes(b"generated")
                self.assertEqual(verify_environment.verify(original, verify_environment.snapshot())[
                    "parent_payload_files_unchanged"], 2)
                for entry in files[:2]:
                    (root / entry).write_bytes(b"same version, different payload")
                    with self.assertRaisesRegex(ValueError, "payload changed"):
                        verify_environment.verify(original, verify_environment.snapshot())
                    (root / entry).write_bytes(b"original")

    def test_scoring_references_and_scripts_must_match_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("scoring/public-manifest.json", "scoring/score.py", "scoring/qualification.py", "score_result.py")
            for name in names:
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("frozen")
            digests = {name: score_result.sha256(root / name) for name in names}
            (root / "scoring-manifest.json").write_text(json.dumps({"files": digests}))
            provenance = score_result.verify_scoring_inputs(root)
            self.assertEqual(provenance["reference_manifest_sha256"], digests[names[0]])
            for name in names:
                (root / name).write_text("changed")
                with self.assertRaisesRegex(ValueError, "Frozen scoring input changed"):
                    score_result.verify_scoring_inputs(root)
                (root / name).write_text("frozen")


if __name__ == "__main__":
    unittest.main()
