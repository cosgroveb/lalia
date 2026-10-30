import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


RUNTIME = Path(__file__).parents[2] / "Support/runtime/lalia_runtime.py"
SPEC = importlib.util.spec_from_file_location("lalia_runtime", RUNTIME)
lalia_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lalia_runtime)


def lsof_responses(paths):
    def run(command, **kwargs):
        if "-iTCP:8002" in command:
            return SimpleNamespace(stdout="73277\n")
        return SimpleNamespace(stdout="p73277\n" + "".join(f"n{path}\n" for path in paths))
    return run


class RuntimeSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.home = self.root / "home"
        self.home.mkdir()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.patches = [
            patch.object(lalia_runtime, "CONFIG", self.config),
            patch.object(lalia_runtime, "ROOT", self.runtime),
            patch.object(lalia_runtime.Path, "home", return_value=self.home),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def test_migrates_legacy_endpoint_and_preserves_summary_settings(self):
        original = {
            "baseURL": "http://127.0.0.1:8000/v1",
            "transcriptionModel": lalia_runtime.MODEL,
            "summaryModel": "my-summary-model",
            "summaryBaseURL": "http://localhost:8000/v1",
            "language": "fr",
        }
        self.config.write_text(json.dumps(original))
        with patch.object(lalia_runtime, "print"):
            lalia_runtime.write_config()
        actual = json.loads(self.config.read_text())
        self.assertEqual(actual["baseURL"], lalia_runtime.BASE_URL)
        self.assertEqual(actual["summaryBaseURL"], original["summaryBaseURL"])
        self.assertEqual(actual["summaryModel"], original["summaryModel"])
        self.assertEqual(actual["language"], "fr")
        self.assertEqual(actual["transcriptionModel"], lalia_runtime.MODEL)
        self.assertEqual(len(list(self.config.parent.glob("config-before-runtime-*.json"))), 1)

    def test_keeps_custom_transcription_endpoint_untouched(self):
        original = {"baseURL": "http://localhost:8123/v1", "summaryModel": "custom"}
        self.config.write_text(json.dumps(original))
        with patch.object(lalia_runtime, "print"):
            lalia_runtime.write_config()
        self.assertEqual(json.loads(self.config.read_text()), original)

    def test_keeps_custom_summary_endpoint_while_migrating_transcription(self):
        original = {
            "baseURL": "http://127.0.0.1:8000/v1",
            "transcriptionModel": lalia_runtime.MODEL,
            "summaryBaseURL": "http://localhost:8123/v1",
            "summaryModel": "custom",
        }
        self.config.write_text(json.dumps(original))
        with patch.object(lalia_runtime, "print"):
            lalia_runtime.write_config()
        actual = json.loads(self.config.read_text())
        self.assertEqual(actual["baseURL"], lalia_runtime.BASE_URL)
        self.assertEqual(actual["summaryBaseURL"], original["summaryBaseURL"])
        self.assertEqual(actual["summaryModel"], "custom")

    def test_migrates_only_the_known_legacy_model(self):
        self.config.write_text(json.dumps({
            "baseURL": "http://127.0.0.1:8000/v1",
            "transcriptionModel": lalia_runtime.LEGACY_MODEL,
        }))
        with patch.object(lalia_runtime, "print"):
            lalia_runtime.write_config()
        actual = json.loads(self.config.read_text())
        self.assertEqual(actual["baseURL"], lalia_runtime.BASE_URL)
        self.assertEqual(actual["transcriptionModel"], lalia_runtime.MODEL)

    def test_preserves_unknown_model_on_legacy_endpoint(self):
        original = {"baseURL": "http://127.0.0.1:8000/v1", "transcriptionModel": "custom-model"}
        self.config.write_text(json.dumps(original))
        with patch.object(lalia_runtime, "print"):
            lalia_runtime.write_config()
        self.assertEqual(json.loads(self.config.read_text()), original)

    def test_legacy_model_key_overrides_current_model_key_like_swift(self):
        original = {
            "baseURL": "http://127.0.0.1:8000/v1",
            "transcriptionModel": lalia_runtime.MODEL,
            "sessionTranscriptionModel": "custom-session-model",
        }
        self.config.write_text(json.dumps(original))
        with patch.object(lalia_runtime, "print"):
            lalia_runtime.write_config()
        self.assertEqual(json.loads(self.config.read_text()), original)

    def test_known_legacy_key_normalizes_model_after_migration(self):
        self.config.write_text(json.dumps({
            "baseURL": "http://127.0.0.1:8000/v1",
            "transcriptionModel": "custom-model",
            "sessionTranscriptionModel": lalia_runtime.MODEL,
        }))
        with patch.object(lalia_runtime, "print"):
            lalia_runtime.write_config()
        actual = json.loads(self.config.read_text())
        self.assertEqual(actual["baseURL"], lalia_runtime.BASE_URL)
        self.assertEqual(actual["transcriptionModel"], lalia_runtime.MODEL)
        self.assertNotIn("sessionTranscriptionModel", actual)

    def test_refuses_to_replace_unrecognized_existing_model(self):
        target = self.runtime / "models" / lalia_runtime.MODEL
        target.mkdir(parents=True)
        (target / "config.json").write_text('{"model_type":"wrong"}')
        with self.assertRaisesRegex(RuntimeError, "Refusing incomplete or unrecognized"):
            lalia_runtime.provision_model()

    def test_accepts_lalia_moss_timed_segment_and_usage_shape(self):
        result = {
            "text": "[0.00][S01] Lalia runtime verification.[1.25]",
            "segments": [{"start": 0.0, "end": 1.25, "text": "[S01] Lalia runtime verification.", "speaker_id": "S01"}],
            "usage": {"type": "tokens", "input_tokens": 18, "output_tokens": 6, "total_tokens": 24},
        }
        lalia_runtime.validate_inference_result(result, 1.3)

    def test_rejects_missing_or_inconsistent_usage(self):
        result = {
            "text": "[0.00][S01] Lalia runtime verification.[1.25]",
            "segments": [{"start": 0.0, "end": 1.25, "text": "[S01] Lalia runtime verification.", "speaker_id": "S01"}],
            "usage": {"type": "tokens", "input_tokens": 18, "output_tokens": 6, "total_tokens": 25},
        }
        with self.assertRaisesRegex(RuntimeError, "token usage"):
            lalia_runtime.validate_inference_result(result, 1.3)

    def test_foreign_port_owner_does_not_restart_service(self):
        with (patch.object(lalia_runtime, "provision_model"),
              patch.object(lalia_runtime, "check", side_effect=lalia_runtime.PortConflictError("foreign")),
              patch.object(lalia_runtime.subprocess, "run") as run):
            with self.assertRaises(lalia_runtime.PortConflictError):
                lalia_runtime.setup()
        run.assert_not_called()

    def test_renamed_server_is_owned_by_loaded_runtime_module(self):
        prefix = "/opt/homebrew/Cellar/lalia-runtime/0.3.3/libexec"
        paths = [f"{prefix}/lib/python3.11/site-packages/setproctitle.cpython-311-darwin.so"]
        with (
            patch.object(lalia_runtime.sys, "prefix", prefix),
            patch.object(lalia_runtime.subprocess, "run", side_effect=lsof_responses(paths)),
            patch.object(lalia_runtime, "request", return_value={"data": [{"id": lalia_runtime.MODEL}]}) as request,
        ):
            lalia_runtime.check()
        request.assert_called_once_with("/models")

    def test_older_runtime_loaded_module_is_restartable(self):
        prefix = "/opt/homebrew/Cellar/lalia-runtime/0.3.3/libexec"
        paths = ["/opt/homebrew/Cellar/lalia-runtime/0.3.2/libexec/lib/python3.11/site-packages/mlx/core.so"]
        with (
            patch.object(lalia_runtime.sys, "prefix", prefix),
            patch.object(lalia_runtime.subprocess, "run", side_effect=lsof_responses(paths)),
            patch.object(lalia_runtime, "request") as request,
        ):
            with self.assertRaisesRegex(RuntimeError, "older Lalia runtime"):
                lalia_runtime.check()
        request.assert_not_called()

    def test_foreign_listener_is_not_owned_by_omlx_process_name(self):
        prefix = "/opt/homebrew/Cellar/lalia-runtime/0.3.3/libexec"
        paths = ["/opt/homebrew/Cellar/omlx/0.7.3/libexec/lib/python3.11/site-packages/mlx/core.so"]
        with (
            patch.object(lalia_runtime.sys, "prefix", prefix),
            patch.object(lalia_runtime.subprocess, "run", side_effect=lsof_responses(paths)),
            patch.object(lalia_runtime, "request") as request,
        ):
            with self.assertRaises(lalia_runtime.PortConflictError):
                lalia_runtime.check()
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
