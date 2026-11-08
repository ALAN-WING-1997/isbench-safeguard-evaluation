"""Verify replay provenance and single-attempt behavior without model loading."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_planner_request import inspect_response, prepare_request, run_once


class PlannerReplayTests(unittest.TestCase):
    def fixture(self, root):
        images = []
        for index in range(5):
            path = root / f"obs_{index}.png"
            path.write_bytes(f"saved-view-{index}".encode())
            images.append(str(path))
        prompt = "Original C prompt with HomeGuard predictions."
        model = {"repo_id": "Qwen/Qwen3-VL-8B-Thinking", "revision": "pinned"}
        request = {
            "role": "planner", "step": 1, "proposal": 1, "format_attempt": 2,
            "jobs": [{"images": images, "prompt": prompt}],
            "prompt_sha256": [hashlib.sha256(prompt.encode()).hexdigest()],
            "image_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in images},
            "settings": {"seed": 0, "max_new_tokens": 8192}, "generation": {"do_sample": False}, "model": model,
        }
        request_path = root / "003_planner_request.json"
        request_path.write_text(json.dumps(request))
        runtime = {"inference": {"seed": 0, "max_new_tokens": 8192, "do_sample": True,
                                   "temperature": 1.0, "top_p": 0.95, "top_k": 20},
                   "request_timeout_seconds": 1800}
        model_path = root / "cached-model"
        model_path.mkdir()
        models = {"planner": {**model, "model_path": str(model_path)}}
        return request_path, runtime, models

    def test_preparation_preserves_exact_jobs_and_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            request_path, runtime, models = self.fixture(Path(temp))
            original_bytes = request_path.read_bytes()
            original = json.loads(original_bytes)
            prepared = prepare_request(request_path, runtime, models)
            self.assertEqual(prepared["jobs"], original["jobs"])
            self.assertEqual(prepared["prompt_sha256"], original["prompt_sha256"])
            self.assertEqual(prepared["image_sha256"], original["image_sha256"])
            self.assertEqual(prepared["settings"], runtime["inference"])
            self.assertEqual(prepared["replay"]["original_settings"], original["settings"])
            self.assertEqual(prepared["replay"]["original_generation"], original["generation"])
            self.assertNotIn("generation", prepared)
            self.assertEqual(prepared["replay"]["source_request_sha256"], hashlib.sha256(original_bytes).hexdigest())
            self.assertFalse(prepared["replay"]["action_executed"])
            self.assertFalse(prepared["replay"]["simulation_started"])
            self.assertEqual(request_path.read_bytes(), original_bytes)

    def test_tampered_prompt_and_hash_inventory_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            request_path, runtime, models = self.fixture(Path(temp))
            original = json.loads(request_path.read_text())
            for field, replacement in (("prompt_sha256", []), ("prompt_sha256", ["invalid"]),
                                       ("prompt_sha256", original["prompt_sha256"] * 2)):
                changed = {**original, field: replacement}
                request_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    prepare_request(request_path, runtime, models)
            original["jobs"][0]["prompt"] += " altered"
            request_path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "prompt content"):
                prepare_request(request_path, runtime, models)

    def test_missing_and_modified_original_images_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            request_path, runtime, models = self.fixture(Path(temp))
            image = Path(json.loads(request_path.read_text())["jobs"][0]["images"][0])
            original = image.read_bytes()
            image.write_bytes(b"different-image")
            with self.assertRaisesRegex(ValueError, "image content"):
                prepare_request(request_path, runtime, models)
            image.write_bytes(original)
            image.unlink()
            with self.assertRaisesRegex(ValueError, "image is missing"):
                prepare_request(request_path, runtime, models)

    def test_guard_request_and_checkpoint_mismatch_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            request_path, runtime, models = self.fixture(Path(temp))
            models["planner"]["revision"] = "different-pin"
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                prepare_request(request_path, runtime, models)
            request = json.loads(request_path.read_text())
            request["role"] = "homeguard"
            request_path.write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, "planner request"):
                prepare_request(request_path, runtime, models)

    def test_summary_is_syntax_only_and_rejects_truncation(self):
        request = {"model": {"repo_id": "Qwen/Qwen3-VL-8B-Thinking"}}
        output = {"raw_output": 'reasoning</think>\n{"action":"UNKNOWN(fake)","caution":null}',
                  "hit_token_limit": False}
        sampled = {"max_new_tokens": 8192, "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
        row = inspect_response({"outputs": [output], "effective_generation_kwargs": sampled}, request)
        self.assertEqual(row["status"], "final_json_available")
        self.assertFalse(row["action_operator_object_validation_performed"])
        self.assertFalse(row["action_executed"])
        self.assertFalse(row["correctness_or_safety_evaluated"])
        self.assertEqual(row["effective_generation"], sampled)
        output["hit_token_limit"] = True
        self.assertEqual(inspect_response({"outputs": [output]}, request)["status"], "output_truncated")
        output["hit_token_limit"] = False
        output["raw_output"] = '{"action":"DONE","caution":null}'
        self.assertEqual(inspect_response({"outputs": [output]}, request)["status"], "unfinished_reasoning")

    def test_worker_is_called_once_and_original_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path, runtime, models = self.fixture(root)
            original_bytes = request_path.read_bytes()

            def worker(command, **kwargs):
                response = Path(command[command.index("--response") + 1])
                response.write_text(json.dumps({"outputs": [{"raw_output": "repeating", "hit_token_limit": True}]}))
                return SimpleNamespace(returncode=0)

            with patch("replay_planner_request.subprocess.run", side_effect=worker) as process:
                output, summary = run_once(request_path, root / "replays", runtime, models, "fake-python")
            process.assert_called_once()
            self.assertEqual(summary["status"], "output_truncated")
            self.assertEqual((output / "original_request.json").read_bytes(), original_bytes)
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "response.json").is_file())
            self.assertEqual((output / "raw_output.txt").read_text(), "repeating")
            original = json.loads(original_bytes)
            for index, image in enumerate(original["jobs"][0]["images"]):
                self.assertEqual((output / f"inputs/obs_{index}.png").read_bytes(), Path(image).read_bytes())

    def test_child_failure_keeps_partial_response_and_summary_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path, runtime, models = self.fixture(root)

            def worker(command, **kwargs):
                response = Path(command[command.index("--response") + 1])
                response.write_text(json.dumps({"error": "GPU worker failure", "outputs": []}))
                return SimpleNamespace(returncode=1)

            with patch("replay_planner_request.subprocess.run", side_effect=worker) as process:
                output, summary = run_once(request_path, root / "replays", runtime, models, "fake-python")
            process.assert_called_once()
            self.assertEqual(summary["status"], "inference_failure")
            self.assertEqual(summary["error"], "GPU worker failure")
            self.assertEqual(json.loads((output / "response.json").read_text())["error"], "GPU worker failure")
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
