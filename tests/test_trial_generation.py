"""Trial decoding controls are explicit and validated before model loading."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from vlm_worker import generation_kwargs


class TrialGenerationTests(unittest.TestCase):
    def test_legacy_requests_keep_greedy_decoding(self):
        self.assertEqual(generation_kwargs({"max_new_tokens": 8192}), {
            "max_new_tokens": 8192, "do_sample": False,
            "repetition_penalty": 1.0, "use_cache": True,
        })

    def test_sampling_passes_all_controls_to_generation(self):
        settings = {"max_new_tokens": 8192, "do_sample": True,
                    "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                    "repetition_penalty": 1.0}
        actual = generation_kwargs(settings)
        self.assertEqual(actual, {**settings, "use_cache": True})
        self.assertEqual(settings, {key: actual[key] for key in settings})

    def test_greedy_does_not_pass_unused_sampling_controls(self):
        actual = generation_kwargs({"max_new_tokens": 2048, "do_sample": False,
                                    "temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertFalse(actual["do_sample"])
        self.assertTrue({"temperature", "top_p", "top_k"}.isdisjoint(actual))

    def test_instruct_sampling_can_use_different_settings(self):
        actual = generation_kwargs({"max_new_tokens": 2048, "do_sample": True,
                                    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                    "repetition_penalty": 1.1})
        self.assertEqual((actual["temperature"], actual["top_p"], actual["top_k"],
                          actual["repetition_penalty"]), (0.7, 0.8, 20, 1.1))

    def test_sampling_requires_explicit_controls(self):
        settings = {"max_new_tokens": 8192, "do_sample": True,
                    "temperature": 1.0, "top_p": 0.95, "top_k": 20}
        for key in ("temperature", "top_p", "top_k"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "explicit"):
                generation_kwargs({k: v for k, v in settings.items() if k != key})

    def test_invalid_values_are_rejected_including_booleans_and_nonfinite(self):
        settings = {"max_new_tokens": 8192, "do_sample": True,
                    "temperature": 1.0, "top_p": 0.95, "top_k": 20}
        bad = {
            "do_sample": (1, "true", None),
            "max_new_tokens": (0, -1, True, 4.5, "8192"),
            "temperature": (0, -0.1, True, float("nan"), float("inf"), "1"),
            "top_p": (0, 1.1, True, float("nan"), float("inf"), "0.95"),
            "top_k": (-1, True, 2.5, "20"),
            "repetition_penalty": (0, -1, True, float("nan"), float("inf"), "1"),
        }
        for key, values in bad.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    generation_kwargs({**settings, key: value})
        for key in ("temperature", "top_p", "top_k"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                generation_kwargs({"max_new_tokens": 2048, "do_sample": False,
                                   key: bad[key][0]})

    def test_boundary_values_are_valid(self):
        actual = generation_kwargs({"max_new_tokens": 1, "do_sample": True,
                                    "temperature": 0.01, "top_p": 1, "top_k": 0,
                                    "repetition_penalty": 0.5})
        self.assertEqual((actual["max_new_tokens"], actual["top_p"], actual["top_k"]), (1, 1, 0))

    def test_worker_consumes_helper_kwargs_and_records_effective_settings(self):
        tree = ast.parse((ROOT / "scripts/vlm_worker.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "generate"]
        self.assertEqual(len(calls), 1)
        self.assertEqual([(key.arg, ast.unparse(key.value)) for key in calls[0].keywords],
                         [(None, "inputs"), (None, "job_decoding")])
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        validation = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                          and ast.unparse(node.value) == "generation_kwargs(settings)")
        first_import = next(i for i, node in enumerate(main.body) if isinstance(node, (ast.Import, ast.ImportFrom)))
        self.assertLess(validation, first_import)
        self.assertIn("effective_generation_kwargs", (ROOT / "scripts/vlm_worker.py").read_text())


if __name__ == "__main__":
    unittest.main()
