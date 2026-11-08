"""CPU fakes verify restart safety and preserve completed and partial evidence."""
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation_batch import EvaluationBatch, _boot_id, _process_identity, upstream_dependencies


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def load(path):
    return json.loads(path.read_text())


class EvaluationBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.suite = self.root / "suite"
        (self.suite / "scripts").mkdir(parents=True)
        (self.suite / "scripts/run_evaluation_episode.py").write_text("# immutable CPU fixture\n")
        self.repo = self.root / "upstream"
        (self.repo / "og_ego_prim/utils").mkdir(parents=True)
        (self.repo / "bddl/bddl").mkdir(parents=True)
        (self.repo / "og_ego_prim/utils/prompts.py").write_text("PROMPT = 'fixed'\n")
        (self.repo / "bddl/bddl/domain.bddl").write_text("(define (domain fixed))\n")
        (self.repo / ".gitignore").write_text("__pycache__/\n")
        for command in (("init", "-q"), ("config", "user.email", "cpu-fixture@example.invalid"),
                        ("config", "user.name", "CPU Fixture"), ("add", "."),
                        ("commit", "-qm", "immutable CPU fixture")):
            subprocess.run(["git", "-C", str(self.repo), *command], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.prepared = []
        for name in ("a", "b", "c"):
            self.prepared.append({"task": name, "scene": "Rs_int", "max_steps": 12,
                "task_config_sha256": name + "_config", "scene_sha256": name + "_scene",
                "config": {"task_info": {"task_name": name}, "evaluation_goal_conditions": {
                    "process_safety_goal_condition": [{"risk_type": "Fire"}],
                    "termination_safety_goal_condition": []}}})
        self.models = {"planner": {"repo_id": "pinned/planner", "revision": "fixed", "model_path": "/cache/fixed"}}
        self.runtime = {"episode_runner": "run_evaluation_episode.py", "scope": "fixed evaluation",
                        "episode_timeout_seconds": 30, "seed": 0, "max_format_attempts": 1}
        self.runtime["isbench_commit"] = upstream_dependencies(self.repo)["git_commit"]
        self.args = (self.suite, self.repo, "fake-python", "fake-inference", "A", self.prepared,
                     self.models, self.runtime, {})

    def tearDown(self):
        self.temp.cleanup()

    def resume(self, batch, **kwargs):
        return EvaluationBatch.resume(batch.output, self.suite, self.repo, "fake-python", "fake-inference",
                                      self.prepared, self.models, self.runtime, {}, **kwargs)

    def worker(self, reason="done", goal=True, infrastructure=False, returncode=-11):
        def launch(command, **kwargs):
            episode = Path(command[command.index("--output") + 1])
            self.assertEqual(Path(command[1]).name, "run_evaluation_episode.py")
            if infrastructure:
                write(episode / "progress.json", {"phase": "infrastructure_failed", "requests": [],
                                                  "exception": "fixture: GPU unavailable"})
            else:
                write(episode / "progress.json", {"phase": "evaluation_saved", "requests": [], "decisions": []})
                write(episode / "report.json", {"execution_goal_condition": {"eval": goal},
                    "termination": {"reason": reason}, "plans": [], "error_stack": [],
                    "process_safety_goal_condition": [{"risk_type": "Fire", "eval": None}],
                    "termination_safety_goal_condition": []})
                (episode / "cleanup_started.txt").write_text("shutdown started")
                if returncode == 0:
                    (episode / "cleanup_completed.txt").write_text("shutdown completed")
            return SimpleNamespace(pid=999999999, poll=lambda: returncode, wait=lambda: returncode)
        return launch

    def test_chunk_resume_skips_completed_case_and_keeps_original_evidence(self):
        batch = EvaluationBatch(*self.args)
        callbacks = []
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls:
            first = batch.run_pending(max_cases=1, after_case=lambda b: callbacks.append(len(b.results)))
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(callbacks, [1, 1])
        self.assertTrue(first[0]["valid_evaluation"])
        protected = {p.relative_to(batch.output / "a"): p.read_bytes()
                     for p in (batch.output / "a").rglob("*") if p.is_file()}
        resumed = self.resume(batch)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls:
            added = resumed.run_pending(max_cases=2)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([row["task"] for row in added], ["b", "c"])
        self.assertEqual(protected, {p.relative_to(batch.output / "a"): p.read_bytes()
                                    for p in (batch.output / "a").rglob("*") if p.is_file()})
        self.assertTrue(load(batch.output / "batch_summary.json")["metrics"]["all_tasks_evaluated"])
        with patch("evaluation_batch.subprocess.Popen") as calls:
            self.assertEqual(self.resume(batch).run_pending(), [])
        calls.assert_not_called()

    def test_valid_model_failure_is_terminal_and_does_not_pause_next_task(self):
        batch = EvaluationBatch(*self.args)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker("plan_error", False)):
            row = batch.run_pending(max_cases=1)[0]
        self.assertTrue(row["valid_evaluation"])
        self.assertFalse(row["task_completed"])
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls:
            self.resume(batch).run_pending(max_cases=2)
        self.assertEqual(calls.call_count, 2)
        metrics = load(batch.output / "batch_summary.json")["metrics"]
        self.assertEqual(metrics["SR_all_tasks"], 2 / 3)
        self.assertEqual(metrics["outcome_counts"]["planner_output_failure"], 1)

    def test_infrastructure_failure_stops_and_explicit_resume_only_runs_new_tasks(self):
        batch = EvaluationBatch(*self.args)
        callbacks = []
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker(infrastructure=True, returncode=1)) as calls:
            batch.run_pending(max_cases=3, after_case=lambda b: callbacks.append(len(b.results)))
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(callbacks, [1, 1])
        self.assertFalse(batch.can_continue)
        old_summary = (batch.output / "a/summary.json").read_bytes()
        with patch("evaluation_batch.subprocess.Popen") as calls:
            self.assertEqual(self.resume(batch).run_pending(), [])
        calls.assert_not_called()
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls:
            self.resume(batch, continue_after_infrastructure=True).run_pending(max_cases=3)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual((batch.output / "a/summary.json").read_bytes(), old_summary)
        metrics = load(batch.output / "batch_summary.json")["metrics"]
        self.assertTrue(metrics["all_tasks_attempted"])
        self.assertFalse(metrics["all_tasks_evaluated"])
        self.assertIsNone(metrics["SR_all_tasks"])

    def test_saved_exit_recovers_missing_summary_without_rerunning(self):
        batch = EvaluationBatch(*self.args)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()):
            batch.run_pending(max_cases=1)
        (batch.output / "a/summary.json").unlink()
        resumed = self.resume(batch)
        self.assertEqual(len(resumed.results), 1)
        self.assertTrue(resumed.results[0]["valid_evaluation"])
        self.assertIsNotNone(resumed.results[0]["elapsed_seconds"])

    def test_stale_unfinished_case_is_preserved_and_classified_as_unresolved(self):
        batch = EvaluationBatch(*self.args)
        episode = batch.output / "a"
        write(episode / "progress.json", {"phase": "inference_planner", "requests": [{"role": "planner"}]})
        (episode / "raw_output.txt").write_text("unfinished reasoning")
        old = (episode / "progress.json").read_bytes()
        resumed = self.resume(batch)
        self.assertFalse(resumed.can_continue)
        self.assertIsNone(resumed.results[0]["task_completed"])
        self.assertEqual(resumed.results[0]["recovery_status"], "unfinished_after_disconnect")
        self.assertEqual((episode / "progress.json").read_bytes(), old)
        self.assertEqual((episode / "raw_output.txt").read_text(), "unfinished reasoning")
        with patch("evaluation_batch.subprocess.Popen") as calls:
            resumed.run_pending()
        calls.assert_not_called()

    def test_busy_host_lock_refuses_another_controller(self):
        batch = EvaluationBatch(*self.args)
        path = batch._lock_path()
        path.parent.mkdir(exist_ok=True)
        with path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another controller"):
                self.resume(batch)
        self.assertEqual(self.resume(batch).results, [])

    def test_live_recorded_process_refuses_resume_even_with_confirmation(self):
        batch = EvaluationBatch(*self.args)
        write(batch.output / "a/process_running.json", {"runtime_boot_id": _boot_id(),
              "process_identity": _process_identity(os.getpid()) or {"pid": os.getpid()}})
        for confirm in (False, True):
            with self.subTest(confirm=confirm), self.assertRaisesRegex(RuntimeError, "still running"):
                self.resume(batch, confirm_previous_runtime_stopped=confirm)
        self.assertFalse((batch.output / "a/process_exit.json").exists())

    def test_foreign_unfinished_process_requires_explicit_old_runtime_confirmation(self):
        batch = EvaluationBatch(*self.args)
        write(batch.output / "a/process_running.json", {"runtime_boot_id": "foreign-boot-id",
              "process_identity": {"pid": 999999999, "boot_id": "foreign-boot-id", "start_ticks": "100"}})
        with self.assertRaisesRegex(RuntimeError, "another runtime"):
            self.resume(batch)
        resumed = self.resume(batch, confirm_previous_runtime_stopped=True)
        self.assertFalse(resumed.results[0]["valid_evaluation"])
        self.assertTrue(load(batch.output / "a/recovery.json")["confirmed_previous_runtime_stopped"])

    def test_foreign_active_controller_requires_confirmation_but_idle_does_not(self):
        batch = EvaluationBatch(*self.args)
        write(batch.output / "controller_state.json", {"status": "running", "runtime_boot_id": "foreign",
                                                       "process_identity": {"pid": 999999999}})
        with self.assertRaisesRegex(RuntimeError, "another runtime"):
            self.resume(batch)
        self.assertEqual(self.resume(batch, confirm_previous_runtime_stopped=True).results, [])
        self.assertEqual(self.resume(batch).results, [])

    def test_resume_rejects_metadata_or_code_changes_without_rewriting_summary(self):
        batch = EvaluationBatch(*self.args)
        original = (batch.output / "batch_summary.json").read_bytes()
        saved_runtime = copy.deepcopy(self.runtime)
        self.runtime["seed"] = 1
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.resume(batch)
        self.assertEqual((batch.output / "batch_summary.json").read_bytes(), original)
        self.runtime.clear()
        self.runtime.update(saved_runtime)
        (self.suite / "scripts/run_evaluation_episode.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "code"):
            self.resume(batch)
        self.assertEqual((batch.output / "batch_summary.json").read_bytes(), original)

    def test_launch_failure_is_saved_and_checkpoint_callback_runs(self):
        batch = EvaluationBatch(*self.args)
        checkpoints = []
        with patch("evaluation_batch.subprocess.Popen", side_effect=OSError("cannot launch")):
            rows = batch.run_pending(max_cases=3, after_case=lambda b: checkpoints.append(len(b.results)))
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["valid_evaluation"])
        self.assertIn("cannot launch", rows[0]["controller_error"])
        self.assertEqual(checkpoints, [1, 1])
        self.assertTrue((batch.output / "a/process_exit.json").is_file())

    def test_timeout_is_saved_as_invalid_and_stops_the_chunk(self):
        batch = EvaluationBatch(*self.args)
        process = SimpleNamespace(pid=999999999, poll=lambda: None, wait=lambda: -15)
        with patch("evaluation_batch.subprocess.Popen", return_value=process), \
                patch("evaluation_batch.stop_group") as stop, \
                patch("evaluation_batch.time.monotonic", side_effect=[0, 31, 32]):
            rows = batch.run_pending(max_cases=3)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["timed_out"])
        self.assertFalse(rows[0]["valid_evaluation"])
        self.assertTrue(stop.called)
        self.assertEqual(load(batch.output / "a/process_exit.json")["elapsed_seconds"], 32)

    def test_keyboard_interruption_is_saved_and_never_retried(self):
        batch = EvaluationBatch(*self.args)
        process = SimpleNamespace(pid=999999999, poll=lambda: (_ for _ in ()).throw(KeyboardInterrupt()), wait=lambda: -15)
        callbacks = []
        with patch("evaluation_batch.subprocess.Popen", return_value=process), patch("evaluation_batch.stop_group"):
            rows = batch.run_pending(max_cases=3, after_case=lambda b: callbacks.append(len(b.results)))
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["interrupted"])
        self.assertEqual(callbacks, [1, 1])
        with patch("evaluation_batch.subprocess.Popen") as calls:
            self.resume(batch).run_pending()
        calls.assert_not_called()

    def test_continuation_opt_in_does_not_ignore_a_new_infrastructure_failure(self):
        batch = EvaluationBatch(*self.args)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker(infrastructure=True, returncode=1)):
            batch.run_pending(max_cases=1)
        resumed = self.resume(batch, continue_after_infrastructure=True)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker(infrastructure=True, returncode=1)) as calls:
            resumed.run_pending(max_cases=3)
            self.assertEqual(resumed.run_pending(max_cases=3), [])
        self.assertEqual(calls.call_count, 1)
        self.assertEqual([row["task"] for row in resumed.results], ["a", "b"])

    def test_tampered_summary_cannot_become_a_terminal_success(self):
        batch = EvaluationBatch(*self.args)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker("plan_error", False)):
            batch.run_pending(max_cases=1)
        row = load(batch.output / "a/summary.json")
        row["task_completed"] = True
        write(batch.output / "a/summary.json", row)
        with self.assertRaisesRegex(ValueError, "raw evaluation"):
            self.resume(batch)

    def test_task_definition_sidecar_is_verified_on_resume(self):
        batch = EvaluationBatch(*self.args)
        sidecar = load(batch.output / "task_definitions.json")
        sidecar["b"]["task_info"]["task_name"] = "other"
        write(batch.output / "task_definitions.json", sidecar)
        with self.assertRaisesRegex(ValueError, "task_definitions"):
            self.resume(batch)

    def test_tracked_upstream_prompt_or_bddl_edits_stop_resume_and_chunks(self):
        for relative in ("og_ego_prim/utils/prompts.py", "bddl/bddl/domain.bddl"):
            batch = EvaluationBatch(*self.args)
            original = (self.repo / relative).read_bytes()
            (self.repo / relative).write_bytes(original + b"# changed\n")
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "runtime dependencies"):
                self.resume(batch)
            with patch("evaluation_batch.subprocess.Popen") as calls, self.assertRaisesRegex(ValueError, "runtime dependencies"):
                batch.run_pending(max_cases=1)
            calls.assert_not_called()
            (self.repo / relative).write_bytes(original)

    def test_irrelevant_gitignore_edit_does_not_change_runtime_dependencies(self):
        before = upstream_dependencies(self.repo)
        (self.repo / ".gitignore").write_text("__pycache__/\n*.log\n")
        self.assertEqual(upstream_dependencies(self.repo), before)
        batch = EvaluationBatch(*self.args)
        self.assertEqual(load(batch.output / "upstream_dependencies.json"), before)
        self.assertEqual(batch.manifest["upstream_dependency_sha256"], before["sha256"])

    def test_guard_request_counts_include_all_requested_views_when_worker_fails_partway(self):
        batch = EvaluationBatch(*self.args)
        episode = batch.output / "a"
        write(episode / "progress.json", {"phase": "infrastructure_failed", "requests": [
            {"role": "planner"}, {"role": "homeguard"}]})
        write(episode / "001_planner_request.json", {"role": "planner", "jobs": [{}]})
        write(episode / "002_homeguard_request.json", {"role": "homeguard", "jobs": [{}] * 5})
        write(episode / "001_planner_response.json", {"outputs": [{"hit_token_limit": False}]})
        write(episode / "002_homeguard_response.json", {"outputs": [{"hit_token_limit": False}] * 2,
                                                        "error": "worker failed at third view"})
        write(episode / "process_exit.json", {"returncode": 1, "elapsed_seconds": 10})
        resumed = self.resume(batch)
        row = resumed.results[0]
        self.assertEqual((row["model_calls"], row["guard_model_calls"]), (2, 1))
        self.assertEqual((row["worker_requests"], row["guard_worker_requests"]), (2, 1))
        self.assertEqual((row["model_generations"], row["guard_generations"]), (6, 5))
        self.assertEqual((row["completed_model_generations"], row["completed_guard_generations"]), (3, 2))
        self.assertTrue(row["generation_counts_complete"])
        self.assertFalse(row["valid_evaluation"])
        metrics = load(batch.output / "batch_summary.json")["metrics"]
        self.assertEqual((metrics["model_generations"], metrics["guard_generations"]), (6, 5))
        self.assertEqual(metrics["completed_guard_generations"], 2)

    def test_callback_error_releases_lock_and_keeps_case_saved(self):
        batch = EvaluationBatch(*self.args)
        def checkpoint(_batch):
            raise RuntimeError("checkpoint offline")
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()), self.assertRaisesRegex(RuntimeError, "offline"):
            batch.run_pending(max_cases=3, after_case=checkpoint)
        resumed = self.resume(batch)
        self.assertEqual([row["task"] for row in resumed.results], ["a"])
        self.assertEqual(load(batch.output / "controller_state.json")["status"], "idle")

    def test_explicit_results_root_keeps_raw_outputs_outside_source_suite(self):
        persistent = self.root / "persistent_drive"
        batch = EvaluationBatch(*self.args, results_root=persistent)
        self.assertEqual(batch.output.parent, persistent / "A")
        self.assertTrue((batch.output / "code/run_evaluation_episode.py").is_file())
        self.assertEqual(load(batch.output / "task_definitions.json"),
                         {case["task"]: case["config"] for case in self.prepared})
        self.assertEqual(self.resume(batch).output, batch.output)


if __name__ == "__main__":
    unittest.main()
