"""CPU fixtures exercise continuous scheduling against the real frozen controller."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import test_evaluation_batch as fixtures
import evaluation_session
from evaluation_batch import EvaluationBatch
from evaluation_session import export_session, run_all_remaining


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


class EvaluationSessionTests(unittest.TestCase):
    setUp = fixtures.EvaluationBatchTests.setUp
    tearDown = fixtures.EvaluationBatchTests.tearDown
    worker = fixtures.EvaluationBatchTests.worker
    resume = fixtures.EvaluationBatchTests.resume

    def test_one_invocation_attempts_all_twenty_tasks_once_despite_multiple_failures(self):
        template = copy.deepcopy(self.prepared[0])
        self.prepared[:] = []
        for index in range(20):
            task = "task_" + str(index).zfill(2)
            case = copy.deepcopy(template)
            case["task"] = case["config"]["task_info"]["task_name"] = task
            self.prepared.append(case)
        batch = EvaluationBatch(*self.args)
        launched, failed_evidence = [], {}

        def launch(command, **kwargs):
            episode = Path(command[command.index("--output") + 1])
            index = int(episode.name.removeprefix("task_"))
            launched.append(episode.name)
            if index in (0, 6, 17):
                self.worker(infrastructure=True)(command, **kwargs)
                fixtures.write(episode / "progress.json", {
                    "phase": "infrastructure_failed", "requests": [],
                    "exception": "AssertionError: Got mismatch in particle setting size: 505, vs. number of expected particles 511!",
                })
                failed_evidence[episode.name] = (episode / "progress.json").read_bytes()
                return SimpleNamespace(pid=999999999, poll=lambda: -11, wait=lambda: -11)
            reason = "plan_error" if index == 8 else "guard_format_error" if index == 12 else "done"
            return self.worker(reason, goal=reason == "done")(command, **kwargs)

        with patch("evaluation_batch.subprocess.Popen", side_effect=launch) as calls:
            current, added = run_all_remaining(batch)
        expected = [case["task"] for case in self.prepared]
        self.assertEqual(calls.call_count, 20)
        self.assertEqual(launched, expected)
        self.assertEqual([row["task"] for row in added], expected)
        self.assertEqual(len(set(launched)), 20)
        self.assertEqual(sum(row["valid_evaluation"] is not True for row in added), 3)
        self.assertEqual(sum(row["task_completed"] is True for row in added), 15)
        saved = fixtures.load(current.output / "batch_summary.json")
        self.assertEqual(saved["pending_tasks"], [])
        self.assertTrue(saved["metrics"]["all_tasks_attempted"])
        self.assertEqual(saved["metrics"]["invalid_evaluations"], 3)
        self.assertIsNone(saved["metrics"]["SR_all_tasks"])
        for task, original in failed_evidence.items():
            self.assertEqual((current.output / task / "progress.json").read_bytes(), original)

    def test_mixed_episode_failures_do_not_pause_or_retry_the_continuous_run(self):
        for task in ("d", "e"):
            case = copy.deepcopy(self.prepared[0])
            case["task"] = case["config"]["task_info"]["task_name"] = task
            self.prepared.append(case)
        batch = EvaluationBatch(*self.args)
        frozen = snapshot(batch.code)
        manifest = (batch.output / "batch_manifest.json").read_bytes()
        workers = iter((self.worker(), self.worker(infrastructure=True, returncode=-11),
                        self.worker("plan_error", False), self.worker("guard_format_error", False),
                        self.worker()))
        checkpoints = []
        with patch("evaluation_batch.subprocess.Popen", side_effect=lambda *a, **k: next(workers)(*a, **k)) as calls:
            current, added = run_all_remaining(batch, after_case=lambda b: checkpoints.append(len(b.results)))
        self.assertEqual(calls.call_count, 5)
        self.assertEqual([row["task"] for row in added], list("abcde"))
        self.assertFalse(added[1]["valid_evaluation"])
        self.assertTrue(added[2]["valid_evaluation"])
        self.assertTrue(added[3]["valid_evaluation"])
        self.assertFalse(added[2]["task_completed"])
        self.assertEqual(checkpoints, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        self.assertEqual(snapshot(current.code), frozen)
        self.assertEqual((current.output / "batch_manifest.json").read_bytes(), manifest)
        saved = fixtures.load(current.output / "batch_summary.json")
        self.assertTrue(saved["metrics"]["all_tasks_attempted"])
        self.assertFalse(saved["metrics"]["all_tasks_evaluated"])
        self.assertIsNone(saved["metrics"]["SR_all_tasks"])
        policy = fixtures.load(current.output / "session_policy.json")
        self.assertEqual(policy["automatic_episode_retries"], 0)
        self.assertEqual(policy["session_driver_sha256"],
                         hashlib.sha256((current.output / "session_driver.py").read_bytes()).hexdigest())

    def test_previously_failed_attempt_is_preserved_before_running_remaining_tasks(self):
        batch = EvaluationBatch(*self.args)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker(infrastructure=True)):
            batch.run_pending(max_cases=1)
        protected = snapshot(batch.output / "a")
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls:
            current, added = run_all_remaining(batch)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([row["task"] for row in added], ["b", "c"])
        self.assertEqual(snapshot(current.output / "a"), protected)
        self.assertEqual(len(current.results), 3)

    def test_timeout_is_preserved_and_next_task_starts_in_the_same_invocation(self):
        batch = EvaluationBatch(*self.args)
        stuck = SimpleNamespace(pid=999999999, poll=lambda: None, wait=lambda: -15)
        workers = iter((lambda *a, **k: stuck, self.worker(), self.worker()))
        with patch("evaluation_batch.subprocess.Popen", side_effect=lambda *a, **k: next(workers)(*a, **k)) as calls, \
                patch("evaluation_batch.stop_group"), \
                patch("evaluation_batch.time.monotonic", side_effect=[0, 31, 32, 40, 41, 50, 51]):
            current, added = run_all_remaining(batch)
        self.assertEqual(calls.call_count, 3)
        self.assertTrue(added[0]["timed_out"])
        self.assertFalse(added[0]["valid_evaluation"])
        self.assertEqual([row["task"] for row in current.results], ["a", "b", "c"])

    def test_user_interrupt_stops_and_only_a_new_invocation_acknowledges_it(self):
        batch = EvaluationBatch(*self.args)
        process = SimpleNamespace(pid=999999999,
            poll=lambda: (_ for _ in ()).throw(KeyboardInterrupt()), wait=lambda: -15)
        with patch("evaluation_batch.subprocess.Popen", return_value=process) as calls, \
                patch("evaluation_batch.stop_group"):
            current, added = run_all_remaining(batch)
        self.assertEqual(calls.call_count, 1)
        self.assertTrue(added[0]["interrupted"])
        self.assertFalse(current.can_continue)
        events = fixtures.load(current.output / "resume_events.json")["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["preserved_invalid_tasks"], [])
        protected = snapshot(current.output / "a")
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls:
            completed, rows = run_all_remaining(current)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([row["task"] for row in rows], ["b", "c"])
        self.assertEqual(snapshot(completed.output / "a"), protected)

    def test_identity_or_live_owner_errors_are_not_suppressed(self):
        batch = EvaluationBatch(*self.args)
        batch.runtime["seed"] = 99
        with patch("evaluation_batch.subprocess.Popen") as calls, \
                self.assertRaisesRegex(ValueError, "Resume inputs differ"):
            run_all_remaining(batch)
        calls.assert_not_called()
        self.assertFalse((batch.output / "session_policy.json").exists())
        batch.runtime["seed"] = 0
        path = batch._lock_path()
        import fcntl
        with path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another controller"):
                run_all_remaining(batch)

    def test_checkpoint_storage_error_stops_without_attempting_another_task(self):
        batch = EvaluationBatch(*self.args)
        def unavailable(_):
            raise OSError("Drive write unavailable")
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls, \
                self.assertRaisesRegex(OSError, "Drive write unavailable"):
            run_all_remaining(batch, after_case=unavailable)
        self.assertEqual(calls.call_count, 1)
        self.assertFalse((batch.output / "b").exists())

    def test_previous_runtime_confirmation_is_required_and_preserved(self):
        batch = EvaluationBatch(*self.args)
        fixtures.write(batch.output / "a/process_running.json", {
            "runtime_boot_id": "foreign-runtime",
            "process_identity": {"pid": 999999999, "boot_id": "foreign-runtime", "start_ticks": "100"},
        })
        with patch("evaluation_batch.subprocess.Popen") as calls, \
                self.assertRaisesRegex(RuntimeError, "another runtime"):
            run_all_remaining(batch)
        calls.assert_not_called()
        self.assertFalse((batch.output / "a/process_exit.json").exists())
        batch._confirm_previous_runtime_stopped = True
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls:
            current, rows = run_all_remaining(batch)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([row["task"] for row in rows], ["b", "c"])
        self.assertTrue(current._confirm_previous_runtime_stopped)
        self.assertTrue(fixtures.load(current.output / "a/recovery.json")["confirmed_previous_runtime_stopped"])

    def test_raw_keyboard_interrupt_from_checkpoint_is_not_swallowed(self):
        batch = EvaluationBatch(*self.args)
        def interrupted(_):
            raise KeyboardInterrupt()
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()) as calls, \
                self.assertRaises(KeyboardInterrupt):
            run_all_remaining(batch, after_case=interrupted)
        self.assertEqual(calls.call_count, 1)
        self.assertFalse((batch.output / "b").exists())

    def test_unexpected_no_progress_raises_instead_of_looping_forever(self):
        batch = EvaluationBatch(*self.args)
        with patch.object(EvaluationBatch, "run_pending", return_value=[]) as calls, \
                self.assertRaisesRegex(RuntimeError, "no unique task progress"):
            run_all_remaining(batch)
        self.assertEqual(calls.call_count, 1)

    def test_reinvoking_a_finished_session_does_not_duplicate_any_episode(self):
        batch = EvaluationBatch(*self.args)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()):
            current, _ = run_all_remaining(batch)
        protected = {case["task"]: snapshot(current.output / case["task"]) for case in self.prepared}
        with patch("evaluation_batch.subprocess.Popen") as calls:
            completed, added = run_all_remaining(current)
        calls.assert_not_called()
        self.assertEqual(added, [])
        self.assertEqual({case["task"]: snapshot(completed.output / case["task"])
                          for case in self.prepared}, protected)

    def test_explicit_inline_source_is_saved_and_changed_driver_is_rejected(self):
        batch = EvaluationBatch(*self.args)
        source = "# notebook-inlined session driver\n"
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()):
            current, _ = run_all_remaining(batch, driver_source=source)
        self.assertEqual((current.output / "session_driver.py").read_text(), source)
        with self.assertRaisesRegex(ValueError, "Saved session driver differs"):
            run_all_remaining(current, driver_source=source + "# changed\n")

    def test_compact_export_includes_host_policy_and_separate_corrected_analysis(self):
        batch = EvaluationBatch(*self.args)
        with patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()):
            current, _ = run_all_remaining(batch)
        original = (current.output / "batch_summary.json").read_bytes()
        def accounting(run_dir):
            self.assertEqual(run_dir, current.output)
            return {"condition": "A", "results": current.results,
                    "metrics": {"task_successes": 3}, "analysis_only": True, "corrections": []}
        for _ in range(2):
            result = export_session(current, accounting_fn=accounting)
            self.assertEqual(fixtures.load(Path(result["corrected_metrics_json"])), {"task_successes": 3})
            with zipfile.ZipFile(result["summary_zip"]) as archive:
                names = archive.namelist()
                self.assertEqual(len(names), len(set(names)))
                self.assertIn(current.output.name + "/session_policy.json", names)
                self.assertIn(current.output.name + "/session_driver.py", names)
                self.assertIn(current.output.name + "/analysis/corrected_metrics.json", names)
                self.assertIn(current.output.name + "/analysis/corrected_per_task.csv", names)
        self.assertEqual((current.output / "batch_summary.json").read_bytes(), original)

    def test_inline_accounting_provenance_is_frozen_and_exported(self):
        batch = EvaluationBatch(*self.args)
        accounting_source = "# accounting source supplied by the notebook\n"
        with patch.dict(evaluation_session.__dict__, {"ACCOUNTING_SOURCE": accounting_source}), \
                patch("evaluation_batch.subprocess.Popen", side_effect=self.worker()):
            current, _ = run_all_remaining(batch)
            completed, added = run_all_remaining(current)
        self.assertEqual(added, [])
        self.assertEqual((completed.output / "session_accounting.py").read_text(), accounting_source)
        policy = fixtures.load(completed.output / "session_policy.json")
        self.assertEqual(policy["session_accounting_sha256"], hashlib.sha256(accounting_source.encode()).hexdigest())
        result = export_session(completed)
        with zipfile.ZipFile(result["summary_zip"]) as archive:
            self.assertEqual(archive.read(completed.output.name + "/session_accounting.py").decode(), accounting_source)
        with patch.dict(evaluation_session.__dict__, {"ACCOUNTING_SOURCE": accounting_source + "# changed\n"}), \
                self.assertRaisesRegex(ValueError, "Saved session accounting differs"):
            run_all_remaining(completed)
        self.assertEqual((completed.output / "session_accounting.py").read_text(), accounting_source)


if __name__ == "__main__":
    unittest.main()
