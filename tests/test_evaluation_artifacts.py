"""CPU tests for portable comparisons and compact evidence exports."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_evaluations import compare_evaluations
from evaluation_exports import export_evaluation
from evaluation_metrics import aggregate_evaluation
from evaluation_protocol import CONDITIONS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class EvaluationArtifactTests(unittest.TestCase):
    def fixture(self, directory, condition="A", omit_infrastructure=False):
        run = Path(directory) / condition
        definitions = {
            "task_success": {"evaluation_goal_conditions": {
                "process_safety_goal_condition": [{"risk_type": "Fire"}],
                "termination_safety_goal_condition": [{"risk_type": "Electrical Shock"}]}},
            "task_plan_error": {"evaluation_goal_conditions": {
                "process_safety_goal_condition": [{"risk_type": "Fire"}],
                "termination_safety_goal_condition": []}},
            "task_infrastructure": {"evaluation_goal_conditions": {
                "process_safety_goal_condition": [],
                "termination_safety_goal_condition": [{"risk_type": "Water Spill"}]}},
        }
        tasks = [{"task": task, "scene": "Scene_a" if task == "task_infrastructure" else "Scene_z",
                  "scene_sha256": "a" * 64, "task_config_sha256": digest(config), "max_steps": 11}
                 for task, config in definitions.items()]
        pins = json.loads((ROOT / "experiments/models.json").read_text())
        roles = ["planner"] + ([CONDITIONS[condition]["guard"]] if CONDITIONS[condition]["guard"] else [])
        models = {role: {**pins[role], "model_path": str(run / "models" / role)} for role in roles}
        manifest = {"condition": condition, "condition_definition": copy.deepcopy(CONDITIONS[condition]),
                    "batch_controller": "evaluation_batch_v1",
                    "upstream_git_commit": "6a406e162feca1c4c455d9c9f1f58832edbdc1fd",
                    "upstream_dependency_sha256": {"og_ego_prim/utils/prompts.py": "d" * 64},
                    "runtime": {"seed": 0, "inference": {"max_new_tokens": 8192, "do_sample": True}},
                    "models": models,
                    "tasks": tasks, "script_sha256": {"runner.py": "c" * 64},
                    "prepared_config_sha256": {task: digest(config) for task, config in definitions.items()}}
        rows = []
        for case in tasks:
            task = case["task"]
            if omit_infrastructure and task == "task_infrastructure":
                continue
            success = task == "task_success"
            valid = task != "task_infrastructure"
            row = {"task": task, "condition": condition, "scene": case["scene"],
                   "valid_evaluation": valid, "task_completed": success if valid else None,
                   "safe_task_completed": success if valid else None,
                   "termination_reason": "done" if success else "plan_error" if valid else "process_error",
                   "triggered_safety_checks": 2 if success else 0,
                   "passed_safety_checks": 2 if success else 0,
                   "untriggered_safety_checks": 0 if success else 1,
                   "elapsed_seconds": 10, "model_calls": 1, "guard_model_calls": 0,
                   "token_budget_hits": int(task == "task_plan_error"),
                   "process_returncode": 0 if valid else -11, "interrupted": False, "timed_out": False,
                   "risk_check_counts": {risk: {"triggered_checks": 1, "passed_checks": 1}
                                         for risk in ("Fire", "Electrical Shock")} if success else {}}
            rows.append(row)
            write_json(run / task / "summary.json", row)
            write_json(run / task / "initial_state.json", {"objects": {
                "object": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}}})
        prepared = [{**case, "config": definitions[case["task"]]} for case in tasks]
        write_json(run / "batch_manifest.json", manifest)
        write_json(run / "upstream_dependencies.json", {"git_commit": manifest["upstream_git_commit"],
                                                       "sha256": manifest["upstream_dependency_sha256"],
                                                       "policy": "Frozen upstream dependency content."})
        write_json(run / "task_definitions.json", definitions)
        write_json(run / "batch_summary.json", {"condition": condition, "results": rows,
                                                "metrics": aggregate_evaluation(rows, prepared)})
        return run

    def four_runs(self, directory):
        return [self.fixture(directory, condition, omit_infrastructure=condition == "D") for condition in "ABCD"]

    def snapshot(self, directory):
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file()}

    def test_exported_partial_comparison_is_standalone_and_retains_selected_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            source, downloaded = Path(temp) / "source", Path(temp) / "downloaded"
            paths = self.four_runs(source)
            for path in paths:
                exported = export_evaluation(path)
                with zipfile.ZipFile(exported["summary_zip"]) as archive:
                    archive.extractall(downloaded)
            # Only the extracted reports are supplied: no task repository or simulator is needed.
            compared = compare_evaluations([downloaded / condition for condition in "ABCD"])
            self.assertEqual(compared["conditions"], list("ABCD"))
            self.assertEqual(len(compared["per_task"]), 12)
            metrics = {row["condition"]: row["metrics"] for row in compared["condition_metrics"]}
            for condition, metric in metrics.items():
                with self.subTest(condition=condition):
                    self.assertEqual((metric["expected_tasks"], metric["valid_evaluations"]), (3, 2))
                    self.assertIsNone(metric["SR_all_tasks"])
                    self.assertIsNone(metric["SSR_all_tasks"])
                    self.assertEqual(metric["SR_evaluated_tasks"], 0.5)
                    self.assertEqual(metric["SR_all_tasks_bounds"], {"lower": 1 / 3, "upper": 2 / 3})
                    self.assertEqual(metric["annotated_safety_checks_selected"], 4)
                    self.assertEqual(metric["scene_coverage"], {"Scene_a": 1, "Scene_z": 2})
                    self.assertEqual(metric["risk_coverage"]["Water Spill"]["evaluated_tasks"], 0)
                    self.assertEqual(metric["risk_coverage"]["Fire"]["selected_tasks"], 2)
                    self.assertEqual(metric["outcome_counts"]["planner_output_failure"], 1)
            self.assertEqual((metrics["A"]["invalid_evaluations"], metrics["A"]["unattempted_tasks"]), (1, 0))
            self.assertEqual((metrics["D"]["invalid_evaluations"], metrics["D"]["unattempted_tasks"]), (0, 1))

    def test_primary_scores_require_complete_valid_coverage_and_count_model_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            for path in paths:
                summary = {"task": "task_infrastructure", "condition": path.name, "scene": "Scene_a",
                           "valid_evaluation": True, "task_completed": False, "safe_task_completed": False,
                           "termination_reason": "guard_format_error", "triggered_safety_checks": 0,
                           "passed_safety_checks": 0, "elapsed_seconds": 10}
                write_json(path / "task_infrastructure/summary.json", summary)
            compared = compare_evaluations(paths)
            for row in compared["condition_metrics"]:
                metric = row["metrics"]
                self.assertTrue(metric["all_tasks_evaluated"])
                self.assertEqual(metric["SR_all_tasks"], 1 / 3)
                self.assertEqual(metric["SSR_all_tasks"], 1 / 3)
                self.assertEqual(metric["SR_evaluated_tasks"], 1 / 3)
                self.assertEqual(metric["SR_all_tasks_bounds"], {"lower": 1 / 3, "upper": 1 / 3})
                self.assertEqual(metric["outcome_counts"], {"task_success": 1, "planner_output_failure": 1,
                                                          "guard_output_failure": 1})

    def test_comparison_rejects_differing_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            changed = json.loads((paths[-1] / "batch_manifest.json").read_text())
            changed["runtime"]["inference"]["max_new_tokens"] = 4096
            write_json(paths[-1] / "batch_manifest.json", changed)
            with self.assertRaisesRegex(ValueError, "protocol.*budgets"):
                compare_evaluations(paths)

    def test_comparison_rejects_changed_frozen_task_definitions(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            definition_path = paths[-1] / "task_definitions.json"
            definitions = json.loads(definition_path.read_text())
            definitions["task_infrastructure"]["evaluation_goal_conditions"]["termination_safety_goal_condition"] = []
            write_json(definition_path, definitions)
            with self.assertRaisesRegex(ValueError, "task definitions.*frozen"):
                compare_evaluations(paths)

    def test_comparison_rejects_changed_guard_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            for index, role in ((2, "homeguard"), (3, "embguard")):
                with self.subTest(role=role):
                    manifest_path = paths[index] / "batch_manifest.json"
                    original = json.loads(manifest_path.read_text())
                    changed = copy.deepcopy(original)
                    changed["models"][role]["revision"] = "0" * 40
                    write_json(manifest_path, changed)
                    with self.assertRaisesRegex(ValueError, "[Mm]odel|[Pp]in|[Rr]evision|[Cc]heckpoint"):
                        compare_evaluations(paths)
                    write_json(manifest_path, original)

    def test_comparison_rejects_changed_c_condition_definition(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            manifest_path = paths[2] / "batch_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["condition_definition"]["prompt"] = "v1"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "[Cc]ondition|[Pp]rotocol"):
                compare_evaluations(paths)

    def test_comparison_rejects_changed_upstream_hash_or_git_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            manifest_path = paths[-1] / "batch_manifest.json"
            sidecar_path = paths[-1] / "upstream_dependencies.json"
            original_manifest = json.loads(manifest_path.read_text())
            original_sidecar = json.loads(sidecar_path.read_text())
            mutations = (("upstream_dependency_sha256", "sha256", {"og_ego_prim/utils/prompts.py": "e" * 64}),
                         ("upstream_git_commit", "git_commit", "0" * 40))
            for manifest_key, sidecar_key, value in mutations:
                with self.subTest(changed=manifest_key):
                    manifest, sidecar = copy.deepcopy(original_manifest), copy.deepcopy(original_sidecar)
                    manifest[manifest_key] = value
                    sidecar[sidecar_key] = value
                    write_json(manifest_path, manifest)
                    write_json(sidecar_path, sidecar)
                    with self.assertRaisesRegex(ValueError, "[Uu]pstream|[Dd]ependenc"):
                        compare_evaluations(paths)
                    write_json(manifest_path, original_manifest)
                    write_json(sidecar_path, original_sidecar)

    def test_comparison_rejects_modified_upstream_sidecar_and_empty_dependency_maps(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            sidecar_path = paths[-1] / "upstream_dependencies.json"
            sidecar = json.loads(sidecar_path.read_text())
            sidecar["sha256"]["og_ego_prim/utils/prompts.py"] = "e" * 64
            write_json(sidecar_path, sidecar)
            with self.assertRaisesRegex(ValueError, "[Uu]pstream|[Dd]ependenc"):
                compare_evaluations(paths)
            for path in paths:
                manifest_path, sidecar_path = path / "batch_manifest.json", path / "upstream_dependencies.json"
                manifest, sidecar = json.loads(manifest_path.read_text()), json.loads(sidecar_path.read_text())
                manifest["upstream_dependency_sha256"], sidecar["sha256"] = {}, {}
                write_json(manifest_path, manifest)
                write_json(sidecar_path, sidecar)
            with self.assertRaisesRegex(ValueError, "[Uu]pstream|[Dd]ependenc"):
                compare_evaluations(paths)

    def test_comparison_rejects_pilot_controller_and_missing_required_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            manifest_path = paths[2] / "batch_manifest.json"
            original = json.loads(manifest_path.read_text())
            mutations = (("controller", lambda manifest: manifest.update(batch_controller="trial_batch_v1")),
                         ("required guard", lambda manifest: manifest["models"].pop("homeguard")))
            for description, mutate in mutations:
                with self.subTest(description=description):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    write_json(manifest_path, changed)
                    with self.assertRaises(ValueError):
                        compare_evaluations(paths)
            write_json(manifest_path, original)

    def test_comparison_rejects_episode_identity_mismatch_and_missing_condition(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.four_runs(temp)
            with self.assertRaisesRegex(ValueError, "A, B, C and D"):
                compare_evaluations(paths[:3])
            summary_path = paths[-1] / "task_success/summary.json"
            summary = json.loads(summary_path.read_text())
            summary["condition"] = "A"
            write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "Episode identity"):
                compare_evaluations(paths)

    def test_compact_zip_and_csv_preserve_failure_rows_and_leave_raw_evidence_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.fixture(temp)
            write_json(run / "resume_events.json", [{"event": "resume", "seed": 0}])
            write_json(run / "controller_state.json", {"next_task": "task_infrastructure"})
            write_json(run / "task_plan_error/report.json", {"process_safety_goal_condition": []})
            write_json(run / "task_plan_error/diagnostics/0001_response.json", {"raw_output": "incomplete"})
            write_json(run / "task_plan_error/diagnostics/0001_request.json", {"prompt": "original prompt"})
            (run / "task_plan_error/initial_view.png").write_bytes(b"synthetic image evidence")
            (run / "task_plan_error/episode.log").write_text("original inference log\n")
            (run / "code").mkdir()
            (run / "code/runner.py").write_text("# frozen synthetic code\n")
            before = self.snapshot(run)
            with patch("evaluation_exports.export_results") as full_export:
                exported = export_evaluation(run)
                full_export.assert_not_called()
            self.assertNotIn("full_evidence", exported)
            self.assertEqual(exported["raw_directory"], str(run.resolve()))
            self.assertEqual(self.snapshot(run), before)
            rows = list(csv.DictReader(io.StringIO(Path(exported["per_task_csv"]).read_text())))
            by_task = {row["task"]: row for row in rows}
            self.assertEqual(len(rows), 3)
            self.assertEqual(by_task["task_plan_error"]["valid_evaluation"], "True")
            self.assertEqual(by_task["task_plan_error"]["termination_reason"], "plan_error")
            self.assertEqual(by_task["task_plan_error"]["task_completed"], "False")
            self.assertEqual(by_task["task_infrastructure"]["valid_evaluation"], "False")
            self.assertEqual(by_task["task_infrastructure"]["process_returncode"], "-11")
            metrics = json.loads(Path(exported["metrics_json"]).read_text())
            self.assertIsNone(metrics["SR_all_tasks"])
            self.assertEqual(metrics["outcome_counts"]["infrastructure_or_interruption"], 1)
            with zipfile.ZipFile(exported["summary_zip"]) as archive:
                names = set(archive.namelist())
                for relative in ("task_definitions.json", "resume_events.json", "controller_state.json", "upstream_dependencies.json",
                                 "code/runner.py", "task_plan_error/report.json", "task_infrastructure/summary.json"):
                    self.assertIn(run.name + "/" + relative, names)
                    self.assertEqual(archive.read(run.name + "/" + relative), (run / relative).read_bytes())
                self.assertEqual(archive.read(run.name + "/analysis/per_task.csv"), Path(exported["per_task_csv"]).read_bytes())
                self.assertEqual(archive.read(run.name + "/analysis/metrics.json"), Path(exported["metrics_json"]).read_bytes())
                self.assertFalse(any(name.endswith(("_request.json", "_response.json", ".png", ".log")) for name in names))
            self.assertEqual(len(list(Path(exported["summary_zip"]).parent.glob("*.zip"))), 1)

    def test_full_evidence_export_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.fixture(temp)
            full_paths = {"full_zip": "explicit-full.zip", "parts": ["explicit-full.part001"]}
            with patch("evaluation_exports.export_results", return_value=full_paths) as full_export:
                exported = export_evaluation(run, full_evidence=True)
                full_export.assert_called_once_with(run.resolve())
            self.assertEqual(exported["full_evidence"], full_paths)

    def test_real_full_export_keeps_evaluation_compact_zip_and_analysis_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.fixture(temp)
            write_json(run / "resume_events.json", [{"event": "resume"}])
            raw_response = run / "task_plan_error/diagnostics/0001_response.json"
            write_json(raw_response, {"raw_output": "saved incomplete output"})
            original_raw = self.snapshot(run)
            compact = export_evaluation(run)
            sidecar_contents = {key: Path(compact[key]).read_bytes() for key in ("per_task_csv", "metrics_json")}
            with zipfile.ZipFile(compact["summary_zip"]) as archive:
                compact_contents = {name: archive.read(name) for name in archive.namelist()}
            exported = export_evaluation(run, full_evidence=True)
            self.assertEqual(Path(exported["summary_zip"]).name, run.name + "_evaluation_summary.zip")
            self.assertNotEqual(exported["summary_zip"], exported["full_evidence"]["summary_zip"])
            self.assertEqual(self.snapshot(run), original_raw)
            for key, content in sidecar_contents.items():
                self.assertEqual(Path(exported[key]).read_bytes(), content)
            with zipfile.ZipFile(exported["summary_zip"]) as archive:
                self.assertEqual({name: archive.read(name) for name in archive.namelist()}, compact_contents)
                self.assertIn(run.name + "/task_definitions.json", archive.namelist())
                self.assertIn(run.name + "/resume_events.json", archive.namelist())
                self.assertIn(run.name + "/upstream_dependencies.json", archive.namelist())
                self.assertIn(run.name + "/analysis/per_task.csv", archive.namelist())
                self.assertIn(run.name + "/analysis/metrics.json", archive.namelist())
            with zipfile.ZipFile(exported["full_evidence"]["full_zip"]) as archive:
                self.assertEqual(archive.read(run.name + "/task_plan_error/diagnostics/0001_response.json"),
                                 raw_response.read_bytes())


if __name__ == "__main__":
    unittest.main()
