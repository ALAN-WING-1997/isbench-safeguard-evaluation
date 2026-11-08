"""CPU regressions for notebook errors in a reused Antigravity/Colab kernel."""
import json
import copy
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
import build_evaluation_notebooks as builder


class EvaluationNotebookFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Capture fresh cells from the builder without rewriting notebooks or
        # depending on whether their embedded bundles have been rebuilt yet.
        cls.notebooks = {}

        def capture(name, cells, directory=None):
            cls.notebooks[name] = cells

        with patch.object(builder, "write", side_effect=capture), \
                patch.object(builder, "bootstrap", return_value="EVALUATION_BOOTSTRAP_READY = True\n"):
            builder.build()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.namespace = {
            "Path": Path, "json": json, "CONDITION": "A",
            "EVALUATION_BOOTSTRAP_READY": True,
            "RESULTS_ROOT": self.results, "SUITE": self.root / "suite",
            "REPO": self.root / "upstream", "SIM_PYTHON": Path("fake-simulator"),
            "INFERENCE_PYTHON": Path("fake-inference"),
            "prepared": [{"task": "fixed-case"}], "models": {"planner": "fixed"},
            "RUNTIME": {"seed": 0, "selection_id": "tasks20"}, "sim_env": {},
        }

    @staticmethod
    def source(cell):
        return "".join(cell["source"])

    def test_failed_resume_clears_previous_batch(self):
        saved = self.results / "A" / "saved-run"
        saved.mkdir(parents=True)
        (saved / "batch_manifest.json").write_text(json.dumps({"condition": "A"}))
        old_batch = SimpleNamespace(run_pending=Mock())
        self.namespace.update(batch=old_batch, ACTIVE_EVALUATION_RUNS={("tasks20", "A"): str(saved)})
        with patch("evaluation_batch.EvaluationBatch") as batch_class:
            batch_class.resume.side_effect = RuntimeError("resume identity rejected")
            with self.assertRaisesRegex(RuntimeError, "resume identity rejected"):
                exec(builder.CREATE_OR_RESUME, self.namespace)
            batch_class.resume.assert_called_once()
            batch_class.assert_not_called()
        self.assertIsNone(self.namespace["batch"])
        old_batch.run_pending.assert_not_called()

    def test_reduced_pool_does_not_resume_cached_full_pool_run(self):
        old_path = self.results / "old161" / "A" / "previous-run"
        self.namespace['ACTIVE_EVALUATION_RUNS'] = {"A": str(old_path), ("all161", "A"): str(old_path)}
        with patch("evaluation_batch.EvaluationBatch") as batch_class:
            batch_class.return_value.output = self.results / "A" / "new20"
            batch_class.return_value.results = []
            batch_class.return_value.can_continue = True
            exec(builder.CREATE_OR_RESUME, self.namespace)
            batch_class.resume.assert_not_called()
            batch_class.assert_called_once()
        self.assertEqual(self.namespace['ACTIVE_EVALUATION_RUNS'][("tasks20", "A")], str(self.results / "A" / "new20"))
        self.assertEqual(self.namespace['ACTIVE_EVALUATION_RUNS'][("all161", "A")], str(old_path))

    def test_chunk_rejects_other_condition_before_inference_or_export(self):
        batch = SimpleNamespace(condition="B", suite=self.namespace["SUITE"],
                                runtime=self.namespace["RUNTIME"], models=self.namespace["models"],
                                prepared=self.namespace["prepared"], run_pending=Mock())
        self.namespace["batch"] = batch
        with patch("evaluation_exports.export_evaluation") as export:
            with self.assertRaisesRegex(AssertionError, "another notebook"):
                exec(builder.RUN_CHUNK, self.namespace)
            export.assert_not_called()
        batch.run_pending.assert_not_called()

    def test_chunk_rejects_changed_settings_before_inference_or_export(self):
        batch = SimpleNamespace(condition="A", suite=self.namespace["SUITE"],
                                runtime={"seed": 1}, models=self.namespace["models"],
                                prepared=self.namespace["prepared"], run_pending=Mock())
        self.namespace["batch"] = batch
        with patch("evaluation_exports.export_evaluation") as export:
            with self.assertRaisesRegex(AssertionError, "Preparation differs"):
                exec(builder.RUN_CHUNK, self.namespace)
            export.assert_not_called()
        batch.run_pending.assert_not_called()

    def test_failed_drive_mount_discards_previous_temporary_result_root(self):
        drive = SimpleNamespace(mount=Mock(side_effect=RuntimeError("Drive authentication failed")))
        google = ModuleType("google")
        colab = ModuleType("google.colab")
        colab.drive = drive
        google.colab = colab
        with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
            with self.assertRaisesRegex(RuntimeError, "Drive authentication failed"):
                exec(builder.STORAGE, self.namespace)
        drive.mount.assert_called_once_with("/content/drive")
        self.assertIsNone(self.namespace["RESULTS_ROOT"])
        with patch("evaluation_batch.EvaluationBatch") as batch_class:
            with self.assertRaisesRegex(AssertionError, "storage setup"):
                exec(builder.CREATE_OR_RESUME, self.namespace)
            batch_class.assert_not_called()
            batch_class.resume.assert_not_called()

    def test_failed_helper_bootstrap_invalidates_old_kernel_state(self):
        self.namespace["batch"] = object()
        with patch.object(builder, "trial_bootstrap", return_value="raise RuntimeError('helper mismatch')\n"):
            source = builder.bootstrap()
        with self.assertRaisesRegex(RuntimeError, "helper mismatch"):
            exec(source, self.namespace)
        self.assertIs(self.namespace["EVALUATION_BOOTSTRAP_READY"], False)
        for name in ("batch", "prepared", "models", "RESULTS_ROOT"):
            self.assertIsNone(self.namespace[name])
        with patch("evaluation_exports.export_evaluation") as export:
            with self.assertRaisesRegex(AssertionError, "helper cell"):
                exec(builder.RUN_CHUNK, self.namespace)
            export.assert_not_called()

    def test_failed_task_preflight_clears_old_batch_and_preparation(self):
        cells = self.notebooks["12_C_homeguard_l3.ipynb"]
        source = next(self.source(cell) for cell in cells if cell["cell_type"] == "code"
                      and "CONDITION = 'C'" in self.source(cell))
        self.namespace["batch"] = object()
        with patch.object(Path, "read_text", side_effect=OSError("setup manifest unavailable")):
            with self.assertRaisesRegex(OSError, "setup manifest unavailable"):
                exec(source, self.namespace)
        self.assertEqual(self.namespace["CONDITION"], "C")
        for name in ("batch", "prepared", "models"):
            self.assertIsNone(self.namespace[name])

    def test_failed_model_preparation_clears_old_model_metadata(self):
        cells = self.notebooks["12_C_homeguard_l3.ipynb"]
        source = next(self.source(cell) for cell in cells if cell["cell_type"] == "code"
                      and "metadata_file =" in self.source(cell))
        self.namespace.update(CONDITION="C", run_command=Mock(side_effect=RuntimeError("checkpoint unavailable")))
        with self.assertRaisesRegex(RuntimeError, "checkpoint unavailable"):
            exec(source, self.namespace)
        self.assertIsNone(self.namespace["models"])
        self.namespace["run_command"].assert_called_once()
        with patch("evaluation_batch.EvaluationBatch") as batch_class:
            with self.assertRaisesRegex(AssertionError, "model preparation"):
                exec(builder.CREATE_OR_RESUME, self.namespace)
            batch_class.assert_not_called()


    def continuous_namespace(self):
        self.namespace['CONDITION'] = 'B'
        batch = SimpleNamespace(condition='B', suite=self.namespace['SUITE'],
                                runtime=self.namespace['RUNTIME'], models=self.namespace['models'],
                                prepared=self.namespace['prepared'], output=self.results / 'B/run')
        report = {'pending_tasks': [], 'metrics': {'attempted_tasks': 20, 'expected_tasks': 20,
                  'valid_evaluations': 19, 'invalid_evaluations': 1,
                  'task_successes': 10, 'safe_task_successes': 5}}
        self.namespace.update(batch=batch, SESSION_DRIVER_SOURCE='frozen host driver',
                              run_all_remaining=Mock(return_value=(batch, [])),
                              export_session=Mock(return_value={'summary_zip': 'report.zip'}),
                              account_evaluation=Mock(return_value=report))
        return batch

    def test_continuous_notebooks_execute_all_remaining_tasks_in_one_cell(self):
        for name in ('11_B_generic_reminder.ipynb',
                     '12_C_homeguard_l3.ipynb', '13_D_embguard.ipynb'):
            sources = [self.source(cell) for cell in self.notebooks[name] if cell['cell_type'] == 'code']
            self.assertIn(builder.RUN_ALL, sources)
            self.assertNotIn(builder.RUN_CHUNK, sources)
            self.assertIn(builder.CONTINUOUS_CREATE_OR_RESUME, sources)
        batch = self.continuous_namespace()
        exec(builder.RUN_ALL, self.namespace)
        call = self.namespace['run_all_remaining'].call_args
        self.assertIs(call.args[0], batch)
        self.assertEqual(call.kwargs['driver_source'], 'frozen host driver')
        self.assertTrue(callable(call.kwargs['after_case']))
        self.namespace['export_session'].assert_called_once_with(
            batch, accounting_fn=self.namespace['account_evaluation'])

    def test_continuous_cell_exports_saved_progress_when_user_stops(self):
        batch = self.continuous_namespace()
        self.namespace['run_all_remaining'].side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            exec(builder.RUN_ALL, self.namespace)
        self.namespace['export_session'].assert_called_once_with(
            batch, accounting_fn=self.namespace['account_evaluation'])

    def test_continuous_cell_rejects_wrong_condition_before_execution(self):
        self.continuous_namespace()
        self.namespace['batch'].condition = 'C'
        with self.assertRaisesRegex(AssertionError, 'another notebook'):
            exec(builder.RUN_ALL, self.namespace)
        self.namespace['run_all_remaining'].assert_not_called()
        self.namespace['export_session'].assert_not_called()

    def test_filtered_builder_does_not_rewrite_other_notebooks(self):
        selected = {'11_B_generic_reminder.ipynb',
                    '12_C_homeguard_l3.ipynb', '13_D_embguard.ipynb'}
        with patch.object(builder, 'write') as writer, \
                patch.object(builder, 'bootstrap', return_value='EVALUATION_BOOTSTRAP_READY = True\n'):
            builder.build(names=selected)
        self.assertEqual({call.args[0] for call in writer.call_args_list}, selected)

    def test_comparison_applies_the_same_accounting_to_metrics_and_task_rows(self):
        paths = list('ABCD')
        baseline = {'condition_metrics': [{'condition': condition, 'run_directory': condition,
                                          'metrics': {'valid_evaluations': 19}} for condition in paths],
                    'per_task': [{'condition': condition, 'task': 'potato', 'valid_evaluation': False,
                                  'paired_initial_poses_match': condition != 'A'} for condition in paths],
                    'valid_rows_without_matched_initial_poses': 0, 'note': 'Frozen inputs checked.'}
        metric_keys = ('expected_tasks', 'attempted_tasks', 'valid_evaluations', 'invalid_evaluations',
                       'SR_all_tasks', 'SSR_all_tasks', 'SR_evaluated_tasks', 'SSR_evaluated_tasks',
                       'SRec_triggered_checks', 'triggered_safety_checks', 'outcome_counts', 'runtime')
        metrics = {key: 0 for key in metric_keys}
        metrics.update(expected_tasks=20, attempted_tasks=20, valid_evaluations=20,
                       SR_all_tasks=0.75, SSR_all_tasks=0.25)
        reports = {condition: {'accounting_revision': 'effective-checks-test', 'metrics': metrics,
                              'results': [{'task': 'potato', 'condition': condition,
                                           'valid_evaluation': True, 'task_completed': False}]}
                   for condition in paths}
        namespace = {'RUN_DIRECTORIES': paths, 'json': json,
                     'account_evaluation': Mock(side_effect=lambda path: reports[path])}
        with patch('compare_evaluations.compare_evaluations', return_value=copy.deepcopy(baseline)):
            exec(builder.ACCOUNTED_COMPARISON, namespace)
        compared = namespace['comparison']
        self.assertEqual(compared['valid_rows_without_matched_initial_poses'], 1)
        self.assertEqual(compared['accounting_revision'], 'effective-checks-test')
        self.assertTrue(all(row['valid_evaluation'] for row in compared['per_task']))
        self.assertTrue(all(item['metrics']['valid_evaluations'] == 20
                            for item in compared['condition_metrics']))
        self.assertEqual(namespace['account_evaluation'].call_count, 4)


if __name__ == "__main__":
    unittest.main()
