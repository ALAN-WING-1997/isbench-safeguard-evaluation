"""CPU preflight tests for frozen selections and complete asset coverage."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation_cases import EvaluationPreparationError, load_selection, prepare_evaluation_cases


class EvaluationCaseTests(unittest.TestCase):
    def fixture(self, temp):
        root = Path(temp)
        repo, dataset, archive = root / "repo", root / "dataset", root / "scenes.tar.gz"
        (repo / "data/tasks").mkdir(parents=True)
        (dataset / "metadata").mkdir(parents=True)
        (dataset / "metadata/avg_category_specs.json").write_text("{}")
        rows = []
        scene_data = {"objects_info": {"init_info": {
            "one": {"class_name": "DatasetObject", "args": {"category": "sponge", "model": "shared"}},
            "two": {"class_name": "DatasetObject", "args": {"category": "sponge", "model": "shared"}},
        }}}
        members = []
        for task, scene, refs in (("z_task", "Scene_z", 2), ("a_task", "Scene_a", 1)):
            config = {"task_info": {"task_name": task},
                      "scene_info": {"default_scene_model": scene, "scene_models": [scene, "Other_scene"]},
                      "planning_context": {"task_instruction": "Do the task."},
                      "evaluation_goal_conditions": {"process_safety_goal_condition": [{"secret": "gold predicate"}],
                                                     "termination_safety_goal_condition": []},
                      "example_planning": [{"action": "DONE"}] * refs}
            data = json.dumps(config).encode()
            (repo / "data/tasks" / (task + ".json")).write_bytes(data)
            rows.append({"task": task, "scene": scene, "task_config_sha256": hashlib.sha256(data).hexdigest()})
            (dataset / "scenes" / scene).mkdir(parents=True)
            relative = f"scenes/{scene}/json/{scene}_task_{task}_0_0_template.json"
            members.append((relative, json.dumps(scene_data).encode()))
        model = dataset / "objects/sponge/shared"
        model.mkdir(parents=True)
        (model / "model.encrypted.usd").write_bytes(b"asset")
        self.write_archive(archive, list(reversed(members)))
        manifest = {"selection_id": "frozen", "cases": rows,
                    "scene_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        return repo, dataset, archive, manifest, members

    def write_archive(self, path, members):
        with tarfile.open(path, "w:gz") as archive:
            for name, data in members:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    def test_exact_selection_order_defaults_hashes_and_cached_models(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, dataset, archive, manifest, _ = self.fixture(temp)
            manifest_path = Path(temp) / "selection.json"
            manifest_path.write_text(json.dumps(manifest))
            selected = load_selection(repo, manifest_path)
            self.assertEqual([row["task"] for row in selected], ["z_task", "a_task"])
            original = copy.deepcopy(manifest)
            real_rglob = Path.rglob
            calls = []
            def counted_rglob(path, pattern):
                calls.append(str(path))
                return real_rglob(path, pattern)
            with patch.object(Path, "rglob", counted_rglob):
                prepared = prepare_evaluation_cases(repo, dataset, archive, selected, manifest["scene_archive_sha256"])
            self.assertEqual(len(calls), 1)
            self.assertEqual([row["task"] for row in prepared], ["z_task", "a_task"])
            self.assertEqual([row["scene"] for row in prepared], ["Scene_z", "Scene_a"])
            self.assertEqual([row["max_steps"] for row in prepared], [12, 11])
            self.assertTrue(all(row["object_references_checked"] == 2 for row in prepared))
            self.assertTrue(all(Path(row["scene_file"]).is_file() for row in prepared))
            self.assertEqual(manifest, original)

    def test_changed_config_or_nondefault_scene_rejects_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, _, _, manifest, _ = self.fixture(temp)
            path = repo / "data/tasks/z_task.json"
            config = json.loads(path.read_text())
            config["planning_context"]["task_instruction"] = "Changed"
            path.write_text(json.dumps(config))
            with self.assertRaises(EvaluationPreparationError) as caught:
                load_selection(repo, manifest)
            self.assertEqual(caught.exception.problems["changed_task_configs"][0]["task"], "z_task")
            other = copy.deepcopy(manifest)
            other["cases"][1]["scene"] = "Other_scene"
            with self.assertRaisesRegex(EvaluationPreparationError, "authored default"):
                load_selection(repo, other)

    def test_aggregates_missing_tasks_scenes_templates_and_models_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, dataset, archive, manifest, members = self.fixture(temp)
            (repo / "data/tasks/a_task.json").unlink()
            (dataset / "scenes/Scene_a").rmdir()
            (dataset / "objects/sponge/shared/model.encrypted.usd").unlink()
            self.write_archive(archive, members[:1])
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest["scene_archive_sha256"] = digest
            with self.assertRaises(EvaluationPreparationError) as caught:
                prepare_evaluation_cases(repo, dataset, archive, manifest, digest)
            problems = caught.exception.problems
            self.assertEqual(problems["missing_task_configs"], ["a_task"])
            self.assertEqual(problems["missing_base_scenes"], ["Scene_a"])
            self.assertEqual(len(problems["missing_scene_templates"]), 1)
            self.assertEqual(problems["missing_object_models"][0]["tasks"], ["z_task"])
            self.assertFalse((repo / "data/scenes").exists())

    def test_archive_hash_mismatch_stops_before_installing_any_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, dataset, archive, manifest, _ = self.fixture(temp)
            with self.assertRaisesRegex(RuntimeError, "pinned dataset"):
                prepare_evaluation_cases(repo, dataset, archive, manifest["cases"], "0" * 64)
            self.assertFalse((repo / "data/scenes").exists())

    def test_subset_is_exact_and_conflicting_scene_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, dataset, archive, manifest, _ = self.fixture(temp)
            selected = manifest["cases"][1:]
            prepared = prepare_evaluation_cases(repo, dataset, archive, selected, manifest["scene_archive_sha256"])
            self.assertEqual([row["task"] for row in prepared], ["a_task"])
            self.assertFalse((repo / "data/scenes/Scene_z").exists())
            path = Path(prepared[0]["scene_file"])
            path.write_bytes(b"existing scene evidence")
            with self.assertRaisesRegex(EvaluationPreparationError, "conflicting_scene_files"):
                prepare_evaluation_cases(repo, dataset, archive, manifest, manifest["scene_archive_sha256"])
            self.assertEqual(path.read_bytes(), b"existing scene evidence")
            self.assertFalse((repo / "data/scenes/Scene_z").exists())

    def test_duplicate_unfrozen_or_path_traversal_selection_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, _, _, manifest, _ = self.fixture(temp)
            for rows in ([manifest["cases"][0]] * 2,
                         [{"task": "z_task", "scene": "Scene_z"}],
                         [{**manifest["cases"][0], "task": "../z_task"}]):
                with self.assertRaises(ValueError):
                    load_selection(repo, rows)

    def test_declared_selection_count_cannot_be_silently_reduced(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, _, _, manifest, _ = self.fixture(temp)
            manifest["expected_tasks"] = 2
            manifest["cases"].pop()
            with self.assertRaisesRegex(ValueError, "expected_tasks"):
                load_selection(repo, manifest)


if __name__ == "__main__":
    unittest.main()
