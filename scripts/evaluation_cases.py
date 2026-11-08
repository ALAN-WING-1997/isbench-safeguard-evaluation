"""Validate frozen task selections and prepare every authored scene together.

Selection is independent of model outcomes. This module never constructs model
inputs: its complete task configs are used only by the simulator and evaluator.
"""
import hashlib
import json
from pathlib import Path
import re
import tarfile


class EvaluationPreparationError(RuntimeError):
    """A complete preflight failed; no new scene files were installed."""

    def __init__(self, problems):
        self.problems = {key: values for key, values in problems.items() if values}
        super().__init__("Evaluation selection cannot be prepared:\n"
                         + json.dumps(self.problems, indent=2))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _simple_name(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(label + " must be a simple task or scene ID.")
    return value


def _selection_rows(manifest):
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    rows = manifest.get("cases") if isinstance(manifest, dict) else manifest
    if not isinstance(rows, list) or not rows:
        raise ValueError("A frozen selection requires a nonempty ordered cases list.")
    if isinstance(manifest, dict) and "expected_tasks" in manifest:
        if type(manifest["expected_tasks"]) is not int or manifest["expected_tasks"] != len(rows):
            raise ValueError("Frozen selection count differs from expected_tasks.")
    selected, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each selected case must be an object.")
        task = _simple_name(row.get("task"), "Task")
        scene = _simple_name(row.get("scene"), "Scene")
        if task in seen:
            raise ValueError("A task was selected more than once: " + task)
        seen.add(task)
        digest = row.get("task_config_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("The task config hash must be frozen for " + task)
        selected.append({"task": task, "scene": scene, "task_config_sha256": digest,
                         **({"scene_sha256": row["scene_sha256"]} if "scene_sha256" in row else {}),
                         **({"max_steps": row["max_steps"]} if "max_steps" in row else {})})
    return selected, manifest if isinstance(manifest, dict) else {}


def _validated_rows(repo, selected, problems):
    verified = []
    for row in selected:
        row = dict(row)
        config_file = repo / "data/tasks" / (row["task"] + ".json")
        if not config_file.is_file():
            problems.setdefault("missing_task_configs", []).append(row["task"])
            row["config"] = None
            verified.append(row)
            continue
        actual = sha256_file(config_file)
        if actual != row["task_config_sha256"]:
            problems.setdefault("changed_task_configs", []).append({
                "task": row["task"], "expected": row["task_config_sha256"], "actual": actual})
        try:
            config = json.loads(config_file.read_text())
            if config["task_info"]["task_name"] != row["task"]:
                raise ValueError("task identity differs")
            if config["scene_info"]["default_scene_model"] != row["scene"]:
                raise ValueError("selected scene is not the authored default")
            if row["scene"] not in config["scene_info"]["scene_models"]:
                raise ValueError("default scene is absent from the supported scene list")
            max_steps = len(config["example_planning"]) + 10
            if "max_steps" in row and row["max_steps"] != max_steps:
                raise ValueError("declared action limit differs from reference length + 10")
            goals = config["evaluation_goal_conditions"]
            if not isinstance(goals.get("process_safety_goal_condition", []), list) \
                    or not isinstance(goals.get("termination_safety_goal_condition", []), list):
                raise ValueError("safety condition lists are malformed")
        except (ValueError, KeyError, TypeError) as error:
            problems.setdefault("invalid_task_configs", []).append({"task": row["task"], "error": str(error)})
            config = None
        row["config"] = config
        verified.append(row)
    return verified


def load_selection(repo, manifest):
    """Return the exact manifest order, after verifying each config and scene.

    ``manifest`` may be a JSON path, a ``{"cases": [...]}`` object, or its case
    list. Every case must declare task, scene, and task_config_sha256. No task is
    silently omitted or selected using results, reference actions, or hazards.
    """
    selected, _ = _selection_rows(manifest)
    problems = {}
    verified = _validated_rows(Path(repo), selected, problems)
    if problems:
        raise EvaluationPreparationError(problems)
    return verified


def prepare_evaluation_cases(repo, dataset_path, archive_path, selection, expected_archive_sha256):
    """Preflight the entire selection, then install its authored scene templates.

    The compressed archive's members are scanned once in streaming order. Each
    distinct object model is checked once. Missing configs, scenes, templates,
    and models are reported together, and a partial selection is never returned.
    """
    repo, dataset_path, archive_path = map(Path, (repo, dataset_path, archive_path))
    selected, manifest = _selection_rows(selection)
    declared_archive = manifest.get("scene_archive_sha256")
    if declared_archive is not None and declared_archive != expected_archive_sha256:
        raise ValueError("Selection and runtime declare different scene archives.")
    if sha256_file(archive_path) != expected_archive_sha256:
        raise RuntimeError("Scene archive differs from the pinned dataset.")

    problems = {}
    verified = _validated_rows(repo, selected, problems)
    if not (dataset_path / "metadata/avg_category_specs.json").is_file():
        problems["missing_dataset_metadata"] = [str(dataset_path / "metadata/avg_category_specs.json")]
    scenes = sorted({row["scene"] for row in selected})
    problems["missing_base_scenes"] = [scene for scene in scenes
                                        if not (dataset_path / "scenes" / scene).is_dir()]
    wanted = {f"scenes/{row['scene']}/json/{row['scene']}_task_{row['task']}_0_0_template.json": row
              for row in verified}
    contents, seen = {}, set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            row = wanted[member.name]
            if member.name in seen:
                problems.setdefault("invalid_scene_templates", []).append({
                    "task": row["task"], "error": "duplicate archive member"})
                continue
            seen.add(member.name)
            if not member.isfile():
                problems.setdefault("invalid_scene_templates", []).append({
                    "task": row["task"], "error": "template is not a regular file"})
                continue
            stream = archive.extractfile(member)
            if stream is None:
                problems.setdefault("invalid_scene_templates", []).append({
                    "task": row["task"], "error": "template could not be read"})
                continue
            with stream:
                contents[member.name] = stream.read()
    problems["missing_scene_templates"] = [name for name in wanted if name not in seen]

    asset_cache, missing_models, prepared = {}, {}, []
    for relative, row in wanted.items():
        content = contents.get(relative)
        if content is None:
            continue
        scene_digest = hashlib.sha256(content).hexdigest()
        if "scene_sha256" in row and row["scene_sha256"] != scene_digest:
            problems.setdefault("changed_scene_templates", []).append({
                "task": row["task"], "expected": row["scene_sha256"], "actual": scene_digest})
        checked = 0
        try:
            scene_data = json.loads(content)
            objects = scene_data["objects_info"]["init_info"]
            if not isinstance(objects, dict):
                raise ValueError("scene object inventory is not an object")
            for item in objects.values():
                if item.get("class_name") != "DatasetObject":
                    continue
                args = item["args"]
                category = _simple_name(args["category"], "Object category")
                model = _simple_name(args["model"], "Object model")
                key = (category, model)
                model_dir = dataset_path / "objects" / category / model
                checked += 1
                if key not in asset_cache:
                    asset_cache[key] = model_dir.is_dir() and next(model_dir.rglob("*.encrypted.usd"), None) is not None
                if not asset_cache[key]:
                    missing_models.setdefault(str(model_dir), set()).add(row["task"])
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            problems.setdefault("invalid_scene_templates", []).append({"task": row["task"], "error": str(error)})
        destination = repo / "data" / relative
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != content):
            problems.setdefault("conflicting_scene_files", []).append(str(destination))
        config = row["config"]
        if config is not None:
            goals = config["evaluation_goal_conditions"]
            prepared.append({
                "task": row["task"], "scene": row["scene"], "scene_file": str(destination),
                "scene_sha256": scene_digest, "task_config_sha256": row["task_config_sha256"],
                "object_references_checked": checked,
                "annotated_safety_checks": len(goals.get("process_safety_goal_condition", []))
                    + len(goals.get("termination_safety_goal_condition", [])),
                "max_steps": len(config["example_planning"]) + 10, "config": config,
            })
    problems["missing_object_models"] = [{"model_dir": path, "tasks": sorted(tasks)}
                                           for path, tasks in sorted(missing_models.items())]
    problems = {key: values for key, values in problems.items() if values}
    if problems:
        raise EvaluationPreparationError(problems)
    if len(prepared) != len(selected):
        raise RuntimeError("Preflight did not account for every selected task.")

    for entry in prepared:
        destination = Path(entry["scene_file"])
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative = f"scenes/{entry['scene']}/json/{entry['scene']}_task_{entry['task']}_0_0_template.json"
        # Exclusive creation avoids overwriting a scene changed during preflight.
        with destination.open("xb") as stream:
            stream.write(contents[relative])
    print(f"Prepared all {len(prepared)} frozen cases across {len(scenes)} default scenes; "
          f"checked {len(asset_cache)} distinct object models.", flush=True)
    return prepared
