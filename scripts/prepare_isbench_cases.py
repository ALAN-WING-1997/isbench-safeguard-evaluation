"""Prepare authored task scenes and check assets without executing reference plans."""
import hashlib
import json
from pathlib import Path
import tarfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_cases(repo, dataset_path, archive_path, cases, expected_archive_sha256):
    repo, dataset_path, archive_path = map(Path, (repo, dataset_path, archive_path))
    if not cases or len({c["task"] for c in cases}) != len(cases):
        raise ValueError("Expected a nonempty list of unique task cases.")
    if sha256_file(archive_path) != expected_archive_sha256:
        raise RuntimeError("Scene archive differs from the pinned dataset.")
    if not (dataset_path / "metadata/avg_category_specs.json").is_file():
        raise RuntimeError("Dataset metadata is missing; complete scene-data setup first.")
    prepared = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for case in cases:
            task_name, scene = case["task"], case["scene"]
            if any("/" in value or "\\" in value or value in (".", "..")
                   for value in (task_name, scene)):
                raise ValueError("Task and scene IDs must be simple names.")
            config_file = repo / "data/tasks" / f"{task_name}.json"
            config = json.loads(config_file.read_text())
            if config["task_info"]["task_name"] != task_name:
                raise RuntimeError("Task identity differs: " + task_name)
            if config["scene_info"]["default_scene_model"] != scene:
                raise RuntimeError("Default scene differs from the selected case: " + task_name)
            if not (dataset_path / "scenes" / scene).is_dir():
                raise RuntimeError("Missing base scene assets: " + scene)
            relative = f"scenes/{scene}/json/{scene}_task_{task_name}_0_0_template.json"
            member = archive.getmember(relative)
            if not member.isfile():
                raise RuntimeError("Expected a regular scene JSON: " + relative)
            with archive.extractfile(member) as stream:
                content = stream.read()
            scene_data = json.loads(content)
            missing, checked = [], 0
            for item in scene_data["objects_info"]["init_info"].values():
                if item.get("class_name") != "DatasetObject":
                    continue
                args = item["args"]
                model_dir = dataset_path / "objects" / args["category"] / args["model"]
                checked += 1
                if not model_dir.is_dir() or not any(model_dir.rglob("*.encrypted.usd")):
                    missing.append(str(model_dir))
            if missing:
                raise RuntimeError(f"Missing objects for {task_name}: {missing[:10]}")
            destination = repo / "data" / relative
            if destination.exists() and destination.read_bytes() != content:
                raise RuntimeError("Existing scene differs; refusing to overwrite: " + str(destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            goals = config["evaluation_goal_conditions"]
            entry = {
                "task": task_name, "scene": scene, "scene_file": str(destination),
                "scene_sha256": hashlib.sha256(content).hexdigest(),
                "task_config_sha256": sha256_file(config_file),
                "object_references_checked": checked,
                "annotated_safety_checks": len(goals.get("process_safety_goal_condition", []))
                    + len(goals.get("termination_safety_goal_condition", [])),
                "max_steps": len(config["example_planning"]) + 10,
                "config": config,
            }
            prepared.append(entry)
            print(f"Prepared: {task_name} | {scene} | max actions: {entry['max_steps']}"
                  f" | formal checks: {entry['annotated_safety_checks']}", flush=True)
    return prepared
