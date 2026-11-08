"""Compare one explicitly selected large run per condition, retaining coverage."""
import json
from pathlib import Path

from compare_trials import compare as compare_paired
from evaluation_metrics import aggregate_evaluation
from evaluation_batch import _digest
from evaluation_protocol import CONDITIONS


def compare_evaluations(paths):
    paths = [Path(path) for path in paths]
    paired = compare_paired(paths)
    if set(paired["conditions"]) != set("ABCD") or len(paths) != 4:
        raise ValueError("Provide one full-task run for each of A, B, C and D.")
    model_lock = json.loads((Path(__file__).resolve().parents[1] / "experiments/models.json").read_text())
    upstream_identity = None
    condition_metrics = []
    for path in paths:
        manifest = json.loads((path / "batch_manifest.json").read_text())
        condition = manifest["condition"]
        current_upstream = (manifest.get("upstream_git_commit"), manifest.get("upstream_dependency_sha256"))
        if not current_upstream[0] or not current_upstream[1]:
            raise ValueError("Missing frozen upstream benchmark dependencies: " + str(path))
        if upstream_identity is None:
            upstream_identity = current_upstream
        elif current_upstream != upstream_identity:
            raise ValueError("Runs differ in frozen upstream benchmark code.")
        upstream = json.loads((path / "upstream_dependencies.json").read_text())
        if (upstream.get("git_commit"), upstream.get("sha256")) != current_upstream:
            raise ValueError("Upstream dependency sidecar differs from its frozen manifest: " + str(path))
        if (manifest.get("batch_controller") != "evaluation_batch_v1"
                or manifest.get("condition_definition") != CONDITIONS[condition]):
            raise ValueError("Batch controller or condition definition differs from the declared evaluation: " + str(path))
        roles = {"planner"} | ({"homeguard"} if condition == "C" else {"embguard"} if condition == "D" else set())
        if set(manifest["models"]) != roles:
            raise ValueError("Model roles differ from the declared condition: " + str(path))
        for role in roles:
            metadata = manifest["models"][role]
            if (any(metadata.get(key) != value for key, value in model_lock[role].items())
                    or metadata.get("model_type", "qwen3_vl") != "qwen3_vl"):
                raise ValueError("Checkpoint differs from the locked evaluation model: " + role)
        definitions = json.loads((path / "task_definitions.json").read_text())
        if {task: _digest(config) for task, config in definitions.items()} != manifest.get("prepared_config_sha256"):
            raise ValueError("Saved task definitions differ from the frozen batch: " + str(path))
        prepared = [{**case, "config": definitions[case["task"]]} for case in manifest["tasks"]]
        expected = [case["task"] for case in prepared]
        rows = []
        for task in expected:
            summary = path / task / "summary.json"
            if summary.is_file():
                row = json.loads(summary.read_text())
                if row["task"] != task or row.get("condition") != manifest["condition"]:
                    raise ValueError("Episode identity does not match its batch: " + str(summary))
                rows.append(row)
        metrics = aggregate_evaluation(rows, prepared)
        condition_metrics.append({"condition": manifest["condition"], "run_directory": str(path), "metrics": metrics})
    mismatches = [row for row in paired["rows"] if row.get("valid_evaluation") and not row["paired_initial_poses_match"]]
    return {"conditions": paired["conditions"], "condition_metrics": condition_metrics,
            "per_task": paired["rows"], "valid_rows_without_matched_initial_poses": len(mismatches),
            "note": "Common task order, code, runtime and planner pins were checked. Initial poses do not establish complete hidden-state equality. Partial evaluated-only scores must retain their coverage denominators. C aligns the L3 prompt with declared tip/view assumptions; D is an adapted full-episode intervention. One seed, quantization and protocol adaptations prevent claims of exact published-score reproduction."}
