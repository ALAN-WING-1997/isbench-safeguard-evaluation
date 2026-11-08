"""Compare explicitly selected runs; do not silently pool partial or different trials."""
import argparse
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def same_initial_objects(left, right, tolerance=1e-4):
    a, b = left.get("objects", {}), right.get("objects", {})
    if not a or a.keys() != b.keys():
        return False
    for name in a:
        if any(abs(x-y) > tolerance for x, y in zip(a[name]["position"], b[name]["position"])):
            return False
        qa, qb = a[name]["orientation"], b[name]["orientation"]
        distance = min(math.sqrt(sum((x-y)**2 for x, y in zip(qa, qb))),
                       math.sqrt(sum((x+y)**2 for x, y in zip(qa, qb))))
        if distance > tolerance:
            return False
    return True


def compare(paths):
    paths = [Path(path) for path in paths]
    manifests = [load(p / "batch_manifest.json") for p in paths]
    if len({m["condition"] for m in manifests}) != len(manifests):
        raise ValueError("Select one run per condition; repeated runs require a separate analysis.")
    baseline = manifests[0]
    signature = lambda m: (m["runtime"], m["models"]["planner"]["repo_id"], m["models"]["planner"]["revision"],
                           [(x["task"], x["scene_sha256"], x["task_config_sha256"], x["max_steps"]) for x in m["tasks"]],
                           m["script_sha256"])
    if any(signature(m) != signature(baseline) for m in manifests[1:]):
        raise ValueError("Runs differ in protocol, planner, case content, budgets, or code. Do not rank them as a paired comparison.")
    rows = []
    for case in baseline["tasks"]:
        states = []
        for path, manifest in zip(paths, manifests):
            episode = path / case["task"]
            row = load(episode / "summary.json") if (episode / "summary.json").is_file() else {"valid_evaluation": False}
            state = load(episode / "initial_state.json") if (episode / "initial_state.json").is_file() else {}
            states.append(state)
            rows.append({"condition": manifest["condition"], "task": case["task"], **row})
        paired = all(same_initial_objects(states[0], state) for state in states[1:]) if len(states) > 1 else None
        for row in rows[-len(paths):]:
            row["paired_initial_poses_match"] = paired
    return {"rows": rows, "conditions": [m["condition"] for m in manifests],
            "note": "Matching initial poses does not prove complete simulator-state equivalence. Review initial images and task-specific state when interpreting differences. Two cases are integration evidence, not a model ranking."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.runs)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
