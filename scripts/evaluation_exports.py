"""Export a compact large-run report; full evidence remains separately available."""
import csv
import io
import json
from pathlib import Path
import zipfile

from export_isbench_results import COMPACT_NAMES, export_results

LARGE_COMPACT_NAMES = COMPACT_NAMES | {"task_definitions.json", "recovery.json", "process_running.json",
                                        "resume_events.json", "controller_state.json", "upstream_dependencies.json"}


def export_evaluation(run_dir, full_evidence=False):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "batch_summary.json").read_text())
    output = run_dir.parent / (run_dir.name + "_downloads")
    output.mkdir(exist_ok=True)
    metrics_file = output / "metrics.json"
    metrics_file.write_text(json.dumps(summary["metrics"], indent=2) + "\n")
    columns = ("task", "scene", "valid_evaluation", "task_completed", "safe_task_completed",
               "termination_reason", "triggered_safety_checks", "passed_safety_checks",
               "untriggered_safety_checks", "elapsed_seconds", "model_calls", "guard_model_calls",
               "worker_requests", "guard_worker_requests", "model_generations", "guard_generations",
               "completed_model_generations", "completed_guard_generations",
               "token_budget_hits", "process_returncode", "interrupted", "timed_out")
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary["results"])
    csv_file = output / "per_task.csv"
    csv_file.write_text(table.getvalue())
    compact = output / (run_dir.name + "_evaluation_summary.zip")
    temporary = compact.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and (path.name in LARGE_COMPACT_NAMES or
                                  (path.suffix == ".py" and "code" in path.relative_to(run_dir).parts)):
                archive.write(path, Path(run_dir.name) / path.relative_to(run_dir))
        archive.write(metrics_file, Path(run_dir.name) / "analysis/metrics.json")
        archive.write(csv_file, Path(run_dir.name) / "analysis/per_task.csv")
    temporary.replace(compact)
    result = {"summary_zip": str(compact), "per_task_csv": str(csv_file),
              "metrics_json": str(metrics_file), "raw_directory": str(run_dir)}
    if full_evidence:
        result["full_evidence"] = export_results(run_dir)
    return result
