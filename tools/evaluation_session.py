"""Run every remaining task using the unchanged, frozen evaluation controller.

This host-side driver is deliberately outside ``scripts/``.  Its continuation
policy changes scheduling only; saved model, simulator and scoring code remains
subject to EvaluationBatch's existing identity and evidence checks.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


def _resume_session(batch):
    return type(batch).resume(
        batch.output, batch.suite, batch.repo, batch.sim_python,
        batch.inference_python, batch.prepared, batch.models, batch.runtime,
        batch.sim_env, continue_after_infrastructure=True,
        confirm_previous_runtime_stopped=batch._confirm_previous_runtime_stopped,
    )


def _atomic_text(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _record_policy(batch, driver_source):
    if not driver_source:
        driver_source = globals().get("SESSION_DRIVER_SOURCE", "")
    if not driver_source:
        source_path = globals().get("__file__")
        if source_path and Path(source_path).name == "evaluation_session.py":
            driver_source = Path(source_path).read_text(encoding="utf-8")
    if not isinstance(driver_source, str) or not driver_source.strip():
        raise ValueError("Supply the session driver source to record its checksum.")
    policy = {
        "session_driver": "evaluation_session_v1",
        "session_driver_sha256": hashlib.sha256(driver_source.encode("utf-8")).hexdigest(),
        "scheduling": "all remaining unattempted tasks in one notebook invocation",
        "episode_isolation": "one fresh native simulator subprocess per task",
        "automatic_episode_retries": 0,
        "continue_after_saved_invalid_episode": True,
        "stop_after_new_user_interruption": True,
        "existing_attempts": "preserved and skipped, including failed attempts",
        "frozen_episode_and_model_code_modified": False,
        "unresolved_infrastructure_failures": "remain invalid and retain original metrics semantics",
        "controller_errors": "identity, storage and live-process errors propagate",
    }
    source_file = batch.output / "session_driver.py"
    policy_file = batch.output / "session_policy.json"
    accounting_source = globals().get("ACCOUNTING_SOURCE")
    accounting_file = batch.output / "session_accounting.py"
    if isinstance(accounting_source, str) and accounting_source.strip():
        policy["session_accounting_sha256"] = hashlib.sha256(accounting_source.encode("utf-8")).hexdigest()
        if accounting_file.exists() and accounting_file.read_text(encoding="utf-8") != accounting_source:
            raise ValueError("Saved session accounting differs from this notebook's accounting source.")
    else:
        accounting_source = None
    if source_file.exists() and source_file.read_text(encoding="utf-8") != driver_source:
        raise ValueError("Saved session driver differs from this notebook's driver.")
    if policy_file.exists() and json.loads(policy_file.read_text()) != policy:
        raise ValueError("Saved session policy differs from this notebook's policy.")
    if not source_file.exists():
        _atomic_text(source_file, driver_source)
    if accounting_source is not None and not accounting_file.exists():
        _atomic_text(accounting_file, accounting_source)
    if not policy_file.exists():
        _atomic_text(policy_file, json.dumps(policy, indent=2) + "\n")


def _pending_tasks(batch):
    attempted = {row["task"] for row in batch.results}
    return [case["task"] for case in batch.prepared if case["task"] not in attempted]


def _show_failure(row):
    if row.get("valid_evaluation") is True and row.get("task_completed") is True:
        return
    print(
        "Attempt retained:", row["task"], "| valid:", row.get("valid_evaluation"),
        "| termination:", row.get("termination_reason"),
        "| timed out:", row.get("timed_out", False),
        "| interrupted:", row.get("interrupted", False),
        "| process exit:", row.get("process_returncode"), flush=True,
    )
    for name in ("exception", "controller_error", "inference_error"):
        if row.get(name):
            print(name + ":", str(row[name])[-2400:], flush=True)


def run_all_remaining(batch, after_case=None, driver_source=""):
    """Return ``(current_batch, new_rows)`` after one pass over unattempted tasks.

    Starting this function explicitly acknowledges previously saved invalid
    attempts.  A newly interrupted task stops this invocation; running the cell
    again resumes only tasks that have never been attempted.  Identity checks,
    live-process checks and checkpoint/export errors are never suppressed.
    """
    if after_case is not None and not callable(after_case):
        raise TypeError("after_case must be callable.")
    batch = _resume_session(batch)
    _record_policy(batch, driver_source)
    newly_attempted = []
    initial_pending = len(_pending_tasks(batch))
    print(
        "Running all remaining tasks | retained:", len(batch.results),
        "| pending:", initial_pending, "| automatic retries: 0", flush=True,
    )
    for _ in range(initial_pending):
        before = set(_pending_tasks(batch))
        if not before:
            break
        rows = batch.run_pending(max_cases=1, after_case=after_case)
        if len(rows) != 1 or rows[0].get("task") not in before:
            raise RuntimeError("The evaluation controller made no unique task progress; inspect its saved evidence.")
        row = rows[0]
        newly_attempted.append(row)
        _show_failure(row)
        if row.get("interrupted"):
            print(
                "Stopped by user. The interrupted attempt is preserved; rerun this cell when ready to continue.",
                flush=True,
            )
            break
        if row.get("valid_evaluation") is not True:
            batch = _resume_session(batch)
            print("Continuing with the next unattempted task; this failure will not be retried.", flush=True)
    print(
        "Session finished | coverage:", len(batch.results), "/", len(batch.prepared),
        "| pending:", len(_pending_tasks(batch)), flush=True,
    )
    return batch, newly_attempted


def export_session(batch, full_evidence=False, accounting_fn=None):
    """Export native reports, retaining the separate host scheduling provenance."""
    from evaluation_exports import export_evaluation
    result = export_evaluation(batch.output, full_evidence=full_evidence)
    compact = Path(result["summary_zip"])
    extras = [(batch.output / name, Path(batch.output.name) / name)
              for name in ("session_policy.json", "session_driver.py", "session_accounting.py")]
    extras = [(path, name) for path, name in extras if path.is_file()]
    if accounting_fn is not None:
        accounted = accounting_fn(batch.output)
        corrected_file = compact.parent / "corrected_evaluation.json"
        corrected_metrics = compact.parent / "corrected_metrics.json"
        corrected_table = compact.parent / "corrected_per_task.csv"
        _atomic_text(corrected_file, json.dumps(accounted, indent=2) + "\n")
        _atomic_text(corrected_metrics, json.dumps(accounted["metrics"], indent=2) + "\n")
        rows = accounted["results"]
        columns = sorted({key for row in rows for key in row})
        table = io.StringIO()
        writer = csv.DictWriter(table, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                         else value for key, value in row.items()} for row in rows)
        _atomic_text(corrected_table, table.getvalue())
        extras.extend((path, Path(batch.output.name) / "analysis" / path.name)
                      for path in (corrected_file, corrected_metrics, corrected_table))
        result.update(corrected_evaluation_json=str(corrected_file),
                      corrected_metrics_json=str(corrected_metrics),
                      corrected_per_task_csv=str(corrected_table))
    if extras:
        # The native exporter creates a fresh ZIP; each provenance file is added
        # once, so repeated checkpoints never accumulate duplicate ZIP members.
        with zipfile.ZipFile(compact, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, name in extras:
                archive.write(path, name)
    return result
