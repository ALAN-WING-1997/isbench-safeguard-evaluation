"""Notebook controller: unique runs, bounded subprocesses, immediate summaries."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from pilot_qwen_summary import aggregate, load_json, summarize
from protocol import CONDITIONS


def save(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def stop_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


class TrialBatch:
    def __init__(self, suite, repo, sim_python, inference_python, condition, prepared, models, runtime, sim_env):
        self.repo, self.sim_python, self.inference_python = map(Path, (repo, sim_python, inference_python))
        self.condition, self.prepared, self.sim_env = condition, prepared, sim_env
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
        self.output = Path(suite) / "results" / condition / stamp
        self.output.mkdir(parents=True, exist_ok=False)
        self.code = self.output / "code"
        shutil.copytree(Path(suite) / "scripts", self.code, ignore=shutil.ignore_patterns("__pycache__"))
        self.runtime = runtime
        self.results, self.can_continue = [], True
        self.manifest = {
            "condition": condition, "condition_definition": CONDITIONS[condition], "models": models,
            "runtime": runtime, "tasks": [{k: v for k, v in case.items() if k != "config"} for case in prepared],
            "script_sha256": {str(p.relative_to(self.code)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(self.code.rglob("*.py"))},
            "reference_validation_performed": False,
            "max_steps_source": "reference length + 10; reference actions are never sent to models",
            "automatic_episode_retries": 0,
        }
        save(self.output / "batch_manifest.json", self.manifest)
        save(self.output / "runtime.json", runtime)
        save(self.output / "model_metadata.json", models)
        save(self.output / "cases.json", self.manifest["tasks"])
        self.save_summary()

    def save_summary(self):
        save(self.output / "batch_summary.json", {
            "condition": self.condition, "results": self.results,
            "metrics": aggregate(self.results, len(self.prepared)),
        })

    def run_case(self, index):
        if not self.can_continue:
            raise RuntimeError("Batch stopped after an infrastructure failure; export the partial evidence.")
        entry = self.prepared[index]
        episode = self.output / entry["task"]
        if episode.exists():
            raise RuntimeError("Case already attempted. Create a fresh batch for a rerun.")
        episode.mkdir()
        save(episode / "case.json", {k: v for k, v in entry.items() if k != "config"})
        command = [str(self.sim_python), str(self.code / "run_trial_episode.py"),
                   "--repo", str(self.repo), "--output", str(episode), "--condition", self.condition,
                   "--runtime", str(self.output / "runtime.json"), "--models", str(self.output / "model_metadata.json"),
                   "--case", str(episode / "case.json"), "--inference-python", str(self.inference_python)]
        start = time.monotonic()
        timed_out = interrupted = False
        print("Starting", self.condition, entry["task"], "| full log:", episode / "episode.log", flush=True)
        with (episode / "episode.log").open("w") as log:
            process = subprocess.Popen(command, cwd=self.repo, env=self.sim_env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            next_update = start
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if now - start > self.runtime["episode_timeout_seconds"]:
                        timed_out = True
                        stop_group(process)
                        break
                    if now >= next_update:
                        progress = load_json(episode / "progress.json")
                        print(progress.get("phase", "starting"), "| requests:", len(progress.get("requests", [])),
                              "| elapsed:", round(now - start), "s", flush=True)
                        next_update = now + 30
                    time.sleep(3)
            except KeyboardInterrupt:
                interrupted = True
            finally:
                stop_group(process)
            returncode = process.wait()
        row = summarize(episode, entry["config"], returncode, timed_out, interrupted, scope=self.runtime["scope"])
        progress = load_json(episode / "progress.json")
        decisions = progress.get("decisions", [])
        responses = [load_json(p) for p in sorted(episode.glob("*_response.json"))]
        row.update(condition=self.condition, scene=entry["scene"], elapsed_seconds=time.monotonic() - start,
                   model_calls=len(progress.get("requests", [])),
                   guard_model_calls=sum(r["role"] != "planner" for r in progress.get("requests", [])),
                   guard_replans=sum(d["decision"] == "replan" for d in decisions),
                   rejected_proposals=sum(d["decision"] in {"replan", "guard_blocked"} for d in decisions),
                   token_budget_hits=sum(o.get("hit_token_limit", False) for r in responses for o in r.get("outputs", [])),
                   failed_safety_checks=row["triggered_safety_checks"] - row["passed_safety_checks"],
                   initial_state_captured=(episode / "initial_state.json").is_file())
        save(episode / "process_exit.json", {"returncode": returncode, "timed_out": timed_out, "interrupted": interrupted})
        save(episode / "summary.json", row)
        self.results.append(row)
        self.save_summary()
        if not row["valid_evaluation"]:
            self.can_continue = False
            print("Infrastructure/inference failure. Export this batch before continuing.", flush=True)
            print(row.get("inference_error") or row.get("exception") or (episode / "episode.log").read_text(errors="replace")[-4000:])
        else:
            print("Saved result: completed =", row["task_completed"], "safe completed =", row["safe_task_completed"],
                  "termination =", row["termination_reason"], flush=True)
        return row
