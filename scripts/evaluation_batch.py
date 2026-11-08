"""Resume-safe controller for a frozen evaluation, without changing model policy."""
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from subprocess import Popen as GitPopen
import tempfile
import time
import uuid

from evaluation_metrics import aggregate_evaluation, risk_check_counts
from pilot_qwen_summary import load_json, summarize
from evaluation_protocol import CONDITIONS
from trial_batch import save, stop_group


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _script_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).rglob("*.py"))}


def _runtime_dependency(path):
    path = Path(path)
    if path.suffix == ".py" or path.parts[0] == "og_ego_prim":
        return True
    if path.parts[:2] == ("data", "scenes") or path.parts[0] in {"assets", "docs"}:
        return False  # Authored scenes are pinned and verified separately per case.
    if path.parts[0] == "bddl" and path.parts[:2] != ("bddl", "bddl"):
        return False
    return path.suffix in {".json", ".yaml", ".yml", ".bddl", ".csv", ".toml", ".txt"}


def upstream_dependencies(repo):
    """Freeze clean tracked runtime sources/configuration and BDDL scoring data."""
    repo = Path(repo)
    def git(*arguments):
        command = ["git", "-C", str(repo), *arguments]
        with GitPopen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            output, error = process.communicate()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command, output, error)
            return output
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        tracked = [name for name in git("ls-files", "-z").decode().split("\0") if name]
        changed = [name for name in git("diff", "--name-only", "-z", "HEAD", "--").decode().split("\0") if name]
    except subprocess.CalledProcessError as error:
        raise ValueError("Cannot verify the upstream Git checkout.") from error
    dirty = [name for name in changed if _runtime_dependency(name)]
    if dirty:
        raise ValueError("Upstream tracked runtime dependencies were modified: " + ", ".join(dirty[:8]))
    selected = sorted(name for name in tracked if _runtime_dependency(name))
    if not selected:
        raise ValueError("The upstream checkout has no tracked runtime dependencies.")
    try:
        hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in selected}
    except OSError as error:
        raise ValueError("An upstream runtime dependency is missing or unreadable.") from error
    return {"git_commit": commit, "sha256": hashes,
            "policy": "Clean tracked Python/source/configuration and BDDL runtime data. Extracted scenes are checked separately by their per-case hashes; package versions are pinned by inference setup."}


def _process_identity(pid):
    """Match a recorded Linux process even if its PID is subsequently reused."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except FileNotFoundError:
        if Path("/proc").is_dir():
            return None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            return {"pid": pid}
        return {"pid": pid}
    except (OSError, IndexError, ValueError):
        # If process metadata cannot be read, be conservative about overlapping
        # processes rather than starting another simulator against the same GPU.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        return {"pid": pid}


def _boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


class EvaluationBatch:
    """One attempt per task; resume skips evidence instead of repeating actions.

    ``after_case(batch)`` is an optional checkpoint callback. ``run_pending``
    invokes it after each newly attempted case and again in ``finally``.
    An infrastructure failure pauses this controller; it is never retried.
    """

    def __init__(self, suite, repo, sim_python, inference_python, condition, prepared, models, runtime, sim_env,
                 results_root=None):
        self._configure(suite, repo, sim_python, inference_python, condition, prepared, models, runtime, sim_env)
        dependencies = upstream_dependencies(self.repo)
        if runtime.get("isbench_commit", dependencies["git_commit"]) != dependencies["git_commit"]:
            raise ValueError("Upstream HEAD differs from runtime.isbench_commit.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
        self.output = Path(results_root if results_root is not None else self.suite / "results") / condition / stamp
        self.output.mkdir(parents=True, exist_ok=False)
        self.code = self.output / "code"
        shutil.copytree(self.suite / "scripts", self.code, ignore=shutil.ignore_patterns("__pycache__"))
        self.manifest = {"condition": condition, "condition_definition": self.condition_definition, "models": models,
                         "runtime": runtime, "tasks": self._tasks(),
                         "prepared_config_sha256": {case["task"]: _digest(case["config"]) for case in prepared},
                         "script_sha256": _script_hashes(self.code),
                         "upstream_git_commit": dependencies["git_commit"],
                         "upstream_dependency_sha256": dependencies["sha256"],
                         "reference_validation_performed": False,
                         "max_steps_source": "reference length + 10; reference actions are never sent to models",
                         "automatic_episode_retries": 0, "batch_controller": "evaluation_batch_v1"}
        for name, value in (("batch_manifest.json", self.manifest), ("runtime.json", runtime),
                            ("model_metadata.json", models), ("cases.json", self._tasks()),
                            ("task_definitions.json", {case["task"]: case["config"] for case in prepared}),
                            ("upstream_dependencies.json", dependencies)):
            save(self.output / name, value)
        self.results, self.can_continue = [], True
        self.save_summary()

    def _configure(self, suite, repo, sim_python, inference_python, condition, prepared, models, runtime, sim_env):
        if condition not in CONDITIONS:
            raise ValueError("Unknown evaluation condition.")
        improved_c = runtime.get("condition_c_improvements", False)
        if type(improved_c) is not bool or (improved_c and condition != "C"):
            raise ValueError("condition_c_improvements must be Boolean and is available only for Condition C.")
        self.condition_definition = dict(CONDITIONS[condition])
        if improved_c:
            self.condition_definition.update(
                name="homeguard_l3_guidance_improved", prompt="v3_grounded_generated_tips",
                homeguard_final_format="homeguard_json_v1", failed_view_retries=1,
                executor_error_feedback=True, unresolved_view_policy="stop_before_planning",
            )
        self.suite, self.repo, self.sim_python, self.inference_python = map(
            Path, (suite, repo, sim_python, inference_python))
        self.condition = condition
        self.prepared, self.models, self.runtime = map(copy.deepcopy, (prepared, models, runtime))
        self.sim_env = dict(sim_env)
        self._allowed_invalid_tasks = set()
        self._confirm_previous_runtime_stopped = False
        names = [case["task"] for case in prepared]
        if not names or len(names) != len(set(names)) or any(Path(name).name != name or name in (".", "..") for name in names):
            raise ValueError("Expected a nonempty selection of unique task names.")
        runner = runtime.get("episode_runner")
        if not isinstance(runner, str) or Path(runner).name != runner or not runner.endswith(".py"):
            raise ValueError("runtime.episode_runner must name one explicit Python script.")
        if not (self.suite / "scripts" / runner).is_file():
            raise ValueError("The selected episode runner is missing.")

    def _tasks(self):
        return [{key: value for key, value in case.items() if key != "config"} for case in self.prepared]

    @classmethod
    def resume(cls, run_dir, suite, repo, sim_python, inference_python, prepared, models, runtime, sim_env,
               continue_after_infrastructure=False, confirm_previous_runtime_stopped=False):
        if type(continue_after_infrastructure) is not bool or type(confirm_previous_runtime_stopped) is not bool:
            raise ValueError("Resume confirmation flags must be Boolean.")
        output = Path(run_dir)
        manifest = load_json(output / "batch_manifest.json")
        if manifest.get("batch_controller") != "evaluation_batch_v1":
            raise ValueError("Expected a saved EvaluationBatch run.")
        batch = cls.__new__(cls)
        batch._configure(suite, repo, sim_python, inference_python, manifest["condition"], prepared, models, runtime, sim_env)
        batch.output, batch.code, batch.manifest = output, output / "code", manifest
        batch.results, batch.can_continue = [], True
        batch._confirm_previous_runtime_stopped = confirm_previous_runtime_stopped
        with batch._controller_lock():
            batch._verify_identity()
            batch._refresh_results()
            if continue_after_infrastructure:
                batch._allowed_invalid_tasks = {row["task"] for row in batch.results
                                                if row.get("valid_evaluation") is not True}
                batch.can_continue = True
            events = load_json(output / "resume_events.json").get("events", [])
            events.append({"resumed_at_utc": datetime.now(timezone.utc).isoformat(),
                           "continue_after_infrastructure": continue_after_infrastructure,
                           "confirm_previous_runtime_stopped": confirm_previous_runtime_stopped,
                           "preserved_invalid_tasks": sorted(batch._allowed_invalid_tasks)})
            save(output / "resume_events.json", {"events": events})
            batch.save_summary()
        return batch

    def _verify_identity(self):
        identity = {"runtime": self.runtime, "models": self.models, "tasks": self._tasks(),
                    "condition_definition": self.condition_definition,
                    "prepared_config_sha256": {case["task"]: _digest(case["config"]) for case in self.prepared}}
        if any(self.manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("Resume inputs differ in condition, task selection/content, runtime, or models.")
        if (_script_hashes(self.code) != self.manifest["script_sha256"]
                or _script_hashes(self.suite / "scripts") != self.manifest["script_sha256"]):
            raise ValueError("Resume code differs from the frozen run.")
        dependencies = upstream_dependencies(self.repo)
        if (dependencies["git_commit"] != self.manifest.get("upstream_git_commit")
                or dependencies["sha256"] != self.manifest.get("upstream_dependency_sha256")):
            raise ValueError("Upstream runtime dependencies differ from the frozen run.")
        for name, expected in (("runtime.json", self.runtime), ("model_metadata.json", self.models),
                               ("cases.json", self._tasks()),
                               ("task_definitions.json", {case["task"]: case["config"] for case in self.prepared}),
                               ("upstream_dependencies.json", dependencies)):
            if load_json(self.output / name) != expected:
                raise ValueError("Saved evaluation metadata differs: " + name)

    @contextmanager
    def _controller_lock(self):
        # Drive's FUSE mount may not implement flock. A host-local lock handles
        # concurrent controllers in this runtime; persistent owner metadata
        # handles unfinished evidence from another runtime conservatively.
        path = self._lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another controller is running this batch; do not start overlapping simulations.") from error
            registered = False
            try:
                owner = load_json(self.output / "controller_state.json")
                if owner.get("status") == "running":
                    self._assert_recorded_process_not_live(owner, "controller")
                save(self.output / "controller_state.json", {
                    "status": "running", "runtime_boot_id": _boot_id(),
                    "process_identity": _process_identity(os.getpid()) or {"pid": os.getpid()},
                    "run_dir": str(self.output.resolve()), "locked_at_utc": datetime.now(timezone.utc).isoformat()})
                registered = True
                yield
            finally:
                if registered:
                    save(self.output / "controller_state.json", {
                        "status": "idle", "runtime_boot_id": _boot_id(),
                        "released_at_utc": datetime.now(timezone.utc).isoformat()})
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _lock_path(self):
        name = hashlib.sha256(str(self.output.resolve()).encode()).hexdigest() + ".lock"
        return Path(tempfile.gettempdir()) / "isbench_evaluation_locks" / name

    def _assert_recorded_process_not_live(self, evidence, label):
        recorded = evidence.get("process_identity")
        boot = evidence.get("runtime_boot_id") or (recorded.get("boot_id") if isinstance(recorded, dict) else None)
        if boot and boot != _boot_id():
            if not self._confirm_previous_runtime_stopped:
                raise RuntimeError("Unfinished " + label + " evidence belongs to another runtime. "
                                   "End the previous runtime before setting confirm_previous_runtime_stopped=True.")
            return
        if not isinstance(recorded, dict) or type(recorded.get("pid")) is not int:
            return
        current = _process_identity(recorded["pid"])
        if current and (len(current) == 1 or all(current.get(key) == value for key, value in recorded.items())):
            raise RuntimeError("A recorded " + label + " process is still running; do not overlap simulators.")

    def _assert_episode_not_live(self, episode):
        if (episode / "process_exit.json").is_file():
            return
        self._assert_recorded_process_not_live(load_json(episode / "process_running.json"), "episode")
        # Also cover a controller disappearing just after Popen, before its PID
        # marker was saved. Do not print unrelated processes' command lines.
        if Path("/proc").is_dir():
            for process_dir in Path("/proc").iterdir():
                if not process_dir.name.isdigit():
                    continue
                try:
                    arguments = (process_dir / "cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
                    index = arguments.index("--output")
                    matches = (arguments[index + 1] == str(episode)
                               and str(self.code / self.runtime["episode_runner"]) in arguments)
                except (OSError, ValueError, IndexError):
                    continue
                if matches and _process_identity(int(process_dir.name)):
                    raise RuntimeError("An episode process is still running: " + str(episode))

    def _summarize(self, episode, case, process, elapsed_seconds):
        row = summarize(episode, case["config"], process.get("returncode"),
                        process.get("timed_out", False), process.get("interrupted", False), scope=self.runtime["scope"])
        progress = load_json(episode / "progress.json")
        decisions = progress.get("decisions", [])
        responses_by_name = {path.name: load_json(path) for path in sorted(episode.glob("*_response.json"))}
        responses = list(responses_by_name.values())
        requests = [load_json(path) for path in sorted(episode.glob("*_request.json"))]
        requested = sum(len(request.get("jobs", [])) for request in requests)
        guard_requested = sum(len(request.get("jobs", [])) for request in requests if request.get("role") != "planner")
        completed = sum(len(response.get("outputs", [])) for response in responses)
        guard_completed = sum(len(response.get("outputs", [])) for name, response in responses_by_name.items()
                              if not name.endswith("_planner_response.json"))
        row.update(condition=self.condition, scene=case["scene"], elapsed_seconds=elapsed_seconds,
                   model_calls=len(progress.get("requests", [])),
                   guard_model_calls=sum(request["role"] != "planner" for request in progress.get("requests", [])),
                   worker_requests=len(requests),
                   guard_worker_requests=sum(request.get("role") != "planner" for request in requests),
                   model_generations=requested, guard_generations=guard_requested,
                   completed_model_generations=completed, completed_guard_generations=guard_completed,
                   generation_counts_complete=len(requests) == len(progress.get("requests", [])),
                   generation_count_basis="model_generations/guard_generations count requested jobs, including jobs lost before completion; completed_* count saved outputs, irrespective of format validity.",
                   guard_replans=sum(decision["decision"] == "replan" for decision in decisions),
                   rejected_proposals=sum(decision["decision"] in {"replan", "guard_blocked"} for decision in decisions),
                   token_budget_hits=sum(output.get("hit_token_limit", False)
                                         for response in responses for output in response.get("outputs", [])),
                   planner_token_budget_hits=sum(output.get("hit_token_limit", False)
                                                 for name, response in responses_by_name.items()
                                                 if name.endswith("_planner_response.json")
                                                 for output in response.get("outputs", [])),
                   failed_safety_checks=row["triggered_safety_checks"] - row["passed_safety_checks"],
                   initial_state_captured=(episode / "initial_state.json").is_file(),
                   risk_check_counts=risk_check_counts(load_json(episode / "report.json")))
        if self.runtime.get("condition_c_improvements", False):
            retries = progress.get("guard_format_retries", [])
            retried = sum(len(event["retried_views"]) for event in retries)
            unresolved = sum(len(event["unresolved_views"]) for event in retries)
            row.update(guard_retried_views=retried, guard_recovered_views=retried - unresolved,
                       guard_unresolved_views=unresolved)
        if process.get("controller_error"):
            row.update(controller_error=process["controller_error"], valid_evaluation=False,
                       task_completed=None, safe_task_completed=None, safety_status="not_evaluated")
        return row

    def _refresh_results(self):
        results = []
        for case in self.prepared:
            episode = self.output / case["task"]
            if not episode.exists():
                continue
            self._assert_episode_not_live(episode)
            if (episode / "summary.json").is_file():
                row = load_json(episode / "summary.json")
                if row.get("task") != case["task"] or row.get("condition") != self.condition:
                    raise ValueError("Saved case summary has an unexpected identity.")
                if not (episode / "process_exit.json").is_file():
                    raise ValueError("Saved summary lacks process-exit evidence: " + case["task"])
                process = load_json(episode / "process_exit.json")
                recomputed = self._summarize(episode, case, process, process.get("elapsed_seconds"))
                if any(row.get(key) != value for key, value in recomputed.items()):
                    raise ValueError("Saved summary differs from its raw evaluation evidence: " + case["task"])
                results.append(row)
                continue
            process = load_json(episode / "process_exit.json")
            if not process:
                process = {"returncode": None, "timed_out": False, "interrupted": True,
                           "recovered_unfinished": True, "elapsed_seconds": None}
                save(episode / "process_exit.json", process)
                save(episode / "recovery.json", {
                    "status": "unfinished_after_disconnect", "automatic_retry": False,
                    "confirmed_previous_runtime_stopped": self._confirm_previous_runtime_stopped,
                    "note": "No live recorded episode or saved exit was found. Original evidence is preserved; no action was rerun.",
                    "last_phase": load_json(episode / "progress.json").get("phase")})
            row = self._summarize(episode, case, process, process.get("elapsed_seconds"))
            if process.get("recovered_unfinished"):
                row["recovery_status"] = "unfinished_after_disconnect"
            save(episode / "summary.json", row)
            results.append(row)
        self.results = results
        invalid = {row["task"] for row in results if row.get("valid_evaluation") is not True}
        self.can_continue = invalid <= self._allowed_invalid_tasks

    def save_summary(self):
        save(self.output / "batch_summary.json", {
            "condition": self.condition, "results": self.results,
            "metrics": aggregate_evaluation(self.results, self.prepared),
            "can_continue": self.can_continue,
            "pending_tasks": [case["task"] for case in self.prepared
                              if case["task"] not in {row["task"] for row in self.results}],
        })

    def _run_case(self, case):
        episode = self.output / case["task"]
        episode.mkdir(exist_ok=False)
        save(episode / "case.json", {key: value for key, value in case.items() if key != "config"})
        command = [str(self.sim_python), str(self.code / self.runtime["episode_runner"]),
                   "--repo", str(self.repo), "--output", str(episode), "--condition", self.condition,
                   "--runtime", str(self.output / "runtime.json"), "--models", str(self.output / "model_metadata.json"),
                   "--case", str(episode / "case.json"), "--inference-python", str(self.inference_python)]
        started, timed_out, interrupted, returncode = time.monotonic(), False, False, None
        launch_error = None
        print("Starting", self.condition, case["task"], "| log:", episode / "episode.log", flush=True)
        with (episode / "episode.log").open("w") as log:
            process = None
            try:
                process = subprocess.Popen(command, cwd=self.repo, env=self.sim_env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save(episode / "process_running.json", {"command": command,
                     "runtime_boot_id": _boot_id(),
                     "process_identity": _process_identity(process.pid) or {"pid": process.pid},
                     "started_at_utc": datetime.now(timezone.utc).isoformat()})
                next_update = started
                while process.poll() is None:
                    now = time.monotonic()
                    if now - started > self.runtime["episode_timeout_seconds"]:
                        timed_out = True
                        stop_group(process)
                        break
                    if now >= next_update:
                        progress = load_json(episode / "progress.json")
                        print(progress.get("phase", "starting"), "| requests:", len(progress.get("requests", [])),
                              "| elapsed:", round(now - started), "s", flush=True)
                        next_update = now + 30
                    time.sleep(3)
            except KeyboardInterrupt:
                interrupted = True
            except Exception as error:
                launch_error = f"{type(error).__name__}: {error}"
                log.write("\nController error: " + launch_error + "\n")
            finally:
                if process is not None:
                    stop_group(process)
                    returncode = process.wait()
        elapsed = time.monotonic() - started
        process_info = {"returncode": returncode, "timed_out": timed_out, "interrupted": interrupted,
                        "elapsed_seconds": elapsed, "controller_error": launch_error}
        save(episode / "process_exit.json", process_info)
        row = self._summarize(episode, case, process_info, elapsed)
        if launch_error:
            row.update(controller_error=launch_error, valid_evaluation=False,
                       task_completed=None, safe_task_completed=None, safety_status="not_evaluated")
        save(episode / "summary.json", row)
        self.results.append(row)
        self.can_continue = row["valid_evaluation"]
        self.save_summary()
        print("Saved:", case["task"], "| valid:", row["valid_evaluation"],
              "| termination:", row["termination_reason"], flush=True)
        return row

    def run_pending(self, max_cases=5, after_case=None):
        """Run at most ``max_cases`` new tasks, preserving all earlier attempts."""
        if type(max_cases) is not int or max_cases <= 0:
            raise ValueError("max_cases must be a positive integer.")
        if after_case is not None and not callable(after_case):
            raise TypeError("after_case must be callable.")
        added = []
        with self._controller_lock():
            verified = False
            try:
                self._verify_identity()
                verified = True
                self._refresh_results()
                if not self.can_continue:
                    print("Batch paused after an infrastructure failure. Export its evidence before continuing.", flush=True)
                    return added
                seen = {row["task"] for row in self.results}
                for case in self.prepared:
                    if case["task"] in seen:
                        continue
                    added.append(self._run_case(case))
                    if after_case is not None:
                        after_case(self)
                    if not self.can_continue or len(added) >= max_cases:
                        break
                return added
            finally:
                if verified:
                    self.save_summary()
                    if after_case is not None:
                        after_case(self)
