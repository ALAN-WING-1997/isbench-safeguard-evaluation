"""Run a fresh A/B/C/D episode with isolated inference and upstream evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from unittest.mock import patch

from protocol import (CONDITIONS, embguard_jobs, gate_decision, guard_assessments,
                      homeguard_format_audit, homeguard_jobs, inference_settings, parse_plan, planner_prompt)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DisabledJudge:
    def __getattr__(self, name):
        raise RuntimeError("API awareness judging is disabled in this open-model trial.")


def run(args):
    repo, output = args.repo.resolve(), args.output.resolve()
    runtime = json.loads(args.runtime.read_text())
    models = json.loads(args.models.read_text())
    entry = json.loads(args.case.read_text())
    condition = CONDITIONS[args.condition]
    actual_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != runtime["isbench_commit"]:
        raise RuntimeError("IS-Bench revision differs from the locked trial protocol.")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "progress.json").exists():
        raise RuntimeError("This episode already exists; start a new batch.")
    task_file = repo / "data/tasks" / (entry["task"] + ".json")
    scene_file = Path(entry["scene_file"])
    if sha(task_file) != entry["task_config_sha256"] or sha(scene_file) != entry["scene_sha256"]:
        raise RuntimeError("Authored task or scene content changed after preparation.")
    task = json.loads(task_file.read_text())
    if task["scene_info"]["default_scene_model"] != entry["scene"]:
        raise RuntimeError("Unexpected default scene.")
    save(output / "task_config.json", task)  # Evidence only; never a worker request.
    os.environ.pop("OMNIGIBSON_NO_OMNIVERSE", None)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    sys.path[:0] = [str(repo / "bddl"), str(repo)]
    inference_env = os.environ.copy()  # Capture before Isaac Sim alters library paths.
    inference_env["OMNIGIBSON_NO_OMNIVERSE"] = "1"
    progress = {"phase": "loading_scene", "task": entry["task"], "condition": args.condition,
                "requests": [], "decisions": [], "executed_actions": 0}
    save(output / "progress.json", progress)
    benchmark = None
    request_number = 0

    def call(role, jobs, step, proposal=None, format_attempt=None):
        nonlocal request_number
        request_number += 1
        prefix = f"{request_number:03d}_{role}"
        request_file = output / (prefix + "_request.json")
        response_file = output / (prefix + "_response.json")
        progress["phase"] = "inference_" + role
        settings = inference_settings(runtime, role)
        request = {"role": role, "step": step, "proposal": proposal, "format_attempt": format_attempt,
                   "jobs": jobs, "settings": settings,
                   "model": {k: v for k, v in models[role].items() if k != "model_path"},
                   "prompt_sha256": [hashlib.sha256(job["prompt"].encode()).hexdigest() for job in jobs],
                   "image_sha256": {p: sha(p) for job in jobs for p in job["images"]}}
        save(request_file, request)
        progress["requests"].append({"request": request_number, "role": role, "step": step,
                                     "proposal": proposal, "format_attempt": format_attempt})
        save(output / "progress.json", progress)
        try:
            subprocess.run([
                str(args.inference_python), str(Path(__file__).parent / "vlm_worker.py"),
                "--request", str(request_file), "--response", str(response_file),
                "--model-path", models[role]["model_path"],
            ], env=inference_env, check=True, timeout=runtime["request_timeout_seconds"])
        except Exception:
            if response_file.is_file():
                failed = json.loads(response_file.read_text())
                progress["inference_error"] = failed.get("error")
            raise
        response = json.loads(response_file.read_text())
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response

    try:
        import numpy as np
        import torch
        random.seed(runtime["seed"])
        np.random.seed(runtime["seed"])
        torch.manual_seed(runtime["seed"])
        import omnigibson as og
        from omnigibson.macros import gm
        gm.HEADLESS = True
        gm.USE_GPU_DYNAMICS = True
        from og_ego_prim.utils.monkey_patch import add_monkey_patch
        add_monkey_patch()
        from og_ego_prim.benchmark import build_benchmark
        from og_ego_prim.primitives.ego_primitives import VALID_PRIMITIVES
        with patch.dict(os.environ, {"OPENAI_API_KEY": "disabled", "OPENAI_API_BASE": "http://127.0.0.1:9"}), \
                patch("openai.OpenAI", return_value=DisabledJudge()):
            benchmark = build_benchmark(
                task=entry["task"], scene=entry["scene"], ego_view=True, online_object_sampling=False,
                debug=False, eval_process_safety=True, eval_termination_safety=True,
                eval_execution=True, eval_awareness=False,
            )
        if Path(benchmark.env_config["scene"]["scene_file"]).resolve() != scene_file.resolve():
            raise RuntimeError("Simulator did not load the prepared authored scene.")
        benchmark.tracker.model = "Qwen3-VL-8B-Thinking_NF4_" + args.condition

        def views(label):
            folder = output / label
            observations = benchmark.get_surrounding_viewer_obs(save_img=str(folder))
            paths = [str(folder / f"obs_{i}.png") for i in range(5)]
            if observations is None or len(observations) != 5 or not all(Path(p).is_file() for p in paths):
                raise RuntimeError("Expected five saved surrounding observations.")
            return paths

        images = views("00_initial")
        poses = {}
        for obj in benchmark.env.scene.objects:
            position, orientation = obj.get_position_orientation()
            poses[obj.name] = {"position": position.tolist(), "orientation": orientation.tolist()}
        save(output / "initial_state.json", {"objects": poses, "image_sha256": {p: sha(p) for p in images}})
        history = []
        terminated = False
        max_steps = entry["max_steps"]
        for step in range(1, max_steps + 1):
            notes = None
            if args.condition == "C":
                guard = call("homeguard", homeguard_jobs(images, task["planning_context"]["task_instruction"]), step)
                notes = guard_assessments(guard, "homeguard")
                audit = homeguard_format_audit(guard)
                save(output / f"step_{step:02d}_homeguard.json", {"assessments": notes, "format_audit": audit})
                for view in audit["views"]:
                    print(f"HomeGuard view {view['view']}: {view['status']}"
                          + (f" ({view['error']})" if view['error'] else ""), flush=True)
                if notes is None:
                    benchmark.tracker.track_termination(reason="guard_format_error", msg="Malformed/truncated HomeGuard output.")
                    terminated = True
                    break
            rejected = []
            plan = None
            max_proposals = runtime["max_guarded_proposals"] if args.condition == "D" else 1
            for proposal in range(1, max_proposals + 1):
                prompt = planner_prompt(repo, task, history, args.condition, notes, rejected)
                plan = None
                for format_attempt in range(runtime["max_format_attempts"]):
                    response = call("planner", [{"images": images, "prompt": prompt}], step, proposal, format_attempt)
                    if len(response.get("outputs", [])) != 1:
                        raise RuntimeError("Planner worker returned an unexpected output count.")
                    result = response["outputs"][0]
                    benchmark.tracker.track_raw_output(step=step, content=result["raw_output"])
                    # The locked Thinking planner's chat template opens reasoning
                    # in its input. Its generated text must close that reasoning.
                    thinking_finished = any(tag in result["raw_output"] for tag in ("</think>", "</thinking>"))
                    if not result.get("hit_token_limit") and thinking_finished:
                        plan = parse_plan(result["raw_output"], VALID_PRIMITIVES, task["planning_context"]["object_list"])
                    benchmark.tracker.save_tracking(str(output / "report_partial.json"))
                    if plan is not None:
                        break
                if plan is None:
                    benchmark.tracker.track_termination(reason="plan_error", msg="Invalid/truncated planner outputs exhausted format attempts.")
                    terminated = True
                    break
                if args.condition != "D" or plan["action"] == "done()":
                    break  # DONE has no physical interaction; final safety is still evaluated.
                assessment = call("embguard", embguard_jobs(images, plan["action"]), step, proposal)
                parsed = guard_assessments(assessment, "embguard")
                decision = gate_decision(parsed, proposal, max_proposals)
                progress["decisions"].append({"step": step, "proposal": proposal, "action": plan["action"],
                                              "assessments": parsed, "decision": decision})
                save(output / "progress.json", progress)
                if decision == "execute":
                    break
                if decision != "replan":
                    benchmark.tracker.track_termination(reason=decision, msg="Guard assessment did not allow a physical action.")
                    terminated = True
                    break
                rejected.append({"action": plan["action"], "assessments": parsed})
                plan = None
            if terminated:
                break
            if plan is None:
                raise RuntimeError("Internal policy error: no accepted action.")
            benchmark.tracker.track_plan(step=step, plan=plan)
            history.append(f"{step}. {plan['action']}")
            progress["phase"] = "executing_action"
            save(output / "progress.json", progress)
            print("TRIAL_ACTION", step, plan["action"], flush=True)
            benchmark.execute_plan(plan)
            progress["executed_actions"] += 1
            benchmark.tracker.save_tracking(str(output / "report_partial.json"))
            if plan["action"] == "done()":
                benchmark.tracker.track_termination(reason="done")
                break
            images = views(f"{step:02d}_{plan['action'].split('(')[0]}")
        else:
            benchmark.tracker.track_termination(reason="exceeding_max_steps", msg=f"Reached {max_steps} executed proposals.")
        benchmark.termination_evaluation()
        benchmark.tracker.save_tracking(str(output / "report.json"))
        progress["phase"] = "evaluation_saved"
        save(output / "progress.json", progress)
        print("TRIAL_EVALUATION_SAVED", flush=True)
        try:
            views("99_final")
            progress["final_views_saved"] = True
        except Exception:
            progress["final_views_saved"] = False
            progress["final_render_exception"] = traceback.format_exc()
        save(output / "progress.json", progress)
    except Exception:
        progress["phase"] = "infrastructure_failed"
        progress["exception"] = traceback.format_exc()
        save(output / "progress.json", progress)
        if benchmark is not None:
            benchmark.tracker.save_tracking(str(output / "report_partial.json"))
        raise
    finally:
        if "og" in locals():
            (output / "cleanup_started.txt").write_text("og.shutdown()\n")
            og.shutdown()
            (output / "cleanup_completed.txt").write_text("clean shutdown\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("repo", "output", "runtime", "models", "case", "inference-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    run(parser.parse_args())
