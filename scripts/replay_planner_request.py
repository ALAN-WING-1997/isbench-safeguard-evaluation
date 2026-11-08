"""Try one saved planner input without starting guards or a simulator.

Preparation verifies the original prompt, five image files and pinned checkpoint.
The result measures final JSON availability only, never task or safety success.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

from protocol import json_answer


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def prepare_request(request_path, runtime, models):
    """Preserve exact jobs and image paths; replace only inference settings.

    Original image paths must still exist on the connected Colab runtime. This
    deliberately refuses relocation or replacement with a different observation.
    """
    request_path = Path(request_path).resolve()
    original = load(request_path)
    if original.get("role") != "planner":
        raise ValueError("Select a saved planner request, not a guard request.")
    jobs = original.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(jobs[0], dict):
        raise ValueError("Expected exactly one original planner job.")
    prompt = jobs[0].get("prompt")
    prompt_hashes = original.get("prompt_sha256")
    if not isinstance(prompt, str) or not isinstance(prompt_hashes, list) or len(prompt_hashes) != 1:
        raise ValueError("The saved prompt or its hash inventory is incomplete.")
    if not valid_digest(prompt_hashes[0]) or hashlib.sha256(prompt.encode()).hexdigest() != prompt_hashes[0]:
        raise ValueError("Saved prompt content differs from its recorded hash.")
    images = jobs[0].get("images")
    if (not isinstance(images, list) or len(images) != 5
            or any(not isinstance(path, str) or not path for path in images)
            or len(set(images)) != 5):
        raise ValueError("Expected the five distinct original image paths.")
    image_hashes = original.get("image_sha256")
    if not isinstance(image_hashes, dict) or set(image_hashes) != set(images):
        raise ValueError("The image hash inventory does not match the five original images.")
    for image in images:
        if not Path(image).is_file():
            raise ValueError("Original image is missing; restore that saved request's images: " + image)
        if not valid_digest(image_hashes[image]) or sha(image) != image_hashes[image]:
            raise ValueError("Saved image content differs from its recorded hash: " + image)
    planner, recorded_model = models.get("planner", {}), original.get("model", {})
    if not isinstance(recorded_model, dict) or any(
            not planner.get(key) or recorded_model.get(key) != planner[key]
            for key in ("repo_id", "revision")):
        raise ValueError("The replay planner checkpoint must match the original request exactly.")
    if not isinstance(runtime.get("inference"), dict):
        raise ValueError("The current runtime has no planner inference settings.")
    prepared = deepcopy(original)
    prepared["settings"] = deepcopy(runtime["inference"])
    prepared.pop("generation", None)  # Keep old decoding only in the provenance block.
    prepared["format_attempt"] = 0
    prepared["replay"] = {
        "source_request": str(request_path), "source_request_sha256": sha(request_path),
        "original_settings": deepcopy(original.get("settings")),
        "original_generation": deepcopy(original.get("generation")),
        "simulation_started": False, "guard_inference_performed": False,
        "action_executed": False, "correctness_or_safety_evaluated": False,
        "automatic_retries": 0,
    }
    return prepared


def inspect_response(response, request, returncode=0, timed_out=False, interrupted=False):
    """Check completion and JSON fields, without claiming an executable action."""
    outputs = response.get("outputs", [])
    outputs = outputs if isinstance(outputs, list) else []
    output = outputs[0] if len(outputs) == 1 and isinstance(outputs[0], dict) else {}
    raw = output.get("raw_output", "")
    raw = raw if isinstance(raw, str) else ""
    closed = any(tag in raw for tag in ("</think>", "</thinking>"))
    thinking_checkpoint = request["model"]["repo_id"].endswith("-Thinking")
    opened = "<think>" in raw or "<thinking>" in raw
    finished = closed or (not thinking_checkpoint and not opened)
    value = None
    if response.get("error") or returncode != 0 or timed_out or interrupted or len(outputs) != 1:
        status = "inference_failure"
    elif output.get("hit_token_limit"):
        status = "output_truncated"
    elif not finished:
        status = "unfinished_reasoning"
    else:
        value = json_answer(raw)
        if (value is not None and isinstance(value.get("action"), str) and value["action"].strip()
                and (value.get("caution") is None or isinstance(value["caution"], str))):
            status = "final_json_available"
        else:
            status = "complete_format_failure"
            value = None
    error = response.get("error")
    if status == "inference_failure" and not error:
        error = ("Worker timed out." if timed_out else "Replay was interrupted." if interrupted
                 else f"Worker failed or returned an invalid output inventory (returncode={returncode}); inspect worker.log.")
    return {
        "status": status, "final_json_available": status == "final_json_available", "final_json": value,
        "thinking_finished": finished, "hit_token_limit": output.get("hit_token_limit"),
        "input_tokens": output.get("input_tokens"), "output_tokens": output.get("output_tokens"),
        "generation_seconds": output.get("generation_seconds"), "returncode": returncode,
        "timed_out": timed_out, "interrupted": interrupted, "error": error,
        "effective_generation": response.get("effective_generation_kwargs", response.get("generation")),
        "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "simulation_started": False, "guard_inference_performed": False, "action_executed": False,
        "action_operator_object_validation_performed": False, "correctness_or_safety_evaluated": False,
        "note": "JSON availability only. Operator, objects, task correctness and safety are not evaluated.",
    }


def run_once(request_path, destination, runtime, models, inference_python):
    prepared = prepare_request(request_path, runtime, models)
    model_path = models["planner"].get("model_path")
    if not model_path or not Path(model_path).is_dir():
        raise ValueError("Prepare the pinned planner first; its cached model_path is missing.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    output = Path(destination) / run_id
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    shutil.copytree(scripts, output / "code", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(request_path, output / "original_request.json")
    evidence = output / "inputs"
    evidence.mkdir()
    for index, image in enumerate(prepared["jobs"][0]["images"]):
        shutil.copyfile(image, evidence / f"obs_{index}.png")
        if sha(evidence / f"obs_{index}.png") != prepared["image_sha256"][image]:
            raise ValueError("Original image changed while its replay evidence was being copied: " + image)
    save(output / "request.json", prepared)
    save(output / "manifest.json", {
        "kind": "single saved planner request replay", "source": prepared["replay"],
        "model": prepared["model"], "inference": prepared["settings"],
        "worker_sha256": sha(output / "code/vlm_worker.py"),
        "saved_image_sha256": {f"inputs/obs_{index}.png": prepared["image_sha256"][image]
                               for index, image in enumerate(prepared["jobs"][0]["images"])},
    })
    print("PLANNER_REPLAY_DIRECTORY:", output, flush=True)
    response_path = output / "response.json"
    command = [str(inference_python), str(output / "code/vlm_worker.py"),
               "--request", str(output / "request.json"), "--response", str(response_path),
               "--model-path", str(model_path)]
    started = time.monotonic()
    timed_out = interrupted = False
    returncode = -1
    for image in prepared["jobs"][0]["images"]:
        if not Path(image).is_file() or sha(image) != prepared["image_sha256"][image]:
            raise ValueError("Original image changed before replay inference: " + image)
    with (output / "worker.log").open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    timeout=runtime["request_timeout_seconds"], check=False)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except KeyboardInterrupt:
            interrupted = True
        except OSError as error:
            save(response_path, {"error": str(error), "outputs": []})
    response = load(response_path) if response_path.is_file() else {
        "error": "The worker did not save a response; inspect worker.log.", "outputs": []}
    summary = inspect_response(response, prepared, returncode, timed_out, interrupted)
    summary["elapsed_seconds"] = time.monotonic() - started
    save(output / "summary.json", summary)
    outputs = response.get("outputs", [])
    outputs = outputs if isinstance(outputs, list) else []
    raw = outputs[0].get("raw_output", "") if len(outputs) == 1 and isinstance(outputs[0], dict) else ""
    (output / "raw_output.txt").write_text(raw if isinstance(raw, str) else "")
    print(json.dumps(summary, indent=2), flush=True)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, required=True)
    args = parser.parse_args()
    run_once(args.request, args.output, load(args.runtime), load(args.models), args.inference_python)
    # Recorded model failures remain exportable notebook outcomes. Preparation
    # failures still raise before the notebook attempts an inference request.
