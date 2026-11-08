"""One isolated, locally cached Qwen3-VL request; model exits before execution."""
import argparse
import json
import math
from pathlib import Path
import time
import traceback


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def chat_messages(job, images):
    """Use the multimodal processor's content-block format for every role."""
    messages = []
    if job.get("system"):
        messages.append({"role": "system", "content": [
            {"type": "text", "text": job["system"]},
        ]})
    messages.append({"role": "user", "content": [
        *[{"type": "image", "image": img} for img in images],
        {"type": "text", "text": job["prompt"]},
    ]})
    return messages


def generation_kwargs(settings):
    """Validate and pass explicit decoding settings, with legacy greedy defaults."""
    do_sample = settings.get("do_sample", False)
    if type(do_sample) is not bool:
        raise ValueError("do_sample must be a Boolean.")
    tokens = settings.get("max_new_tokens")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")
    penalty = settings.get("repetition_penalty", 1.0)
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("repetition_penalty must be positive and finite.")
    sampling = {}
    for key in ("temperature", "top_p", "top_k"):
        if key in settings:
            sampling[key] = settings[key]
        elif do_sample:
            raise ValueError(f"Sampling requires an explicit {key}.")
    if "temperature" in sampling:
        value = sampling["temperature"]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("temperature must be positive and finite.")
    if "top_p" in sampling:
        value = sampling["top_p"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("top_p must be in (0, 1].")
    if "top_k" in sampling:
        value = sampling["top_k"]
        if type(value) is not int or value < 0:
            raise ValueError("top_k must be a nonnegative integer.")
    kwargs = {"max_new_tokens": tokens, "do_sample": do_sample,
              "repetition_penalty": penalty, "use_cache": True}
    if do_sample:
        kwargs.update(sampling)
    return kwargs


def main(request_path, response_path, model_path):
    request = json.loads(request_path.read_text())
    settings = request["settings"]
    decoding = generation_kwargs(settings)
    import bitsandbytes as bnb
    from importlib.metadata import version
    from PIL import Image
    import torch
    from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The trial requires an NVIDIA GPU with BF16 support.")
    free, total = torch.cuda.mem_get_info()
    if free < settings["minimum_free_gib"] * 1024**3:
        raise RuntimeError(f"Only {free/1024**3:.2f} GiB GPU memory free beside the simulator.")
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3_vl":
        raise RuntimeError("This worker supports the pinned Qwen3-VL checkpoints only.")
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        min_pixels=settings["min_pixels"], max_pixels=settings["max_pixels"],
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=quantization, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False,
    ).eval()
    quantized = [name for name, mod in model.named_modules() if isinstance(mod, bnb.nn.Linear4bit)]
    if not model.is_loaded_in_4bit or not quantized or any("visual" in name.split(".") for name in quantized):
        raise RuntimeError("Expected NF4 text layers and an unquantized vision encoder.")
    placement = {key: str(value) for key, value in model.hf_device_map.items()}
    if any(value not in {"0", "cuda:0"} for value in placement.values()):
        raise RuntimeError("CPU/disk model offloading is not part of this trial.")
    numerical_checks = {"lm_head_forward_calls": 0}

    def check_logits(module, args, output):
        numerical_checks["lm_head_forward_calls"] += 1
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Non-finite model logits; stop before interpreting this output.")

    hook = model.lm_head.register_forward_hook(check_logits)
    result = {
        "outputs": [], "model_path": str(model_path), "settings": settings,
        "effective_generation_kwargs": decoding,
        "device_map": placement, "quantized_text_linear_count": len(quantized),
        "gpu_free_before_model_bytes": free, "gpu_total_bytes": total,
        "model_load_seconds": time.monotonic() - started,
        "model_memory_footprint_bytes": model.get_memory_footprint(),
        "package_versions": {name: version(name) for name in
                             ("torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "huggingface-hub")},
    }
    save(response_path, result)
    homeguard_decoder = None
    for job in request["jobs"]:
        images = [Image.open(path).convert("RGB") for path in job["images"]]
        messages = chat_messages(job, images)
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to("cuda")
        input_tokens = inputs.input_ids.shape[1]
        if input_tokens > settings["max_input_tokens"]:
            raise RuntimeError(f"Input exceeds the fixed trial budget: {input_tokens} tokens.")
        job_decoding = decoding
        constraint = None
        if job.get("response_format") is not None:
            from homeguard_json import HOMEGUARD_JSON_FORMAT, HomeGuardJSONDecoder

            if job["response_format"] != HOMEGUARD_JSON_FORMAT or request.get("role") != "homeguard":
                raise ValueError("Unsupported response format for this inference job.")
            if homeguard_decoder is None:
                homeguard_decoder = HomeGuardJSONDecoder(processor.tokenizer)
                result["package_versions"]["lm-format-enforcer"] = version("lm-format-enforcer")
            constraint = homeguard_decoder.constraint(input_tokens)
            job_decoding = {**decoding, "logits_processor": [constraint]}
        job_started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, **job_decoding)
        torch.cuda.synchronize()
        output_ids = generated[:, input_tokens:]
        raw = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output_tokens = int(output_ids.shape[1])
        output = {
            "view": job.get("view"), "raw_output": raw, "input_tokens": int(input_tokens),
            "output_tokens": output_tokens, "hit_token_limit": output_tokens >= settings["max_new_tokens"],
            "generation_seconds": time.monotonic() - job_started,
            "image_sizes": [list(img.size) for img in images],
        }
        if constraint is not None:
            output.update(response_format=job["response_format"], format_error=constraint.format_error(raw))
        result["outputs"].append(output)
        result.update(
            elapsed_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            numerical_checks=numerical_checks.copy(),
        )
        save(response_path, result)
        del inputs, generated, output_ids
        for img in images:
            img.close()
    hook.remove()
    print("VLM_RESPONSE_SAVED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.request, args.response, args.model_path)
    except Exception:
        result = {}
        if args.response.is_file():
            result = json.loads(args.response.read_text())
        result["error"] = traceback.format_exc()
        save(args.response, result)
        raise
