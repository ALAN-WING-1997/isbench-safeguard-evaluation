"""Resolve only locked public model revisions and verify local inference assets."""
import argparse
import json
from pathlib import Path


def check_multimodal_chat(processor):
    """Exercise system/image preprocessing on CPU before starting a simulator."""
    from PIL import Image
    from vlm_worker import chat_messages
    with Image.new("RGB", (64, 64), color="gray") as image:
        messages = chat_messages({"system": "Processor format check.", "prompt": "Describe this image."}, [image])
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
    input_tokens = int(inputs["input_ids"].shape[-1])
    if input_tokens <= 0:
        raise RuntimeError("Multimodal processor preflight produced no input tokens.")
    return {"system_text_blocks": True, "input_tokens": input_tokens, "model_generation_performed": False}


def prepare(lock_path, roles, metadata_path):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    lock = json.loads(lock_path.read_text())
    metadata = {}
    for role in roles:
        entry = lock[role]
        model_path = Path(snapshot_download(repo_id=entry["repo_id"], revision=entry["revision"]))
        if model_path.name != entry["revision"]:
            raise RuntimeError("Expected an immutable Hugging Face snapshot path.")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if config.model_type != "qwen3_vl":
            raise RuntimeError("Unexpected checkpoint architecture: " + role)
        index = json.loads((model_path / "model.safetensors.index.json").read_text())
        if any(not (model_path / name).is_file() for name in set(index["weight_map"].values())):
            raise RuntimeError("Incomplete checkpoint: " + role)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if not processor.chat_template:
            raise RuntimeError("The processor has no multimodal chat template.")
        metadata[role] = {**entry, "model_path": str(model_path), "model_type": config.model_type,
                          "weight_bytes": index.get("metadata", {}).get("total_size"),
                          "processor_class": type(processor).__name__}
        if role == "embguard":
            metadata[role]["chat_processor_preflight"] = check_multimodal_chat(processor)
            print("EMBGUARD_CHAT_PROCESSOR_READY (CPU preprocessing; no generation)", flush=True)
        print("CHECKPOINT_READY:", role, entry["revision"], flush=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", choices=("planner", "homeguard", "embguard"), required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.lock, args.roles, args.metadata)
