# Source provenance

Checked on 8 October 2026. Public manifest/configuration inspection confirms all three checkpoint repositories expose merged Qwen3-VL weights and inference assets. GPU loading and simulator integration remain untested locally.

| Component | Repository/revision |
| --- | --- |
| IS-Bench | `AI45Lab/IS-Bench` at `6a406e162feca1c4c455d9c9f1f58832edbdc1fd` |
| Planner | `Qwen/Qwen3-VL-8B-Thinking` at `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b` |
| HomeGuard | `Ursulalala/HomeGuard-8B` at `0766d4cf3eadb577c23c084538aa65e36bc59dc6` |
| EMBGuard | `EMBGuard/EMBGuard-4B` at `87015b974d49e132dcc15e58c8fb770ea3b33182` |
| HomeGuard prompt source | `AI45Lab/HomeGuard` at `1e88112af9d82bb39cf561ca46bec8180e764bfc`, `evaluation/prompt.py` |
| EMBGuard prompt source | `dongwxxkchoi/EMBGuard` at `45856cccf919fba7d9abb271e8665520a1577378`, `src/guardrail/prompts/guardrail_prompt.py` |

`scripts/vendor/homeguard_prompt.py` and `scripts/vendor/embguard_prompt.py` are unchanged local snapshots of the official prompt files. EMBGuard's code repository carries the MIT license. Retain upstream provenance when sharing adapted code; model/data licensing belongs to its respective source repositories.

Simulator setup, scene-data preparation, case preparation, evaluator summarization, and export helpers originate in this workspace's validated `isbench-open-model-evaluation` suite. The new suite copies them and adapts export coverage. It does not modify those original files.

- [HomeGuard paper](https://arxiv.org/abs/2603.14367)
- [HomeGuard official code](https://github.com/AI45Lab/HomeGuard)
- [HomeGuard model](https://huggingface.co/Ursulalala/HomeGuard-8B)
- [EMBGuard paper](https://arxiv.org/abs/2605.30924)
- [EMBGuard official code](https://github.com/dongwxxkchoi/EMBGuard)
- [EMBGuard model](https://huggingface.co/EMBGuard/EMBGuard-4B)
- [Qwen3-VL official model](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
- [IS-Bench](https://github.com/AI45Lab/IS-Bench)

The inference environment pins PyTorch 2.5.1/CUDA 12.1, torchvision 0.20.1, Transformers 4.57.6, Accelerate 1.11.0, bitsandbytes 0.48.2, huggingface-hub 0.36.0, Pillow 11.1.0, and NumPy 1.26.4. Public package metadata was checked for the new transformer/inference package versions. Worker responses record actual installed versions, precision, token budget, quantized layers, GPU placement, and numerical checks.
