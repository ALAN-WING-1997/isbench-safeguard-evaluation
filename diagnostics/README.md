> **9 October update:** the diagnostic has already established that the saved sink input completes at 8192 tokens. The final-answer parser is now fixed in the clean [rerun notebooks](../notebooks/archive/rerun_8192/README.md), and both saved 8192-token responses pass offline reinspection. The original diagnostic notebook, outputs, summary, and input ZIP remain preserved with their earlier parser. No further diagnostic generation is required before running the updated A trial.

# Planner generation diagnostic

Open `notebooks/archive/pilot/03_planner_generation_diagnostic.ipynb` in the GPU Colab runtime prepared by notebook 02. Upload `diagnostics/inputs/qwen_planner_diagnostic_inputs.zip` when the notebook's input cell opens the upload picker. The ZIP contains the original A/B prompts, initial images, first planner responses, and primitive/object allowlists. It excludes evaluation annotations and reference plans.

The notebook starts with A's sink case and sampling with an 8192-token cap. Its next cell tests greedy/8192 and sampling/2048. Archived greedy/2048 responses serve as baseline observations. Other tasks and a second sampling seed are optional. Every variant receives one attempt, with no automatic retry or simulator process. The generation settings and output cap are the only model-input changes.

The diagnostic scripts and results use `/content/isbench-safeguard-diagnostics`, separate from the A–D suite. Trial notebooks, runtime settings, and saved raw results are unchanged. Inference uses the same checkpoint, NF4/BF16/SDPA settings, and image budgets. A completed action format is not a correctness, task-success, or safety measurement. Memory use without a resident simulator is not evidence that the longer budget fits during a full episode.

`tools/build_diagnostic_notebook.py` regenerates the diagnostic worker from the unchanged trial worker, packages the selected saved inputs with deterministic ZIP metadata, and embeds the diagnostic code into the notebook. Saved input hashes are verified while building and running. The generation worker records its exact settings, raw output, token count, duration, GPU memory, package versions, and numerical checks. The runner retains failures and supports exporting partial evidence.

Local verification: seven diagnostic CPU tests cover input fidelity for all four cases, generation-setting plumbing, invalid settings, baseline truncation, final-action parsing, archive integrity, and one-attempt control flow. The original 13 tests also remain applicable. Actual diagnostic GPU generation has not been run locally.

Sources: [Transformers generation parameters](https://huggingface.co/docs/transformers/main_classes/text_generation), [Qwen3-VL Thinking generation settings](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking#generation-hyperparameters).
