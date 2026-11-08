# Sampled planner trials

This archive records the sampled two-case pilot. It uses the pinned **Qwen3-VL-8B-Thinking** planner in every condition and changes decoding and format retries.

| Setting | Planner A–D | HomeGuard / EMBGuard |
| --- | --- | --- |
| Decoding | Sampling | Greedy |
| Temperature | 1.0 | Not used |
| Top-p / top-k | 0.95 / 20 | Not used |
| Output limit | 8192 tokens | 2048 tokens |
| Repetition penalty | 1.0 | 1.0 |
| Seed | 0 | 0 |
| Planner format attempts per proposal | 1 | Not applicable |

D retains up to three different safety-driven proposals when EMBGuard rejects an action. That is distinct from repeating the same planner request because formatting failed. Each proposal now has one planner call.

The planner's sampling controls follow the specific dense 8B Thinking model card's visual-input settings. The smaller output cap, NF4 loading and one-attempt policy are documented adaptations, not the complete official inference recipe or paper score reproduction. Sampling may avoid a greedy loop; it does not guarantee final JSON or correct/safe actions. One fixed seed is an integration check, not a statistical success-rate estimate.

## Recommended next run

1. Finish the active Colab trial, or interrupt it and export the partial results. Do not run another planner or simulation concurrently on the same GPU. Opening these local notebooks cannot change an already running process's settings.
2. If the Colab environments, scene assets and cache are intact, skip setup notebooks 00–02. Their new copies are available for a fresh runtime.
3. Run `03_planner_request_replay.ipynb` first. Its default path is the recent C sink planner request. It reuses the exact saved prompt and images, including HomeGuard's notes already in that prompt, with one sampled planner call. No additional input ZIP is required on the same runtime. If the old files are absent, adjust the path after restoring the original files; missing or altered input is rejected.
4. Inspect and download the replay summary/raw response. `final_json_available` means completed reasoning and the requested JSON fields are present; it does not validate operator names, object arguments, task correctness or safety. That validation remains in the full trial.
5. Once the replay completes with usable output, run A, B, C and D notebooks sequentially. Each has preparation, fresh batch, sink, fan and export cells. All include partial export and second-case skipping after infrastructure failure. D also retains the old-batch recovery cell.
6. Use `20_compare_trials.ipynb` with one selected run per condition from this new version. It rejects older greedy/code/settings snapshots. Paired conclusions require rerunning all four conditions under the same settings.

The new runtime helper directory is `/content/isbench-safeguard-evaluation-sampling-8192`. Model caches, environments and scene files are reused. Every batch has a unique result directory and code/settings snapshot. Worker responses save `effective_generation_kwargs`, so the chosen controls can be checked from evidence.

The HomeGuard coordinate-format adapter and per-view audit remain included. EMBGuard's system-message content blocks and CPU processor preflight remain included. Reference actions, oracle safety labels and training datasets remain excluded from model requests.

## Files

- `00_simulator_setup.ipynb`, `01_scene_data.ipynb`, `02_inference_setup.ipynb`: fresh-runtime setup copies.
- `03_planner_request_replay.ipynb`: one saved request without simulator/guard calls.
- `10_A_baseline_trial.ipynb`, `11_B_generic_reminder_trial.ipynb`, `12_C_homeguard_trial.ipynb`, `13_D_embguard_trial.ipynb`: the four trial conditions.
- `20_compare_trials.ipynb`: compare selected matched runs.

[Qwen3-VL-8B-Thinking generation settings](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking#generation-hyperparameters). A future Instruct comparison needs its own checkpoint pin and a parser that does not require a Thinking closing tag; changing a notebook's model name alone is insufficient.
