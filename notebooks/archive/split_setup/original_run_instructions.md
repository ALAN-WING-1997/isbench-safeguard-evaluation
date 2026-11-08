# Original instructions for the 20-task one-seed comparison

These notes record the earlier setup and execution workflow. See the [current Colab guide](../../README.md) for notebook names and current execution steps.

The original A–D notebooks evaluate **20 selected tasks**, covering **all 16 authored default scenes** and **all eight annotated risk categories**, with seed **0**. The [task table and protocol](../../../docs/large_seed0_protocol.md) record the selection, settings, paper adaptations and scoring rules. Selection favors short reference plans and varied actions, without using model results. Scores describe this curated pool rather than the full benchmark.

## Run improved C only

On a fresh L4 / High-RAM Colab runtime, open [00–03 full setup](../../00_setup_colab.ipynb) and choose **Run All**. It runs the simulator, scene data, inference environment, and task preflight stages in order. Wait for `ALL_SETUP_READY`. The setup notebook replaces opening 00, 01, 02 and 03 individually.

Open [14 improved C](../../14_C_improved.ipynb) on the same runtime. It checks all 20 original tasks and the locked HomeGuard tokenizer, then runs only the improved condition. The run uses the original seed, models, scenes, action budgets and output token caps. It tests constrained JSON with one failed-view retry, evidence-focused hazard assessment, and planner guidance with execution feedback together.

Keep Drive enabled to save the run under `large_seed0/tasks20_c_improved/C/<run-id>`. Save the printed `RESULT_DIRECTORY`. After reconnecting, paste that path into the improved notebook’s `RESUME_RUN`; already attempted tasks remain in the record. The comparison cell checks original and improved C against the same evaluation pins and reports coverage, termination reasons, token limits and safety outcomes. It reports the combined intervention at seed 0. Use notebook 20 for the original A–D comparison.

## Run original A–D

1. On a fresh L4 / High-RAM Colab runtime, run [00 simulator setup](00_simulator_setup.ipynb), [01 scene data](01_scene_data.ipynb) and [02 inference setup](02_inference_setup.ipynb). Skip completed setup when the previous runtime is intact.
2. Run [03 task preflight](03_task_preflight.ipynb). Proceed after `TASKS_READY: 20`; fix missing assets rather than removing tasks.
3. Run [A baseline](../../10_A_baseline.ipynb), [B generic reminder](../../11_B_generic_reminder.ipynb), [C HomeGuard L3](../../12_C_homeguard_l3.ipynb) and [D EMBGuard](../../13_D_embguard.ipynb). Each is standalone after shared setup; no diagnostic ZIP upload is needed.
4. In each condition, keep Google Drive storage enabled, prepare models, and create a run. Save the printed `RESULT_DIRECTORY`. The chunk cell attempts the next **five new tasks**; repeat it four times for the 20 tasks, unless an infrastructure failure pauses the run. This is 80 planned episodes across A–D. Download the compact report whenever convenient.
5. Use [20 comparison](../../20_compare_ABCD.ipynb) with exactly one saved directory per condition. It can inspect partial progress, but primary pool-wide SR/SSR remain null until all 20 selected tasks have valid evaluations.

Do not share one GPU between active batches. Separate GPU runtimes can run different conditions in parallel, using distinct condition/run directories. A–D must use the same embedded notebook revision and settings.

## Continue after reconnecting

Open the same condition notebook, rerun helper/preparation/storage/model cells as needed, paste the saved directory into `RESUME_RUN`, and rerun create/resume followed by the chunk cell. Completed episodes—including unsuccessful model outcomes—are retained and skipped. Repeating create/resume in an intact kernel reuses its known run. After a kernel reset, the notebook lists existing runs and requires an explicit path, preventing accidental duplicate batches.

If the old runtime is still working, reconnect to it and wait; do not start a second owner. If an unfinished owner belongs to a different runtime, end that runtime before setting `CONFIRM_PREVIOUS_RUNTIME_STOPPED=True`. If an infrastructure failure paused the run, inspect its episode log and correct the problem before setting `CONTINUE_AFTER_INFRASTRUCTURE=True`. This continues only unattempted tasks; it preserves the failed attempt as unresolved. A planner-format error or guard refusal is a scored model failure and does not need this flag.

In Antigravity, use the connected runtime file browser for the printed ZIP path if Colab’s download widget is unavailable. Drive mounting may require opening this notebook in the Colab browser for authentication. A fresh runtime needs the same setup and cached checkpoints downloaded again if the cache was lost. Saved Drive results survive runtime replacement.

## What changed from the pilot

A and B retain the sampled pilot’s behavior. C now puts HomeGuard predictions into the upstream **V3/L3 safety-tips prompt**, with the five-view conversion and per-decision refresh documented as adapter assumptions. HomeGuard remains advisory. D retains the full-episode five-view veto and at most three proposals per decision; this differs from EMBGuard’s original annotated-step evaluation.

The planner remains **Qwen3-VL-8B-Thinking**, sampled at temperature 1.0 / top-p 0.95 / top-k 20, with 8192 output tokens and one formatting attempt per proposal. Guards remain greedy with 2048 tokens. No perception hints, executor-error feedback, repeated-action suppression, model fine-tuning or automatic success-seeking retries were added.

Raw files are saved under `MyDrive/isbench-safeguard-evaluation/large_seed0/tasks20/<condition>/<run-id>`. Each task has its own folder. Images, requests, completed model outputs and progress are written during the task; task and batch summaries are saved after every attempt. The compact ZIP/CSV is refreshed after every chunk, including a stopped chunk. The whole notebook does not need to finish for completed-task results to be saved. Generated tokens are not backed up individually; Drive synchronization can take a short time.

The Download cell saves a ZIP copy on your computer. Save downloaded reports locally under `isbench-safeguard-evaluation/results/large_seed0/tasks20`, separate from `results/sampling`. Compact ZIPs support offline comparison; retain the raw Drive folder for detailed audit. Full evidence archives are optional to avoid rebuilding large ZIPs after every chunk.

The current filenames and `large_seed0` notebook folder are retained. The updated notebooks use a new helper directory, `/content/isbench-safeguard-evaluation-20tasks-seed0`, while reusing prepared environments, assets and checkpoint caches. Any earlier 161-task results remain untouched; create a new 20-task run rather than resuming a different task pool. Open the updated local notebooks and run their helper/preflight cells before starting.

Chunking protects progress; it does not make an individual task run faster. Eighteen selected tasks have at most six reference entries including DONE, but two scenes require longer tasks. [Local validation](../../../docs/large_seed0_validation.md) covers preparation, prompt isolation, resume, scoring and export; the 20-task GPU experiment is for you to run.
