# Colab notebook guide

## Shared setup

Run [00 setup](00_setup_colab.ipynb) on an **L4 GPU / High-RAM** Colab runtime. It installs the simulator, prepares the scene and object data, sets up inference, and verifies the 20 selected tasks. Wait for `ALL_SETUP_READY`. The [four individual setup stages](archive/split_setup/) are preserved in the archive.

## A–D evaluation

Open [A baseline](10_A_baseline.ipynb), [B generic reminder](11_B_generic_reminder.ipynb), [C HomeGuard L3](12_C_homeguard_l3.ipynb), or [D EMBGuard](13_D_embguard.ipynb) in the prepared runtime. Prepare the pinned models and create a run. Each notebook embeds its helpers and repeats the task preflight.

A runs in chunks of five tasks. B, C and D have a **Run all remaining tasks** cell. Use one active simulator batch per GPU. With Drive enabled, results are saved under `MyDrive/isbench-safeguard-evaluation/large_seed0/tasks20/<condition>/<run-id>`. Save the printed `RESULT_DIRECTORY`.

Use [Compare A–D](20_compare_ABCD.ipynb) with one matching run directory per condition. The comparison reports incomplete coverage and checks initial poses. See the [protocol](../docs/large_seed0_protocol.md) for scoring and the [results guide](../results/README.md) for the audited run selection.

## Improved C

Open [14 C improved](14_C_improved.ipynb) after shared setup. The notebook checks the pinned HomeGuard tokenizer and tests constrained JSON with one retry per failed view, evidence-focused hazard assessment, and planner guidance with execution feedback. It uses the same task selection, seed, models, and action and token budgets as original C.

With Drive enabled, the run goes under `MyDrive/isbench-safeguard-evaluation/large_seed0/tasks20_c_improved/C/<run-id>`. To compare against your own original C run, set `ORIGINAL_C_RUN` in the comparison cell to that run's `RESULT_DIRECTORY`.

## Resume and export

After reconnecting, repeat setup if Colab replaced the runtime. Open the same condition notebook, rerun its preparation, storage and model cells, and paste the saved `RESULT_DIRECTORY` into `RESUME_RUN`. The notebook skips prior attempts and checks run identity and ownership. If a previous owner is recorded, end that runtime before setting `CONFIRM_PREVIOUS_RUNTIME_STOPPED=True`. In A's chunked workflow, inspect an infrastructure failure before using `CONTINUE_AFTER_INFRASTRUCTURE=True` to advance to the remaining tasks.

The compact ZIP contains reports, manifests and per-task evidence. Drive retains raw images, requests, responses and logs. Use the Colab download widget or the connected runtime file browser to download the printed ZIP path.

## Archive

The [two-case pilot notes](archive/pilot/README.md) describe the earlier trial, reruns and diagnostics. Those experiments used a separate task pool and protocol.
