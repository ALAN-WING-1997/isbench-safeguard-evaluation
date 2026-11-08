# Two-case safeguard pilot

These notes describe the earlier two-case experiments. See the [current evaluation](../../../README.md) for A–D on the 20-task pool and improved C.

An independent **two-case integration trial** of four conditions, with one common open-weight planner:

| Condition | Planner prompt | Safeguard |
| --- | --- | --- |
| A: baseline | IS-Bench V0 | None |
| B: generic reminder | IS-Bench V1 | None |
| C: HomeGuard guidance | V1 plus model-predicted safety notes | HomeGuard-8B before each decision |
| D: EMBGuard intervention | V1 plus feedback after rejected proposals | EMBGuard-4B checks proposed physical actions |

All conditions use the locked **Qwen3-VL-8B-Thinking** checkpoint. Models are downloaded at pinned revisions and loaded with NF4 text weights, BF16 vision/computation, and SDPA. No fine-tuning is performed. These are trial notebooks, not measured results or full reproductions of the published score tables.

## Run in Colab

Use an **L4 GPU / High-RAM** runtime. Run in this order on the same runtime:

1. [00: Simulator setup](../sampling_8192/00_simulator_setup.ipynb)
2. [01: Scene data](../sampling_8192/01_scene_data.ipynb)
3. [02: Separate inference environment](../sampling_8192/02_inference_setup.ipynb)
4. [03: Test one saved planner request first](../sampling_8192/03_planner_request_replay.ipynb)
5. [A: Baseline](../sampling_8192/10_A_baseline_trial.ipynb)
6. [B: Generic reminder](../sampling_8192/11_B_generic_reminder_trial.ipynb)
7. [C: HomeGuard guidance](../sampling_8192/12_C_homeguard_trial.ipynb)
8. [D: EMBGuard intervention](../sampling_8192/13_D_embguard_trial.ipynb)
9. [Compare matched sampled runs](../sampling_8192/20_compare_trials.ipynb)

The current [sampled-protocol instructions](../sampling_8192/README.md) retain Qwen3-VL-8B-Thinking, use temperature 1.0 / top-p 0.95 / top-k 20 with an 8192-token cap, and allow one planner formatting attempt per proposal. Guards remain greedy at 2048 tokens. The worker now passes and records those controls. This is a separate revision from all saved greedy runs. Test the latest C prompt/images once in notebook 03 before another full simulation; a completed JSON answer does not establish task or safety success. Export the active run before starting new work on the same GPU.

An intact runtime from the previous study can reuse simulator/data preparation. Run 02 to create the separate `isbench-guard-vl` inference environment; it leaves `isbench-gpu`'s model packages unchanged. Notebook 01 includes two small benchmark import dependencies needed if setting up a completely fresh runtime. If reusing an older prepared runtime, those should already be installed by the previous model notebook.

For the observed planner truncation in the saved A/B runs, use [03: Planner generation diagnostic](03_planner_generation_diagnostic.ipynb) with the local saved input ZIP at `diagnostics/inputs/qwen_planner_diagnostic_inputs.zip`. It replays the original observations without starting OmniGibson, testing a larger output budget and sampling separately. See [diagnostic instructions](../../../diagnostics/README.md). The saved sink diagnostic completed at 8192 tokens under both greedy and sampling decoding. The current rerun notebooks apply the larger planner budget and accept plain or fenced final JSON. The original notebooks, diagnostic ZIP, diagnostic summary, and earlier A/B evidence are preserved. See [rerun instructions](../rerun_8192/README.md).

Each condition notebook is standalone after runtime setup: it embeds canonical helpers, the two-case manifest, model locks, and runtime settings. You do not need to upload the local scripts folder. Execute its cells in order: prepare cases/models → create a unique batch → run sink → run fan → export. The second trial is a separate cell so the first integration outcome can be inspected. Export partial evidence when a run fails, before the temporary VM expires.

For the saved C `guard_format_error`, use the [format-fix instructions](../homeguard_format_fix/README.md). This needs no additional diagnostic notebook or input ZIP. All ten saved guard responses pass offline format reinspection while retaining their original verdicts, hazard text and coordinates. The new C notebook uses a separate helper directory and records each conversion. Earlier runs retain their original failure classifications. Its revised C interface has a different code snapshot, so the existing comparison notebook will refuse automatic pooling with old runs; retain the revision label for later comparison analysis.

For D's processor `TypeError` and unexecuted export cell, use the [D fix and recovery instructions](../embguard_message_fix/README.md). The new notebook fixes system-message content blocks, checks the processor before simulation, exports partial sink failures automatically, and skips the fan after an infrastructure failure. An early recovery cell can export the previous D batch if it remains on the connected runtime, without repeating inference. D's original executed notebook remains evidence of the failed integration.

The trial tasks are `clean_a_kitchen_sink__with_blender` and `moving_electric_fan_on_sofa`, both in `Rs_int`. Reference plans are never executed or provided to models; their lengths only set the common action budget. Each case uses a fresh simulator process and the saved authored scene. Initial poses and observations are recorded for reset auditing.

## Storage and GPU behavior

The planner and HomeGuard downloads are roughly 16 GB each, and EMBGuard roughly 9 GB. NF4 is applied at loading time and does not shrink the original downloads. Assets and model snapshots are cached and shared between notebooks. Inference runs in separate short-lived processes: the planner and guard are not simultaneously loaded, although the simulator remains resident. The first episode checks actual available GPU memory and numerical output. The saved A/B trials loaded the planner alongside OmniGibson, but their 2048-token responses truncated. The 8192-token diagnostic ran without the simulator. Full rerun completion and C/D guard compatibility still require GPU verification.

## What the trial changes

- C evaluates HomeGuard separately on all five views, passes the view-indexed predictions to the V1 planner, and refreshes them before each decision. Boxes are advisory and are not converted into physical trajectories. This is an adapted prompt, not the exact paper L3 prompt.
- D evaluates the proposed action separately on all five views. All views must say safe before execution. An unsafe judgment returns the action and explanations to the planner; at most three physical-action proposals are allowed per decision. Malformed/truncated assessments stop the episode as a model-format failure. DONE receives final-state evaluation rather than a physical-action guard call.
- Planner formatting permits one attempt per proposal in every condition. The common planner cap is 8192 tokens with sampling; guard caps remain 2048 with greedy decoding. D's distinct safety-driven proposals remain limited to three. Plain and fenced final JSON receive the same operator/object/argument validation. Truncation is recorded; the runner does not recover an action from unfinished reasoning. Older greedy runs remain separate evidence.
- Evaluation safety labels, annotated tips, and example actions never enter model requests. Shared task goals, object abilities, and wash rules remain common planner inputs. Awareness/API judging and executor-error feedback are disabled.

See [the protocol](../../../docs/trial_protocol.md) for scoring and scope. C/D compare complete systems with different intervention mechanisms and guard model sizes, not isolated model quality. Model checkpoints can be evaluated without downloading HomeSafe or EMBHazard training datasets.

[Local validation](../../../docs/validation.md) covers notebook syntax, annotation isolation, withholding unsafe proposals, scoring, and export. Full episodes using the updated planner budget still require a Colab run.
