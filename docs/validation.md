# Trial setup validation

The notebooks have been created and checked locally. No new model generation or OmniGibson episode has been run. Real GPU memory use, numerical behavior, model response formatting, simulator execution, and latency still require the first Colab trial.

Local checks completed:

- 13 CPU tests passed, covering annotation isolation with actual upstream task prompts, strict action/guard parsing, five-view gating, blocked-episode scoring, reset comparison, paired-protocol checks, and evidence export.
- The real episode loop was exercised with mocked simulator and inference boundaries. Three unsafe proposals executed nothing; a rejected proposal was replaced by a safe alternative; malformed HomeGuard stopped before planning; malformed EMBGuard stopped before execution. These tests establish control flow, not simulator/model compatibility.
- All eight notebooks passed structural checks. All 47 Python code cells and every Python helper compiled. The six notebooks that embed helpers contain the current canonical helper/config bundle, with empty outputs and unique cell IDs. Formal validation through the optional `nbformat` package was unavailable in the system Python.
- All 3,486 files in the saved pre-change inventory of the previous experiment suite and upstream checkout retained identical hashes.

Recheck after editing canonical helpers:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s isbench-safeguard-evaluation/tests -v
PYTHONDONTWRITEBYTECODE=1 python3 isbench-safeguard-evaluation/tools/build_notebooks.py  # updates only notebooks/archive/sampling_8192
PYTHONDONTWRITEBYTECODE=1 python3 isbench-safeguard-evaluation/tools/validate_notebooks.py
```

Start with A's first case on the GPU. Preserve and export any failure before changing a dependency or generation setting. If the common protocol changes, rerun all affected conditions with the same settings instead of combining incompatible runs.

## Parser and budget update, 9 October 2026

The updated parser was applied offline to the user's saved sink diagnostic responses. Both `sample_8192` (3189 output tokens) and `greedy_8192` (2753 output tokens) have completed reasoning and a valid plain-JSON action: `soak_inside(sponge.n.01_1, liquid_soap__bottle.n.01_1)`. The `sample_2048` response remains rejected as truncated. This reinspection runs no model and executes no simulator action. Original diagnostic summaries retain the earlier parser's classification; they are not rewritten.

All 24 CPU tests pass, including saved-output regression, plain-JSON execution through the mocked episode loop, 8192/2048 planner/guard budget propagation, and rejecting a JSON-looking Thinking-model output that never closes reasoning. All 14 notebooks (9 preserved originals and 5 clean reruns), 86 Python cells, and 12 embedded bundles validate. Historical notebook bundles are checked against their preserved hashes; new bundles must match current canonical sources.

The rerun bootstrap clears cached trial helper imports before loading the new helpers, allowing reuse of an already prepared notebook kernel. A separate runtime suite directory avoids conflicts with old files. Model/simulator environments, shared scene paths, and checkpoint cache paths remain the same. GPU inference at the larger budget with OmniGibson resident has not been rerun locally.

A bootstrap smoke check loaded the old cached parser, executed the new notebook bootstrap, and confirmed that the next import used the updated parser and 8192/2048 settings. Re-executing the bootstrap also succeeded. All 207 protected original notebook, result, and input-archive files retained identical hashes.

## HomeGuard format update, 9 October 2026

The new C notebook preserves the executed C notebook and failed batch `20261009T061355_157248Z_48df1495`. Its prompt retains the released safety-assessment instruction and replaces the pseudocode output schema with literal JSON syntax. The box adapter handles only defined representations: four-key coordinate dictionaries, flat single boxes, and empty/plain-text state suffixes. Verdicts, hazard strings and coordinates must be present, validated and unchanged; state suffixes remain in the audit. No extra inference request, inferred safety label or simulator action is introduced by normalization.

Reinspection of all ten saved responses yields two accepted and eight normalized assessments, with no remaining format rejection. Each batch now supplies all five assessments to the planner in a mocked episode-loop check. This is parsing/control-flow evidence; it does not establish visual grounding, hazard correctness, planner completion or task success. The original reports are not rewritten. See `homeguard_format_reinspection.json` for response hashes, conversions and normalized assessments.

All 29 CPU tests pass, including regression against the ten actual responses, exact preservation of independently decoded hazard text and expected coordinates, rejection of ambiguous/duplicate/incomplete/nonfinite outputs, truncation refusal, and the existing planner/EMBGuard behavior. All 15 notebooks, 93 Python cells and 13 embedded bundles validate. Historical 8192 notebook bundles now join the preserved bundle inventory; only the new C copy must match current canonical helpers. GPU generation with the clarified prompt and the full simulator episode still require the user's Colab run.

A bootstrap smoke check verified replacement of cached helpers from the older 8192 suite, exact extraction of the new embedded sources, successful repeated bootstrap, and unchanged 8192/2048 planner/guard budgets. All 358 files in the pre-fix inventory of notebooks, results and diagnostic input archives retain identical hashes.

## EMBGuard message/export update, 9 October 2026

The executed D notebook records a sink failure in multimodal preprocessing: its system message was a string, while the processor's visual scan expected content dictionaries. EMBGuard did not produce an assessment. The batch rejected the subsequent fan run after that infrastructure failure, and the export cell was not executed. The raw D batch is not present locally; recovery depends on its files remaining in Colab.

The inference helper now represents system instructions as text content blocks, preserving text and role. A CPU-only processor preflight checks a system/image request during EMBGuard preparation. The new D notebook can export the old batch directly by path; after a new sink infrastructure failure it creates partial archives automatically and skips the fan cell, allowing export to run. Download-widget failures leave archives available through the runtime file browser.

All 34 CPU tests pass. New checks reproduce the processor operation that failed, verify unchanged no-system user messages, exercise the CPU preflight with a mock processor, and execute the generated notebook's failure/export/recovery cells against temporary saved batches. Tests verify archived log evidence and recovery without a live `batch` variable. All 16 notebooks, 101 Python cells and 14 embedded helper bundles validate. The previously delivered C notebook is frozen in the historical bundle inventory. Actual pinned-processor preprocessing and subsequent GPU/simulator inference remain to be checked on Colab.

## Sampled planner and one-attempt revision, 9 October 2026

All four conditions now use the pinned Thinking planner with sampling (temperature 1.0, top-p .95, top-k 20, repetition penalty 1.0), seed 0, output limit 8192 and one formatting attempt per proposal. Runtime construction explicitly strips planner sampling fields for guards, which retain greedy decoding and their 2048-token cap. D retains three distinct safety-driven proposals. Worker responses record effective generation kwargs and validate settings before loading GPU weights. No checkpoint, scene, action schema or scoring change was made.

Nine new notebooks in `sampling_8192` cover setup 00–02, saved-request replay 03, A–D and comparison. Every embedded bundle matches current canonical helpers. Earlier executed notebooks and archives are frozen. The replay checks original prompt/image hashes and model revision, then calls the planner once, retaining raw output/errors and image evidence. It performs only completed final-JSON checks; no guard/simulator/action execution or correctness/safety evaluation occurs. Recorded worker failures remain exportable.

All 53 CPU tests pass, including sampled kwargs reaching the worker, guard decoding isolation, one truncated attempt stopping each condition, continued safety-driven D replanning, replay provenance and missing/tampered-input rejection, one worker call and archived child errors, strict comparison rejection of incompatible decoding/attempt settings, previous C/D fixes, and historical diagnostic-worker renderer compatibility. All 25 notebooks, 153 Python cells and 21 embedded bundles validate. Actual sampled GPU generation and its effect on task success remain untested locally.

All seven new standalone bundles are identical. A bootstrap smoke check replaces cached helpers from the previous D suite, removes stale helper paths, verifies sampled planner versus greedy guard kwargs and 1/3 format/proposal budgets, and succeeds when repeated. All 360 protected pre-change notebook, result and diagnostic-input files retain identical hashes.
