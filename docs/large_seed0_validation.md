# Local validation of the one-seed evaluation

Validated on 2026-10-11. This records implementation checks, not measured model performance.

- **112 CPU tests passed** across the safeguard suite. They cover prompt/annotation isolation, C’s exact upstream L3 template, unchanged A/B/D prompts, guard gating, task preparation, fixed denominators, resume ownership, interruption/timeout recovery, no reruns of model failures, frozen benchmark dependencies, partial comparisons and compact/full exports.
- Notebook validation passed for **34 notebooks**, **208 Python cells** and **28 embedded helper bundles**. The nine `large_seed0` notebooks have no execution outputs, compile successfully and embed the current canonical scripts/configurations. Historical bundles remain frozen rather than being rebuilt.
- SHA256 verification confirmed **779 prior notebook/result files remained unchanged**.
- The frozen manifest selects **20 pinned task configurations**, covering **all 16 authored default scenes** and **all eight annotated risk categories**. All 20 config hashes and required templates in the actual pinned scene archive were verified. Asset-presence fixtures are synthetic; these checks do not establish that real assets load or decrypt in OmniGibson.
- The live local upstream checkout passes source verification with **504 tracked runtime dependency hashes** at the pinned commit. Tests reject modified upstream prompts/configuration and a changed upstream commit while permitting unrelated documentation/ignore-file edits.
- Compact reports can be extracted and compared without the original task repository. Tests retain invalid and unattempted tasks in coverage, reject altered task definitions or guard checkpoints, and verify that optional full export does not overwrite the larger-run compact ZIP or raw evidence.
- Notebook flow tests reject stale state after failed helpers, storage authentication, model/task preparation or resume, including a prior batch from another condition in a reused kernel.

The 20-task simulation, new C L3 integration, real Drive mounting and fresh-runtime Colab recovery require GPU/runtime verification by running the notebooks. Existing pilot results remain separate evidence. No new task-success, safety-success or runtime measurements are claimed here.

These counts describe the October validation snapshot. See the [current repository instructions](../README.md#local-checks) to repeat the checks on the reorganized notebooks.
