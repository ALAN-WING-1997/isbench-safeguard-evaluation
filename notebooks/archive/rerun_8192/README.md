# Updated trial notebooks

This archive records the 8192-token greedy A–D rerun. The [original pilot notebooks](../pilot/) contain the earlier execution outputs.

## What changed

- Planner final answers can be plain JSON or a JSON Markdown fence. The same action, operator, argument, and object validation applies.
- Every condition uses greedy planner decoding with an 8192-token output cap, including reasoning tokens. The diagnostic showed that this larger budget completed the saved sink input.
- HomeGuard/EMBGuard caps remain 2048 tokens. Model revisions, NF4 loading, views, image budgets, prompts, and scoring stay the same.
- New runtime files and results use `/content/isbench-safeguard-evaluation-8192`. Shared environments, scene assets, and model downloads are reused. Cached helper imports are cleared before loading the updated code.

## Run next

1. Open `10_A_baseline_trial.ipynb` from this directory and connect it to your prepared Colab runtime. If your existing runtime still has the simulator, scenes, and inference environment, skip setup notebooks 00–02. A fresh runtime still requires those original setup notebooks.
2. Run the notebook's cells from the beginning. The first cell should print a planner cap of 8192 and a guard cap of 2048. The new notebook embeds all required helpers; no diagnostic ZIP is needed.
3. Run the sink case, then the fan case. Each starts a separate fresh simulator process. This is the actual trial rerun; no further standalone diagnostic is required first.
4. Export the summary and full evidence, even if an episode fails. Download the full archive parts listed by the export cell through the runtime file browser. Put the extracted evidence under the local suite's `results` directory, keeping earlier runs.
5. Continue with the B, C, and D notebooks in this directory using the same settings. Use this directory's comparison notebook with four new run directories.

The token budget is a maximum; a completed answer can finish sooner. A two-hour episode timeout and 30-minute worker timeout remain in place. A longer budget can increase runtime and memory use. Full simulator performance and task/safety success must be established by these runs.

Do not run the older 10–13 notebooks for this rerun: those preserve the original 2048-token protocol. Do not mix their results with the new protocol in the paired comparison.
