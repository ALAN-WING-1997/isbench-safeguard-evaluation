# C: HomeGuard format fix

This archive records the HomeGuard format fix for the two-case pilot. See the [original pilot](../pilot/) and [8192-token greedy rerun](../rerun_8192/) for earlier notebook outputs.

1. Open this notebook in Antigravity and connect it to the prepared Colab GPU runtime.
2. Run the cells from the top. If the runtime still has its environments and scene assets, setup notebooks 00–02 do not need to run again. The preparation cells verify them and reuse cached models.
3. Run the sink, then the fan, then export results even if a case fails. Save the new full evidence under the suite's `results/raw/` alongside the old runs.

No diagnostic notebook or input ZIP is needed. The notebook embeds all updated helpers. It creates `/content/isbench-safeguard-evaluation-homeguard-format-v2`, so the older helper files do not conflict. Each run gets a fresh batch directory.

The HomeGuard prompt replaces the released pseudocode schema with literal JSON syntax. The parser accepts coordinate dictionaries with exactly `x_min`, `y_min`, `x_max`, `y_max`; a single four-number box; and one empty or plain-text state suffix following a box. It converts only the representation, preserving verdict, hazard text and coordinates. State suffixes are recorded as model predictions in the audit, not interpreted as safety labels. All five assessments must be readable before the planner is called. Unknown or duplicate fields, missing values, invalid coordinates, ambiguous suffixes and token truncation still stop the episode.

Each `step_XX_homeguard.json` includes the five assessments and `format_audit`, with `accepted`, `normalized` or `rejected` status and exact conversions/errors. Raw worker responses remain saved separately. Normalization uses no extra model call. Models, quantization, decoding, token budgets, scene resets and scoring retain their existing settings.

Offline, all ten responses from C batch `20261009T061355_157248Z_48df1495` can now be read: two unchanged, eight normalized. This does not validate their hazard predictions or make the original failed episodes successful. The new GPU simulation must run to produce a new outcome; planner truncation can still occur independently.

This is a C interface revision. The existing comparison helper deliberately refuses automatic pooling with different code snapshots. Keep this version identified in analysis; matching planner settings alone does not bypass that check. No A/B rerun is needed to check this C fix.
