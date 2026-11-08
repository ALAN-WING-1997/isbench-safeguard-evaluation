# IS-Bench safeguard evaluation

This project compares four safeguard conditions and a revised HomeGuard policy on the same **20 IS-Bench tasks across 16 default scenes**, using seed **0** and a pinned Qwen3-VL-8B-Thinking planner. The tasks were selected for scene and risk coverage. The results describe this task pool and one seed. See the [evaluation protocol](docs/large_seed0_protocol.md) and [source provenance](docs/sources.md) for the task selection, model revisions and paper adaptations.

## Run in Colab

Open [shared setup](notebooks/00_setup_colab.ipynb) on an L4 GPU with a High-RAM runtime. Choose **Run All**, then wait for `ALL_SETUP_READY`. Use the same runtime for the condition you want to run.

| Notebook | Condition |
| --- | --- |
| [A baseline](notebooks/10_A_baseline.ipynb) | IS-Bench V0 planner prompt |
| [B generic reminder](notebooks/11_B_generic_reminder.ipynb) | IS-Bench V1 with a generic safety reminder |
| [C HomeGuard L3](notebooks/12_C_homeguard_l3.ipynb) | HomeGuard predictions inform the IS-Bench V3/L3 planner |
| [D EMBGuard](notebooks/13_D_embguard.ipynb) | EMBGuard checks proposed actions before execution |
| [C improved](notebooks/14_C_improved.ipynb) | Constrained HomeGuard responses, a retry for failed views, and revised guidance with execution feedback |

Use [Compare A–D](notebooks/20_compare_ABCD.ipynb) with four saved runs. Improved C includes its own comparison with original C. The [notebook guide](notebooks/README.md) explains storage, continuation and comparison. Earlier pilots, diagnostic notebooks and the individual setup stages are in the [notebook archive](notebooks/archive/).

## Results

The [results guide](results/README.md) documents how the five runs were audited. The [condition metrics](results/published/condition_metrics.csv) and [per-task outcomes](results/published/per_task.csv) report coverage, task success and safe task success. Full-pool success rates require valid evaluations for all 20 tasks. The raw evidence, including simulation images, worker requests and responses, logs and ZIP exports, is stored locally or in the original Drive runs.

## Repository layout

- `scripts/` contains the evaluation, inference, export and comparison code.
- `experiments/` contains the pinned model revisions, task selection and runtime settings.
- `tools/` builds and validates notebooks and contains the session and accounting helpers.
- `tests/` contains CPU checks. `docs/` records protocols, validation and provenance.

## Local checks

From this repository, run:

```bash
python3 -m unittest discover -s tests
python3 tools/validate_notebooks.py
```

The CPU suite reports skips for checks that need the upstream checkout or saved local run artifacts. To enable the upstream-dependent checks, put the pinned checkout next to this repository:

```bash
git clone https://github.com/AI45Lab/IS-Bench.git ../is-bench-reproduction
git -C ../is-bench-reproduction checkout --detach 6a406e162feca1c4c455d9c9f1f58832edbdc1fd
```

The committed notebooks preserve their saved helper snapshots and execution outputs. To generate fresh active notebooks from the current source, run `python3 tools/build_evaluation_notebooks.py` and `python3 tools/build_c_improved_notebook.py`. The builders replace the generated notebooks, so keep a separate copy of any execution outputs you need.
