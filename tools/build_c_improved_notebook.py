"""Build the isolated 20-task, seed-0 improved Condition C notebook."""
from build_evaluation_notebooks import (
    CONTINUOUS_DOWNLOAD, CONTINUOUS_FULL_EXPORT, CREATE_OR_RESUME, DESTINATION,
    IMPORT_CHECKS, PREFLIGHT, RUN_ALL, SETUP_NAME, STORAGE, SUITE_PATH, continuous_bootstrap,
)
from build_notebooks import cell, write

NAME = '14_C_improved.ipynb'
IMPROVED_SUITE = SUITE_PATH + '-c-improved'
RESULTS_NAME = 'tasks20_c_improved'
ORIGINAL_C_RUN_ID = '20261012T204052_116460Z_60a5f149'


def improved_bootstrap():
    source = continuous_bootstrap()
    source = source.replace(f'\nSUITE = Path({SUITE_PATH!r})\n',
                            f'\nSUITE = Path({IMPROVED_SUITE!r})\n', 1)
    source = source.replace("helper_roots = (SUITE / 'scripts',",
                            f"helper_roots = (SUITE / 'scripts', Path({(SUITE_PATH + '/scripts')!r}),", 1)
    return source + """
assert 'scripts/condition_c.py' in SOURCES and 'scripts/homeguard_json.py' in SOURCES
RUNTIME['condition_c_improvements'] = True
print('Improved Condition C enabled.')
"""


PREPARE_MODELS = """models = None
metadata_file = SUITE / 'models_C.json'
run_command(INFERENCE_PYTHON, SUITE / 'scripts/prepare_models.py', '--lock', SUITE / 'experiments/models.json',
            '--roles', 'planner', 'homeguard', '--metadata', metadata_file)
prepared_models = json.loads(metadata_file.read_text())

# Check/install in the dedicated inference interpreter, not the notebook kernel.
version_check = (
    "from importlib.metadata import version; assert version('lm-format-enforcer') == '0.11.3'"
)
if subprocess.run([str(INFERENCE_PYTHON), '-c', version_check], capture_output=True).returncode:
    run_command(INFERENCE_PYTHON, '-m', 'pip', 'install', 'lm-format-enforcer==0.11.3')
run_command(INFERENCE_PYTHON, '-c', version_check)
run_command(INFERENCE_PYTHON, '-c', '''
import sys
from transformers import AutoProcessor
sys.path.insert(0, sys.argv[1])
from homeguard_json import preflight_homeguard_json
processor = AutoProcessor.from_pretrained(sys.argv[2], local_files_only=True, trust_remote_code=False)
preflight_homeguard_json(processor.tokenizer)
print("HOMEGUARD_JSON_PREFLIGHT_READY")
''', SUITE / 'scripts', prepared_models['homeguard']['model_path'])
models = prepared_models
print(json.dumps(models, indent=2))
"""


COMPARE_C = f"""from compare_trials import same_initial_objects

# Replace this path if original C was saved somewhere else.
ORIGINAL_C_RUN = Path('/content/drive/MyDrive/isbench-safeguard-evaluation/large_seed0/tasks20/C/{ORIGINAL_C_RUN_ID}')
assert batch is not None and batch.runtime['condition_c_improvements'] is True
if not (ORIGINAL_C_RUN / 'batch_manifest.json').is_file():
    print('To compare with original C, set ORIGINAL_C_RUN to its saved result directory.')
else:
    improved_run = batch.output
    run_paths = {{'original_C': ORIGINAL_C_RUN, 'improved_C': improved_run}}
    baseline, candidate = [
        json.loads((path / 'batch_manifest.json').read_text()) for path in run_paths.values()
    ]
    assert baseline['condition'] == candidate['condition'] == 'C', 'Both runs must be Condition C.'
    baseline_runtime, candidate_runtime = dict(baseline['runtime']), dict(candidate['runtime'])
    assert baseline_runtime.pop('condition_c_improvements', False) is False, 'Choose original C as baseline.'
    assert candidate_runtime.pop('condition_c_improvements', False) is True, 'Choose improved C as candidate.'
    assert baseline_runtime == candidate_runtime and baseline_runtime['seed'] == 0, 'Runtime settings differ.'
    assert all(
        baseline[key] == candidate[key]
        for key in ('upstream_git_commit', 'upstream_dependency_sha256', 'prepared_config_sha256')
    ), 'Frozen benchmark or task definitions differ.'
    case_keys = ('task', 'scene', 'scene_sha256', 'task_config_sha256', 'max_steps')
    assert len(baseline['tasks']) == len(candidate['tasks']) == 20 and [
        tuple(case[key] for key in case_keys) for case in baseline['tasks']
    ] == [
        tuple(case[key] for key in case_keys) for case in candidate['tasks']
    ], 'Task order, assets or action budgets differ.'
    locked_models = json.loads((SUITE / 'experiments/models.json').read_text())
    for manifest in (baseline, candidate):
        assert set(manifest['models']) == {{'planner', 'homeguard'}}, 'Expected planner and HomeGuard only.'
        for role, metadata in manifest['models'].items():
            assert all(metadata.get(key) == val for key, val in locked_models[role].items()), \
                'Locked model revision differs.'

    reports = {{label: account_evaluation(path) for label, path in run_paths.items()}}
    rows = {{
        label: {{row['task']: row for row in report['results']}}
        for label, report in reports.items()
    }}
    per_task = []
    jointly_valid = 0
    pose_matched = []
    for case in baseline['tasks']:
        task = case['task']
        pair = {{label: indexed.get(task) for label, indexed in rows.items()}}
        state_paths = [path / task / 'initial_state.json' for path in run_paths.values()]
        poses_match = all(path.is_file() for path in state_paths) and same_initial_objects(
            *(json.loads(path.read_text()) for path in state_paths)
        )
        valid_pair = all(row and row.get('valid_evaluation') is True for row in pair.values())
        jointly_valid += int(valid_pair)
        if valid_pair and poses_match:
            pose_matched.append(pair)
        per_task.append({{
            'task': task, 'scene': case['scene'], 'both_evaluations_valid': valid_pair,
            'initial_poses_match': poses_match, **pair,
        }})

    metrics = {{label: report['metrics'] for label, report in reports.items()}}
    original = metrics['original_C']
    if 'planner_token_budget_hits' not in original:
        # The original run counted all model limit hits. Derive the planner
        # subset only when the raw outputs are present for every saved attempt.
        raw_planner_hits = 0
        for row in reports['original_C']['results']:
            expected = row['completed_model_generations'] - row['completed_guard_generations']
            outputs = [
                output
                for path in (ORIGINAL_C_RUN / row['task']).glob('*_planner_response.json')
                for output in json.loads(path.read_text()).get('outputs', [])
            ]
            if len(outputs) != expected or any('hit_token_limit' not in output for output in outputs):
                break
            raw_planner_hits += sum(output['hit_token_limit'] is True for output in outputs)
        else:
            original['planner_token_budget_hits'] = raw_planner_hits
    fields = ('attempted_tasks', 'valid_evaluations', 'task_successes', 'safe_task_successes',
              'SR_all_tasks', 'SSR_all_tasks', 'SR_evaluated_tasks', 'SSR_evaluated_tasks',
              'SR_all_tasks_bounds', 'SSR_all_tasks_bounds', 'triggered_safety_checks',
              'passed_safety_checks', 'completed_model_generations', 'completed_guard_generations',
              'token_budget_hits', 'planner_token_budget_hits', 'guard_retried_views',
              'guard_recovered_views', 'guard_unresolved_views', 'termination_counts')
    headline = {{
        label: {{key: metric.get(key) for key in fields}}
        for label, metric in metrics.items()
    }}
    paired_totals = {{
        label: {{key: sum(pair[label].get(key) is True for pair in pose_matched)
                for key in ('task_completed', 'safe_task_completed')}}
        for label in run_paths
    }}
    comparison = {{
        'runs': {{label: str(path) for label, path in run_paths.items()}},
        'metrics': metrics, 'per_task': per_task, 'jointly_valid_tasks': jointly_valid,
        'jointly_valid_tasks_with_matched_poses': len(pose_matched),
        'completion_counts_on_jointly_valid_pose_matched_tasks': paired_totals,
        'note': 'One seed tests the combined intervention. Matching poses do not establish equal hidden simulator state. Check coverage may differ.',
    }}
    from datetime import datetime, timezone
    import uuid
    comparison_dir = RESULTS_ROOT / 'analysis' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                                   + '_' + uuid.uuid4().hex[:8])
    comparison_dir.mkdir(parents=True, exist_ok=False)
    (comparison_dir / 'c_comparison.json').write_text(json.dumps(comparison, indent=2) + '\\n')
    print(json.dumps(headline, indent=2))
    print('Jointly valid:', jointly_valid, '| matching recorded poses:', len(pose_matched))
    print('COMPARISON_DIRECTORY:', comparison_dir)
"""


def build():
    storage = STORAGE.replace('/large_seed0/tasks20', f'/large_seed0/{RESULTS_NAME}')
    create = CREATE_OR_RESUME.replace(
        "active_run_key = (RUNTIME['selection_id'], CONDITION)",
        "active_run_key = (RUNTIME['selection_id'], CONDITION, str(RESULTS_ROOT))",
    )
    write(NAME, [
        cell('markdown', f"""# Improved Condition C on the original 20 tasks

This run tests all three changes together. HomeGuard generates schema-constrained JSON, retries only unreadable views once, and stops the episode before a physical action if a view stays unresolved. HomeGuard gets task progress and available objects so it can focus on visually supported, relevant risks. The V3 planner receives consolidated evidence, explicit object and action constraints, and actual execution feedback.

The task order, scene and asset pins, seed 0, planner and HomeGuard revisions, action limits, sampling settings and output token caps match original C. Each task starts a fresh simulator. This experiment estimates the combined effect; it cannot attribute a change to one intervention.

On a fresh L4 / High-RAM Colab runtime, open **{SETUP_NAME}** and choose **Run All**. Wait for `ALL_SETUP_READY`, then connect this notebook to the same runtime. Reuse completed setup and model downloads in an intact runtime. This notebook repeats task preflight and includes all revised helpers. No diagnostic ZIP is needed.

Google Drive results default to `large_seed0/{RESULTS_NAME}/C/<run-id>`. Save the printed `RESULT_DIRECTORY` for resuming. Run only one simulator batch on the GPU at a time.
"""),
        cell('code', improved_bootstrap()),
        cell('markdown', '## Verify the original task selection and environments\n'),
        cell('code', "batch = None\nprepared = None\nmodels = None\nCONDITION = 'C'\n"
             + PREFLIGHT + "\nassert len(prepared) == 20\n" + IMPORT_CHECKS),
        cell('markdown', '## Mount Drive for persistent results\n\n'
             'Antigravity can mount Drive from this cell. If authorization is unavailable in the IDE, '
             'use the Colab browser. With Drive disabled, save the exports before the runtime ends.\n'),
        cell('code', storage),
        cell('markdown', '## Prepare the pinned models and validate JSON constraints\n\n'
             'This installs the pinned format enforcer into the inference environment as needed '
             'and checks it against the locked HomeGuard tokenizer before starting any episode.\n'),
        cell('code', PREPARE_MODELS),
        cell('markdown', '## Create or resume the improved C run\n\n'
             'For the first run, keep the defaults. On reconnect, paste this run’s printed path into '
             '`RESUME_RUN`. Already attempted tasks, including failures, are retained. Resume rejects '
             'a different runtime configuration and an active owner.\n'),
        cell('code', create),
        cell('markdown', '## Run the remaining tasks\n\n'
             'Run once to attempt all remaining tasks. Each attempt writes to Drive. Simulator failures '
             'remain visible as invalid evaluations and do not trigger an episode retry. '
             'The single HomeGuard retry is limited to failed camera views within a decision.\n'),
        cell('code', RUN_ALL),
        cell('markdown', '## Export the compact results\n\n'
             'Use the printed summary ZIP path in the runtime file browser if Antigravity does not '
             'show a download. Raw worker requests, responses and camera images remain in the Drive run directory.\n'),
        cell('code', CONTINUOUS_DOWNLOAD),
        cell('markdown', '## Compare with original C\n\n'
             'This checks the common task and runtime pins while allowing the deliberate policy difference. '
             'Both runs use the same effective safety-check accounting. Use the per-task evidence and '
             'check coverage to interpret the differences; skipped or invalid episodes stay visible.\n'),
        cell('code', COMPARE_C),
        cell('markdown', '## Optional full evidence archive\n\n'
             'Raw evidence is already saved in the run directory. Enable this only to package it for download.\n'),
        cell('code', CONTINUOUS_FULL_EXPORT),
    ], directory=DESTINATION)


if __name__ == '__main__':
    build()
