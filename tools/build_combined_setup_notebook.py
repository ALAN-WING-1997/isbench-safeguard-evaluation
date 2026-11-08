"""Combine the four setup stages into one Colab notebook."""
import hashlib

from build_notebooks import cell

NAME = '00_setup_colab.ipynb'
STAGES = (
    ('00_simulator_setup.ipynb', 'Simulator setup'),
    ('01_scene_data.ipynb', 'Scene and object data'),
    ('02_inference_setup.ipynb', 'Inference environment and planner checkpoint'),
    ('03_task_preflight.ipynb', 'Verify the 20 tasks'),
)

INTRODUCTION = '''# Complete setup for the 20-task evaluation

This combines **00, 01, 02 and 03** in their original execution order. Select an **L4 GPU / High-RAM** Colab runtime, connect this notebook, and choose **Run All**. The stages install the simulator, prepare the scene and object data, prepare the inference environment and planner checkpoint, and verify the selected tasks.

Wait for **ALL_SETUP_READY** at the bottom. Open your evaluation notebook in the same runtime to run or resume a condition. Setup reuses available caches.
'''

COMPLETION = """assert EVALUATION_BOOTSTRAP_READY, 'Complete the setup stages above.'
assert SIM_PYTHON.is_file() and INFERENCE_PYTHON.is_file(), 'A required Python environment is missing.'
assert len(prepared) == selection_spec['expected_tasks'] == 20, 'The 20-task preflight did not complete.'
prepared_scenes = {case['scene'] for case in prepared}
assert len(prepared_scenes) == selection_spec['expected_default_scenes'] == 16
assert prepared_scenes == set(selection_spec['default_scenes']), 'Prepared scenes differ from the selection.'
setup_model_lock = json.loads((SUITE / 'experiments/models.json').read_text())['planner']
setup_planner = json.loads((SUITE / 'planner_metadata.json').read_text())['planner']
assert all(setup_planner[key] == setup_model_lock[key] for key in ('repo_id', 'revision')), 'Planner checkpoint differs from the lock.'
assert Path(setup_planner['model_path']).is_dir(), 'Planner checkpoint cache is missing.'
print('ALL_SETUP_READY | 20 tasks | 16 scenes | seed:', RUNTIME['seed'])
print('Open your condition notebook in this same runtime. For C, use its saved Drive run in Create or resume.')
"""


def _combined_cell(kind, source, origin):
    result = cell(kind, source)
    result['id'] = hashlib.sha256((origin + '\n' + kind + '\n' + source).encode()).hexdigest()[:12]
    result['metadata'] = {'setup_origin': origin}
    return result


def _stage_markdown(source, stage, title):
    if source.startswith('# '):
        source = f'## Stage {stage}: {title}\n' + '\n'.join(source.splitlines()[1:])
    replacements = {
        'Connect this notebook to the **same Colab runtime** that completed `00_simulator_setup.ipynb`.':
            'Continue in the same runtime after the simulator setup stage above.',
        'Run on the same L4 / High-RAM Colab runtime after notebooks 00 and 01.':
            'Continue after the simulator and scene-data stages above.',
        'Run after shared setup 00–02, or reuse an intact prepared runtime.':
            'Continue after the three setup stages above.',
        'Continue with **03_task_preflight** before the evaluation notebooks.':
            'Continue to the task preflight stage below before the evaluation notebooks.',
        'After `SCENE_DATA_READY`, prepare inference in notebook 02 and verify the selected 20 tasks in notebook 03.':
            'After `SCENE_DATA_READY`, the inference and task preflight stages follow below.',
        'Model inference dependencies go into the separate environment in notebook 02.':
            'Model inference dependencies go into the separate environment in the next stage.',
        'Proceed to A, B, C or D after `TASKS_READY: 20`.':
            'Proceed to the final readiness check below after `TASKS_READY: 20`.',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def combined_cells(stages):
    cells = [_combined_cell('markdown', INTRODUCTION, 'combined:introduction')]
    for stage, (name, title) in enumerate(STAGES, 1):
        for index, original in enumerate(stages[name]):
            kind = original['cell_type']
            if kind not in ('code', 'markdown'):
                raise ValueError('Unsupported setup cell: ' + kind)
            source = ''.join(original['source'])
            if kind == 'markdown':
                source = _stage_markdown(source, stage, title)
            cells.append(_combined_cell(kind, source, name + ':' + str(index)))
    cells.extend([
        _combined_cell('markdown', '## Setup complete\n\nRun this final check, then open your condition notebook in the same runtime. Its preparation cells select and verify the condition; C and D also prepare their guard checkpoints.\n', 'combined:ready-description'),
        _combined_cell('code', COMPLETION, 'combined:ready-check'),
    ])
    return cells


def build():
    from build_evaluation_notebooks import build as build_evaluation
    build_evaluation(names={NAME})


if __name__ == '__main__':
    build()
