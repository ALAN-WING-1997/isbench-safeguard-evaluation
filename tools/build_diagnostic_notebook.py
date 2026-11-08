"""Build a separate diagnostic notebook and a verified bundle of saved A/B inputs."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

from build_notebooks import ARCHIVE, ROOT, cell, write

NAME = '03_planner_generation_diagnostic.ipynb'
BUNDLE = ROOT / 'diagnostics/inputs/qwen_planner_diagnostic_inputs.zip'


def diagnostic_sources():
    names = ('scripts/protocol.py', 'scripts/prepare_models.py', 'diagnostics/settings.py',
             'diagnostics/runner.py', 'diagnostics/worker.py', 'experiments/models.json')
    return {name: (ROOT / name).read_text() for name in names}


def diagnostic_worker_source():
    """Render the diagnostic worker without modifying any saved artifact."""
    source = (ROOT / 'scripts/vlm_worker.py').read_text()
    # Diagnostic variants use the request's separate generation object, rather
    # than the trial's settings object. Remove the trial helper to avoid hiding
    # the imported diagnostic helper with the same function name.
    helpers = [node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == 'generation_kwargs']
    assert len(helpers) == 1, 'Expected one trial generation helper.'
    lines = source.splitlines(keepends=True)
    helper = helpers[0]
    del lines[helper.lineno - 1:helper.end_lineno]
    source = ''.join(lines)
    old = '    decoding = generation_kwargs(settings)\n'
    new = '    decoding = generation_kwargs(request)  # Validate before loading weights.\n'
    assert source.count(old) == 1
    source = source.replace(old, new)
    assert source.count('import traceback\n') == 1
    source = source.replace('import traceback\n', 'import traceback\n\nfrom settings import generation_kwargs\n')
    old = '        "effective_generation_kwargs": decoding,\n'
    assert source.count(old) == 1
    source = source.replace(old, old + '        "generation": decoding, "diagnostic": True,\n')
    source = '# Generated from scripts/vlm_worker.py by tools/build_diagnostic_notebook.py.\n' + source
    ast.parse(source)
    return source


def build_worker():
    (ROOT / 'diagnostics/worker.py').write_text(diagnostic_worker_source())


def bundle_inputs():
    primitive_source = (ROOT.parent / 'is-bench-reproduction/og_ego_prim/primitives/ego_primitives.py').read_text()
    assignments = [n for n in ast.parse(primitive_source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'VALID_PRIMITIVES' for t in n.targets)]
    primitives = ast.literal_eval(assignments[0].value)
    files, cases = {}, []
    selected = {'A': '20261008T035402_826524Z_0a794fe5', 'B': '20261008T060912_151374Z_bc9ba72e'}
    for condition, run_id in selected.items():
        run = ROOT / 'results/raw' / run_id
        manifest = json.loads((run / 'batch_manifest.json').read_text())
        assert manifest['condition'] == condition
        for task in ('clean_a_kitchen_sink__with_blender', 'moving_electric_fan_on_sofa'):
            episode = run / task
            case_id = condition + '/' + task
            request = json.loads((episode / '001_planner_request.json').read_text())
            assert request['role'] == 'planner' and len(request['jobs']) == 1
            assert hashlib.sha256(request['jobs'][0]['prompt'].encode()).hexdigest() == request['prompt_sha256'][0]
            files[case_id + '/request.json'] = (episode / '001_planner_request.json').read_bytes()
            files[case_id + '/baseline_response.json'] = (episode / '001_planner_response.json').read_bytes()
            config = json.loads((episode / 'task_config.json').read_text())
            # Evaluation annotations and reference actions are excluded from this bundle.
            files[case_id + '/context.json'] = json.dumps({'objects': config['planning_context']['object_list'],
                'allowed_primitives': primitives}, indent=2).encode()
            files[case_id + '/source.json'] = json.dumps({'condition': condition, 'task': task, 'batch': run_id,
                'request_sha256': hashlib.sha256(files[case_id + '/request.json']).hexdigest(),
                'original_worker_sha256': manifest['script_sha256']['vlm_worker.py'],
                'original_episode_summary': {k: v for k, v in json.loads((episode / 'summary.json').read_text()).items()
                    if k in ('termination_reason', 'action_count', 'token_budget_hits', 'model_calls')}}, indent=2).encode()
            for i, old_path in enumerate(request['jobs'][0]['images']):
                image = (episode / '00_initial' / f'obs_{i}.png').read_bytes()
                assert hashlib.sha256(image).hexdigest() == request['image_sha256'][old_path]
                files[case_id + f'/images/obs_{i}.png'] = image
            assert len(request['jobs'][0]['images']) == 5
            cases.append({'id': case_id, 'condition': condition, 'task': task, 'source_batch': run_id})
    files['manifest.json'] = json.dumps({'cases': cases, 'file_sha256': {name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()}, 'evaluation_annotations_included': False}, indent=2).encode()
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, content)
    return hashlib.sha256(BUNDLE.read_bytes()).hexdigest()


def build():
    build_worker()
    digest = bundle_inputs()
    bootstrap = '''import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

DIAG = Path('/content/isbench-safeguard-diagnostics')
SOURCES = ''' + repr(diagnostic_sources()) + '''
for name, source in SOURCES.items():
    path = DIAG / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != source:
        raise RuntimeError('Diagnostic helper differs: ' + str(path) + '. Export results and use a new DIAG folder.')
    path.write_text(source)
sys.path[:0] = [str(DIAG / 'diagnostics'), str(DIAG / 'scripts')]
INFERENCE_PYTHON = Path('/content/miniforge3/envs/isbench-guard-vl/bin/python')
INPUT_BUNDLE_SHA256 = ''' + repr(digest) + '''

def run_command(*args):
    with subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, stdin=subprocess.DEVNULL) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, args)
print('Diagnostic folder:', DIAG)
'''
    write(NAME, [
        cell('markdown', '''# Qwen planner generation diagnostic

This replays the **exact saved first-step prompts and five images** from your A/B sink and fan trials. It runs model inference only. No OmniGibson process, action execution, HomeGuard, or EMBGuard is started.

Use the GPU runtime prepared by **02_inference_setup**. If that environment is already installed, reuse it. On a fresh runtime prepare the inference environment first; the downloaded planner can be cached. Restart the Colab session if a simulator process is still holding GPU memory.

Upload this notebook and the supplied **qwen_planner_diagnostic_inputs.zip**. You do not need the full experiment archives or the IS-Bench checkout. The ZIP contains prompts, original responses, images, and the primitive/object allowlists needed for format validation. Evaluation safety annotations and reference plans are excluded.

The archived **greedy / 2048** responses are the baseline. New variants are **sampling / 2048**, **greedy / 8192**, and **sampling / 8192**. Sampling uses temperature=1.0, top-p=0.95, top-k=20. Quantization, image budgets, checkpoint revision, and prompt content remain the same. This diagnostic changes no A–D defaults or saved experiment results.
'''),
        cell('code', bootstrap),
        cell('markdown', '''## 1. Load and verify saved inputs

If the input ZIP is already in `/content`, it is reused. Otherwise this cell opens Colab's upload picker. Select **qwen_planner_diagnostic_inputs.zip**. Every image and prompt is checked against the original request, and the entire archive has a fixed checksum.
'''),
        cell('code', '''from runner import extract_inputs, prepared_request, inspect_response, load
INPUT_ZIP = Path('/content/qwen_planner_diagnostic_inputs.zip')
if not INPUT_ZIP.is_file():
    from google.colab import files
    uploaded = files.upload()
    if INPUT_ZIP.name in uploaded:
        INPUT_ZIP.write_bytes(uploaded[INPUT_ZIP.name])
    assert INPUT_ZIP.is_file(), 'Upload qwen_planner_diagnostic_inputs.zip.'
    del uploaded
INPUTS = extract_inputs(INPUT_ZIP, DIAG / ('inputs_' + uuid.uuid4().hex[:8]), INPUT_BUNDLE_SHA256)
case_entries = load(INPUTS / 'manifest.json')['cases']
for entry in case_entries:
    case = INPUTS / entry['id']
    prepared_request(case, 'sample_8192', 0)
    baseline = inspect_response(load(case / 'baseline_response.json'), load(case / 'context.json'))
    print(entry['id'], '| archived baseline:', baseline['status'], '| output tokens:', baseline['output_tokens'])
'''),
        cell('markdown', '''## 2. Check the existing inference environment and pinned planner

This uses the environment from notebook 02 without reinstalling packages. Cached weights are reused; missing weights are downloaded at the original locked revision. There is no simulator import. Model GPU loading happens in the next section.
'''),
        cell('code', '''assert INFERENCE_PYTHON.is_file(), 'Prepare the isbench-guard-vl environment using notebook 02 first.'
run_command(INFERENCE_PYTHON, '-c', "from importlib.metadata import version; import torch; "
    "from transformers import Qwen3VLForConditionalGeneration; "
    "assert version('transformers') == '4.57.6'; assert torch.cuda.is_available(); "
    "assert torch.cuda.is_bf16_supported(); print('GPU:', torch.cuda.get_device_name(0)); "
    "print('Free GPU GiB:', torch.cuda.mem_get_info()[0]/1024**3)")
metadata_file = DIAG / 'planner_metadata.json'
run_command(INFERENCE_PYTHON, DIAG / 'scripts/prepare_models.py', '--lock', DIAG / 'experiments/models.json',
            '--roles', 'planner', '--metadata', metadata_file)
models = load(metadata_file)
'''),
        cell('markdown', '''## 3. Select one case and create a fresh diagnostic batch

Start with **A's sink case**. Available case IDs were printed above. Creating another batch makes a new directory and preserves earlier results. The batch snapshots its code and all input evidence.
'''),
        cell('code', '''from runner import DiagnosticBatch
CASE_ID = 'A/clean_a_kitchen_sink__with_blender'
SEED = 0
assert CASE_ID in [entry['id'] for entry in case_entries]
batch = DiagnosticBatch(DIAG, INPUTS, INFERENCE_PYTHON, models)
print('RESULT_DIRECTORY:', batch.output)
'''),
        cell('markdown', '''## 4. First test: sampling with an 8192-token cap

Run this cell first and inspect its result before continuing. The worker samples with Qwen's recommended VL settings and exits afterward. There are no automatic retries. An 8192-token response can take about 14 minutes at the roughly 10 tokens/second seen in your old runs; it may finish earlier. The worker limit is 40 minutes. Memory and numerical errors are retained and stop this batch.

`valid_final_action` means a completed reasoning section plus a final JSON action with a valid primitive and object IDs. It does not establish that the action is correct or safe. A truncated answer is never salvaged from the reasoning text.
'''),
        cell('code', "first_result = batch.run(CASE_ID, 'sample_8192', SEED)\n"),
        cell('markdown', '''## 5. Controls: isolate output budget from sampling

Run this cell after the first result has been inspected. **Greedy/8192** changes the budget alone relative to the archived baseline. **Sampling/2048** changes sampling alone. **Sampling/8192** changes both. Each new variant gets exactly one attempt, with the same seed, prompt and observations. The archived baseline is displayed rather than regenerated.
'''),
        cell('code', '''control_results = []
for variant in ('greedy_8192', 'sample_2048'):
    row = batch.run(CASE_ID, variant, SEED)
    control_results.append(row)
    if batch.failed:
        break
'''),
        cell('markdown', '''## 6. Optional: test another seed or the other saved cases

Leave both options false for the initial diagnostic. A different seed tests whether the sampled result is sensitive to randomness. Repeating the same seed is not an independent draw. The other-case option adds all three variants for each remaining A/B task; this can take hours. One successful sample is not a reliability estimate.
'''),
        cell('code', '''RUN_SECOND_SEED = False
RUN_OTHER_CASES = False
if RUN_SECOND_SEED and not batch.failed:
    batch.run(CASE_ID, 'sample_8192', seed=1)
if RUN_OTHER_CASES and not batch.failed:
    for entry in case_entries:
        if entry['id'] == CASE_ID:
            continue
        for variant in ('sample_8192', 'greedy_8192', 'sample_2048'):
            batch.run(entry['id'], variant, SEED)
            if batch.failed:
                break
        if batch.failed:
            break
'''),
        cell('markdown', '''## 7. Compare format completion and inspect final actions

Read the raw output for each selected result. A valid action format is only a prerequisite to running the benchmark. Completion here uses saved images without a resident simulator; GPU memory results therefore cannot establish that the longer setting fits during a full episode.
'''),
        cell('code', '''from IPython.display import display, Markdown
rows = []
for entry in case_entries:
    if entry['id'] not in {CASE_ID, *[r['case'] for r in batch.rows]}:
        continue
    case = INPUTS / entry['id']
    baseline = inspect_response(load(case / 'baseline_response.json'), load(case / 'context.json'))
    rows.append({'case': entry['id'], 'variant': 'archived_greedy_2048', 'seed': 0, **baseline})
rows.extend(batch.rows)
lines = ['| Case | Variant | Seed | Status | Output tokens | Seconds | Peak GPU GiB |',
         '| --- | --- | --- | --- | --- | --- | --- |']
for r in rows:
    lines.append(f"| {r['case']} | {r['variant']} | {r['seed']} | {r['status']} | {r['output_tokens']} | {round(r['generation_seconds'] or 0, 1)} | {r['peak_allocated_gib']:.2f} |")
display(Markdown('\\n'.join(lines)))
for r in batch.rows:
    print(r['case'], r['variant'], 'seed', r['seed'], '| parsed plan:', r['parsed_plan'])
    print('Raw output:', batch.output / 'runs' / r['case'] / (r['variant'] + '_seed' + str(r['seed'])) / 'raw_output.txt')
'''),
        cell('markdown', '''## 8. Export all evidence, including failures

Run even after a failed worker. The ZIP includes saved observations/prompts, original baseline responses, model metadata, code snapshots, exact generation settings, new raw responses and logs, and the diagnostic summary. Preserve it under the new suite's results folder locally.
'''),
        cell('code', '''archive = batch.export()
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    print('Download:', archive)
'''),
        cell('markdown', '''## Interpreting the diagnostic

If greedy/8192 completes, the larger budget helped for that input. If sampling/2048 completes, sampling helped within the original budget. If only sampling/8192 completes, those two controls did not individually resolve the observed failure in this diagnostic. A single draw cannot establish a general causal effect or the best settings. If all variants fail, inspect the outputs before increasing budgets further or switching checkpoints.

The generation cap counts both reasoning and final-answer tokens. Sampling controls next-token selection; it does not directly control reasoning length or guarantee safety. NF4, BF16, SDPA, input image budgets, and the original checkpoint are retained. No model fine-tuning occurs.

After selecting settings, update and rerun the common A–D planner protocol in new batches. This notebook does not update those defaults automatically.

Sources: [Transformers generation parameters](https://huggingface.co/docs/transformers/main_classes/text_generation), [Qwen3-VL Thinking recommended VL settings](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking#generation-hyperparameters).
'''),
    ], directory=ARCHIVE / 'pilot')
    print('Built', NAME, '| input ZIP:', BUNDLE, '| MiB:', round(BUNDLE.stat().st_size/1024**2, 2), '| SHA256:', digest)


if __name__ == '__main__':
    build()
