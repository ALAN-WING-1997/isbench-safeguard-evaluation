"""Replay saved planner requests, retaining exact prompt and image provenance."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile

from settings import VARIANTS, generation_kwargs
from protocol import parse_plan


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_inputs(archive, destination, expected_sha):
    archive, destination = Path(archive), Path(destination)
    if sha(archive) != expected_sha:
        raise ValueError('Input ZIP differs from the saved diagnostic bundle.')
    if destination.exists():
        raise ValueError('Use a fresh input directory; existing data is not overwritten.')
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            relative = Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in item.filename:
                raise ValueError('Invalid archive path.')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Input archive contains a symbolic link.')
        z.extractall(destination)
    return destination


def prepared_request(case, variant, seed):
    """Change only generation settings and relocated paths, never prompt/image content."""
    case = Path(case)
    request = load(case / 'request.json')
    if request['role'] != 'planner' or len(request['jobs']) != 1 or len(request['jobs'][0]['images']) != 5:
        raise ValueError('Expected the original five-view planner request.')
    original = request['jobs'][0]
    if hashlib.sha256(original['prompt'].encode()).hexdigest() != request['prompt_sha256'][0]:
        raise ValueError('Prompt content changed.')
    mapped, hashes = [], {}
    for i, old in enumerate(original['images']):
        image = case / 'images' / f'obs_{i}.png'
        digest = sha(image)
        if digest != request['image_sha256'][old]:
            raise ValueError('Saved image differs from the original request: ' + str(image))
        mapped.append(str(image.resolve()))
        hashes[str(image.resolve())] = digest
    variant_config = VARIANTS[variant]
    request['jobs'][0]['images'] = mapped
    request['image_sha256'] = hashes
    request['settings'] = {**request['settings'], 'seed': seed, 'max_new_tokens': variant_config['max_new_tokens']}
    request['generation'] = {key: value for key, value in variant_config.items() if key != 'max_new_tokens'}
    request['diagnostic'] = {'variant': variant, 'source': load(case / 'source.json'), 'simulation_started': False}
    generation_kwargs(request)
    return request


def inspect_response(response, context, returncode=0, timed_out=False, interrupted=False):
    """Keep format completion separate from correctness and safe execution."""
    outputs = response.get('outputs', [])
    output = outputs[0] if len(outputs) == 1 else {}
    raw = output.get('raw_output', '')
    ended_thinking = '</think>' in raw or '</thinking>' in raw
    truncated = output.get('hit_token_limit')
    plan = None
    status = 'complete_format_failure'
    if response.get('error') or returncode != 0 or timed_out or interrupted or len(outputs) != 1:
        status = 'inference_failure'
    elif truncated:
        status = 'output_truncated'
    elif not ended_thinking:
        status = 'unfinished_reasoning'
    else:
        plan = parse_plan(raw, context['allowed_primitives'], context['objects'])
        if plan is not None:
            status = 'valid_final_action'
    return {'status': status, 'valid_final_action': status == 'valid_final_action', 'parsed_plan': plan,
            'thinking_finished': ended_thinking, 'hit_token_limit': truncated,
            'input_tokens': output.get('input_tokens'), 'output_tokens': output.get('output_tokens'),
            'generation_seconds': output.get('generation_seconds'),
            'peak_allocated_gib': response.get('peak_allocated_bytes', 0) / 1024**3,
            'returncode': returncode, 'timed_out': timed_out, 'interrupted': interrupted,
            'error': response.get('error'), 'raw_output_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'action_executed': False, 'correctness_or_safety_evaluated': False}


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


class DiagnosticBatch:
    def __init__(self, root, inputs, inference_python, models):
        self.root, self.inputs, self.inference_python = map(Path, (root, inputs, inference_python))
        self.models = models
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
        self.output = self.root / 'results' / run_id
        self.output.mkdir(parents=True, exist_ok=False)
        self.code = self.output / 'code'
        for folder in ('scripts', 'diagnostics'):
            shutil.copytree(self.root / folder, self.code / folder, ignore=shutil.ignore_patterns('__pycache__'))
        self.saved_inputs = self.output / 'inputs'
        shutil.copytree(self.inputs, self.saved_inputs)
        self.rows, self.failed = [], False
        save(self.output / 'manifest.json', {'kind': 'saved-observation inference-only diagnostic',
            'models': models, 'variants': VARIANTS, 'input_bundle': load(self.inputs / 'manifest.json'),
            'code_sha256': {str(p.relative_to(self.code)): sha(p) for p in sorted(self.code.rglob('*.py'))},
            'simulation_started': False, 'automatic_retries': 0})
        self.save_summary()

    def save_summary(self):
        save(self.output / 'diagnostic_summary.json', {'results': self.rows, 'infrastructure_failed': self.failed,
            'note': 'Format completion only; no task success or safety measurement. Saved baseline outcomes are not rerun.'})

    def run(self, case_id, variant='sample_8192', seed=0, timeout_seconds=2400):
        if self.failed:
            raise RuntimeError('Export the failed diagnostic before starting a new batch.')
        known = [entry['id'] for entry in load(self.inputs / 'manifest.json')['cases']]
        if case_id not in known or variant not in VARIANTS or type(seed) is not int or seed < 0:
            raise ValueError('Unknown case/variant or invalid seed.')
        folder = self.output / 'runs' / case_id / (variant + '_seed' + str(seed))
        folder.mkdir(parents=True, exist_ok=False)
        request = prepared_request(self.saved_inputs / case_id, variant, seed)
        planner = self.models['planner']
        if any(request['model'][key] != planner[key] for key in ('repo_id', 'revision')):
            raise ValueError('Planner differs from the saved run.')
        request_file, response_file = folder / 'request.json', folder / 'response.json'
        save(request_file, request)
        env = os.environ.copy()
        env['OMNIGIBSON_NO_OMNIVERSE'] = '1'
        env['PYTHONPATH'] = str(self.code / 'scripts')
        env['PYTHONUNBUFFERED'] = '1'
        command = [str(self.inference_python), str(self.code / 'diagnostics/worker.py'),
            '--request', str(request_file), '--response', str(response_file), '--model-path', planner['model_path']]
        print('Starting', case_id, variant, 'seed', seed, '| log:', folder / 'worker.log', flush=True)
        started = time.monotonic()
        timed_out = interrupted = False
        with (folder / 'worker.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            next_update = started + 30
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if now - started > timeout_seconds:
                        timed_out = True
                        break
                    if now >= next_update:
                        print('Generating | elapsed:', round(now - started), 's', flush=True)
                        next_update = now + 30
                    time.sleep(3)
            except KeyboardInterrupt:
                interrupted = True
            finally:
                stop(process)
            returncode = process.wait()
        response = load(response_file) if response_file.is_file() else {}
        context = load(self.saved_inputs / case_id / 'context.json')
        row = inspect_response(response, context, returncode, timed_out, interrupted)
        row.update(case=case_id, variant=variant, seed=seed, elapsed_seconds=time.monotonic()-started)
        save(folder / 'summary.json', row)
        if response.get('outputs'):
            (folder / 'raw_output.txt').write_text(response['outputs'][0]['raw_output'])
        self.rows.append(row)
        self.failed = row['status'] == 'inference_failure'
        self.save_summary()
        print(json.dumps(row, indent=2), flush=True)
        if self.failed:
            print('Export partial evidence. Inspect worker.log and response.json before another diagnostic.', flush=True)
        return row

    def export(self):
        self.save_summary()
        # Four cases fit in a modest archive; saved images and every raw response are included.
        archive = shutil.make_archive(str(self.output), 'zip', root_dir=self.output.parent, base_dir=self.output.name)
        print('DIAGNOSTIC_ARCHIVE:', archive, flush=True)
        return archive
