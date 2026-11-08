"""Validate generation plumbing and replay provenance without a GPU."""
import copy
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'diagnostics'), str(ROOT / 'scripts'), str(ROOT / 'tools')]
from settings import generation_kwargs
from runner import DiagnosticBatch, extract_inputs, inspect_response, load, prepared_request
from build_diagnostic_notebook import diagnostic_sources, diagnostic_worker_source, build_worker

BUNDLE = ROOT / 'diagnostics/inputs/qwen_planner_diagnostic_inputs.zip'
BUNDLE_MISSING = 'Saved planner diagnostic input ZIP is not available locally.'


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if BUNDLE.exists():
            cls.temp = tempfile.TemporaryDirectory()
            cls.addClassCleanup(cls.temp.cleanup)
            cls.inputs = extract_inputs(BUNDLE, Path(cls.temp.name) / 'inputs', hashlib.sha256(BUNDLE.read_bytes()).hexdigest())

    @unittest.skipUnless(BUNDLE.exists(), BUNDLE_MISSING)
    def test_relocated_inputs_are_identical_for_all_four_saved_cases(self):
        for case in load(self.inputs / 'manifest.json')['cases']:
            path = self.inputs / case['id']
            original = load(path / 'request.json')
            request = prepared_request(path, 'sample_8192', 0)
            self.assertEqual(request['jobs'][0]['prompt'], original['jobs'][0]['prompt'])
            self.assertEqual(set(request['image_sha256'].values()), set(original['image_sha256'].values()))
            self.assertEqual(request['model'], original['model'])
            self.assertEqual(request['settings']['max_pixels'], original['settings']['max_pixels'])
            self.assertEqual(request['settings']['max_new_tokens'], 8192)
            self.assertEqual(request['generation'], {'do_sample': True, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20})
            context = load(path / 'context.json')
            self.assertEqual(set(context), {'objects', 'allowed_primitives'})

    @unittest.skipUnless(BUNDLE.exists(), BUNDLE_MISSING)
    def test_archived_baselines_are_all_truncated(self):
        for case in load(self.inputs / 'manifest.json')['cases']:
            path = self.inputs / case['id']
            row = inspect_response(load(path / 'baseline_response.json'), load(path / 'context.json'))
            self.assertEqual(row['status'], 'output_truncated')
            self.assertEqual(row['output_tokens'], 2048)
            self.assertFalse(row['valid_final_action'])
            self.assertFalse(row['action_executed'])

    def test_saved_sink_responses_pass_updated_parser_without_generation(self):
        saved = ROOT / 'results/diagnostic/20261008T081504_084394Z_16a700ad'
        if not saved.exists():
            self.skipTest('User diagnostic archive is not available in this checkout.')
        case = 'A/clean_a_kitchen_sink__with_blender'
        context = load(saved / 'inputs' / case / 'context.json')
        for variant in ('sample_8192', 'greedy_8192'):
            row = inspect_response(load(saved / 'runs' / case / (variant + '_seed0') / 'response.json'), context)
            self.assertEqual(row['status'], 'valid_final_action')
            self.assertEqual(row['parsed_plan'], {
                'action': 'soak_inside(sponge.n.01_1, liquid_soap__bottle.n.01_1)', 'caution': None})
            self.assertFalse(row['action_executed'])
        row = inspect_response(load(saved / 'runs' / case / 'sample_2048_seed0/response.json'), context)
        self.assertEqual(row['status'], 'output_truncated')
        self.assertFalse(row['valid_final_action'])

    @unittest.skipUnless(BUNDLE.exists(), BUNDLE_MISSING)
    def test_sampling_settings_reach_generation_call(self):
        path = self.inputs / load(self.inputs / 'manifest.json')['cases'][0]['id']
        sample = generation_kwargs(prepared_request(path, 'sample_8192', 1))
        self.assertTrue(sample['do_sample'])
        self.assertEqual((sample['max_new_tokens'], sample['temperature'], sample['top_p'], sample['top_k']), (8192, 1.0, 0.95, 20))
        greedy = generation_kwargs(prepared_request(path, 'greedy_8192', 0))
        self.assertFalse(greedy['do_sample'])
        self.assertNotIn('temperature', greedy)
        self.assertNotIn('top_p', greedy)
        self.assertNotIn('top_k', greedy)
        worker = (ROOT / 'diagnostics/worker.py').read_text()
        self.assertIn('model.generate(**inputs, **generation_kwargs(request))', worker)

    @unittest.skipUnless(BUNDLE.exists(), BUNDLE_MISSING)
    def test_worker_regeneration_uses_diagnostic_variants_without_overwriting_artifacts(self):
        original_worker = (ROOT / 'diagnostics/worker.py').read_bytes()
        source = diagnostic_worker_source()
        tree = ast.parse(source)
        # Model and GPU imports are inside main; loading these definitions is CPU-only.
        definitions = ast.Module(body=[node for node in tree.body if not isinstance(node, ast.If)],
                                 type_ignores=[])
        namespace = {}
        exec(compile(definitions, '<rendered-diagnostic-worker>', 'exec'), namespace)
        self.assertIs(namespace['generation_kwargs'], generation_kwargs)
        case = self.inputs / load(self.inputs / 'manifest.json')['cases'][0]['id']
        for variant in ('sample_8192', 'greedy_8192', 'sample_2048'):
            request = prepared_request(case, variant, 0)
            self.assertEqual(namespace['generation_kwargs'](request), generation_kwargs(request))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        validated = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                         and ast.unparse(node.value) == 'generation_kwargs(request)')
        first_import = next(i for i, node in enumerate(main.body) if isinstance(node, (ast.Import, ast.ImportFrom)))
        self.assertLess(validated, first_import)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'generate']
        self.assertEqual(len(calls), 1)
        self.assertEqual([(key.arg, ast.unparse(key.value)) for key in calls[0].keywords],
                         [(None, 'inputs'), (None, 'job_decoding')])
        with patch('build_diagnostic_notebook.Path.write_text') as output:
            build_worker()
        output.assert_called_once_with(source)
        self.assertIn('diagnostics/worker.py', diagnostic_sources())
        self.assertEqual((ROOT / 'diagnostics/worker.py').read_bytes(), original_worker)

    @unittest.skipUnless(BUNDLE.exists(), BUNDLE_MISSING)
    def test_invalid_sampling_settings_are_rejected(self):
        path = self.inputs / load(self.inputs / 'manifest.json')['cases'][0]['id']
        original = prepared_request(path, 'sample_8192', 0)
        for key, value in (('temperature', 0), ('temperature', float('nan')), ('top_p', 1.1), ('top_k', 2.5), ('do_sample', 'true')):
            request = copy.deepcopy(original)
            request['generation'][key] = value
            with self.assertRaises(ValueError):
                generation_kwargs(request)

    def test_thinking_must_finish_before_accepting_a_final_action(self):
        context = {'objects': [], 'allowed_primitives': {}}
        raw = '<think>reasoning</think>\n```json\n{"action":"DONE","caution":null}\n```'
        response = {'outputs': [{'raw_output': raw, 'hit_token_limit': False}]}
        self.assertTrue(inspect_response(response, context)['valid_final_action'])
        response['outputs'][0]['raw_output'] = 'Still thinking.\n```json\n{"action":"DONE"}\n```'
        self.assertEqual(inspect_response(response, context)['status'], 'unfinished_reasoning')
        response['outputs'][0]['hit_token_limit'] = True
        self.assertEqual(inspect_response(response, context)['status'], 'output_truncated')
        response['error'] = 'OOM'
        self.assertEqual(inspect_response(response, context)['status'], 'inference_failure')

    @unittest.skipUnless(BUNDLE.exists(), BUNDLE_MISSING)
    def test_input_bundle_integrity_is_required(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            extract_inputs(BUNDLE, Path(self.temp.name) / 'bad', 'incorrect-hash')
        path = self.inputs / load(self.inputs / 'manifest.json')['cases'][0]['id']
        with patch('runner.sha', return_value='incorrect-image-hash'):
            with self.assertRaisesRegex(ValueError, 'image differs'):
                prepared_request(path, 'sample_8192', 0)

    @unittest.skipUnless(BUNDLE.exists(), BUNDLE_MISSING)
    def test_batch_runs_one_worker_and_never_retries_or_executes_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('scripts/protocol.py', 'diagnostics/worker.py'):
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text('# snapshot fixture\n')
            case_id = load(self.inputs / 'manifest.json')['cases'][0]['id']
            original = load(self.inputs / case_id / 'request.json')['model']
            batch = DiagnosticBatch(root, self.inputs, Path('fake-python'), {'planner': {**original, 'model_path': 'fake-weights'}})

            def fake_worker(command, **kw):
                response = Path(command[command.index('--response')+1])
                response.write_text(json.dumps({'outputs': [{'raw_output': '</think>```json\n{"action":"DONE"}\n```',
                    'hit_token_limit': False, 'output_tokens': 30}]}))
                return SimpleNamespace(poll=lambda: 0, wait=lambda: 0)

            with patch('runner.subprocess.Popen', side_effect=fake_worker) as calls:
                row = batch.run(case_id)
            self.assertEqual(calls.call_count, 1)
            self.assertTrue(row['valid_final_action'])
            self.assertFalse(row['action_executed'])
            self.assertEqual(len(load(batch.output / 'diagnostic_summary.json')['results']), 1)
            with self.assertRaises(FileExistsError):
                batch.run(case_id)


if __name__ == '__main__':
    unittest.main()
