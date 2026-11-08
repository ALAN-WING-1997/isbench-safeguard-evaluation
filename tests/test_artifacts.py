"""Check paired-run validation and preservation of exported trial evidence."""
import hashlib
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch, Mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_trials import compare
from export_isbench_results import export_results


class ArtifactTests(unittest.TestCase):
    def d_notebook_source(self, prefix):
        notebook = json.loads((ROOT / 'notebooks/archive/sampling_8192/13_D_embguard_trial.ipynb').read_text())
        return next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and ''.join(c['source']).startswith(prefix))

    def test_d_failure_exports_evidence_and_skips_second_case(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'D_trial'
            run.mkdir()
            (run / 'batch_summary.json').write_text('{"condition":"D","results":[]}')
            (run / 'episode.log').write_text('TypeError: string indices must be integers')
            batch = SimpleNamespace(output=run, can_continue=False,
                run_case=Mock(return_value={'valid_evaluation':False}), save_summary=Mock())
            ns = {'batch': batch, 'json': json}
            for prefix in ('first_result =', 'if batch.can_continue:',
                           'from export_isbench_results import export_results\nbatch.save_summary()'):
                exec(self.d_notebook_source(prefix), ns)
            batch.run_case.assert_called_once_with(0)
            self.assertIsNone(ns['second_result'])
            self.assertTrue(Path(ns['partial_exports']['summary_zip']).is_file())
            with zipfile.ZipFile(ns['exports']['full_zip']) as archive:
                self.assertIn('TypeError', archive.read('D_trial/episode.log').decode())

    def test_d_recovery_exports_old_batch_without_kernel_batch_object(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'old_D'
            run.mkdir()
            (run / 'batch_summary.json').write_text('{"condition":"D","results":[]}')
            (run / 'worker_error.json').write_text('{"error":"processor TypeError"}')
            tree = ast.parse(self.d_notebook_source('from export_isbench_results import export_results\nPREVIOUS_D_RUN'))
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREVIOUS_D_RUN' for t in node.targets):
                    node.value = ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(run))],keywords=[])
            ast.fix_missing_locations(tree)
            colab = ModuleType('google.colab')
            colab.files = SimpleNamespace(download=Mock(side_effect=RuntimeError('No browser download widget')))
            ns = {'Path': Path, 'json': json}
            with patch.dict(sys.modules, {'google.colab':colab}):
                exec(compile(tree,'recovery_cell','exec'), ns)
            self.assertNotIn('batch',ns)
            self.assertTrue(Path(ns['recovered_exports']['summary_zip']).is_file())
            with zipfile.ZipFile(ns['recovered_exports']['full_zip']) as archive:
                self.assertIn('old_D/worker_error.json',archive.namelist())

    def test_comparison_rejects_different_generation_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / c for c in 'AB']
            runtime = json.loads((ROOT / 'experiments/runtime.json').read_text())
            for path in paths:
                path.mkdir()
                manifest = {'condition': path.name, 'runtime': runtime,
                    'models': {'planner': {'repo_id': 'same', 'revision': 'same'}}, 'tasks': [], 'script_sha256': {}}
                (path / 'batch_manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(compare(paths)['conditions'], ['A', 'B'])
            other = json.loads((paths[1] / 'batch_manifest.json').read_text())
            other['runtime']['inference']['max_new_tokens'] += 1
            (paths[1] / 'batch_manifest.json').write_text(json.dumps(other))
            with self.assertRaisesRegex(ValueError, 'budgets'):
                compare(paths)

    def test_comparison_rejects_different_sampling_or_format_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / c for c in 'AB']
            runtime = json.loads((ROOT / 'experiments/runtime.json').read_text())
            for path in paths:
                path.mkdir()
                manifest = {'condition': path.name, 'runtime': runtime,
                    'models': {'planner': {'repo_id': 'same', 'revision': 'same'}}, 'tasks': [], 'script_sha256': {}}
                (path / 'batch_manifest.json').write_text(json.dumps(manifest))
            original = json.loads((paths[1] / 'batch_manifest.json').read_text())
            for key, value in (('do_sample', False), ('temperature', .7), ('max_format_attempts', 3)):
                other = json.loads(json.dumps(original))
                if key == 'max_format_attempts':
                    other['runtime'][key] = value
                else:
                    other['runtime']['inference'][key] = value
                (paths[1] / 'batch_manifest.json').write_text(json.dumps(other))
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    compare(paths)

    def test_split_export_preserves_code_and_raw_guard_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'trial'
            (run / 'code').mkdir(parents=True)
            (run / 'batch_summary.json').write_text('{"results": []}')
            (run / 'code/protocol.py').write_bytes((ROOT / 'scripts/protocol.py').read_bytes())
            raw = '{"outputs": [{"raw_output": "unsafe"}]}'
            (run / '001_embguard_response.json').write_text(raw)
            exported = export_results(run, part_size_bytes=200)
            restored = b''.join(Path(p).read_bytes() for p in exported['parts'])
            manifest = json.loads(Path(exported['manifest']).read_text())
            self.assertEqual(hashlib.sha256(restored).hexdigest(), manifest['archive_sha256'])
            self.assertEqual(restored, Path(exported['full_zip']).read_bytes())
            with zipfile.ZipFile(io.BytesIO(restored)) as archive:
                self.assertEqual(archive.read('trial/001_embguard_response.json').decode(), raw)
            with zipfile.ZipFile(exported['summary_zip']) as archive:
                self.assertIn('trial/code/protocol.py', archive.namelist())


if __name__ == '__main__':
    unittest.main()
