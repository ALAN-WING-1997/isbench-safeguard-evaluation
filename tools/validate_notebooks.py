"""Validate notebook code and recorded helper snapshots."""
import ast
import hashlib
import json

from build_c_improved_notebook import NAME as IMPROVED_NAME
from build_diagnostic_notebook import NAME as DIAGNOSTIC_NAME, diagnostic_sources
from build_evaluation_notebooks import DESTINATION, NAMES as EVALUATION_NAMES
from build_notebooks import ARCHIVE, ROOT


HOST_HELPERS = {
    'ACCOUNTING_SOURCE': 'evaluation_accounting.py',
    'SESSION_DRIVER_SOURCE': 'evaluation_session.py',
}


def validate():
    expected = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / 'scripts').rglob('*.py'))}
    expected.update({str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / 'experiments').glob('*.json'))})
    host_helpers = {name: (ROOT / 'tools' / filename).read_text() for name, filename in HOST_HELPERS.items()}
    preserved = json.loads((ROOT / 'tools/preserved_notebook_bundles.json').read_text())
    notebooks = sorted(DESTINATION.rglob('*.ipynb'))
    assert preserved.keys() <= {str(path.relative_to(ROOT)) for path in notebooks}, 'Missing preserved notebooks.'
    assert {path.name for path in DESTINATION.glob('*.ipynb')} == EVALUATION_NAMES | {IMPROVED_NAME}, 'Incomplete evaluation notebook set.'
    count = code_count = embedded = 0
    for path in sorted(ROOT.rglob('*.py')):
        compile(path.read_text(), str(path), 'exec')
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert notebook['nbformat'] == 4 and notebook['nbformat_minor'] == 5
        ids = [c['id'] for c in notebook['cells']]
        assert len(set(ids)) == len(ids), f'Duplicate cell IDs: {path}'
        pins = preserved.get(str(path.relative_to(ROOT)), {})
        seen_helpers = {}
        expected_helpers = {'SOURCES': diagnostic_sources() if path.name == DIAGNOSTIC_NAME else expected,
                            **host_helpers}
        for i, cell in enumerate(notebook['cells']):
            assert cell['cell_type'] in ('markdown', 'code')
            source = ''.join(cell['source'])
            if cell['cell_type'] != 'code':
                continue
            # Executed notebook outputs are user evidence, including large runs.
            assert cell['execution_count'] is None or type(cell['execution_count']) is int
            assert isinstance(cell['outputs'], list)
            tree = ast.parse(source, filename=f'{path.name}:cell{i}')
            compile(tree, f'{path.name}:cell{i}', 'exec')
            code_count += 1
            for statement in tree.body:
                targets = (statement.targets if isinstance(statement, ast.Assign) else
                           [statement.target] if isinstance(statement, ast.AnnAssign) else [])
                for target in targets:
                    if not isinstance(target, ast.Name) or target.id not in expected_helpers:
                        continue
                    symbol = target.id
                    source = ast.literal_eval(statement.value)
                    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
                    if path.is_relative_to(ARCHIVE) and symbol in pins:
                        assert digest == pins[symbol], f'Archived helper {symbol} changed: {path}'
                    else:
                        assert source == expected_helpers[symbol] or digest == pins.get(symbol), (
                            f'Unrecognized embedded helper {symbol}: {path}')
                    if symbol in HOST_HELPERS:
                        compile(source, HOST_HELPERS[symbol], 'exec')
                    assert seen_helpers.setdefault(symbol, digest) == digest, f'Mixed {symbol} snapshots: {path}'
        assert pins.keys() <= seen_helpers.keys(), f'Missing recorded helpers: {path}'
        if path.name not in ('00_simulator_setup.ipynb', '01_scene_data.ipynb', 'air_filter_partial_views.ipynb'):
            assert 'SOURCES' in seen_helpers, f'Missing standalone helpers: {path}'
            embedded += 1
        count += 1
    print(f'Validated {count} notebooks, {code_count} Python cells, and {embedded} complete embedded helper bundles.')


if __name__ == '__main__':
    validate()
