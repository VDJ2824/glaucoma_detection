"""Check notebook structure/imports and run scaffold cells without external effects."""
import ast
import importlib
import json
from pathlib import Path
import pytest
from glaucoma.utils.notebook_runtime import configure_workspace, save_session

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['00_environment_setup','01_dataset_exploration','02_dataset_preparation',
         '03_individual_model_training','04_model_evaluation','05_ensemble_optimization',
         '06_gradcam_analysis','07_cross_dataset_validation','08_tensorflowjs_conversion']


@pytest.mark.parametrize('name', NAMES)
def test_notebook_schema_and_python_syntax(name):
    notebook = json.loads((ROOT/'notebooks'/f'{name}.ipynb').read_text())
    assert (notebook['nbformat'], notebook['nbformat_minor']) == (4, 5)
    assert notebook['metadata']['kernelspec']['name'] == 'python3'
    assert len({cell['id'] for cell in notebook['cells']}) == len(notebook['cells'])
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('glaucoma'):
                    module = importlib.import_module(node.module)
                    for symbol in node.names:
                        if not hasattr(module, symbol.name):
                            importlib.import_module(f'{node.module}.{symbol.name}')


@pytest.mark.parametrize('name', NAMES[1:])
def test_scaffold_run_all_has_no_processing_side_effects(name, tmp_path, monkeypatch):
    cfg = configure_workspace(ROOT/'configs/project.yaml', tmp_path/'missing-dataset', tmp_path/'drive')
    session = save_session(cfg, tmp_path/'drive/session.json')
    monkeypatch.setenv('GLAUCOMA_SESSION', str(session))
    before = {p for p in tmp_path.rglob('*')}
    notebook = json.loads((ROOT/'notebooks'/f'{name}.ipynb').read_text())
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), name, 'exec'), namespace)
    assert {p for p in tmp_path.rglob('*')} == before
    assert not (tmp_path/'missing-dataset').exists()
