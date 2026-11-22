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
    # Colab saves some cell IDs in metadata instead of the nbformat cell field.
    ids = [cell.get('id', cell['metadata'].get('id')) for cell in notebook['cells']]
    assert all(ids) and len(set(ids)) == len(notebook['cells'])
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            if name not in NAMES[:3]:
                assert cell['outputs'] == [] and cell['execution_count'] is None
            else:
                # The user's executed Colab setup includes a standalone shell diagnostic.
                # Leave its recorded outputs and IPython cells intact.
                source = '\n'.join((line[:len(line)-len(line.lstrip())] + 'pass  # IPython command') if line.lstrip().startswith(('!', '%')) else line for line in source.splitlines())
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('glaucoma'):
                    module = importlib.import_module(node.module)
                    for symbol in node.names:
                        if not hasattr(module, symbol.name):
                            importlib.import_module(f'{node.module}.{symbol.name}')


@pytest.mark.parametrize('name', NAMES[3:])
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



def test_training_notebook_defaults_do_not_construct_or_fit(tmp_path, monkeypatch):
    from glaucoma.models import factory
    from glaucoma.training import trainer
    def forbidden(*args, **kwargs):
        raise AssertionError('Default notebook must not construct models or fit')
    monkeypatch.setattr(factory, 'build_model', forbidden)
    monkeypatch.setattr(trainer, 'train_stage', forbidden)
    cfg=configure_workspace(ROOT/'configs/project.yaml',tmp_path/'missing-data',tmp_path/'drive')
    session=save_session(cfg,tmp_path/'drive/session.json')
    monkeypatch.setenv('GLAUCOMA_SESSION',str(session))
    notebook=json.loads((ROOT/'notebooks/03_individual_model_training.ipynb').read_text())
    namespace={}
    before=set(tmp_path.rglob('*'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            exec(compile(''.join(cell['source']),'03_individual_model_training','exec'),namespace)
    assert set(tmp_path.rglob('*'))==before
    assert namespace['model'] is None and not namespace['RUN_STAGE1'] and not namespace['RUN_STAGE2']
