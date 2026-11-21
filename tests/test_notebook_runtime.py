"""Notebook persistence contracts, tested without hosted services or TensorFlow."""
import json
from pathlib import Path
import pytest
from glaucoma.utils.notebook_runtime import configure_workspace, save_session, load_session

CONFIG = Path(__file__).resolve().parents[1] / 'configs/project.yaml'


def test_all_outputs_persist_under_requested_root(tmp_path):
    original = CONFIG.read_bytes()
    dataset = tmp_path / 'readonly-dataset'
    dataset.mkdir()
    marker = dataset / 'original.txt'
    marker.write_text('untouched')
    cfg = configure_workspace(CONFIG, dataset, tmp_path/'drive', 'mobilenet-trial-01')
    run = tmp_path/'drive/runs/mobilenet-trial-01'
    assert Path(cfg['artifacts']['root']) == run
    assert cfg['export']['output_root'] == str(run/'exports')
    for name in ('metrics','histories','heatmaps','exports','manifests','environment'):
        assert Path(cfg['artifacts'][name]).is_dir()
        assert Path(cfg['artifacts'][name]).is_relative_to(run)
    for value in cfg['checkpoints'].values():
        assert Path(value).is_dir() and Path(value).is_relative_to(run)
    assert Path(cfg['metadata']['output']).parent == run/'manifests'
    assert Path(cfg['metadata']['validation_output']).parent == run/'manifests'
    assert cfg['datasets']['acrima']['root'] == str(dataset)
    assert marker.read_text() == 'untouched' and list(dataset.iterdir()) == [marker]
    assert CONFIG.read_bytes() == original
    session = save_session(cfg, run/'environment/session.json')
    assert load_session(session) == cfg


def test_does_not_create_missing_dataset(tmp_path):
    dataset = tmp_path/'not-downloaded'
    configure_workspace(CONFIG, dataset, tmp_path/'outputs')
    assert not dataset.exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', '..', '', 'a/b'])
def test_invalid_run_names_rejected(tmp_path, name):
    with pytest.raises(ValueError,match='run name'):
        configure_workspace(CONFIG, tmp_path/'data', tmp_path/'outputs', name)
    assert not (tmp_path/'outputs').exists()


def test_outputs_must_not_modify_raw_dataset(tmp_path):
    with pytest.raises(ValueError,match='outside'):
        configure_workspace(CONFIG, tmp_path, tmp_path/'artifacts')
    assert not (tmp_path/'artifacts').exists()


def test_session_cannot_overwrite_dataset(tmp_path):
    cfg = configure_workspace(CONFIG, tmp_path/'data', tmp_path/'outputs')
    with pytest.raises(ValueError,match='outside'):
        save_session(cfg, tmp_path/'data/session.json')


def test_missing_session_error(tmp_path):
    with pytest.raises(FileNotFoundError,match='notebook 00'):
        load_session(tmp_path/'missing.json')


def test_symlinked_artifact_directory_cannot_modify_raw_data(tmp_path):
    dataset = tmp_path/'raw'; dataset.mkdir()
    root = tmp_path/'drive/runs/development'; root.mkdir(parents=True)
    (root/'metrics').symlink_to(dataset, target_is_directory=True)
    with pytest.raises(ValueError,match='inside'):
        configure_workspace(CONFIG, dataset, tmp_path/'drive')
    assert list(dataset.iterdir()) == []
