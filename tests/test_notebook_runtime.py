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


@pytest.fixture
def portable_bundle(tmp_path, image_factory):
    import pandas as pd
    from glaucoma.config import load_config
    cfg = load_config(CONFIG)
    input_root = tmp_path / 'attached'
    paths = [image_factory('attached/database/Im001_ACRIMA.jpg'),
             image_factory('attached/database/Im002_g_ACRIMA.jpg', seed=2),
             image_factory('attached/database/Im003_ACRIMA.jpg', seed=3)]
    frame = pd.DataFrame({'image_id': ['001','002','003'],
        'image_path': [f'/content/drive/old/database/{p.name}' for p in paths],
        'dataset_name': ['acrima']*3, 'patient_id': ['','',''], 'label': ['0','1','0'],
        'split': ['train','validation','test'], 'source_key': ['0001','0002','0003']})
    source_csv = input_root/'acrima_metadata.csv'; frame.to_csv(source_csv,index=False)
    session = input_root/'session.json'; session.write_text(json.dumps(cfg))
    return cfg, frame, session, source_csv, input_root/'database'


def test_relocate_paths_preserves_frozen_rows_and_methodology(portable_bundle,tmp_path):
    import pandas as pd
    from glaucoma.utils.notebook_runtime import relocate_frozen_session
    cfg,frame,session,csv,images=portable_bundle
    before={p:p.read_bytes() for p in session.parent.rglob('*') if p.is_file()}
    result,derived=relocate_frozen_session(session,csv,images,tmp_path/'working/artifacts')
    relocated=pd.read_csv(result['metadata']['output'],dtype=str,keep_default_na=False)
    pd.testing.assert_frame_equal(frame.drop(columns='image_path'),relocated.drop(columns='image_path'))
    assert all(Path(value).parent==images for value in relocated.image_path)
    for key in ('model','training','preprocessing','augmentation','split','seed'):
        assert result[key]==cfg[key]
    assert load_session(derived)==result
    assert all(p.read_bytes()==contents for p,contents in before.items())
    assert set(p for p in session.parent.rglob('*') if p.is_file())==set(before)
    previous_times={p:p.stat().st_mtime_ns for p in (derived,Path(result['metadata']['output']))}
    again,again_path=relocate_frozen_session(session,csv,images,tmp_path/'working/artifacts')
    assert again==result and again_path==derived
    assert previous_times=={p:p.stat().st_mtime_ns for p in previous_times}


def test_relocation_rejects_basename_collisions(portable_bundle,tmp_path):
    from glaucoma.utils.notebook_runtime import relocate_frozen_session
    _,frame,session,csv,images=portable_bundle
    frame.loc[1,'image_path']='/another/folder/'+Path(frame.loc[0,'image_path']).name
    frame.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='colliding'):
        relocate_frozen_session(session,csv,images,tmp_path/'working')
    assert not (tmp_path/'working').exists()


def test_relocation_fails_on_missing_image_without_outputs(portable_bundle,tmp_path):
    from glaucoma.utils.notebook_runtime import relocate_frozen_session
    _,frame,session,csv,images=portable_bundle
    frame.loc[0,'image_path']='/old/missing.jpg'; frame.to_csv(csv,index=False)
    with pytest.raises(ValueError,match='missing'):
        relocate_frozen_session(session,csv,images,tmp_path/'working')
    assert not (tmp_path/'working').exists()


def test_relocation_refuses_changed_existing_manifest(portable_bundle,tmp_path):
    import pandas as pd
    from glaucoma.utils.notebook_runtime import relocate_frozen_session
    _,_,session,csv,images=portable_bundle
    cfg,_=relocate_frozen_session(session,csv,images,tmp_path/'working')
    output=Path(cfg['metadata']['output'])
    frame=pd.read_csv(output,dtype=str,keep_default_na=False); frame.loc[0,'split']='test';frame.to_csv(output,index=False)
    before=output.read_bytes()
    with pytest.raises(FileExistsError,match='manifest differs'):
        relocate_frozen_session(session,csv,images,tmp_path/'working')
    assert output.read_bytes()==before


def test_relocation_never_overwrites_input_session(portable_bundle,tmp_path):
    from glaucoma.utils.notebook_runtime import relocate_frozen_session
    _,_,session,csv,images=portable_bundle
    root=tmp_path/'working/runs/development/environment';root.mkdir(parents=True)
    protected=root/'session.json';protected.write_bytes(session.read_bytes())
    before=protected.read_bytes()
    with pytest.raises(ValueError,match='original'):
        relocate_frozen_session(protected,csv,images,tmp_path/'working')
    assert protected.read_bytes()==before
