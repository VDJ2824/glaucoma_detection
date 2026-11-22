from pathlib import Path
import pytest
import yaml
from glaucoma.config import load_config


def test_defaults(cfg):
    assert cfg['seed'] == 42
    assert cfg['model']['input_shape'] == [224, 224, 3]
    assert cfg['training']['batch_size'] == 16
    assert cfg['training']['stage2']['epochs'] == 20
    assert cfg['training']['stage2']['learning_rate'] == 0.0001
    assert cfg['split']['ratios'] == {'train': .7, 'validation': .15, 'test': .15}
    assert Path(cfg['datasets']['acrima']['root']).is_absolute()


def test_environment_and_cwd_independence(cfg, tmp_path, monkeypatch):
    path = Path('configs/project.yaml').resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DATASET_ROOT', str(tmp_path/'download'))
    monkeypatch.setenv('ARTIFACTS_ROOT', str(tmp_path/'outputs'))
    other = load_config(path)
    assert other['datasets']['acrima']['root'] == str(tmp_path/'download')
    assert other['checkpoints']['core'] == str(tmp_path/'outputs/checkpoints/core')
    assert other['metadata'] == cfg['metadata']


@pytest.mark.parametrize('section,key,value', [('model','selected','unknown'), ('model','input_shape',[20,20,3]), ('split','mode','random'), ('split','ratios',{'train':.8,'validation':.2,'test':.2})])
def test_reject_invalid_config(tmp_path, section, key, value):
    doc = yaml.safe_load(Path('configs/project.yaml').read_text())
    doc[section][key] = value
    p = tmp_path/'bad.yaml'; p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):
        load_config(p)


@pytest.mark.parametrize('section,key,value', [
    ('training','batch_size',0),
    ('training','early_stopping',{'mode':'min','patience':-1,'min_delta':.001}),
    ('model','head',{'pooling':'global_average','dropout':1,'units':1,'activation':'sigmoid'}),
    ('ensemble','members',['vgg16']),
    ('export','members',['vgg19']),
    ('augmentation','brightness_delta',-1),
])
def test_reject_invalid_training_and_scope(tmp_path, section, key, value):
    doc = yaml.safe_load(Path('configs/project.yaml').read_text())
    doc[section][key] = value
    p = tmp_path/'bad.yaml'; p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError): load_config(p)


def test_invalid_dataset_label_mode_rejected(tmp_path):
    doc=yaml.safe_load(Path('configs/project.yaml').read_text())
    doc['datasets']['acrima']['label_mode']='guess'
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError,match='label mode'):load_config(path)
