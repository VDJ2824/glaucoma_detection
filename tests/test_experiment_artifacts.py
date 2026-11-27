"""Read-only checks of supplied result records; no TensorFlow or model fitting."""
import json
import runpy
import shutil
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
API=runpy.run_path(str(ROOT/'scripts/verify_experiment_artifacts.py'))


def test_archived_artifacts_histories_and_selections_agree():
    result=API['verify_bundle'](ROOT)
    assert result['files_verified']==24
    assert result['stages_verified']==6
    assert result['epoch_rows_verified']==84
    assert len(result['learning_rate_timing_differences'])==8
    assert result['floating_point_range_notes']==[{'model':'resnet50','stage':'stage2','epoch':16,'metric':'auc','value':1.0000001192092896}]
    assert all(bundle['selection']['stage']=='stage2' for bundle in result['models'].values())


def test_comparison_is_recomputable_from_preserved_sources():
    actual=API['comparison_data'](API['verify_bundle'](ROOT))
    stored=json.loads((ROOT/'docs/experiments/core_models_comparison_2026-11-27.json').read_text())
    assert actual==stored
    assert not stored['test_results_provided']
    mobile=next(row for row in stored['fine_tuning_deltas'] if row['model']=='mobilenet_v2')
    assert mobile['validation_recall_delta']<0 # do not claim every fine-tuning metric improved
    assert all(row['validation_loss_decrease']>0 for row in stored['fine_tuning_deltas'])


def test_artifact_tampering_is_rejected(tmp_path):
    copied=tmp_path/'docs/experiments'
    shutil.copytree(ROOT/'docs/experiments',copied)
    altered=copied/'resnet50_kaggle_2026-11-27/histories/stage2/epochs.csv'
    altered.write_bytes(altered.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='byte/hash mismatch'):
        API['verify_bundle'](tmp_path)


def test_probability_metrics_above_tolerance_are_rejected(tmp_path):
    source=ROOT/'docs/experiments/resnet50_kaggle_2026-11-27'
    shutil.copytree(source,tmp_path/'model')
    path=tmp_path/'model/histories/stage2/history.json'
    history=json.loads(path.read_text());history['auc'][0]=1.1
    path.write_text(json.dumps(history))
    with pytest.raises(ValueError,match='outside numerical tolerance'):
        API['verify_stage'](tmp_path/'model','resnet50','stage2')
