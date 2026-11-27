"""Synthetic predictions exercise selection and test isolation without real CNNs."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from glaucoma.ensemble.predictions import MEMBERS, align_predictions, voting_predictions, validate_weights
from glaucoma.ensemble.weights import optimize_weights, search_protocol
from glaucoma.evaluation.metrics import classification_metrics, binary_crossentropy, roc_points
from glaucoma.evaluation.checkpoints import canonical_hash, training_manifest_digest, verify_selected_checkpoints
from glaucoma.evaluation.inference import manifest_partition, predict_model
from glaucoma.evaluation.comparison import disagreement_analysis, save_evaluation
from glaucoma.evaluation.workflow import freeze_validation, load_frozen, run_validation, run_final_test


def aligned(split='validation'):
    return pd.DataFrame({'dataset_name':['acrima']*4,'image_id':['001','002','003','004'],
       'split':[split]*4,'label':[0,1,0,1],
       MEMBERS[0]:[.1,.9,.2,.8],MEMBERS[1]:[.3,.7,.7,.4],MEMBERS[2]:[.8,.7,.1,.8]})


def test_image_key_alignment_accepts_shuffling_and_preserves_ids():
    frame=aligned()
    predictions={name:frame[['dataset_name','image_id','split','label',name]].rename(columns={name:'score'}).sample(frac=1,random_state=i) for i,name in enumerate(MEMBERS)}
    result=align_predictions(predictions,'validation',frame)
    pd.testing.assert_frame_equal(result,frame)
    bad=copy.deepcopy(predictions);bad[MEMBERS[1]].iloc[0,bad[MEMBERS[1]].columns.get_loc('label')]=1-bad[MEMBERS[1]].iloc[0].label
    with pytest.raises(ValueError,match='labels disagree'):align_predictions(bad,'validation',frame)
    bad=copy.deepcopy(predictions);bad[MEMBERS[0]]=bad[MEMBERS[0]].iloc[:-1]
    with pytest.raises(ValueError,match='Missing or extra'):align_predictions(bad,'validation',frame)
    bad=copy.deepcopy(predictions);bad[MEMBERS[0]]=pd.concat([bad[MEMBERS[0]],bad[MEMBERS[0]].iloc[:1]])
    with pytest.raises(ValueError,match='Duplicate'):align_predictions(bad,'validation',frame)


@pytest.mark.parametrize('weights', [[-.1,.6,.5],[0,0,0],[.5,.5,.5],[np.nan,0,1],[1,0],{'vgg16':1}])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):validate_weights(weights)


def test_voting_and_threshold_boundary():
    scores=np.array([[.5,.4,.4],[.5,.5,.1],[.1,.9,.8]])
    results=voting_predictions(scores,[1,0,0])
    assert results['majority_voting']['labels'].tolist()==[0,1,1]
    assert results['majority_voting']['scores'] is None
    assert results['weighted_soft_voting']['labels'].tolist()==[1,1,0]
    np.testing.assert_allclose(results['equal_soft_voting']['scores'],scores.mean(axis=1))


def test_predefined_search_validation_only_and_reproducible():
    frame=aligned()
    weights,candidates=optimize_weights(frame)
    assert len(candidates)==232 and candidates.selected.sum()==1
    assert sum(weights.values())==pytest.approx(1)
    assert all(v>=0 for v in weights.values())
    assert weights[MEMBERS[0]]==1 # may select a single model; no forced ensemble gain
    assert optimize_weights(frame)[0]==weights
    with pytest.raises(ValueError,match='validation'):optimize_weights(aligned('test'))
    mixed=frame.copy();mixed.loc[0,'split']='test'
    with pytest.raises(ValueError):optimize_weights(mixed)
    frame.loc[:,list(MEMBERS)]=np.tile([.4,.4,.4],(4,1))
    tied,_=optimize_weights(frame)
    np.testing.assert_allclose(list(tied.values()),[1/3]*3)


def test_metrics_exact_confusion_undefined_cases_and_score_policy():
    result=classification_metrics([0,0,1,1],[0,1,0,1],[.1,.7,.2,.9])
    assert result['confusion_matrix']==[[1,1],[1,1]]
    for key in ('accuracy','precision','recall','specificity','f1'):assert result[key]==.5
    assert result['roc_auc']==.75
    assert result['binary_crossentropy']==pytest.approx(-np.mean(np.log([.9,.3,.2,.9])))
    hard=classification_metrics([0,1],[0,1])
    assert hard['roc_auc'] is None and hard['binary_crossentropy'] is None
    one=classification_metrics([0,0],[0,0],[.1,.2])
    assert one['roc_auc'] is None and one['recall'] is None and one['precision'] is None
    assert roc_points([0,0],[.1,.2])==[]
    assert roc_points([0,1],[.1,.9])[0]['threshold'] is None
    assert np.isfinite(binary_crossentropy([0,1],[0,1]))
    with pytest.raises(ValueError):classification_metrics([0,1],[0,1],[np.nan,.1])


def test_disagreement_and_actual_plot_artifacts(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    frame=aligned('test')
    rows,summary=disagreement_analysis(frame)
    assert summary['nonunanimous_count']==3
    assert summary['unanimous_count']==1
    assert len(summary['pairwise'])==3 and len(rows)==4
    results=save_evaluation(frame,{name:1/3 for name in MEMBERS},.5,tmp_path)
    assert set(results)=={*MEMBERS,'majority_voting','equal_soft_voting','weighted_soft_voting'}
    for name in ('metrics.json','metrics_comparison.csv','confusion_matrices.png','roc_curves.png','roc_points.csv','prediction_disagreement.csv'):
        assert (tmp_path/name).is_file()
    assert json.loads((tmp_path/'metrics.json').read_text())['split']=='test'


@pytest.fixture
def manifest_case(tmp_path,image_factory,cfg):
    root=tmp_path/'images'
    rows=[]
    for i,split in enumerate(['train','train','validation','validation','test','test']):
        path=image_factory(f'images/{i:03}.png',seed=i)
        rows.append(dict(dataset_name='acrima',image_id=f'{i:03}',image_path=str(path),patient_id='',label=i%2,split=split))
    cfg=copy.deepcopy(cfg)
    cfg['datasets']['acrima']['root']=str(root)
    cfg['metadata']['output']=str(tmp_path/'frozen.csv')
    cfg['artifacts']={'root':str(tmp_path/'artifacts')}
    cfg['checkpoints']['core']=str(tmp_path/'checkpoints/core')
    pd.DataFrame(rows).to_csv(cfg['metadata']['output'],index=False)
    return cfg,Path(cfg['metadata']['output'])


def test_partition_never_validates_test_labels_or_uses_augmentation(manifest_case,monkeypatch):
    cfg,path=manifest_case
    frame=pd.read_csv(path,dtype=str,keep_default_na=False)
    frame.loc[frame['split'].eq('test'),'label']='must-not-be-read'
    frame.loc[frame['split'].eq('test'),'image_path']='/does-not-exist/test.png'
    frame.to_csv(path,index=False)
    validation=manifest_partition(path,cfg,'validation')
    seen=[]
    def preprocess(path,local,split,rng=None):
        seen.append((local['model']['selected'],split,rng))
        assert split=='validation' and rng is None
        return np.zeros((224,224,3),dtype=np.float32)
    monkeypatch.setattr('glaucoma.evaluation.inference.preprocess_image',preprocess)
    class Model:
        def __call__(self,x,training):
            assert training is False
            return np.full((len(x),1),.25)
    cfg['training']['batch_size']=1
    for name in MEMBERS:
        scores=predict_model(Model(),validation,cfg,name,'validation')
        assert scores.image_id.tolist()==['002','003'] and len(scores)==2
    assert len(seen)==6
    with pytest.raises(ValueError):manifest_partition(path,cfg,'test')


def test_test_gate_requires_frozen_state_before_test_access(manifest_case,monkeypatch,tmp_path):
    cfg,path=manifest_case
    def forbidden(*a,**k):raise AssertionError('Test access before frozen configuration')
    monkeypatch.setattr('glaucoma.evaluation.workflow.predict_partition',forbidden)
    with pytest.raises(PermissionError):run_validation(cfg,enabled=False)
    with pytest.raises(PermissionError):run_final_test(cfg,enabled=False)
    with pytest.raises(FileNotFoundError,match='Freeze'):run_final_test(cfg,enabled=True)


def test_full_workflow_on_synthetic_predictions_freezes_and_locks_test(manifest_case,monkeypatch):
    from glaucoma.evaluation import workflow
    cfg,path=manifest_case
    provenance={name:{'checkpoint_sha256':name,'contract_hash':'synthetic','path':'synthetic'} for name in MEMBERS}
    monkeypatch.setattr(workflow,'verify_selected_checkpoints',lambda *a,**k:copy.deepcopy(provenance))
    calls=[]
    def predict(local,manifest,split,checkpoints,flat_image_root=None,expected_graphs=None):
        if split=='test':assert (Path(cfg['artifacts']['root'])/'phase4/default/frozen_ensemble.json').is_file()
        calls.append(split)
        subset=manifest_partition(manifest,local,split)
        table=subset[['dataset_name','image_id','split','label']].copy()
        for i,name in enumerate(MEMBERS):table[name]=np.where(table.label.eq(1),.85-i*.1,.15+i*.1)
        individuals={name:table[['dataset_name','image_id','split','label',name]].rename(columns={name:'score'}) for name in MEMBERS}
        return table,individuals,{name:'graph' for name in MEMBERS}
    monkeypatch.setattr(workflow,'predict_partition',predict)
    first=run_validation(cfg,enabled=True)
    assert calls==['validation']
    directory=Path(first['run_directory'])
    frozen=load_frozen(directory)
    assert frozen['threshold']==.5 and frozen['test_used_for_selection'] is False
    assert (directory/'validation/weight_search.csv').is_file()
    with pytest.raises(FileExistsError):run_validation(cfg,enabled=True)
    changed=copy.deepcopy(cfg);changed['preprocessing']['clahe']['enabled']=True
    with pytest.raises(ValueError,match='changed'):run_final_test(changed,enabled=True)
    assert calls==['validation']
    original_manifest=path.read_bytes();path.write_bytes(original_manifest+b'\n')
    with pytest.raises(ValueError,match='changed'):run_final_test(cfg,enabled=True)
    path.write_bytes(original_manifest)
    provenance[MEMBERS[0]]['checkpoint_sha256']='modified'
    with pytest.raises(ValueError,match='provenance'):run_final_test(cfg,enabled=True)
    provenance[MEMBERS[0]]['checkpoint_sha256']=MEMBERS[0]
    assert calls==['validation']
    final=run_final_test(cfg,enabled=True)
    assert calls==['validation','test'] and final['partition']=='test'
    assert (directory/'test_complete.json').is_file()
    assert load_frozen(directory)==frozen
    with pytest.raises(FileExistsError):run_final_test(cfg,enabled=True)


def test_frozen_file_tampering_rejected(manifest_case,tmp_path):
    cfg,path=manifest_case
    directory=tmp_path/'evaluation'
    freeze_validation(aligned(),{}, {},cfg,path,directory)
    file=directory/'frozen_ensemble.json'
    frozen=json.loads(file.read_text());frozen['threshold']=.1;file.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='checksum'):load_frozen(directory)


def test_training_digest_compatible_and_selected_contracts_verified(manifest_case,tmp_path):
    from glaucoma.training.data import load_training_metadata
    from glaucoma.evaluation.checkpoints import read_manifest
    cfg,path=manifest_case
    frame=load_training_metadata(path,cfg)
    import hashlib
    digest=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    assert training_manifest_digest(read_manifest(path))==digest
    records={}
    for name in MEMBERS:
        local=copy.deepcopy(cfg);local['model']['selected']=name
        contract={key:local[key] for key in ('model','training','preprocessing','augmentation','seed')}
        contract['training_validation_manifest_sha256']=digest
        contract_hash=canonical_hash(contract)
        for stage,loss in [('stage1',.2),('stage2',.1)]:
            directory=Path(cfg['checkpoints']['core'])/name/stage;directory.mkdir(parents=True)
            (directory/'best.keras').write_bytes(f'synthetic checkpoint marker {name} {stage}'.encode())
            (directory/'contract.json').write_text(json.dumps({**contract,'runtime':{'note':'synthetic'}}))
            (directory/'state.json').write_text(json.dumps({'complete':True,'best_epoch':2,'best_val_loss':loss,'contract_hash':contract_hash}))
        record=tmp_path/f'{name}.json'
        record.write_text(json.dumps({'stage':'stage2','epoch':2,'val_loss':.1,'contract_hash':contract_hash,'complete':True,
          'checkpoint':str(Path(cfg['checkpoints']['core'])/name/'stage2/best.keras')}))
        records[name]=record
    verified=verify_selected_checkpoints(cfg,path,records)
    assert set(verified)==set(MEMBERS)
    assert all(item['selection']['stage']=='stage2' for item in verified.values())
    damaged=Path(cfg['checkpoints']['core'])/MEMBERS[0]/'stage2/contract.json'
    contract=json.loads(damaged.read_text());contract['seed']=7;damaged.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='digest'):verify_selected_checkpoints(cfg,path,records)


def test_loaded_model_identity_checked_with_synthetic_keras_graph():
    tf=pytest.importorskip('tensorflow')
    from glaucoma.evaluation.checkpoints import verify_model_identity
    x=tf.keras.Input((224,224,3))
    feature=tf.keras.layers.Conv2D(1280,1)(tf.keras.layers.AveragePooling2D(32)(x))
    backbone=tf.keras.Model(x,feature,name='mobilenetv2_synthetic')
    outer=tf.keras.Input((224,224,3))
    y=tf.keras.layers.GlobalAveragePooling2D()(backbone(outer,training=False))
    y=tf.keras.layers.Dropout(.2)(y)
    model=tf.keras.Model(outer,tf.keras.layers.Dense(1,activation='sigmoid')(y),name='mobilenet_v2')
    assert len(verify_model_identity(model,'mobilenet_v2'))==64
    with pytest.raises(ValueError,match='identity'):verify_model_identity(model,'resnet50')
    tf.keras.backend.clear_session()


def test_saturated_probabilities_are_valid_grid_candidates():
    frame=aligned();frame.loc[:,list(MEMBERS)]=1.0
    weights,candidates=optimize_weights(frame)
    assert np.isfinite(candidates.validation_binary_crossentropy).all()
    np.testing.assert_allclose(list(weights.values()),[1/3]*3)
    result=voting_predictions(np.ones((2,3)),[.1,.2,.7])
    np.testing.assert_allclose(result['weighted_soft_voting']['scores'],[1,1])


def test_notebook04_to05_handoff_reuses_predictions_and_preserves_reports(manifest_case,monkeypatch):
    from glaucoma.evaluation import workflow
    cfg,path=manifest_case
    calls=[]
    monkeypatch.setattr(workflow,'verify_selected_checkpoints',lambda *a,**k:{name:{'source':'synthetic'} for name in MEMBERS})
    def predict(local,manifest,split,checkpoints,flat_image_root=None,**kwargs):
        assert split=='validation'
        calls.append(split)
        rows=manifest_partition(manifest,local,split)
        table=rows[['dataset_name','image_id','split','label']].copy()
        for i,name in enumerate(MEMBERS):table[name]=np.where(table.label.eq(1),.81-i*.1,.19+i*.1)
        individual={name:table[['dataset_name','image_id','split','label',name]].rename(columns={name:'score'}) for name in MEMBERS}
        return table,individual,{name:'synthetic-graph' for name in MEMBERS}
    monkeypatch.setattr(workflow,'predict_partition',predict)
    optimizer=workflow.optimize_weights
    def forbidden(*a,**k):raise AssertionError('Wrong notebook responsibility')
    monkeypatch.setattr(workflow,'optimize_weights',forbidden)
    model_result=workflow.run_model_validation(cfg,enabled=True)
    directory=Path(model_result['run_directory'])
    assert calls==['validation']
    assert not (directory/'frozen_ensemble.json').exists()
    assert not (directory/'validation/weight_search.csv').exists()
    metrics=json.loads((directory/'validation/metrics.json').read_text())
    assert set(metrics['methods'])==set(MEMBERS)
    originals={p:p.read_bytes() for p in (directory/'validation').rglob('*') if p.is_file()}
    cached,_=workflow.load_saved_validation(cfg,directory,path)
    assert cached.image_id.tolist()==['002','003']
    with pytest.raises(FileNotFoundError,match='Freeze'):workflow.run_final_test(cfg,enabled=True)
    monkeypatch.setattr(workflow,'optimize_weights',optimizer)
    monkeypatch.setattr(workflow,'predict_partition',forbidden)
    monkeypatch.setattr(workflow,'verify_selected_checkpoints',forbidden)
    ensemble_result=workflow.run_ensemble_validation(cfg,enabled=True)
    assert calls==['validation']
    assert Path(ensemble_result['frozen_ensemble']).is_file()
    assert not (directory/'test_started.json').exists()
    assert all(p.read_bytes()==data for p,data in originals.items())
    ensemble_metrics=json.loads((directory/'validation/ensemble/metrics.json').read_text())
    assert set(ensemble_metrics['methods'])=={*MEMBERS,'majority_voting','equal_soft_voting','weighted_soft_voting'}
    source=directory/f'validation/individual_predictions/{MEMBERS[0]}.csv'
    source.write_bytes(source.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='modified'):workflow.load_frozen(directory)


def test_notebook05_requires_notebook04_outputs_and_opt_in(manifest_case):
    from glaucoma.evaluation.workflow import run_ensemble_validation,run_model_validation
    cfg,_=manifest_case
    with pytest.raises(PermissionError):run_model_validation(cfg)
    with pytest.raises(PermissionError):run_ensemble_validation(cfg)
    with pytest.raises(FileNotFoundError,match='Notebook 04'):run_ensemble_validation(cfg,enabled=True)


def test_readonly_kaggle_handoff_copies_only_before_ensemble_search(manifest_case,monkeypatch,tmp_path):
    from glaucoma.evaluation import workflow
    cfg,path=manifest_case
    monkeypatch.setattr(workflow,'verify_selected_checkpoints',lambda *a,**k:{name:{'source':'synthetic'} for name in MEMBERS})
    def predict(local,manifest,split,checkpoints,flat_image_root=None,**kwargs):
        rows=manifest_partition(manifest,local,split)
        table=rows[['dataset_name','image_id','split','label']].copy()
        for name in MEMBERS:table[name]=np.where(table.label.eq(1),.9,.1)
        individual={name:table[['dataset_name','image_id','split','label',name]].rename(columns={name:'score'}) for name in MEMBERS}
        return table,individual,{name:'synthetic' for name in MEMBERS}
    monkeypatch.setattr(workflow,'predict_partition',predict)
    source=Path(workflow.run_model_validation(cfg,enabled=True)['run_directory'])
    before={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    target=tmp_path/'writable-phase5'
    workflow.copy_model_validation(source,target)
    assert {p.relative_to(target):p.read_bytes() for p in target.rglob('*') if p.is_file()}==before
    assert {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}==before
    with pytest.raises(FileExistsError):workflow.copy_model_validation(source,target)
    (source/'test_started.json').write_text('{}')
    with pytest.raises(ValueError,match='pre-ensemble'):workflow.copy_model_validation(source,tmp_path/'another-run')
