"""Actual TF 2.20 CPU execution, synthetic inputs only; never download weights."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip('tensorflow')
from glaucoma.models.factory import build_model, get_backbone
from glaucoma.training.data import load_training_metadata, make_dataset
from glaucoma.training.trainer import configure_stage, train_stage, recover_best_model, select_best_checkpoint
from glaucoma.training.artifacts import stage_paths
from glaucoma.utils.notebook_runtime import configure_workspace
from glaucoma.utils.reproducibility import set_random_seed


@pytest.fixture(autouse=True)
def clear_tensorflow():
    tf.keras.backend.clear_session()
    set_random_seed(42)
    yield
    tf.keras.backend.clear_session()


@pytest.mark.parametrize('name',['mobilenet_v2','efficientnet_b0','resnet50'])
def test_real_core_factory_forward_offline(name):
    model=build_model(name,weights=None)
    assert model.input_shape==(None,224,224,3) and model.output_shape==(None,1)
    assert not get_backbone(model).trainable
    assert len(model.layers)==5 # input, backbone, GAP, dropout, sigmoid (no double normalization)
    assert model.layers[-2].rate==.2 and model.layers[-1].activation==tf.keras.activations.sigmoid
    from glaucoma.evaluation.checkpoints import verify_model_identity
    assert len(verify_model_identity(model,name))==64
    output=model(tf.zeros((1,224,224,3)),training=False).numpy()
    assert output.shape==(1,1) and np.isfinite(output).all() and ((output>=0)&(output<=1)).all()
    assert any(isinstance(x,tf.keras.layers.Rescaling) for x in get_backbone(model).layers)==(name=='efficientnet_b0')
    configure_stage(model,'stage2')
    layers=get_backbone(model).layers
    assert all(not x.trainable for x in layers[:-20])
    assert all(not x.trainable for x in layers if isinstance(x,tf.keras.layers.BatchNormalization))
    assert any(x.trainable_weights for x in layers[-20:])


def tiny_model():
    # Small serializable stand-in for testing the real fit/checkpoint machinery.
    inputs=tf.keras.Input((224,224,3))
    x=tf.keras.layers.AveragePooling2D(32)(inputs)
    for i in range(10):
        x=tf.keras.layers.Conv2D(2,1,name=f'conv_{i}')(x)
        x=tf.keras.layers.BatchNormalization(name=f'bn_{i}')(x)
        x=tf.keras.layers.Activation('relu')(x)
    backbone=tf.keras.Model(inputs,x,name='tiny_backbone')
    backbone.trainable=False
    outer=tf.keras.Input((224,224,3))
    y=backbone(outer,training=False)
    y=tf.keras.layers.GlobalAveragePooling2D()(y)
    y=tf.keras.layers.Dropout(.2)(y)
    return tf.keras.Model(outer,tf.keras.layers.Dense(1,activation='sigmoid')(y),name='mobilenet_v2')


@pytest.fixture
def training_case(tmp_path,image_factory):
    cfg=configure_workspace('configs/project.yaml',tmp_path/'images',tmp_path/'artifacts','synthetic')
    cfg['model']['selected']='mobilenet_v2'
    cfg['model']['weights']=None
    cfg['training']['batch_size']=2
    cfg['training']['stage1']['epochs']=1
    cfg['training']['stage2']['epochs']=1
    rows=[]
    for i in range(6):
        image=image_factory(f'images/image{i}.png',seed=i)
        rows.append(dict(image_id=str(i),image_path=str(image),dataset_name='acrima',patient_id='',label=i%2,split='train' if i<3 else 'validation'))
    # A deliberately nonexistent test path: no code may stat or decode it.
    rows.append(dict(image_id='test',image_path='/never/read/test.png',dataset_name='acrima',patient_id='',label='not used',split='test'))
    path=Path(cfg['metadata']['output']);pd.DataFrame(rows).to_csv(path,index=False)
    return cfg,path


def test_frozen_loader_partial_batches_no_test_access(training_case):
    cfg,path=training_case
    original=path.read_bytes()
    frame=load_training_metadata(path,cfg)
    assert set(frame.split)=={'train','validation'} and len(frame)==6
    for split in ('train','validation'):
        ds=make_dataset(frame,cfg,split)
        assert int(tf.data.experimental.cardinality(ds))==2
        batches=list(ds.as_numpy_iterator())
        assert [len(x) for x,y in batches]==[2,1]
        assert all(y.shape==(len(x),1) for x,y in batches)
    a=list(make_dataset(frame,cfg,'validation').as_numpy_iterator())
    b=list(make_dataset(frame,cfg,'validation',start_epoch=9).as_numpy_iterator())
    for (x,y),(z,w) in zip(a,b):np.testing.assert_array_equal(x,z)
    assert path.read_bytes()==original
    with pytest.raises(ValueError,match='test'):make_dataset(frame,cfg,'test')


def test_trainability_and_frozen_bn_statistics():
    model=tiny_model()
    configure_stage(model,'stage1')
    assert len(model.trainable_weights)==2
    configure_stage(model,'stage2')
    backbone=get_backbone(model)
    assert all(not x.trainable for x in backbone.layers[:-20])
    before=[w.numpy().copy() for layer in backbone.layers if isinstance(layer,tf.keras.layers.BatchNormalization) for w in layer.non_trainable_weights]
    model.train_on_batch(np.ones((2,224,224,3),dtype=np.float32),np.array([[0],[1]],dtype=np.float32))
    after=[w.numpy() for layer in backbone.layers if isinstance(layer,tf.keras.layers.BatchNormalization) for w in layer.non_trainable_weights]
    for x,y in zip(before,after):np.testing.assert_array_equal(x,y)
    assert float(model.optimizer.learning_rate.numpy())==pytest.approx(.0001)


def test_training_opt_in_and_baseline_separation(cfg):
    with pytest.raises(PermissionError):train_stage(cfg,'stage1')
    cfg['model']['selected']='vgg16'
    with pytest.raises(ValueError,match='VGG'):train_stage(cfg,'stage1',enabled=True)
    with pytest.raises(ValueError):build_model('vgg16',weights=None)


def test_two_stage_fit_persistence_and_best_recovery(training_case,monkeypatch):
    cfg,path=training_case
    model=tiny_model()
    original=path.read_bytes()
    first=train_stage(cfg,'stage1',model=model,enabled=True,verbose=0)
    assert first['completed_epochs']==1
    p1=stage_paths(cfg,'stage1')
    expected=tf.keras.models.load_model(p1['best'])(tf.zeros((1,224,224,3)),training=False).numpy()
    # Deliberately corrupt the in-memory model; Stage 2 must load Stage 1 best, not use it.
    for variable in model.trainable_weights:variable.assign(tf.ones_like(variable)*100)
    from glaucoma.training import trainer
    best_weights=tf.keras.models.load_model(p1['best'],compile=False).get_weights()
    original_configure=trainer.configure_stage
    def verify_warm_start(restored,stage,*args,**kwargs):
        if stage=='stage2':
            for actual,saved in zip(restored.get_weights(),best_weights):
                np.testing.assert_array_equal(actual,saved)
        return original_configure(restored,stage,*args,**kwargs)
    monkeypatch.setattr(trainer,'configure_stage',verify_warm_start)
    second=train_stage(cfg,'stage2',enabled=True,verbose=0)
    for stage in ('stage1','stage2'):
        paths=stage_paths(cfg,stage)
        assert all(paths[key].is_file() for key in ('best','last','state','history','csv','metrics','contract'))
        history=json.loads(paths['history'].read_text())
        assert all(key in history for key in ('loss','accuracy','auc','precision','recall','val_loss','val_accuracy','val_auc'))
        assert json.loads(paths['state'].read_text())['complete']
    recovered=recover_best_model(cfg,'stage1')
    np.testing.assert_allclose(recovered(tf.zeros((1,224,224,3)),training=False).numpy(),expected)
    choice=select_best_checkpoint(cfg,save_selection=True)
    assert choice['val_loss']==min(first['best_val_loss'],second['best_val_loss'])
    assert path.read_bytes()==original
    assert not Path('/never/read/test.png').exists()
    with pytest.raises(FileExistsError):train_stage(cfg,'stage1',model=tiny_model(),enabled=True,verbose=0)


def test_missing_stage1_prevents_fine_tuning(training_case):
    cfg,_=training_case
    with pytest.raises(FileNotFoundError,match='Stage 1'):train_stage(cfg,'stage2',enabled=True,verbose=0)


def test_random_seeds_reproduce_offline_head_weights():
    set_random_seed(42);first=tiny_model().get_weights()
    tf.keras.backend.clear_session();set_random_seed(42);second=tiny_model().get_weights()
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)


def test_interrupted_run_resumes_at_epoch_boundary(training_case,monkeypatch):
    from glaucoma.training import trainer
    cfg,path=training_case
    cfg['training']['stage1']['epochs']=2
    original_callbacks=trainer.make_callbacks
    class Interrupt(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            raise InterruptedError('simulated runtime disconnect')
    monkeypatch.setattr(trainer,'make_callbacks',lambda *args,**kwargs: original_callbacks(*args,**kwargs)+[Interrupt()])
    with pytest.raises(InterruptedError):
        train_stage(cfg,'stage1',model=tiny_model(),enabled=True,verbose=0)
    paths=stage_paths(cfg,'stage1')
    state=json.loads(paths['state'].read_text())
    assert state['completed_epochs']==1 and not state['complete']
    changed=copy.deepcopy(cfg);changed['seed']=7
    with pytest.raises(ValueError,match='contract'):
        train_stage(changed,'stage1',enabled=True,resume=True,verbose=0)
    monkeypatch.setattr(trainer,'make_callbacks',original_callbacks)
    result=train_stage(cfg,'stage1',enabled=True,resume=True,verbose=0)
    assert result['completed_epochs']==2
    history=json.loads(paths['history'].read_text())
    assert len(history['loss'])==2
    assert len(pd.read_csv(paths['csv']))==2


def test_callbacks_and_learning_rate_reduction(training_case):
    from glaucoma.training.callbacks import make_callbacks
    cfg,_=training_case
    cfg['training']['reduce_lr']={'factor':.5,'patience':1,'min_delta':.0001,'min_lr':1e-7,'cooldown':0}
    callbacks=make_callbacks(cfg,stage_paths(cfg,'stage1'),'synthetic-contract')
    for klass in (tf.keras.callbacks.EarlyStopping,tf.keras.callbacks.ModelCheckpoint,
                  tf.keras.callbacks.ReduceLROnPlateau,tf.keras.callbacks.CSVLogger):
        assert any(isinstance(x,klass) for x in callbacks)
    reduce=next(x for x in callbacks if isinstance(x,tf.keras.callbacks.ReduceLROnPlateau))
    model=configure_stage(tiny_model(),'stage1')
    reduce.set_model(model);reduce.on_train_begin()
    reduce.on_epoch_end(0,{'val_loss':1.0});reduce.on_epoch_end(1,{'val_loss':1.1})
    assert float(model.optimizer.learning_rate.numpy())==pytest.approx(.0005)


def test_imagenet_arguments_without_network(monkeypatch):
    calls=[]
    def fake_application(**kwargs):
        calls.append(kwargs)
        return get_backbone(tiny_model())
    monkeypatch.setattr(tf.keras.applications,'MobileNetV2',fake_application)
    model=build_model('mobilenet_v2',weights='imagenet')
    assert calls==[{'include_top':False,'weights':'imagenet','input_shape':(224,224,3)}]
    assert model.output_shape==(None,1)


@pytest.mark.parametrize('name,application', [('mobilenet_v2','mobilenet_v2'),('resnet50','resnet50'),('efficientnet_b0','efficientnet')])
def test_existing_normalization_matches_keras(name,application):
    from glaucoma.data.preprocessing import normalize
    image=np.random.default_rng(42).uniform(0,255,(2,8,8,3)).astype(np.float32)
    expected=getattr(tf.keras.applications,application).preprocess_input(image.copy())
    np.testing.assert_allclose(normalize(image,name),expected,rtol=1e-6,atol=1e-6)


def test_cli_requires_opt_in_before_reading_session():
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'scripts/train_core.py','--session','/does/not/exist.json','--model','mobilenet_v2'],capture_output=True,text=True)
    assert result.returncode==2 and 'Training disabled' in result.stderr
