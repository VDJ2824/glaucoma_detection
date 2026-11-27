"""Offline analytical and actual core-architecture Grad-CAM checks; no downloads."""
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
from glaucoma.explainability.gradcam import GradCAM, GradCAMError, normalize_heatmap
from glaucoma.explainability.fusion import fuse_heatmaps, reverse_letterbox, overlay_heatmap
from glaucoma.explainability.workflow import select_examples, load_analysis_context, run_explanations
from glaucoma.ensemble.predictions import MEMBERS


def tiny_model(name='mobilenet_v2', shape=(4,4,3), zero=False):
    import tensorflow as tf
    x=tf.keras.Input(shape)
    y=tf.keras.layers.Conv2D(2,1,strides=32 if shape[0]==224 else 1,use_bias=False,name={'mobilenet_v2':'Conv_1','efficientnet_b0':'top_conv','resnet50':'conv5_block3_3_conv'}[name])(x)
    backbone=tf.keras.Model(x,y,name='backbone')
    backbone.trainable=False
    outer=tf.keras.Input(shape)
    y=backbone(outer,training=False)
    y=tf.keras.layers.GlobalAveragePooling2D()(y)
    y=tf.keras.layers.Dropout(.2)(y)
    output=tf.keras.layers.Dense(1,activation='sigmoid')(y)
    model=tf.keras.Model(outer,output,name=name)
    kernel=np.zeros((1,1,3,2),dtype=np.float32);kernel[0,0,0]=[1,-1]
    backbone.layers[-1].set_weights([kernel])
    model.layers[-1].set_weights([np.array([[0],[0]] if zero else [[1],[-1]],np.float32),np.zeros(1,np.float32)])
    return model


def test_nested_frozen_signed_targets_are_analytically_correct():
    tf=pytest.importorskip('tensorflow')
    model=tiny_model()
    x=np.zeros((4,4,3),np.float32);x[:,:2,0]=1;x[:,2:,0]=-1
    before=[v.numpy().copy() for v in model.weights]
    trainability=[layer.trainable for layer in model.layers]
    cam=GradCAM(model,'mobilenet_v2')
    positive=cam.explain(x,1);negative=cam.explain(x,0)
    assert positive.status==negative.status=='ok'
    np.testing.assert_array_equal(positive.heatmap,np.tile([1,1,0,0],(4,1)))
    np.testing.assert_array_equal(negative.heatmap,np.tile([0,0,1,1],(4,1)))
    assert positive.probability==pytest.approx(float(model(x[None],training=False)[0,0]))
    assert positive.probability==negative.probability==.5
    assert trainability==[layer.trainable for layer in model.layers]
    for a,b in zip(before,model.weights):np.testing.assert_array_equal(a,b.numpy())
    np.testing.assert_array_equal(cam.explain(x).heatmap,positive.heatmap)
    tf.keras.backend.clear_session()


def test_zero_gradient_relu_suppression_and_invalid_cases():
    pytest.importorskip('tensorflow')
    model=tiny_model(zero=True);cam=GradCAM(model,'mobilenet_v2')
    result=cam.explain(np.ones((4,4,3),np.float32))
    assert result.status=='zero_gradient' and not result.heatmap.any()
    cam=GradCAM(tiny_model(),'mobilenet_v2')
    assert cam.explain(np.ones((4,4,3)),0).status=='no_positive_activation'
    for target in [-1,2,True]:
        with pytest.raises(GradCAMError,match='target_class'):cam.explain(np.ones((4,4,3)),target)
    for x in [np.ones((2,4,4,3)),np.full((4,4,3),np.nan)]:
        with pytest.raises(GradCAMError,match='finite preprocessed'):cam.explain(x)
    with pytest.raises(GradCAMError,match='Missing spatial'):GradCAM(model,'mobilenet_v2','missing')
    with pytest.raises(GradCAMError,match='named core'):GradCAM(model,'vgg16')
    for x in [np.full((2,2),np.nan),-np.ones((2,2)),np.zeros((2,2,1))]:
        with pytest.raises(GradCAMError,match='Invalid heatmap'):normalize_heatmap(x)


@pytest.mark.parametrize('name',MEMBERS)
def test_actual_nested_core_backbone_gradients_offline(name):
    tf=pytest.importorskip('tensorflow')
    from glaucoma.models.factory import build_model
    from glaucoma.data.preprocessing import normalize
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
    model=build_model(name,weights=None)
    x=normalize(np.random.default_rng(42).integers(0,256,(224,224,3),dtype=np.uint8),name)
    cam=GradCAM(model,name)
    for target in (0,1):
        result=cam.explain(x,target)
        assert result.heatmap.shape==(7,7) and np.isfinite(result.heatmap).all()
        assert result.status in ('ok','no_positive_activation','zero_gradient')
        assert result.probability==pytest.approx(float(model(x[None],training=False)[0,0]),abs=1e-6)
    tf.keras.backend.clear_session()


def test_fusion_alignment_and_frozen_weights():
    maps=dict(zip(MEMBERS,[np.array([[1,0],[0,0]]),np.array([[0,1],[0,0]]),np.array([[0,0],[0,1]])]))
    np.testing.assert_array_equal(fuse_heatmaps(maps,shape=(2,2)),np.array([[1,1],[0,1]]))
    weighted=fuse_heatmaps(maps,dict(zip(MEMBERS,[0,.15,.85])),shape=(2,2))
    np.testing.assert_allclose(weighted,np.array([[0,.15/.85],[0,1]]),atol=1e-7)
    before={n:m.copy() for n,m in maps.items()}
    assert fuse_heatmaps(maps,shape=(6,4)).shape==(6,4)
    for n in maps:np.testing.assert_array_equal(maps[n],before[n])
    with pytest.raises(ValueError):fuse_heatmaps(maps,[-1,1,1])
    with pytest.raises(ValueError):fuse_heatmaps({'mobilenet_v2':maps['mobilenet_v2']})
    maps['resnet50']=np.zeros((2,2))
    with pytest.raises(GradCAMError,match='Fusion unavailable'):fuse_heatmaps(maps)


@pytest.mark.parametrize('shape',[(100,201,3),(201,100,3),(1,901,3),(55,55,3)])
def test_reverse_letterbox_matches_existing_rounding_and_padding(shape):
    h,w=shape[:2];scale=min(224/h,224/w);nh,nw=max(1,round(h*scale)),max(1,round(w*scale))
    top,left=(224-nh)//2,(224-nw)//2
    # Heat only the padding: the correctly restored original must be empty.
    heatmap=np.ones((224,224),np.float32);heatmap[top:top+nh,left:left+nw]=0
    restored=reverse_letterbox(heatmap,shape)
    assert restored.shape==(h,w) and not restored.any()
    rgb=np.full(shape,100,np.uint8)
    np.testing.assert_array_equal(overlay_heatmap(rgb,restored),rgb)
    heatmap[:]=0;heatmap[top:top+nh,left:left+nw]=1
    np.testing.assert_allclose(reverse_letterbox(heatmap,shape),np.ones((h,w)),atol=1e-6)


def synthetic_predictions():
    return pd.DataFrame({'dataset_name':['acrima']*4,'image_id':['d','a','c','b'], 'split':['validation']*4,
                         'label':[1,0,0,1], 'mobilenet_v2':[.9,.1,.9,.9],
                         'efficientnet_b0':[.9,.1,.1,.9], 'resnet50':[.1,.1,.9,.9]})


def test_selection_is_key_ordered_and_reports_missing_categories():
    frame=synthetic_predictions();weights=[0,.15,.85]
    selected,summary=select_examples(frame,weights,.5,per_category=2)
    other,second=select_examples(frame.sample(frac=1,random_state=3),weights,.5,per_category=2)
    pd.testing.assert_frame_equal(selected,other);assert summary==second
    assert selected.image_id.tolist()==['a','b','c','d']
    assert 'disagreement' in selected.loc[selected.image_id.eq('d'),'categories'].iloc[0]
    perfect=frame.copy();perfect['label']=np.array([0,0,1,1])
    _,summary=select_examples(perfect,weights,.5)
    assert summary['categories']['incorrect']['available']==0
    with pytest.raises(ValueError):select_examples(frame,weights,.5,method='invented')


def test_disabled_workflow_and_test_opt_in_precede_any_file_access(tmp_path):
    with pytest.raises(PermissionError,match='disabled'):
        run_explanations({},tmp_path/'missing',tmp_path/'missing')
    with pytest.raises(PermissionError,match='post-hoc'):
        load_analysis_context({},tmp_path/'missing',tmp_path/'missing',split='test')


def test_private_output_guards(cfg,tmp_path):
    from glaucoma.explainability.workflow import _private_directory
    local=copy.deepcopy(cfg)
    local.setdefault('artifacts', {'root':str(tmp_path/'artifacts')})
    local['datasets']['acrima']['root']=str(tmp_path/'dataset')
    local['artifacts']['heatmaps']=str(tmp_path/'dataset')
    with pytest.raises(ValueError,match='separate'):_private_directory(local,'run',tmp_path/'phase4',None)
    local['artifacts']['heatmaps']=str(Path(__file__).resolve().parents[1]/'docs/private')
    with pytest.raises(ValueError,match='ignored artifacts'):_private_directory(local,'run',tmp_path/'phase4',None)


@pytest.fixture
def analysis_case(tmp_path, monkeypatch):
    pytest.importorskip('tensorflow')
    from glaucoma.utils.notebook_runtime import configure_workspace
    from glaucoma.evaluation.workflow import freeze_validation
    from glaucoma.data.preprocessing import preprocess_image
    from glaucoma.explainability import workflow
    cfg=configure_workspace('configs/project.yaml',tmp_path/'images',tmp_path/'artifacts','synthetic')
    cfg['datasets']['acrima']['root']=str(tmp_path/'images')
    cfg['preprocessing']['pad_value']=255  # synthetic uniform signal, independent of padding evidence
    image_root=tmp_path/'images';image_root.mkdir()
    models={name:tiny_model(name,shape=(224,224,3)) for name in MEMBERS}
    rows=[]
    for split in ['validation','test']:
        for i in range(2):
            path=image_root/f'{split}{i}.png'
            assert cv2.imwrite(str(path),np.full((31,59,3),255,np.uint8))
            rows.append(dict(image_id=path.name,image_path=str(path),dataset_name='acrima',patient_id='',label=i,split=split))
    manifest=tmp_path/'manifest.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
    tables={}
    for split in ['validation','test']:
        table=pd.DataFrame([r for r in rows if r['split']==split])[['dataset_name','image_id','split','label']]
        for name,model in models.items():
            local=copy.deepcopy(cfg);local['model']['selected']=name
            tensor=preprocess_image(image_root/f'{split}0.png',local,split)
            table[name]=float(model(tensor[None],training=False)[0,0])
        tables[split]=table
    records={name:{'model_name':name,'checkpoint_sha256':'synthetic','contract_hash':'contract','training_manifest_sha256':'training',
                   'input_shape':[224,224,3],'preprocessing':cfg['preprocessing'],'class_mapping':{'0':'non_glaucoma','1':'glaucoma'},
                   'selection':{'stage':'stage2','epoch':1,'val_loss':1.0,'complete':True,'contract_hash':'contract'}} for name in MEMBERS}
    graphs={name:'synthetic-'+name for name in MEMBERS}
    evaluation=tmp_path/'evaluation'
    frozen=freeze_validation(tables['validation'],records,graphs,cfg,manifest,evaluation)
    (evaluation/'test').mkdir();tables['test'].to_csv(evaluation/'test/aligned_predictions.csv',index=False)
    monkeypatch.setattr(workflow,'verify_selected_checkpoints',lambda *a,**kw:records)
    monkeypatch.setattr(workflow,'load_verified_model',lambda cfg,name,record:(models[name],graphs[name]))
    return cfg,manifest,evaluation,frozen


def test_private_end_to_end_panels_empty_map_status_and_source_preservation(analysis_case):
    cfg,manifest,evaluation,frozen=analysis_case
    before={p:p.read_bytes() for p in [manifest,*evaluation.rglob('*')] if p.is_file()}
    result=run_explanations(cfg,manifest,evaluation,enabled=True,name='synthetic',per_category=1)
    root=Path(result['directory'])
    assert result['examples']==2 and result['panels']==4
    assert (root/'complete.json').is_file()
    for path in root.glob('example_*/class_1/explanation.json'):
        data=json.loads(path.read_text())
        assert all(m['status']=='ok' for m in data['methods'].values())
        for name in ['equal_heatmap_fusion','weighted_heatmap_fusion']:
            assert (path.parent/f'{name}_overlay.png').is_file()
            assert np.load(path.parent/f'{name}_image.npy').shape==(31,59)
    for path in root.glob('example_*/class_0/explanation.json'):
        data=json.loads(path.read_text())
        assert data['methods']['mobilenet_v2']['status']=='no_positive_activation'
        assert data['methods']['equal_heatmap_fusion']['status'].startswith('unavailable')
        assert not (path.parent/'equal_heatmap_fusion_overlay.png').exists()
    for path,payload in before.items():assert path.read_bytes()==payload
    with pytest.raises(FileExistsError):run_explanations(cfg,manifest,evaluation,enabled=True,name='synthetic')


def test_test_analysis_requires_matching_completed_run_and_manifest(analysis_case):
    cfg,manifest,evaluation,frozen=analysis_case
    with pytest.raises(ValueError,match='Complete Phase 4'):
        load_analysis_context(cfg,manifest,evaluation,'test',allow_test=True)
    (evaluation/'test_complete.json').write_text(json.dumps({'frozen_seal':frozen['seal_sha256'],'partition':'test','n':2}))
    aligned,_,provenance=load_analysis_context(cfg,manifest,evaluation,'test',allow_test=True)
    assert len(aligned)==2 and provenance['source_prediction_integrity'].startswith('unsealed')
    broken=copy.deepcopy(cfg);broken['preprocessing']['pad_value']=128
    with pytest.raises(ValueError,match='methodology'):load_analysis_context(broken,manifest,evaluation,'test',allow_test=True)
    manifest.write_bytes(manifest.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='manifest changed'):load_analysis_context(cfg,manifest,evaluation,'test',allow_test=True)


def test_saved_score_mismatch_fails_without_silently_updating_predictions(analysis_case):
    cfg,manifest,evaluation,frozen=analysis_case
    (evaluation/'test_complete.json').write_text(json.dumps({'frozen_seal':frozen['seal_sha256'],'partition':'test','n':2}))
    path=evaluation/'test/aligned_predictions.csv'
    frame=pd.read_csv(path);frame['mobilenet_v2']=.01;frame.to_csv(path,index=False)
    payload=path.read_bytes()
    with pytest.raises(ValueError,match='saved Phase 4 prediction'):
        run_explanations(cfg,manifest,evaluation,enabled=True,split='test',allow_test=True,name='mismatch')
    assert path.read_bytes()==payload
    assert not (Path(cfg['artifacts']['heatmaps'])/'phase5/mismatch/complete.json').exists()


def test_disconnected_or_nonfinite_gradients_fail_explicitly():
    tf=pytest.importorskip('tensorflow')
    x=tf.keras.Input((4,4,3))
    conv=tf.keras.layers.Conv2D(2,1,name='Conv_1')(x)
    stopped=tf.keras.layers.Lambda(tf.stop_gradient)(conv)
    backbone=tf.keras.Model(x,stopped,name='stopped_backbone')
    outer=tf.keras.Input((4,4,3));y=backbone(outer)
    y=tf.keras.layers.GlobalAveragePooling2D()(y);y=tf.keras.layers.Dropout(.2)(y)
    model=tf.keras.Model(outer,tf.keras.layers.Dense(1,activation='sigmoid')(y),name='mobilenet_v2')
    with pytest.raises(GradCAMError,match='Disconnected'):
        GradCAM(model,'mobilenet_v2').explain(np.ones((4,4,3)))
    model=tiny_model()
    layer=model.layers[-1];weights=layer.get_weights();weights[0][:]=np.nan;layer.set_weights(weights)
    with pytest.raises(GradCAMError,match='Nonfinite'):
        GradCAM(model,'mobilenet_v2').explain(np.ones((4,4,3)))


def test_changed_checkpoint_provenance_or_graph_fails_before_explanations(analysis_case,monkeypatch):
    from glaucoma.explainability import workflow
    cfg,manifest,evaluation,frozen=analysis_case
    changed=copy.deepcopy(frozen['checkpoints']);changed['resnet50']['checkpoint_sha256']='changed'
    monkeypatch.setattr(workflow,'verify_selected_checkpoints',lambda *a,**kw:changed)
    with pytest.raises(ValueError,match='checkpoint provenance changed'):
        run_explanations(cfg,manifest,evaluation,enabled=True,name='bad-checkpoint')
    assert not (Path(cfg['artifacts']['heatmaps'])/'phase5/bad-checkpoint').exists()
    monkeypatch.setattr(workflow,'verify_selected_checkpoints',lambda *a,**kw:frozen['checkpoints'])
    monkeypatch.setattr(workflow,'load_verified_model',lambda *a,**kw:(None,'changed'))
    with pytest.raises(ValueError,match='graph differs'):
        run_explanations(cfg,manifest,evaluation,enabled=True,name='bad-graph')
    assert not (Path(cfg['artifacts']['heatmaps'])/'phase5/bad-graph/complete.json').exists()


def test_symlinked_output_cannot_escape_into_image_storage(cfg,tmp_path):
    from glaucoma.explainability.workflow import _private_directory
    local=copy.deepcopy(cfg)
    images=tmp_path/'images';images.mkdir()
    heatmaps=tmp_path/'heatmaps';heatmaps.mkdir()
    (heatmaps/'phase5').symlink_to(images,target_is_directory=True)
    local['datasets']['acrima']['root']=str(images)
    local['artifacts']={'root':str(tmp_path/'artifacts'),'heatmaps':str(heatmaps)}
    with pytest.raises(ValueError,match='separate'):_private_directory(local,'run',tmp_path/'phase4',None)
