import shutil
import pandas as pd
import pytest
from glaucoma.data.adapters import AcrimaAdapter, COLUMNS, create_adapter
from glaucoma.data.validation import validate_metadata
from glaucoma.data.loading import prepare_dataset


def row(path, **kwargs):
    return dict(image_id=path.name, image_path=str(path), dataset_name='acrima', patient_id='', label=0, split='', **kwargs)


def test_explicit_folders_and_no_patient_fabrication(tmp_path, image_factory):
    image_factory('normal/a.png'); image_factory('positive/b.png', seed=2)
    data = AcrimaAdapter(tmp_path, {'normal':0, 'positive':1}).discover()
    assert data.columns.tolist() == COLUMNS
    assert sorted(data.label) == [0,1]
    assert data.patient_id.eq('').all() and data.split.eq('').all()
    with pytest.raises(ValueError):
        AcrimaAdapter(tmp_path, {'DR':2}).discover()


def test_annotations_preserve_patient_and_official_split(tmp_path, image_factory):
    path = image_factory()
    ann = tmp_path/'labels.csv'
    pd.DataFrame([dict(image_id='a', image_path=path.name, label=1, patient_id='real-patient', split='test')]).to_csv(ann,index=False)
    data = AcrimaAdapter(tmp_path, annotations_csv=ann).discover()
    assert data.iloc[0].patient_id == 'real-patient'
    assert data.iloc[0].split == 'test'
    assert data.iloc[0].image_path == str(path)


def test_no_inference_of_labels(tmp_path, image_factory):
    image_factory()
    ann=tmp_path/'bad.csv'
    pd.DataFrame([{'image_id':'x','image_path':'image.png'}]).to_csv(ann,index=False)
    with pytest.raises(ValueError):
        AcrimaAdapter(tmp_path, annotations_csv=ann).discover()


def test_validation_errors(tmp_path, image_factory):
    good = image_factory()
    corrupt = tmp_path/'broken.jpg'; corrupt.write_text('not an image')
    data = pd.DataFrame([row(good),row(corrupt),row(tmp_path/'missing.png')])
    data['label'] = data.label.astype(object)
    data.loc[0,'label']=''
    result=validate_metadata(data,tmp_path)
    assert result.has_errors
    assert set(result.issues.code) == {'corrupt_image','invalid_path','missing_or_invalid_label'}


def test_outside_root_path_rejected(tmp_path, image_factory):
    good=image_factory()
    result=validate_metadata(pd.DataFrame([row(good)]),tmp_path/'other')
    assert result.has_errors and 'invalid_path' in result.issues.code.tolist()


def test_decoded_duplicates_and_conflicts(tmp_path,image_factory):
    path=image_factory()
    copy=tmp_path/'copy.png'; shutil.copyfile(path,copy)
    data=pd.DataFrame([row(path),row(copy)])
    result=validate_metadata(data,tmp_path)
    assert not result.has_errors and len(result.metadata)==1
    assert result.issues.iloc[0].code=='duplicate_removed'
    assert validate_metadata(data,tmp_path,'error').has_errors
    for field,value in [('label',1),('patient_id','other'),('split','test')]:
        conflict=data.copy(); conflict.loc[1,field]=value
        assert 'conflicting_duplicate' in validate_metadata(conflict,tmp_path).issues.code.tolist()


def test_duplicate_id_different_pixels(tmp_path,image_factory):
    a,b=image_factory(),image_factory('b.png',seed=2)
    data=pd.DataFrame([row(a),row(b)]);data['image_id']='same'
    assert 'duplicate_image_id' in validate_metadata(data,tmp_path).issues.code.tolist()


def test_unsupported_dataset(cfg):
    cfg['datasets']['selected']='refuge'
    with pytest.raises(NotImplementedError): create_adapter(cfg)


def test_end_to_end_originals_unchanged(cfg,tmp_path,image_factory):
    for label,folder in enumerate(('non_glaucoma','glaucoma')):
        for i in range(40): image_factory(f'raw/{folder}/{i}.png',seed=i+label*100)
    root=tmp_path/'raw'
    originals={p:p.read_bytes() for p in root.rglob('*.png')}
    cfg['datasets']['acrima']['root']=str(root)
    cfg['datasets']['acrima']['label_mode']='folders'
    cfg['metadata']['output']=str(tmp_path/'derived/metadata.csv')
    cfg['metadata']['validation_output']=str(tmp_path/'derived/issues.csv')
    # Exact duplicate removed before splitting; not another sample.
    shutil.copyfile(root/'glaucoma/0.png',root/'glaucoma/copy.png')
    frame=prepare_dataset(cfg)
    assert len(frame)==80 and set(frame.split)=={'train','validation','test'}
    assert pd.read_csv(cfg['metadata']['output']).columns.tolist()==COLUMNS
    assert all(p.read_bytes()==content for p,content in originals.items())
    assert len(list(root.rglob('*.png')))==81
    cfg['metadata']['output']=str(root/'metadata.csv')
    with pytest.raises(ValueError,match='outside'): prepare_dataset(cfg)


def test_validation_failure_does_not_write_metadata(cfg,tmp_path,image_factory):
    image_factory('raw/non_glaucoma/a.png')
    bad=tmp_path/'raw/glaucoma/b.png';bad.parent.mkdir();bad.write_bytes(b'corrupt')
    cfg['datasets']['acrima']['root']=str(tmp_path/'raw')
    cfg['datasets']['acrima']['label_mode']='folders'
    cfg['metadata']['output']=str(tmp_path/'output.csv')
    cfg['metadata']['validation_output']=str(tmp_path/'issues.csv')
    with pytest.raises(ValueError,match='validation failed'):prepare_dataset(cfg)
    assert not (tmp_path/'output.csv').exists()
    assert 'corrupt_image' in pd.read_csv(tmp_path/'issues.csv').code.tolist()
    assert bad.read_bytes()==b'corrupt'


@pytest.mark.parametrize('name,label', [
    ('Im310_g_ACRIMA.jpg',1), ('Im304_ACRIMA.jpg',0),
    ('im001_G_acrima.JPG',1), ('IM999_ACRIMA.JpG',0),
])
def test_filename_labels_case_insensitive(tmp_path,image_factory,name,label):
    image=image_factory(name)
    before=image.read_bytes()
    data=AcrimaAdapter(tmp_path,label_mode='filename').discover()
    assert data.iloc[0].label==label
    assert data.iloc[0].image_id==name and data.iloc[0].image_path==str(image)
    assert data.patient_id.eq('').all() and data.split.eq('').all()
    assert image.read_bytes()==before


@pytest.mark.parametrize('name', ['Im310_ACRIMA_g.jpg','Im_g_ACRIMA.jpg','Im310_x_ACRIMA.jpg',
                                  'Im310_g_ACRIMA_copy.jpg','Im304_ACRIMA.png',
                                  'Im304_ACRIMA.jpeg','random.jpg','Im304_ACRIMA.jpg.bak',
                                  'Im304_ACRIMA.jpg\n'])
def test_filename_mode_rejects_unexpected_names(tmp_path,name):
    (tmp_path/name).write_bytes(b'not decoded by discovery')
    with pytest.raises(ValueError,match='Unexpected entries'):
        AcrimaAdapter(tmp_path,label_mode='filename').discover()


def test_filename_mode_rejects_nested_or_empty_directory(tmp_path):
    with pytest.raises(ValueError,match='No canonical'):AcrimaAdapter(tmp_path,label_mode='filename').discover()
    (tmp_path/'nested').mkdir()
    with pytest.raises(ValueError,match='Unexpected entries'):AcrimaAdapter(tmp_path,label_mode='filename').discover()


def test_filename_mode_rejects_external_symlinks(tmp_path,image_factory):
    image=image_factory('outside.jpg')
    root=tmp_path/'flat';root.mkdir()
    (root/'Im001_ACRIMA.jpg').symlink_to(image)
    with pytest.raises(ValueError,match='Unexpected'):AcrimaAdapter(root,label_mode='filename').discover()


def test_configured_filename_mode(cfg,tmp_path,image_factory):
    image_factory('Im310_g_ACRIMA.jpg');image_factory('Im304_ACRIMA.jpg',seed=2)
    cfg['datasets']['acrima']['root']=str(tmp_path)
    cfg['datasets']['acrima']['label_mode']='filename'
    assert create_adapter(cfg).discover().label.value_counts().to_dict()=={0:1,1:1}


@pytest.mark.parametrize('kwargs', [dict(label_mode='unknown'),dict(label_mode='csv'),
    dict(label_mode='filename',annotations_csv='annotations.csv'),dict(label_mode='folders',annotations_csv='annotations.csv')])
def test_label_modes_reject_ambiguous_or_invalid_settings(tmp_path,kwargs):
    with pytest.raises(ValueError):AcrimaAdapter(tmp_path,**kwargs)


def test_explicit_csv_mode_remains_the_annotation_authority(cfg,tmp_path,image_factory):
    path=image_factory('Im310_g_ACRIMA.jpg')
    ann=tmp_path/'annotations.csv'
    pd.DataFrame([dict(image_id='image-only-id',image_path=path.name,label=0,patient_id='verified-key')]).to_csv(ann,index=False)
    cfg['datasets']['acrima'].update(root=str(tmp_path),label_mode='csv',annotations_csv=str(ann))
    row=create_adapter(cfg).discover().iloc[0]
    assert row.label=='0' and row.patient_id=='verified-key'


def test_legacy_config_without_mode_still_supports_folders(cfg,tmp_path,image_factory):
    image_factory('normal/example.png')
    image_factory('positive/example.png',seed=2)
    cfg['datasets']['acrima'].pop('label_mode',None)
    cfg['datasets']['acrima'].update(root=str(tmp_path),class_directories={'normal':0,'positive':1})
    assert create_adapter(cfg).discover().label.tolist()==[0,1]
