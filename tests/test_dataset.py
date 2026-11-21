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
    cfg['metadata']['output']=str(tmp_path/'output.csv')
    cfg['metadata']['validation_output']=str(tmp_path/'issues.csv')
    with pytest.raises(ValueError,match='validation failed'):prepare_dataset(cfg)
    assert not (tmp_path/'output.csv').exists()
    assert 'corrupt_image' in pd.read_csv(tmp_path/'issues.csv').code.tolist()
    assert bad.read_bytes()==b'corrupt'
