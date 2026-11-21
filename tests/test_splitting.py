import pandas as pd
import pytest
from glaucoma.data.splitting import split_metadata


def metadata(patients=False):
    return pd.DataFrame([dict(image_id=str(i), image_path=f'/unused/{i}.png', dataset_name='acrima', patient_id=f'p{i//2}' if patients else '', label=(i//2)%2, split='') for i in range(200)])


def test_image_stratification_and_reproducibility():
    data=metadata()
    a=split_metadata(data); b=split_metadata(data)
    pd.testing.assert_frame_equal(a,b)
    assert a.split.value_counts().to_dict()=={'train':140,'validation':30,'test':30}
    assert all(abs(x.label.mean()-.5)<.05 for _,x in a.groupby('split'))
    assert data.split.eq('').all()
    assert a.patient_id.eq('').all()


def test_group_isolation_and_stratification():
    data=split_metadata(metadata(True))
    assert data.groupby('patient_id').split.nunique().max()==1
    assert all(x.label.nunique()==2 for _,x in data.groupby('split'))
    assert all(abs(len(x)/len(data)-r)<.06 for s,r in [('train',.7),('validation',.15),('test',.15)] for _,x in [(s,data[data.split==s])])
    pd.testing.assert_frame_equal(data,split_metadata(metadata(True)))


def test_partial_patient_ids_rejected():
    data=metadata(True);data.loc[0,'patient_id']=''
    with pytest.raises(ValueError,match='Partial'):split_metadata(data)


def test_official_splits_preserved_and_leakage_rejected():
    data=metadata(True);data['split']=['train']*140+['validation']*30+['test']*30
    pd.testing.assert_frame_equal(split_metadata(data,mode='official'),data)
    data.loc[0,'split']='test'
    with pytest.raises(ValueError,match='leakage'):split_metadata(data,mode='official')


def test_missing_official_or_existing_splits_rejected():
    data=metadata()
    with pytest.raises(ValueError,match='Official'):split_metadata(data,mode='official')
    data.loc[0,'split']='test'
    with pytest.raises(ValueError,match='Existing'):split_metadata(data)


def test_insufficient_classes_rejected():
    data=metadata().iloc[:4]
    with pytest.raises(ValueError):split_metadata(data)
