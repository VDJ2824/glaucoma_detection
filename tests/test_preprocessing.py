import cv2
import numpy as np
import pandas as pd
import pytest
from glaucoma.data.preprocessing import preprocess_image, normalize
from glaucoma.data.loading import iter_batches


def test_rgb_letterbox_normalization(cfg,tmp_path):
    p=tmp_path/'color.png'
    bgr=np.full((20,40,3),[10,20,30],dtype=np.uint8);cv2.imwrite(str(p),bgr)
    cfg['model']['selected']='efficientnet_b0'
    out=preprocess_image(p,cfg,'test')
    assert out.shape==(224,224,3) and out.dtype==np.float32
    np.testing.assert_array_equal(out[112,112],[30,20,10])
    assert np.all(out[:56]==0) and np.all(out[168:]==0)
    np.testing.assert_allclose(normalize(np.array([[[30,20,10]]]),'resnet50'),[[[10-103.939,20-116.779,30-123.68]]],atol=1e-5)
    np.testing.assert_allclose(normalize(np.array([[[0,127.5,255]]]),'mobilenet_v2'),[[[-1,0,1]]])


def test_training_only_augmentation(cfg,image_factory):
    path=image_factory()
    before=path.read_bytes()
    val=preprocess_image(path,cfg,'validation')
    test=preprocess_image(path,cfg,'test',np.random.default_rng(2))
    np.testing.assert_array_equal(val,test)
    train=preprocess_image(path,cfg,'train',np.random.default_rng(42))
    np.testing.assert_array_equal(train,preprocess_image(path,cfg,'train',np.random.default_rng(42)))
    assert not np.array_equal(val,train)
    assert path.read_bytes()==before
    with pytest.raises(ValueError,match='generator'):preprocess_image(path,cfg,'train')
    with pytest.raises(ValueError):preprocess_image(path,cfg,'unknown')


def test_clahe_and_disabled_augmentation(cfg,image_factory):
    path=image_factory()
    cfg['preprocessing']['clahe']['enabled']=True
    cfg['augmentation']['enabled']=False
    out=preprocess_image(path,cfg,'train')
    assert out.shape==(224,224,3) and np.isfinite(out).all()
    np.testing.assert_array_equal(out,preprocess_image(path,cfg,'test'))


def test_batches_never_augment_eval(cfg,image_factory):
    p=image_factory()
    data=pd.DataFrame([dict(image_path=str(p),label=1,split='validation')]*17)
    a=list(iter_batches(data,cfg,'validation',epoch=0))
    b=list(iter_batches(data,cfg,'validation',epoch=10))
    assert [len(x) for x,y in a]==[16,1]
    for (x,y),(z,w) in zip(a,b):np.testing.assert_array_equal(x,z)
