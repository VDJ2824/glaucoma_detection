import shutil
import numpy as np
import pytest
from glaucoma.data.exploration import explore_dataset, plot_samples, plot_class_distribution


@pytest.fixture
def exploration_fixture(cfg,tmp_path,image_factory):
    root=tmp_path/'flat'
    image_factory('flat/Im001_ACRIMA.jpg',shape=(20,40,3))
    image_factory('flat/Im002_g_ACRIMA.jpg',seed=2,shape=(30,50,3))
    shutil.copyfile(root/'Im001_ACRIMA.jpg',root/'Im003_ACRIMA.jpg')
    (root/'Im004_g_ACRIMA.jpg').write_bytes(b'corrupt')
    cfg['datasets']['acrima']['root']=str(root)
    cfg['datasets']['acrima']['label_mode']='filename'
    return cfg,root


def test_exploration_retains_original_counts_and_reports_errors(exploration_fixture):
    cfg,root=exploration_fixture
    before={p:p.read_bytes() for p in root.iterdir()}
    result=explore_dataset(cfg)
    assert result.class_distribution['count'].tolist()==[2,2]
    assert result.class_distribution.percentage.tolist()==[50.,50.]
    assert len(result.metadata)==4 and len(result.dimensions)==3
    assert set(result.validation.issues.code)=={'duplicate_image','corrupt_image'}
    assert result.validation.has_errors
    assert result.dimension_statistics.loc['min','height']==20
    assert result.dimension_statistics.loc['max','width']==50
    assert result.metadata.patient_id.eq('').all() and result.metadata.split.eq('').all()
    assert {p:p.read_bytes() for p in root.iterdir()}==before


def test_plotting_uses_rgb_samples_and_is_reproducible(exploration_fixture):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg,_=exploration_fixture
    result=explore_dataset(cfg)
    distribution=plot_class_distribution(result)
    assert [p.get_height() for p in distribution.axes[0].patches]==[2,2]
    first=plot_samples(result,per_class=2,seed=42)
    second=plot_samples(result,per_class=2,seed=42)
    assert [ax.get_title() for ax in first.axes]==[ax.get_title() for ax in second.axes]
    import cv2
    row=result.dimensions.iloc[0]
    expected=cv2.cvtColor(cv2.imread(row.image_path),cv2.COLOR_BGR2RGB)
    # Both non-glaucoma samples have the same original decoded pixels.
    np.testing.assert_array_equal(first.axes[0].images[0].get_array(),expected)
    assert len(first.axes[3].images)==0 # corrupt image excluded, not filled with a fake sample
    for fig in (distribution,first,second):
        fig.canvas.draw()
        plt.close(fig)
    with pytest.raises(ValueError):plot_samples(result,per_class=0)


def test_all_corrupt_dataset_has_empty_dimensions(cfg,tmp_path):
    (tmp_path/'Im001_ACRIMA.jpg').write_bytes(b'corrupt')
    cfg['datasets']['acrima']['root']=str(tmp_path)
    result=explore_dataset(cfg)
    assert result.dimensions.empty
    assert result.dimension_statistics.loc['count','width']==0
