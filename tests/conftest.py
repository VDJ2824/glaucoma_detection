from pathlib import Path
import cv2
import numpy as np
import pytest
from glaucoma.config import load_config


@pytest.fixture
def cfg():
    return load_config(Path(__file__).resolve().parents[1] / 'configs/project.yaml')


@pytest.fixture
def image_factory(tmp_path):
    def make(name='image.png', seed=1, shape=(30, 60, 3)):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        return path
    return make
