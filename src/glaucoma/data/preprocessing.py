"""OpenCV/NumPy preprocessing; no TensorFlow dependency or file mutation."""
import cv2
import numpy as np
from glaucoma.config import CORE_MODELS, BASELINES


def normalize(image, model):
    x = image.astype(np.float32)
    if model in ('resnet50', 'vgg16', 'vgg19'):
        return x[..., ::-1].copy() - np.array([103.939, 116.779, 123.68], dtype=np.float32)
    if model == 'mobilenet_v2':
        return x / 127.5 - 1.0
    if model == 'efficientnet_b0':
        # Keras EfficientNet embeds rescaling and expects float RGB [0, 255].
        return x
    raise ValueError(f'Unsupported normalization model: {model}')


def augment(image, settings, rng):
    """Internal transform on RGB [0,255]; called only for training samples."""
    x = image.copy()
    if rng.random() < settings['horizontal_flip_probability']:
        x = cv2.flip(x, 1)
    angle = rng.uniform(-settings['rotation_degrees'], settings['rotation_degrees'])
    h, w = x.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1)
    x = cv2.warpAffine(x, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    delta = rng.uniform(-settings['brightness_delta'], settings['brightness_delta']) * 255
    return np.clip(x.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def preprocess_image(path, cfg, split, rng=None):
    if split not in ('train', 'validation', 'test'):
        raise ValueError('A known split is required before preprocessing')
    model = cfg['model']['selected']
    if model not in CORE_MODELS + BASELINES:
        raise ValueError('Unknown model')
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Cannot decode image: {path}')
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pre = cfg['preprocessing']
    if pre['clahe']['enabled']:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        settings = pre['clahe']
        lab[..., 0] = cv2.createCLAHE(clipLimit=settings['clip_limit'], tileGridSize=tuple(settings['tile_grid_size'])).apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    height, width, _ = cfg['model']['input_shape']
    h, w = image.shape[:2]
    scale = min(height/h, width/w)
    new_w, new_h = max(1, round(w*scale)), max(1, round(h*scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), pre['pad_value'], dtype=np.uint8)
    top, left = (height-new_h)//2, (width-new_w)//2
    canvas[top:top+new_h, left:left+new_w] = resized
    if split == 'train' and cfg['augmentation']['enabled']:
        if rng is None:
            raise ValueError('Pass a seeded persistent NumPy generator for training augmentation')
        canvas = augment(canvas, cfg['augmentation'], rng)
    return normalize(canvas, model)
