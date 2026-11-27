"""Aligned heatmap fusion and geometry-correct overlays, not score Grad-CAM."""
import cv2
import numpy as np
from glaucoma.ensemble.predictions import MEMBERS, validate_weights
from .gradcam import normalize_heatmap, GradCAMError


def resize_heatmap(heatmap, shape):
    if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('Positive (height, width) required')
    return cv2.resize(normalize_heatmap(heatmap), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def fuse_heatmaps(heatmaps, weights=None, shape=(224, 224)):
    """Normalize each spatial map, align on the model canvas, fuse, normalize.

    Require all three informative maps; never silently drop a failed model or
    renormalize its weight. This is a visual mixture of per-model logit maps,
    not a gradient of the probability-weighted ensemble score.
    """
    if set(heatmaps) != set(MEMBERS):
        raise ValueError('Exactly three named core heatmaps required')
    maps = [resize_heatmap(heatmaps[name], shape) for name in MEMBERS]
    if any(not m.any() for m in maps):
        raise GradCAMError('Fusion unavailable: at least one individual heatmap has no positive activation')
    values = validate_weights(weights if weights is not None else [1/3]*3)
    return normalize_heatmap(np.tensordot(values, np.stack(maps), axes=1))


def reverse_letterbox(heatmap, original_shape, input_shape=(224, 224)):
    """Exactly mirror the existing preprocessing round/offset rule, then crop.

    Upsample to the input canvas BEFORE removing padding. Interpolation cannot
    recover lost spatial detail or perfectly separate a boundary receptive field.
    """
    h, w = original_shape[:2]
    height, width = input_shape[:2]
    if min(h, w, height, width) < 1:
        raise ValueError('Invalid original/input image geometry')
    scale = min(height/h, width/w)
    new_w, new_h = max(1, round(w*scale)), max(1, round(h*scale))
    top, left = (height-new_h)//2, (width-new_w)//2
    canvas = resize_heatmap(heatmap, (int(height), int(width)))
    crop = canvas[top:top+new_h, left:left+new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def overlay_heatmap(rgb, image_heatmap, alpha=0.4):
    image = np.asarray(rgb)
    heatmap = np.asarray(image_heatmap, dtype=np.float32)
    if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
        or heatmap.shape != image.shape[:2] or not np.isfinite(heatmap).all()
        or (heatmap < 0).any() or (heatmap > 1).any() or not 0 <= alpha <= 1):
        raise ValueError('RGB uint8 image, matching [0,1] map and alpha in [0,1] required')
    # A zero/nonlocalized map must not produce a misleading colored overlay.
    if not heatmap.any():
        return image.copy()
    colors = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.clip(image*(1-alpha) + colors*alpha, 0, 255).astype(np.uint8)
