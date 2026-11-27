"""Inference-only binary logit Grad-CAM through the nested core backbone.

No normalization is performed here: callers pass the existing preprocessed tensor.
A non-glaucoma target explains -z, and a glaucoma target explains z, where p=sigmoid(z).
"""
from dataclasses import dataclass
import numpy as np
from glaucoma.models.factory import get_backbone

TARGET_LAYERS = {'mobilenet_v2': 'Conv_1', 'efficientnet_b0': 'top_conv',
                 'resnet50': 'conv5_block3_3_conv'}


class GradCAMError(ValueError):
    """Disconnected gradients, nonfinite tensors or an incompatible model."""


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    probability: float
    target_class: int
    target_layer: str
    status: str
    gradient_max_abs: float
    target_logit: float


def normalize_heatmap(heatmap):
    """Normalize a nonnegative spatial map; preserve an explicitly empty map."""
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all() or (array < 0).any():
        raise GradCAMError('Invalid heatmap: expected finite nonnegative H x W values')
    maximum = float(array.max())
    return array / maximum if maximum > 0 else np.zeros_like(array)


class GradCAM:
    """Reusable probe; shares weights without changing trainability or compiling.

    The probe exposes both the internal convolution and the backbone output in
    one connected graph. Replaying the unchanged inference head outside that
    nested graph avoids Keras 3's disconnected outer-input/internal-layer graph.
    Input watching keeps gradients available even for a fully frozen backbone.
    """
    def __init__(self, model, model_name, target_layer=None):
        import tensorflow as tf
        if model_name not in TARGET_LAYERS or model.name != model_name:
            raise GradCAMError('Expected a named core classifier')
        self.model = model
        self.backbone = get_backbone(model)
        self.layer_name = target_layer or TARGET_LAYERS[model_name]
        try:
            layer = self.backbone.get_layer(self.layer_name)
        except ValueError as exc:
            raise GradCAMError(f'Missing spatial target convolution: {self.layer_name}') from exc
        if not isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or len(layer.output.shape) != 4:
            raise GradCAMError('Target must be a spatial convolution inside the backbone')
        index = model.layers.index(self.backbone)
        self.head = model.layers[index+1:]
        if (len(self.head) != 3 or not isinstance(self.head[0], tf.keras.layers.GlobalAveragePooling2D)
            or not isinstance(self.head[1], tf.keras.layers.Dropout)
            or not isinstance(self.head[2], tf.keras.layers.Dense)
            or self.head[2].units != 1 or self.head[2].activation != tf.keras.activations.sigmoid):
            raise GradCAMError('Expected GAP / Dropout / binary sigmoid head')
        self.probe = tf.keras.Model(self.backbone.inputs, [layer.output, self.backbone.output])

    def explain(self, preprocessed_image, target_class=1):
        import tensorflow as tf
        if type(target_class) is not int or target_class not in (0, 1):
            raise GradCAMError('target_class must be integer 0 or 1')
        array = np.asarray(preprocessed_image, dtype=np.float32)
        if array.shape == tuple(self.model.input_shape[1:]):
            array = array[None]
        if array.shape != (1, *self.model.input_shape[1:]) or not np.isfinite(array).all():
            raise GradCAMError('One finite preprocessed image matching the model input is required')
        tensor = tf.convert_to_tensor(array)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            activations, features = self.probe(tensor, training=False)
            for layer in self.head[:-1]:
                features = layer(features, training=False)
            dense = self.head[-1]
            logit = tf.linalg.matmul(features, dense.kernel)
            if dense.use_bias:
                logit = tf.nn.bias_add(logit, dense.bias)
            target = logit[0, 0] if target_class == 1 else -logit[0, 0]
        gradients = tape.gradient(target, activations)
        if gradients is None:
            raise GradCAMError('Disconnected target-layer gradients')
        a, g = activations.numpy()[0], gradients.numpy()[0]
        if not np.isfinite(a).all() or not np.isfinite(g).all() or not np.isfinite(logit.numpy()).all():
            raise GradCAMError('Nonfinite activations, gradients or logits')
        magnitude = float(np.max(np.abs(g)))
        raw = np.maximum(np.sum(a * g.mean(axis=(0, 1)), axis=-1), 0)
        heatmap = normalize_heatmap(raw)
        status = 'zero_gradient' if magnitude == 0 else 'no_positive_activation' if not heatmap.any() else 'ok'
        return GradCAMResult(heatmap, float(tf.math.sigmoid(logit)[0, 0]), target_class,
                             self.layer_name, status, magnitude, float(target))
