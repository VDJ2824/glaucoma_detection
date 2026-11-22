"""Core classifiers consuming already preprocessed images; lazy TensorFlow import."""
from .registry import CORE_APPLICATIONS


def build_model(model_name, weights='imagenet', input_shape=(224, 224, 3), dropout=0.2):
    """Explicit invocation may fetch ImageNet weights. None is entirely offline.

    No extra normalization is applied: ResNet50 expects BGR/mean subtraction,
    MobileNetV2 expects [-1,1], and EfficientNetB0 embeds its own rescaling and
    expects RGB [0,255]. All are provided by the existing data pipeline.
    """
    if model_name not in CORE_APPLICATIONS:
        raise ValueError('Select a core model; VGG experiments are separate')
    if tuple(input_shape) != (224, 224, 3) or dropout != 0.2:
        raise ValueError('Phase 3 requires input (224,224,3) and dropout 0.2')
    if weights not in ('imagenet', None):
        raise ValueError('weights must be imagenet or None')
    import tensorflow as tf
    backbone = getattr(tf.keras.applications, CORE_APPLICATIONS[model_name])(
        include_top=False, weights=weights, input_shape=tuple(input_shape))
    backbone.trainable = False
    inputs = tf.keras.Input(shape=tuple(input_shape), name='preprocessed_image')
    features = backbone(inputs, training=False)  # keep BN statistics fixed in both stages
    features = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pool')(features)
    features = tf.keras.layers.Dropout(dropout, name='head_dropout')(features)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='glaucoma_probability')(features)
    return tf.keras.Model(inputs, outputs, name=model_name)


def get_backbone(model):
    import tensorflow as tf
    backbones = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if len(backbones) != 1:
        raise ValueError('Expected exactly one nested backbone model')
    return backbones[0]


def forward_smoke_test(model):
    """Offline synthetic forward pass only; no dataset access or fit call."""
    import numpy as np
    import tensorflow as tf
    result = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False).numpy()
    if result.shape != (1, 1) or not np.isfinite(result).all() or not ((0 <= result) & (result <= 1)).all():
        raise RuntimeError('Classifier forward-pass check failed')
    return {'input_shape': [1, 224, 224, 3], 'output_shape': list(result.shape), 'status': 'passed',
            'note': 'Synthetic tensor check, not a measured dataset prediction or clinical result'}
