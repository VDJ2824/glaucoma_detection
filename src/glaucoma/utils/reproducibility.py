"""Explicit reproducibility settings; no import-time TensorFlow initialization."""
import os
import random
import numpy as np


def set_random_seed(seed=42, deterministic=True):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    random.seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' if deterministic else '0'
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
