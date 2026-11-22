"""Frozen CSV -> finite tf.data batches using the unmodified iter_batches pipeline."""
import copy
import math
from pathlib import Path
import numpy as np
import pandas as pd
from glaucoma.data.adapters import COLUMNS
from glaucoma.data.loading import iter_batches


def load_training_metadata(path, cfg):
    """Read saved assignments, keep train/validation only; never discover or split.

    Test CSV rows are discarded before label/path checks. No test image is opened
    or stat'ed, and no train/validation decoding occurs until iteration.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not set(COLUMNS).issubset(frame):
        raise ValueError('Frozen CSV requires the six metadata schema columns')
    if not frame.split.isin(['train', 'validation', 'test']).all():
        raise ValueError('Frozen CSV contains missing/unknown splits; prepare it separately')
    frame = frame.loc[frame.split.isin(['train', 'validation']), COLUMNS].copy().reset_index(drop=True)
    if set(frame.split) != {'train', 'validation'}:
        raise ValueError('Both frozen train and validation partitions are required')
    if not frame.label.isin(['0', '1']).all():
        raise ValueError('Training/validation labels must be binary glaucoma annotations')
    if frame.image_id.str.strip().eq('').any() or frame.image_id.duplicated().any():
        raise ValueError('Image IDs must be present and unique across train/validation')
    root = Path(cfg['datasets'][cfg['datasets']['selected']]['root']).resolve()
    paths = []
    for value in frame.image_path:
        path = Path(value).resolve()
        if not value.strip() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Invalid train/validation image path; originals are not relocated automatically')
        paths.append(str(path))
    if len(set(paths)) != len(paths):
        raise ValueError('Repeated image paths in frozen train/validation metadata')
    frame['image_path'] = paths
    frame['patient_id'] = frame.patient_id.str.strip()
    known = frame.patient_id.ne('')
    if cfg['split']['patient_level'] and known.any():
        if not known.all():
            raise ValueError('Partial patient IDs in patient-grouped frozen metadata')
        groups = frame.dataset_name + ':' + frame.patient_id
        if frame.groupby(groups).split.nunique().gt(1).any():
            raise ValueError('Patient leakage across frozen train/validation partitions')
    frame['label'] = frame.label.astype(int)
    return frame


def make_dataset(metadata, cfg, split, start_epoch=0):
    """Finite dataset with final partial batch; no second normalization or caching.

    Training gets a new seeded augmentation stream per iterator/epoch. Validation
    always uses epoch zero. Start_epoch permits epoch-boundary recovery. Keras
    may create extra iterators; exact augmentation replay after interruption is
    not promised, while partitions and per-iterator seeds remain reproducible.
    """
    if split not in ('train', 'validation'):
        raise ValueError('Training datasets may not access the test partition')
    import tensorflow as tf
    cfg = copy.deepcopy(cfg)
    frame = metadata.loc[metadata.split.eq(split)].copy()
    if frame.empty:
        raise ValueError(f'Empty {split} partition')
    if type(start_epoch) is not int or start_epoch < 0:
        raise ValueError('Invalid initial epoch')
    epoch = start_epoch
    def generator():
        nonlocal epoch
        current = epoch if split == 'train' else 0
        if split == 'train':
            epoch += 1
        for images, labels in iter_batches(frame, cfg, split, epoch=current):
            yield images.astype(np.float32, copy=False), labels.astype(np.float32).reshape(-1, 1)
    shape = tuple(cfg['model']['input_shape'])
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec((None, *shape), tf.float32), tf.TensorSpec((None, 1), tf.float32)))
    steps = math.ceil(len(frame) / cfg['training']['batch_size'])
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(steps))
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).prefetch(1)
