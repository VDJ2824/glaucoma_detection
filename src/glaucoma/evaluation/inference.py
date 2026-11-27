"""Deterministic image-keyed inference using the unchanged preprocessing module."""
import copy
import os
from pathlib import Path
import numpy as np
import pandas as pd
from glaucoma.data.preprocessing import preprocess_image
from glaucoma.ensemble.predictions import MEMBERS, KEYS, align_predictions, binary_labels, probabilities
from glaucoma.utils.reproducibility import set_random_seed
from .checkpoints import read_manifest, load_verified_model


def manifest_partition(path, cfg, split, flat_image_root=None):
    if split not in ('validation', 'test'):
        raise ValueError('Inference is restricted to validation/test')
    frame = read_manifest(path)
    subset = frame.loc[frame['split'].eq(split)].copy()
    if subset.empty or subset.duplicated(KEYS).any() or subset.image_id.str.strip().eq('').any():
        raise ValueError('Empty or duplicate/missing manifest image keys')
    subset['label'] = binary_labels(subset.label)
    # Test rows/paths/labels are not checked while selecting validation rows.
    root = Path(flat_image_root or cfg['datasets'][cfg['datasets']['selected']]['root']).resolve()
    paths = [(root/Path(value).name).resolve() if flat_image_root else Path(value).resolve() for value in subset.image_path]
    if len(set(paths)) != len(paths) or any(not value.strip() for value in subset.image_path):
        raise ValueError('Colliding/blank physical image paths')
    if any(not p.is_relative_to(root) or not p.is_file() for p in paths):
        raise ValueError('Missing or out-of-root inference image path')
    subset['_physical_path'] = [str(p) for p in paths]
    # Known grouping and image/path leakage can be checked without using other labels.
    other = frame.loc[~frame['split'].eq(split)]
    if set(map(tuple, subset[KEYS].to_numpy())) & set(map(tuple, other[KEYS].to_numpy())):
        raise ValueError('Image key leakage across frozen splits')
    other_paths = {os.path.abspath(root/Path(p).name) if flat_image_root else os.path.abspath(p) for p in other.image_path}
    if set(map(str,paths)) & other_paths:
        raise ValueError('Image path leakage across frozen splits')
    known = subset.loc[subset.patient_id.str.strip().ne('')]
    other_known = other.loc[other.patient_id.str.strip().ne('')]
    groups = set(zip(known.dataset_name, known.patient_id.str.strip()))
    if groups & set(zip(other_known.dataset_name, other_known.patient_id.str.strip())):
        raise ValueError('Known patient leakage across frozen splits')
    return subset.sort_values(KEYS, kind='stable').reset_index(drop=True)


def predict_model(model, frame, cfg, name, split):
    if name not in MEMBERS or not frame['split'].eq(split).all() or split not in ('validation', 'test'):
        raise ValueError('Invalid core model/partition')
    local = copy.deepcopy(cfg)
    local['model']['selected'] = name
    batch_size = local['training']['batch_size']
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive inference batch size required')
    output = []
    for offset in range(0, len(frame), batch_size):
        rows = frame.iloc[offset:offset+batch_size]
        x = np.stack([preprocess_image(value, local, split=split) for value in rows['_physical_path']])
        y = model(x, training=False)
        scores = y.numpy() if hasattr(y, 'numpy') else np.asarray(y)
        if scores.shape != (len(rows),1):
            raise ValueError('Model output shape differs from expected N x 1')
        output.extend(probabilities(scores[:,0]).tolist())
    result = frame[KEYS+['split','label']].copy()
    result['score'] = output
    return result


def predict_partition(cfg, manifest_path, split, checkpoints, flat_image_root=None, expected_graphs=None):
    """Load models sequentially to limit memory, with no training or augmentation."""
    import tensorflow as tf
    frame = manifest_partition(manifest_path, cfg, split, flat_image_root)
    predictions, graphs = {}, {}
    for name in MEMBERS:
        tf.keras.backend.clear_session()
        set_random_seed(cfg['seed'])
        model, graph_hash = load_verified_model(cfg, name, checkpoints[name])
        if expected_graphs is not None and graph_hash != expected_graphs[name]:
            raise ValueError('Loaded model graph differs from the frozen validation model')
        predictions[name] = predict_model(model, frame, cfg, name, split)
        graphs[name] = graph_hash
        del model
    tf.keras.backend.clear_session()
    return align_predictions(predictions, split, frame), predictions, graphs
