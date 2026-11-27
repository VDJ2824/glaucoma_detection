"""Strict core-score alignment and hard/soft voting; no inference or fitting."""
import numpy as np
import pandas as pd
from glaucoma.models.registry import CORE_APPLICATIONS

MEMBERS = tuple(CORE_APPLICATIONS)
KEYS = ['dataset_name', 'image_id']


def binary_labels(values):
    a = np.asarray(values)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError('Nonempty one-dimensional labels required')
    try:
        numbers = a.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Labels must be binary 0/1') from exc
    if not np.isfinite(numbers).all() or not np.isin(numbers, [0, 1]).all():
        raise ValueError('Labels must be binary 0/1')
    return numbers.astype(int)


def probabilities(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0 or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError('Scores must be finite probabilities in [0,1]')
    return a


def validate_weights(weights):
    if isinstance(weights, dict):
        if set(weights) != set(MEMBERS):
            raise ValueError('Weights must name exactly the three core models')
        weights = [weights[name] for name in MEMBERS]
    a = np.asarray(weights, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or (a < 0).any() or not np.isclose(a.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError('Three nonnegative weights summing to one required')
    return a / a.sum()


def check_threshold(threshold):
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must be strictly between zero and one')
    return float(threshold)


def score_matrix(scores):
    a = probabilities(scores)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Expected aligned N x 3 core probability matrix')
    return a


def voting_predictions(scores, weights, threshold=0.5):
    a = score_matrix(scores)
    threshold = check_threshold(threshold)
    votes = (a >= threshold).astype(int)
    equal = a.mean(axis=1)
    weighted = np.clip(a @ validate_weights(weights), 0, 1)  # convex-combination roundoff only
    return {'majority_voting': {'labels': (votes.sum(axis=1) >= 2).astype(int), 'scores': None},
            'equal_soft_voting': {'labels': (equal >= threshold).astype(int), 'scores': equal},
            'weighted_soft_voting': {'labels': (weighted >= threshold).astype(int), 'scores': weighted}}


def align_predictions(predictions, split, reference=None):
    """Join by dataset/image ID, fail on missing/extra rows or label disagreement.

    Scores are never aligned by incidental CSV row order. Reference establishes
    frozen-manifest ordering when provided. Patient IDs/paths are not output.
    """
    if set(predictions) != set(MEMBERS) or split not in ('validation', 'test'):
        raise ValueError('Exactly three core models and validation/test split required')
    checked = {}
    for name in MEMBERS:
        frame = predictions[name].copy()
        required = {*KEYS, 'split', 'label', 'score'}
        if not required.issubset(frame) or frame.empty or not frame['split'].eq(split).all():
            raise ValueError(f'{name}: wrong schema or split')
        for key in KEYS:
            frame[key] = frame[key].astype(str)
            if frame[key].str.strip().eq('').any():
                raise ValueError('Missing image key')
        if frame.duplicated(KEYS).any():
            raise ValueError('Duplicate image keys')
        frame['label'] = binary_labels(frame.label)
        frame['score'] = probabilities(frame.score.to_numpy())
        checked[name] = frame
    base = checked[MEMBERS[0]][KEYS + ['split', 'label']].copy() if reference is None else reference[KEYS + ['split', 'label']].copy()
    base['label'] = binary_labels(base.label)
    for key in KEYS:
        base[key] = base[key].astype(str)
    if base.duplicated(KEYS).any() or not base['split'].eq(split).all():
        raise ValueError('Invalid reference manifest keys/split')
    expected = set(map(tuple, base[KEYS].to_numpy()))
    for name, frame in checked.items():
        if set(map(tuple, frame[KEYS].to_numpy())) != expected:
            raise ValueError('Missing or extra model prediction rows')
        merged = base[KEYS + ['label']].merge(frame[KEYS + ['label', 'score']], on=KEYS, validate='one_to_one', sort=False)
        if not np.array_equal(merged.label_x, merged.label_y):
            raise ValueError('Ground-truth labels disagree between models/reference')
        base[name] = merged.score.to_numpy()
    return base.reset_index(drop=True)
