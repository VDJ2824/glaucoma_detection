"""Reproducible stratified original-image splits, preserving patient groups."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SPLITS = ('train', 'validation', 'test')


def split_metadata(frame, ratios=None, seed=42, mode='generated', stratify=True, patient_level=True):
    ratios = ratios or dict(zip(SPLITS, (0.70, 0.15, 0.15)))
    if set(ratios) != set(SPLITS) or any(v <= 0 for v in ratios.values()) or not np.isclose(sum(ratios.values()), 1):
        raise ValueError('Invalid split ratios')
    data = frame.copy().fillna('').reset_index(drop=True)
    labels = pd.to_numeric(data.label, errors='coerce')
    if data.empty or labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError('Nonempty metadata with binary glaucoma labels required')
    data['label'] = labels.astype(int)
    known = data.patient_id.astype(str).str.strip().ne('')
    # IDs are scoped to their originating dataset, never invented.
    groups = data.dataset_name.astype(str) + ':' + data.patient_id.astype(str)
    if patient_level and known.any() and not known.all():
        raise ValueError('Partial patient IDs: resolve missing IDs or explicitly disable patient grouping')
    if mode == 'official':
        if not data.split.isin(SPLITS).all():
            raise ValueError('Official split required for every image')
        if patient_level and known.all() and data.groupby(groups)['split'].nunique().gt(1).any():
            raise ValueError('Patient leakage across official splits')
    elif mode == 'generated':
        if data.split.ne('').any():
            raise ValueError('Existing splits present: choose official mode; do not silently overwrite')
        if patient_level and known.all():
            # Grouped stratification is approximate: search deterministic candidates
            # by size and label proportions, never by model performance.
            best, best_score = None, float('inf')
            outer = GroupShuffleSplit(n_splits=256, train_size=ratios['train'], random_state=seed)
            try:
                for attempt, (train, rest) in enumerate(outer.split(data, groups=groups)):
                    inner = GroupShuffleSplit(n_splits=1, train_size=ratios['validation'] / (ratios['validation'] + ratios['test']), random_state=seed+attempt+1)
                    try:
                        val_idx, test_idx = next(inner.split(rest, groups=groups.iloc[rest]))
                    except ValueError:
                        continue
                    parts = (train, rest[val_idx], rest[test_idx])
                    if stratify and any(data.iloc[idx].label.nunique() != 2 for idx in parts):
                        continue
                    score = sum(abs(len(idx)/len(data)-ratios[name]) for name, idx in zip(SPLITS, parts))
                    if stratify:
                        score += sum(abs(data.iloc[idx].label.mean()-data.label.mean()) for idx in parts)
                    if score < best_score:
                        best, best_score = parts, score
            except ValueError as exc:
                raise ValueError('Insufficient patient groups for splitting') from exc
            if best is None:
                raise ValueError('Cannot form nonempty grouped stratified splits; obtain more groups or adjust configuration')
            parts = best
        else:
            indices = np.arange(len(data))
            try:
                train, rest = train_test_split(indices, train_size=round(len(data) * ratios['train']), random_state=seed, stratify=data.label if stratify else None)
                val, test = train_test_split(rest, train_size=round(len(data) * ratios['validation']), random_state=seed, stratify=data.iloc[rest].label if stratify else None)
                parts = (train, val, test)
            except ValueError as exc:
                raise ValueError('Insufficient samples per class for requested stratification') from exc
        for name, indices in zip(SPLITS, parts):
            data.loc[indices, 'split'] = name
    else:
        raise ValueError('Unknown split mode')
    if set(data.split) != set(SPLITS):
        raise ValueError('All three splits must be nonempty')
    if stratify and any(part.label.nunique() != 2 for _, part in data.groupby('split')):
        raise ValueError('Each stratified split must contain both classes')
    return data
