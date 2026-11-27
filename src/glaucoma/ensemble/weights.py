"""Predefined finite simplex search on validation predictions only."""
import numpy as np
import pandas as pd
from glaucoma.evaluation.metrics import BCE_EPSILON, binary_crossentropy
from .predictions import MEMBERS, binary_labels, check_threshold, score_matrix, validate_weights


def search_protocol(grid_denominator=20, threshold=0.5):
    if type(grid_denominator) is not int or not 1 <= grid_denominator <= 100:
        raise ValueError('Grid denominator must be an integer from 1 to 100')
    return {'members': list(MEMBERS), 'grid_denominator': grid_denominator,
            'include_exact_equal_weights': True, 'criterion': 'minimum_validation_binary_crossentropy',
            'tie_tolerance': 1e-12, 'tie_break': 'closest_to_equal_then_lexicographic',
            'threshold': check_threshold(threshold), 'threshold_rule': 'score_greater_than_or_equal',
            'threshold_selection': 'predefined_not_tuned', 'bce_epsilon': BCE_EPSILON}


def optimize_weights(aligned, protocol=None):
    protocol = protocol or search_protocol()
    if protocol != search_protocol(protocol['grid_denominator'], protocol['threshold']):
        raise ValueError('Unsupported or modified predefined protocol')
    if not aligned['split'].eq('validation').all() or aligned.empty:
        raise ValueError('Weight optimization may use validation predictions only')
    y = binary_labels(aligned.label)
    if np.unique(y).size != 2:
        raise ValueError('Both classes required for validation weight selection')
    scores = score_matrix(aligned[list(MEMBERS)].to_numpy())
    n = protocol['grid_denominator']
    candidates = [(i/n, j/n, (n-i-j)/n) for i in range(n+1) for j in range(n-i+1)]
    if n % 3:
        candidates.append((1/3, 1/3, 1/3))
    rows = []
    for weights in candidates:
        weights = validate_weights(weights)
        loss = binary_crossentropy(y, np.clip(scores @ weights, 0, 1))
        rows.append({**dict(zip(MEMBERS, weights)), 'validation_binary_crossentropy': loss,
                     'distance_to_equal': float(sum((value-1/3)**2 for value in weights))})
    minimum = min(row['validation_binary_crossentropy'] for row in rows)
    eligible = [i for i, row in enumerate(rows) if row['validation_binary_crossentropy'] <= minimum+protocol['tie_tolerance']]
    selected = min(eligible, key=lambda i: (rows[i]['distance_to_equal'], tuple(rows[i][name] for name in MEMBERS)))
    for i, row in enumerate(rows):
        row['selected'] = i == selected
    return {name: rows[selected][name] for name in MEMBERS}, pd.DataFrame(rows)
