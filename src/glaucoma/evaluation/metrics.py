"""Binary metrics with explicit undefined cases and no fabricated probabilities."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from glaucoma.ensemble.predictions import binary_labels, probabilities

BCE_EPSILON = 1e-7


def binary_crossentropy(labels, scores):
    y = binary_labels(labels)
    p = probabilities(scores)
    if p.shape != y.shape:
        raise ValueError('Score and label shapes differ')
    p = np.clip(p, BCE_EPSILON, 1-BCE_EPSILON)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log1p(-p)))


def classification_metrics(labels, predicted, scores=None):
    y, pred = binary_labels(labels), binary_labels(predicted)
    if y.shape != pred.shape:
        raise ValueError('Prediction/label lengths differ')
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    undefined = []
    def ratio(name, numerator, denominator):
        if not denominator:
            undefined.append(name)
            return None
        return float(numerator/denominator)
    result = {'n': len(y), 'accuracy': float((tn+tp)/len(y)),
              'precision': ratio('precision', tp, tp+fp), 'recall': ratio('recall', tp, tp+fn),
              'specificity': ratio('specificity', tn, tn+fp), 'f1': ratio('f1', 2*tp, 2*tp+fp+fn),
              'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
              'confusion_matrix': [[tn, fp], [fn, tp]], 'roc_auc': None, 'binary_crossentropy': None,
              'undefined_metrics': undefined, 'score_metrics_note': None}
    if scores is None:
        result['score_metrics_note'] = 'Hard voting has no probability score: ROC-AUC/BCE not applicable.'
    else:
        p = probabilities(scores)
        result['binary_crossentropy'] = binary_crossentropy(y, p)
        if np.unique(y).size == 2:
            result['roc_auc'] = float(roc_auc_score(y, p))
        else:
            result['undefined_metrics'].append('roc_auc')
            result['score_metrics_note'] = 'ROC-AUC undefined for a single-class partition.'
    return result


def roc_points(labels, scores):
    y, p = binary_labels(labels), probabilities(scores)
    if y.shape != p.shape:
        raise ValueError('Score/label lengths differ')
    if np.unique(y).size < 2:
        return []
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    return [{'false_positive_rate': float(f), 'true_positive_rate': float(t),
             'threshold': float(s) if np.isfinite(s) else None} for f, t, s in zip(fpr, tpr, thresholds)]
