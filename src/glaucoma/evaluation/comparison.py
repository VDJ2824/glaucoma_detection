"""Metrics, plots and disagreement artifacts from actual aligned predictions only."""
from pathlib import Path
import numpy as np
import pandas as pd
from glaucoma.ensemble.predictions import MEMBERS, KEYS, voting_predictions, check_threshold, score_matrix, binary_labels
from glaucoma.training.artifacts import write_json
from .metrics import classification_metrics, roc_points


def evaluate_predictions(aligned, weights=None, threshold=0.5, include_ensembles=True):
    threshold = check_threshold(threshold)
    scores = aligned[list(MEMBERS)].to_numpy(dtype=float)
    outputs = {name: {'labels': (scores[:,i] >= threshold).astype(int), 'scores': scores[:,i]} for i,name in enumerate(MEMBERS)}
    if include_ensembles:
        outputs.update(voting_predictions(scores, weights, threshold))
    metrics = {name: classification_metrics(aligned.label.to_numpy(), data['labels'], data['scores']) for name,data in outputs.items()}
    return outputs, metrics


def disagreement_analysis(aligned, threshold=0.5):
    threshold = check_threshold(threshold)
    votes = (score_matrix(aligned[list(MEMBERS)].to_numpy(dtype=float)) >= threshold).astype(int)
    y = binary_labels(aligned.label.to_numpy())
    rows = aligned[KEYS+['split','label']].copy()
    for i,name in enumerate(MEMBERS):
        rows[f'{name}_label'] = votes[:,i]
    rows['positive_votes'] = votes.sum(axis=1)
    rows['unanimous'] = (rows.positive_votes == 0) | (rows.positive_votes == 3)
    rows['models_correct'] = (votes == y[:,None]).sum(axis=1)
    pairwise = []
    for i,a in enumerate(MEMBERS):
        for j in range(i+1,3):
            count = int((votes[:,i] != votes[:,j]).sum())
            pairwise.append({'model_a':a,'model_b':MEMBERS[j], 'disagree_count':count, 'disagree_fraction':count/len(rows),
                             'both_wrong_count':int(((votes[:,i] != y) & (votes[:,j] != y)).sum())})
    return rows, {'n':len(rows), 'unanimous_count':int(rows.unanimous.sum()),
                  'nonunanimous_count':int((~rows.unanimous).sum()),
                  'nonunanimous_fraction':float((~rows.unanimous).mean()), 'pairwise':pairwise}


def save_evaluation(aligned, weights, threshold, directory, include_ensembles=True):
    """Save private score tables, confusion matrices and score-based ROC curves."""
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs, metrics = evaluate_predictions(aligned, weights, threshold, include_ensembles)
    aligned.to_csv(directory/'aligned_predictions.csv', index=False)
    combined = aligned.copy()
    for name,result in outputs.items():
        combined[f'{name}_label'] = result['labels']
        if result['scores'] is not None:
            combined[f'{name}_score'] = result['scores']
    combined.to_csv(directory/('all_strategy_predictions.csv' if include_ensembles else 'individual_strategy_predictions.csv'), index=False)
    write_json(directory/'metrics.json', {'split':str(aligned['split'].iloc[0]), 'threshold':threshold, 'methods':metrics})
    table = pd.DataFrame([{'method':name, **{key:value for key,value in data.items() if key not in ('confusion_matrix','undefined_metrics','score_metrics_note')}} for name,data in metrics.items()])
    table.to_csv(directory/'metrics_comparison.csv', index=False)
    if include_ensembles:
        disagreements, summary = disagreement_analysis(aligned, threshold)
        disagreements.to_csv(directory/'prediction_disagreement.csv', index=False)
        write_json(directory/'disagreement_summary.json', summary)
    nrows = (len(metrics)+2)//3
    fig, axes = plt.subplots(nrows,3,figsize=(13,4*nrows),squeeze=False)
    for ax,(name,data) in zip(axes.flat,metrics.items()):
        cm = np.asarray(data['confusion_matrix'])
        ax.imshow(cm,cmap='Blues',vmin=0)
        for (i,j),value in np.ndenumerate(cm):
            ax.text(j,i,str(value),ha='center',va='center',color='white' if value > cm.max()/2 else 'black')
        ax.set(title=name,xlabel='Predicted',ylabel='True',xticks=[0,1],yticks=[0,1])
    fig.tight_layout();fig.savefig(directory/'confusion_matrices.png',dpi=150);plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,6))
    all_points=[]
    count=0
    for name,result in outputs.items():
        if result['scores'] is None:
            continue
        points=roc_points(aligned.label.to_numpy(),result['scores'])
        if points:
            ax.plot([p['false_positive_rate'] for p in points],[p['true_positive_rate'] for p in points],label=f"{name} (AUC {metrics[name]['roc_auc']:.4f})")
            all_points.extend([{'method':name,**point} for point in points]);count+=1
    ax.plot([0,1],[0,1],'--',color='grey')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Score-based ROC' + (' (hard majority excluded)' if include_ensembles else ' — individual models'))
    if count:ax.legend(fontsize=8)
    else:ax.text(.1,.5,'ROC undefined: only one ground-truth class')
    fig.tight_layout();fig.savefig(directory/'roc_curves.png',dpi=150);plt.close(fig)
    pd.DataFrame(all_points,columns=['method','false_positive_rate','true_positive_rate','threshold']).to_csv(directory/'roc_points.csv',index=False)
    return metrics


def save_individual_evaluation(aligned, threshold, directory):
    """Notebook 04 report: only the three CNNs, without voting or weight search."""
    return save_evaluation(aligned, None, threshold, directory, include_ensembles=False)
