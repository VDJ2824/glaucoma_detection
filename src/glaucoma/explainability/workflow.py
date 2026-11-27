"""Private post-hoc analysis of verified Phase 4 predictions and selected models."""
import copy
import json
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from glaucoma.ensemble.predictions import MEMBERS, KEYS, align_predictions, voting_predictions
from glaucoma.evaluation.checkpoints import (sha256_file, read_manifest, verify_selected_checkpoints,
                                            load_verified_model)
from glaucoma.evaluation.workflow import load_frozen, inference_contract
from glaucoma.evaluation.inference import manifest_partition
from glaucoma.training.artifacts import write_json
from glaucoma.utils.reproducibility import set_random_seed
from .gradcam import GradCAM, GradCAMError
from .fusion import resize_heatmap, fuse_heatmaps, reverse_letterbox, overlay_heatmap
from glaucoma.data.preprocessing import preprocess_image


def select_examples(aligned, weights, threshold, *, method='weighted_soft_voting', per_category=2):
    """Stable key-ordered examples, without adding or changing any predictions.

    Correct/incorrect refer to the specified existing classifier, not the target
    class explained. Categories can overlap; selected keys are deduplicated.
    """
    if type(per_category) is not int or not 1 <= per_category <= 50:
        raise ValueError('per_category must be an integer from 1 to 50')
    individuals = {n: aligned[KEYS+['split','label',n]].rename(columns={n:'score'}) for n in MEMBERS}
    if aligned.empty or aligned['split'].nunique() != 1:
        raise ValueError('One nonempty validation/test prediction partition required')
    frame = align_predictions(individuals, str(aligned['split'].iloc[0])).sort_values(KEYS, kind='stable').reset_index(drop=True)
    scores = frame[list(MEMBERS)].to_numpy()
    outputs = voting_predictions(scores, weights, threshold)
    if method in MEMBERS:
        predicted = (frame[method].to_numpy() >= threshold).astype(int)
    elif method in outputs:
        predicted = outputs[method]['labels']
    else:
        raise ValueError('Unknown example-selection classifier')
    votes = scores >= threshold
    masks = {'correct': predicted == frame.label.to_numpy(),
             'incorrect': predicted != frame.label.to_numpy(),
             'disagreement': np.any(votes != votes[:, :1], axis=1)}
    categories = {}
    counts = {}
    for category, mask in masks.items():
        indices = np.flatnonzero(mask)
        counts[category] = {'available': int(len(indices)), 'selected': int(min(len(indices), per_category))}
        for index in indices[:per_category]:
            categories.setdefault(int(index), []).append(category)
    selected = frame.iloc[sorted(categories)].copy()
    selected['categories'] = [categories[int(i)] for i in selected.index]
    return selected.reset_index(drop=True), {'method':method, 'ordering':'dataset_name,image_id ascending',
                                            'per_category':per_category, 'categories':counts}


def load_analysis_context(cfg, manifest_path, evaluation_directory, split='validation', *,
                          allow_test=False, flat_image_root=None):
    """Read verified private prediction records, never decode images or infer.

    Test analysis requires explicit opt-in AND an existing completion record.
    There is no model/weight selection, threshold tuning or test evaluation here.
    """
    if split not in ('validation', 'test'):
        raise ValueError('Explain validation or previously evaluated test only')
    if split == 'test' and allow_test is not True:
        raise PermissionError('Test explanations require explicit post-hoc opt-in')
    directory = Path(evaluation_directory)
    frozen = load_frozen(directory)
    if sha256_file(manifest_path) != frozen['manifest_sha256']:
        raise ValueError('Frozen dataset manifest changed')
    context = inference_contract(cfg, flat_image_root)
    # Physical image storage may move, but the manifest/preprocessing stay frozen.
    portable = lambda d: {k:v for k,v in d.items() if k not in ('dataset_root', 'flat_image_root')}
    if portable(context) != portable(frozen['inference_contract']):
        raise ValueError('Frozen inference methodology changed')
    if split == 'test':
        complete_path = directory/'test_complete.json'
        if not complete_path.is_file():
            raise ValueError('Complete Phase 4 test evaluation before post-hoc explanations')
        complete = json.loads(complete_path.read_text())
        if complete.get('frozen_seal') != frozen['seal_sha256'] or complete.get('partition') != 'test':
            raise ValueError('Test completion does not match the frozen protocol')
    path = directory/split/'aligned_predictions.csv'
    saved = pd.read_csv(path, dtype={'image_id':str, 'dataset_name':str}, float_precision='round_trip')
    reference = read_manifest(manifest_path)
    reference = reference.loc[reference['split'].eq(split)].sort_values(KEYS, kind='stable').reset_index(drop=True)
    individuals = {n:saved[KEYS+['split','label',n]].rename(columns={n:'score'}) for n in MEMBERS}
    aligned = align_predictions(individuals, split, reference)
    if split == 'test' and complete.get('n') != len(aligned):
        raise ValueError('Test completion count mismatch')
    return aligned, frozen, {'source_predictions':str(path.resolve()), 'source_predictions_sha256':sha256_file(path),
                              'source_prediction_integrity':'sealed validation' if split == 'validation' else 'unsealed Phase 4 test CSV; selected scores rechecked during explanation'}


def _private_directory(cfg, name, evaluation_directory, flat_image_root):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Use a simple explanation run name')
    root = (Path(cfg['artifacts'].get('heatmaps', Path(cfg['artifacts']['root'])/'heatmaps')).expanduser()/'phase5'/name).resolve()
    repo = Path(__file__).resolve().parents[3]
    if root.is_relative_to(repo) and not root.is_relative_to(repo/'artifacts'):
        raise ValueError('Private image-linked outputs inside this repository must stay under ignored artifacts/')
    protected = [cfg['datasets'][cfg['datasets']['selected']]['root'], evaluation_directory,
                 *cfg.get('checkpoints', {}).values()]
    if flat_image_root:
        protected.append(flat_image_root)
    if any(root.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(root) for p in protected):
        raise ValueError('Explanation outputs must be separate from images and Phase 4 source records')
    if root.exists():
        raise FileExistsError('Use a fresh private explanation run; existing artifacts are preserved')
    return root


def _checkpoint_equivalent(actual, frozen):
    # Checkpoint relocation changes physical paths only; bytes/contracts must match.
    for name in MEMBERS:
        for key in ('model_name','checkpoint_sha256','contract_hash','training_manifest_sha256','input_shape','preprocessing','class_mapping'):
            if actual[name][key] != frozen[name][key]:
                raise ValueError(f'{name}: checkpoint provenance changed: {key}')
        for key in ('stage','epoch','val_loss','complete','contract_hash'):
            if actual[name]['selection'][key] != frozen[name]['selection'][key]:
                raise ValueError(f'{name}: selected checkpoint changed')


def _write_rgb(path, image):
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Cannot write private image artifact: {path}')


def save_comparative_panel(rgb, overlays, metadata, path):
    """Six panels: original, three individual maps and two fused maps."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(15, 10))
    FigureCanvasAgg(fig)  # headless local/hosted rendering without altering notebook backend
    axes = fig.subplots(2, 3)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"Original — ground truth: {metadata['ground_truth']}\nTarget: {metadata['target_class']} ({metadata['target_name']})")
    for ax, name in zip(list(axes.flat)[1:], [*MEMBERS, 'equal_heatmap_fusion', 'weighted_heatmap_fusion']):
        data = metadata['methods'][name]
        ax.imshow(overlays.get(name, rgb))
        ax.set_title(f"{name}\np(glaucoma)={data['score']:.5f}; predicted={data['predicted_class']}\n{data['status']}", fontsize=10)
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle('Post-hoc Grad-CAM / heatmap fusion — not segmentation or clinical evidence', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    fig.clear()


def run_explanations(cfg, manifest_path, evaluation_directory, *, enabled=False, split='validation',
                     allow_test=False, name='default', per_category=2, method='weighted_soft_voting',
                     target_classes=(0, 1), checkpoint_roots=None, expected_records=None,
                     flat_image_root=None, score_atol=1e-5, score_rtol=1e-4):
    """Read-only models + private outputs. No training, resplitting or optimization."""
    if enabled is not True:
        raise PermissionError('Grad-CAM disabled; explicitly pass enabled=True')
    if not target_classes or any(type(c) is not int or c not in (0, 1) for c in target_classes) or len(set(target_classes)) != len(target_classes):
        raise ValueError('Select unique binary target classes')
    if not (0 <= score_atol <= 1e-4 and 0 <= score_rtol <= 1e-3):
        raise ValueError('Score tolerances must be small, finite and explicit')
    root = _private_directory(cfg, name, evaluation_directory, flat_image_root)
    aligned, frozen, provenance = load_analysis_context(cfg, manifest_path, evaluation_directory, split,
                                                      allow_test=allow_test, flat_image_root=flat_image_root)
    selected, selection = select_examples(aligned, frozen['weights'], frozen['threshold'], method=method, per_category=per_category)
    checkpoints = verify_selected_checkpoints(cfg, manifest_path, expected_records, checkpoint_roots)
    _checkpoint_equivalent(checkpoints, frozen['checkpoints'])
    if checkpoint_roots and any(root.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(root)
                                for p in checkpoint_roots.values()):
        raise ValueError('Explanation outputs must be separate from checkpoint storage')
    # Validate only this partition's physical image mapping, using unchanged rules.
    physical = manifest_partition(manifest_path, cfg, split, flat_image_root)
    chosen = selected.merge(physical[KEYS+['_physical_path']], on=KEYS, validate='one_to_one')
    root.mkdir(parents=True)
    write_json(root/'selected_examples.json', selected.to_dict(orient='records'))
    write_json(root/'analysis.json', {'status':'started', 'privacy':'PRIVATE: image-linked artifacts; do not commit or send to Gemini',
        'split':split, 'post_hoc_only':True, 'frozen_seal':frozen['seal_sha256'], 'frozen_weights':frozen['weights'],
        'threshold':frozen['threshold'], 'target_classes':list(target_classes), 'target_definition':'z for class 1, -z for class 0; p=sigmoid(z)',
        'selection':selection, **provenance, 'score_atol':score_atol, 'score_rtol':score_rtol,
        'fusion':'individually max-normalized maps aligned on input canvas, convex fusion, max-normalize; not direct ensemble-score Grad-CAM',
        'checkpoints':checkpoints, 'model_graph_sha256':frozen['model_graph_sha256'],
        'inference_contract':frozen['inference_contract']})
    import tensorflow as tf
    records = {}
    for i, row in chosen.iterrows():
        image = cv2.imread(row['_physical_path'], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode selected image')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for target in target_classes:
            key = (int(i), target)
            directory = root/f'example_{i:03d}'/f'class_{target}'
            directory.mkdir(parents=True)
            records[key] = {'directory':directory, 'rgb':rgb, 'maps':{}, 'overlays':{}, 'metadata':{
                'dataset_name':row.dataset_name, 'image_id':row.image_id, 'split':split,
                'ground_truth':int(row.label), 'categories':row.categories, 'target_class':target,
                'original_shape':list(rgb.shape), 'input_shape':cfg['model']['input_shape'],
                'source_image_sha256':sha256_file(row['_physical_path']),
                'target_name':'glaucoma' if target else 'non_glaucoma', 'methods':{}}}
    # Sequential model loading limits memory; one probe per core, all examples/classes.
    for model_name in MEMBERS:
        tf.keras.backend.clear_session()
        set_random_seed(cfg['seed'])
        model, graph = load_verified_model(cfg, model_name, checkpoints[model_name])
        if graph != frozen['model_graph_sha256'][model_name]:
            raise ValueError('Loaded model graph differs from Phase 4')
        cam = GradCAM(model, model_name)
        local = copy.deepcopy(cfg)
        local['model']['selected'] = model_name
        for i, row in chosen.iterrows():
            tensor = preprocess_image(row['_physical_path'], local, split=split)
            for target in target_classes:
                record = records[(int(i), target)]
                try:
                    result = cam.explain(tensor, target)
                except GradCAMError as exc:
                    # Disconnected/nonfinite results are fatal, not silently colored.
                    raise GradCAMError(f'{model_name}/example_{i}/class_{target}: {exc}') from exc
                if not np.isclose(result.probability, row[model_name], atol=score_atol, rtol=score_rtol):
                    raise ValueError('Recomputed model score differs from saved Phase 4 prediction; investigate without retuning')
                canvas = resize_heatmap(result.heatmap, tuple(cfg['model']['input_shape'][:2]))
                original = reverse_letterbox(canvas, record['rgb'].shape, cfg['model']['input_shape'])
                record['maps'][model_name] = canvas
                overlay = overlay_heatmap(record['rgb'], original)
                record['overlays'][model_name] = overlay
                np.save(record['directory']/f'{model_name}_native.npy', result.heatmap)
                np.save(record['directory']/f'{model_name}_image.npy', original)
                _write_rgb(record['directory']/f'{model_name}_overlay.png', overlay)
                record['metadata']['methods'][model_name] = {'score':float(row[model_name]), 'recomputed_score':result.probability,
                    'predicted_class':int(row[model_name] >= frozen['threshold']),
                    'status':'padding_only_activation' if result.status == 'ok' and not original.any() else result.status,
                    'target_layer':result.target_layer, 'target_logit':result.target_logit, 'gradient_max_abs':result.gradient_max_abs}
        del cam, model
    tf.keras.backend.clear_session()
    for (i, target), record in records.items():
        row = chosen.iloc[i]
        scores = np.asarray([row[n] for n in MEMBERS], dtype=float)
        votes = voting_predictions(scores[None], frozen['weights'], frozen['threshold'])
        informative = all(record['metadata']['methods'][n]['status'] == 'ok' for n in MEMBERS)
        for name, score_name, weights in [('equal_heatmap_fusion','equal_soft_voting',None),
                                          ('weighted_heatmap_fusion','weighted_soft_voting',frozen['weights'])]:
            score = float(votes[score_name]['scores'][0])
            status = 'ok' if informative else 'unavailable: at least one individual map is noninformative'
            record['metadata']['methods'][name] = {'score':score, 'predicted_class':int(score >= frozen['threshold']), 'status':status,
                                                  'interpretation':'heatmap fusion; not ensemble-score Grad-CAM'}
            if informative:
                fused = fuse_heatmaps(record['maps'], weights, tuple(cfg['model']['input_shape'][:2]))
                original = reverse_letterbox(fused, record['rgb'].shape, cfg['model']['input_shape'])
                np.save(record['directory']/f'{name}_canvas.npy', fused)
                np.save(record['directory']/f'{name}_image.npy', original)
                overlay = overlay_heatmap(record['rgb'], original)
                record['overlays'][name] = overlay
                _write_rgb(record['directory']/f'{name}_overlay.png', overlay)
        write_json(record['directory']/'explanation.json', record['metadata'])
        save_comparative_panel(record['rgb'], record['overlays'], record['metadata'], record['directory']/'comparison.png')
    write_json(root/'complete.json', {'status':'complete', 'examples':len(chosen), 'panels':len(records),
                                     'note':'Post-hoc only; no evaluated model, split, threshold or ensemble weights changed.'})
    return {'directory':str(root), 'examples':len(chosen), 'panels':len(records), 'selection':selection}
