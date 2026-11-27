"""Read-only checkpoint/contract verification using Phase 3 selection and recovery."""
import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
from glaucoma.data.adapters import COLUMNS
from glaucoma.ensemble.predictions import MEMBERS, binary_labels
from glaucoma.models.factory import get_backbone
from glaucoma.training.artifacts import stage_paths
from glaucoma.training.trainer import select_best_checkpoint, recover_best_model


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_manifest(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not set(COLUMNS).issubset(frame) or frame.empty:
        raise ValueError('Frozen six-column manifest required')
    if not frame['split'].isin(['train', 'validation', 'test']).all():
        raise ValueError('Unknown/missing frozen split assignment')
    return frame[COLUMNS].copy()


def training_manifest_digest(frame):
    """Match Phase 3's canonical train/validation CSV without stat'ing any image.

    Use the original manifest paths. Physical flat-directory relocation is a
    separate in-memory inference option and never changes this contract digest.
    """
    data = frame.loc[frame['split'].isin(['train', 'validation']), COLUMNS].copy().reset_index(drop=True)
    if set(data['split']) != {'train', 'validation'}:
        raise ValueError('Both frozen training and validation partitions required')
    data['label'] = binary_labels(data.label)
    data['patient_id'] = data.patient_id.str.strip()
    data['image_path'] = [str(Path(value).resolve()) for value in data.image_path]
    if data.duplicated(['dataset_name', 'image_id']).any():
        raise ValueError('Duplicate train/validation image keys')
    return hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()


def verify_selected_checkpoints(cfg, manifest_path, expected_records=None, checkpoint_roots=None):
    """Require both completed stages, true contract digests and original manifest.

    expected_records optionally maps each core name to a committed selection.json.
    checkpoint_roots maps names to bases containing <name>/<stage>/ artifacts,
    allowing private Kaggle attachments while leaving stored contracts unchanged.
    """
    if expected_records is not None and set(expected_records) != set(MEMBERS):
        raise ValueError('Selection references must cover all core models')
    if checkpoint_roots is not None and set(checkpoint_roots) != set(MEMBERS):
        raise ValueError('Checkpoint roots must cover all core models')
    frame = read_manifest(manifest_path)
    manifest_digest = training_manifest_digest(frame)
    result = {}
    for name in MEMBERS:
        local = copy.deepcopy(cfg)
        local['model']['selected'] = name
        if checkpoint_roots:
            local['checkpoints']['core'] = str(checkpoint_roots[name])
        for stage in ('stage1', 'stage2'):
            paths = stage_paths(local, stage)
            if not all(paths[key].is_file() for key in ('state', 'contract', 'best')):
                raise FileNotFoundError(f'{name}/{stage}: best.keras, state.json and contract.json are required')
            state = json.loads(paths['state'].read_text())
            if state.get('complete') is not True:
                raise ValueError(f'{name}/{stage}: training is not complete')
        selection = select_best_checkpoint(local)
        stage = selection['stage']
        paths = stage_paths(local, stage)
        contract = json.loads(paths['contract'].read_text())
        payload = {key: value for key, value in contract.items() if key != 'runtime'}
        if canonical_hash(payload) != selection['contract_hash']:
            raise ValueError(f'{name}: training contract digest mismatch')
        # Validate both stage contracts, not just the selected one.
        for other_stage in ('stage1', 'stage2'):
            other = json.loads(stage_paths(local, other_stage)['contract'].read_text())
            if canonical_hash({k:v for k,v in other.items() if k != 'runtime'}) != selection['contract_hash']:
                raise ValueError(f'{name}: stage contracts disagree')
        if contract['training_validation_manifest_sha256'] != manifest_digest:
            raise ValueError(f'{name}: frozen training/validation manifest does not match training; supply the original canonical CSV')
        if (contract['model']['selected'] != name or contract['model']['input_shape'] != [224,224,3]
            or contract['model']['input_shape'] != cfg['model']['input_shape']
            or contract['model']['head'] != cfg['model']['head'] or contract['preprocessing'] != cfg['preprocessing']
            or contract['seed'] != cfg['seed']):
            raise ValueError(f'{name}: identity/input/preprocessing contract incompatible')
        if expected_records is not None:
            recorded = json.loads(Path(expected_records[name]).read_text())
            for key in ('stage', 'epoch', 'val_loss', 'contract_hash', 'complete'):
                if recorded.get(key) != selection.get(key):
                    raise ValueError(f'{name}: checkpoint selection differs from experiment record')
            suffix = f'/checkpoints/core/{name}/{stage}/best.keras'
            if not recorded['checkpoint'].endswith(suffix):
                raise ValueError('Recorded checkpoint identity is invalid')
        result[name] = {'model_name': name, 'selection': selection, 'path': str(paths['best'].resolve()),
                        'checkpoint_sha256': sha256_file(paths['best']), 'contract_hash': selection['contract_hash'],
                        'training_manifest_sha256': manifest_digest, 'input_shape': [224,224,3],
                        'preprocessing': contract['preprocessing'], 'class_mapping': {'0':'non_glaucoma', '1':'glaucoma'},
                        'checkpoint_root': local['checkpoints']['core']}
    return result


def verify_model_identity(model, name):
    import tensorflow as tf
    if model.name != name or model.input_shape != (None,224,224,3) or model.output_shape != (None,1):
        raise ValueError('Loaded model identity or tensor shape mismatch')
    backbone = get_backbone(model)
    prefix = {'mobilenet_v2':'mobilenetv2', 'efficientnet_b0':'efficientnetb0', 'resnet50':'resnet50'}[name]
    head = model.layers[-3:]
    channels = 2048 if name == 'resnet50' else 1280
    if (len(model.layers) != 5 or backbone.output_shape != (None,7,7,channels)
        or not backbone.name.lower().startswith(prefix) or not isinstance(head[0], tf.keras.layers.GlobalAveragePooling2D)
        or not isinstance(head[1], tf.keras.layers.Dropout) or head[1].rate != .2
        or not isinstance(head[2], tf.keras.layers.Dense) or head[2].units != 1
        or head[2].activation != tf.keras.activations.sigmoid):
        raise ValueError('Loaded backbone/head is not a compatible core classifier')
    return canonical_hash(json.loads(model.to_json()))


def load_verified_model(cfg, name, record):
    if sha256_file(record['path']) != record['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since verification')
    local = copy.deepcopy(cfg)
    local['model']['selected'] = name
    local['checkpoints']['core'] = record['checkpoint_root']
    model = recover_best_model(local, stage=record['selection']['stage'])
    graph_hash = verify_model_identity(model, name)
    return model, graph_hash
