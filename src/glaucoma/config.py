"""Central YAML configuration; no TensorFlow dependency or downloads."""
from pathlib import Path
import math
import os
import yaml

CORE_MODELS = ('resnet50', 'mobilenet_v2', 'efficientnet_b0')
BASELINES = ('vgg16', 'vgg19')


def load_config(path='configs/project.yaml'):
    path = Path(path).resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a YAML mapping')
    root = (path.parent / cfg['project_root']).resolve()
    cfg['project_root'] = str(root)
    if cfg['model']['selected'] not in CORE_MODELS + BASELINES:
        raise ValueError('Unknown model selection')
    if cfg['model']['input_shape'] != [224, 224, 3]:
        raise ValueError('Expected input_shape [224, 224, 3]')
    ratios = cfg['split']['ratios']
    if set(ratios) != {'train', 'validation', 'test'} or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in ratios.values()
    ) or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError('Split ratios must be positive and sum to one')
    if cfg['split']['mode'] not in ('generated', 'official'):
        raise ValueError('Unknown split mode')
    if cfg['metadata']['duplicate_policy'] not in ('drop_identical', 'error'):
        raise ValueError('Unknown duplicate policy')
    if type(cfg['training']['batch_size']) is not int or cfg['training']['batch_size'] <= 0 or type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('Invalid batch size or seed')
    for stage in ('stage1', 'stage2'):
        values = cfg['training'][stage]
        if type(values['epochs']) is not int or values['epochs'] <= 0 or not 0 < values['learning_rate'] < 1:
            raise ValueError('Invalid training stage')
    head = cfg['model']['head']
    if head['pooling'] != 'global_average' or not 0 <= head['dropout'] < 1 or head['units'] != 1 or head['activation'] != 'sigmoid':
        raise ValueError('Invalid binary classification head')
    early = cfg['training']['early_stopping']
    if early['mode'] not in ('min', 'max', 'auto') or type(early['patience']) is not int or early['patience'] < 0 or not math.isfinite(early['min_delta']) or early['min_delta'] < 0:
        raise ValueError('Invalid early stopping parameters')
    for section in ('ensemble', 'export'):
        if tuple(cfg[section]['members']) != CORE_MODELS:
            raise ValueError('Only the three core models may be ensembled/exported')
    pre = cfg['preprocessing']
    if (pre['color'], pre['resize'], pre['normalization']) != ('RGB', 'letterbox', 'model_specific'):
        raise ValueError('Unsupported preprocessing configuration')
    if not 0 <= pre['pad_value'] <= 255:
        raise ValueError('Invalid padding value')
    clahe = pre['clahe']
    if clahe['clip_limit'] <= 0 or len(clahe['tile_grid_size']) != 2 or any(x <= 0 for x in clahe['tile_grid_size']):
        raise ValueError('Invalid CLAHE parameters')
    aug = cfg['augmentation']
    if not 0 <= aug['horizontal_flip_probability'] <= 1 or not 0 <= aug['brightness_delta'] <= 1 or not 0 <= aug['rotation_degrees'] <= 180:
        raise ValueError('Invalid augmentation settings')
    def resolve(value):
        p = Path(os.path.expandvars(os.path.expanduser(value)))
        return str((root / p).resolve())
    dataset_root = os.getenv('DATASET_ROOT')
    for name, spec in cfg['datasets'].items():
        if name == 'selected':
            continue
        spec['root'] = resolve(dataset_root if dataset_root and name == cfg['datasets']['selected'] else spec['root'])
        if spec.get('annotations_csv'):
            spec['annotations_csv'] = resolve(spec['annotations_csv'])
    if cfg['datasets']['selected'] not in cfg['datasets']:
        raise ValueError('Unknown dataset')
    for key in ('output', 'validation_output'):
        cfg['metadata'][key] = resolve(cfg['metadata'][key])
    artifact_root = os.getenv('ARTIFACTS_ROOT')
    for key in cfg['checkpoints']:
        cfg['checkpoints'][key] = resolve(str(Path(artifact_root) / 'checkpoints' / key) if artifact_root else cfg['checkpoints'][key])
    cfg['export']['output_root'] = resolve(cfg['export']['output_root'])
    return cfg
