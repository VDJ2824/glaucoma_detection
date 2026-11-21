"""Hosted notebook configuration and diagnostics, independent of training logic.

Importing this module does not import TensorFlow, mount storage or write files.
"""
import copy
import json
from pathlib import Path
import re
import sys

from glaucoma.config import load_config


def configure_workspace(config_path, dataset_root, artifact_root, run_name='development'):
    """Create derived-output directories and return a config, leaving YAML untouched.

    artifact_root may be a mounted Google Drive folder or Kaggle working folder.
    Dataset directories are never created or changed. Run names isolate experiments.
    """
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_name):
        raise ValueError('Use a simple run name without path separators')
    cfg = copy.deepcopy(load_config(config_path))
    dataset = Path(dataset_root).expanduser().resolve()
    root = (Path(artifact_root).expanduser().resolve() / 'runs' / run_name).resolve()
    if root.is_relative_to(dataset):
        raise ValueError('Artifacts must be outside the original dataset directory')
    selected = cfg['datasets']['selected']
    cfg['datasets'][selected]['root'] = str(dataset)
    cfg['artifacts'] = {'root': str(root)}
    for name in ('metrics', 'histories', 'heatmaps', 'exports', 'manifests', 'environment'):
        cfg['artifacts'][name] = str(root / name)
    cfg['checkpoints'] = {kind: str(root / 'checkpoints' / kind) for kind in ('core', 'baselines')}
    cfg['metadata']['output'] = str(root / 'manifests' / f'{selected}_metadata.csv')
    cfg['metadata']['validation_output'] = str(root / 'manifests' / f'{selected}_validation.csv')
    cfg['export']['output_root'] = cfg['artifacts']['exports']
    directories = [Path(path) for path in [*cfg['artifacts'].values(), *cfg['checkpoints'].values()]]
    if any(path.resolve().is_relative_to(dataset) for path in directories):
        raise ValueError('Artifact directory resolves inside the original dataset directory')
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
    return cfg


def save_session(cfg, path):
    """Save resolved settings for independent notebooks; contains local paths, no secrets."""
    path = Path(path).expanduser().resolve()
    dataset = Path(cfg['datasets'][cfg['datasets']['selected']]['root']).resolve()
    if path.is_relative_to(dataset):
        raise ValueError('Session settings must be outside the original dataset directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    return path


def load_session(path):
    """Load settings saved by notebook 00; rerun 00 if runtime mount paths change."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Run notebook 00 first, then set SESSION_FILE to its saved path: {path}')
    return json.loads(path.read_text())


def verify_tensorflow():
    """Import TensorFlow explicitly and verify real tensor execution on GPU or CPU."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    memory_growth = []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            memory_growth.append('enabled')
        except RuntimeError as exc:
            memory_growth.append(str(exc))
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        result = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
    values = result.numpy()
    if values.shape != (2, 2) or not (values == 2).all():
        raise RuntimeError('TensorFlow matrix multiplication verification failed')
    return {
        'python': sys.version,
        'tensorflow': tf.__version__,
        'built_with_cuda': tf.test.is_built_with_cuda(),
        'physical_gpus': [device.name for device in gpus],
        'execution_device': result.device,
        'memory_growth': memory_growth,
        'tensor_check': 'passed',
    }
