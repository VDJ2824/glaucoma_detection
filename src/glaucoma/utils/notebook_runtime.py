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
    return _configure_storage(load_config(config_path), dataset_root, artifact_root, run_name)


def _configure_storage(config, dataset_root, artifact_root, run_name, create_directories=True):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_name):
        raise ValueError('Use a simple run name without path separators')
    cfg = copy.deepcopy(config)
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
    if create_directories:
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



def relocate_frozen_session(session_path, metadata_path, dataset_root, artifact_root, run_name='development'):
    """Explicit flat-directory path relocation only; no new labels or splits.

    Original JSON/CSV/images remain untouched. Reject basename collisions,
    missing/out-of-root files and conflicting derived outputs. Repeated calls
    with identical inputs reuse derived files without overwriting them.
    """
    import pandas as pd
    from glaucoma.data.adapters import COLUMNS
    session_path = Path(session_path).expanduser().resolve()
    metadata_path = Path(metadata_path).expanduser().resolve()
    original = load_session(session_path)
    frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if not set(COLUMNS).issubset(frame.columns) or frame.empty:
        raise ValueError('A nonempty frozen metadata CSV with the six schema columns is required')
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('The relocated image directory must already exist')
    names = [Path(value).name for value in frame.image_path]
    if any(not name or name in ('.', '..') for name in names) or len(set(names)) != len(names):
        raise ValueError('Blank or colliding basenames: flat relocation cannot preserve this manifest safely')
    new_paths = [(root / name).resolve() for name in names]
    if any(not path.is_relative_to(root) or not path.is_file() for path in new_paths):
        raise ValueError('Some relocated images are missing or escape the dataset directory')
    # Resolve and check destinations before creating directories or writing outputs.
    target = (Path(artifact_root).expanduser().resolve() / 'runs' / run_name).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_name):
        raise ValueError('Use a simple run name without path separators')
    selected = original['datasets']['selected']
    output_csv = target / 'manifests' / f'{selected}_metadata.csv'
    output_session = target / 'environment' / 'session.json'
    protected = {session_path, metadata_path}
    if any(path.resolve() in protected for path in (output_csv, output_session)):
        raise ValueError('Relocation must not overwrite the original manifest or session')
    relocated = frame.copy()
    relocated['image_path'] = [str(path) for path in new_paths]
    # Compare every non-path column, not only aggregate class/split counts.
    assert relocated.drop(columns='image_path').equals(frame.drop(columns='image_path'))
    if output_csv.exists():
        prior = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
        if not prior.equals(relocated):
            raise FileExistsError('Derived manifest differs; choose a new output run instead of overwriting it')
    # Compute destinations without writing, preserving the loaded run configuration.
    cfg = _configure_storage(original, root, artifact_root, run_name, create_directories=False)
    if output_session.exists() and load_session(output_session) != cfg:
        raise FileExistsError('Derived session differs; choose a new output run instead of overwriting it')
    cfg = _configure_storage(original, root, artifact_root, run_name)
    if not output_csv.exists():
        temporary = output_csv.with_name(output_csv.name + '.tmp')
        relocated.to_csv(temporary, index=False)
        temporary.replace(output_csv)
    if not output_session.exists():
        save_session(cfg, output_session)
    return cfg, output_session
