"""Per-model/stage artifact locations and atomic JSON records."""
import json
from pathlib import Path
from glaucoma.models.registry import CORE_APPLICATIONS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stage_paths(cfg, stage):
    name = cfg['model']['selected']
    if name not in CORE_APPLICATIONS or stage not in ('stage1', 'stage2'):
        raise ValueError('Core model and stage1/stage2 required')
    checkpoint = Path(cfg['checkpoints']['core']) / name / stage
    artifact_root = Path(cfg.get('artifacts', {}).get('root', Path(cfg['checkpoints']['core']).parents[1]))
    artifacts = cfg.get('artifacts', {})
    histories = Path(artifacts.get('histories', artifact_root / 'histories')) / name / stage
    metrics = Path(artifacts.get('metrics', artifact_root / 'metrics')) / name
    paths = {'directory': checkpoint, 'best': checkpoint / 'best.keras', 'last': checkpoint / 'last.keras',
             'state': checkpoint / 'state.json', 'contract': checkpoint / 'contract.json',
             'history': histories / 'history.json', 'csv': histories / 'epochs.csv',
             'metrics': metrics / f'{stage}.json', 'selection': metrics / 'selection.json'}
    dataset = Path(cfg['datasets'][cfg['datasets']['selected']]['root']).resolve()
    if any(path.resolve().is_relative_to(dataset) for path in paths.values()):
        raise ValueError('Training artifacts must be outside original dataset storage')
    return paths
