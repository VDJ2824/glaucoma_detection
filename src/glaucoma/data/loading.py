"""Dataset-independent preparation and lazy, training-only augmentation batches."""
from pathlib import Path
import numpy as np
from .adapters import create_adapter
from .validation import validate_metadata
from .splitting import split_metadata
from .preprocessing import preprocess_image


def prepare_dataset(cfg):
    """Writes derived CSVs only. Fails closed on any validation error."""
    adapter = create_adapter(cfg)
    result = validate_metadata(adapter.discover(), cfg['datasets'][cfg['datasets']['selected']]['root'], cfg['metadata']['duplicate_policy'])
    report = Path(cfg['metadata']['validation_output'])
    output = Path(cfg['metadata']['output'])
    if report == output:
        raise ValueError('Metadata and validation output paths must differ')
    raw_root = Path(cfg['datasets'][cfg['datasets']['selected']]['root']).resolve()
    if any(p.resolve().is_relative_to(raw_root) for p in (report, output)):
        raise ValueError('Derived outputs must be outside the original dataset root')
    # Refuse to overwrite any annotation file or original image.
    originals = {Path(p).resolve() for p in result.metadata.image_path if str(p)}
    annotations = cfg['datasets'][cfg['datasets']['selected']].get('annotations_csv')
    if annotations:
        originals.add(Path(annotations).resolve())
    if any(p.resolve() in originals for p in (report, output)):
        raise ValueError('Output would overwrite source data')
    report.parent.mkdir(parents=True, exist_ok=True)
    result.issues.to_csv(report, index=False)
    if result.has_errors:
        raise ValueError(f'Dataset validation failed; see {report}')
    frame = split_metadata(result.metadata, seed=cfg['seed'], **cfg['split'])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def iter_batches(metadata, cfg, split, epoch=0):
    """One deterministic epoch; no augmented files are written or cached."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split')
    frame = metadata.loc[metadata.split.eq(split)].reset_index(drop=True)
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], epoch]))
    indices = np.arange(len(frame))
    if split == 'train':
        rng.shuffle(indices)
    size = cfg['training']['batch_size']
    for offset in range(0, len(indices), size):
        rows = frame.iloc[indices[offset:offset+size]]
        yield (np.stack([preprocess_image(row.image_path, cfg, split, rng) for row in rows.itertuples()]), rows.label.to_numpy(dtype=np.float32))
