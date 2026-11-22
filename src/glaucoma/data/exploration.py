"""Read-only original-image exploration. No splitting, augmentation or training."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .adapters import create_adapter
from .validation import ValidationResult, validate_metadata


@dataclass
class ExplorationResult:
    metadata: pd.DataFrame
    validation: ValidationResult
    class_distribution: pd.DataFrame
    dimensions: pd.DataFrame
    dimension_statistics: pd.DataFrame


def explore_dataset(cfg):
    """Audit all originals, retaining duplicates in exploratory counts.

    Existing validation runs with duplicate_policy='error' for reporting only;
    findings do not discard rows or alter files. Invalid/corrupt paths cannot
    contribute dimension measurements or sample visualizations.
    """
    frame = create_adapter(cfg).discover()
    root = cfg['datasets'][cfg['datasets']['selected']]['root']
    validation = validate_metadata(frame, root, duplicate_policy='error')
    labels = pd.to_numeric(frame.label, errors='coerce')
    counts = labels.value_counts().reindex([0, 1], fill_value=0)
    distribution = pd.DataFrame({'label': [0, 1], 'class_name': ['non_glaucoma', 'glaucoma'],
                                 'count': counts.to_numpy()})
    distribution['percentage'] = distribution['count'] / len(frame) * 100 if len(frame) else 0.0
    invalid = set(validation.issues.loc[validation.issues.code.isin(['invalid_path', 'corrupt_image']), 'row'])
    dimensions = []
    for i, row in frame.reset_index(drop=True).iterrows():
        if i in invalid:
            continue
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Image changed or became unreadable after validation; rerun exploration')
        height, width = image.shape[:2]
        dimensions.append({'image_id': row.image_id, 'image_path': row.image_path, 'label': row.label,
                           'width': width, 'height': height, 'aspect_ratio': width / height})
    dimensions = pd.DataFrame(dimensions, columns=['image_id', 'image_path', 'label', 'width', 'height', 'aspect_ratio'])
    stats = dimensions[['width', 'height', 'aspect_ratio']].astype(float).describe()
    return ExplorationResult(frame, validation, distribution, dimensions, stats)


def plot_class_distribution(result):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    table = result.class_distribution
    ax.bar(table.class_name, table['count'])
    ax.set(ylabel='Original image count', title='ACRIMA class distribution (before deduplication)')
    for i, row in enumerate(table.itertuples()):
        ax.text(i, row.count, str(row.count), ha='center', va='bottom')
    fig.tight_layout()
    return fig


def plot_samples(result, per_class=3, seed=42):
    """Deterministic class-balanced RGB previews of unmodified decoded originals."""
    import matplotlib.pyplot as plt
    if type(per_class) is not int or per_class < 1:
        raise ValueError('per_class must be a positive integer')
    fig, axes = plt.subplots(2, per_class, figsize=(4 * per_class, 7), squeeze=False)
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        rows = result.dimensions.loc[result.dimensions.label.astype(str).eq(str(label))]
        chosen = rng.choice(len(rows), size=min(per_class, len(rows)), replace=False)
        for col in range(per_class):
            ax = axes[label, col]
            ax.axis('off')
            if col >= len(chosen):
                ax.set_title(f'Label {label}: no further valid images')
                continue
            row = rows.iloc[chosen[col]]
            image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Sample became unreadable; rerun exploration')
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(f'{Path(row.image_path).name}\nlabel {label}; {row.width}×{row.height}')
    fig.tight_layout()
    return fig
