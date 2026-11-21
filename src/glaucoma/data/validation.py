"""Validate and deduplicate originals before splitting or augmentation."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import cv2
import pandas as pd
from .adapters import COLUMNS


@dataclass
class ValidationResult:
    metadata: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_errors(self):
        return not self.issues.empty and self.issues['severity'].eq('error').any()


def validate_metadata(frame, dataset_root, duplicate_policy='drop_identical'):
    if duplicate_policy not in ('drop_identical', 'error'):
        raise ValueError('Unknown duplicate policy')
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing metadata columns: {sorted(missing)}')
    data = frame[COLUMNS].copy().fillna('').reset_index(drop=True)
    data['patient_id'] = data['patient_id'].astype(str).str.strip()
    root = Path(dataset_root).resolve()
    issues, hashes, remove = [], {}, set()
    def issue(i, code, severity='error'):
        issues.append({'row': i, 'image_id': data.at[i, 'image_id'], 'code': code, 'severity': severity})
    for i, row in data.iterrows():
        if not str(row.image_id).strip():
            issue(i, 'missing_image_id')
        if not str(row.dataset_name).strip():
            issue(i, 'missing_dataset_name')
        if str(row.label) not in ('0', '1'):
            issue(i, 'missing_or_invalid_label')
        if str(row.split) not in ('', 'train', 'validation', 'test'):
            issue(i, 'invalid_split')
        if not str(row.image_path).strip():
            issue(i, 'invalid_path'); continue
        path = Path(row.image_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            issue(i, 'invalid_path'); continue
        data.at[i, 'image_path'] = str(path)
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                issue(i, 'corrupt_image'); continue
            # Hash decoded pixels to catch copies with different filenames/metadata.
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        except (OSError, cv2.error):
            issue(i, 'corrupt_image'); continue
        if digest in hashes:
            previous = data.loc[hashes[digest]]
            conflicts = any(str(previous[k]) != str(row[k]) for k in ('label', 'patient_id', 'split', 'dataset_name'))
            if conflicts or duplicate_policy == 'error':
                issue(i, 'conflicting_duplicate' if conflicts else 'duplicate_image')
            else:
                issue(i, 'duplicate_removed', 'warning'); remove.add(i)
        else:
            hashes[digest] = i
    retained = data.drop(index=list(remove))
    for i in retained.index[retained['image_id'].duplicated(keep=False)]:
        issue(i, 'duplicate_image_id')
    return ValidationResult(retained.reset_index(drop=True), pd.DataFrame(issues, columns=['row', 'image_id', 'code', 'severity']))
