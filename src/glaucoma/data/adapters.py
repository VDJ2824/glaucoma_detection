"""Dataset adapters produce original-image metadata, never augmented samples."""
from abc import ABC, abstractmethod
from pathlib import Path
import re
import pandas as pd

COLUMNS = ['image_id', 'image_path', 'dataset_name', 'patient_id', 'label', 'split']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class DatasetAdapter(ABC):
    @abstractmethod
    def discover(self) -> pd.DataFrame:
        """Return COLUMNS; unknown patient IDs/splits remain empty."""


class AcrimaAdapter(DatasetAdapter):
    """Read explicit folders, CSV annotations, or canonical flat ACRIMA filenames."""
    def __init__(self, root, class_directories=None, annotations_csv=None, label_mode=None):
        self.root = Path(root).resolve()
        self.class_directories = class_directories
        self.annotations_csv = annotations_csv
        # No explicit mode preserves the original constructor's CSV/folder behavior.
        self.label_mode = label_mode if label_mode is not None else ('csv' if annotations_csv else 'folders')
        if self.label_mode not in ('folders', 'csv', 'filename'):
            raise ValueError(f'Unknown ACRIMA label mode: {self.label_mode}')
        if self.label_mode == 'csv' and not annotations_csv:
            raise ValueError('CSV label mode requires annotations_csv')
        if self.label_mode != 'csv' and annotations_csv:
            raise ValueError('annotations_csv requires CSV label mode; do not mix label sources')

    def discover(self):
        if not self.root.is_dir():
            raise ValueError(f'Dataset directory does not exist: {self.root}')
        if self.label_mode == 'filename':
            return self._discover_filenames()
        if self.label_mode == 'csv':
            frame = pd.read_csv(self.annotations_csv, dtype=str, keep_default_na=False)
            required = {'image_id', 'image_path', 'label'}
            if not required.issubset(frame.columns):
                raise ValueError(f'Annotations require {sorted(required)}')
            if 'dataset_name' in frame and not frame['dataset_name'].isin(['', 'acrima']).all():
                raise ValueError('Annotations contain another dataset name')
            for column in COLUMNS:
                if column not in frame:
                    frame[column] = ''
            frame['dataset_name'] = 'acrima'
            frame['image_path'] = frame['image_path'].map(self._path)
            return frame[COLUMNS]
        mapping = self.class_directories
        if not mapping or set(mapping.values()) != {0, 1}:
            raise ValueError('Provide an explicit folder-to-glaucoma-label mapping (0 and 1)')
        rows = []
        for directory, label in mapping.items():
            folder = (self.root / directory).resolve()
            if not folder.is_relative_to(self.root) or not folder.is_dir():
                raise ValueError(f'Invalid class directory: {directory}')
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    rows.append([path.relative_to(self.root).as_posix(), str(path.resolve()), 'acrima', '', label, ''])
        return pd.DataFrame(rows, columns=COLUMNS)

    def _discover_filenames(self):
        # Full matching is deliberate: absence of _g_ alone is never a label.
        pattern = re.compile(r'Im[0-9]+(?P<glaucoma>_g)?_ACRIMA\.jpg', re.IGNORECASE)
        rows, unexpected = [], []
        for path in sorted(self.root.iterdir()):
            match = pattern.fullmatch(path.name)
            if not path.is_file() or not path.resolve().is_relative_to(self.root) or not match:
                unexpected.append(path.name)
                continue
            rows.append([path.name, str(path.resolve()), 'acrima', '',
                         int(match.group('glaucoma') is not None), ''])
        if unexpected:
            raise ValueError('Unexpected entries in flat ACRIMA filename mode: ' + ', '.join(unexpected))
        if not rows:
            raise ValueError('No canonical ACRIMA images found in the flat directory')
        return pd.DataFrame(rows, columns=COLUMNS)

    def _path(self, value):
        # Preserve blank paths so validation reports them as invalid.
        return str((self.root / value).resolve()) if value.strip() else ''


ADAPTERS = {'acrima': AcrimaAdapter}


def create_adapter(cfg):
    name = cfg['datasets']['selected']
    if name not in ADAPTERS:
        raise NotImplementedError(f'{name}: no verified glaucoma-label adapter implemented')
    spec = cfg['datasets'][name]
    return ADAPTERS[name](spec['root'], spec.get('class_directories'), spec.get('annotations_csv'), label_mode=spec.get('label_mode'))
