"""Dataset adapters produce original-image metadata, never augmented samples."""
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

COLUMNS = ['image_id', 'image_path', 'dataset_name', 'patient_id', 'label', 'split']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class DatasetAdapter(ABC):
    @abstractmethod
    def discover(self) -> pd.DataFrame:
        """Return COLUMNS; unknown patient IDs/splits remain empty."""


class AcrimaAdapter(DatasetAdapter):
    """Read explicit binary class folders or user-supplied glaucoma annotations."""
    def __init__(self, root, class_directories=None, annotations_csv=None):
        self.root = Path(root).resolve()
        self.class_directories = class_directories
        self.annotations_csv = annotations_csv

    def discover(self):
        if not self.root.is_dir():
            raise ValueError(f'Dataset directory does not exist: {self.root}')
        if self.annotations_csv:
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

    def _path(self, value):
        # Preserve blank paths so validation reports them as invalid.
        return str((self.root / value).resolve()) if value.strip() else ''


ADAPTERS = {'acrima': AcrimaAdapter}


def create_adapter(cfg):
    name = cfg['datasets']['selected']
    if name not in ADAPTERS:
        raise NotImplementedError(f'{name}: no verified glaucoma-label adapter implemented')
    spec = cfg['datasets'][name]
    return ADAPTERS[name](spec['root'], spec.get('class_directories'), spec.get('annotations_csv'))
