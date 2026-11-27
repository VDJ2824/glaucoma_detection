"""Public experiment archive checks without private predictions or inference."""
import runpy
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / 'scripts/verify_phase4_artifacts.py'))


def test_phase4_aggregate_archive_is_consistent():
    assert API['verify_bundle'](ROOT) == {
        'files_verified': 20, 'files_withheld': 17,
        'methods': 6, 'images_per_partition': 106,
    }


@pytest.mark.parametrize('kind', ['altered_metric', 'private_predictions', 'unexpected_weights'])
def test_unreviewed_or_altered_artifacts_are_rejected(tmp_path, kind):
    target = tmp_path / API['RECORD']
    shutil.copytree(ROOT / API['RECORD'], target)
    if kind == 'altered_metric':
        path = target / 'test/metrics.json'
        path.write_bytes(path.read_bytes() + b'\n')
        message = 'byte/hash mismatch'
    elif kind == 'private_predictions':
        (target / 'test/aligned_predictions.csv').write_text('image_id,label\nprivate,1\n')
        message = 'Withheld artifact published'
    else:
        (target / 'model.keras').write_bytes(b'not for publication')
        message = 'Unexpected/unreviewed'
    with pytest.raises(ValueError, match=message):
        API['verify_bundle'](tmp_path)
