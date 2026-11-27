"""Verify public Phase 4 aggregates/provenance without models, images or private scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

RECORD = Path('docs/experiments/phase4_kaggle_2026-11-27')
MEMBERS = ('mobilenet_v2', 'efficientnet_b0', 'resnet50')
METHODS = (*MEMBERS, 'majority_voting', 'equal_soft_voting', 'weighted_soft_voting')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def seal(record):
    body = {k: v for k, v in record.items() if k != 'seal_sha256'}
    require(digest(json.dumps(body, sort_keys=True, allow_nan=False).encode()) == record['seal_sha256'], 'Invalid canonical seal')


def verify_bundle(repo_root):
    root = Path(repo_root)
    directory = root / RECORD
    manifest = read(directory / 'artifact_manifest.json')
    preserved = set()
    withheld = set()
    entries = {item['archive_member']: item for item in manifest['files']}
    require(len(entries) == len(manifest['files']) == 37, 'Unexpected source member inventory')
    for name, item in entries.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe artifact path')
        path = directory / name
        if item['disposition'] == 'withheld':
            require(not path.exists(), 'Withheld artifact published')
            withheld.add(name)
            continue
        require(item['disposition'] == 'preserved', 'Unknown disposition')
        payload = path.read_bytes()
        require(len(payload) == item['bytes'] and digest(payload) == item['sha256'], f'Artifact byte/hash mismatch: {name}')
        require(path.suffix in {'.png', '.csv', '.json'}, 'Unexpected file type')
        if path.suffix == '.png':
            require(payload.startswith(b'\x89PNG\r\n\x1a\n'), 'Invalid PNG')
        preserved.add(name)
    require(len(preserved) == 20 and len(withheld) == 17, 'Incorrect privacy inventory')
    expected_files = preserved | {'README.md', 'artifact_manifest.json'}
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual_files == expected_files, 'Unexpected/unreviewed archive file')
    for name in preserved:
        if name.endswith('.csv'):
            with (directory / name).open(newline='') as stream:
                header = next(csv.reader(stream))
            require(not {'image_id', 'image_path', 'patient_id', 'label', 'split'}.intersection(header), 'Image-level CSV published')
    frozen = read(directory / 'frozen_ensemble.json')
    handoff = read(directory / 'validation_predictions.json')
    seal(frozen)
    seal(handoff)
    require(digest((directory / 'validation_predictions.json').read_bytes()) == frozen['model_validation_record_sha256'], 'Handoff link mismatch')
    require(frozen['protocol'] == read(directory / 'predefined_protocol.json'), 'Protocol mismatch')
    require(frozen['members'] == list(MEMBERS) and frozen['threshold'] == 0.5, 'Frozen members/threshold changed')
    require(frozen['status'] == 'frozen_before_test' and frozen['selection_partition'] == 'validation' and frozen['test_used_for_selection'] is False, 'Wrong selection scope')
    require(frozen['weights'] == dict(zip(MEMBERS, (0.0, 0.15, 0.85))), 'Frozen weights changed')
    require(handoff['manifest_sha256'] == frozen['manifest_sha256'], 'Manifest link mismatch')
    for name, value in handoff['prediction_files'].items():
        require(entries[name]['sha256'] == value, 'Private prediction provenance mismatch')
    require(frozen['validation_prediction_sha256'] == entries['validation/aligned_predictions.csv']['sha256'], 'Validation hash mismatch')
    for marker in ('test_started.json', 'test_complete.json'):
        require(read(directory / marker)['frozen_seal'] == frozen['seal_sha256'], 'Test marker seal mismatch')
    require(read(directory / 'test_complete.json')['n'] == 106, 'Test completion count mismatch')
    model_dirs = read(root / 'docs/experiments/core_models_artifact_manifest_2026-11-27.json')['model_directories']
    for name in MEMBERS:
        selection = read(root / model_dirs[name] / 'metrics/selection.json')
        require(frozen['checkpoints'][name]['selection'] == selection, 'Earlier checkpoint selection changed')
        require(handoff['checkpoints'][name] == frozen['checkpoints'][name], 'Checkpoint handoff differs')
    with (directory / 'validation/weight_search.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 232, 'Wrong search candidate count')
    selected = [row for row in rows if row['selected'].lower() == 'true']
    require(len(selected) == 1, 'Wrong selected count')
    require(all(close(selected[0][name], frozen['weights'][name]) for name in MEMBERS), 'Search weight mismatch')
    require(close(selected[0]['validation_binary_crossentropy'], min(float(row['validation_binary_crossentropy']) for row in rows)), 'Search did not minimize validation BCE')
    bundles = {}
    for split, prefix, methods, counts in [
        ('validation', 'validation', MEMBERS, (46, 60)),
        ('validation', 'validation/ensemble', METHODS, (46, 60)),
        ('test', 'test', METHODS, (47, 59)),
    ]:
        bundle = read(directory / prefix / 'metrics.json')
        require(bundle['split'] == split and bundle['threshold'] == 0.5 and set(bundle['methods']) == set(methods), 'Metric scope mismatch')
        with (directory / prefix / 'metrics_comparison.csv').open(newline='') as stream:
            table = {row['method']: row for row in csv.DictReader(stream)}
        require(set(table) == set(methods), 'CSV methods differ')
        for name, m in bundle['methods'].items():
            tn, fp, fn, tp = [m[k] for k in ('tn', 'fp', 'fn', 'tp')]
            require(all(type(v) is int and v >= 0 for v in (tn, fp, fn, tp)), 'Invalid confusion count')
            require((tn + fp, fn + tp) == counts and m['n'] == sum(counts), 'Class counts differ')
            require(m['confusion_matrix'] == [[tn, fp], [fn, tp]], 'Confusion matrix differs')
            derived = {'accuracy': (tn+tp)/m['n'], 'precision': tp/(tp+fp), 'recall': tp/(tp+fn), 'specificity': tn/(tn+fp), 'f1': 2*tp/(2*tp+fp+fn)}
            require(all(close(m[k], v) for k, v in derived.items()), 'Confusion-derived metric mismatch')
            for key, value in table[name].items():
                if key != 'method':
                    require(value == '' if m[key] is None else close(value, m[key]), f'CSV/JSON mismatch: {key}')
            require((m['roc_auc'] is None and m['binary_crossentropy'] is None) if name == 'majority_voting' else (0 <= m['roc_auc'] <= 1 and m['binary_crossentropy'] >= 0), 'Invalid score metric')
        bundles[prefix] = bundle['methods']
    for name in MEMBERS:
        require(bundles['validation'][name] == bundles['validation/ensemble'][name], 'Individual validation results changed')
    test = bundles['test']
    equal = test['equal_soft_voting']
    require(close(equal['accuracy'], max(test[name]['accuracy'] for name in MEMBERS)), 'Equal accuracy claim unsupported')
    scores = [m for m in test.values() if m['roc_auc'] is not None]
    require(equal['roc_auc'] == max(m['roc_auc'] for m in scores) and equal['binary_crossentropy'] == min(m['binary_crossentropy'] for m in scores), 'Equal score ranking unsupported')
    require(test['weighted_soft_voting']['accuracy'] < equal['accuracy'], 'Weighted comparison unsupported')
    return {'files_verified': len(preserved), 'files_withheld': len(withheld), 'methods': len(METHODS), 'images_per_partition': 106}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.repo_root), indent=2))
    print('Aggregate integrity verified. Private scores, checkpoint bytes and execution chronology are not independently verified.')


if __name__ == '__main__':
    main()
