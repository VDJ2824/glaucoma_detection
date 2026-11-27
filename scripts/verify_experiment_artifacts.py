"""Verify archived core-model results without loading images, models or the source ZIP."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODELS = ('mobilenet_v2', 'efficientnet_b0', 'resnet50')
STAGES = ('stage1', 'stage2')
MANIFEST = 'docs/experiments/core_models_artifact_manifest_2026-11-27.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_stage(directory, model, stage):
    metrics = read_json(directory / 'metrics' / f'{stage}.json')
    history = read_json(directory / 'histories' / stage / 'history.json')
    with (directory / 'histories' / stage / 'epochs.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    count = metrics['completed_epochs']
    require(metrics['model_name'] == model and metrics['stage'] == stage, f'{model}/{stage}: mislabeled summary')
    require(type(count) is int and count > 0 and len(rows) == count, f'{model}/{stage}: inconsistent epoch counts')
    require(history and all(len(values) == count for values in history.values()), f'{model}/{stage}: truncated history')
    require(set(rows[0]) == {'epoch', *history}, f'{model}/{stage}: CSV/history columns differ')
    differences, range_notes = [], []
    for index, row in enumerate(rows):
        require(int(row['epoch']) == index, f'{model}/{stage}: CSV must use contiguous zero-based epochs')
        for key, values in history.items():
            recorded, csv_value = float(values[index]), float(row[key])
            require(math.isfinite(recorded) and math.isfinite(csv_value), f'{model}/{stage}: nonfinite value')
            if key.endswith(('accuracy', 'auc', 'precision', 'recall')):
                require(-1e-6 <= recorded <= 1+1e-6, f'{model}/{stage}: metric outside numerical tolerance')
                if not 0 <= recorded <= 1:
                    range_notes.append({'model': model, 'stage': stage, 'epoch': index+1, 'metric': key, 'value': recorded})
            if not close(recorded, csv_value):
                require(key == 'learning_rate', f'{model}/{stage}: {key} mismatch at epoch {index+1}')
                require(0 < recorded < csv_value, f'{model}/{stage}: unexplained learning-rate discrepancy')
                if index+1 < count:
                    require(close(recorded, rows[index+1]['learning_rate']), f'{model}/{stage}: LR does not match next epoch')
                differences.append({'model': model, 'stage': stage, 'epoch': index+1,
                                    'csv_learning_rate': csv_value, 'json_learning_rate': recorded})
    best_index = min(range(count), key=lambda index: history['val_loss'][index])
    require(metrics['best_epoch'] == best_index+1, f'{model}/{stage}: best epoch not minimum validation loss')
    require(close(metrics['best_val_loss'], history['val_loss'][best_index]), f'{model}/{stage}: best validation loss mismatch')
    for key, value in metrics['best_epoch_metrics'].items():
        require(key in history and close(value, history[key][best_index]), f'{model}/{stage}: best-epoch metric mismatch: {key}')
    require(metrics['best_checkpoint'].endswith(f'/checkpoints/core/{model}/{stage}/best.keras'), f'{model}/{stage}: wrong checkpoint reference')
    return metrics, differences, range_notes, count


def verify_bundle(repo_root):
    root = Path(repo_root).resolve()
    manifest = read_json(root / MANIFEST)
    expected = set()
    for model in MODELS:
        for stage in STAGES:
            expected.update({f'histories/{model}/{stage}/epochs.csv', f'histories/{model}/{stage}/history.json', f'metrics/{model}/{stage}.json'})
        expected.update({f'metrics/{model}/selection.json', f'metrics/{model}/training_history.png'})
    require(len(manifest['files']) == 24 and {item['archive_member'] for item in manifest['files']} == expected,
            'Manifest does not cover exactly the 24 expected artifacts')
    require(set(manifest['model_directories']) == set(MODELS), 'Unexpected model directories')
    for item in manifest['files']:
        path = (root / item['repository_path']).resolve()
        require(path.is_relative_to(root / 'docs/experiments'), 'Artifact outside experiment documentation')
        payload = path.read_bytes()
        require(len(payload) == item['bytes'] and hashlib.sha256(payload).hexdigest() == item['sha256'],
                f'Artifact byte/hash mismatch: {item["repository_path"]}')
        if path.suffix == '.png':
            require(payload.startswith(b'\x89PNG\r\n\x1a\n'), 'Invalid PNG signature')
    bundles, differences, range_notes, epoch_count = {}, [], [], 0
    for model in MODELS:
        directory = root / manifest['model_directories'][model]
        bundle = {}
        for stage in STAGES:
            summary, stage_differences, stage_range_notes, count = verify_stage(directory, model, stage)
            bundle[stage] = summary
            differences.extend(stage_differences)
            range_notes.extend(stage_range_notes)
            epoch_count += count
        selection = read_json(directory / 'metrics/selection.json')
        best = min((bundle[stage] for stage in STAGES), key=lambda summary: summary['best_val_loss'])
        require(selection['stage'] == best['stage'] and selection['epoch'] == best['best_epoch'] and selection['checkpoint'] == best['best_checkpoint'],
                f'{model}: selection disagrees with stage summaries')
        require(selection['complete'] is True and close(selection['val_loss'], best['best_val_loss']), f'{model}: invalid selection record')
        require(bundle['stage1']['contract_hash'] == bundle['stage2']['contract_hash'] == selection['contract_hash'],
                f'{model}: stage contract hashes disagree')
        bundle['selection'] = selection
        bundles[model] = bundle
    # Preserve the older MobileNetV2 account and verify the archive corroborates it.
    existing = read_json(root / 'docs/experiments/mobilenet_v2_kaggle_2026-11-26/results.json')
    for key in (*STAGES, 'selection'):
        require(existing[key] == bundles['mobilenet_v2'][key], 'Archive differs from the preserved MobileNetV2 record')
    return {'files_verified': 24, 'stages_verified': 6, 'epoch_rows_verified': epoch_count,
            'learning_rate_timing_differences': differences, 'floating_point_range_notes': range_notes, 'models': bundles}


def comparison_data(verified):
    stages, deltas, selections = [], [], []
    for model, bundle in verified['models'].items():
        for stage in STAGES:
            summary = bundle[stage]
            stages.append({'model': model, 'stage': stage, 'best_epoch': summary['best_epoch'],
                           'completed_epochs': summary['completed_epochs'], **summary['best_epoch_metrics']})
        first, second = (bundle[stage] for stage in STAGES)
        a, b = first['best_epoch_metrics'], second['best_epoch_metrics']
        decrease = a['val_loss'] - b['val_loss']
        deltas.append({'model': model, 'validation_loss_decrease': decrease,
                       'validation_loss_relative_decrease_percent': 100*decrease/a['val_loss'],
                       'validation_accuracy_delta_percentage_points': 100*(b['val_accuracy']-a['val_accuracy']),
                       'validation_auc_delta': b['val_auc']-a['val_auc'],
                       'validation_precision_delta': b['val_precision']-a['val_precision'],
                       'validation_recall_delta': b['val_recall']-a['val_recall']})
        selections.append({'model': model, **bundle['selection']})
    return {'record_date': '2026-11-27', 'result_scope': 'validation-selected individual-model training',
            'test_results_provided': False, 'stage_metrics': stages, 'fine_tuning_deltas': deltas,
            'selected_checkpoints': selections,
            'verification': {key: value for key, value in verified.items() if key != 'models'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    verified = verify_bundle(args.repo_root)
    comparison = comparison_data(verified)
    stored = read_json(args.repo_root / 'docs/experiments/core_models_comparison_2026-11-27.json')
    require(stored == comparison, 'Stored comparison JSON differs from recomputed evidence')
    print(f'Verified {verified["files_verified"]} artifacts, {verified["stages_verified"]} stages and {verified["epoch_rows_verified"]} epoch rows.')
    print(f'Preserved {len(verified["learning_rate_timing_differences"])} documented learning-rate timing differences; other CSV/JSON metrics agree.')
    print('Best epochs, checkpoint selections, original MobileNetV2 record and derived comparison are consistent. No test results are asserted.')


if __name__ == '__main__':
    main()
