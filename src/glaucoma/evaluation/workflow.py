"""Validation -> frozen protocol -> separate opt-in final test inference."""
import json
import re
from pathlib import Path
from glaucoma.ensemble.predictions import MEMBERS, KEYS, align_predictions, check_threshold, validate_weights
from glaucoma.ensemble.weights import search_protocol, optimize_weights
from glaucoma.training.artifacts import write_json
from .checkpoints import canonical_hash, sha256_file, verify_selected_checkpoints, read_manifest
from .inference import predict_partition
from .comparison import save_evaluation, save_individual_evaluation


def run_directory(cfg, name='default', output_root=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name):
        raise ValueError('Use a simple evaluation run name')
    root = Path(output_root or cfg['artifacts']['root']).expanduser().resolve()/'phase4'/name
    dataset = Path(cfg['datasets'][cfg['datasets']['selected']]['root']).resolve()
    if root.resolve().is_relative_to(dataset):
        raise ValueError('Evaluation outputs must be outside original dataset storage')
    return root


def inference_contract(cfg, flat_image_root=None):
    return {'input_shape':cfg['model']['input_shape'],'preprocessing':cfg['preprocessing'],
            'dataset_name':cfg['datasets']['selected'],
            'dataset_root':str(Path(cfg['datasets'][cfg['datasets']['selected']]['root']).resolve()),
            'flat_image_root':str(Path(flat_image_root).resolve()) if flat_image_root else None,
            'seed':cfg['seed'],'batch_size':cfg['training']['batch_size'],
            'class_mapping':{'0':'non_glaucoma','1':'glaucoma'},'inference_training_argument':False}


def require_opt_in(enabled):
    if enabled is not True:
        raise PermissionError('Evaluation disabled; explicitly pass enabled=True')


def _write_once(path, value):
    """No API overwrites a frozen record or a consumed-test marker."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def _save_model_predictions(directory, predictions):
    directory.mkdir(parents=True,exist_ok=True)
    for name,frame in predictions.items():
        frame.to_csv(directory/f'{name}.csv',index=False)


def freeze_validation(aligned, checkpoints, graphs, cfg, manifest_path, directory, protocol=None, flat_image_root=None):
    """Freeze only a real validation table and verified checkpoint provenance.

    Low-level API also accepts synthetic tables in tests; it never opens test
    images. The high-level workflow produces these inputs from actual models.
    """
    directory = Path(directory)
    if (directory/'test_started.json').exists() or (directory/'frozen_ensemble.json').exists():
        raise FileExistsError('This run is frozen/test consumed; do not retune')
    protocol = protocol or search_protocol()
    weights,candidates = optimize_weights(aligned,protocol)
    validation_dir = directory/'validation'
    validation_dir.mkdir(parents=True,exist_ok=True)
    aligned_path = validation_dir/'aligned_predictions.csv'
    if aligned_path.exists():
        _require_same_predictions(aligned, _read_prediction_csv(aligned_path))
    else:
        aligned.to_csv(aligned_path,index=False)
    candidates.to_csv(validation_dir/'weight_search.csv',index=False)
    config = {'schema_version':1,'status':'frozen_before_test','members':list(MEMBERS),
              'protocol':protocol,'weights':weights,'threshold':protocol['threshold'],
              'validation_prediction_sha256':sha256_file(validation_dir/'aligned_predictions.csv'),
              'manifest_sha256':sha256_file(manifest_path),'inference_contract':inference_contract(cfg,flat_image_root),
              'checkpoints':checkpoints,'model_graph_sha256':graphs,
              'selection_partition':'validation','test_used_for_selection':False}
    record_path = directory/'validation_predictions.json'
    if record_path.is_file():
        config['model_validation_record_sha256'] = sha256_file(record_path)
    config['seal_sha256'] = canonical_hash(config)
    _write_once(directory/'frozen_ensemble.json',config)
    return config


def load_frozen(directory):
    directory = Path(directory)
    path = directory/'frozen_ensemble.json'
    if not path.is_file():
        raise FileNotFoundError('Freeze validation weights and threshold before final test access')
    config = json.loads(path.read_text())
    seal = config.get('seal_sha256')
    if canonical_hash({key:value for key,value in config.items() if key != 'seal_sha256'}) != seal:
        raise ValueError('Frozen configuration checksum mismatch')
    if (config.get('schema_version') != 1 or config.get('status') != 'frozen_before_test'
        or config.get('members') != list(MEMBERS) or config.get('selection_partition') != 'validation'
        or config.get('test_used_for_selection') is not False):
        raise ValueError('Invalid frozen protocol')
    validate_weights(config['weights']);check_threshold(config['threshold'])
    if config['protocol'] != search_protocol(config['protocol']['grid_denominator'],config['threshold']):
        raise ValueError('Frozen search/threshold policy changed')
    prediction_file = directory/'validation/aligned_predictions.csv'
    if sha256_file(prediction_file) != config['validation_prediction_sha256']:
        raise ValueError('Frozen validation predictions were modified')
    if 'model_validation_record_sha256' in config:
        if sha256_file(directory/'validation_predictions.json') != config['model_validation_record_sha256']:
            raise ValueError('Notebook 04 validation provenance changed after freeze')
        _read_validation_record(directory)
    return config


def _read_prediction_csv(path):
    import pandas as pd
    return pd.read_csv(path, dtype={'image_id':str, 'dataset_name':str}, float_precision='round_trip')


def _require_same_predictions(expected, actual):
    import pandas as pd
    try:
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Saved aligned validation predictions disagree with individual score files') from exc


def _read_validation_record(directory):
    directory = Path(directory)
    file = directory/'validation_predictions.json'
    if not file.is_file():
        raise FileNotFoundError('Run Notebook 04 model validation before Notebook 05 ensemble learning')
    record = json.loads(file.read_text())
    if canonical_hash({k:v for k,v in record.items() if k != 'seal_sha256'}) != record.get('seal_sha256'):
        raise ValueError('Notebook 04 validation record checksum mismatch')
    if (record.get('schema_version') != 1 or record.get('status') != 'individual_validation_complete'
        or record.get('partition') != 'validation' or record.get('members') != list(MEMBERS)):
        raise ValueError('Invalid Notebook 04 validation record')
    expected = {'validation/aligned_predictions.csv',
                *(f'validation/individual_predictions/{name}.csv' for name in MEMBERS)}
    if set(record['prediction_files']) != expected:
        raise ValueError('Missing or unexpected cached prediction files')
    for relative, digest in record['prediction_files'].items():
        if sha256_file(directory/relative) != digest:
            raise ValueError('Notebook 04 validation prediction file was modified')
    return record


def load_saved_validation(cfg, directory, manifest_path=None, flat_image_root=None):
    """Notebook 05 reads verified cached CSVs only; no CNN or image loading."""
    directory = Path(directory)
    record = _read_validation_record(directory)
    manifest_path = Path(manifest_path or cfg['metadata']['output'])
    if sha256_file(manifest_path) != record['manifest_sha256'] or inference_contract(cfg,flat_image_root) != record['inference_contract']:
        raise ValueError('Manifest or inference settings changed since Notebook 04')
    reference = read_manifest(manifest_path)
    reference = reference.loc[reference['split'].eq('validation')].sort_values(KEYS,kind='stable').reset_index(drop=True)
    individuals = {name:_read_prediction_csv(directory/f'validation/individual_predictions/{name}.csv') for name in MEMBERS}
    aligned = align_predictions(individuals, 'validation', reference)
    _require_same_predictions(aligned, _read_prediction_csv(directory/'validation/aligned_predictions.csv'))
    return aligned, record


def run_model_validation(cfg, manifest_path=None, *, enabled=False, name='default', output_root=None,
                         expected_records=None, checkpoint_roots=None, flat_image_root=None, threshold=0.5):
    """Notebook 04: selected checkpoints, validation scores and individual reports.

    No voting, ensemble optimization, freeze or test access happens here.
    """
    require_opt_in(enabled)
    threshold = check_threshold(threshold)
    manifest_path = Path(manifest_path or cfg['metadata']['output'])
    directory = run_directory(cfg,name,output_root)
    if flat_image_root and directory.resolve().is_relative_to(Path(flat_image_root).resolve()):
        raise ValueError('Evaluation outputs must be outside physical image storage')
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Use a fresh Phase 4 directory; model validation cannot overwrite a run')
    checkpoints = verify_selected_checkpoints(cfg,manifest_path,expected_records,checkpoint_roots)
    aligned,predictions,graphs = predict_partition(cfg,manifest_path,'validation',checkpoints,flat_image_root)
    _save_model_predictions(directory/'validation/individual_predictions',predictions)
    save_individual_evaluation(aligned,threshold,directory/'validation')
    files = ['validation/aligned_predictions.csv',
             *(f'validation/individual_predictions/{model}.csv' for model in MEMBERS)]
    record = {'schema_version':1, 'status':'individual_validation_complete', 'partition':'validation',
              'members':list(MEMBERS), 'individual_threshold':threshold,
              'manifest_sha256':sha256_file(manifest_path),'inference_contract':inference_contract(cfg,flat_image_root),
              'checkpoints':checkpoints,'model_graph_sha256':graphs,
              'prediction_files':{relative:sha256_file(directory/relative) for relative in files}}
    record['seal_sha256'] = canonical_hash(record)
    _write_once(directory/'validation_predictions.json',record)
    return {'run_directory':str(directory),'partition':'validation',
            'validation_record':str(directory/'validation_predictions.json'),
            'individual_metrics':str(directory/'validation/metrics.json')}


def run_ensemble_validation(cfg, manifest_path=None, *, enabled=False, name='default', output_root=None,
                            flat_image_root=None, protocol=None):
    """Notebook 05: reuse Notebook 04 predictions, optimize on validation, freeze.

    Does not load checkpoints, decode images or regenerate predictions.
    """
    require_opt_in(enabled)
    directory = run_directory(cfg,name,output_root)
    if flat_image_root and directory.resolve().is_relative_to(Path(flat_image_root).resolve()):
        raise ValueError('Evaluation outputs must be outside physical image storage')
    if (directory/'test_started.json').exists() or (directory/'frozen_ensemble.json').exists():
        raise FileExistsError('This run is frozen/test consumed; do not retune')
    manifest_path = Path(manifest_path or cfg['metadata']['output'])
    protocol = protocol or search_protocol()
    if protocol != search_protocol(protocol['grid_denominator'],protocol['threshold']):
        raise ValueError('Unsupported selection procedure')
    # Publish the predefined search before consuming cached validation scores.
    record = _read_validation_record(directory)
    if record['individual_threshold'] != protocol['threshold']:
        raise ValueError('Preserve the threshold predefined in Notebook 04; do not tune it in Notebook 05')
    protocol_path = directory/'predefined_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('The previously declared search protocol cannot be changed')
    else:
        _write_once(protocol_path,protocol)
    aligned,record = load_saved_validation(cfg,directory,manifest_path,flat_image_root)
    frozen = freeze_validation(aligned,record['checkpoints'],record['model_graph_sha256'],cfg,
                               manifest_path,directory,protocol,flat_image_root)
    save_evaluation(aligned,frozen['weights'],frozen['threshold'],directory/'validation/ensemble')
    return {'run_directory':str(directory),'partition':'validation',
            'frozen_ensemble':str(directory/'frozen_ensemble.json'),
            'weights':frozen['weights'],'threshold':frozen['threshold']}


def run_validation(cfg, manifest_path=None, *, enabled=False, name='default', output_root=None,
                   expected_records=None, checkpoint_roots=None, flat_image_root=None, protocol=None):
    """Compatibility orchestration of both validation steps; notebooks use separate APIs."""
    require_opt_in(enabled)
    protocol = protocol or search_protocol()
    run_model_validation(cfg,manifest_path,enabled=True,name=name,output_root=output_root,
                         expected_records=expected_records,checkpoint_roots=checkpoint_roots,
                         flat_image_root=flat_image_root,threshold=protocol['threshold'])
    return run_ensemble_validation(cfg,manifest_path,enabled=True,name=name,output_root=output_root,
                                   flat_image_root=flat_image_root,protocol=protocol)


def run_final_test(cfg, manifest_path=None, *, enabled=False, name='default', output_root=None,
                   expected_records=None, checkpoint_roots=None, flat_image_root=None):
    require_opt_in(enabled)
    directory = run_directory(cfg,name,output_root)
    if flat_image_root and directory.resolve().is_relative_to(Path(flat_image_root).resolve()):
        raise ValueError('Evaluation outputs must be outside physical image storage')
    frozen = load_frozen(directory)  # MUST precede even reading the test partition
    if (directory/'test_started.json').exists():
        raise FileExistsError('Final test already started for this frozen run; never retune using its results')
    manifest_path = Path(manifest_path or cfg['metadata']['output'])
    if sha256_file(manifest_path) != frozen['manifest_sha256'] or inference_contract(cfg,flat_image_root) != frozen['inference_contract']:
        raise ValueError('Frozen manifest or inference configuration changed')
    checkpoints = verify_selected_checkpoints(cfg,manifest_path,expected_records,checkpoint_roots)
    if checkpoints != frozen['checkpoints']:
        raise ValueError('Selected checkpoint/contract provenance changed after freeze')
    _write_once(directory/'test_started.json',{'frozen_seal':frozen['seal_sha256'],'status':'test_access_started'})
    # Only here may labels/paths/images of the test partition be consumed.
    aligned,predictions,graphs = predict_partition(cfg,manifest_path,'test',checkpoints,flat_image_root,expected_graphs=frozen['model_graph_sha256'])
    if graphs != frozen['model_graph_sha256']:
        raise ValueError('Loaded model graphs differ from the validation run')
    import pandas as pd
    validation = pd.read_csv(directory/'validation/aligned_predictions.csv',dtype={'image_id':str,'dataset_name':str})
    if set(zip(aligned.dataset_name,aligned.image_id)) & set(zip(validation.dataset_name,validation.image_id)):
        raise ValueError('Validation/test image overlap')
    _save_model_predictions(directory/'test/individual_predictions',predictions)
    metrics = save_evaluation(aligned,frozen['weights'],frozen['threshold'],directory/'test')
    write_json(directory/'test_complete.json',{'frozen_seal':frozen['seal_sha256'],'partition':'test','n':len(aligned),
               'note':'Fixed protocol evaluated; no guarantee that an ensemble outperforms individual models.'})
    return {'run_directory':str(directory),'partition':'test','metrics':metrics}


def copy_model_validation(source_directory, destination_directory):
    """Stage a pre-ensemble Notebook 04 run from read-only Kaggle inputs.

    Copy reports byte-for-byte into a fresh writable artifact directory. Never
    copy or reset an already frozen/test-consumed experiment to enable retuning.
    No images/checkpoints are loaded and no predictions are regenerated.
    """
    import shutil
    source, target = Path(source_directory).resolve(), Path(destination_directory).resolve()
    if any((source/name).exists() for name in ('frozen_ensemble.json','test_started.json','test_complete.json','predefined_protocol.json')):
        raise ValueError('Only a pre-ensemble Notebook 04 handoff may be copied')
    _read_validation_record(source)
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Source and destination must be separate artifact directories')
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Do not overwrite an existing evaluation run')
    sources = [source/'validation_predictions.json', source/'validation']
    if any(p.is_symlink() for parent in sources for p in [parent,*parent.rglob('*')]):
        raise ValueError('Symlinked handoff artifacts are not supported')
    target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'validation_predictions.json',target/'validation_predictions.json')
    shutil.copytree(source/'validation',target/'validation')
    _read_validation_record(target)
    return str(target)
