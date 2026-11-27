"""Validation -> frozen protocol -> separate opt-in final test inference."""
import json
import re
from pathlib import Path
from glaucoma.ensemble.predictions import MEMBERS, check_threshold, validate_weights
from glaucoma.ensemble.weights import search_protocol, optimize_weights
from glaucoma.training.artifacts import write_json
from .checkpoints import canonical_hash, sha256_file, verify_selected_checkpoints
from .inference import predict_partition
from .comparison import save_evaluation


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
    aligned.to_csv(validation_dir/'aligned_predictions.csv',index=False)
    candidates.to_csv(validation_dir/'weight_search.csv',index=False)
    config = {'schema_version':1,'status':'frozen_before_test','members':list(MEMBERS),
              'protocol':protocol,'weights':weights,'threshold':protocol['threshold'],
              'validation_prediction_sha256':sha256_file(validation_dir/'aligned_predictions.csv'),
              'manifest_sha256':sha256_file(manifest_path),'inference_contract':inference_contract(cfg,flat_image_root),
              'checkpoints':checkpoints,'model_graph_sha256':graphs,
              'selection_partition':'validation','test_used_for_selection':False}
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
    return config


def run_validation(cfg, manifest_path=None, *, enabled=False, name='default', output_root=None,
                   expected_records=None, checkpoint_roots=None, flat_image_root=None, protocol=None):
    require_opt_in(enabled)
    manifest_path = Path(manifest_path or cfg['metadata']['output'])
    directory = run_directory(cfg,name,output_root)
    if flat_image_root and directory.resolve().is_relative_to(Path(flat_image_root).resolve()):
        raise ValueError('Evaluation outputs must be outside physical image storage')
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Use a fresh Phase 4 directory; validation cannot overwrite a run')
    # Predefine the policy before inference. No threshold or protocol tuning API.
    protocol = protocol or search_protocol()
    if protocol != search_protocol(protocol['grid_denominator'],protocol['threshold']):
        raise ValueError('Unsupported selection procedure')
    checkpoints = verify_selected_checkpoints(cfg,manifest_path,expected_records,checkpoint_roots)
    directory.mkdir(parents=True,exist_ok=True)
    _write_once(directory/'predefined_protocol.json',protocol)
    aligned,predictions,graphs = predict_partition(cfg,manifest_path,'validation',checkpoints,flat_image_root)
    _save_model_predictions(directory/'validation/individual_predictions',predictions)
    frozen = freeze_validation(aligned,checkpoints,graphs,cfg,manifest_path,directory,protocol,flat_image_root)
    save_evaluation(aligned,frozen['weights'],frozen['threshold'],directory/'validation')
    return {'run_directory':str(directory),'frozen_ensemble':str(directory/'frozen_ensemble.json'),
            'weights':frozen['weights'],'threshold':frozen['threshold'],'partition':'validation'}


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
