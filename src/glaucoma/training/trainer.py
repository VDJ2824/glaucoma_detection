"""Opt-in core-only two-stage training of a frozen metadata manifest."""
import copy
import hashlib
import json
import platform
from pathlib import Path
from glaucoma.models.factory import get_backbone
from glaucoma.models.registry import CORE_APPLICATIONS
from glaucoma.utils.reproducibility import set_random_seed
from .artifacts import stage_paths, write_json
from .callbacks import DEFAULT_REDUCE_LR, make_callbacks
from .data import load_training_metadata, make_dataset


def configure_stage(model, stage, last_n_layers=20):
    """Change trainability then compile a fresh Adam optimizer for each new stage."""
    import tensorflow as tf
    if stage not in ('stage1', 'stage2') or last_n_layers != 20:
        raise ValueError('Expected stage1/stage2 and a last-20-layer fine-tuning window')
    backbone = get_backbone(model)
    backbone.trainable = stage == 'stage2'
    if stage == 'stage2':
        for index, layer in enumerate(backbone.layers):
            layer.trainable = index >= len(backbone.layers) - last_n_layers and not isinstance(layer, tf.keras.layers.BatchNormalization)
    compile_model(model, 0.001 if stage == 'stage1' else 0.0001)
    return model


def compile_model(model, learning_rate):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=[
                      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(name='auc')])


def _validate_settings(cfg):
    if cfg['model']['selected'] not in CORE_APPLICATIONS:
        raise ValueError('Core training excludes VGG baselines')
    if cfg['model']['head'] != {'pooling': 'global_average', 'dropout': 0.2, 'units': 1, 'activation': 'sigmoid'}:
        raise ValueError('Phase 3 requires GAP, Dropout(0.2) and one sigmoid output')
    if cfg['model']['input_shape'] != [224, 224, 3]:
        raise ValueError('Expected input shape [224,224,3]')
    for stage, maximum, rate in [('stage1', 10, .001), ('stage2', 20, .0001)]:
        settings = cfg['training'][stage]
        if type(settings['epochs']) is not int or not 1 <= settings['epochs'] <= maximum or settings['learning_rate'] != rate:
            raise ValueError(f'{stage} requires at most {maximum} epochs and learning rate {rate}')
    if cfg['training']['stage2']['unfreeze_last_n_layers'] != 20 or not cfg['training']['stage2']['freeze_batch_normalization']:
        raise ValueError('Stage 2 must unfreeze the last 20 layers while freezing BatchNormalization')
    if cfg['model'].get('weights') not in ('imagenet', None):
        raise ValueError('Record imagenet or None initialization in the run configuration')
    if type(cfg['training']['batch_size']) is not int or cfg['training']['batch_size'] < 1:
        raise ValueError('Batch size must be a positive integer')
    reduce = cfg['training']['reduce_lr']
    if not 0 < reduce['factor'] < 1 or not 0 < reduce['min_lr'] <= 0.0001 or reduce['min_delta'] < 0:
        raise ValueError('Invalid ReduceLROnPlateau settings')
    if any(type(reduce[key]) is not int or reduce[key] < 0 for key in ('patience', 'cooldown')):
        raise ValueError('LR patience/cooldown must be nonnegative integers')
    early = cfg['training']['early_stopping']
    if early['monitor'] != 'val_loss' or early['mode'] != 'min':
        raise ValueError('Checkpoint selection and stopping require val_loss/min')


def _contract(cfg, metadata):
    # Record the frozen training/validation rows, never read test images.
    manifest = hashlib.sha256(metadata.to_csv(index=False).encode()).hexdigest()
    contract = {key: cfg[key] for key in ('model', 'training', 'preprocessing', 'augmentation', 'seed')}
    contract['training_validation_manifest_sha256'] = manifest
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return contract, digest


def train_stage(cfg, stage, *, model=None, enabled=False, resume=False, metadata_path=None, verbose=1):
    """Train only with enabled=True. Never build models, resplit or use test pixels.

    Stage 1 requires an explicitly constructed core model; Stage 2 ALWAYS starts
    from the saved best Stage 1 checkpoint. Resume restores last.keras at a fully
    committed epoch boundary, including optimizer/callback states. An existing
    run cannot be overwritten without choosing a new artifact/run directory.
    """
    if enabled is not True:
        raise PermissionError('Training is disabled; explicitly pass enabled=True')
    cfg = copy.deepcopy(cfg)
    cfg['training']['reduce_lr'] = {**DEFAULT_REDUCE_LR, **cfg['training'].get('reduce_lr', {})}
    _validate_settings(cfg)
    paths = stage_paths(cfg, stage)
    metadata = load_training_metadata(metadata_path or cfg['metadata']['output'], cfg)
    contract, digest = _contract(cfg, metadata)
    set_random_seed(cfg['seed'])
    import tensorflow as tf
    state = None
    initial_epoch = 0
    if resume:
        if not paths['state'].is_file() or not paths['last'].is_file():
            raise FileNotFoundError('No complete epoch-boundary recovery state found')
        state = json.loads(paths['state'].read_text())
        if state['contract_hash'] != digest:
            raise ValueError('Recovery settings or frozen metadata differ from the saved contract')
        if state['complete']:
            raise ValueError('Stage is already complete; recover its best checkpoint instead')
        model = tf.keras.models.load_model(paths['last'])
        if int(model.optimizer.iterations.numpy()) != state['optimizer_iterations']:
            raise ValueError('Interrupted checkpoint/state write; use best checkpoint recovery in a new run')
        initial_epoch = state['completed_epochs']
    else:
        if paths['directory'].exists() and any(paths['directory'].iterdir()):
            raise FileExistsError('Stage artifacts exist; resume explicitly or use a new run directory')
        if stage == 'stage2':
            previous = stage_paths(cfg, 'stage1')
            if not previous['state'].is_file() or not previous['best'].is_file():
                raise FileNotFoundError('Stage 2 requires a saved best Stage 1 checkpoint and state')
            previous_state = json.loads(previous['state'].read_text())
            if previous_state['contract_hash'] != digest or not previous_state['complete']:
                raise ValueError('Complete Stage 1 with matching frozen metadata/settings before Stage 2')
            model = tf.keras.models.load_model(previous['best'], compile=False)
        elif model is None:
            raise ValueError('Construct a core model explicitly before starting Stage 1')
        if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, 1):
            raise ValueError('Model tensor shapes do not match the Phase 3 contract')
        if model.name != cfg['model']['selected']:
            raise ValueError('Model name does not match selected preprocessing architecture')
        configure_stage(model, stage)
    if initial_epoch >= cfg['training'][stage]['epochs']:
        # An interruption after the final epoch but before finalization is recoverable.
        if not state:
            raise ValueError('No epochs available')
    paths['directory'].mkdir(parents=True, exist_ok=True)
    write_json(paths['contract'], {**contract, 'runtime': {
        'tensorflow': tf.__version__, 'keras': tf.keras.__version__, 'python': platform.python_version()}})
    if initial_epoch < cfg['training'][stage]['epochs'] and not (state and state.get('stopped_early')):
        model.fit(make_dataset(metadata, cfg, 'train', start_epoch=initial_epoch),
                  validation_data=make_dataset(metadata, cfg, 'validation'),
                  epochs=cfg['training'][stage]['epochs'], initial_epoch=initial_epoch,
                  callbacks=make_callbacks(cfg, paths, digest, state), shuffle=False, verbose=verbose)
    if not paths['best'].is_file() or not paths['state'].is_file():
        raise RuntimeError('No valid best checkpoint saved; inspect validation metrics')
    state = json.loads(paths['state'].read_text())
    state['complete'] = True
    write_json(paths['state'], state)
    best_index = state['best_epoch'] - 1
    result = {'model_name': cfg['model']['selected'], 'stage': stage,
              'best_checkpoint': str(paths['best']), 'last_checkpoint': str(paths['last']),
              'best_epoch': state['best_epoch'], 'completed_epochs': state['completed_epochs'],
              'best_val_loss': state['best_val_loss'],
              'best_epoch_metrics': {key: values[best_index] for key, values in state['history'].items()},
              'history_path': str(paths['history']), 'contract_hash': digest,
              'tensorflow_version': tf.__version__, 'keras_version': tf.keras.__version__}
    write_json(paths['metrics'], result)
    return result


def select_best_checkpoint(cfg, save_selection=False):
    """Choose lowest validation loss across stages; never evaluate the test set."""
    candidates = []
    for stage in ('stage1', 'stage2'):
        paths = stage_paths(cfg, stage)
        if paths['state'].is_file() and paths['best'].is_file():
            state = json.loads(paths['state'].read_text())
            candidates.append({'stage': stage, 'checkpoint': str(paths['best']),
                               'val_loss': state['best_val_loss'], 'epoch': state['best_epoch'],
                               'complete': state['complete'], 'contract_hash': state['contract_hash']})
    if not candidates:
        raise FileNotFoundError('No best checkpoint available for this model/run')
    if len({item['contract_hash'] for item in candidates}) != 1:
        raise ValueError('Stage checkpoint contracts differ')
    selected = min(candidates, key=lambda item: item['val_loss'])
    if save_selection:
        write_json(stage_paths(cfg, 'stage1')['selection'], selected)
    return selected


def recover_best_model(cfg, stage=None):
    """Load a full best .keras model including optimizer; does not start training."""
    import tensorflow as tf
    path = stage_paths(cfg, stage)['best'] if stage else Path(select_best_checkpoint(cfg)['checkpoint'])
    if not path.is_file():
        raise FileNotFoundError(f'Best checkpoint is missing: {path}')
    return tf.keras.models.load_model(path)


def plot_training_history(cfg):
    """Plot recorded train/validation histories only; never manufacture metrics."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    offset = 0
    for stage in ('stage1', 'stage2'):
        path = stage_paths(cfg, stage)['history']
        if not path.is_file():
            continue
        history = json.loads(path.read_text())
        epochs = list(range(offset + 1, offset + len(history['loss']) + 1))
        for ax, metric in zip(axes, ('loss', 'accuracy')):
            for key in (metric, f'val_{metric}'):
                ax.plot(epochs, history[key], label=f'{stage} {key}')
            ax.set(xlabel='Completed epoch', ylabel=metric)
        offset += len(history['loss'])
    if offset == 0:
        plt.close(fig)
        raise FileNotFoundError('No recorded training history for this model/run')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    return fig
