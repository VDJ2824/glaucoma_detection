"""Persistent epoch-boundary checkpoints, histories and callback recovery."""
import math
from .artifacts import write_json

DEFAULT_REDUCE_LR = {'factor': 0.5, 'patience': 2, 'min_delta': 0.0001, 'min_lr': 1e-7, 'cooldown': 0}


def make_callbacks(cfg, paths, contract_hash, state=None):
    import tensorflow as tf
    state = state or {}
    early_cfg = cfg['training']['early_stopping']
    lr_cfg = {**DEFAULT_REDUCE_LR, **cfg['training'].get('reduce_lr', {})}
    early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=early_cfg['patience'],
        min_delta=early_cfg['min_delta'], restore_best_weights=early_cfg['restore_best_weights'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(paths['best']), monitor='val_loss', mode='min', save_best_only=True,
        initial_value_threshold=state.get('best_val_loss'))
    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', mode='min', factor=lr_cfg.get('factor', 0.5),
        patience=lr_cfg.get('patience', 2), min_delta=lr_cfg.get('min_delta', 0.0001),
        min_lr=lr_cfg.get('min_lr', 1e-7), cooldown=lr_cfg.get('cooldown', 0))
    csv = tf.keras.callbacks.CSVLogger(str(paths['csv']), append=bool(state))

    class Recorder(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history = state.get('history', {})
            self.best = state.get('best_val_loss', float('inf'))
            self.best_epoch = state.get('best_epoch', 0)

        def on_train_begin(self, logs=None):
            if not state:
                return
            # These callbacks reset on_train_begin; restore afterwards.
            for callback, key in ((early, 'early_stopping'), (reduce, 'reduce_lr')):
                for attribute, value in state.get(key, {}).items():
                    setattr(callback, attribute, value)
            if early.restore_best_weights and paths['best'].is_file():
                early.best_weights = tf.keras.models.load_model(paths['best'], compile=False).get_weights()

        def on_epoch_end(self, epoch, logs=None):
            logs = {key: float(value) for key, value in (logs or {}).items()}
            if 'val_loss' not in logs or not all(math.isfinite(value) for value in logs.values()):
                raise FloatingPointError('Nonfinite or missing metrics; recover the previous good checkpoint')
            if logs['val_loss'] < self.best:
                self.best, self.best_epoch = logs['val_loss'], epoch + 1
            logs['learning_rate'] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            for key, value in logs.items():
                self.history.setdefault(key, []).append(value)
            def callback_state(callback, keys):
                return {key: (int(getattr(callback, key)) if key in ('wait', 'best_epoch', 'cooldown_counter')
                              else float(getattr(callback, key))) for key in keys}
            record = {'contract_hash': contract_hash, 'completed_epochs': epoch + 1, 'complete': False,
                      'best_val_loss': self.best, 'best_epoch': self.best_epoch, 'history': self.history,
                      'stopped_early': bool(self.model.stop_training),
                      'optimizer_iterations': int(self.model.optimizer.iterations.numpy()),
                      'early_stopping': callback_state(early, ('best', 'wait', 'best_epoch')),
                      'reduce_lr': callback_state(reduce, ('best', 'wait', 'cooldown_counter'))}
            # Save last before state, and detect a mismatched interrupted write on resume.
            temporary = paths['last'].with_name('last.tmp.keras')
            self.model.save(temporary)
            temporary.replace(paths['last'])
            write_json(paths['history'], self.history)
            write_json(paths['state'], record)

    for key in ('best', 'csv', 'history', 'state'):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    return [tf.keras.callbacks.TerminateOnNaN(), early, checkpoint, reduce, csv, Recorder()]
