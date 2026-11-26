# Phase 3: individual core CNN transfer learning

This phase implements core construction and two-stage training. Existing dataset processing and notebooks 00–02 remain unchanged, including the latest Colab edits pulled before implementation. No dataset is downloaded or trained during development; verification uses synthetic images/tensors and randomly initialized models. A separately supplied completed Kaggle experiment is now documented outside the reusable notebook.

## Runtime and installation

Use Python 3.10–3.13. Dependency markers retain **TensorFlow 2.20.0** on Python 3.10–3.12 and select **TensorFlow 2.21.x** on Python 3.13, matching the supplied successful Kaggle environment. The implementation uses public `tf.keras` APIs, native `.keras` checkpoints and finite `tf.data` inputs; it does not enable legacy `tf_keras` or depend on a particular CUDA device. CPU tests cover TensorFlow 2.20.0 with Keras 3.15.1 and Python 3.12. The run contract also records the actual TensorFlow/Keras/Python versions. GPU availability, Drive I/O and ImageNet retrieval still require checks in the actual hosted environment.

```bash
source .venv/bin/activate
python -m pip install -r requirements-training.txt -r requirements-dev.txt
python -m pytest -q
```

The data-only development install can run non-training tests without TensorFlow; `tests/test_training.py` is explicitly skipped if TensorFlow is absent. Install training dependencies to exercise actual model/fit tests. `requirements.txt` and notebook research installs use the same Python-version-specific TensorFlow/NumPy markers. Installing packages may replace preinstalled versions; restart the notebook kernel afterwards. TensorFlow.js conversion remains a separate, future compatibility exercise; this phase does not claim conversion parity.

## Architecture and preprocessing

`models/factory.py` exposes `build_model(name, weights='imagenet')` for `mobilenet_v2`, `efficientnet_b0`, and `resnet50`. Explicit model construction with ImageNet initialization can fetch weights; `weights=None` is offline. Imports alone never construct or download a model. `models/core.py` re-exports the core API; `models/registry.py` contains only these three applications. Optional VGG baseline code is separate and is not accepted by the core factory/trainer.

Each application uses `include_top=False` and `(224,224,3)`, followed by global average pooling, dropout 0.2 and one sigmoid output. The existing preprocessing remains the only external normalization: MobileNetV2 RGB [-1,1], ResNet50 BGR/mean subtraction, and EfficientNetB0 float RGB [0,255] for its built-in rescaling. No second normalization, new augmentation or segmentation stage is inserted. See [Keras EfficientNet input contract](https://keras.io/api/applications/efficientnet/efficientnet_models/).

Stage 1 freezes the backbone, compiles Adam at 0.001 with binary crossentropy and trains the head for at most 10 epochs. Stage 2 loads the **best Stage 1 checkpoint from disk**, unfreezes the last 20 backbone layers while leaving BatchNormalization layers frozen, and recompiles fresh Adam at 0.0001 for at most 20 epochs. The last-20 window counts all backbone layers before excluding BatchNormalization. The backbone is always called with `training=False` to preserve BatchNormalization statistics; this also disables backbone stochastic layers while head dropout still trains. See [TensorFlow transfer-learning guidance](https://www.tensorflow.org/guide/keras/transfer_learning).

EarlyStopping and best-checkpoint selection monitor `val_loss` in `min` mode. ReduceLROnPlateau defaults to factor 0.5, patience 2, min_delta 0.0001, min_lr 1e-7 and cooldown 0. CSVLogger and JSON records persist epoch metrics. Both stages record loss, accuracy, precision, recall and ROC-AUC for training/validation, with a sigmoid threshold of 0.5 for thresholded metrics. These are research measurements, not established clinical operating points. Do not interpret AUC from a partition with only one class as useful discrimination.

## Frozen data loading

`training/data.py` reads the already saved six-column metadata CSV. It discards test rows before validating image paths and labels, requires nonempty train/validation partitions, checks duplicate IDs/paths and supplied patient-group isolation, and passes only those partitions to the unmodified `iter_batches`. It never invokes adapters, `prepare_dataset`, splitters or deduplication and never opens test images. Test annotations may exist in the single CSV, but no test pixels, labels or predictions enter fitting or checkpoint selection.

The wrapper yields float32 `[B,224,224,3]` images and `[B,1]` labels, includes the final partial batch and exposes finite cardinality. Training shuffling/augmentation is already performed by `iter_batches`; Keras-level shuffling is disabled. Validation always uses deterministic preprocessing without augmentation. Do not move original files during a run; frozen absolute paths must remain valid. Do not edit the CSV or regenerate splits to resume a model.

## Opt-in API

```python
from glaucoma.utils.reproducibility import set_random_seed
from glaucoma.models.factory import build_model
from glaucoma.training.trainer import train_stage, select_best_checkpoint, recover_best_model

# cfg is a copy of the existing saved session, with the chosen model set.
# Use a fresh configured run directory for every independent experiment.
set_random_seed(cfg['seed'])
model = build_model(cfg['model']['selected'], weights=cfg['model']['weights'])
# These calls deliberately opt in to actual training; do not run as a smoke test.
stage1 = train_stage(cfg, 'stage1', model=model, enabled=True)
stage2 = train_stage(cfg, 'stage2', enabled=True)
selection = select_best_checkpoint(cfg, save_selection=True)
recovered = recover_best_model(cfg)  # loads a real checkpoint; never fits
```

`enabled=True` is required by the trainer. The factory does not fit a model. Stage 1 requires an explicitly built model unless resuming. Stage 2 ignores in-memory models and starts from the saved best Stage 1 checkpoint; a completed, matching Stage 1 is required for new fine-tuning. Stage limits/rates, head contract and BatchNormalization policy are validated. The config's model weights field must describe the actual initialization (`None` for offline/random experiments). A random model is not ImageNet transfer learning.

The CLI is also opt-in:

```bash
python scripts/train_core.py --session /path/session.json --model mobilenet_v2 --stage stage1
# The above refuses to fit. When intentionally ready, explicitly add:
# --enable-training --allow-imagenet-download
# Use --weights none for deliberately random-initialized offline experiments.
```

Supported stages are `stage1`, `stage2`, `both`. `--resume` resumes a single explicit stage, not both. `--metadata` may point to an existing frozen CSV. The CLI does not produce new splits or implement optional VGG training.

## Persistence and recovery

With notebook 00's saved Drive configuration:

```text
<run>/checkpoints/core/<model>/<stage>/
  best.keras       # full model and optimizer at lowest validation loss
  last.keras       # full model and optimizer at last committed epoch
  contract.json    # model/settings, frozen train/validation digest and runtime versions
  state.json       # completed epochs, callback/optimizer state and history
<run>/histories/<model>/<stage>/
  history.json
  epochs.csv
<run>/metrics/<model>/
  stage1.json
  stage2.json
  selection.json
```

Best checkpoint and early stopping have distinct semantics: the checkpoint retains every strictly better validation loss, while early stopping uses its configured `min_delta`. Stage summaries contain actual logged metrics at the chosen best epoch, not test metrics. Global selection compares Stage 1 and Stage 2 validation losses, retaining Stage 1 on ties. Fine-tuning can perform worse; the Stage 1 checkpoint remains available. Selection can also identify a recoverable best checkpoint from an interrupted stage and marks its `complete` status explicitly.

New training refuses to overwrite a stage directory. For independent runs, use a new run directory while pointing to the same verified frozen metadata. `resume=True` loads `last.keras`, restores optimizer state and callback counters, and appends history from the last completed epoch. Settings and the frozen train/validation manifest must match. Final epoch interruption before finalization is handled without repeating that epoch. Callback state/history are committed after a completed epoch, so interrupted batches do not produce recovery points. Files are written via temporary paths and rename; Drive must be mounted/writable and still needs external backup. A checkpoint/state iteration mismatch fails rather than silently mixing partial writes; recover `best.keras` for inspection or a separately documented new experiment.

The seed helper initializes Python, NumPy and TensorFlow and enables deterministic ops. Seeds/epochs drive existing augmentation streams; exact augmentation replay after interrupted input prefetch, GPU kernels, software changes or cross-platform restarts is not guaranteed. Record the runtime, source commit, session config and manifests with each experiment. Best recovery is not a claim of bit-for-bit uninterrupted training.

## Notebook 03

Notebook 03 is independent: configure repository/session paths, optionally mount Drive, clone a nonexistent destination and install the training manifest. It adds `src` to imports and loads the saved session without rewriting it. The default code path constructs no model, reads no images, fits nothing and writes no training artifacts.

1. For offline verification, set `BUILD_MODEL=True`, `WEIGHTS=None`, and `RUN_FORWARD_SMOKE=True`. Leave both stage flags false. The summary and synthetic forward test do not require metadata/images or pretrained downloads.
2. To inspect the saved CSV only, set `LOAD_METADATA=True` and point `METADATA_CSV` to the actual existing file if it differs from the session. No splitting or image decoding occurs.
3. When deliberately ready for transfer learning, use `WEIGHTS='imagenet'`, allow the download, build one model and enable Stage 1. Set the same initialization/config for later recovery. Enable Stage 2 only after Stage 1 completes.
4. Recovery uses `RESUME_STAGE1` or `RESUME_STAGE2` with the corresponding stage flag; do not construct a new model for resume. Keep all configuration/manifest settings unchanged.
5. Enable `SHOW_HISTORY` to plot existing logged train/validation curves. Optional plot saving uses the configured persistent metrics directory. The final cell records best-checkpoint selection only after an enabled training stage.

Old session snapshots do not automatically pick up new YAML keys: Notebook 03 supplies the documented ReduceLROnPlateau defaults if absent. Existing stage settings should already be 10/20 epochs at 0.001/0.0001; inspect them before training. Notebook 00–02 edits and saved sessions are not overwritten. No new accuracy, loss, AUC or clinical performance is claimed by implementation tests.


## Kaggle portability and Python 3.13

The supplied executed notebook demonstrates Python 3.13.16 / TensorFlow 2.21.0 / Keras 3.13.2 on Kaggle; see the [experiment provenance](experiments/mobilenet_v2_kaggle_2026-11-26/README.md). Current local unit tests run under Python 3.12 / TensorFlow 2.20.0. Python-version guard/marker tests are simulated; they are not a claim that the complete suite ran locally on Python 3.13.

The package now permits Python >=3.10,<3.14. On Python 3.13, NumPy >=2.1 is needed instead of the previous <2.1 constraint ([NumPy 2.1 release notes](https://numpy.org/doc/stable/release/2.1.0-notes.html)); the TensorFlow dependency selects the observed 2.21 line ([official supported wheels](https://www.tensorflow.org/install/pip)). Older local/Colab environments retain their tested dependency selections. Conversion remains a separate compatibility task. Do not reinstall/downgrade a working hosted GPU stack just to satisfy the old pinned training manifest.

Notebook 03 uses Kaggle environment markers or `/kaggle/working` before checking Colab's `/content`; it never hard-codes Kaggle for all users. Colab still mounts Drive explicitly. Local runs use the repository and its artifact directory. Notebook 03 is independent, so earlier user-edited notebooks 00–02 have not been rewritten.

For a Drive-to-Kaggle bundle:

1. Attach your existing images and frozen session/CSV privately. Set `KAGGLE_INPUT_ROOT` to the actual directory directly containing `database/`, `acrima_metadata.csv` and `session.json`. The defaults are placeholders, not a dataset discovery/download command.
2. Set `RELOCATE_FROZEN_SESSION=True`, leave `METADATA_CSV=None`, and choose a writable `OUTPUT_ARTIFACT_ROOT` plus `RUN_NAME`. This explicitly calls `relocate_frozen_session` in the shared notebook runtime utility.
3. The utility reads metadata as strings, remaps only `image_path` by basename to the existing flat image directory, checks every target exists and rejects basename collisions. It preserves row order and all other columns (including labels, splits and grouping keys), clones the existing configuration rather than rebuilding training settings, and creates derived files under `<artifact_root>/runs/<RUN_NAME>`.
4. Source CSV/JSON/images are untouched. Existing derived files are reused only if identical; conflicting outputs fail instead of being overwritten. For subsequent runs set `SESSION_FILE` to the printed derived session and disable relocation. Different directory structures need a separately verified path mapping, not a guessed basename collapse.
5. Training remains separately opt-in. No helper regenerates splits, changes a model or relaxes checkpoint-contract checks. In particular, relocating a manifest does not rewrite old checkpoint contracts: recovery requires the effective paths/configuration originally used for that checkpoint.

The helper checks path existence across the frozen manifest during explicit portability setup; it does not decode any images. Training continues to use only the existing train/validation loader and never test pixels. The notebook no longer creates backup ZIPs automatically. Keep model backups outside Git, save Kaggle outputs/export them privately, and retain original metrics/provenance alongside them.
