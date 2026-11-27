# Phase 4: validation-selected ensemble and final test evaluation

Phase 4 implements checkpoint verification, deterministic prediction, majority/equal/weighted voting, predefined validation weight selection, frozen protocol records, metrics, plots and disagreement analysis. **No real Phase 4 predictions, optimized weights or test scores have been generated during implementation.** The actual private Kaggle checkpoints are not in this repository. Existing training histories under `docs/experiments/` remain historical training/validation records, not Phase 4 inference outputs.

## Prerequisites and checkpoint identity

Use the saved training session and the original canonical frozen metadata CSV. Provide the full private checkpoint tree for both stages of all three core models:

```text
<checkpoint-base>/
  mobilenet_v2/stage1/{best.keras,state.json,contract.json}
  mobilenet_v2/stage2/{best.keras,state.json,contract.json}
  efficientnet_b0/stage1/...
  efficientnet_b0/stage2/...
  resnet50/stage1/...
  resnet50/stage2/...
```

The archive of metrics/graphs alone cannot substitute for these files. Notebook 04 defaults to the committed per-model `metrics/selection.json` references for the completed ACRIMA experiments. Bases may be configured per model for private Kaggle attachments; stored checkpoint path strings need not match the current attachment path, but model/stage identities and provenance must match.

`evaluation/checkpoints.py` reuses Phase 3's `select_best_checkpoint` and `recover_best_model`. It requires both completed stages, recalculates each saved contract hash (excluding the separate runtime field), verifies the original training/validation manifest digest and compares the selected stage/epoch/loss/hash against the provided experiment record. It verifies input/head/preprocessing/seed compatibility and hashes actual selected checkpoint bytes. Model loading then checks the core name, `(224,224,3)` input, single sigmoid output, expected backbone family/feature dimensions and the original GAP → Dropout(0.2) → Dense(1) head. VGG is never admitted. These structural checks and fingerprints verify the supplied run; they are not an adversarial proof of training provenance for arbitrary untrusted model files.

The training-manifest digest is calculated with Phase 3's canonical column order, train/validation rows, patient-key trimming, integer labels and canonical stored paths. It does not stat or open test images. **Do not rewrite the frozen CSV or edit a contract hash to force compatibility.** If moving from Kaggle to Colab/local storage, retain the original canonical manifest and set `FLAT_IMAGE_ROOT` to the current flat image directory. Only in-memory physical path lookup changes. Basename collisions/missing files fail. If a previously rewritten manifest no longer matches the recorded digest, supply the original canonical training CSV rather than reconstructing unknown assignments.

## Prediction alignment and preprocessing

`evaluation/inference.py` reads only the requested validation or test partition for labels/path checks and image decoding. The CSV file is necessarily read to filter rows; validation does not inspect test labels or use them in a metric/objective. Whole-file hashing binds the frozen file without selecting on its test labels. Metadata-only image-key/path/known-patient overlap checks can compare split membership without decoding other partitions.

Every model uses the existing `data.preprocessing.preprocess_image` with `split='validation'` or `'test'`. Training augmentation is never called. Letterboxing, RGB conversion, optional CLAHE and architecture-specific normalization remain unchanged: MobileNetV2 [-1,1], ResNet50 BGR/mean subtraction, EfficientNetB0 RGB [0,255] for embedded rescaling. No second normalization is added.

Models are loaded sequentially to limit CPU/GPU memory, called with `training=False`, and use reproducibility settings. Partial final batches are included. Stable dataset/image-key ordering and strict key/label joins prevent positional score mismatches. Duplicate/missing/extra rows, invalid labels, nonfinite/out-of-range probabilities and inconsistent model labels fail. Image IDs remain strings (including leading zeroes); no patient IDs are inferred. Individual CSVs save dataset name, image ID, split, ground truth and probability. Paths/patient keys are not included in prediction tables.

Deterministic execution does not promise bit-for-bit equality across TensorFlow versions, GPU kernels or platforms. Save runtime/source records with actual experiments. Phase 4's ROC-AUC is computed from predicted scores using scikit-learn; it need not exactly match the threshold-binned AUC recorded by Keras during training. Existing logged validation summaries are not copied into new evaluation results.

## Predefined ensemble protocol

The default `search_protocol()` is declared before inference and saved before validation prediction:

- Members in explicit order: `mobilenet_v2`, `efficientnet_b0`, `resnet50`.
- Per-model/soft-vote positive decision: probability **>= 0.5**. The threshold is predefined and not tuned.
- Majority voting: at least two positive model decisions; three models have no voting tie.
- Equal soft voting: mean probability with exact conceptual weights 1/3 each.
- Weighted soft voting: nonnegative weights summing to one (floating tolerance 1e-12; normalized for arithmetic).
- Search: all simplex triples at steps of 1/20 (0.05), plus exact equal weights. This gives **232 candidates**, including single-model corners.
- Criterion: minimum validation binary crossentropy. Probabilities are clipped to `[1e-7,1-1e-7]` for BCE only. Convex-combination outputs are bounded for floating-point roundoff; original individual scores are not altered.
- Loss ties within 1e-12 prefer minimum squared distance to equal weights, then lexicographic weights in the explicit member order.

No test prediction/table is accepted by `optimize_weights`. Both validation classes are required for selection. Candidate losses and selected weights are saved for audit. The procedure can choose an individual-model corner; it does not force a mixed ensemble or guarantee that any ensemble beats individual models on unseen data. Do not alter the grid/criterion/threshold after seeing test performance. Validation scores after optimizing weights are selection-biased, especially because the same validation partition already selected CNN checkpoints.

## Freeze and final test gate

Run validation in a new `phase4/<name>` directory. `frozen_ensemble.json` binds:

- Protocol, member order, selected weights, threshold and comparison rule.
- Original manifest file hash and aligned validation-prediction file hash.
- Checkpoint byte hashes, selections and training-contract hashes.
- Loaded model graph hashes, input/preprocessing/class contract, seed and inference batch size.
- Dataset identity and physical root/flat-root lookup configuration.

A canonical checksum detects accidental changes to the frozen record. This is an audit mechanism, not a security boundary against someone deliberately rewriting code/records. New validation refuses to overwrite any existing run, and the freeze API refuses refreezing or retuning a test-consumed run.

`run_final_test` first loads/verifies the freeze and its validation file, then rechecks the original manifest, inference settings and checkpoint/contract provenance. Only after these checks does it create `test_started.json` and consume test labels/paths/images. Each loaded model is checked against the frozen graph before its test inference. All six methods use the same image-aligned test rows and frozen threshold. Test labels do not influence weights, thresholds or checkpoint selection.

A test-start marker prevents another test attempt or hidden retuning in that run, even if a runtime failure prevents completion. Investigate such a failure without deleting evidence or using partial results to select parameters; recovery of an unchanged frozen evaluation requires an explicitly reviewed operational procedure. Never start fresh validation experiments based on observed test results. Successful evaluation creates `test_complete.json`. The frozen file is retained unchanged.

## Metrics, plots and disagreement

Each individual CNN and each voting method receives accuracy, precision, recall/sensitivity, specificity, F1 and a fixed-label-order confusion matrix (`[[TN,FP],[FN,TP]]`). Zero denominators are **null**, not invented zero/perfect scores, and are listed in `undefined_metrics`. Accuracy is always defined for nonempty input.

Individual probabilities and the two soft votes also receive rank-based ROC-AUC when both ground-truth classes exist, plus clipped-probability BCE. **Hard majority voting has no probability score:** its AUC/BCE are explicitly not applicable, and no fabricated confidence or calibrated vote-fraction probability is assigned. Score-based single-class AUC is null with a reason; BCE remains defined. ROC points record the initial infinite threshold as null for strict JSON/CSV compatibility.

Artifacts include six confusion matrices, a shared ROC figure for the five score-based methods, ROC points, comparative JSON/CSV metrics, wide strategy predictions, per-image hard decisions/correct-model counts and pairwise disagreement/both-wrong counts. These are computed only from predictions actually generated in an enabled run. No prefilled experimental scores or charts are shipped for Phase 4.

## Notebook 04 execution

`04_model_evaluation.ipynb` supports independent Kaggle, Colab and local setup with Python 3.10–3.13 and the existing TensorFlow training dependencies. Kaggle detection precedes Colab; Colab Drive mounting is explicit. Install packages only if needed, then restart the kernel. No dataset or pretrained-weight downloads occur.

1. Set `REPO_ROOT`, `SESSION_FILE`, the original `MANIFEST_CSV`, checkpoint base(s), optional `FLAT_IMAGE_ROOT`, output root and a fresh evaluation name. Attach actual private weights/states/contracts; do not commit them.
2. Optionally enable `VERIFY_CHECKPOINTS` to check metadata/artifact availability without opening test images.
3. Enable **only `RUN_VALIDATION`**. Review the saved weights, search candidates, validation tables/disagreement and frozen record.
4. Turn validation off, retain the same paths/settings, and explicitly enable **`RUN_FINAL_TEST`**. The notebook disallows both flags together to encourage separate review and invocation.
5. Review actual test artifacts without retuning. Preserve Kaggle working outputs through saved outputs/private export, or write directly to mounted Drive on Colab. Notebook outputs should be cleared before committing reusable source.

All setup/clone/install/checkpoint/inference/display flags default to False. The notebook's default path does not load checkpoints, decode images or write evaluation artifacts. Source logic resides in reusable modules under `src/glaucoma/ensemble/` and `src/glaucoma/evaluation/`.

## Storage layout

Outputs live under the configured artifact root (or explicit output override), outside image storage:

```text
<artifact-root>/phase4/<evaluation-name>/
  predefined_protocol.json
  frozen_ensemble.json
  validation/
    individual_predictions/<model>.csv
    aligned_predictions.csv
    weight_search.csv
    all_strategy_predictions.csv
    metrics.json
    metrics_comparison.csv
    confusion_matrices.png
    roc_curves.png
    roc_points.csv
    prediction_disagreement.csv
    disagreement_summary.json
  test_started.json          # only after final test opt-in and freeze checks
  test/                     # same prediction/metric/plot set, no weight search
  test_complete.json
```

Checkpoint and training directories are read-only inputs. Frozen dataset files are never rewritten. Outputs, predictions, model files and datasets must remain outside Git; the default repository artifact directory is already ignored.

## Scientific limits and verification status

Image-level splitting without real patient identifiers does not establish patient independence; known key/path/patient leakage fails, but these checks do not detect every resized/cropped or renamed duplicate. This is single-dataset ACRIMA research. External-dataset generalization, calibration, confidence intervals, clinical utility and superiority over any individual model are not established. The test set is small, should remain a final fixed evaluation, and must not become another selection set.

Implementation tests use synthetic predictions/images and offline randomly initialized structures, including shuffled alignment, boundary/invalid weights, validation-only selection, threshold ties, saturated scores, exact metrics, one-class/undefined cases, provenance/configuration changes, freeze/test ordering, disagreement and actual plot-file generation. No real checkpoints or test images were available locally, so the real Kaggle Phase 4 run remains to be performed explicitly by the user. No CNN was retrained and no experimental result was fabricated.
