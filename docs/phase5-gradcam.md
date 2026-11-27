# Phase 5 — private Grad-CAM analysis

[Notebook 06](../notebooks/06_gradcam_analysis.ipynb) now orchestrates reusable, inference-only core explanations. It does not retrain, modify checkpoints, regenerate splits, optimize ensemble weights or replace the completed Phase 4 evaluation. Real retinal explanations require the private original inputs; no retinal maps or performance claims are supplied by this implementation.

## Required private inputs and platform setup

Use the same environment requirements as Phase 3/4 (`requirements-training.txt`). Python 3.10–3.13 is supported by the existing dependency markers: TensorFlow 2.20 on earlier Python versions, TensorFlow 2.21 on Python 3.13. Actual Grad-CAM execution is checked locally with TensorFlow 2.20 CPU and randomly initialized models, not downloaded weights; hosted runtime behavior must also be checked against the actual private checkpoints. No dependency changes or automatic dataset/pretrained-weight downloads are introduced.

Notebook 06 independently detects Kaggle first, then Colab, then local execution. Set `REPO_ROOT`, `SESSION_FILE`, `MANIFEST_CSV`, `CHECKPOINT_ROOT` or per-model `CHECKPOINT_ROOTS`, and `EVALUATION_DIRECTORY`. Optional clone, installation and Colab Drive mounting use the same explicit setup flags as 04/05. On Kaggle, attach your saved private checkpoint bundle and the complete private Phase 4 evaluation output. On Colab, use mounted Drive for persistent heatmaps. Locally, use ignored `artifacts/`.

The session and original frozen manifest must retain the Phase 4 preprocessing, input shape, seed, batch size, class mapping and dataset identity. Do not rewrite canonical manifest paths. `FLAT_IMAGE_ROOT` may map basenames to the current physical image directory without changing the original CSV. Selected checkpoint files may also move physically, but their bytes, stage/epoch/loss, training/validation manifest digest, contract, model identity and graph fingerprints must agree with the original frozen records. The existing selection/recovery utilities are reused read-only, with committed individual-model selection records as additional notebook checks. Both stages' `best.keras`, `state.json` and `contract.json` remain required by the existing verifier.

The complete private Phase 4 directory must include `frozen_ensemble.json`, `validation_predictions.json`, validation aligned/individual CSVs and, for test analysis, `test/aligned_predictions.csv` and the matching `test_complete.json`. The aggregate archive under `docs/experiments/` deliberately omits these image-linked CSVs and cannot be used instead. Frozen and validation handoff checksums are verified before analysis. Original test CSVs have no Phase 4 checksum binding: analysis records their current hash, aligns keys/labels with the unchanged manifest, and rechecks each explained model score. This does not authenticate the whole historical test CSV or establish execution chronology.

## Explicit execution and example selection

All flags default to False: `RUN_PLATFORM_SETUP`, `CLONE_REPOSITORY`, `INSTALL_DEPENDENCIES`, `INSPECT_SAVED_PREDICTIONS`, `RUN_GRADCAM`, `ALLOW_TEST_ANALYSIS`, `SHOW_PRIVATE_PANELS`. Running all cells with defaults loads session settings and imports modules, without checkpoint/image/prediction access or artifact generation. Installation and Drive mounting are opt-in.

1. Configure the private paths and optionally inspect saved predictions using `INSPECT_SAVED_PREDICTIONS`.
2. Choose `SPLIT="validation"` initially. `SELECTION_METHOD` defaults to the already evaluated frozen weighted classifier; it can name any core or Phase 4 voting method for retrospective example categorization. It does not change predictions or weights.
3. `select_examples` sorts by dataset and image ID, takes up to `PER_CATEGORY` correct, incorrect and core-disagreement examples, and deduplicates overlapping keys. It reports missing categories instead of inventing cases. Core disagreement means differing individual predicted classes at the frozen threshold. This is deterministic qualitative sampling, not an unbiased explanation-quality study.
4. Choose `TARGET_CLASSES=(0, 1)` to explain both classes separately. Every map in a panel uses the same target, regardless of the individual predicted class.
5. Set `RUN_GRADCAM=True` explicitly and choose a fresh `EXPLANATION_NAME`. Existing outputs are never overwritten.
6. For test analysis, set `SPLIT="test"` and explicitly opt into `ALLOW_TEST_ANALYSIS=True`. The corresponding Phase 4 test completion record must already exist and match the frozen seal before test predictions are read. These explanations are **post-hoc only**: do not use observed test maps/errors to retune the evaluated system or claim another untouched evaluation.

## Connected nested-backbone Grad-CAM

`src/glaucoma/explainability/gradcam.py` exposes `GradCAM`, `GradCAMResult`, `GradCAMError` and strict map normalization. Default spatial targets are the final convolution before subsequent backbone normalization/activation/residual operations:

| Core model | Target convolution | Native map size at project input |
| --- | --- | --- |
| MobileNetV2 | `Conv_1` | 7 × 7 |
| EfficientNetB0 | `top_conv` | 7 × 7 |
| ResNet50 | `conv5_block3_3_conv` | 7 × 7 |

The probe returns both the internal convolution and the original backbone output through one connected graph, then replays the existing GAP, inference Dropout and Dense head. This handles Keras 3 nested models without connecting an internal tensor directly to an unrelated outer input. Explicit input watching permits differentiation even when the entire backbone is frozen. Model trainability, variables, normalization, BN statistics and dropout behavior remain unchanged; all calls use `training=False`. The workflow loads each core sequentially to limit memory use.

The API expects an already preprocessed single RGB-derived tensor. The workflow calls the existing `preprocess_image` exactly once per image/model with `split="validation"` or `"test"`; no augmentation or extra normalization is applied. MobileNetV2 uses [-1,1], ResNet50 uses the existing BGR/mean subtraction, and EfficientNetB0 receives [0,255] for its embedded rescaling.

For the binary classifier `p(glaucoma)=sigmoid(z)`, class 1 explains `z`, and class 0 explains `-z`. Using logits avoids additional sigmoid saturation in the gradient target; scores shown to users remain the saved glaucoma probability. Non-glaucoma maps are not obtained by inverting the glaucoma map. Spatially averaged target gradients weight the feature channels; their weighted sum is ReLU-clipped and divided by its positive maximum. No claim of equivalence to all probability-targeted Grad-CAM implementations is made.

Disconnected gradients and nonfinite inputs/activations/gradients/logits/maps fail explicitly. A zero gradient yields an all-zero map with `zero_gradient`; nonzero gradients with no positive ReLU evidence yield `no_positive_activation`. A map that disappears after padding removal is marked `padding_only_activation`. Empty overlays retain the original pixels and are labelled; no artificial localization is fabricated. Invalid model identity/head/layer is rejected. Saved and replayed probabilities must agree within explicit small tolerances (`atol=1e-5`, `rtol=1e-4` by default); mismatches fail rather than updating scores. Tolerances address numerical runtime variation, not changes in methodology.

## Fusion and original geometry

`fusion.py` requires exactly three named core maps. Each finite, nonnegative map is max-normalized and bilinearly resized onto the same model-input canvas. It computes either one-third weighting or the **unaltered Phase 4 frozen weights**, then max-normalizes the convex mixture. If any member has no informative original-image map, workflow fusion is unavailable; no member is dropped and no weight is redistributed. This includes zero-weight members as an explicit conservative completeness policy.

These outputs are **heatmap fusion, not direct ensemble-score Grad-CAM**. Individual maps explain signed logits; weights were selected for probability-score BCE in Phase 4, not explanation quality. Per-map and final normalization discard absolute magnitude, so fused brightness is not a probability, uncertainty measure or calibrated feature attribution. Majority voting does not have a separate gradient map.

Before overlaying, maps are upsampled to the 224 × 224 input canvas, cropped using the identical scale, `round` and centered integer padding offsets from the existing letterbox preprocessing, then resized to the original height and width. Padding must be removed before resizing to the original aspect ratio. This preserves geometric alignment but cannot recover fine spatial detail or undo receptive-field mixing at padding boundaries. Overlays use the original RGB image, not the normalized/CLAHE model tensor.

## Private artifact contract

Outputs go under `<cfg.artifacts.heatmaps>/phase5/<EXPLANATION_NAME>/`:

```text
analysis.json                     Frozen seal/weights, contracts/graphs, targets, selection, source CSV hash
selected_examples.json            PRIVATE image IDs, labels, saved scores and categories
example_000/class_0/              One directory per example/target
  <core>_native.npy               Normalized native convolution map
  <core>_image.npy                Map in original image coordinates
  <core>_overlay.png              Private individual overlay
  equal_heatmap_fusion_canvas.npy # Present only when fusion is informative
  equal_heatmap_fusion_image.npy
  equal_heatmap_fusion_overlay.png
  weighted_heatmap_fusion_*.npy / *_overlay.png
  explanation.json               Ground truth, scores/classes, geometry, source image hash, status/layer/logit
  comparison.png                 Original + 3 individual + 2 fusion panels
example_000/class_1/
...
complete.json                     Written only on successful completion
```

Every panel shows the ground truth, common target, saved individual/fusion scores, predicted classes and map statuses. Noninformative fusion slots show the original image with an unavailable label. Panel files render through Matplotlib's noninteractive Agg canvas so local headless execution does not depend on a GUI backend.

Output guards reject dataset, physical image, checkpoint and Phase 4 source storage overlaps. Inside this repository, image-linked outputs must stay under ignored root `artifacts/`; `docs/` and other tracked areas are rejected. Use private storage on Drive/Kaggle and review any exported files before sharing. Do not commit retinal images, overlays, arrays, prediction CSVs, populated notebook outputs or backup ZIPs without verifying original redistribution permissions. Neither source images nor any explanation artifact is sent to Gemini or another service.

## Interpretation and verification

Grad-CAM is coarse sensitivity analysis, not an optic-disc/cup segmentation mask, clinical diagnosis, localization ground truth, proof of anatomical reasoning or causal evidence. An attractive heatmap does not establish classifier correctness. Failures, absent maps and model disagreement are valid observations, not grounds for silently changing the evaluated model. Image-level splitting does not guarantee patient independence, and ACRIMA-only analysis does not establish external validity. Real checkpoint explanations still require explicit private hosted execution and review.

Run `python -m pytest -q`. Synthetic tests cover analytic opposite-class maps, fully frozen/nested backbones, actual core architectures with `weights=None`, score parity and unchanged weights/trainability, empty/invalid maps, alignment/frozen weights, odd/extreme letterbox geometry, deterministic category selection, private output guards, disabled notebook defaults, completed-test gating, score mismatch rejection and private end-to-end panel generation. Existing dataset, training, evaluation and archived-result tests remain in place. These tests do not measure real explanation quality or fabricate retinal results.
