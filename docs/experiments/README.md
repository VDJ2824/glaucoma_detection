# Core-model and ensemble experiment records

All three core models have completed two-stage Kaggle training, as reported by the user and recorded in the supplied metrics/selection artifacts. This directory contains analytical documentation and original CSV/JSON/PNG result artifacts only, without datasets, weights or source ZIPs.

Start with the [Stage 1/Stage 2 comparative report](core_models_comparison_2026-11-27.md). All measurements are **validation-selected results, not held-out test performance**.

| Model | Earlier account | Complete archive record |
| --- | --- | --- |
| MobileNetV2 | [Preserved 26 November account](mobilenet_v2_kaggle_2026-11-26/README.md) | [Full history/graph addendum](mobilenet_v2_kaggle_2026-11-26/ARTIFACTS.md) |
| EfficientNetB0 | — | [Record imported 27 November](efficientnet_b0_kaggle_2026-11-27/README.md) |
| ResNet50 | — | [Record imported 27 November](resnet50_kaggle_2026-11-27/README.md) |

[Archive-member manifest](core_models_artifact_manifest_2026-11-27.json) · [Full-precision comparison](core_models_comparison_2026-11-27.json)

All 24 original source artifacts are preserved byte-for-byte, including the CSV CRLF line endings; scoped `.gitattributes` rules prevent checkout normalization. New record-directory dates are import dates, not assumed training dates. The original MobileNetV2 account remains unchanged; its addendum supplies the previously unavailable complete histories.

Run the repository's read-only verifier to check hashes, 84 epoch records, summaries, selections, expected numerical notes and derived comparisons:

```bash
python scripts/verify_experiment_artifacts.py
```

## Phase 4 evaluation completed

[Validation and held-out test comparative report](phase4_kaggle_2026-11-27/README.md) covers all six methods, the original frozen configuration, aggregate metrics, original confusion matrices/ROC plots and the privacy review. These held-out test results are distinct from the validation-selected training measurements above. Equal-weight voting matched the best individual test accuracy and had the lowest test BCE and highest ROC-AUC; the validation-optimized mixture performed worse on test. No weights were changed. Per-image predictions and fine-grained ROC score exports are withheld.

```bash
python scripts/verify_phase4_artifacts.py
```
