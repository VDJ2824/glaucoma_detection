# Colab and Kaggle notebooks

These notebooks orchestrate `src/glaucoma` modules. Dataset tools, read-only exploration (01) and environment setup are implemented; Phase 3 training (03) is implemented with explicit opt-in flags; Phase 4 preserves 04 for individual model evaluation and 05 for cached-score ensemble learning/final test; Phase 5 Grad-CAM is implemented with private, opt-in analysis; cross-dataset validation and conversion remain documented scaffolds. Training starts only when explicitly enabled in 03; no notebook downloads datasets. Notebook 01 now performs read-only exploration when run; it requires an existing dataset. The preparation call in 02 is disabled until you explicitly enable it.

| Notebook | Purpose |
| --- | --- |
| `00_environment_setup.ipynb` | Mount Drive on Colab; optionally clone/install; verify TensorFlow/GPU; save persistent session configuration |
| `01_dataset_exploration.ipynb` | Implemented class/dimension summaries, RGB sample plots and duplicate/corruption checks |
| `02_dataset_preparation.ipynb` | Opt-in shared validation, deduplication, split preparation and preprocessing inspection |
| `03_individual_model_training.ipynb` | Implemented independent setup, opt-in two-stage training, histories and checkpoint recovery |
| `04_model_evaluation.ipynb` | Individual-model checkpoint verification, validation inference/metrics/plots and saved prediction handoff |
| `05_ensemble_optimization.ipynb` | Reuse 04 validation scores; voting, validation-only weight search/freeze and separate final test of all methods |
| `06_gradcam_analysis.ipynb` | Implemented nested-backbone Grad-CAM, signed class targets, equal/frozen-weight fusion and private panels |
| `07_cross_dataset_validation.ipynb` | Future external validation after verified adapters exist |
| `08_tensorflowjs_conversion.ipynb` | Future isolated core-only conversion and browser parity checks |

Start with 00 → 01 → 02 → 03 → 04 (individual validation predictions/reports) → 05 (cached-score ensemble learning/freeze) → 05 (separate frozen final test evaluation) → 06 → 07 → 08. Train each core model independently at 03. Optional VGG experiments never gate this sequence or enter the application.

## Independent kernels

Run 00 in each new hosted runtime. Set the repository path, dataset path, artifact root and run name; mounting/installing requires the platform's normal interaction. Subsequent notebooks load the saved `session.json`. They do not inherit variables from 00 in another kernel: paste its printed path into `SESSION_FILE`, or export `GLAUCOMA_SESSION` in that kernel. If TensorFlow dependencies were installed/changed, restart the kernel before verification. Keep installation disabled on the rerun.

The session JSON is a resolved configuration snapshot, not a replacement for editing `configs/project.yaml`. After changing model selection or dataset mapping, rerun configuration in 00 with the intended run name; choose a new name to preserve earlier settings. For a different Colab/Kaggle machine, use Notebook 03's explicit frozen-session relocation helper to create path-adjusted copies; do not regenerate splits. Original image IDs, labels, grouping keys and assignments are preserved, and existing checkpoint contract checks still apply.

## Persistent outputs

`configure_workspace` in `glaucoma.utils.notebook_runtime` creates this derived-only layout:

```text
<artifact_root>/runs/<run_name>/
  checkpoints/core/       # future <model>/<stage>/ checkpoints
  checkpoints/baselines/  # optional VGG checkpoints
  metrics/
  histories/
  heatmaps/
  exports/                # future converted core model directories
  manifests/              # metadata + validation CSVs
  environment/            # session.json, TF check, pip freeze, source commit
```

Training callbacks use `cfg['checkpoints']`; metrics/history writers use `cfg['artifacts']`; exporters must use `cfg['export']['output_root']`. No writer should hardcode `/content` or the frontend directory. The helper does not create the dataset directory, edit original images, construct models or write model artifacts. Reusing a run name can overwrite derived files; use distinct names for independent experiments.

On **Colab**, choose an artifact root under `/content/drive/MyDrive/...` after mounting; files under `/content` alone are ephemeral. Drive may be slower for frequent writes; future training should use an explicit save strategy without treating temporary local checkpoints as durable. See [Colab storage/limits](https://research.google.com/colaboratory/faq.html) and [Drive mount example](https://colab.research.google.com/notebooks/io.ipynb).

On **Kaggle**, attach your own labeled dataset under `/kaggle/input`, put artifacts under `/kaggle/working`, and save a notebook version that includes outputs before leaving the runtime. Export the artifacts and upload them to Google Drive if Drive is your long-term archive. There is no automatic Drive mount/sync in Kaggle, and interactive session files alone are not a backup. A new saved run can start with a clean state: reattach needed prior artifacts and configure their paths explicitly. Review [Kaggle notebook settings and saved versions](https://www.kaggle.com/docs/notebooks).

Dependencies/cloning may require Internet, while dataset processing itself is local. No tokens, dataset files, patient information or populated notebook outputs should be committed. See the root README for GitHub and platform setup, and [dataset instructions](../docs/datasets.md) for ACRIMA organization.

See [Phase 3 instructions](../docs/phase3-training.md) for TensorFlow 2.20.x/2.21.x, Python 3.10–3.13, offline weights=None smoke tests and the frozen-manifest recovery contract. Existing notebook 00–02 edits are retained.

The completed Kaggle MobileNetV2 run is preserved in [separate experiment documentation](../docs/experiments/mobilenet_v2_kaggle_2026-11-26/README.md). Notebook 03 has no executed outputs, automatic ZIP creation or enabled training flags.

Phase 4: see [ensemble/evaluation guide](../docs/phase4-ensemble-evaluation.md). Notebook 04 owns individual-model validation; Notebook 05 reuses its cached score/provenance files for ensemble learning and final testing. Shared modules avoid duplicate inference or metric/search implementations. For separate Kaggle runtimes, save 04 outputs and use 05's opt-in verified handoff copy into fresh writable storage; do not modify or regenerate the original scores. Actual Kaggle validation and held-out test aggregate results are archived in the [Phase 4 report](../docs/experiments/phase4_kaggle_2026-11-27/README.md); per-image prediction files are withheld. Both notebooks remain reusable and disabled by default.

Phase 5: [Grad-CAM guide](../docs/phase5-gradcam.md). Notebook 06 independently configures Kaggle/Colab/local execution, reuses verified models and saved predictions, and saves only private image-linked artifacts. Its inspection, explanation, test-analysis and display flags default to False. Test explanations require existing Phase 4 completion and remain post-hoc; the public aggregate archive cannot replace the complete private run. Keep executed outputs and overlays out of Git.
