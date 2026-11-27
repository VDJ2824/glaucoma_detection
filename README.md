# Lightweight Glaucoma Detection via Ensemble Learning

A modular B.Tech CSE major-project scaffold for a lightweight, explainable, privacy-preserving retinal fundus image classification research prototype. Planned inference runs locally in the browser; optional Gemini-generated analytical text is requested through Flask.

**Current status:** individual two-stage Kaggle training is complete for MobileNetV2, EfficientNetB0 and ResNet50, with supplied results archived under `docs/experiments/`. The dataset foundation and Phase 3 training implementation are unit tested. Notebook 03 includes independent setup, offline smoke checks, explicit training flags, persistent histories/checkpoints and recovery. The frontend remains a landing placeholder and Flask provides a health endpoint with disabled reports. Phase 4 checkpoint-verified inference, validation-only ensemble selection and frozen-protocol evaluation are implemented; actual Phase 4 validation and held-out test evaluation have completed on Kaggle, with aggregate results archived under `docs/experiments/`. Repeating inference requires the private Kaggle checkpoints. Image quality assessment, Grad-CAM, conversion, Gemini requests and PDFs remain unimplemented. No datasets or trained weights are bundled; nothing downloads or trains models automatically.

Start with [ACRIMA organization and preparation instructions](docs/datasets.md). For read-only exploration, install `requirements-dev.txt`, confirm the selected filename/folder/CSV label mode, refresh notebook 00 and run notebook 01. The preparation command is a separate opt-in step that assigns splits; leave it for the preparation phase. Run `python -m pytest -q` for synthetic-data tests.

This is experimental research, not a medical device or a clinical diagnosis. Model scores are not established clinical probabilities. The term “lightweight” is a project goal: the memory, latency and download cost of all three core models must be measured on target devices.

## Architecture boundaries

| Area | Models | Intended use |
| --- | --- | --- |
| Core system | ResNet50, MobileNetV2, EfficientNetB0 | Independent training; weighted predictions; individual and ensemble Grad-CAM; TF.js export; browser inference; analytical reports |
| Optional baselines | VGG16, VGG19 | Learning, experimentation and comparative evaluation; optional research Grad-CAM only |

VGG training is optional and never a prerequisite for the main project. Its entry point, configuration role, experiments and artifact paths are separate. Shared builders and training utilities will avoid duplication, but application registries and export validation must admit only the three core models.

## Directory structure

```text
configs/                    Central project/data/training YAML
src/glaucoma/
  config.py                 YAML loading, validation and path overrides
  data/                     Loading, preprocessing and image quality
  models/                   Shared factory; separate core and baseline registries
  training/                 Shared trainer and callbacks
  evaluation/               Metrics and comparative research
  ensemble/                 Core prediction fusion and weight selection
  explainability/           Individual Grad-CAM and heatmap fusion
  export/                   TF.js conversion and parity validation
  utils/                    Reproducibility
scripts/                    Dataset preparation; disabled training/evaluation/export entry points
experiments/
  baselines/                Optional VGG research
  segmentation/             Future disc/cup segmentation
  alternative_models/       Future architecture experiments
notebooks/                  Nine ordered Colab/Kaggle notebooks and persistence guidance
backend/                    Flask API, analytics schema and Gemini placeholders
frontend/
  src/components/           Upload, quality, scores, heatmap and report placeholders
  src/services/             Local inference/explainability/PDF and report API placeholders
  src/utils/                Browser preprocessing placeholder
  public/models/            Future core TF.js artifacts only
artifacts/                  Ignored checkpoints, metrics, heatmaps and exports
reports/generated/          Ignored generated reports
data/                      Ignored local datasets (see data/README.md)
docs/                       Design, privacy and research workflow
```

## Local environment setup

Use Python 3.10–3.13 and Node.js 20.19+ or 22.12+. Dependencies are declared as version ranges; lock exact versions after installation and compatibility validation. Installation is optional at this scaffold stage.

For dataset development and unit tests (CPU only, no TensorFlow needed):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Research/notebook environment (Python 3.10–3.13, TensorFlow 2.20.x/2.21.x, CPU supported; no training starts on install):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
cp .env.example .env
```

For the backend alone, create an independent environment and install `requirements-backend.txt`; TensorFlow is not required. Start from the repository root:

```bash
python -m pip install -r requirements-backend.txt
python -m backend.app
```

`GET /api/health` reports scaffold status. `POST /api/reports` always returns HTTP 501 and never calls Gemini. The API binds to localhost by default and has a small request-size limit and an explicit development CORS origin. These settings are not a complete deployment security policy.

Frontend:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

This displays a static project placeholder. `npm run build` will build the frontend once dependencies are installed. Never put Gemini credentials in frontend files or variables prefixed with `VITE_`. Vite reads `frontend/.env`; Flask reads root `.env`. Dataset tooling reads exported environment variables; it does not auto-load `.env`.

## GitHub repository setup

Create an empty repository on GitHub without an initial README, then run these commands in your local project folder after reviewing the staged files. Replace the URL with your repository. Check `git rev-parse --show-toplevel` first. If it reports a parent directory, run `git init` **inside `glaucoma_detection`** to create an independent project repository; do not rename branches or add remotes in the parent repository. Skip initialization only if the reported Git root is already this exact project directory, and inspect `git remote -v` before adding a remote.

```bash
git init
git branch -M main
git add .
git diff --cached --stat
git diff --cached
git commit -m "Add modular glaucoma research project and notebook scaffolds"
git remote add origin https://github.com/YOUR_USERNAME/glaucoma_detection.git
git push -u origin main
```

These are instructions only; no repository has been published or pushed by this task. `.gitignore` excludes local datasets, artifacts, secrets, environments and model weights. Keep supplemental annotations/private paths under ignored `data/` and run artifacts under ignored `artifacts/`. Do not commit populated notebook outputs, patient identifiers or credentials. Authenticate private repositories through the platform's supported secure flow; never paste access tokens into notebook code or clone URLs. After pushing, record a commit/tag and set `REPO_REF` in notebook 00 for repeatability. Source revisions and split manifests should accompany every future experiment.

## Google Colab execution

1. Open `notebooks/00_environment_setup.ipynb` from your GitHub repository using Colab's notebook opener, or upload that notebook. Select a GPU accelerator in runtime settings if available; CPU setup works too.
2. Edit `REPO_URL`, `REPO_REF` and `REPO_DIR`. Leave the code clone on runtime storage (normally `/content/glaucoma_detection`), and enable `CLONE_REPOSITORY` only when the destination does not exist. Existing clones are never automatically pulled/reset.
3. Run the Drive-mount cell and authorize access. Keep datasets in `/content/drive/MyDrive/glaucoma_project/data/raw/ACRIMA` or your own configured path. Keep the artifact root at `/content/drive/MyDrive/glaucoma_project/artifacts` or another mounted Drive directory.
4. Enable `INSTALL_DEPENDENCIES` for initial installation. It uses `requirements-notebooks.txt`, which can replace preinstalled packages. After installation the notebook stops deliberately: restart the kernel, disable installation, rerun setup, and check TensorFlow's GPU list and tiny tensor computation. This project's supported Python/TensorFlow ranges must match the chosen runtime; a platform upgrade is not guaranteed compatible.
5. Confirm ACRIMA label mapping/annotation paths in `configs/project.yaml`, then set `DATASET_ROOT`, `ARTIFACTS_ROOT` and a distinct `RUN_NAME` in 00. The shared helper redirects checkpoints, metrics, histories, heatmaps, converted models and manifests to the persistent run folder. It saves resolved settings and environment records there.
6. Open the next notebook. Run 00 again in a fresh runtime when necessary; in each separate kernel, paste the printed session JSON path into `SESSION_FILE`. Do not rely on Python variables from a different notebook. Package installation and mounts must exist in that runtime.

Colab runtime disks are temporary and GPU availability varies. Keep durable outputs on mounted Drive and verify saves before ending a session. See [Colab's storage and resource FAQ](https://research.google.com/colaboratory/faq.html) and [official Drive example](https://colab.research.google.com/notebooks/io.ipynb). No setup cell downloads a dataset, creates a pretrained model or starts training.

## Kaggle Notebooks execution

Import notebook 00 and attach your own verified ACRIMA dataset through Kaggle's input controls. Configure the session's accelerator and Internet options as needed for GitHub/pip. Edit the dataset path to the actual attached directory under `/kaggle/input` and use a repository clone under `/kaggle/working`. If Internet is unavailable, attach a repository archive and use an existing compatible environment instead of cloning/installing.

Notebook 00 skips Colab Drive mounting on Kaggle and defaults outputs to `/kaggle/working/glaucoma_artifacts/runs/<run_name>`. Save a notebook version with outputs, then download the artifacts and upload/copy them to Google Drive for a durable archive. This project does not automatically authenticate or synchronize Kaggle with Drive. A saved batch run uses a clean session, so all setup/configuration must be reproducible there. See [Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks).

## Dataset folder organization

Keep original downloads and filenames unchanged. The configuration defaults to the inspected flat ACRIMA release: point its root at the directory holding `Im<number>_g_ACRIMA.jpg` (label 1) and `Im<number>_ACRIMA.jpg` (label 0). Matching is case-insensitive and unexpected entries are rejected. For the folder layout below, explicitly select `label_mode: folders`; for supplemental annotations select `label_mode: csv`. See the provenance details in `docs/datasets.md`. The following is an alternative **folder-mode example**:

```text
Google Drive/MyDrive/glaucoma_project/
  data/raw/ACRIMA/
    non_glaucoma/    # explicitly verified label 0
    glaucoma/        # explicitly verified label 1
  artifacts/runs/<run_name>/
    checkpoints/core/
    checkpoints/baselines/
    metrics/
    histories/
    heatmaps/
    exports/
    manifests/
    environment/
```

In folder mode, if the original folders differ, change `datasets.acrima.class_directories`; do not reorganize originals just for the example. For noncanonical flat filenames, provide a verified annotation CSV. Paths in that CSV are relative to the configured ACRIMA root. Missing patient IDs remain blank; unavailable glaucoma labels are errors. Other datasets have reserved configuration only. See [the complete ACRIMA guide](docs/datasets.md).

## Recommended notebook execution order

| Notebook | Purpose and status |
| --- | --- |
| `00_environment_setup.ipynb` | Implemented platform setup, diagnostics and persistent output configuration |
| `01_dataset_exploration.ipynb` | Implemented read-only class/dimension analysis, samples and integrity checks |
| `02_dataset_preparation.ipynb` | Documented shared dataset pipeline; processing explicitly opt-in |
| `03_individual_model_training.ipynb` | Implemented opt-in independent two-stage training, offline smoke tests and checkpoint recovery |
| `04_model_evaluation.ipynb` | Individual-model validation predictions, metrics, confusion matrices/ROC and saved handoff |
| `05_ensemble_optimization.ipynb` | Cached validation voting/weight search, ensemble freeze and separate final test of all methods |
| `06_gradcam_analysis.ipynb` | Scaffold for individual and ensemble core explanations |
| `07_cross_dataset_validation.ipynb` | Scaffold pending verified external-dataset adapters |
| `08_tensorflowjs_conversion.ipynb` | Scaffold for isolated core conversion and browser parity |

Start with 00 → 01 → 02 → 03. Repeat 03 independently for each core model. Use 04 only for individual-model validation predictions/reports. Use 05 to reuse those scores for voting, validation-only weight optimization and freezing; then run its separate final-test opt-in with ensemble learning disabled. Continue with 06 → 07 → 08. Test/external results must not drive weight or threshold selection. Optional VGG training stays separate and is never required. Notebook 08 will use a separate conversion environment; it does not install converters or export models yet.

All core logic stays in reusable Python modules. Notebooks 03–05 are implemented; later explainability/conversion notebooks remain scaffolds and load saved session settings. See [notebook details and output contracts](notebooks/README.md). No trained checkpoints are bundled. Full supplied CSV/JSON histories, stage summaries, selections and original graphs for all three models are preserved separately in documentation. The trainer saves real run artifacts to the configured persistent paths; converted models remain future work.

## Completed individual-model training

All three core models completed Stage 1 head training and Stage 2 fine-tuning on Kaggle. The user-supplied result archive has been organized under [docs/experiments](docs/experiments/README.md), preserving all six epoch CSVs, six JSON histories, six stage metric summaries, three checkpoint selections and three original PNG graphs byte-for-byte. The earlier MobileNetV2 account remains unchanged and now has a full-history addendum. No datasets, model weights or source ZIPs are included.

The selections all choose Stage 2 by minimum validation loss:

| Model | Selected stage epoch | Stage 2 completed epochs | Validation loss | Validation accuracy | Validation ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2 | 9 | 14 | 0.038979 | 99.0566% | 0.999637604 |
| EfficientNetB0 | 18 | 20 | 0.014796 | 100.0000% | 0.999999940 |
| ResNet50 | 18 | 20 | 0.012717 | 100.0000% | 1.000000000 |

These are **validation-selected measurements, not held-out test results or clinical performance**. Perfect observed validation accuracy does not establish perfect generalization. See the [comparative report](docs/experiments/core_models_comparison_2026-11-27.md) for all Stage 1/Stage 2 training and validation metrics, fine-tuning deltas, epochs, checkpoint references, numerical notes and provenance. TensorFlow 2.21.0 and Keras 3.13.2 are recorded for all stage summaries; new-model Python/GPU details are not supplied by the archive.

Verify artifact hashes, all 84 epoch rows, selections and derived values without TensorFlow or the source ZIP:

```bash
python scripts/verify_experiment_artifacts.py
```

Notebook 03 remains generic, output-free and safe to open: training/setup/download flags are disabled, `weights=None` is the offline default, and model selection uses a valid configurable core-model name. Kaggle-first detection, Python 3.13 support and opt-in frozen-path relocation are retained. The import does not retrain models, recalculate splits or alter architecture/training logic.

## Phase 3 training and CPU smoke tests

See [the complete Phase 3 guide](docs/phase3-training.md) for the factory, frozen-manifest loader, two-stage API/CLI, per-epoch persistence and recovery contracts. Install `requirements-training.txt` plus `requirements-dev.txt` to run all tests. Python 3.10–3.12 retains TensorFlow 2.20.0/NumPy <2.1; Python 3.13 selects TensorFlow 2.21.x/NumPy >=2.1. Leave package installation disabled on an already working hosted stack. Without TensorFlow, training tests explicitly skip; the dataset suite still runs.

Notebook 03 supports its own Drive/repository setup and loads the existing session. By default, model construction, pretrained downloads and both training stages are disabled. For an offline CPU smoke test, set `BUILD_MODEL=True`, `WEIGHTS=None` and `RUN_FORWARD_SMOKE=True`; no images or training are needed. For actual future transfer learning, select ImageNet weights and explicitly enable training. VGG baselines remain separate.

Stage 1 freezes the backbone and uses Adam/BCE at 0.001 for up to 10 epochs. Stage 2 restores the best Stage 1 checkpoint, unfreezes the last 20 layers excluding BatchNormalization and recompiles Adam at 0.0001 for up to 20 epochs. Existing normalization/augmentation code is unchanged, validation is never augmented, and training never reads test pixels or regenerates splits.

The trainer includes EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, per-model/stage JSON histories, validation-selected best checkpoints and epoch-boundary recovery with configuration/manifest checks. The reported individual-model validation runs are documented above; held-out test performance and generalization remain unmeasured.

## Phase 4: ensemble learning and performance evaluation

[Notebook 04 — Model Evaluation](notebooks/04_model_evaluation.ipynb) handles checkpoint verification, deterministic image-aligned validation scores, individual metrics, confusion matrices and ROC plots. [Notebook 05 — Ensemble Learning](notebooks/05_ensemble_optimization.ipynb) reuses its checksummed prediction handoff for majority/equal/weighted voting, validation-only weight optimization, freezing and separate final test evaluation of all methods. The existing checkpoint selection/recovery and preprocessing functions are reused. It requires actual private `.keras` checkpoints plus both stages' `state.json`/`contract.json`, the original frozen manifest and saved session; the committed summaries alone are insufficient.

Weights are selected on validation predictions only using a predefined 0.05-step simplex grid plus exact equal weights (232 candidates), minimizing validation BCE with deterministic ties. The default threshold is fixed at 0.5, not tuned. Weights, threshold, manifest/checkpoint/contract/graph fingerprints and inference settings are sealed before the separate final-test function can access test images. Actual Kaggle validation and held-out test results are archived in the [Phase 4 comparative report](docs/experiments/phase4_kaggle_2026-11-27/README.md). Equal-weight voting matched MobileNetV2’s best individual test accuracy (96.23%) and achieved the lowest test BCE (0.102667) and highest ROC-AUC (0.996394). Validation-optimized voting achieved 92.45% test accuracy; its original weights and threshold remain frozen. Per-image predictions are withheld after privacy/redistribution review; no ensembles are assumed to outperform individuals.

Enabled runs save individual/aligned predictions and ground truth, candidate searches, frozen configuration, comparative metrics, confusion matrices, score-based ROC curves and disagreement analysis beneath `<artifact-root>/phase4/<name>`. Hard majority voting has no probability-based BCE/AUC; undefined metrics are explicitly null. All execution flags default to False. Both output-free notebooks support Kaggle, Colab and local execution without retraining or resplitting. Notebook 05 learning never reloads CNNs or regenerates validation predictions; its comparisons go under `validation/ensemble/`, preserving the individual reports under `validation/`. A shared opt-in handoff-copy utility stages read-only Kaggle outputs into fresh writable artifact storage without changing their bytes.

See [the Phase 4 guide](docs/phase4-ensemble-evaluation.md) for private checkpoint layout, original-path contract compatibility, safe portability, the one-attempt test gate, artifact layout and image-level/single-dataset limitations. Training validation records remain distinct from the now-archived held-out test measurements. Verify the Phase 4 archive with `python scripts/verify_phase4_artifacts.py`.

## Configuration and training settings

`configs/project.yaml` is the central source of truth and replaces the original JSON scaffolds. It sets the three core models, optional baselines, 224×224×3 inputs, batch size 16, seed 42, 70/15/15 splits, classification head, preprocessing, augmentation, checkpoints and early stopping. Stage 1 is 10 epochs at 0.001; stage 2 is 20 epochs at 0.0001 with selective fine-tuning and frozen batch normalization. These settings are not validated hyperparameters.

Dataset paths resolve relative to the config's project root. Export `DATASET_ROOT` for selected dataset storage and `ARTIFACTS_ROOT` for checkpoints, including Colab/Kaggle mounts. Core training uses CPU or available TensorFlow accelerators and is tested with TensorFlow 2.20.0 on CPU. Train models independently; model construction with ImageNet weights can download parameters only when explicitly invoked.

`scripts/train_core.py` now supports explicit opt-in core training from a saved session and frozen CSV. Evaluation/export scripts remain disabled. `scripts/prepare_dataset.py` remains a separate preparation entry point and is never called during training. VGG experiments remain optional; only core models are intended for browser export and the application. Use the separate conversion requirements later and validate exact compatibility.

Experimental ensemble weights are intentionally unset in central training configuration; Phase 4 selects and freezes them from actual validation predictions in a separate immutable run artifact. Equal and weighted heatmap fusion are both planned, with the ensemble heatmap as the main web visualization. Do not invent trained weights, thresholds or performance results.

## Privacy and reports

Images, image previews, quality checks, predictions, heatmaps, PDF generation and image insertion must stay in the browser. Only after explicit consent may an allowlisted analytical JSON summary travel to Flask and Gemini. Never transmit retinal images, image URLs/base64, file names, patient identifiers, or unrestricted user text. The future schema must reject unknown fields and invalid values before any external call. No request body logging or report persistence is planned by default.

Gemini should narrate supplied results without inventing findings, medical advice or clinical diagnoses. Generated reports must state that this is an experimental prototype, summarize image quality and uncertainty, explain limitations and clarify that a heatmap does not establish pathology. Users should be able to generate a local structured PDF without consenting to Gemini. See [privacy design](docs/privacy.md).

## Validation and next steps

The dataset foundation includes tested patient grouping and preprocessing. Core construction and two-stage training are now implemented; the next research step is deliberate real-data training/evaluation using the frozen manifest. No real-data training occurred during implementation. Evaluate on an untouched test set after selecting thresholds and ensemble weights using validation data. Add browser conversion only after model and preprocessing metadata are stable. Validate probability and Grad-CAM parity before enabling inference or reports.

Required future checks include dataset leakage, calibration, sensitivity/specificity and class imbalance, cross-dataset generalization, quality thresholds, browser memory/latency, tensor disposal, payload privacy/consent, and local PDF behavior. Individual Grad-CAM target layers, TF.js gradient support and heatmap fusion require architecture-specific validation; ordinary model conversion alone does not guarantee usable browser Grad-CAM.

Reported validation measurements do not establish clinical validity or deployment readiness. See [research workflow](docs/research-workflow.md) and [architecture](docs/architecture.md).
