# Lightweight Glaucoma Detection via Ensemble Learning

A modular B.Tech CSE major-project scaffold for a lightweight, explainable, privacy-preserving retinal fundus image classification research prototype. Planned inference runs locally in the browser; optional Gemini-generated analytical text is requested through Flask.

**Current status:** configuration and ACRIMA dataset discovery, validation, deduplication, splitting, preprocessing and training-only augmentation are implemented and unit tested. The frontend remains a landing placeholder and Flask provides a health endpoint with disabled reports. Training, inference, image quality assessment, Grad-CAM, conversion, Gemini requests and PDFs remain unimplemented. No datasets or trained weights are bundled; nothing downloads or trains models automatically.

Start with [ACRIMA organization and preparation instructions](docs/datasets.md). For data work alone, install `requirements-dev.txt` and run `python scripts/prepare_dataset.py --config configs/project.yaml` after placing your download and confirming the label mapping. Run `python -m pytest -q` for synthetic-data tests.

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

Use Python 3.10–3.12 and Node.js 20.19+ or 22.12+. Dependencies are declared as version ranges; lock exact versions after installation and compatibility validation. Installation is optional at this scaffold stage.

For dataset development and unit tests (CPU only, no TensorFlow needed):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Research/notebook environment (Python 3.10–3.12, CPU supported; no training starts):

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

Create an empty repository on GitHub without an initial README, then run these commands in your local project folder after reviewing the staged files. Replace the URL with your repository. Check `git rev-parse --show-toplevel` first: this workspace currently inherits a Git repository from a parent directory. Run `git init` **inside `glaucoma_detection`** to create an independent project repository; do not rename branches or add remotes in the parent repository. Skip initialization only if the reported Git root is already this exact project directory, and inspect `git remote -v` before adding a remote.

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

Keep original downloads and filenames unchanged. The following is an **example**, not an assumption about the archive you receive:

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

If the original folders differ, change `datasets.acrima.class_directories`; do not reorganize originals just for the example. For flat image folders, provide a verified annotation CSV. Paths in that CSV are relative to the configured ACRIMA root. Missing patient IDs remain blank; unavailable glaucoma labels are errors. Other datasets have reserved configuration only. See [the complete ACRIMA guide](docs/datasets.md).

## Recommended notebook execution order

| Notebook | Purpose and status |
| --- | --- |
| `00_environment_setup.ipynb` | Implemented platform setup, diagnostics and persistent output configuration |
| `01_dataset_exploration.ipynb` | Scaffold for original-label, class and image review |
| `02_dataset_preparation.ipynb` | Documented shared dataset pipeline; processing explicitly opt-in |
| `03_individual_model_training.ipynb` | Scaffold for independent two-stage training |
| `04_model_evaluation.ipynb` | Scaffold for validation analysis and final held-out metrics |
| `05_ensemble_optimization.ipynb` | Scaffold for validation-only core ensemble weight selection |
| `06_gradcam_analysis.ipynb` | Scaffold for individual and ensemble core explanations |
| `07_cross_dataset_validation.ipynb` | Scaffold pending verified external-dataset adapters |
| `08_tensorflowjs_conversion.ipynb` | Scaffold for isolated core conversion and browser parity |

Start with 00 → 01 → 02 → 03. Repeat 03 independently for each core model. Use 04 for validation analysis, then 05 to freeze ensemble weights; return to 04 for final test evaluation. Continue with 06 → 07 → 08. Test/external results must not drive weight or threshold selection. Optional VGG training stays separate and is never required. Notebook 08 will use a separate conversion environment; it does not install converters or export models yet.

All core logic stays in reusable Python modules. Later notebooks import the relevant modules, mark unimplemented steps explicitly, and load saved session settings. See [notebook details and output contracts](notebooks/README.md). Training checkpoints, histories, metrics and converted models do not exist yet; future implementations must use the configured persistent paths rather than runtime-local defaults.

## Configuration and future training

`configs/project.yaml` is the central source of truth and replaces the original JSON scaffolds. It sets the three core models, optional baselines, 224×224×3 inputs, batch size 16, seed 42, 70/15/15 splits, classification head, preprocessing, augmentation, checkpoints and early stopping. Stage 1 is 10 epochs at 0.001; stage 2 is 20 epochs at 0.0001 with selective fine-tuning and frozen batch normalization. These settings are not validated hyperparameters.

Dataset paths resolve relative to the config's project root. Export `DATASET_ROOT` for selected dataset storage and `ARTIFACTS_ROOT` for checkpoints, including Colab/Kaggle mounts. GPU training remains future work; CPU data processing is supported now. Train each model independently when training is implemented. ImageNet weights must only be fetched by explicitly invoked future training.

Training/evaluation/export scripts remain disabled and produce no artifacts. `scripts/prepare_dataset.py` is the implemented local-only entry point. VGG experiments remain optional; only core models are intended for browser export and the application. Use the separate conversion requirements later and validate exact compatibility.

Ensemble weights are not yet configured; validation-based weight selection remains future work. Equal and weighted heatmap fusion are both planned, with the ensemble heatmap as the main web visualization. Do not invent trained weights, thresholds or performance results.

## Privacy and reports

Images, image previews, quality checks, predictions, heatmaps, PDF generation and image insertion must stay in the browser. Only after explicit consent may an allowlisted analytical JSON summary travel to Flask and Gemini. Never transmit retinal images, image URLs/base64, file names, patient identifiers, or unrestricted user text. The future schema must reject unknown fields and invalid values before any external call. No request body logging or report persistence is planned by default.

Gemini should narrate supplied results without inventing findings, medical advice or clinical diagnoses. Generated reports must state that this is an experimental prototype, summarize image quality and uncertainty, explain limitations and clarify that a heatmap does not establish pathology. Users should be able to generate a local structured PDF without consenting to Gemini. See [privacy design](docs/privacy.md).

## Validation and next steps

The dataset foundation includes tested patient grouping and preprocessing. Next implement core model construction and independent two-stage training, then validate them on real, verified data. Evaluate on an untouched test set after selecting thresholds and ensemble weights using validation data. Add browser conversion only after model and preprocessing metadata are stable. Validate probability and Grad-CAM parity before enabling inference or reports.

Required future checks include dataset leakage, calibration, sensitivity/specificity and class imbalance, cross-dataset generalization, quality thresholds, browser memory/latency, tensor disposal, payload privacy/consent, and local PDF behavior. Individual Grad-CAM target layers, TF.js gradient support and heatmap fusion require architecture-specific validation; ordinary model conversion alone does not guarantee usable browser Grad-CAM.

No performance, clinical validity or deployment readiness is claimed. See [research workflow](docs/research-workflow.md) and [architecture](docs/architecture.md).
