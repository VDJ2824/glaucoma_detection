# Dataset preparation and ACRIMA download organization

## Organizing your downloaded images

Keep the downloaded archive and extracted original images unchanged. Do not rename/move original images solely to fit this example: set the configured root and folder mapping to match your extraction instead. No command in this project downloads datasets, edits originals, creates labels or fabricates patient IDs.

### Filename label mode and provenance

The [original dataset paper, Diaz-Pinto et al. (2019)](https://doi.org/10.1186/s12938-019-0649-y) identifies ACRIMA as a glaucoma-labelled dataset and confirms **705 images: 396 glaucomatous and 309 normal**. These totals match the user's Colab inspection. This establishes the class meanings and published counts, not patient identity.

The supplied flat-release naming convention is `Im<number>_g_ACRIMA.jpg` for glaucoma and `Im<number>_ACRIMA.jpg` for normal. The [OpenMedLab ACRIMA catalogue](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/ACRIMA.md) independently describes the same filename stems. Verification on 22 November 2026: the original paper was accessible, but the original Figshare shared release returned HTTP 403 and the original project's `Characteristics.md` was empty. Therefore, **the original archive's filename labels were not independently audited here**: filename support implements the user's inspected release convention, corroborated by the catalogue. Aggregate counts alone do not validate every image's annotation. Keep the original release provenance and use a verified annotation CSV for renamed or differently labelled copies.

`configs/project.yaml` now defaults to explicit filename mode. Point `root` (or exported `DATASET_ROOT`) directly at the **flat image directory**, not a wrapper containing `Images/`:

```text
data/raw/ACRIMA/
  Im310_g_ACRIMA.jpg
  Im304_ACRIMA.jpg
  ...
```

```yaml
datasets:
  selected: acrima
  acrima:
    root: data/raw/ACRIMA
    label_mode: filename
    annotations_csv: null
```

The adapter uses a **full, case-insensitive match**, equivalent to `Im[0-9]+(_g)?_ACRIMA\.jpg`. Label 1 requires the optional `_g` group; label 0 requires the complete normal pattern. It never labels arbitrary files normal merely because `_g_` is absent. Examples such as `im310_G_acrima.JPG` are accepted. Extra suffixes, unknown disease markers, `.png`/`.jpeg` copies, unrelated files, sidecars and nested directories are explicitly rejected in this strict flat mode. Keep ancillary files outside the configured image directory without changing original images. Empty directories and symlinks escaping the dataset root also fail. Image numbers remain image IDs only; `patient_id` and `split` stay blank. No filenames are renamed, moved or edited. Discovery assigns labels without decoding; existing validation subsequently detects unreadable JPEGs.

### Existing folder and CSV modes

For an existing labelled folder layout, set `label_mode: folders` and map exact relative directory names to glaucoma labels. Folder mode remains recursive and supports JPG/JPEG, PNG, BMP, TIF and TIFF. Files outside mapped folders are not scanned. A typical **project convention**, not the original archive layout, is:

```yaml
datasets:
  selected: acrima
  acrima:
    root: data/raw/ACRIMA
    label_mode: folders
    class_directories: {non_glaucoma: 0, glaucoma: 1}
    annotations_csv: null
```

Never infer labels from numeric ordering or treat diabetic-retinopathy labels as glaucoma labels. Do not reorganize original files to satisfy a particular mode.

For verified supplemental annotations or noncanonical filenames, select CSV mode:

```yaml
datasets:
  selected: acrima
  acrima:
    root: data/raw/ACRIMA
    label_mode: csv
    class_directories: null
    annotations_csv: data/annotations/acrima.csv
```

Modes are explicit and never fall back when discovery fails. CSV mode requires an annotation path; other modes reject a nonempty `annotations_csv` to avoid competing label sources. Old configuration without `label_mode` and existing direct `AcrimaAdapter(...)` callers retain the previous behavior: CSV when an annotation path is provided, folders otherwise.

Example annotation format (illustrative rows only, not actual ACRIMA labels):

```csv
image_id,image_path,label,patient_id,split
original-name-1,images/original-name-1.jpg,0,,
original-name-2,images/original-name-2.jpg,1,,
```

`image_id`, `image_path` and `label` are mandatory annotation columns. Image paths resolve relative to the configured dataset root; absolute paths are accepted only if inside that root. `patient_id` and `split` are optional and stay blank if unavailable. IDs must come from supplied dataset metadata; file stems are not assumed to be patient IDs. If patient IDs are only partially available, preparation fails unless you explicitly set `split.patient_level: false`. This is an acknowledged image-level experiment with unknown patient leakage, not a patient-independent evaluation.

The adapter standardizes `dataset_name` to `acrima`. It does not parse arbitrary external annotation formats: translate a verified annotation file into the documented schema while retaining its source/provenance separately. Do not include actual private patient identifiers in source control; metadata CSVs can contain sensitive paths and grouping keys and belong under ignored `data/`.

## Read-only exploration in Notebook 01

After refreshing notebook 00's saved session for the selected label mode and actual dataset path, run `01_dataset_exploration.ipynb`. It calls `glaucoma.data.exploration.explore_dataset` to discover originals and reuse `validate_metadata`. It includes:

- Original class counts/percentages and a distribution chart, keeping duplicates in exploratory counts.
- Original decoded width, height and aspect-ratio statistics (count, mean, standard deviation, min, quartiles and max).
- Seeded per-class RGB sample previews, excluding unreadable/out-of-root files and showing missing samples honestly.
- Exact decoded-pixel duplicate, conflicting annotation, corruption and invalid-path findings without deleting rows or files.

Validation errors are displayed for review, not silently repaired. Samples and dimension measurements include valid duplicate originals; audit findings distinguish duplicates. Near-duplicate limitations of pixel hashing still apply. No split assignments, augmentation, training or image writes occur. Analytical CSV export is optional and disabled by default; keep notebook outputs/private paths and image previews out of Git. Reusable plotting helpers import Matplotlib only when called; it is included in research and development dependencies.

## Install and run locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For an exploration-only workflow, run notebooks 00 and 01. Notebook 02 and the preparation command assign splits and belong to a separate opt-in preparation phase. The command below is for that later phase.

TensorFlow and GPU support are not needed for dataset preparation or tests. `requirements-data.txt` installs only runtime data dependencies; `requirements-dev.txt` also adds pytest. Model training remains disabled.

Paths in YAML resolve against `project_root` relative to the YAML file. The default `project_root: ..` points to the repository. The preparation CLI also accepts an exported `PROJECT_CONFIG` as its default config path. Exported `DATASET_ROOT` overrides the selected dataset's root, while `ARTIFACTS_ROOT` overrides checkpoint locations. `.env` is not automatically loaded by this CLI. Example:

```bash
export DATASET_ROOT=/absolute/path/to/your/ACRIMA
python scripts/prepare_dataset.py --config configs/project.yaml
```

The command writes only derived files, by default:

- `data/processed/acrima_validation.csv`: row/image ID, issue code and severity; written before splitting.
- `data/processed/acrima_metadata.csv`: validated, deduplicated metadata with assigned splits, written only after successful validation and splitting.

A failure raises an error; inspect the validation report and fix the source mapping or annotations, not the images. A metadata file from an earlier successful run may still exist after a failed rerun; only consume output from a successful command. Outputs are prohibited inside the original dataset root and cannot overwrite the annotation source. Changed configuration can produce a new split: retain a frozen manifest and its config for any completed experiment.

## Metadata CSV contract

| Column | Meaning |
| --- | --- |
| `image_id` | Unique original-image identifier; folder discovery uses the relative path, never a fabricated patient ID |
| `image_path` | Resolved absolute local path inside the dataset root |
| `dataset_name` | Adapter's dataset key (`acrima` initially) |
| `patient_id` | Supplied patient grouping key, otherwise blank; grouping is dataset-scoped |
| `label` | Binary glaucoma annotation: `0` non-glaucoma, `1` glaucoma; missing/unknown labels are errors |
| `split` | `train`, `validation`, `test`; assigned only after deduplication, or supplied officially |

CSV readers should use `keep_default_na=False` and string dtype for IDs, preserving leading zeroes. Do not treat missing IDs as one shared patient group.

## Validation, deduplication and splitting

The sequence is **discover originals → validate paths/labels and decode images → deduplicate → split → lazy preprocess/augment**. No augmented files enter deduplication or splitting.

Validation detects missing/invalid labels, empty IDs, duplicate image IDs, out-of-root/missing paths, corrupt images and invalid split names. SHA-256 of decoded BGR pixels and dimensions detects identical images even with different filenames or metadata. It is not a perceptual/near-duplicate detector: differently compressed, resized or cropped versions may need a separate future audit.

`metadata.duplicate_policy: drop_identical` keeps the first discovered copy only when label, dataset, patient key and existing split agree; removed copies remain untouched on disk and receive warning rows in the report. Conflicting duplicate labels, groups or official splits always fail. Use `error` to reject all duplicates instead.

Generated splitting defaults to seeded 70/15/15 and stratification. Without patient IDs, scikit-learn makes image-level stratified splits and the CLI warns that patient independence cannot be established. With complete patient IDs, deterministic candidate group splits approximate the requested sizes and class proportions without dividing any patient across splits; exact ratios may be impossible. All splits must be nonempty and, when stratification is enabled, contain both classes. Insufficient samples/groups fail explicitly. Group assignment uses labels and size only, never model performance. Patients with differing eye-level labels remain grouped.

For documented official splits, fill every annotation's `split` and set `split.mode: official`. Splits are preserved exactly; preparation checks that all three exist, both classes exist if requested, and known patients do not cross partitions. Duplicate images crossing official splits are errors. Partial official assignments fail; generated mode refuses to overwrite any existing assignment. Disabling stratification relaxes class-presence checks, not group isolation. Patient leakage checks can only apply to supplied IDs.

## Preprocessing and augmentation

`preprocess_image(path, cfg, split, rng=None)` decodes BGR and converts to RGB, optionally applies CLAHE to LAB luminance, then letterboxes to 224×224 while retaining aspect ratio and centering with the configured padding value. It returns float32 values using the selected architecture's normalization:

- ResNet50/VGG: RGB to BGR, then subtract ImageNet channel means.
- MobileNetV2: RGB in `[-1, 1]`.
- Keras EfficientNetB0: float RGB in `[0, 255]`, for its embedded preprocessing. Do not divide by 255 again when integrating the model.

Optional training augmentation uses horizontal flips, bounded rotation and brightness changes before normalization. Validation and test paths ignore augmentation even if globally enabled. CLAHE is deterministic preprocessing applied consistently to every split. Training augmentation requires an explicit seeded persistent NumPy generator to avoid repeating the same transform for every image. These are initial research transformations that need empirical validation.

`iter_batches(metadata, cfg, split, epoch=0)` yields image arrays and float labels, shuffles training only, includes the final partial batch, and derives reproducible generator state from seed and epoch. Pass only metadata returned by successful preparation. It does not train models or create a `tf.data.Dataset`; training can wrap this shared interface later. It never writes images.

## Adding other datasets

RIM-ONE DL, DRISHTI-GS, REFUGE, AIROGS and JustRAIGS have path entries only. Selecting them currently raises `NotImplementedError`; no labels or splits are assumed and there is no automatic download.

Implement `DatasetAdapter.discover()` returning the six schema columns and register it in `ADAPTERS` in `src/glaucoma/data/adapters.py`. Verify that dataset's glaucoma label semantics, license, patient keys and official split protocol. Add adapter-specific tests with synthetic fixtures. The shared validation, splitting, preprocessing and batch interface remain unchanged; training code does not need dataset-specific branching. Future segmentation should use a separate mask-aware contract rather than repurposing this classification label column.
