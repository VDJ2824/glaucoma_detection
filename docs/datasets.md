# Dataset preparation and ACRIMA download organization

## Organizing the files you download tomorrow

Keep the downloaded archive and extracted original images unchanged. Do not rename/move original images solely to fit this example: set the configured root and folder mapping to match your extraction instead. No command in this project downloads datasets, edits originals, creates labels or fabricates patient IDs.

The simplest supported **example layout** is:

```text
data/raw/ACRIMA/
  non_glaucoma/
    original-image-name-1.jpg
    ...
  glaucoma/
    original-image-name-2.jpg
    ...
```

These folder names are project conventions, not a claim about ACRIMA's archive layout. If the downloaded folders have other names, edit `datasets.acrima.class_directories` in `configs/project.yaml`, mapping each exact relative folder name to `0` (non-glaucoma) or `1` (glaucoma). Verify these meanings against the download's glaucoma annotations/documentation. Never infer labels from file numbering or treat diabetic retinopathy labels as glaucoma labels. Folder discovery recursively includes JPG/JPEG, PNG, BMP, TIF and TIFF files. Non-image files are ignored; an image file with a supported extension that cannot decode is an error. Folders not listed in the mapping are not scanned.

If your download has a single flat image folder or separate annotations, use a manually verified annotation CSV instead. Do not copy images into inferred class folders:

```yaml
datasets:
  selected: acrima
  acrima:
    root: data/raw/ACRIMA
    class_directories: null
    annotations_csv: data/annotations/acrima.csv
```

Example annotation format (illustrative rows only, not actual ACRIMA labels):

```csv
image_id,image_path,label,patient_id,split
original-name-1,images/original-name-1.jpg,0,,
original-name-2,images/original-name-2.jpg,1,,
```

`image_id`, `image_path` and `label` are mandatory annotation columns. Image paths resolve relative to the configured dataset root; absolute paths are accepted only if inside that root. `patient_id` and `split` are optional and stay blank if unavailable. IDs must come from supplied dataset metadata; file stems are not assumed to be patient IDs. If patient IDs are only partially available, preparation fails unless you explicitly set `split.patient_level: false`. This is an acknowledged image-level experiment with unknown patient leakage, not a patient-independent evaluation.

The adapter standardizes `dataset_name` to `acrima`. It does not parse arbitrary external annotation formats: translate a verified annotation file into the documented schema while retaining its source/provenance separately. Do not include actual private patient identifiers in source control; metadata CSVs can contain sensitive paths and grouping keys and belong under ignored `data/`.

## Install and run locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/prepare_dataset.py --config configs/project.yaml
python -m pytest -q
```

TensorFlow and GPU support are not needed for dataset preparation or tests. `requirements-data.txt` installs only runtime data dependencies; `requirements-dev.txt` also adds pytest. Model training remains disabled.

Paths in YAML resolve against `project_root` relative to the YAML file. The default `project_root: ..` points to the repository. The CLI also accepts an exported `PROJECT_CONFIG` as its default config path. Exported `DATASET_ROOT` overrides the selected dataset's root, while `ARTIFACTS_ROOT` overrides checkpoint locations. `.env` is not automatically loaded by this CLI. Example:

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
