# Research workflow

1. Establish dataset provenance/license, labels, patient grouping, inclusion criteria and independent train/validation/test splits. Keep private data outside version control.
2. Implement reproducible preprocessing, class mapping and quality assessment. Do not apply augmentation to validation/test data.
3. Build ResNet50, MobileNetV2 and EfficientNetB0 independently with ImageNet weights. Train the head, then selectively fine-tune at a lower learning rate. Save each stage and record seeds, runtime, dataset version and hyperparameters.
4. Report accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, confusion matrices, calibration and uncertainty. Define behavior for datasets missing either class. Choose ensemble weights and operating thresholds on validation data only, then evaluate the final frozen system on the test set.
5. Generate individual core Grad-CAM and compare equal/weighted ensemble fusion after spatial alignment and normalization. Assess explanation stability; heatmaps do not prove causal or clinical validity.
6. Optionally train VGG16/VGG19 with comparable splits and report their compute/performance separately. Optional VGG heatmaps never enter the application.
7. Export only core models, validate Python/browser scores and heatmaps, measure artifact sizes and CPU/browser latency and memory on realistic hardware. Quantization is a later experiment requiring renewed validation.
8. Implement local upload, quality feedback, inference, visualization and jsPDF export. Add consented structured text reports through Flask only after privacy validation exists.

Colab/Kaggle notebooks should be thin wrappers around the shared modules, with configurable mounted dataset/checkpoint locations. Do not require all models to train in one session. CPU execution must remain possible, even if full training is slow.
