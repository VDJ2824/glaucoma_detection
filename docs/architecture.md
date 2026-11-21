# Architecture contract

Research pipeline: configurable local dataset → grouped splits → architecture-specific preprocessing → independent core training → held-out evaluation → validation-derived ensemble metadata → core-only TF.js export.

Browser pipeline: local image → quality assessment → three core predictions → weighted result → individual Grad-CAM → equal/weighted aligned heatmap fusion → ensemble visualization → local PDF. Expose individual scores alongside the ensemble and identify the fusion method. A quality failure must be communicated and must not silently produce a confident assessment.

Optional report pipeline: explicit consent → strict structured analytics allowlist → Flask → Gemini textual response → browser review and local PDF insertion. The Flask service does not need image-processing or training dependencies.

The reusable model factory will map architecture names to ImageNet applications, preprocessing and a lightweight classification head. No TensorFlow import or network activity is necessary to inspect the current placeholder modules. Keep core and baseline registries separate; do not infer core membership from all registered architectures.

Persist class order, input range, resize/color conventions, preprocessing placement, output semantics, model versions, validation-selected thresholds/weights and Grad-CAM target layers with each export. EfficientNetB0 may embed input rescaling: inspect the actual graph and avoid applying rescaling twice. Maintain Python/browser parity fixtures before enabling predictions.

Browser Grad-CAM requires access to differentiable feature maps and class scores. Verify conversion preserves the required operations, expose an appropriate gradient-capable graph or an explicitly designed feature/head split, and check both numerical parity and memory cost. Do not substitute static sample heatmaps for actual explanations.

Future segmentation and alternative experiments live outside core classification. Add task-specific data contracts, losses, metrics and export routes when needed without coupling them to the current browser app.
