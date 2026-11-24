# Brain Tumor Detection and Classification Using MRI Images

A leakage-aware comparison of a custom four-block CNN and ImageNet-pretrained models
(ResNet50, VGG19, DenseNet121, EfficientNetB0) on the Kaggle four-class brain MRI dataset
(glioma, meningioma, pituitary tumour, no tumour), with ReLU versus GELU experiments,
stratified cross-validation, Grad-CAM and a saved final model.

**Minor Project-I**, B.Tech Artificial Intelligence, Interdisciplinary Centre for Artificial Intelligence,
ZHCET, Aligarh Muslim University. Author: Farhan Ahmad Ansari. Supervisor: Professor Rashid Ali.

> **Educational research prototype. It is not a clinically validated diagnostic system.**

## Why this project
The reference study (Anand et al., *Scientific Reports* 16:4709, 2026) reports 98.5% accuracy for a custom CNN.
Inspection of the dataset showed that 268 of the 394 images in its original Testing folder (68%) have an exact or
near-duplicate in the Training folder. This project removes duplicates, groups near-duplicates, builds nested
class-stratified group-aware splits (about 70:30, 80:20, 90:10), fixes the model-selection rules before any test
evaluation, and reports what remains.

## Main results (leak-checked protocol)
| Item | Result |
|---|---|
| Cleaned dataset | 2,870 images in 2,095 duplicate groups (394 pixel-identical copies removed) |
| Cross-validated accuracy, pretrained models | about 0.88 to 0.93 |
| Custom 4-block CNN | about 0.79 validation macro-F1 (below the pretrained models) |
| **Final model** | **DenseNet121, ReLU head, trained on 90:10: test accuracy 0.930 (95% CI 0.897 to 0.953), macro-F1 0.933, 328 test images** |
| ReLU vs GELU | no detectable difference |

The reference paper's ranking was not reproduced. Several architectures are statistically indistinguishable;
see the report for limitations (single seed, nested test sets, no patient IDs, source cues that track class).

## Repository layout
```
report/      final project report (PDF)
notebook/    the Kaggle notebook (outputs cleared; a copy with outputs is also included)
final/       final model (final_model.keras), labels, configs, test metrics, model card
splits/      frozen dataset splits
logs/        configuration, selection records, run registry, leakage verification
results/     result tables (validation, cross-validation, test) and removed-image log
figures/     figures used in the report
predict.py   predict the class of one MRI image with the saved model
```

## Data
The dataset is **not** included. Download it from Kaggle:
<https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri>

## Predict with the saved model
```
pip install -r requirements.txt
python predict.py path/to/mri_image.jpg
```
The script applies the training preprocessing (grayscale, centred padding to a square, bicubic resize to 224 x 224).
It was adapted from the prediction cell of the notebook, which was verified on Kaggle.

## Reproduce
The experiments were run in the notebook on a Kaggle Tesla T4 (TensorFlow 2.21.0, Keras 3.13.2, seed 42).
Add the dataset as an input, enable a GPU and internet, and run the cells in order. A full run takes many GPU hours;
the run registry in `logs/` lets interrupted runs resume. GPU training is not bit-reproducible, so numbers can differ slightly.
