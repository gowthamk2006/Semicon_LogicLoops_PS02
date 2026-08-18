# SEMICON

## Reference-Pattern Localization in Search Images

**SEMICON** is a computer-vision project developed at **Chennai Institute of Technology** for locating a selected reference pattern inside a search SEM image and estimating its center position as image coordinates `(x, y)`.

The final submitted system uses a **CNN-based spatial-correlation localization model (Model D)**. The model receives a reference image and a search image and produces a spatial score map from which the target center is estimated using **soft-argmax**.

The project also includes the Generator 5 dataset-generation pipeline and the earlier classical/experimental training code used during development.

## Team

- Senthoor Kumaran S
- Gowtham K
- M C Thejeeswaran
- Madhesh Raj M

**Institution:** Chennai Institute of Technology

---

## 1. Project Objective

Given:

1. a **reference image** containing the pattern of interest, and
2. a **search image** containing the pattern at an unknown location,

the system estimates the center coordinate of the reference pattern inside the search image.

The localization output is:

```text
Predicted center X : x px
Predicted center Y : y px
```

The project is designed for SEM-image localization under variations such as:

- blur
- brightness/contrast changes
- drift
- noise
- rotation
- scaling
- shear

A major challenge is **periodic ambiguity**, where visually similar structures can occur at multiple locations. The learned spatial-correlation approach is designed to learn discriminative spatial features rather than relying only on raw pixel similarity.

---

## 2. Dataset and Data Pipeline

The project uses the **Generator 5** synthetic dataset pipeline.

Each sample contains:

```text
Reference image
       +
Search image
       +
Ground-truth target center
       ↓
Training / validation / test metadata
```

The generated dataset contains augmented reference/search pairs representing image transformations such as:

- rotation
- shear
- drift
- noise
- blur
- brightness/contrast
- scaling

The final dataset used during development contains:

```text
Training   : 1680 samples
Validation : 210 samples
Test       : 210 samples
```

The reference and search images used for the final model are:

```text
1000 × 1000 pixels
```

The metadata CSV files are stored under:

```text
generator5_augmented/training_metadata/
```

with:

```text
train.csv
validation.csv
test.csv
```

---

## 3. Final Model — Model D

The final submitted deep-learning model is **Model D: Spatial Correlation Localization**.

### Architecture

```text
Reference Image ──→ CNN Encoder ──┐
                                  ├──→ Spatial Correlation
Search Image ─────→ CNN Encoder ──┘
                                           ↓
                                  Correlation Head
                                           ↓
                                  Spatial Score Map
                                           ↓
                                     Soft-Argmax
                                           ↓
                                      (x, y)
```

### Model configuration

```text
Raw input                 : 1000 × 1000
Internal image size       : 128 × 128
Encoder channels          : 1 → 32 → 64 → 96 → 128 → 128
Correlation               : spatial/depthwise feature correlation
Correlation head          : 1 → 32 → 16 → 1
Output                    : spatial score map
Decoder                   : Soft-Argmax
Soft-Argmax temperature   : 0.05
Trainable parameters      : 338,369
```

The model was trained using:

```text
Optimizer                : AdamW
Learning rate            : 0.0001
Weight decay             : 0.0001
Gradient clipping        : 5.0
Batch size               : 4
Final training run       : 20 epochs
```

The final checkpoint selected during development is the **best validation checkpoint**, rather than automatically using the last training epoch.

---

## 4. Final Model File

The standalone submission model is stored as:

```text
models/model.pt
```

The filename `model.pt` is intentional so that the inference package has a simple, fixed default model location.

If the model is moved to another location, the inference command can be supplied with the new checkpoint path when supported by the installed `localize.py`.

---

## 5. Repository Structure

The final submission is organized as:

```text
semicon_submission/
│
├── README.md
├── requirements.txt
├── INFERENCE_MANUAL.md
│
├── models/
│   └── model.pt
│
├── inference/
│   └── localize.py
│
├── generator/
│   ├── generator5.py
│   ├── augment_generator5.py
│   ├── prepare_generator5_training_metadata.py
│   ├── fix_generator5_filename_pairing.py
│   └── validate_generator5_dataset.py
│
├── training/
│   └── [development/training scripts retained as required]
│
├── references/
│
└── results/
```

The `models/` directory contains the checkpoint required for standalone inference.

Large generated datasets are not required for standalone inference.

---

## 6. Environment Requirements

Install the Python dependencies using:

```bash
pip install -r requirements.txt
```

The inference application requires:

- Python 3.x
- PyTorch
- NumPy
- OpenCV
- pandas

The exact versions should be taken from `requirements.txt`.

### GPU

A GPU is **not required for standalone inference**.

The submitted inference program can run on CPU. A CUDA-capable GPU is recommended for model training and large-scale evaluation.

---

## 7. Standalone Inference

The main inference entry point is:

```text
inference/localize.py
```

It accepts two image locations:

```text
--reference
--search
```

and loads the trained model from:

```text
models/model.pt
```

### Windows command

From the repository root:

```bat
python inference\localize.py ^
  --reference "E:\path\to\reference.png" ^
  --search "E:\path\to\search.png"
```

### Example

```bat
python inference\localize.py ^
  --reference "E:\semicon\generator5_augmented\reference\0001_aug_01_rotation.png" ^
  --search "E:\semicon\generator5_augmented\search\0001_aug_01_rotation.png"
```

### Expected output

The program reports:

```text
SEMICON / DRIFT-SENSE
MODEL D LOCALIZATION INFERENCE

Reference : ...
Search    : ...
Model     : ...

RESULT
------------------------------------------------------------------------
Predicted center X : xxx.xx px
Predicted center Y : yyy.yy px

========================================================================
LOCALIZATION COMPLETE
========================================================================
```

The important final outputs are the predicted center coordinates:

```text
(x, y)
```

in the coordinate system of the original **1000 × 1000** search image.

---

## 8. Changing the Model Location

The model does not need to remain physically inside the repository if the inference script supports the `--model` argument.

Example:

```bat
python inference\localize.py ^
  --model "E:\my_models\model.pt" ^
  --reference "E:\images\reference.png" ^
  --search "E:\images\search.png"
```

If `--model` is omitted, the default submission checkpoint is:

```text
models/model.pt
```

This makes it possible to keep the inference code unchanged while replacing or relocating the checkpoint.

---

## 9. Image Requirements

For the final model, the expected raw images are:

```text
Reference : 1000 × 1000 pixels
Search    : 1000 × 1000 pixels
```

The images should be readable by OpenCV and should normally be grayscale SEM images.

The inference code performs the required preprocessing internally.

The user does **not** need to manually resize the images to 128 × 128.

---

## 10. Training and Development

Several approaches were investigated during development.

### Classical NCC baseline

A normalized cross-correlation approach was used as a classical baseline.

The baseline helps establish the difficulty of locating a reference pattern inside a larger/repetitive search image.

### Learned localization experiments

Multiple CNN-based localization approaches were investigated during development, including:

- direct coordinate regression
- spatial score-map localization
- feature correlation
- true sliding feature correlation

The final submitted learned architecture is **Model D**.

The development experiments are retained separately from the standalone inference entry point.

---

## 11. Evaluation

The official test set contains:

```text
210 samples
```

For the localization task, the primary evaluation metrics are based on coordinate error.

For a ground-truth coordinate `(x, y)` and prediction `(x̂, ŷ)`:

```text
Euclidean Error
= sqrt((x̂ - x)² + (ŷ - y)²)
```

Recommended reported metrics include:

```text
Mean Euclidean Error
Median Euclidean Error
Minimum Error
Maximum Error
Standard Deviation
MAE-X
MAE-Y
RMSE-X
RMSE-Y
Accuracy within 5 px
Accuracy within 10 px
Accuracy within 25 px
Accuracy within 50 px
Accuracy within 100 px
```

A complete final evaluation should use **all 210 test samples**.

A separate set of 30 representative samples can be used for visual demonstration, and the worst-error sample should be reported separately.

---

## 12. Why mAP / Confusion Matrix Are Not Primary Metrics

The final model performs **continuous coordinate localization**.

It predicts:

```text
(x, y)
```

rather than discrete classes or bounding boxes.

Therefore:

- conventional classification accuracy is not the primary metric;
- a conventional confusion matrix is not directly applicable;
- precision/recall/F1 are not directly applicable to the continuous coordinate output;
- mAP@50 and mAP@95 are object-detection metrics based on bounding-box overlap and are not directly defined for this point-localization output.

For this project, **Euclidean localization error and tolerance-based localization accuracy** directly measure the required task.

If an external evaluation specification requires detection-style metrics, the exact specification should be followed and a corresponding bounding-box representation should be defined explicitly rather than inventing incompatible metrics.

---

## 13. Reproducibility

The complete development workflow is:

```text
Dataset Generation
        ↓
Augmentation
        ↓
Metadata Preparation
        ↓
Train / Validation / Test Split
        ↓
Baseline and Model Experiments
        ↓
Model D Training
        ↓
Best Validation Checkpoint
        ↓
210-Sample Test Evaluation
        ↓
Standalone Inference
```

For standalone inference, only the following are required:

```text
Python environment
        +
requirements.txt
        +
inference/localize.py
        +
models/model.pt
        +
reference image
        +
search image
```

The training dataset and CSV metadata are **not required** to run standalone inference.

---

## 14. Important Submission Notes

1. Keep the final checkpoint at:

```text
models/model.pt
```

2. Keep the inference entry point at:

```text
inference/localize.py
```

3. Do not modify the model architecture in `localize.py` unless the checkpoint architecture is changed at the same time.

4. If `model.pt` is moved, provide its new location through the supported model-path option or restore it to:

```text
models/model.pt
```

5. Reference and search images used for inference should be 1000 × 1000 pixels.

6. The reported final test metrics must be calculated on the complete 210-image test split.

---

## 15. Project Status

The final submission focuses on:

```text
Model D
   ↓
Spatial correlation
   ↓
Localization score map
   ↓
Soft-argmax
   ↓
Predicted (x, y)
```

The repository separates **training/development code** from the **standalone inference application**, allowing the trained model to be demonstrated without requiring the complete training dataset.

---

## License

This repository is intended for academic/project submission and evaluation.
