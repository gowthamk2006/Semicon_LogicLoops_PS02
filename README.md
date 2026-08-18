# SEMICON

## Reference-Pattern Localization in Search Images

**SEMICON** is a computer-vision project developed at **Chennai Institute of Technology** for locating a selected reference pattern inside a larger search image and estimating its position as image coordinates `(x, y)`.

The project investigates both a classical computer-vision baseline and deep-learning approaches for robust localization under image transformations and periodic/repetitive structures.

## Team

- Senthoor Kumaran S
- Gowtham K
- M C Thejeeswaran
- Madhesh Raj M

**Institution:** Chennai Institute of Technology

---

## Project Objective

Given:

1. a reference image containing the pattern of interest, and
2. a larger search image containing multiple possible locations,

the system estimates the location of the reference pattern in the search image.

The final localization output is expressed as the center coordinate:

```text
(x, y)
```

in pixels of the search image.

A major challenge addressed by the project is **periodic ambiguity**: visually similar blocks can occur at multiple locations, causing conventional template matching to select an incorrect but visually similar region.

---

## Dataset and Data Pipeline

The project uses a generated dataset containing reference/search image pairs. The Generator 5 pipeline was developed to create augmented examples representing several image variations, including:

- blur
- brightness/contrast changes
- drift
- noise
- rotation
- scaling
- shear

The generated dataset is accompanied by training metadata and a fixed train/validation/test split.

The repository contains the scripts used for dataset generation and preparation. 
---

## Methods

### 1. Classical NCC Baseline

The first localization approach uses normalized cross-correlation:

```text
Reference
    ↓
Resize to 100 × 100
    ↓
Full-image template matching
    ↓
TM_CCOEFF_NORMED
    ↓
Best matching location
    ↓
Predicted (x, y)
```

This provides a classical baseline against which the learned models can be compared.

The standalone implementation is available in:

```text
inference/localize.py
```

### 2. Model A

Model A uses a shared convolutional encoder for the reference and search images followed by depthwise cross-correlation and a heatmap-based localization head.

The training implementation is provided in:

```text
training/train_model_A.py
```

The model uses a coordinate-aware objective together with heatmap localization and soft-argmax coordinate extraction.

### 3. Model B

Model B extends the learned localization approach with hard-negative training and an InfoNCE objective. For each sample, the target block is treated as the positive while the other blocks provide hard negatives.

The training implementation is provided in:

```text
training/train_model_B.py
```

The purpose of this experiment is to improve discrimination between the correct location and visually similar periodic alternatives.

---

## Repository Structure

```text
SEMICON/
│
├── README.md
├── requirements.txt
│
├── generator/
│   ├── generator5.py
│   ├── augment_generator5.py
│   ├── prepare_generator5_training_metadata.py
│   ├── fix_generator5_filename_pairing.py
│   └── validate_generator5_dataset.py
│
├── inference/
│   └── localize.py
│
├── training/
│   ├── baseline_ncc_generator5.py
│   ├── train_model_A.py
│   ├── train_model_B.py
│   └── evaluate_model_AB.py
│
├── results/
└── references/
```

---

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build appropriate for the CUDA version available on the target machine.

---

## Standalone Inference

The repository includes a standalone classical localization inference script.

From the repository root:

```bash
python inference/localize.py --reference path/to/reference.png --search path/to/search.png
```

Example:

```bash
python inference/localize.py --reference examples/reference.png --search examples/search.png
```

The program reports:

- predicted center X
- predicted center Y
- NCC score
- matched top-left coordinate

An optional top-k output can be requested:

```bash
python inference/localize.py --reference reference.png --search search.png --top-k 3
```

The inference script does not require the training dataset, CSV metadata, or a trained checkpoint.

---

## Training

### Model A

Training can be started with the project training script after preparing the dataset and metadata:

```bash
python training/train_model_A.py --epochs 20 --batch-size 4
```

### Model B

Model B supports the hard-negative/InfoNCE training phases implemented in the script.

Example:

```bash
python training/train_model_B.py --data-root /path/to/generator5_augmented --phase 2 --epochs 20 --batch-size 8
```

The exact dataset paths can be supplied through the command-line arguments supported by each training script.

---

## Classical Baseline Evaluation

The NCC baseline can be evaluated using:

```bash
python training/baseline_ncc_generator5.py
```

The evaluation uses the prepared test metadata and reports localization error and block-level matching metrics.

---

## Model A vs Model B Evaluation

The evaluation utility compares the trained learned models on the fixed test split:

```bash
python training/evaluate_model_AB.py --data-root /path/to/generator5_augmented --device cuda
```

It is an evaluation tool rather than the standalone inference entry point.

---

## Reproducibility

The project separates the workflow into:

```text
Dataset generation
        ↓
Metadata preparation
        ↓
Classical baseline
        ↓
Model A training
        ↓
Model B / hard-negative training
        ↓
Test evaluation
        ↓
Standalone inference
```

The repository contains the scripts required to reproduce the major processing and training stages. Large generated datasets and runtime-specific files are not included.

---

## Current Submission Note

The repository contains both the classical NCC baseline and the deep-learning training implementations. The standalone inference entry point is intentionally lightweight and does not require access to the training dataset or metadata.

Model checkpoint files are kept separate from the source-code repository when they are too large for normal Git storage. If a trained checkpoint is supplied separately for evaluation, its expected path should be documented alongside the corresponding model code.

---

## License

This repository is intended for academic/project submission and evaluation.
