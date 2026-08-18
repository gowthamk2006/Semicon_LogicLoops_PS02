# SEMICON / DRIFT-SENSE — Inference Manual

## 1. Overview

The SEMICON / DRIFT-SENSE inference program takes:

1. A reference image
2. A search image
3. A trained Model D checkpoint

and predicts the target center `(X, Y)` inside the search image.

The current Model D inference expects both input images to be **1000 × 1000 pixels**.

## 2. Folder Structure

```text
semicon_submission/
│
├── inference/
│   └── localize.py
│
├── training/
│   └── train_model_D.py
│
├── models/
│   └── model.pt
│
├── generator/
│   ├── generator5.py
│   ├── augment_generator5.py
│   ├── fix_generator5_filename_pairing.py
│   ├── prepare_generator5_training_metadata.py
│   └── validate_generator5_dataset.py
│
├── references/
├── results/
├── requirements.txt
└── README.md
```

### Files required specifically for inference

```text
inference/localize.py
training/train_model_D.py
models/model.pt
```

`localize.py` loads the exact `ModelD` architecture from
`training/train_model_D.py`. Keep these two files together.

## 3. Required Environment

- Python 3.x
- PyTorch
- NumPy
- OpenCV

The repository also includes pandas for the training/evaluation utilities.

Install dependencies:

```cmd
pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
numpy
opencv-python
pandas
torch
```

A GPU is **not required** for inference. The program uses CUDA when
available and otherwise runs on CPU.

## 4. Model Checkpoint

The default trained model is:

```text
models/model.pt
```

The model may also be stored anywhere else. Its location can be
supplied at runtime with `--model`; `localize.py` does not need to be
edited.

## 5. Input Image Requirements

The program requires:

- **Reference image** — the pattern/template to be localized.
- **Search image** — the image in which the target is located.

Both images must currently be:

```text
1000 × 1000 pixels
```

The program internally resizes them for Model D processing.

## 6. Basic Inference Command

From the repository root:

```cmd
python inference\localize.py ^
  --reference "PATH_TO_REFERENCE_IMAGE" ^
  --search "PATH_TO_SEARCH_IMAGE"
```

If `--model` is omitted, the program automatically uses:

```text
models\model.pt
```

## 7. Command-Line Arguments

### `--reference`

Replace the value with the reference image path.

Example:

```cmd
--reference "E:\images\reference.png"
```

### `--search`

Replace the value with the search image path.

Example:

```cmd
--search "E:\images\search.png"
```

### `--model`

Optional path to the trained checkpoint.

Example:

```cmd
--model "D:\trained_models\model.pt"
```

This allows the model location to be changed every time the program is
run without changing the Python source code.

## 8. Complete Example — Default Model

If:

```text
Reference:
E:\semicon\generator5_augmented\reference\0001_aug_01_rotation.png

Search:
E:\semicon\generator5_augmented\search\0001_aug_01_rotation.png

Model:
E:\semicon\semicon_submission\models\model.pt
```

run:

```cmd
cd /d E:\semicon\semicon_submission

python inference\localize.py ^
  --reference "E:\semicon\generator5_augmented\reference\0001_aug_01_rotation.png" ^
  --search "E:\semicon\generator5_augmented\search\0001_aug_01_rotation.png"
```

Because the model is in `models\model.pt`, `--model` is optional.

## 9. Complete Example — Custom Model Location

If the model is stored at:

```text
D:\trained_models\semicon_model.pt
```

run:

```cmd
python inference\localize.py ^
  --reference "E:\images\reference.png" ^
  --search "E:\images\search.png" ^
  --model "D:\trained_models\semicon_model.pt"
```

Only the paths need to be changed.

## 10. Command Template

```cmd
python inference\localize.py ^
  --reference "YOUR_REFERENCE_IMAGE_PATH" ^
  --search "YOUR_SEARCH_IMAGE_PATH" ^
  --model "YOUR_MODEL_PATH"
```

If using the default `models\model.pt`, omit the `--model` line:

```cmd
python inference\localize.py ^
  --reference "YOUR_REFERENCE_IMAGE_PATH" ^
  --search "YOUR_SEARCH_IMAGE_PATH"
```

### What must be changed?

Change:

```text
YOUR_REFERENCE_IMAGE_PATH
```

to the actual reference image location.

Change:

```text
YOUR_SEARCH_IMAGE_PATH
```

to the actual search image location.

Optionally change:

```text
YOUR_MODEL_PATH
```

to the location of the trained checkpoint.

Do not change the `--reference`, `--search`, or `--model` option names.

## 11. Expected Output

A successful run produces output similar to:

```text
========================================================================
SEMICON / DRIFT-SENSE
MODEL D LOCALIZATION INFERENCE
========================================================================
Reference : ...
Search    : ...
Model     : ...
Input     : 1000 x 1000 px
Method    : Model D spatial correlation
Decoder   : Soft-Argmax

Device    : cpu

RESULT
------------------------------------------------------------------------
Predicted center X : 649.72 px
Predicted center Y : 496.84 px

========================================================================
LOCALIZATION COMPLETE
========================================================================
```

The predicted coordinates are the estimated target center in the
original 1000 × 1000 search image.

For example:

```text
Predicted center X : 649.72 px
Predicted center Y : 496.84 px
```

means:

```text
(X, Y) = (649.72, 496.84)
```

## 12. Quick Start

### Step 1 — Download or clone the repository

Verify the folder structure and the presence of `models/model.pt`.

### Step 2 — Install dependencies

```cmd
pip install -r requirements.txt
```

### Step 3 — Enter the repository

```cmd
cd /d E:\semicon\semicon_submission
```

### Step 4 — Run inference

```cmd
python inference\localize.py ^
  --reference "PATH_TO_REFERENCE_IMAGE" ^
  --search "PATH_TO_SEARCH_IMAGE"
```

### Step 5 — If the model is elsewhere

```cmd
python inference\localize.py ^
  --reference "PATH_TO_REFERENCE_IMAGE" ^
  --search "PATH_TO_SEARCH_IMAGE" ^
  --model "PATH_TO_MODEL"
```

### Step 6 — Read the result

Use:

```text
Predicted center X
Predicted center Y
```

as the predicted localization coordinate.

## 13. Important Notes

- Reference and search images must be 1000 × 1000 pixels for the current implementation.
- The checkpoint must match the `ModelD` architecture in `training/train_model_D.py`.
- Do not modify the Model D architecture when using the supplied checkpoint.
- The model path can be changed at runtime using `--model`.
- A CUDA GPU is optional for inference.
- The inference program does not require the training dataset or CSV files.
