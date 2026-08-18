# SEMICON / DRIFT-SENSE
# Model D Localization Manual

## 1. Purpose

`localization/localize.py` is the standalone inference program for the
trained Model D localization system.

It accepts a reference SEM image, a search SEM image, and a trained Model D
checkpoint, then predicts the center of the reference pattern in the search
image.

The script contains the Model D architecture required to load the supplied
checkpoint. Therefore, **`train_model_D.py` is not required during inference**.

The training source remains in the repository for reproducibility and the
training-code submission requirement.

## 2. Required Inference Files

The minimum inference package is:

```text
localize.py
model.pt
```

The reference and search images can be anywhere.

Example:

```text
D:\SEMICON\
├── model\
│   └── model.pt
├── inference\
│   └── localize.py
└── images\
    ├── reference.png
    └── search.png
```

## 3. Environment

Recommended:

```text
Python 3.10+
```

Install repository dependencies:

```bat
pip install -r requirements.txt
```

The inference program uses PyTorch, NumPy, and OpenCV.

CPU inference is supported. CUDA is used automatically when a compatible
CUDA-enabled PyTorch installation and GPU are available.

## 4. Image Requirements

```text
Reference image : 1000 × 1000 pixels
Search image    : 1000 × 1000 pixels
```

Images are processed as grayscale.

## 5. Command

General form:

```bat
python localize.py ^
  --reference "<REFERENCE_PATH>" ^
  --search "<SEARCH_PATH>" ^
  --model "<MODEL_PATH>"
```

Example:

```bat
python localize.py ^
  --reference "D:\images\reference.png" ^
  --search "F:\test_data\search.png" ^
  --model "E:\semicon\semicon_submission\models\model.pt"
```

The three paths may be completely different locations.

## 6. Running From the Repository

From:

```text
E:\semicon\semicon_submission
```

run:

```bat
python localization\localize.py ^
  --reference "D:\reference.png" ^
  --search "D:\search.png" ^
  --model "E:\semicon\semicon_submission\models\model.pt"
```

## 7. Expected Output

A successful run displays approximately:

```text
========================================================================
SEMICON / DRIFT-SENSE
MODEL D LOCALIZATION INFERENCE
========================================================================
Reference : D:\reference.png
Search    : D:\search.png
Model     : E:\semicon\semicon_submission\models\model.pt
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

The predicted coordinates are in the original **1000 × 1000 pixel**
coordinate system.

## 8. Coordinate Convention

```text
(0,0) ───────────────────────→ X
  │
  │
  │
  ↓
  Y
```

For a 1000 × 1000 image:

```text
X: 0–999 px
Y: 0–999 px
```

## 9. Model Architecture

Model D uses:

- CNN-based feature extraction
- Spatial correlation
- Correlation score-map generation
- Soft-argmax coordinate decoding

The standalone inference file contains the architecture needed to load
`model.pt`.

## 10. Training Files

The repository also contains:

```text
training/train_model_D.py
training/semicon.ipynb
```

These are provided for training reproducibility and submission requirements.
They are not required for standalone inference.

## 11. Thirty Test Cases

Clean test inputs are stored under:

```text
test_cases/test cases/
```

Each case contains:

```text
reference.png
search.png
```

Example:

```bat
python localization\localize.py ^
  --reference "test_cases\test cases\case_01\reference.png" ^
  --search "test_cases\test cases\case_01\search.png" ^
  --model "models\model.pt"
```

Generated result visualizations are stored separately under:

```text
test_cases/test results/
```

## 12. Troubleshooting

### `--model` is unrecognized

Run:

```bat
python localize.py --help
```

The current standalone file must show:

```text
--reference
--search
--model
```

If it only shows the older arguments, replace the old `localize.py` with the
current standalone version.

### Training file not found

If the program asks for:

```text
training\train_model_D.py
```

you are running the older non-standalone inference script. Replace it with
the current standalone `localize.py`.

### Checkpoint mismatch

If you see missing keys, unexpected keys, or size mismatches, verify that:

1. `model.pt` is the final Model D checkpoint.
2. You are using the current standalone `localize.py`.
3. The checkpoint has not been replaced by a different model.

### Image not found

Check paths with:

```bat
dir "D:\reference.png"
dir "D:\search.png"
dir "E:\semicon\semicon_submission\models\model.pt"
```

Use quotes around paths containing spaces.

### CPU output

```text
Device : cpu
```

is normal. CUDA is optional for inference.

## 13. Final Checklist

- [ ] Current standalone `localize.py`
- [ ] Correct Model D `model.pt`
- [ ] Reference image exists
- [ ] Search image exists
- [ ] Both images are 1000 × 1000
- [ ] Python environment installed
- [ ] Requirements installed
- [ ] Command paths are correct

Run:

```bat
python localize.py ^
  --reference "<REFERENCE>" ^
  --search "<SEARCH>" ^
  --model "<MODEL>"
```

A successful run ends with:

```text
LOCALIZATION COMPLETE
```
