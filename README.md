# SEMICON / DRIFT-SENSE

## Model D — SEM Image Localization

SEMICON / DRIFT-SENSE is a deep-learning-based system for locating a
reference SEM image within a search SEM image.

The final inference system uses **Model D**, a CNN-based spatial-correlation
architecture with soft-argmax coordinate decoding.

## Repository Structure

```text
semicon_submission/
├── README.md
├── LOCALIZATION_MANUAL.md
├── GENERATOR5_MANUAL.md
├── REFERENCES.md
├── requirements.txt
├── models/
│   └── model.pt
├── generator/
│   ├── generator5.py
│   ├── augment_generator5.py
│   ├── fix_generator5_filename_pairing.py
│   ├── prepare_generator5_training_metadata.py
│   └── validate_generator5_dataset.py
├── localization/
│   └── localize.py
├── training/
│   ├── train_model_D.py
│   ├── evaluate_model_D.py
│   └── semicon.ipynb
└── test_cases/
    ├── test cases/
    │   ├── case_01/ ... case_30/
    │   │   ├── reference.png
    │   │   └── search.png
    └── test results/
        ├── cases/
        │   ├── case_01.png ... case_30.png
        └── worst_case/
            ├── worst_case.png
            └── explanation.txt

```

## Main Components

| Component | Purpose |
|---|---|
| `models/model.pt` | Trained Model D checkpoint |
| `localization/localize.py` | Standalone inference program |
| `training/train_model_D.py` | Model D training implementation |
| `training/evaluate_model_D.py` | Test-set evaluation |
| `training/semicon.ipynb` | Training notebook |
| `generator/generator5.py` | Configurable DRAM/FinFET dataset generator |
| `GENERATOR5_MANUAL.md` | Generator usage manual |
| `LOCALIZATION_MANUAL.md` | Localization usage manual |
| `REFERENCES.md` | Supporting references |
| `test_cases/` | 30 clean test cases and generated results |

## 1. Model and Inference

The trained checkpoint is:

```text
models/model.pt
```

The standalone inference script is:

```text
localization/localize.py
```

The inference script contains the Model D architecture required to load the
checkpoint, so **`training/train_model_D.py` is not required at inference
time**. The training source is retained separately for reproducibility and
submission requirements.

Run from the repository root:

```bat
python localization\localize.py ^
  --reference "D:\images\reference.png" ^
  --search "D:\images\search.png" ^
  --model "E:\semicon\semicon_submission\models\model.pt"
```

The three paths can point to different locations.

The output reports the predicted center in the original **1000 × 1000 pixel**
coordinate system.

## 2. Image Requirements

```text
Reference : 1000 × 1000 pixels
Search    : 1000 × 1000 pixels
```

## 3. Synthetic Dataset Generation

The configurable generator is:

```text
generator/generator5.py
```

It accepts architecture style, number of pairs, and output directory.

DRAM:

```bat
python generator\generator5.py ^
  --architecture DRAM ^
  --pairs 30 ^
  --output-dir "D:\RANE\dram_30"
```

FinFET:

```bat
python generator\generator5.py ^
  --architecture FinFET ^
  --pairs 30 ^
  --output-dir "D:\RANE\finfet_30"
```

Generated output:

```text
output/
├── reference/
├── search/
├── visualization/
├── metadata.json
└── generation_config.json
```

`metadata.json` records the ground-truth target position, including
`center_x` and `center_y` in the final 1000 × 1000 search-image coordinate
system.

See `GENERATOR5_MANUAL.md`.

## 4. Dataset Generation Principle

Generator 5 follows:

```text
Complete physical scene
        ↓
Natural target crop selected
        ↓
Independent reference capture
        ↓
Independent complete-scene search capture
        ↓
10× downsampling
        ↓
1000 × 1000 search image
        ↓
Ground-truth coordinates recorded
```

The reference image is **not pasted into the search image**.

Ground-truth annotations are kept out of the clean search images and are
shown only in visualization outputs.

## 5. Training and Evaluation

Training:

```text
training/train_model_D.py
training/semicon.ipynb
```

Evaluation:

```text
training/evaluate_model_D.py
```

## 6. Test Cases

Thirty clean input cases are under:

```text
test_cases/test cases/
```

Each case contains:

```text
reference.png
search.png
```

Generated visual results are under:

```text
test_cases/test results/
```

The worst-case visualization and explanation are under:

```text
test_cases/test results/worst_case/
```

## 7. Documentation

- `LOCALIZATION_MANUAL.md` — standalone inference setup and commands.
- `GENERATOR5_MANUAL.md` — generator setup, parameters, outputs, and GT.
- `REFERENCES.md` — supporting sources.

## 8. Requirements

Install the repository requirements with:

```bat
pip install -r requirements.txt
```

For dataset generation, the generator uses Python, NumPy, and Pillow.
