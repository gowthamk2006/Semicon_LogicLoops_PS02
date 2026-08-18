# SEMICON / DRIFT-SENSE
# Generator 5 Usage Manual

## 1. Purpose

`generator5.py` generates synthetic semiconductor image pairs for
SEMICON / DRIFT-SENSE.

Each generated pair contains:

- One **reference image**
- One **search image**
- Ground-truth target coordinates
- An optional visualization showing the ground-truth target location

The generator supports two architecture styles:

- `DRAM`
- `FinFET`

The search image is generated from a complete physical scene. The
reference image is captured independently from the selected natural
physical crop. The reference image is **not pasted into the search image**.

---

## 2. Requirements

Python 3.10+ is recommended.

Install the required packages:

```bat
pip install numpy pillow
```

The generator uses:

- Python standard library
- NumPy
- Pillow

No PyTorch installation is required just to run the generator.

---

## 3. File

Place the modified generator file somewhere accessible, for example:

```text
E:\Downloads\generator5.py
```

Before running it, verify that it is the **modified configurable version**.

Run:

```bat
python generator5.py --help
```

The help output should include:

```text
--architecture
--pairs
--output-dir
--seed
```

If it only shows `--pairs` and `--seed`, you are using the older
version of `generator5.py`.

---

## 4. Required Parameters

The generator requires three main parameters.

### 4.1 Architecture style

Use:

```text
--architecture DRAM
```

or:

```text
--architecture FinFET
```

This determines the semiconductor architecture used to generate the
physical scene.

### 4.2 Number of pairs

Use:

```text
--pairs 30
```

to generate 30 reference/search pairs.

For example:

```text
--pairs 100
```

generates 100 pairs.

### 4.3 Output directory

Use:

```text
--output-dir "E:\RANE\dram_30"
```

This specifies where all generated data will be stored.

The output directory can be anywhere for which you have write access.

---

## 5. Basic Commands

### Generate 30 DRAM pairs

```bat
python generator5.py --architecture DRAM --pairs 30 --output-dir "E:\RANE\dram_30"
```

### Generate 30 FinFET pairs

```bat
python generator5.py --architecture FinFET --pairs 30 --output-dir "E:\RANE\finfet_30"
```

### Generate 100 DRAM pairs

```bat
python generator5.py --architecture DRAM --pairs 100 --output-dir "E:\RANE\dram_100"
```

### Generate 100 FinFET pairs

```bat
python generator5.py --architecture FinFET --pairs 100 --output-dir "E:\RANE\finfet_100"
```

---

## 6. Random Seed

The generator supports an optional seed:

```bat
--seed 20260816
```

For example:

```bat
python generator5.py --architecture DRAM --pairs 30 --output-dir "E:\RANE\dram_30" --seed 20260816
```

Using the same seed and the same generator configuration makes the
generation deterministic.

If `--seed` is omitted, the generator uses its built-in default seed.

---

## 7. Output Folder Structure

For:

```bat
python generator5.py --architecture DRAM --pairs 30 --output-dir "E:\RANE\dram_30"
```

the generated directory will contain:

```text
E:\RANE\dram_30\
│
├── reference\
│   ├── reference_0001.png
│   ├── reference_0002.png
│   ├── ...
│   └── reference_0030.png
│
├── search\
│   ├── search_0001.png
│   ├── search_0002.png
│   ├── ...
│   └── search_0030.png
│
├── visualization\
│   ├── pair_0001.png
│   ├── pair_0002.png
│   ├── ...
│   └── pair_0030.png
│
├── metadata.json
└── generation_config.json
```

---

## 8. Reference Images

The reference images are:

```text
1000 x 1000 pixels
```

Each reference image is independently captured from the selected
1000 x 1000 physical target crop.

The reference image is **not copied or pasted into the search image**.

---

## 9. Search Images

The search images are also:

```text
1000 x 1000 pixels
```

Internally, the generator first creates a:

```text
10000 x 10000 physical scene
```

The natural target is selected from that completed scene.

The complete search scene is independently captured and then
downsampled by a factor of 10 to produce the final 1000 x 1000 search
image.

Therefore, the target occupies approximately:

```text
100 x 100 pixels
```

in the final search image.

---

## 10. Ground-Truth Coordinates

Ground-truth information is stored in:

```text
metadata.json
```

For every generated pair, the metadata records the target position
in the final 1000 x 1000 search-image coordinate system.

Important fields include:

```json
"target_x"
"target_y"
"target_width"
"target_height"
"center_x"
"center_y"
```

The most important fields for localization evaluation are:

```text
center_x
center_y
```

These represent the true center of the reference pattern in the
search image.

Example:

```json
{
    "pair_id": 1,
    "architecture": "DRAM",
    "reference_file": "reference_0001.png",
    "search_file": "search_0001.png",
    "center_x": 543.21,
    "center_y": 421.78
}
```

---

## 11. Visualization Images

The `visualization` folder contains debugging/evaluation images.

These images show the selected target location in the search image.

The ground-truth box is drawn **only in the visualization**.

The actual search image:

```text
search/search_XXXX.png
```

does not contain the ground-truth annotation.

Therefore, the clean search image can be directly used for model
training or inference.

---

## 12. Generation Configuration

The file:

```text
generation_config.json
```

records the main generation settings, including:

- Architecture
- Number of pairs
- Random seed
- Image dimensions
- Physical scene dimensions
- Downsampling factor
- Ground-truth recording
- Image variation settings
- Generation pipeline

This file provides a record of how the dataset was generated.

---

## 13. Recommended Dataset Generation

For a DRAM dataset:

```bat
python generator5.py --architecture DRAM --pairs 300 --output-dir "E:\RANE\dram_300"
```

For a FinFET dataset:

```bat
python generator5.py --architecture FinFET --pairs 300 --output-dir "E:\RANE\finfet_300"
```

For a quick test before generating a large dataset, first generate a
small number:

```bat
python generator5.py --architecture DRAM --pairs 5 --output-dir "E:\RANE\test"
```

Check the generated files before generating hundreds of pairs.

---

## 14. Checking the Generated Dataset

After generation, verify that the output contains:

```text
reference/
search/
visualization/
metadata.json
generation_config.json
```

For 30 pairs, there should be:

```text
30 reference images
30 search images
30 visualization images
```

The metadata file should contain 30 records.

---

## 15. Important Design Property

The generator follows this sequence:

```text
Complete physical scene
        ↓
Select natural target crop
        ↓
Independent reference capture
        ↓
Independent complete-scene search capture
        ↓
10× downsample
        ↓
Final search image
        ↓
Record ground-truth coordinates
```

The reference image is never pasted into the search image.

This is important because the objective is to train/evaluate a
localization system that finds naturally occurring structure rather
than learning to detect an artificially pasted template.

---

## 16. Command Summary

General form:

```bat
python generator5.py --architecture <DRAM|FinFET> --pairs <NUMBER> --output-dir "<OUTPUT_PATH>"
```

Example:

```bat
python generator5.py --architecture DRAM --pairs 30 --output-dir "E:\RANE\dram_30"
```

With a fixed seed:

```bat
python generator5.py --architecture DRAM --pairs 30 --output-dir "E:\RANE\dram_30" --seed 20260816
```

---

## 17. Troubleshooting

### Error: unrecognized arguments

If you see:

```text
error: unrecognized arguments: --architecture ...
```

you are probably running the older `generator5.py`.

Run:

```bat
python generator5.py --help
```

The configurable version must show:

```text
--architecture
--pairs
--output-dir
--seed
```

Replace the old file with the modified `generator5.py` if necessary.

### Error: output directory cannot be created

Check that the specified drive and parent directory exist and that
you have permission to write there.

For example:

```bat
E:\RANE
```

must be accessible before using:

```bat
--output-dir "E:\RANE\dram_30"
```

### Want to test first

Use only 1 or 2 pairs:

```bat
python generator5.py --architecture DRAM --pairs 2 --output-dir "E:\RANE\test"
```

---

## 18. Final Notes

Do not manually edit the generated `metadata.json` if it is being used
as ground truth.

The generated `search` images are the clean model inputs.

The `visualization` images are for human inspection and verification.

The `center_x` and `center_y` values in `metadata.json` are the
ground-truth localization coordinates for evaluation.
