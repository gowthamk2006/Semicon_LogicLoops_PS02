from pathlib import Path
import json
import csv

ROOT = Path(r"E:\semicon\generator5_augmented")
REF_DIR = ROOT / "reference"
SEARCH_DIR = ROOT / "search"
METADATA_JSON = ROOT / "metadata.json"
METADATA_CSV = ROOT / "metadata.csv"


def common_stem(filename):
    """
    Convert either:

        reference_0001_aug_01_rotation.png
        search_0001_aug_01_rotation.png

    into the same logical pair key:

        0001_aug_01_rotation
    """
    stem = Path(filename).stem

    if stem.startswith("reference_"):
        return stem[len("reference_"):]

    if stem.startswith("search_"):
        return stem[len("search_"):]

    # Also support the alternative suffix convention.
    if stem.endswith("_reference"):
        return stem[:-len("_reference")]

    if stem.endswith("_search"):
        return stem[:-len("_search")]

    return stem


def paired_filename(filename):
    return common_stem(filename) + ".png"


print("=" * 72)
print("GENERATOR 5 FILENAME PAIRING REPAIR - V2")
print("=" * 72)

if not ROOT.exists():
    raise FileNotFoundError(f"Dataset folder not found: {ROOT}")

if not REF_DIR.exists() or not SEARCH_DIR.exists():
    raise FileNotFoundError(
        "reference/ or search/ folder is missing."
    )

ref_files = sorted(REF_DIR.glob("*.png"))
search_files = sorted(SEARCH_DIR.glob("*.png"))

print(f"Reference images before : {len(ref_files)}")
print(f"Search images before    : {len(search_files)}")

if len(ref_files) != len(search_files):
    raise RuntimeError(
        "Reference and search image counts differ. "
        "No files were renamed."
    )

# ------------------------------------------------------------
# Determine logical pair keys.
# ------------------------------------------------------------
ref_keys = {}
search_keys = {}

for path in ref_files:
    key = common_stem(path.name)

    if key in ref_keys:
        raise RuntimeError(
            f"Duplicate reference logical key: {key}"
        )

    ref_keys[key] = path

for path in search_files:
    key = common_stem(path.name)

    if key in search_keys:
        raise RuntimeError(
            f"Duplicate search logical key: {key}"
        )

    search_keys[key] = path

missing_reference = sorted(
    set(search_keys) - set(ref_keys)
)

missing_search = sorted(
    set(ref_keys) - set(search_keys)
)

if missing_reference or missing_search:
    print("\nPairing check FAILED.")
    print(
        "Missing reference keys:",
        missing_reference[:20]
    )
    print(
        "Missing search keys:",
        missing_search[:20]
    )
    raise RuntimeError(
        "Reference/search logical pairs do not match. "
        "No files were renamed."
    )

print("Logical reference/search pairing : PASSED")

# ------------------------------------------------------------
# Rename safely using temporary names first.
#
# This avoids collisions when:
#
# reference_x.png -> x.png
# search_x.png    -> x.png
#
# because both folders are independent, but the temporary phase
# makes the operation robust anyway.
# ------------------------------------------------------------
ref_temp = {}
search_temp = {}

for index, path in enumerate(ref_files):
    temp = REF_DIR / f"__PAIRING_TMP_REF_{index:05d}.png"
    path.rename(temp)
    ref_temp[index] = temp

for index, path in enumerate(search_files):
    temp = SEARCH_DIR / f"__PAIRING_TMP_SEARCH_{index:05d}.png"
    path.rename(temp)
    search_temp[index] = temp

# Rebuild mappings from the original logical keys.
ref_temp_by_key = {}
search_temp_by_key = {}

for index, path in ref_temp.items():
    original_key = common_stem(ref_files[index].name)
    ref_temp_by_key[original_key] = path

for index, path in search_temp.items():
    original_key = common_stem(search_files[index].name)
    search_temp_by_key[original_key] = path

# Write common filenames.
for key in sorted(ref_temp_by_key):
    final_name = key + ".png"

    ref_temp_by_key[key].rename(
        REF_DIR / final_name
    )

    search_temp_by_key[key].rename(
        SEARCH_DIR / final_name
    )

# ------------------------------------------------------------
# Update metadata.json.
# ------------------------------------------------------------
if METADATA_JSON.exists():
    with open(
        METADATA_JSON,
        "r",
        encoding="utf-8"
    ) as f:
        records = json.load(f)

    for row in records:
        if "reference_file" in row:
            row["reference_file"] = (
                common_stem(
                    row["reference_file"]
                )
                + ".png"
            )

        if "search_file" in row:
            row["search_file"] = (
                common_stem(
                    row["search_file"]
                )
                + ".png"
            )

    with open(
        METADATA_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f,
            indent=4
        )

    print("metadata.json updated.")

# ------------------------------------------------------------
# Update metadata.csv.
# ------------------------------------------------------------
if METADATA_CSV.exists():
    with open(
        METADATA_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if rows:
        fieldnames = list(rows[0].keys())

        for row in rows:
            if "reference_file" in row:
                row["reference_file"] = (
                    common_stem(
                        row["reference_file"]
                    )
                    + ".png"
                )

            if "search_file" in row:
                row["search_file"] = (
                    common_stem(
                        row["search_file"]
                    )
                    + ".png"
                )

        with open(
            METADATA_CSV,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)

    print("metadata.csv updated.")

# ------------------------------------------------------------
# Final verification.
# ------------------------------------------------------------
ref_after = {
    p.stem
    for p in REF_DIR.glob("*.png")
}

search_after = {
    p.stem
    for p in SEARCH_DIR.glob("*.png")
}

print()
print(f"Reference images after  : {len(ref_after)}")
print(f"Search images after     : {len(search_after)}")
print(
    "Exact filename pairing  :",
    ref_after == search_after
)

if len(ref_after) != 2100:
    raise RuntimeError(
        f"Expected 2100 reference images, "
        f"found {len(ref_after)}."
    )

if len(search_after) != 2100:
    raise RuntimeError(
        f"Expected 2100 search images, "
        f"found {len(search_after)}."
    )

if ref_after != search_after:
    raise RuntimeError(
        "Final reference/search filename pairing failed."
    )

print()
print("=" * 72)
print("FILENAME PAIRING REPAIR COMPLETE")
print("=" * 72)
print()
print("No image pixels were changed.")
print("No augmentation was regenerated.")
print("Only filenames and metadata references were normalized.")