import csv
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

ROOT = Path(r"E:\semicon\generator5_augmented")
REF_DIR = ROOT / "reference"
SEARCH_DIR = ROOT / "search"
META = ROOT / "metadata.csv"

EXPECTED_VARIANTS = {
    "rotation",
    "shear",
    "drift",
    "noise",
    "blur",
    "brightness_contrast",
    "scaling",
}

print("=" * 75)
print("GENERATOR 5 AUGMENTED DATASET VALIDATION")
print("=" * 75)

# ------------------------------------------------------------
# 1. File counts
# ------------------------------------------------------------
ref_files = sorted(REF_DIR.glob("*.png"))
search_files = sorted(SEARCH_DIR.glob("*.png"))

print(f"\nReference images : {len(ref_files)}")
print(f"Search images    : {len(search_files)}")

errors = []
warnings = []

if len(ref_files) != len(search_files):
    errors.append(
        f"Reference/search count mismatch: "
        f"{len(ref_files)} vs {len(search_files)}"
    )

if len(ref_files) != 2100:
    warnings.append(
        f"Expected 2100 images for 300 x 7; found {len(ref_files)}."
    )

# ------------------------------------------------------------
# 2. Metadata
# ------------------------------------------------------------
if not META.exists():
    errors.append(f"Missing metadata file: {META}")
    rows = []
else:
    with META.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

print(f"Metadata rows    : {len(rows)}")

if rows and len(rows) != len(ref_files):
    errors.append(
        f"Metadata/image mismatch: {len(rows)} metadata rows "
        f"for {len(ref_files)} image pairs."
    )

# ------------------------------------------------------------
# 3. Detect useful column names
# ------------------------------------------------------------
def find_col(names):
    for name in names:
        if name in rows[0]:
            return name
    return None

if rows:
    pair_col = find_col([
        "original_pair_id",
        "pair_id",
        "base_pair_id",
    ])

    variant_col = find_col([
        "augmentation",
        "augmentation_type",
        "variant",
    ])

    block_row_col = find_col([
        "selected_block_row",
        "block_row",
    ])

    block_col_col = find_col([
        "selected_block_col",
        "block_col",
    ])

    target_type_col = find_col([
        "target_selection",
        "target_type",
        "target_mode",
    ])

    cx_col = find_col([
        "center_x",
        "target_center_x",
    ])

    cy_col = find_col([
        "center_y",
        "target_center_y",
    ])

    print("\nDetected metadata columns:")
    print(f"  pair       : {pair_col}")
    print(f"  variant    : {variant_col}")
    print(f"  block row  : {block_row_col}")
    print(f"  block col  : {block_col_col}")
    print(f"  target     : {target_type_col}")
    print(f"  center x   : {cx_col}")
    print(f"  center y   : {cy_col}")

    # --------------------------------------------------------
    # 4. Variant distribution
    # --------------------------------------------------------
    if variant_col:
        variant_counts = Counter(
            row[variant_col]
            for row in rows
        )

        print("\nAugmentation distribution:")
        for name, count in sorted(
            variant_counts.items()
        ):
            print(f"  {name:<24} {count}")

        missing = EXPECTED_VARIANTS - set(
            variant_counts
        )

        if missing:
            errors.append(
                "Missing augmentation types: "
                + ", ".join(sorted(missing))
            )

    # --------------------------------------------------------
    # 5. Seven unique target blocks per base pair
    # --------------------------------------------------------
    if (
        pair_col
        and block_row_col
        and block_col_col
    ):
        pair_blocks = defaultdict(list)

        for row in rows:
            try:
                pair_id = row[pair_col]
                block = (
                    int(row[block_row_col]),
                    int(row[block_col_col])
                )
                pair_blocks[pair_id].append(
                    block
                )
            except (ValueError, TypeError):
                errors.append(
                    "Invalid block coordinate in metadata."
                )

        duplicate_pairs = []

        for pair_id, blocks in pair_blocks.items():
            if len(blocks) != len(set(blocks)):
                duplicate_pairs.append(
                    pair_id
                )

        print(
            f"\nBase pairs checked : {len(pair_blocks)}"
        )

        if duplicate_pairs:
            errors.append(
                f"{len(duplicate_pairs)} base pairs contain "
                "duplicate target blocks."
            )
            print(
                "  UNIQUE BLOCK TEST : FAILED"
            )
            print(
                "  First duplicate pairs:",
                duplicate_pairs[:10]
            )
        else:
            print(
                "  UNIQUE BLOCK TEST : PASSED"
            )

        bad_variant_counts = [
            pair_id
            for pair_id, blocks
            in pair_blocks.items()
            if len(blocks) != 7
        ]

        if bad_variant_counts:
            errors.append(
                f"{len(bad_variant_counts)} base pairs do not "
                "contain exactly 7 variants."
            )
        else:
            print(
                "  7 VARIANTS/PAIR TEST : PASSED"
            )

    # --------------------------------------------------------
    # 6. Target type distribution
    # --------------------------------------------------------
    if target_type_col:
        target_counts = Counter(
            row[target_type_col]
            for row in rows
        )

        print("\nTarget distribution:")
        for name, count in sorted(
            target_counts.items()
        ):
            print(f"  {name:<24} {count}")

    # --------------------------------------------------------
    # 7. Ground-truth coordinate validation
    # --------------------------------------------------------
    if cx_col and cy_col:
        bad_gt = 0

        for row in rows:
            try:
                x = float(row[cx_col])
                y = float(row[cy_col])

                if not (
                    0 <= x <= 1000
                    and 0 <= y <= 1000
                ):
                    bad_gt += 1

            except (ValueError, TypeError):
                bad_gt += 1

        print(
            f"\nGT coordinate test : "
            f"{'PASSED' if bad_gt == 0 else 'FAILED'}"
        )

        if bad_gt:
            errors.append(
                f"{bad_gt} rows have invalid GT coordinates."
            )

# ------------------------------------------------------------
# 8. Actual image dimensions
# ------------------------------------------------------------
print("\nChecking image dimensions...")

bad_reference_size = []
bad_search_size = []

for path in ref_files:
    try:
        with Image.open(path) as im:
            if im.size != (1000, 1000):
                bad_reference_size.append(
                    (path.name, im.size)
                )
    except Exception as e:
        errors.append(
            f"Cannot open reference {path.name}: {e}"
        )

for path in search_files:
    try:
        with Image.open(path) as im:
            if im.size != (1000, 1000):
                bad_search_size.append(
                    (path.name, im.size)
                )
    except Exception as e:
        errors.append(
            f"Cannot open search {path.name}: {e}"
        )

print(
    "  Reference 1000x1000 : "
    + ("PASSED" if not bad_reference_size else "FAILED")
)

print(
    "  Search 1000x1000    : "
    + ("PASSED" if not bad_search_size else "FAILED")
)

if bad_reference_size:
    errors.append(
        f"{len(bad_reference_size)} reference images "
        "have incorrect dimensions."
    )

if bad_search_size:
    errors.append(
        f"{len(bad_search_size)} search images "
        "have incorrect dimensions."
    )

# ------------------------------------------------------------
# 9. Filename pairing
# ------------------------------------------------------------
ref_names = {
    p.stem
    for p in ref_files
}

search_names = {
    p.stem
    for p in search_files
}

if ref_names != search_names:
    errors.append(
        "Reference/search filenames are not one-to-one."
    )
else:
    print(
        "  Reference/search pairing : PASSED"
    )

# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------
print("\n" + "=" * 75)

if errors:
    print("DATASET VALIDATION : FAILED")
    print("\nErrors:")
    for error in errors:
        print("  [ERROR]", error)
else:
    print("DATASET VALIDATION : PASSED")

if warnings:
    print("\nWarnings:")
    for warning in warnings:
        print("  [WARNING]", warning)

print("=" * 75)

if errors:
    raise SystemExit(1)