
from pathlib import Path
import argparse
import json
import math
import csv
import shutil

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    raise SystemExit(
        "OpenCV is required. Install with: pip install opencv-python"
    )


# ============================================================
# DEFAULT PATHS
# ============================================================

DATASET_ROOT = Path(r"E:\semicon\generator5_augmented")
METADATA_FILE = DATASET_ROOT / "metadata.csv"
OUTPUT_ROOT = DATASET_ROOT / "training_metadata"

GRID_ROWS = 4
GRID_COLS = 4
SEARCH_SIZE = 1000
REFERENCE_SIZE = 1000

NCC_THRESHOLD = 0.97
SPLIT_SEED = 20260817

TRAIN_BASE_PAIRS = 240
VAL_BASE_PAIRS = 30
TEST_BASE_PAIRS = 30


# ============================================================
# HELPERS
# ============================================================

def fnum(value, default=0.0):
    """Safely convert CSV numeric values, including NaN."""
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def transform_point(
    x,
    y,
    width,
    height,
    rotation_deg=0.0,
    shear_deg=0.0,
    scale_factor=1.0,
):
    """
    EXACTLY follows the coordinate transform used by the current
    augment_generator5.py transform_point() function.

    Order used by that generator's GT calculation:
        scale -> shear -> rotation
    """
    cx = width / 2.0
    cy = height / 2.0

    px = x - cx
    py = y - cy

    px *= scale_factor
    py *= scale_factor

    shear = math.tan(math.radians(shear_deg))
    px = px + shear * py

    angle = math.radians(rotation_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rx = cos_a * px - sin_a * py
    ry = sin_a * px + cos_a * py

    return rx + cx, ry + cy


def drift_point(
    x,
    y,
    width,
    height,
    dx,
    dy,
    seed,
    smoothness=0.65,
):
    """
    EXACTLY follows drift_point() in augment_generator5.py.
    """
    rng = np.random.default_rng(int(seed))

    n = 11

    yp = np.linspace(
        0,
        height - 1,
        n,
        dtype=np.float32
    )

    xc = np.cumsum(
        rng.normal(
            0,
            smoothness,
            n
        )
    ).astype(np.float32)

    yc = np.cumsum(
        rng.normal(
            0,
            smoothness * 0.7,
            n
        )
    ).astype(np.float32)

    xc -= xc.mean()
    yc -= yc.mean()

    xd = (
        float(np.interp(y, yp, xc))
        + float(dx) * float(y) / max(height - 1, 1)
    )

    yd = (
        float(np.interp(y, yp, yc))
        + float(dy) * float(y) / max(height - 1, 1)
    )

    return x + xd, y + yd


def nominal_block_center(row, col):
    """Nominal center of one 250x250 block in the 1000x1000 Search."""
    block_w = SEARCH_SIZE / GRID_COLS
    block_h = SEARCH_SIZE / GRID_ROWS

    return (
        (col + 0.5) * block_w,
        (row + 0.5) * block_h,
    )


def transformed_block_center(row, col, row_data):
    """
    Transform a nominal block center using the SAME geometric
    transformation family used for that augmented Search image.
    """
    x, y = nominal_block_center(row, col)

    augmentation = str(row_data["augmentation"])

    if augmentation in {
        "rotation",
        "shear",
        "scaling",
    }:
        rotation = fnum(
            row_data.get("search_rotation_deg"),
            0.0
        )

        shear = fnum(
            row_data.get("search_shear_deg"),
            0.0
        )

        scale = fnum(
            row_data.get("search_scale_factor"),
            1.0
        )

        return transform_point(
            x,
            y,
            SEARCH_SIZE,
            SEARCH_SIZE,
            rotation,
            shear,
            scale,
        )

    if augmentation == "drift":
        dx = fnum(
            row_data.get("search_drift_x_px"),
            0.0
        )

        dy = fnum(
            row_data.get("search_drift_y_px"),
            0.0
        )

        seed = int(
            fnum(
                row_data.get("search_drift_seed"),
                0
            )
        )

        return drift_point(
            x,
            y,
            SEARCH_SIZE,
            SEARCH_SIZE,
            dx,
            dy,
            seed,
            0.65,
        )

    return x, y


def crop_center(
    image,
    cx,
    cy,
    width,
    height,
):
    """
    Crop a fixed-size candidate centered on (cx,cy).
    Returns None when the complete candidate cannot fit.
    """
    h, w = image.shape[:2]

    width = max(8, int(round(width)))
    height = max(8, int(round(height)))

    left = int(round(cx - width / 2.0))
    top = int(round(cy - height / 2.0))

    right = left + width
    bottom = top + height

    if left < 0 or top < 0:
        return None

    if right > w or bottom > h:
        return None

    return image[top:bottom, left:right]


def resize_reference_for_matching(reference):
    """
    The physical Reference is 1000x1000 while its footprint in the
    final Search is about 100x100. Downsample the Reference by 10x.
    """
    return cv2.resize(
        reference,
        (REFERENCE_SIZE // 10, REFERENCE_SIZE // 10),
        interpolation=cv2.INTER_AREA,
    )


def candidate_block_coordinates(row_data):
    """
    Return all 16 transformed block centers.
    """
    result = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cx, cy = transformed_block_center(
                row,
                col,
                row_data
            )

            result.append({
                "row": row,
                "col": col,
                "center_x": round(float(cx), 4),
                "center_y": round(float(cy), 4),
            })

    return result


def target_block_string(row_data):
    return (
        f"{int(row_data['selected_block_row'])}_"
        f"{int(row_data['selected_block_col'])}"
    )


# ============================================================
# NCC AMBIGUITY AUDIT
# ============================================================

def run_ncc_audit(
    row_data,
    reference_path,
    search_path,
    threshold=NCC_THRESHOLD,
):
    """
    Compare the 10x-downsampled Reference against:

        1. the actual target crop
        2. the 15 other block-centered candidate crops

    The 15 other blocks are the hard-negative candidates.

    A sample is marked ambiguous when any WRONG block obtains
    NCC > threshold.

    No images are modified.
    """
    reference = cv2.imread(
        str(reference_path),
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        str(search_path),
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise RuntimeError(
            f"Could not read reference image:\n{reference_path}"
        )

    if search is None:
        raise RuntimeError(
            f"Could not read search image:\n{search_path}"
        )

    reference_small = resize_reference_for_matching(
        reference
    )

    target_w = fnum(
        row_data["target_width"],
        100.0
    )

    target_h = fnum(
        row_data["target_height"],
        100.0
    )

    target_cx = fnum(
        row_data["center_x"],
        500.0
    )

    target_cy = fnum(
        row_data["center_y"],
        500.0
    )

    target_crop = crop_center(
        search,
        target_cx,
        target_cy,
        target_w,
        target_h,
    )

    if target_crop is None:
        return {
            "ambiguous": False,
            "audit_status": "target_crop_out_of_bounds",
            "true_ncc": None,
            "max_wrong_ncc": None,
            "best_wrong_block": "",
            "hard_negative_nccs": {},
        }

    target_crop = cv2.resize(
        target_crop,
        (
            reference_small.shape[1],
            reference_small.shape[0],
        ),
        interpolation=cv2.INTER_AREA,
    )

    true_result = cv2.matchTemplate(
        target_crop,
        reference_small,
        cv2.TM_CCOEFF_NORMED
    )

    true_ncc = float(true_result[0, 0])

    selected_row = int(
        row_data["selected_block_row"]
    )

    selected_col = int(
        row_data["selected_block_col"]
    )

    candidates = candidate_block_coordinates(
        row_data
    )

    hard_negative_nccs = {}

    best_wrong_ncc = -1.0
    best_wrong_block = ""

    for candidate in candidates:
        row = candidate["row"]
        col = candidate["col"]

        if row == selected_row and col == selected_col:
            continue

        crop = crop_center(
            search,
            candidate["center_x"],
            candidate["center_y"],
            target_w,
            target_h,
        )

        if crop is None:
            # A candidate outside the valid Search field cannot be
            # considered a valid matching candidate.
            continue

        crop = cv2.resize(
            crop,
            (
                reference_small.shape[1],
                reference_small.shape[0],
            ),
            interpolation=cv2.INTER_AREA,
        )

        result = cv2.matchTemplate(
            crop,
            reference_small,
            cv2.TM_CCOEFF_NORMED
        )

        ncc = float(result[0, 0])

        key = f"{row}_{col}"
        hard_negative_nccs[key] = round(ncc, 6)

        if ncc > best_wrong_ncc:
            best_wrong_ncc = ncc
            best_wrong_block = key

    ambiguous = (
        best_wrong_ncc >= threshold
    )

    return {
        "ambiguous": bool(ambiguous),
        "audit_status": "ok",
        "true_ncc": round(true_ncc, 6),
        "max_wrong_ncc": (
            round(best_wrong_ncc, 6)
            if best_wrong_ncc >= 0
            else None
        ),
        "best_wrong_block": best_wrong_block,
        "hard_negative_nccs": hard_negative_nccs,
    }


# ============================================================
# SPLIT CREATION
# ============================================================

def create_base_scene_split(
    base_ids,
    seed=SPLIT_SEED,
):
    """
    Split at BASE-SCENE level so all 7 augmentations remain together.
    """
    base_ids = sorted(
        int(x)
        for x in base_ids
    )

    expected = (
        TRAIN_BASE_PAIRS
        + VAL_BASE_PAIRS
        + TEST_BASE_PAIRS
    )

    if len(base_ids) != expected:
        raise RuntimeError(
            f"Expected {expected} base scenes, "
            f"found {len(base_ids)}."
        )

    rng = np.random.default_rng(
        seed
    )

    shuffled = np.array(
        base_ids,
        dtype=np.int64
    )

    rng.shuffle(shuffled)

    train_ids = set(
        int(x)
        for x in shuffled[
            :TRAIN_BASE_PAIRS
        ]
    )

    val_ids = set(
        int(x)
        for x in shuffled[
            TRAIN_BASE_PAIRS:
            TRAIN_BASE_PAIRS + VAL_BASE_PAIRS
        ]
    )

    test_ids = set(
        int(x)
        for x in shuffled[
            TRAIN_BASE_PAIRS + VAL_BASE_PAIRS:
        ]
    )

    return (
        train_ids,
        val_ids,
        test_ids,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare enriched training metadata for the validated "
            "Generator5 augmented dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(DATASET_ROOT)
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=NCC_THRESHOLD
    )

    parser.add_argument(
        "--skip-ncc",
        action="store_true",
        help="Create coordinates/splits but skip the expensive NCC audit."
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.dataset
    )

    metadata_path = (
        dataset_root /
        "metadata.csv"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found:\n{metadata_path}"
        )

    reference_dir = (
        dataset_root /
        "reference"
    )

    search_dir = (
        dataset_root /
        "search"
    )

    if not reference_dir.exists():
        raise FileNotFoundError(
            f"Reference directory not found:\n{reference_dir}"
        )

    if not search_dir.exists():
        raise FileNotFoundError(
            f"Search directory not found:\n{search_dir}"
        )

    output_root = (
        dataset_root /
        "training_metadata"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 76)
    print("GENERATOR 5 TRAINING METADATA PREPARATION")
    print("=" * 76)
    print()
    print(f"Dataset : {dataset_root}")
    print(f"Metadata: {metadata_path}")
    print()
    print("No images will be modified.")
    print("No images will be regenerated.")
    print()
    print(f"NCC ambiguity threshold: {args.threshold}")
    print()

    df = pd.read_csv(
        metadata_path
    )

    print(
        f"Metadata rows: {len(df)}"
    )

    required = [
        "original_pair_id",
        "augmentation",
        "reference_file",
        "search_file",
        "center_x",
        "center_y",
        "target_width",
        "target_height",
        "selected_block_row",
        "selected_block_col",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Required metadata columns are missing:\n"
            + "\n".join(
                f"  {x}"
                for x in missing
            )
        )

    base_ids = sorted(
        int(x)
        for x in df["original_pair_id"].unique()
    )

    print(
        f"Base scenes: {len(base_ids)}"
    )

    if len(base_ids) != 300:
        raise RuntimeError(
            f"Expected 300 base scenes, "
            f"found {len(base_ids)}."
        )

    train_ids, val_ids, test_ids = (
        create_base_scene_split(
            base_ids
        )
    )

    def assign_split(pair_id):
        pair_id = int(pair_id)

        if pair_id in train_ids:
            return "train"

        if pair_id in val_ids:
            return "validation"

        if pair_id in test_ids:
            return "test"

        raise RuntimeError(
            f"Pair {pair_id} is not assigned to a split."
        )

    df["split"] = [
        assign_split(x)
        for x in df["original_pair_id"]
    ]

    # --------------------------------------------------------
    # Verify split integrity before expensive NCC work.
    # --------------------------------------------------------
    split_counts = (
        df.groupby("split")
        .size()
        .to_dict()
    )

    print()
    print("Split:")
    print(
        f"  train      : "
        f"{split_counts.get('train', 0)} augmented pairs"
    )
    print(
        f"  validation : "
        f"{split_counts.get('validation', 0)} augmented pairs"
    )
    print(
        f"  test       : "
        f"{split_counts.get('test', 0)} augmented pairs"
    )

    for split_name, ids in (
        ("train", train_ids),
        ("validation", val_ids),
        ("test", test_ids),
    ):
        observed = set(
            int(x)
            for x in df.loc[
                df["split"] == split_name,
                "original_pair_id"
            ].unique()
        )

        if observed != ids:
            raise RuntimeError(
                f"Base-scene leakage/integrity failure in {split_name}."
            )

    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    ):
        raise RuntimeError(
            "Base-scene split overlap detected."
        )

    # --------------------------------------------------------
    # Add all 16 transformed block coordinates.
    # --------------------------------------------------------
    print()
    print("Generating transformed 4x4 block coordinates...")

    all_coordinates = []
    target_blocks = []
    periodic_groups = []

    for _, row in df.iterrows():
        coords = candidate_block_coordinates(
            row
        )

        all_coordinates.append(
            json.dumps(
                coords,
                separators=(",", ":")
            )
        )

        target_blocks.append(
            target_block_string(row)
        )

        # Initial periodic group:
        # all 16 blocks belonging to the same generated base scene
        # are the competing periodic locations for this sample.
        periodic_groups.append(
            f"pair_{int(row['original_pair_id']):04d}"
        )

    df["all_block_coordinates"] = (
        all_coordinates
    )

    df["target_block"] = (
        target_blocks
    )

    df["periodic_group"] = (
        periodic_groups
    )

    # --------------------------------------------------------
    # NCC ambiguity audit.
    # --------------------------------------------------------
    if args.skip_ncc:
        print()
        print("NCC audit skipped by --skip-ncc.")
        df["ambiguous"] = False
        df["ambiguity_audit_status"] = (
            "not_run"
        )
        df["true_ncc"] = np.nan
        df["max_wrong_ncc"] = np.nan
        df["best_wrong_block"] = ""
        df["hard_negative_nccs"] = "{}"

    else:
        print()
        print("=" * 76)
        print("NCC AMBIGUITY AUDIT")
        print("=" * 76)
        print(
            "Each Reference is downsampled 10x."
        )
        print(
            "Each target is compared against the 15 OTHER blocks."
        )
        print(
            f"Ambiguous if max wrong-block NCC >= "
            f"{args.threshold:.3f}"
        )
        print()

        ambiguous_values = []
        audit_status_values = []
        true_ncc_values = []
        max_wrong_values = []
        best_wrong_values = []
        hard_negative_values = []

        for index, (_, row) in enumerate(
            df.iterrows(),
            1
        ):
            reference_path = (
                reference_dir /
                Path(
                    str(row["reference_file"])
                ).name
            )

            search_path = (
                search_dir /
                Path(
                    str(row["search_file"])
                ).name
            )

            result = run_ncc_audit(
                row,
                reference_path,
                search_path,
                args.threshold
            )

            ambiguous_values.append(
                result["ambiguous"]
            )

            audit_status_values.append(
                result["audit_status"]
            )

            true_ncc_values.append(
                result["true_ncc"]
            )

            max_wrong_values.append(
                result["max_wrong_ncc"]
            )

            best_wrong_values.append(
                result["best_wrong_block"]
            )

            hard_negative_values.append(
                json.dumps(
                    result["hard_negative_nccs"],
                    separators=(",", ":")
                )
            )

            if (
                index == 1
                or index % 100 == 0
                or index == len(df)
            ):
                print(
                    f"  [{index:04d}/{len(df):04d}]"
                )

        df["ambiguous"] = (
            ambiguous_values
        )

        df["ambiguity_audit_status"] = (
            audit_status_values
        )

        df["true_ncc"] = (
            true_ncc_values
        )

        df["max_wrong_ncc"] = (
            max_wrong_values
        )

        df["best_wrong_block"] = (
            best_wrong_values
        )

        df["hard_negative_nccs"] = (
            hard_negative_values
        )

    # --------------------------------------------------------
    # Save enriched full metadata.
    # --------------------------------------------------------
    enriched_path = (
        output_root /
        "metadata_enriched.csv"
    )

    df.to_csv(
        enriched_path,
        index=False
    )

    enriched_json_path = (
        output_root /
        "metadata_enriched.json"
    )

    records = df.to_dict(
        orient="records"
    )

    with open(
        enriched_json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f,
            indent=2,
            allow_nan=True
        )

    # --------------------------------------------------------
    # Save split-specific metadata.
    # No image copying.
    # --------------------------------------------------------
    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_df = df[
            df["split"] == split_name
        ].copy()

        split_df.to_csv(
            output_root /
            f"{split_name}.csv",
            index=False
        )

    # --------------------------------------------------------
    # Save the base-scene split itself.
    # --------------------------------------------------------
    split_rows = []

    for pair_id in sorted(train_ids):
        split_rows.append({
            "original_pair_id": pair_id,
            "split": "train",
        })

    for pair_id in sorted(val_ids):
        split_rows.append({
            "original_pair_id": pair_id,
            "split": "validation",
        })

    for pair_id in sorted(test_ids):
        split_rows.append({
            "original_pair_id": pair_id,
            "split": "test",
        })

    pd.DataFrame(
        split_rows
    ).to_csv(
        output_root /
        "base_scene_split.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------
    print()
    print("=" * 76)
    print("TRAINING METADATA PREPARATION COMPLETE")
    print("=" * 76)

    print()
    print("Base scenes:")
    print(
        f"  train      : {len(train_ids)}"
    )
    print(
        f"  validation : {len(val_ids)}"
    )
    print(
        f"  test       : {len(test_ids)}"
    )

    print()
    print("Augmented pairs:")
    print(
        f"  train      : "
        f"{len(df[df['split'] == 'train'])}"
    )
    print(
        f"  validation : "
        f"{len(df[df['split'] == 'validation'])}"
    )
    print(
        f"  test       : "
        f"{len(df[df['split'] == 'test'])}"
    )

    if not args.skip_ncc:
        ambiguous_count = int(
            df["ambiguous"].sum()
        )

        print()
        print("NCC ambiguity:")
        print(
            f"  ambiguous : "
            f"{ambiguous_count} / {len(df)} "
            f"({100.0 * ambiguous_count / len(df):.2f}%)"
        )

        print(
            f"  threshold : {args.threshold:.3f}"
        )

        valid_ncc = df[
            df["ambiguity_audit_status"] == "ok"
        ]

        if len(valid_ncc):
            print(
                f"  mean true NCC : "
                f"{valid_ncc['true_ncc'].mean():.4f}"
            )

            print(
                f"  mean max wrong NCC : "
                f"{valid_ncc['max_wrong_ncc'].mean():.4f}"
            )

    print()
    print("Output:")
    print(
        f"  {enriched_path}"
    )
    print(
        f"  {output_root / 'train.csv'}"
    )
    print(
        f"  {output_root / 'validation.csv'}"
    )
    print(
        f"  {output_root / 'test.csv'}"
    )
    print(
        f"  {output_root / 'base_scene_split.csv'}"
    )

    print()
    print("No image files were changed.")
    print("No image files were copied.")
    print("No new images were generated.")
    print()
    print("=" * 76)


if __name__ == "__main__":
    main()
