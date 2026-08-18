from pathlib import Path
import argparse
import json
import math

import cv2
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(r"E:\semicon\generator5_augmented")
DEFAULT_METADATA = (
    DEFAULT_ROOT / "training_metadata" / "test.csv"
)
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "training_metadata" / "ncc_baseline"
)

TEMPLATE_SCALE = 10.0
DEFAULT_TOP_K = 3
DISTANCE_THRESHOLD_PX = 5.0


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def load_gray(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )
    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{path}"
        )
    return image


def downsample_reference(reference):
    """
    Physical Reference is 1000x1000 while its footprint in
    the final Search is approximately 100x100.
    """
    return cv2.resize(
        reference,
        (
            reference.shape[1] // 10,
            reference.shape[0] // 10
        ),
        interpolation=cv2.INTER_AREA
    )


def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2
        + (y1 - y2) ** 2
    )


def find_top_peaks(result, top_k=3, min_distance=20):
    """
    Extract top-k spatially separated peaks from the NCC map.
    """
    work = result.copy()
    peaks = []

    h, w = work.shape

    for _ in range(top_k):
        _, max_value, _, max_location = cv2.minMaxLoc(
            work
        )

        if max_value <= -1:
            break

        x, y = max_location

        peaks.append({
            "x": int(x),
            "y": int(y),
            "score": float(max_value),
        })

        # Suppress a circular neighborhood around this peak.
        yy, xx = np.ogrid[:h, :w]
        mask = (
            (xx - x) ** 2
            + (yy - y) ** 2
            <= min_distance ** 2
        )
        work[mask] = -1.0

    return peaks


def evaluate_sample(
    row,
    reference_dir,
    search_dir,
    top_k=3,
):
    reference_file = Path(
        str(row["reference_file"])
    ).name

    search_file = Path(
        str(row["search_file"])
    ).name

    reference_path = (
        reference_dir / reference_file
    )

    search_path = (
        search_dir / search_file
    )

    reference = load_gray(
        reference_path
    )

    search = load_gray(
        search_path
    )

    template = downsample_reference(
        reference
    )

    if (
        template.shape[0] > search.shape[0]
        or template.shape[1] > search.shape[1]
    ):
        raise RuntimeError(
            "Template is larger than Search:\n"
            f"template={template.shape}\n"
            f"search={search.shape}\n"
            f"reference={reference_path}\n"
            f"search={search_path}"
        )

    result = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    peaks = find_top_peaks(
        result,
        top_k=top_k,
        min_distance=max(
            template.shape[0],
            template.shape[1]
        ) // 2
    )

    # matchTemplate gives the top-left coordinate.
    # Convert it to template center.
    half_w = template.shape[1] / 2.0
    half_h = template.shape[0] / 2.0

    for peak in peaks:
        peak["center_x"] = (
            peak["x"] + half_w
        )
        peak["center_y"] = (
            peak["y"] + half_h
        )

    gt_x = safe_float(
        row["center_x"]
    )
    gt_y = safe_float(
        row["center_y"]
    )

    for rank, peak in enumerate(
        peaks,
        start=1
    ):
        peak["rank"] = rank
        peak["distance_to_gt"] = distance(
            peak["center_x"],
            peak["center_y"],
            gt_x,
            gt_y
        )

    top1 = (
        peaks[0]
        if len(peaks) >= 1
        else None
    )

    top3_correct = any(
        p["distance_to_gt"]
        <= DISTANCE_THRESHOLD_PX
        for p in peaks
    )

    top1_correct = (
        top1 is not None
        and top1["distance_to_gt"]
        <= DISTANCE_THRESHOLD_PX
    )

    wrong_periodic = (
        top1 is not None
        and top1["distance_to_gt"]
        > DISTANCE_THRESHOLD_PX
    )

    result_row = {
        "original_pair_id": row["original_pair_id"],
        "augmentation": row["augmentation"],
        "reference_file": reference_file,
        "search_file": search_file,
        "target_type": (
            row["target_type"]
            if "target_type" in row
            else ""
        ),
        "selected_block_row": row[
            "selected_block_row"
        ],
        "selected_block_col": row[
            "selected_block_col"
        ],
        "gt_x": gt_x,
        "gt_y": gt_y,
        "template_width": template.shape[1],
        "template_height": template.shape[0],
        "top1_x": (
            top1["center_x"]
            if top1 else np.nan
        ),
        "top1_y": (
            top1["center_y"]
            if top1 else np.nan
        ),
        "top1_score": (
            top1["score"]
            if top1 else np.nan
        ),
        "top1_error_px": (
            top1["distance_to_gt"]
            if top1 else np.nan
        ),
        "top1_correct": bool(
            top1_correct
        ),
        "top3_correct": bool(
            top3_correct
        ),
        "wrong_periodic_match": bool(
            wrong_periodic
        ),
        "top_peaks_json": json.dumps(
            peaks,
            separators=(",", ":")
        ),
    }

    return result_row, result, peaks


def print_metrics(results):
    df = pd.DataFrame(results)

    total = len(df)

    if total == 0:
        print("No samples were evaluated.")
        return

    valid = df[
        df["top1_error_px"].notna()
    ].copy()

    print()
    print("=" * 76)
    print("CLASSICAL NCC BASELINE RESULTS")
    print("=" * 76)

    print()
    print(f"Samples evaluated : {len(valid)}")

    if len(valid):
        errors = valid[
            "top1_error_px"
        ].to_numpy()

        print(
            f"Mean pixel error   : "
            f"{np.mean(errors):.3f}"
        )

        print(
            f"Median pixel error : "
            f"{np.median(errors):.3f}"
        )

        mae_x = np.mean(
            np.abs(
            valid["top1_x"].to_numpy()
            - valid["gt_x"].to_numpy()
            )
        )

        mae_y = np.mean(
            np.abs(
                valid["top1_y"].to_numpy()
                - valid["gt_y"].to_numpy()
            )
        )

        print(
        f"MAE-x              : "
        f"{mae_x:.3f}"
        )
        
        print(
        f"MAE-y              : "
        f"{mae_y:.3f}"
        )
        for threshold in [1, 2, 5, 10]:
            percentage = (
                100.0
                * np.mean(
                    errors <= threshold
                )
            )

            print(
                f"% within {threshold:2d} px      : "
                f"{percentage:.2f}%"
            )

        top1 = (
            100.0
            * valid["top1_correct"].mean()
        )

        top3 = (
            100.0
            * valid["top3_correct"].mean()
        )

        wrong = (
            100.0
            * valid[
                "wrong_periodic_match"
            ].mean()
        )

        print()
        print(
            f"Top-1 accuracy      : "
            f"{top1:.2f}%"
        )

        print(
            f"Top-3 accuracy      : "
            f"{top3:.2f}%"
        )

        print(
            f"Wrong-periodic rate : "
            f"{wrong:.2f}%"
        )

    # --------------------------------------------------------
    # Target-type breakdown
    # --------------------------------------------------------
    if "target_type" in valid.columns:
        print()
        print("-" * 76)
        print("TARGET-TYPE BREAKDOWN")
        print("-" * 76)

        for target_type, group in (
            valid.groupby("target_type")
        ):
            if len(group) == 0:
                continue

            print()
            print(
                f"{target_type}: "
                f"{len(group)} samples"
            )

            print(
                f"  mean error      : "
                f"{group['top1_error_px'].mean():.3f}"
            )

            print(
                f"  median error    : "
                f"{group['top1_error_px'].median():.3f}"
            )

            print(
                f"  top-1           : "
                f"{100.0 * group['top1_correct'].mean():.2f}%"
            )

            print(
                f"  top-3           : "
                f"{100.0 * group['top3_correct'].mean():.2f}%"
            )

            print(
                f"  wrong-periodic  : "
                f"{100.0 * group['wrong_periodic_match'].mean():.2f}%"
            )

    # --------------------------------------------------------
    # Augmentation breakdown
    # --------------------------------------------------------
    print()
    print("-" * 76)
    print("AUGMENTATION BREAKDOWN")
    print("-" * 76)

    for augmentation, group in (
        valid.groupby("augmentation")
    ):
        print()
        print(
            f"{augmentation}: "
            f"{len(group)} samples"
        )

        print(
            f"  mean error      : "
            f"{group['top1_error_px'].mean():.3f}"
        )

        print(
            f"  top-1           : "
            f"{100.0 * group['top1_correct'].mean():.2f}%"
        )

        print(
            f"  top-3           : "
            f"{100.0 * group['top3_correct'].mean():.2f}%"
        )

        print(
            f"  wrong-periodic  : "
            f"{100.0 * group['wrong_periodic_match'].mean():.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classical full-search NCC baseline for "
            "Generator5 augmented test set."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_ROOT)
    )

    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA)
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT)
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.dataset
    )

    metadata_path = Path(
        args.metadata
    )

    output_root = Path(
        args.output
    )

    reference_dir = (
        dataset_root / "reference"
    )

    search_dir = (
        dataset_root / "search"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{metadata_path}"
        )

    df = pd.read_csv(
        metadata_path
    )

    required = [
        "reference_file",
        "search_file",
        "center_x",
        "center_y",
        "original_pair_id",
        "augmentation",
        "selected_block_row",
        "selected_block_col",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing metadata columns:\n"
            + "\n".join(
                f"  {x}"
                for x in missing
            )
        )

    print("=" * 76)
    print("GENERATOR 5 CLASSICAL NCC BASELINE")
    print("=" * 76)
    print()
    print(
        f"Metadata : {metadata_path}"
    )
    print(
        f"Samples  : {len(df)}"
    )
    print()
    print(
        "Reference: 1000x1000 -> 100x100"
    )
    print(
        "Search   : 1000x1000"
    )
    print(
        "Method   : TM_CCOEFF_NORMED"
    )
    print(
        "Search   : FULL IMAGE"
    )
    print(
        f"Top-K    : {args.top_k}"
    )
    print(
        f"Correct  : <= {DISTANCE_THRESHOLD_PX:.1f} px"
    )
    print()

    results = []

    # Save only a small number of diagnostic heatmaps
    # initially, so the output directory does not become huge.
    diagnostic_limit = 30
    diagnostic_count = 0

    for index, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):
        result_row, match_map, peaks = (
            evaluate_sample(
                row,
                reference_dir,
                search_dir,
                top_k=args.top_k
            )
        )

        results.append(
            result_row
        )

        # Save heatmaps for wrong predictions and the
        # first few samples.
        should_save = (
            diagnostic_count
            < diagnostic_limit
            and (
                index <= 10
                or result_row[
                    "wrong_periodic_match"
                ]
            )
        )

        if should_save:
            pair_id = int(
                result_row[
                    "original_pair_id"
                ]
            )

            augmentation = str(
                result_row[
                    "augmentation"
                ]
            ).replace(
                " ",
                "_"
            )

            heatmap_path = (
                output_root
                / f"pair_{pair_id:04d}_"
                  f"{augmentation}_"
                  f"{index:04d}_ncc.png"
            )

            normalized = cv2.normalize(
                match_map,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            normalized = normalized.astype(
                np.uint8
            )

            cv2.imwrite(
                str(heatmap_path),
                normalized
            )

            diagnostic_count += 1

        if (
            index == 1
            or index % 25 == 0
            or index == len(df)
        ):
            print(
                f"[{index:03d}/{len(df):03d}]"
            )

    results_df = pd.DataFrame(
        results
    )

    results_csv = (
        output_root /
        "ncc_results.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save compact JSON summary
    # --------------------------------------------------------
    valid = results_df[
        results_df[
            "top1_error_px"
        ].notna()
    ]

    summary = {
        "samples": int(len(valid)),
        "method": "TM_CCOEFF_NORMED",
        "reference_downsample_factor": 10,
        "search_size": [1000, 1000],
        "template_size": [100, 100],
        "correct_threshold_px": DISTANCE_THRESHOLD_PX,
        "mean_pixel_error": (
            float(
                valid[
                    "top1_error_px"
                ].mean()
            )
            if len(valid)
            else None
        ),
        "median_pixel_error": (
            float(
                valid[
                    "top1_error_px"
                ].median()
            )
            if len(valid)
            else None
        ),
        "top1_accuracy": (
            float(
                valid[
                    "top1_correct"
                ].mean()
                * 100.0
            )
            if len(valid)
            else None
        ),
        "top3_accuracy": (
            float(
                valid[
                    "top3_correct"
                ].mean()
                * 100.0
            )
            if len(valid)
            else None
        ),
        "wrong_periodic_match_rate": (
            float(
                valid[
                    "wrong_periodic_match"
                ].mean()
                * 100.0
            )
            if len(valid)
            else None
        ),
    }

    with open(
        output_root / "ncc_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print_metrics(
        results
    )

    print()
    print("=" * 76)
    print("NCC BASELINE COMPLETE")
    print("=" * 76)
    print()
    print(
        f"Results : {results_csv}"
    )
    print(
        f"Summary : "
        f"{output_root / 'ncc_summary.json'}"
    )
    print(
        f"Heatmaps saved: "
        f"{diagnostic_count}"
    )
    print()
    print(
        "No dataset images were modified."
    )


if __name__ == "__main__":
    main()