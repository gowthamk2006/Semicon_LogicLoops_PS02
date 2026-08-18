"""
SEMICON / DRIFT-SENSE - Standalone Localization Inference

Usage:
    python localize.py --reference reference.png --search search.png

This standalone fallback uses the same classical NCC idea as the
Generator 5 baseline: the reference is resized to 100x100 pixels and
matched against the full search image using TM_CCOEFF_NORMED.

It requires only OpenCV and NumPy. No CSV, metadata, dataset, or
trained model weights are required.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REFERENCE_SIZE = 100
METHOD = cv2.TM_CCOEFF_NORMED


def read_gray(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found:\n{path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image:\n{path}")
    if image.size == 0:
        raise RuntimeError(f"Image is empty:\n{path}")
    return image


def run_ncc(reference, search):
    template = cv2.resize(
        reference,
        (REFERENCE_SIZE, REFERENCE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    if (template.shape[0] > search.shape[0] or
            template.shape[1] > search.shape[1]):
        raise ValueError(
            "Reference template is larger than the search image."
        )

    response = cv2.matchTemplate(
        search,
        template,
        METHOD,
    )

    _, max_score, _, max_location = cv2.minMaxLoc(response)

    x = int(max_location[0])
    y = int(max_location[1])

    return response, float(max_score), x, y


def top_k_peaks(response, k):
    work = response.copy()
    results = []

    half = REFERENCE_SIZE // 2

    for _ in range(k):
        _, score, _, location = cv2.minMaxLoc(work)

        x = int(location[0])
        y = int(location[1])

        results.append({
            "score": float(score),
            "x": float(x + REFERENCE_SIZE / 2.0),
            "y": float(y + REFERENCE_SIZE / 2.0),
        })

        y0 = max(0, y - half)
        y1 = min(work.shape[0], y + half + 1)
        x0 = max(0, x - half)
        x1 = min(work.shape[1], x + half + 1)
        work[y0:y1, x0:x1] = -1.0

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Locate a reference pattern inside a search image "
            "using normalized cross-correlation."
        )
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference image.",
    )
    parser.add_argument(
        "--search",
        required=True,
        help="Path to the search image.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of matches to report. Default: 1.",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    print("=" * 72)
    print("SEMICON / DRIFT-SENSE")
    print("STANDALONE LOCALIZATION INFERENCE")
    print("=" * 72)
    print(f"Reference : {args.reference}")
    print(f"Search    : {args.search}")
    print(f"Template  : {REFERENCE_SIZE} x {REFERENCE_SIZE} px")
    print("Method    : TM_CCOEFF_NORMED")
    print("Search    : FULL IMAGE")
    print()

    try:
        reference = read_gray(args.reference)
        search = read_gray(args.search)

        response, score, top_left_x, top_left_y = run_ncc(
            reference,
            search,
        )

    except Exception as exc:
        print("INFERENCE FAILED")
        print("-" * 72)
        print(str(exc))
        return 1

    center_x = top_left_x + REFERENCE_SIZE / 2.0
    center_y = top_left_y + REFERENCE_SIZE / 2.0

    print("RESULT")
    print("-" * 72)
    print(f"Predicted center X : {center_x:.2f} px")
    print(f"Predicted center Y : {center_y:.2f} px")
    print(f"NCC score          : {score:.6f}")
    print(
        f"Matched top-left   : "
        f"({top_left_x}, {top_left_y})"
    )

    if args.top_k > 1:
        print()
        print(f"TOP-{args.top_k} MATCHES")
        print("-" * 72)

        for rank, item in enumerate(
            top_k_peaks(response, args.top_k),
            start=1,
        ):
            print(
                f"{rank}. "
                f"center=({item['x']:.2f}, {item['y']:.2f}) "
                f"score={item['score']:.6f}"
            )

    print()
    print("=" * 72)
    print("LOCALIZATION COMPLETE")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
