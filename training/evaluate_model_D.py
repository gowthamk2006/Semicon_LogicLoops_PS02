from pathlib import Path
import argparse
import importlib.util
import json
import cv2
import numpy as np
import pandas as pd
import torch


# ============================================================
# SEMICON / DRIFT-SENSE
# FINAL TEST EVALUATION - MODEL D
#
# IMPORTANT:
# This evaluator deliberately IMPORTS THE ORIGINAL
# train_model_D.py so the checkpoint is evaluated with the
# EXACT Model D architecture used during training.
#
# This avoids architecture drift between training/evaluation.
# ============================================================

DATASET_ROOT = Path(
    "/content/drive/MyDrive/semicon/generator5_augmented"
)

TEST_CSV = (
    DATASET_ROOT
    / "training_metadata"
    / "test.csv"
)

REFERENCE_DIR = DATASET_ROOT / "reference"
SEARCH_DIR = DATASET_ROOT / "search"

CHECKPOINT = Path(
    "/content/drive/MyDrive/semicon/checkpoints/model_D/model_D_best.pt"
)

TRAIN_SCRIPT = Path(
    "/content/train_model_D.py"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/semicon/test_results/model_D"
)

RAW_SIZE = 1000


# ============================================================
# LOAD ORIGINAL TRAINING MODULE
# ============================================================

def load_training_module(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Original Model D training script not found:\n{path}\n\n"
            "Upload the exact train_model_D.py used for training "
            "to /content/ before running this evaluator."
        )

    spec = importlib.util.spec_from_file_location(
        "original_train_model_D",
        str(path),
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def find_model_class(module):

    # Normal expected name.
    if hasattr(module, "ModelD"):
        return module.ModelD

    # Fallback names.
    for name in [
        "SpatialCorrelationModel",
        "SpatialCorrelationNet",
        "Model",
    ]:
        if hasattr(module, name):
            candidate = getattr(module, name)

            if isinstance(candidate, type):
                return candidate

    raise RuntimeError(
        "Could not find the Model D class in train_model_D.py.\n"
        "Expected a class named ModelD."
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    if image.shape != (
        RAW_SIZE,
        RAW_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected image shape {image.shape}\n"
            f"File: {path}\n"
            f"Expected: {(RAW_SIZE, RAW_SIZE)}"
        )

    # Model D used a 128x128 internal representation.
    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA,
    )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    return torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)


# ============================================================
# SOFT ARGMAX
# ============================================================

def soft_argmax(score_map, temperature=0.05):

    b, _, h, w = score_map.shape

    probabilities = torch.softmax(
        score_map.reshape(b, -1)
        / temperature,
        dim=1,
    )

    yy, xx = torch.meshgrid(
        torch.arange(
            h,
            device=score_map.device,
            dtype=torch.float32,
        ),
        torch.arange(
            w,
            device=score_map.device,
            dtype=torch.float32,
        ),
        indexing="ij",
    )

    xx = xx.reshape(1, -1)
    yy = yy.reshape(1, -1)

    px = (
        probabilities * xx
    ).sum(dim=1)

    py = (
        probabilities * yy
    ).sum(dim=1)

    if w > 1:
        px = (
            px
            / (w - 1)
            * (RAW_SIZE - 1)
        )

    if h > 1:
        py = (
            py
            / (h - 1)
            * (RAW_SIZE - 1)
        )

    return torch.stack(
        [px, py],
        dim=1,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint,
    device,
):

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
    )

    if (
        isinstance(checkpoint_data, dict)
        and "model_state_dict" in checkpoint_data
    ):
        state_dict = (
            checkpoint_data[
                "model_state_dict"
            ]
        )
    else:
        state_dict = checkpoint_data

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return checkpoint_data


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-script",
        default=str(TRAIN_SCRIPT),
    )

    parser.add_argument(
        "--test-csv",
        default=str(TEST_CSV),
    )

    parser.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print(
        "SEMICON / DRIFT-SENSE"
    )
    print(
        "FINAL TEST EVALUATION - MODEL D"
    )
    print("=" * 72)

    print(
        "Device     :",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU        :",
            torch.cuda.get_device_name(0),
        )

    print(
        "Test CSV   :",
        args.test_csv,
    )

    print(
        "Checkpoint :",
        args.checkpoint,
    )

    print(
        "Train code :",
        args.train_script,
    )

    test_csv = Path(args.test_csv)
    checkpoint = Path(args.checkpoint)
    train_script = Path(args.train_script)

    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{test_csv}"
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint}"
        )

    # --------------------------------------------------------
    # Load EXACT architecture from original training script.
    # --------------------------------------------------------

    print()
    print(
        "Loading original Model D architecture..."
    )

    training_module = load_training_module(
        train_script
    )

    ModelClass = find_model_class(
        training_module
    )

    model = ModelClass().to(device)

    print(
        "Model class:",
        ModelClass.__name__,
    )

    # --------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------

    checkpoint_data = load_checkpoint(
        model,
        checkpoint,
        device,
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    if (
        isinstance(checkpoint_data, dict)
        and "epoch" in checkpoint_data
    ):
        print(
            "Checkpoint epoch:",
            checkpoint_data["epoch"],
        )

    if (
        isinstance(checkpoint_data, dict)
        and "metrics" in checkpoint_data
    ):
        print(
            "Checkpoint metrics:",
            checkpoint_data["metrics"],
        )

    # --------------------------------------------------------
    # Load test CSV.
    # --------------------------------------------------------

    df = pd.read_csv(test_csv)

    required = [
        "reference_file",
        "search_file",
        "center_x",
        "center_y",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    print()
    print(
        "Test samples:",
        len(df),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # --------------------------------------------------------
    # Test.
    # --------------------------------------------------------

    with torch.no_grad():

        for i, row in df.iterrows():

            reference_path = (
                REFERENCE_DIR
                / Path(
                    str(row["reference_file"])
                ).name
            )

            search_path = (
                SEARCH_DIR
                / Path(
                    str(row["search_file"])
                ).name
            )

            reference = load_image(
                reference_path
            ).to(device)

            search = load_image(
                search_path
            ).to(device)

            output = model(
                reference,
                search,
            )

            # Some training implementations return
            # (score_map, auxiliary_output). Handle both.
            if isinstance(output, (tuple, list)):
                score_map = output[0]
            else:
                score_map = output

            prediction = soft_argmax(
                score_map,
                temperature=args.temperature,
            )[0].cpu().numpy()

            pred_x = float(
                prediction[0]
            )

            pred_y = float(
                prediction[1]
            )

            true_x = float(
                row["center_x"]
            )

            true_y = float(
                row["center_y"]
            )

            error = float(
                np.hypot(
                    pred_x - true_x,
                    pred_y - true_y,
                )
            )

            results.append(
                {
                    "index": int(i),
                    "reference_file":
                        str(row["reference_file"]),
                    "search_file":
                        str(row["search_file"]),
                    "true_x": true_x,
                    "true_y": true_y,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "error_px": error,
                }
            )

            if (
                i == 0
                or (i + 1) % 25 == 0
                or i == len(df) - 1
            ):
                print(
                    f"[{i + 1:3d}/{len(df)}] "
                    f"error={error:.2f} px"
                )

    # --------------------------------------------------------
    # Metrics.
    # --------------------------------------------------------

    errors = np.asarray(
        [
            item["error_px"]
            for item in results
        ],
        dtype=np.float64,
    )

    summary = {
        "model": "Model D",
        "test_samples": len(results),
        "checkpoint": str(checkpoint),
        "mean_error_px": float(
            errors.mean()
        ),
        "median_error_px": float(
            np.median(errors)
        ),
        "min_error_px": float(
            errors.min()
        ),
        "max_error_px": float(
            errors.max()
        ),
        "within_1_px_percent": float(
            (errors <= 1).mean() * 100
        ),
        "within_2_px_percent": float(
            (errors <= 2).mean() * 100
        ),
        "within_5_px_percent": float(
            (errors <= 5).mean() * 100
        ),
        "within_10_px_percent": float(
            (errors <= 10).mean() * 100
        ),
    }

    sorted_results = sorted(
        results,
        key=lambda x: x["error_px"],
    )

    summary["best_samples"] = (
        sorted_results[:10]
    )

    summary["worst_samples"] = (
        sorted_results[-10:][::-1]
    )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------

    predictions_path = (
        OUTPUT_DIR
        / "model_D_test_predictions.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "model_D_test_summary.json"
    )

    pd.DataFrame(
        results
    ).to_csv(
        predictions_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MODEL D - FINAL TEST RESULT"
    )
    print("=" * 72)

    print(
        f"Test samples       : "
        f"{len(results)}"
    )

    print(
        f"Mean error         : "
        f"{summary['mean_error_px']:.3f} px"
    )

    print(
        f"Median error       : "
        f"{summary['median_error_px']:.3f} px"
    )

    print(
        f"Minimum error      : "
        f"{summary['min_error_px']:.3f} px"
    )

    print(
        f"Maximum error      : "
        f"{summary['max_error_px']:.3f} px"
    )

    print(
        f"Within 1 px        : "
        f"{summary['within_1_px_percent']:.2f}%"
    )

    print(
        f"Within 2 px        : "
        f"{summary['within_2_px_percent']:.2f}%"
    )

    print(
        f"Within 5 px        : "
        f"{summary['within_5_px_percent']:.2f}%"
    )

    print(
        f"Within 10 px       : "
        f"{summary['within_10_px_percent']:.2f}%"
    )

    print()
    print(
        "Predictions CSV:"
    )
    print(
        predictions_path
    )

    print()
    print(
        "Summary JSON:"
    )
    print(
        summary_path
    )

    print()
    print("=" * 72)
    print(
        "MODEL D TEST EVALUATION COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()