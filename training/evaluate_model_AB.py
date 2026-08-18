"""
SEMICON / DRIFT-SENSE
FINAL TEST EVALUATION - MODEL A vs MODEL B

Evaluates the existing Generator 5 test split (normally 210 samples).
No training. No image modification.

Run in Colab:
!python /content/evaluate_model_AB.py \
    --data-root /content/generator5_augmented \
    --device cuda
"""

from pathlib import Path
import argparse
import ast
import json

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SEARCH_SIZE = 1000
REFERENCE_MODEL_SIZE = 100
HEATMAP_SIZE = 500
TOP_K = 3


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to read image:\n{path}")
    return image


def parse_coordinates(value):
    if isinstance(value, list):
        data = value
    else:
        text = str(value).strip()
        try:
            data = json.loads(text)
        except Exception:
            data = ast.literal_eval(text)

    if not isinstance(data, list):
        raise RuntimeError("all_block_coordinates is not a list.")

    result = []
    for item in data:
        result.append({
            "row": int(item.get("row", item.get("block_row", 0))),
            "col": int(item.get("col", item.get("block_col", 0))),
            "center_x": float(item["center_x"]),
            "center_y": float(item["center_y"]),
        })
    return result


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 16, 1),
            ConvBlock(16, 32, 2),
            ConvBlock(32, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 128, 1),
        )

    def forward(self, x):
        return self.encoder(x)


def depthwise_cross_correlation(search_features, reference_features):
    batch_size, channels, _, _ = search_features.shape

    search_grouped = search_features.reshape(
        1, batch_size * channels,
        search_features.shape[-2],
        search_features.shape[-1],
    )

    reference_grouped = reference_features.reshape(
        batch_size * channels, 1,
        reference_features.shape[-2],
        reference_features.shape[-1],
    )

    correlation = F.conv2d(
        search_grouped,
        reference_grouped,
        groups=batch_size * channels,
    )

    correlation = correlation.reshape(
        batch_size, channels,
        correlation.shape[-2],
        correlation.shape[-1],
    )

    return correlation.mean(dim=1)


class HeatmapHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, correlation):
        return F.interpolate(
            self.layers(correlation.unsqueeze(1)),
            size=(HEATMAP_SIZE, HEATMAP_SIZE),
            mode="bilinear",
            align_corners=False,
        )


class ModelAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.heatmap_head = HeatmapHead()

    def forward(self, reference, search):
        ref_features = self.encoder(reference)
        search_features = self.encoder(search)
        correlation = depthwise_cross_correlation(
            search_features, ref_features
        )
        return self.heatmap_head(correlation)


def soft_argmax_2d(logits):
    batch_size, _, height, width = logits.shape
    values = logits[:, 0].float()

    probabilities = torch.softmax(
        values.reshape(batch_size, height * width), dim=1
    )

    x = torch.linspace(0, 1, width, device=logits.device)
    y = torch.linspace(0, 1, height, device=logits.device)

    x_grid = x.view(1, 1, width).expand(
        batch_size, height, width
    ).reshape(batch_size, height * width)

    y_grid = y.view(1, height, 1).expand(
        batch_size, height, width
    ).reshape(batch_size, height * width)

    px = (probabilities * x_grid).sum(dim=1) * SEARCH_SIZE
    py = (probabilities * y_grid).sum(dim=1) * SEARCH_SIZE
    return px, py


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return checkpoint


def prepare_pair(row, reference_dir, search_dir):
    ref_path = reference_dir / Path(str(row["reference_file"])).name
    search_path = search_dir / Path(str(row["search_file"])).name

    reference = read_gray(ref_path)
    search = read_gray(search_path)

    reference = cv2.resize(
        reference, (REFERENCE_MODEL_SIZE, REFERENCE_MODEL_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    ref_tensor = torch.from_numpy(
        reference.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)

    search_tensor = torch.from_numpy(
        search.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)

    return ref_tensor, search_tensor


def block_accuracy(px, py, row, coordinates):
    if not coordinates:
        return None, None

    distances = np.array([
        np.hypot(
            px - item["center_x"],
            py - item["center_y"]
        )
        for item in coordinates
    ])

    order = np.argsort(distances)

    target_index = None
    if not pd.isna(row.get("selected_block_row", np.nan)) and \
       not pd.isna(row.get("selected_block_col", np.nan)):
        tr = int(row["selected_block_row"])
        tc = int(row["selected_block_col"])
        for i, item in enumerate(coordinates):
            if item["row"] == tr and item["col"] == tc:
                target_index = i
                break

    if target_index is None:
        return None, None

    top1 = int(order[0] == target_index)
    top3 = int(target_index in order[:TOP_K])
    return top1, top3


def evaluate(model_name, checkpoint_path, df, root, device):
    print()
    print("=" * 76)
    print(f"EVALUATING {model_name}")
    print("=" * 76)

    model = ModelAB().to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    reference_dir = Path(root) / "reference"
    search_dir = Path(root) / "search"

    records = []

    with torch.inference_mode():
        for i, row in df.iterrows():
            ref, search = prepare_pair(
                row, reference_dir, search_dir
            )

            ref = ref.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)

            if device.type == "cuda":
                ref = ref.contiguous(memory_format=torch.channels_last)
                search = search.contiguous(memory_format=torch.channels_last)

            logits = model(ref, search)
            px, py = soft_argmax_2d(logits)

            px = float(px[0].cpu())
            py = float(py[0].cpu())

            gx = float(row["center_x"])
            gy = float(row["center_y"])

            ex = px - gx
            ey = py - gy
            error = float(np.hypot(ex, ey))

            top1 = None
            top3 = None

            if "all_block_coordinates" in row.index:
                try:
                    coords = parse_coordinates(
                        row["all_block_coordinates"]
                    )
                    top1, top3 = block_accuracy(
                        px, py, row, coords
                    )
                except Exception:
                    pass

            augmentation = row.get(
                "augmentation",
                row.get("augmentation_type", "unknown")
            )

            target_type = row.get("target_type", "unknown")

            records.append({
                "index": int(i),
                "reference_file": row["reference_file"],
                "search_file": row["search_file"],
                "augmentation": augmentation,
                "target_type": target_type,
                "gt_x": gx,
                "gt_y": gy,
                "pred_x": px,
                "pred_y": py,
                "error_x": ex,
                "error_y": ey,
                "error": error,
                "within_1": int(error <= 1),
                "within_2": int(error <= 2),
                "within_5": int(error <= 5),
                "within_10": int(error <= 10),
                "top1": top1,
                "top3": top3,
                "wrong_periodic": (
                    None if top1 is None else 1 - top1
                ),
            })

            if (i + 1) % 25 == 0 or i == len(df) - 1:
                print(f"[{i + 1:03d}/{len(df):03d}]")

    result = pd.DataFrame(records)

    summary = {
        "model": model_name,
        "samples": int(len(result)),
        "mean_error_px": float(result["error"].mean()),
        "median_error_px": float(result["error"].median()),
        "mae_x_px": float(result["error_x"].abs().mean()),
        "mae_y_px": float(result["error_y"].abs().mean()),
        "within_1_percent": float(result["within_1"].mean() * 100),
        "within_2_percent": float(result["within_2"].mean() * 100),
        "within_5_percent": float(result["within_5"].mean() * 100),
        "within_10_percent": float(result["within_10"].mean() * 100),
    }

    valid_top = result["top1"].notna()
    if valid_top.any():
        summary["top1_percent"] = float(
            result.loc[valid_top, "top1"].mean() * 100
        )
        summary["top3_percent"] = float(
            result.loc[valid_top, "top3"].mean() * 100
        )
        summary["wrong_periodic_percent"] = float(
            result.loc[valid_top, "wrong_periodic"].mean() * 100
        )
    else:
        summary["top1_percent"] = None
        summary["top3_percent"] = None
        summary["wrong_periodic_percent"] = None

    print()
    print(f"{model_name} RESULTS")
    print(f"Mean error       : {summary['mean_error_px']:.3f} px")
    print(f"Median error     : {summary['median_error_px']:.3f} px")
    print(f"MAE-x            : {summary['mae_x_px']:.3f} px")
    print(f"MAE-y            : {summary['mae_y_px']:.3f} px")
    print(f"<= 1 px          : {summary['within_1_percent']:.2f}%")
    print(f"<= 2 px          : {summary['within_2_percent']:.2f}%")
    print(f"<= 5 px          : {summary['within_5_percent']:.2f}%")
    print(f"<= 10 px         : {summary['within_10_percent']:.2f}%")

    if summary["top1_percent"] is not None:
        print(f"Top-1            : {summary['top1_percent']:.2f}%")
        print(f"Top-3            : {summary['top3_percent']:.2f}%")
        print(
            f"Wrong-periodic   : "
            f"{summary['wrong_periodic_percent']:.2f}%"
        )

    return result, summary


def print_breakdown(result, column):
    if column not in result.columns:
        return

    unique = result[column].dropna().astype(str).unique()
    if len(unique) == 0 or (len(unique) == 1 and unique[0] == "unknown"):
        return

    print()
    print("-" * 76)
    print(f"{column.upper()} BREAKDOWN")
    print("-" * 76)

    for value, group in result.groupby(column, dropna=False):
        print(f"\n{value}: {len(group)} samples")
        print(f"  mean error : {group['error'].mean():.3f} px")
        print(f"  median     : {group['error'].median():.3f} px")
        print(f"  <= 5 px    : {group['within_5'].mean() * 100:.2f}%")
        print(f"  <= 10 px   : {group['within_10'].mean() * 100:.2f}%")

        valid = group["top1"].notna()
        if valid.any():
            print(f"  top-1      : {group.loc[valid, 'top1'].mean() * 100:.2f}%")
            print(f"  top-3      : {group.loc[valid, 'top3'].mean() * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/content/generator5_augmented"
    )
    parser.add_argument("--model-a", default=None)
    parser.add_argument("--model-b", default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto"
    )
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    root = Path(args.data_root)

    model_a_path = Path(args.model_a) if args.model_a else (
        root / "model_A_clean" / "model_A_best.pt"
    )
    model_b_path = Path(args.model_b) if args.model_b else (
        root / "model_B" / "model_B_last.pt"
    )

    test_csv = root / "training_metadata" / "test.csv"

    print("=" * 76)
    print("SEMICON / DRIFT-SENSE")
    print("FINAL TEST EVALUATION - MODEL A vs MODEL B")
    print("=" * 76)
    print(f"Device   : {device}")
    if device.type == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Test CSV : {test_csv}")
    print(f"Model A  : {model_a_path}")
    print(f"Model B  : {model_b_path}")

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found:\n{test_csv}")
    if not model_a_path.exists():
        raise FileNotFoundError(f"Model A checkpoint not found:\n{model_a_path}")
    if not model_b_path.exists():
        raise FileNotFoundError(f"Model B checkpoint not found:\n{model_b_path}")

    df = pd.read_csv(test_csv)
    print(f"Test samples: {len(df)}")

    output_dir = root / "final_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    a_result, a_summary = evaluate(
        "Model A", model_a_path, df, root, device
    )
    print_breakdown(a_result, "augmentation")
    print_breakdown(a_result, "target_type")

    b_result, b_summary = evaluate(
        "Model B", model_b_path, df, root, device
    )
    print_breakdown(b_result, "augmentation")
    print_breakdown(b_result, "target_type")

    a_result.to_csv(
        output_dir / "model_A_test_results.csv", index=False
    )
    b_result.to_csv(
        output_dir / "model_B_test_results.csv", index=False
    )

    with open(
        output_dir / "model_A_test_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(a_summary, f, indent=2)

    with open(
        output_dir / "model_B_test_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(b_summary, f, indent=2)

    comparison = pd.DataFrame([a_summary, b_summary])
    comparison.to_csv(
        output_dir / "model_A_vs_B_comparison.csv",
        index=False
    )

    print()
    print("=" * 76)
    print("FINAL MODEL A vs MODEL B COMPARISON")
    print("=" * 76)

    columns = [
        "model",
        "mean_error_px",
        "median_error_px",
        "within_5_percent",
        "within_10_percent",
        "top1_percent",
        "top3_percent",
        "wrong_periodic_percent",
    ]

    print(comparison[columns].to_string(index=False))

    print()
    print("Saved to:")
    print(output_dir)
    print("=" * 76)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()
