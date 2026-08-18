"""
MODEL A v2
Correlation + Gaussian heatmap + differentiable soft-argmax coordinate loss.

Dataset:
E:\\semicon\\generator5_augmented\\
  reference/
  search/
  training_metadata/
    train.csv
    validation.csv
    test.csv

Run sanity check:
    python train_model_A_v2.py --sanity-check

Train:
    python train_model_A_v2.py --epochs 20 --batch-size 4

Resume:
    python train_model_A_v2.py --resume --epochs 50

This version adds DIRECT coordinate supervision to the original Model A:
    total_loss = heatmap_loss + coordinate_loss

It still does NOT use:
    InfoNCE
    hard-negative mining
    global context
    refinement head
"""

from pathlib import Path
import argparse
import json
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"E:\semicon\generator5_augmented")

TRAIN_CSV = DATASET_ROOT / "training_metadata" / "train.csv"
VAL_CSV = DATASET_ROOT / "training_metadata" / "validation.csv"

REFERENCE_DIR = DATASET_ROOT / "reference"
SEARCH_DIR = DATASET_ROOT / "search"
OUTPUT_ROOT = DATASET_ROOT / "model_A_v2"

SEARCH_SIZE = 1000
REFERENCE_SIZE = 1000
REFERENCE_DOWNSAMPLE = 10
REFERENCE_MODEL_SIZE = 100
HEATMAP_SIZE = 500

GAUSSIAN_SIGMA = 4.0
FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0

COORDINATE_LOSS_WEIGHT = 1.0
SOFTMAX_TEMPERATURE = 1.0

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-4
DEFAULT_NUM_WORKERS = 0

SEED = 20260817


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

def read_grayscale(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{path}"
        )

    return image


def normalize_image(image):
    image = image.astype(np.float32) / 255.0

    return torch.from_numpy(
        image
    ).unsqueeze(0)


class Generator5Dataset(Dataset):

    def __init__(
        self,
        csv_path,
        reference_dir,
        search_dir
    ):
        self.df = pd.read_csv(
            csv_path
        )

        self.reference_dir = Path(
            reference_dir
        )

        self.search_dir = Path(
            search_dir
        )

        required = [
            "reference_file",
            "search_file",
            "center_x",
            "center_y",
            "original_pair_id",
            "augmentation"
        ]

        missing = [
            column
            for column in required
            if column not in self.df.columns
        ]

        if missing:
            raise RuntimeError(
                "Missing required metadata columns:\n"
                + "\n".join(
                    f"  {column}"
                    for column in missing
                )
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        reference_path = (
            self.reference_dir
            / Path(
                str(row["reference_file"])
            ).name
        )

        search_path = (
            self.search_dir
            / Path(
                str(row["search_file"])
            ).name
        )

        reference = read_grayscale(
            reference_path
        )

        search = read_grayscale(
            search_path
        )

        if reference.shape != (
            REFERENCE_SIZE,
            REFERENCE_SIZE
        ):
            raise RuntimeError(
                f"Bad Reference size: "
                f"{reference_path} -> "
                f"{reference.shape}"
            )

        if search.shape != (
            SEARCH_SIZE,
            SEARCH_SIZE
        ):
            raise RuntimeError(
                f"Bad Search size: "
                f"{search_path} -> "
                f"{search.shape}"
            )

        # Reference is native 1000x1000.
        # Only for coarse matching do we downsample it 10x.
        reference = cv2.resize(
            reference,
            (
                REFERENCE_MODEL_SIZE,
                REFERENCE_MODEL_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        reference = normalize_image(
            reference
        )

        search = normalize_image(
            search
        )

        center_x = float(
            row["center_x"]
        )

        center_y = float(
            row["center_y"]
        )

        heatmap_x = (
            center_x
            * HEATMAP_SIZE
            / SEARCH_SIZE
        )

        heatmap_y = (
            center_y
            * HEATMAP_SIZE
            / SEARCH_SIZE
        )

        return {
            "reference": reference,
            "search": search,

            "center_x": torch.tensor(
                center_x,
                dtype=torch.float32
            ),

            "center_y": torch.tensor(
                center_y,
                dtype=torch.float32
            ),

            "heatmap_x": torch.tensor(
                heatmap_x,
                dtype=torch.float32
            ),

            "heatmap_y": torch.tensor(
                heatmap_y,
                dtype=torch.float32
            ),

            "original_pair_id": int(
                row["original_pair_id"]
            ),

            "augmentation": str(
                row["augmentation"]
            )
        }


# ============================================================
# SHARED CNN ENCODER
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


class SharedEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(1, 16, 1),
            ConvBlock(16, 32, 2),
            ConvBlock(32, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 128, 1)
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# DEPTHWISE CROSS-CORRELATION
# ============================================================

def depthwise_cross_correlation(
    search_features,
    reference_features
):
    batch_size, channels, hs, ws = (
        search_features.shape
    )

    _, _, hr, wr = (
        reference_features.shape
    )

    search_grouped = (
        search_features.reshape(
            1,
            batch_size * channels,
            hs,
            ws
        )
    )

    reference_grouped = (
        reference_features.reshape(
            batch_size * channels,
            1,
            hr,
            wr
        )
    )

    correlation = F.conv2d(
        search_grouped,
        reference_grouped,
        stride=1,
        padding=0,
        groups=batch_size * channels
    )

    hout, wout = (
        correlation.shape[-2],
        correlation.shape[-1]
    )

    correlation = (
        correlation.reshape(
            batch_size,
            channels,
            hout,
            wout
        )
    )

    # One scalar similarity value per spatial location.
    correlation = correlation.mean(
        dim=1
    )

    return correlation


# ============================================================
# HEATMAP HEAD
# ============================================================

class HeatmapHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, correlation):

        x = F.relu(
            self.conv1(
                correlation
            ),
            inplace=True
        )

        x = self.conv2(
            x
        )

        x = F.interpolate(
            x,
            size=(
                HEATMAP_SIZE,
                HEATMAP_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        return x


# ============================================================
# MODEL A v2
# ============================================================

class ModelA(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SharedEncoder()

        self.heatmap_head = (
            HeatmapHead()
        )

    def forward(
        self,
        reference,
        search
    ):

        reference_features = (
            self.encoder(
                reference
            )
        )

        search_features = (
            self.encoder(
                search
            )
        )

        correlation = (
            depthwise_cross_correlation(
                search_features,
                reference_features
            )
        )

        heatmap_logits = (
            self.heatmap_head(
                correlation.unsqueeze(1)
            )
        )

        return {
            "heatmap_logits": heatmap_logits,
            "correlation": correlation,
            "reference_features":
                reference_features,
            "search_features":
                search_features
        }


# ============================================================
# GAUSSIAN HEATMAP TARGET
# ============================================================

def make_gaussian_heatmap(
    center_x,
    center_y,
    height,
    width,
    sigma=GAUSSIAN_SIGMA
):

    batch_size = (
        center_x.shape[0]
    )

    device = center_x.device

    y = torch.arange(
        height,
        device=device,
        dtype=torch.float32
    ).view(
        1,
        height,
        1
    )

    x = torch.arange(
        width,
        device=device,
        dtype=torch.float32
    ).view(
        1,
        1,
        width
    )

    cx = center_x.view(
        batch_size,
        1,
        1
    )

    cy = center_y.view(
        batch_size,
        1,
        1
    )

    distance_squared = (
        (x - cx) ** 2
        + (y - cy) ** 2
    )

    return torch.exp(
        -distance_squared
        / (
            2.0 * sigma * sigma
        )
    )


# ============================================================
# FOCAL HEATMAP LOSS
# ============================================================

def focal_heatmap_loss(
    logits,
    target
):

    probability = torch.sigmoid(
        logits
    ).clamp(
        1e-6,
        1.0 - 1e-6
    )

    positive = target.eq(1.0)
    negative = ~positive

    positive_loss = (
        -torch.pow(
            1.0 - probability,
            FOCAL_ALPHA
        )
        * torch.log(
            probability
        )
        * positive.float()
    )

    negative_weight = torch.pow(
        1.0 - target,
        FOCAL_BETA
    )

    negative_loss = (
        -negative_weight
        * torch.pow(
            probability,
            FOCAL_ALPHA
        )
        * torch.log(
            1.0 - probability
        )
        * negative.float()
    )

    positive_count = (
        positive.float()
        .sum(
            dim=(1, 2, 3)
        )
        .clamp(
            min=1.0
        )
    )

    loss = (
        positive_loss
        + negative_loss
    ).sum(
        dim=(1, 2, 3)
    ) / positive_count

    return loss.mean()


# ============================================================
# DIFFERENTIABLE SOFT-ARGMAX
# ============================================================

def soft_argmax_2d(
    logits,
    temperature=SOFTMAX_TEMPERATURE
):
    """
    Convert heatmap logits to differentiable expected x,y.

    Returns normalized coordinates in [0,1].

    This is used ONLY for training the coordinate loss.
    Final reported Top-1 localization still uses hard argmax.
    """

    if logits.ndim != 4:
        raise ValueError(
            "Expected [B,1,H,W], got "
            f"{tuple(logits.shape)}"
        )

    batch_size, channels, height, width = (
        logits.shape
    )

    if channels != 1:
        raise ValueError(
            "Expected one heatmap channel."
        )

    scaled = (
        logits[:, 0]
        / temperature
    )

    probabilities = torch.softmax(
        scaled.reshape(
            batch_size,
            height * width
        ),
        dim=1
    )

    x = torch.linspace(
        0.0,
        1.0,
        width,
        device=logits.device,
        dtype=logits.dtype
    )

    y = torch.linspace(
        0.0,
        1.0,
        height,
        device=logits.device,
        dtype=logits.dtype
    )

    x_grid = (
        x.view(1, 1, width)
        .expand(
            batch_size,
            height,
            width
        )
        .reshape(
            batch_size,
            height * width
        )
    )

    y_grid = (
        y.view(1, height, 1)
        .expand(
            batch_size,
            height,
            width
        )
        .reshape(
            batch_size,
            height * width
        )
    )

    predicted_x = (
        probabilities
        * x_grid
    ).sum(dim=1)

    predicted_y = (
        probabilities
        * y_grid
    ).sum(dim=1)

    return predicted_x, predicted_y


# ============================================================
# COORDINATE LOSS
# ============================================================

def coordinate_loss(
    predicted_x_norm,
    predicted_y_norm,
    ground_truth_x,
    ground_truth_y
):

    target_x_norm = (
        ground_truth_x
        / SEARCH_SIZE
    )

    target_y_norm = (
        ground_truth_y
        / SEARCH_SIZE
    )

    loss_x = F.smooth_l1_loss(
        predicted_x_norm,
        target_x_norm
    )

    loss_y = F.smooth_l1_loss(
        predicted_y_norm,
        target_y_norm
    )

    return loss_x + loss_y


# ============================================================
# HARD ARGMAX FOR EVALUATION
# ============================================================

def decode_heatmap(logits):

    probability = torch.sigmoid(
        logits
    )

    batch_size = (
        probability.shape[0]
    )

    flat = probability.reshape(
        batch_size,
        -1
    )

    indices = flat.argmax(
        dim=1
    )

    y = (
        indices
        // HEATMAP_SIZE
    )

    x = (
        indices
        % HEATMAP_SIZE
    )

    x = (
        x.float()
        * SEARCH_SIZE
        / HEATMAP_SIZE
    )

    y = (
        y.float()
        * SEARCH_SIZE
        / HEATMAP_SIZE
    )

    return x, y


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    pred_x,
    pred_y,
    gt_x,
    gt_y
):

    error_x = pred_x - gt_x
    error_y = pred_y - gt_y

    distance = torch.sqrt(
        error_x ** 2
        + error_y ** 2
    )

    return {
        "mae_x": float(
            torch.abs(
                error_x
            ).mean().item()
        ),

        "mae_y": float(
            torch.abs(
                error_y
            ).mean().item()
        ),

        "mean_error": float(
            distance.mean().item()
        ),

        "median_error": float(
            distance.median().item()
        ),

        "within_1px": float(
            (
                distance <= 1.0
            ).float().mean().item()
            * 100.0
        ),

        "within_2px": float(
            (
                distance <= 2.0
            ).float().mean().item()
            * 100.0
        ),

        "within_5px": float(
            (
                distance <= 5.0
            ).float().mean().item()
            * 100.0
        ),

        "within_10px": float(
            (
                distance <= 10.0
            ).float().mean().item()
            * 100.0
        )
    }


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_coordinate_loss = 0.0
    sample_count = 0

    predictions_x = []
    predictions_y = []
    ground_truth_x = []
    ground_truth_y = []

    for batch in loader:

        reference = batch[
            "reference"
        ].to(
            device,
            non_blocking=True
        )

        search = batch[
            "search"
        ].to(
            device,
            non_blocking=True
        )

        heatmap_x = batch[
            "heatmap_x"
        ].to(device)

        heatmap_y = batch[
            "heatmap_y"
        ].to(device)

        gt_x = batch[
            "center_x"
        ].to(device)

        gt_y = batch[
            "center_y"
        ].to(device)

        target = (
            make_gaussian_heatmap(
                heatmap_x,
                heatmap_y,
                HEATMAP_SIZE,
                HEATMAP_SIZE
            )
            .unsqueeze(1)
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            output = model(
                reference,
                search
            )

            logits = output[
                "heatmap_logits"
            ]

            heat_loss = (
                focal_heatmap_loss(
                    logits,
                    target
                )
            )

            soft_x, soft_y = (
                soft_argmax_2d(
                    logits
                )
            )

            coord_loss = (
                coordinate_loss(
                    soft_x,
                    soft_y,
                    gt_x,
                    gt_y
                )
            )

            loss = (
                heat_loss
                + COORDINATE_LOSS_WEIGHT
                * coord_loss
            )

            if training:
                loss.backward()

                optimizer.step()

        batch_size = (
            reference.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_heatmap_loss += (
            heat_loss.item()
            * batch_size
        )

        total_coordinate_loss += (
            coord_loss.item()
            * batch_size
        )

        sample_count += batch_size

        # Evaluation remains hard argmax for fair comparison
        # with the NCC baseline.
        pred_x, pred_y = (
            decode_heatmap(
                logits.detach()
            )
        )

        predictions_x.append(
            pred_x.cpu()
        )

        predictions_y.append(
            pred_y.cpu()
        )

        ground_truth_x.append(
            gt_x.detach().cpu()
        )

        ground_truth_y.append(
            gt_y.detach().cpu()
        )

    pred_x = torch.cat(
        predictions_x
    )

    pred_y = torch.cat(
        predictions_y
    )

    gt_x = torch.cat(
        ground_truth_x
    )

    gt_y = torch.cat(
        ground_truth_y
    )

    metrics = calculate_metrics(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    denominator = max(
        sample_count,
        1
    )

    metrics["loss"] = (
        total_loss
        / denominator
    )

    metrics["heatmap_loss"] = (
        total_heatmap_loss
        / denominator
    )

    metrics["coordinate_loss"] = (
        total_coordinate_loss
        / denominator
    )

    return metrics


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_error,
    history,
    args
):

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "best_val_error":
                best_val_error,

            "history":
                history,

            "config":
                vars(args)
        },
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    device
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    start_epoch = (
        int(
            checkpoint.get(
                "epoch",
                0
            )
        )
        + 1
    )

    best_val_error = float(
        checkpoint.get(
            "best_val_error",
            float("inf")
        )
    )

    history = checkpoint.get(
        "history",
        []
    )

    return (
        start_epoch,
        best_val_error,
        history
    )


# ============================================================
# SANITY CHECK
# ============================================================

def run_sanity_check(
    model,
    dataset,
    device
):

    print()
    print("=" * 76)
    print("MODEL A v2 FORWARD/BACKWARD SANITY CHECK")
    print("=" * 76)

    sample = dataset[0]

    reference = (
        sample["reference"]
        .unsqueeze(0)
        .to(device)
    )

    search = (
        sample["search"]
        .unsqueeze(0)
        .to(device)
    )

    gt_x = (
        sample["center_x"]
        .unsqueeze(0)
        .to(device)
    )

    gt_y = (
        sample["center_y"]
        .unsqueeze(0)
        .to(device)
    )

    heatmap_x = (
        sample["heatmap_x"]
        .unsqueeze(0)
        .to(device)
    )

    heatmap_y = (
        sample["heatmap_y"]
        .unsqueeze(0)
        .to(device)
    )

    print(
        f"Reference input : "
        f"{tuple(reference.shape)}"
    )

    print(
        f"Search input    : "
        f"{tuple(search.shape)}"
    )

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DEFAULT_LR
    )

    output = model(
        reference,
        search
    )

    logits = output[
        "heatmap_logits"
    ]

    target = (
        make_gaussian_heatmap(
            heatmap_x,
            heatmap_y,
            HEATMAP_SIZE,
            HEATMAP_SIZE
        )
        .unsqueeze(1)
    )

    heat_loss = (
        focal_heatmap_loss(
            logits,
            target
        )
    )

    soft_x, soft_y = (
        soft_argmax_2d(
            logits
        )
    )

    coord_loss = (
        coordinate_loss(
            soft_x,
            soft_y,
            gt_x,
            gt_y
        )
    )

    total_loss = (
        heat_loss
        + COORDINATE_LOSS_WEIGHT
        * coord_loss
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss.backward()

    optimizer.step()

    print(
        f"Reference features: "
        f"{tuple(output['reference_features'].shape)}"
    )

    print(
        f"Search features    : "
        f"{tuple(output['search_features'].shape)}"
    )

    print(
        f"Correlation        : "
        f"{tuple(output['correlation'].shape)}"
    )

    print(
        f"Heatmap logits     : "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Initial heatmap loss: "
        f"{heat_loss.item():.6f}"
    )

    print(
        f"Initial coordinate loss: "
        f"{coord_loss.item():.6f}"
    )

    print(
        f"Initial total loss: "
        f"{total_loss.item():.6f}"
    )

    print(
        f"Soft-argmax x: "
        f"{soft_x.item() * SEARCH_SIZE:.2f} "
        f"(GT {gt_x.item():.2f})"
    )

    print(
        f"Soft-argmax y: "
        f"{soft_y.item() * SEARCH_SIZE:.2f} "
        f"(GT {gt_y.item():.2f})"
    )

    print()
    print(
        "FORWARD + BACKWARD SANITY CHECK: PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Model A v2 with "
            "soft-argmax coordinate supervision."
        )
    )

    parser.add_argument(
        "--train-csv",
        default=str(TRAIN_CSV)
    )

    parser.add_argument(
        "--val-csv",
        default=str(VAL_CSV)
    )

    parser.add_argument(
        "--reference-dir",
        default=str(REFERENCE_DIR)
    )

    parser.add_argument(
        "--search-dir",
        default=str(SEARCH_DIR)
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT)
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu"
        ],
        default="auto"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--resume-from",
        default=None
    )

    parser.add_argument(
        "--sanity-check",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    if args.device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            "cuda"
        )

    elif args.device == "cpu":

        device = torch.device(
            "cpu"
        )

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 76)
    print(
        "SEMICON / DRIFT-SENSE"
    )
    print(
        "MODEL A v2 - CORRELATION + HEATMAP + SOFT-ARGMAX"
    )
    print("=" * 76)

    print()
    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Train CSV    : {args.train_csv}"
    )

    print(
        f"Val CSV      : {args.val_csv}"
    )

    print(
        f"Epochs       : {args.epochs}"
    )

    print(
        f"Batch size   : {args.batch_size}"
    )

    print(
        f"Learning rate: {args.lr}"
    )

    print()
    print("Architecture:")
    print(
        "  Encoder channels: "
        "16 -> 32 -> 64 -> 128 -> 128"
    )

    print(
        "  Encoder strides : "
        "1 -> 2 -> 2 -> 2 -> 1"
    )

    print(
        "  Reference       : "
        "1000 -> 100 pixels"
    )

    print(
        "  Correlation     : "
        "depthwise cross-correlation"
    )

    print(
        "  Heatmap         : "
        "500 x 500"
    )

    print(
        "  Soft-argmax     : YES"
    )

    print(
        "  Coordinate loss : "
        "SmoothL1, weight=1.0"
    )

    print()
    print("Not included:")
    print("  InfoNCE          : NO")
    print("  Hard-negative    : NO")
    print("  Global context   : NO")
    print("  Refinement head  : NO")

    train_dataset = (
        Generator5Dataset(
            args.train_csv,
            args.reference_dir,
            args.search_dir
        )
    )

    val_dataset = (
        Generator5Dataset(
            args.val_csv,
            args.reference_dir,
            args.search_dir
        )
    )

    print()
    print(
        f"Training samples  : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    model = ModelA().to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    if args.sanity_check:

        run_sanity_check(
            model,
            train_dataset,
            device
        )

        return

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    start_epoch = 1
    best_val_error = float("inf")
    history = []

    checkpoint_path = None

    if args.resume_from:

        checkpoint_path = Path(
            args.resume_from
        )

    elif args.resume:

        checkpoint_path = (
            output_dir
            / "model_A_v2_last.pt"
        )

    if checkpoint_path is not None:

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"{checkpoint_path}"
            )

        (
            start_epoch,
            best_val_error,
            history
        ) = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            device
        )

        print()
        print(
            f"Resuming at epoch "
            f"{start_epoch}"
        )

    if start_epoch > args.epochs:

        print(
            "Checkpoint already reached "
            f"epoch {args.epochs}."
        )

        print(
            "Use a larger --epochs value "
            "to continue."
        )

        return

    for epoch in range(
        start_epoch,
        args.epochs + 1
    ):

        epoch_start = time.time()

        print()
        print("=" * 76)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("=" * 76)

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer
        )

        with torch.no_grad():

            val_metrics = run_epoch(
                model,
                val_loader,
                device
            )

        elapsed = (
            time.time()
            - epoch_start
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
                "seconds": elapsed
            }
        )

        print()
        print(
            f"Train total loss : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Train heatmap    : "
            f"{train_metrics['heatmap_loss']:.6f}"
        )

        print(
            f"Train coordinate : "
            f"{train_metrics['coordinate_loss']:.6f}"
        )

        print(
            f"Train mean error : "
            f"{train_metrics['mean_error']:.3f} px"
        )

        print(
            f"Val total loss   : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Val heatmap      : "
            f"{val_metrics['heatmap_loss']:.6f}"
        )

        print(
            f"Val coordinate   : "
            f"{val_metrics['coordinate_loss']:.6f}"
        )

        print(
            f"Val mean error   : "
            f"{val_metrics['mean_error']:.3f} px"
        )

        print(
            f"Val median error : "
            f"{val_metrics['median_error']:.3f} px"
        )

        print(
            f"Val <= 5 px      : "
            f"{val_metrics['within_5px']:.2f}%"
        )

        print(
            f"Val <= 10 px     : "
            f"{val_metrics['within_10px']:.2f}%"
        )

        print(
            f"Epoch time       : "
            f"{elapsed:.1f} s"
        )

        save_checkpoint(
            output_dir
            / "model_A_v2_last.pt",
            model,
            optimizer,
            epoch,
            best_val_error,
            history,
            args
        )

        print(
            "  -> Latest checkpoint saved."
        )

        if (
            val_metrics["mean_error"]
            < best_val_error
        ):

            best_val_error = (
                val_metrics["mean_error"]
            )

            save_checkpoint(
                output_dir
                / "model_A_v2_best.pt",
                model,
                optimizer,
                epoch,
                best_val_error,
                history,
                args
            )

            print(
                "  -> BEST checkpoint saved."
            )

        with open(
            output_dir
            / "training_history.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=2
            )

    with open(
        output_dir
        / "model_A_v2_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "model": "Model A v2",
                "encoder":
                    "16-32-64-128-128",
                "strides":
                    "1-2-2-2-1",
                "reference_downsample":
                    10,
                "heatmap_size":
                    500,
                "heatmap_loss":
                    "Gaussian focal",
                "coordinate_loss":
                    "SmoothL1",
                "coordinate_loss_weight":
                    COORDINATE_LOSS_WEIGHT,
                "soft_argmax":
                    True,
                "softmax_temperature":
                    SOFTMAX_TEMPERATURE,
                "InfoNCE":
                    False,
                "hard_negative_mining":
                    False,
                "global_context":
                    False,
                "refinement":
                    False
            },
            file,
            indent=2
        )

    print()
    print("=" * 76)
    print("MODEL A v2 TRAINING COMPLETE")
    print("=" * 76)

    print(
        f"Best validation error: "
        f"{best_val_error:.3f} px"
    )

    print(
        f"Latest checkpoint: "
        f"{output_dir / 'model_A_v2_last.pt'}"
    )

    print(
        f"Best checkpoint: "
        f"{output_dir / 'model_A_v2_best.pt'}"
    )


if __name__ == "__main__":
    main()