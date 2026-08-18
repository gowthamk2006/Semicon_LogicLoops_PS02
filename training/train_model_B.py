"""
SEMICON / DRIFT-SENSE
MODEL B - CORRELATION + HEATMAP + COORDINATE + HARD-NEGATIVE INFONCE

Model B is Model A plus all 15 non-target blocks as hard negatives.

For every training sample:
    Reference
       |
       +---- positive target block
       |
       +---- 15 other blocks from the SAME Search image

No additional augmentation is applied to negatives.

Loss:
    L = heatmap_loss
        + coordinate_loss
        + lambda_infonce * InfoNCE

InfoNCE:
    temperature = 0.1

Training:
    Phase 1:
        heatmap + coordinate
        lambda_infonce = 0

    Phase 2:
        heatmap + coordinate + InfoNCE
        lambda_infonce ramps from 0 to 1

    Phase 3:
        optional continuation using lambda_infonce = 1

The script is designed for the existing Generator 5 dataset.

IMPORTANT:
The current Generator 5 metadata must contain:
    all_block_coordinates
    target_block
    selected_block_row
    selected_block_col
    center_x
    center_y

If all_block_coordinates is stored as JSON text, this script parses it.

Example:

    !python /content/train_model_B.py \
        --data-root /content/generator5_augmented \
        --phase 1 \
        --epochs 5 \
        --batch-size 8 \
        --num-workers 2 \
        --device cuda

Then Phase 2:

    !python /content/train_model_B.py \
        --data-root /content/generator5_augmented \
        --phase 2 \
        --epochs 20 \
        --batch-size 8 \
        --num-workers 2 \
        --device cuda \
        --resume

For a quick test:

    --phase 1 --epochs 2
"""

from pathlib import Path
import argparse
import ast
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

DEFAULT_DATA_ROOT = "/content/generator5_augmented"

SEARCH_SIZE = 1000
REFERENCE_SIZE = 1000
REFERENCE_MODEL_SIZE = 100

HEATMAP_SIZE = 500

GAUSSIAN_SIGMA = 4.0

FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0

COORDINATE_LOSS_WEIGHT = 1.0

INFONCE_TEMPERATURE = 0.1

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-4
DEFAULT_NUM_WORKERS = 2

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
# IMAGE HELPERS
# ============================================================

def read_gray(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read image:\n{path}"
        )

    return image


# ============================================================
# METADATA HELPERS
# ============================================================

def parse_all_block_coordinates(value):
    """
    Converts the all_block_coordinates metadata field into
    a list of 16 dictionaries.

    Accepted examples:
        JSON string
        Python literal string

    Expected logical form:
        [
            {"row": 0, "col": 0, "center_x": ..., "center_y": ...},
            ...
        ]

    Also accepts common alternative key names.
    """

    if isinstance(value, list):
        data = value

    else:
        text = str(value).strip()

        if not text:
            raise RuntimeError(
                "Empty all_block_coordinates field."
            )

        try:
            data = json.loads(text)
        except Exception:
            try:
                data = ast.literal_eval(text)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to parse all_block_coordinates:\n"
                    f"{text[:500]}"
                ) from exc

    if not isinstance(data, list):
        raise RuntimeError(
            "all_block_coordinates must contain a list."
        )

    if len(data) != 16:
        raise RuntimeError(
            "Expected 16 block coordinates, "
            f"found {len(data)}."
        )

    output = []

    for item in data:

        if not isinstance(item, dict):
            raise RuntimeError(
                "Each block coordinate must be a dictionary."
            )

        row = item.get(
            "row",
            item.get(
                "block_row",
                item.get(
                    "selected_block_row"
                )
            )
        )

        col = item.get(
            "col",
            item.get(
                "block_col",
                item.get(
                    "selected_block_col"
                )
            )
        )

        center_x = item.get(
            "center_x"
        )

        center_y = item.get(
            "center_y"
        )

        if (
            row is None
            or col is None
            or center_x is None
            or center_y is None
        ):
            raise RuntimeError(
                "Invalid block coordinate entry:\n"
                f"{item}"
            )

        output.append(
            {
                "row": int(row),
                "col": int(col),
                "center_x": float(center_x),
                "center_y": float(center_y)
            }
        )

    return output


def find_target_block_index(
    coordinates,
    target_row,
    target_col
):
    for index, item in enumerate(coordinates):

        if (
            item["row"] == int(target_row)
            and
            item["col"] == int(target_col)
        ):
            return index

    raise RuntimeError(
        "Target block was not found in "
        "all_block_coordinates."
    )


# ============================================================
# DATASET
# ============================================================

class Generator5HardNegativeDataset(Dataset):

    def __init__(
        self,
        csv_path,
        reference_dir,
        search_dir
    ):
        self.df = pd.read_csv(
            csv_path
        )

        required = [
            "reference_file",
            "search_file",
            "center_x",
            "center_y",
            "all_block_coordinates",
            "selected_block_row",
            "selected_block_col"
        ]

        missing = [
            column
            for column in required
            if column not in self.df.columns
        ]

        if missing:
            raise RuntimeError(
                "Missing required Model B metadata columns:\n"
                + "\n".join(missing)
            )

        reference_dir = Path(
            reference_dir
        )

        search_dir = Path(
            search_dir
        )

        self.reference_paths = [
            reference_dir
            / Path(str(name)).name
            for name in self.df[
                "reference_file"
            ]
        ]

        self.search_paths = [
            search_dir
            / Path(str(name)).name
            for name in self.df[
                "search_file"
            ]
        ]

        self.center_x = (
            self.df["center_x"]
            .astype(np.float32)
            .to_numpy()
        )

        self.center_y = (
            self.df["center_y"]
            .astype(np.float32)
            .to_numpy()
        )

        self.target_rows = (
            self.df[
                "selected_block_row"
            ]
            .astype(int)
            .to_numpy()
        )

        self.target_cols = (
            self.df[
                "selected_block_col"
            ]
            .astype(int)
            .to_numpy()
        )

        self.all_coordinates = []

        for value in self.df[
            "all_block_coordinates"
        ]:
            coordinates = parse_all_block_coordinates(
                value
            )

            self.all_coordinates.append(
                coordinates
            )

        self.target_indices = []

        for index in range(
            len(self.df)
        ):
            target_index = (
                find_target_block_index(
                    self.all_coordinates[index],
                    self.target_rows[index],
                    self.target_cols[index]
                )
            )

            self.target_indices.append(
                target_index
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        reference = read_gray(
            self.reference_paths[index]
        )

        search = read_gray(
            self.search_paths[index]
        )

        if reference.shape != (
            REFERENCE_SIZE,
            REFERENCE_SIZE
        ):
            raise RuntimeError(
                f"Unexpected reference shape "
                f"{reference.shape}:\n"
                f"{self.reference_paths[index]}"
            )

        if search.shape != (
            SEARCH_SIZE,
            SEARCH_SIZE
        ):
            raise RuntimeError(
                f"Unexpected search shape "
                f"{search.shape}:\n"
                f"{self.search_paths[index]}"
            )

        reference = cv2.resize(
            reference,
            (
                REFERENCE_MODEL_SIZE,
                REFERENCE_MODEL_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        reference = (
            torch.from_numpy(
                reference.astype(
                    np.float32
                ) / 255.0
            )
            .unsqueeze(0)
        )

        search = (
            torch.from_numpy(
                search.astype(
                    np.float32
                ) / 255.0
            )
            .unsqueeze(0)
        )

        coordinates = self.all_coordinates[index]

        negative_indices = [
            i
            for i in range(16)
            if i != self.target_indices[index]
        ]

        negative_centers = []

        for i in negative_indices:
            negative_centers.append(
                [
                    coordinates[i]["center_x"],
                    coordinates[i]["center_y"]
                ]
            )

        negative_centers = torch.tensor(
            negative_centers,
            dtype=torch.float32
        )

        return {
            "reference": reference,
            "search": search,
            "center_x": torch.tensor(
                self.center_x[index],
                dtype=torch.float32
            ),
            "center_y": torch.tensor(
                self.center_y[index],
                dtype=torch.float32
            ),
            "negative_centers": negative_centers
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

        self.encoder = nn.Sequential(
            ConvBlock(1, 16, 1),
            ConvBlock(16, 32, 2),
            ConvBlock(32, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 128, 1)
        )

    def forward(self, x):
        return self.encoder(x)


# ============================================================
# CROSS CORRELATION
# ============================================================

def depthwise_cross_correlation(
    search_features,
    reference_features
):
    batch_size, channels, _, _ = (
        search_features.shape
    )

    search_grouped = search_features.reshape(
        1,
        batch_size * channels,
        search_features.shape[-2],
        search_features.shape[-1]
    )

    reference_grouped = reference_features.reshape(
        batch_size * channels,
        1,
        reference_features.shape[-2],
        reference_features.shape[-1]
    )

    correlation = F.conv2d(
        search_grouped,
        reference_grouped,
        groups=batch_size * channels
    )

    correlation = correlation.reshape(
        batch_size,
        channels,
        correlation.shape[-2],
        correlation.shape[-1]
    )

    return correlation.mean(dim=1)


# ============================================================
# HEATMAP HEAD
# ============================================================

class HeatmapHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, correlation):

        x = correlation.unsqueeze(1)

        x = self.layers(x)

        return F.interpolate(
            x,
            size=(
                HEATMAP_SIZE,
                HEATMAP_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )


# ============================================================
# MODEL B
# ============================================================

class ModelB(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SharedEncoder()

        self.heatmap_head = HeatmapHead()

    def encode(
        self,
        reference,
        search
    ):
        reference_features = self.encoder(
            reference
        )

        search_features = self.encoder(
            search
        )

        return (
            reference_features,
            search_features
        )

    def correlate(
        self,
        reference_features,
        search_features
    ):
        return depthwise_cross_correlation(
            search_features,
            reference_features
        )

    def forward(
        self,
        reference,
        search
    ):
        reference_features, search_features = (
            self.encode(
                reference,
                search
            )
        )

        correlation = self.correlate(
            reference_features,
            search_features
        )

        heatmap_logits = self.heatmap_head(
            correlation
        )

        return (
            heatmap_logits,
            reference_features,
            search_features,
            correlation
        )


# ============================================================
# GAUSSIAN TARGET
# ============================================================

def make_gaussian_target(
    center_x,
    center_y,
    height,
    width,
    sigma
):
    batch_size = center_x.shape[0]

    yy = torch.arange(
        height,
        device=center_x.device,
        dtype=torch.float32
    ).view(
        1,
        height,
        1
    )

    xx = torch.arange(
        width,
        device=center_x.device,
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
        (xx - cx) ** 2
        + (yy - cy) ** 2
    )

    return torch.exp(
        -distance_squared
        / (
            2.0
            * sigma
            * sigma
        )
    )


# ============================================================
# FOCAL LOSS
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

    positive = target.eq(
        1.0
    )

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
        * (~positive).float()
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

    return (
        (
            positive_loss
            + negative_loss
        )
        .sum(
            dim=(1, 2, 3)
        )
        / positive_count
    ).mean()


# ============================================================
# SOFT ARGMAX
# ============================================================

def soft_argmax_2d(
    logits
):
    batch_size, _, height, width = (
        logits.shape
    )

    values = logits[:, 0].float()

    probabilities = torch.softmax(
        values.reshape(
            batch_size,
            height * width
        ),
        dim=1
    )

    x_coordinates = torch.linspace(
        0.0,
        1.0,
        width,
        device=logits.device
    )

    y_coordinates = torch.linspace(
        0.0,
        1.0,
        height,
        device=logits.device
    )

    x_grid = (
        x_coordinates
        .view(1, 1, width)
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
        y_coordinates
        .view(1, height, 1)
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
    predicted_x,
    predicted_y,
    ground_truth_x,
    ground_truth_y
):
    target_x = (
        ground_truth_x
        / SEARCH_SIZE
    )

    target_y = (
        ground_truth_y
        / SEARCH_SIZE
    )

    return (
        F.smooth_l1_loss(
            predicted_x,
            target_x
        )
        +
        F.smooth_l1_loss(
            predicted_y,
            target_y
        )
    )


# ============================================================
# HARD NEGATIVE SCORE
# ============================================================

def sample_block_score(
    search_features,
    reference_features,
    center_x,
    center_y
):
    """
    Samples the correlation map at the expected center of a block.

    Search feature stride is approximately 8 pixels.
    Correlation map positions correspond to template placements.

    For the current Model-A geometry, we use center/8 as the
    sampling coordinate. This keeps Model B aligned with Model A
    for the requested ablation.
    """

    correlation = depthwise_cross_correlation(
        search_features,
        reference_features
    )

    height = correlation.shape[-2]
    width = correlation.shape[-1]

    x = (
        center_x
        / SEARCH_SIZE
        * (width - 1)
    )

    y = (
        center_y
        / SEARCH_SIZE
        * (height - 1)
    )

    x = x.round().long().clamp(
        0,
        width - 1
    )

    y = y.round().long().clamp(
        0,
        height - 1
    )

    batch_indices = torch.arange(
        correlation.shape[0],
        device=correlation.device
    )

    return correlation[
        batch_indices,
        y,
        x
    ]


def infonce_hard_negative_loss(
    search_features,
    reference_features,
    positive_x,
    positive_y,
    negative_centers,
    temperature
):
    """
    Positive score:
        correlation score at target block.

    Negative scores:
        correlation score at all 15 other blocks.

    InfoNCE:
        -log exp(positive/tau) /
             [exp(positive/tau) + sum exp(negative/tau)]
    """

    correlation = depthwise_cross_correlation(
        search_features,
        reference_features
    )

    batch_size = correlation.shape[0]

    height = correlation.shape[-2]
    width = correlation.shape[-1]

    def score_at_xy(x, y):

        x_index = (
            x
            / SEARCH_SIZE
            * (width - 1)
        ).round().long().clamp(
            0,
            width - 1
        )

        y_index = (
            y
            / SEARCH_SIZE
            * (height - 1)
        ).round().long().clamp(
            0,
            height - 1
        )

        batch_indices = torch.arange(
            batch_size,
            device=correlation.device
        )

        return correlation[
            batch_indices,
            y_index,
            x_index
        ]

    positive_score = score_at_xy(
        positive_x,
        positive_y
    )

    negative_scores = []

    for negative_index in range(
        negative_centers.shape[1]
    ):
        negative_x = (
            negative_centers[
                :,
                negative_index,
                0
            ]
        )

        negative_y = (
            negative_centers[
                :,
                negative_index,
                1
            ]
        )

        negative_scores.append(
            score_at_xy(
                negative_x,
                negative_y
            )
        )

    negative_scores = torch.stack(
        negative_scores,
        dim=1
    )

    logits = torch.cat(
        [
            positive_score.unsqueeze(1),
            negative_scores
        ],
        dim=1
    )

    labels = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=correlation.device
    )

    return F.cross_entropy(
        logits / temperature,
        labels
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predicted_x,
    predicted_y,
    ground_truth_x,
    ground_truth_y
):
    error_x = (
        predicted_x
        - ground_truth_x
    )

    error_y = (
        predicted_y
        - ground_truth_y
    )

    distance = torch.sqrt(
        error_x ** 2
        + error_y ** 2
    )

    return {
        "mean_error": float(
            distance.mean()
        ),
        "median_error": float(
            distance.median()
        ),
        "mae_x": float(
            error_x.abs().mean()
        ),
        "mae_y": float(
            error_y.abs().mean()
        ),
        "within_1": float(
            (distance <= 1)
            .float()
            .mean()
            * 100
        ),
        "within_2": float(
            (distance <= 2)
            .float()
            .mean()
            * 100
        ),
        "within_5": float(
            (distance <= 5)
            .float()
            .mean()
            * 100
        ),
        "within_10": float(
            (distance <= 10)
            .float()
            .mean()
            * 100
        )
    }


# ============================================================
# PHASE 2 LAMBDA
# ============================================================

def get_infonce_lambda(
    phase,
    epoch,
    ramp_epochs
):
    if phase == 1:
        return 0.0

    if phase == 2:
        if ramp_epochs <= 0:
            return 1.0

        return min(
            1.0,
            float(epoch - 1)
            / float(ramp_epochs)
        )

    return 1.0


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    infonce_lambda=0.0
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_coordinate_loss = 0.0
    total_infonce_loss = 0.0

    total_samples = 0

    all_predicted_x = []
    all_predicted_y = []
    all_ground_truth_x = []
    all_ground_truth_y = []

    use_amp = (
        device.type == "cuda"
    )

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

        ground_truth_x = batch[
            "center_x"
        ].to(
            device,
            non_blocking=True
        )

        ground_truth_y = batch[
            "center_y"
        ].to(
            device,
            non_blocking=True
        )

        negative_centers = batch[
            "negative_centers"
        ].to(
            device,
            non_blocking=True
        )

        if use_amp:

            reference = reference.contiguous(
                memory_format=torch.channels_last
            )

            search = search.contiguous(
                memory_format=torch.channels_last
            )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp
            ):

                (
                    logits,
                    reference_features,
                    search_features,
                    correlation
                ) = model(
                    reference,
                    search
                )

                heatmap_x = (
                    ground_truth_x
                    * HEATMAP_SIZE
                    / SEARCH_SIZE
                )

                heatmap_y = (
                    ground_truth_y
                    * HEATMAP_SIZE
                    / SEARCH_SIZE
                )

                target = (
                    make_gaussian_target(
                        heatmap_x,
                        heatmap_y,
                        HEATMAP_SIZE,
                        HEATMAP_SIZE,
                        GAUSSIAN_SIGMA
                    )
                    .unsqueeze(1)
                )

                heat_loss = (
                    focal_heatmap_loss(
                        logits,
                        target
                    )
                )

                predicted_x_normalized, predicted_y_normalized = (
                    soft_argmax_2d(
                        logits
                    )
                )

                coord_loss = (
                    coordinate_loss(
                        predicted_x_normalized,
                        predicted_y_normalized,
                        ground_truth_x,
                        ground_truth_y
                    )
                )

                if infonce_lambda > 0.0:

                    infonce_loss = (
                        infonce_hard_negative_loss(
                            search_features,
                            reference_features,
                            ground_truth_x,
                            ground_truth_y,
                            negative_centers,
                            INFONCE_TEMPERATURE
                        )
                    )

                else:

                    infonce_loss = (
                        torch.zeros(
                            (),
                            device=device
                        )
                    )

                loss = (
                    heat_loss
                    +
                    COORDINATE_LOSS_WEIGHT
                    * coord_loss
                    +
                    infonce_lambda
                    * infonce_loss
                )

            if training:

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

        batch_size = reference.shape[0]

        total_loss += (
            float(
                loss.detach()
            )
            * batch_size
        )

        total_heatmap_loss += (
            float(
                heat_loss.detach()
            )
            * batch_size
        )

        total_coordinate_loss += (
            float(
                coord_loss.detach()
            )
            * batch_size
        )

        total_infonce_loss += (
            float(
                infonce_loss.detach()
            )
            * batch_size
        )

        total_samples += batch_size

        all_predicted_x.append(
            (
                predicted_x_normalized
                .detach()
                .float()
                .cpu()
                * SEARCH_SIZE
            )
        )

        all_predicted_y.append(
            (
                predicted_y_normalized
                .detach()
                .float()
                .cpu()
                * SEARCH_SIZE
            )
        )

        all_ground_truth_x.append(
            ground_truth_x
            .detach()
            .float()
            .cpu()
        )

        all_ground_truth_y.append(
            ground_truth_y
            .detach()
            .float()
            .cpu()
        )

    predicted_x = torch.cat(
        all_predicted_x
    )

    predicted_y = torch.cat(
        all_predicted_y
    )

    ground_truth_x = torch.cat(
        all_ground_truth_x
    )

    ground_truth_y = torch.cat(
        all_ground_truth_y
    )

    metrics = calculate_metrics(
        predicted_x,
        predicted_y,
        ground_truth_x,
        ground_truth_y
    )

    metrics["loss"] = (
        total_loss
        / total_samples
    )

    metrics["heatmap_loss"] = (
        total_heatmap_loss
        / total_samples
    )

    metrics["coordinate_loss"] = (
        total_coordinate_loss
        / total_samples
    )

    metrics["infonce_loss"] = (
        total_infonce_loss
        / total_samples
    )

    return metrics


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_error,
    phase
):
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scaler_state_dict":
                scaler.state_dict(),
            "best_error":
                best_error
        },
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
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

    scaler.load_state_dict(
        checkpoint[
            "scaler_state_dict"
        ]
    )

    return (
        checkpoint["epoch"] + 1,
        checkpoint.get(
            "best_error",
            float("inf")
        ),
        checkpoint.get(
            "phase",
            1
        )
    )


# ============================================================
# DATALOADER
# ============================================================

def create_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    device
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": (
            device.type == "cuda"
        )
    }

    if num_workers > 0:

        kwargs[
            "persistent_workers"
        ] = True

        kwargs[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        **kwargs
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=1
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
        "--lambda-ramp-epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed(SEED)

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

    if device.type == "cuda":

        torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True

        torch.backends.cudnn.allow_tf32 = True

    data_root = Path(
        args.data_root
    )

    train_csv = (
        data_root
        / "training_metadata"
        / "train.csv"
    )

    validation_csv = (
        data_root
        / "training_metadata"
        / "validation.csv"
    )

    reference_dir = (
        data_root
        / "reference"
    )

    search_dir = (
        data_root
        / "search"
    )

    output_dir = (
        data_root
        / "model_B"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    last_checkpoint = (
        output_dir
        / "model_B_last.pt"
    )

    best_checkpoint = (
        output_dir
        / "model_B_best.pt"
    )

    print("=" * 76)
    print(
        "SEMICON / DRIFT-SENSE"
    )
    print(
        "MODEL B - HARD NEGATIVE INFONCE"
    )
    print("=" * 76)

    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Data root    : {data_root}"
    )

    print(
        f"Phase        : {args.phase}"
    )

    print(
        f"Epochs       : {args.epochs}"
    )

    print(
        f"Batch size   : {args.batch_size}"
    )

    print(
        f"Workers      : {args.num_workers}"
    )

    print(
        f"InfoNCE tau  : "
        f"{INFONCE_TEMPERATURE}"
    )

    train_dataset = (
        Generator5HardNegativeDataset(
            train_csv,
            reference_dir,
            search_dir
        )
    )

    validation_dataset = (
        Generator5HardNegativeDataset(
            validation_csv,
            reference_dir,
            search_dir
        )
    )

    print(
        f"Training samples  : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        "Hard negatives per sample: 15"
    )

    model = ModelB().to(
        device
    )

    if device.type == "cuda":

        model = model.to(
            memory_format=torch.channels_last
        )

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    train_loader = create_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        device
    )

    validation_loader = create_loader(
        validation_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        )
    )

    start_epoch = 1

    best_error = float(
        "inf"
    )

    if args.resume:

        if not last_checkpoint.exists():

            raise FileNotFoundError(
                "No Model B checkpoint found:\n"
                f"{last_checkpoint}"
            )

        (
            start_epoch,
            best_error,
            saved_phase
        ) = load_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scaler,
            device
        )

        print(
            f"Resuming from epoch "
            f"{start_epoch}"
        )

        print(
            f"Checkpoint phase: "
            f"{saved_phase}"
        )

    print()
    print("=" * 76)
    print(
        "MODEL B TRAINING"
    )
    print("=" * 76)

    for epoch in range(
        start_epoch,
        args.epochs + 1
    ):

        epoch_start = time.perf_counter()

        infonce_lambda = (
            get_infonce_lambda(
                args.phase,
                epoch,
                args.lambda_ramp_epochs
            )
        )

        print()
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )

        print(
            f"InfoNCE lambda : "
            f"{infonce_lambda:.3f}"
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            infonce_lambda
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            None,
            None,
            infonce_lambda
        )

        if device.type == "cuda":

            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - epoch_start
        )

        print()
        print(
            f"Train loss       : "
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
            f"Train InfoNCE    : "
            f"{train_metrics['infonce_loss']:.6f}"
        )

        print(
            f"Train mean error : "
            f"{train_metrics['mean_error']:.3f} px"
        )

        print()
        print(
            f"Val loss         : "
            f"{validation_metrics['loss']:.6f}"
        )

        print(
            f"Val heatmap      : "
            f"{validation_metrics['heatmap_loss']:.6f}"
        )

        print(
            f"Val coordinate   : "
            f"{validation_metrics['coordinate_loss']:.6f}"
        )

        print(
            f"Val InfoNCE      : "
            f"{validation_metrics['infonce_loss']:.6f}"
        )

        print(
            f"Val mean error   : "
            f"{validation_metrics['mean_error']:.3f} px"
        )

        print(
            f"Val median error : "
            f"{validation_metrics['median_error']:.3f} px"
        )

        print(
            f"Val <= 1 px      : "
            f"{validation_metrics['within_1']:.2f}%"
        )

        print(
            f"Val <= 2 px      : "
            f"{validation_metrics['within_2']:.2f}%"
        )

        print(
            f"Val <= 5 px      : "
            f"{validation_metrics['within_5']:.2f}%"
        )

        print(
            f"Val <= 10 px     : "
            f"{validation_metrics['within_10']:.2f}%"
        )

        print(
            f"Epoch time       : "
            f"{elapsed / 60:.2f} min"
        )

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scaler,
            epoch,
            best_error,
            args.phase
        )

        print(
            "Latest checkpoint saved."
        )

        if (
            validation_metrics[
                "mean_error"
            ]
            < best_error
        ):

            best_error = (
                validation_metrics[
                    "mean_error"
                ]
            )

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scaler,
                epoch,
                best_error,
                args.phase
            )

            print(
                "Best checkpoint saved."
            )

    print()
    print("=" * 76)
    print(
        "MODEL B TRAINING COMPLETE"
    )
    print("=" * 76)

    print(
        f"Best validation error: "
        f"{best_error:.3f} px"
    )

    print(
        "Checkpoints:"
    )

    print(
        f"  {last_checkpoint}"
    )

    print(
        f"  {best_checkpoint}"
    )


if __name__ == "__main__":
    main()
