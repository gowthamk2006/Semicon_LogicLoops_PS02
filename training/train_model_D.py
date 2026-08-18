from pathlib import Path
import argparse
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

DATASET_ROOT = Path(
    "/content/drive/MyDrive/semicon/generator5_augmented"
)

TRAIN_CSV = DATASET_ROOT / "training_metadata" / "train.csv"
VAL_CSV = DATASET_ROOT / "training_metadata" / "validation.csv"

REFERENCE_DIR = DATASET_ROOT / "reference"
SEARCH_DIR = DATASET_ROOT / "search"

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/semicon/checkpoints/model_D_20epoch"
)

IMAGE_SIZE = 1000
MODEL_IMAGE_SIZE = 128

SEED = 20260818


# ============================================================
# SEED
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

class Generator5Dataset(Dataset):

    def __init__(
        self,
        csv_path,
        reference_dir,
        search_dir
    ):
        self.df = pd.read_csv(csv_path)

        self.reference_dir = Path(reference_dir)
        self.search_dir = Path(search_dir)

        required = [
            "reference_file",
            "search_file",
            "center_x",
            "center_y"
        ]

        missing = [
            c for c in required
            if c not in self.df.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing columns: {missing}"
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
                f"Could not read reference:\n"
                f"{reference_path}"
            )

        if search is None:
            raise RuntimeError(
                f"Could not read search:\n"
                f"{search_path}"
            )

        if reference.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Reference has shape "
                f"{reference.shape}, expected "
                f"(1000,1000)"
            )

        if search.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Search has shape "
                f"{search.shape}, expected "
                f"(1000,1000)"
            )

        # ----------------------------------------------------
        # Preserve spatial structure.
        # Both images are resized only for computational
        # efficiency. The target remains in original
        # 1000x1000 coordinates.
        # ----------------------------------------------------

        reference = cv2.resize(
            reference,
            (
                MODEL_IMAGE_SIZE,
                MODEL_IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        search = cv2.resize(
            search,
            (
                MODEL_IMAGE_SIZE,
                MODEL_IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        reference = (
            reference.astype(
                np.float32
            ) / 255.0
        )

        search = (
            search.astype(
                np.float32
            ) / 255.0
        )

        reference = torch.from_numpy(
            reference
        ).unsqueeze(0)

        search = torch.from_numpy(
            search
        ).unsqueeze(0)

        # Normalize original coordinates.
        x = (
            float(row["center_x"])
            / IMAGE_SIZE
        )

        y = (
            float(row["center_y"])
            / IMAGE_SIZE
        )

        target = torch.tensor(
            [x, y],
            dtype=torch.float32
        )

        return (
            reference,
            search,
            target
        )


# ============================================================
# SPATIAL ENCODER
# ============================================================

class SpatialEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                96,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                96,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# CORRELATION
# ============================================================

def depthwise_correlation(
    search,
    reference
):
    """
    Spatially preserves matching information.

    search:
        [B,C,H,W]

    reference:
        [B,C,H,W]

    Returns:
        [B,C,H,W]
    """

    # Normalize each feature vector at each spatial location.
    search = F.normalize(
        search,
        dim=1
    )

    reference = F.normalize(
        reference,
        dim=1
    )

    # Element-wise feature similarity at corresponding
    # spatial positions.
    correlation = (
        search * reference
    ).sum(
        dim=1,
        keepdim=True
    )

    return correlation


# ============================================================
# MODEL D
# ============================================================

class ModelD(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = SpatialEncoder()

        # Instead of global average pooling, keep the
        # spatial correlation map.

        self.correlation_head = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                kernel_size=1
            )
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

        correlation = depthwise_correlation(
            search_features,
            reference_features
        )

        score_map = self.correlation_head(
            correlation
        )

        return score_map


# ============================================================
# SOFT-ARGMAX
# ============================================================

def soft_argmax(
    score_map,
    temperature=0.05
):

    b, _, h, w = score_map.shape

    logits = (
        score_map
        .reshape(
            b,
            -1
        )
        / temperature
    )

    probabilities = F.softmax(
        logits,
        dim=1
    )

    ys = torch.linspace(
        0.0,
        1.0,
        h,
        device=score_map.device
    )

    xs = torch.linspace(
        0.0,
        1.0,
        w,
        device=score_map.device
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    xx = xx.reshape(
        1,
        -1
    )

    yy = yy.reshape(
        1,
        -1
    )

    pred_x = (
        probabilities * xx
    ).sum(
        dim=1
    )

    pred_y = (
        probabilities * yy
    ).sum(
        dim=1
    )

    return torch.stack(
        [
            pred_x,
            pred_y
        ],
        dim=1
    )


# ============================================================
# GAUSSIAN TARGET MAP
# ============================================================

def make_target_map(
    target,
    height,
    width,
    sigma=1.5
):

    device = target.device

    b = target.shape[0]

    x = (
        target[:, 0]
        * (width - 1)
    )

    y = (
        target[:, 1]
        * (height - 1)
    )

    grid_x = torch.arange(
        width,
        device=device,
        dtype=torch.float32
    ).view(
        1,
        1,
        width
    )

    grid_y = torch.arange(
        height,
        device=device,
        dtype=torch.float32
    ).view(
        1,
        height,
        1
    )

    cx = x.view(
        b,
        1,
        1
    )

    cy = y.view(
        b,
        1,
        1
    )

    distance = (
        (grid_x - cx) ** 2
        +
        (grid_y - cy) ** 2
    )

    target_map = torch.exp(
        -distance
        /
        (
            2.0
            * sigma
            * sigma
        )
    )

    return target_map.unsqueeze(1)


# ============================================================
# EPOCH
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
    total_samples = 0

    all_errors = []

    mse = nn.MSELoss()

    for reference, search, target in loader:

        reference = reference.to(
            device,
            non_blocking=True
        )

        search = search.to(
            device,
            non_blocking=True
        )

        target = target.to(
            device,
            non_blocking=True
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            score_map = model(
                reference,
                search
            )

            target_map = make_target_map(
                target,
                score_map.shape[-2],
                score_map.shape[-1]
            )

            # Normalize the score map through sigmoid
            # so the loss is numerically stable.
            prediction_map = torch.sigmoid(
                score_map
            )

            heatmap_loss = mse(
                prediction_map,
                target_map
            )

            prediction = soft_argmax(
                score_map
            )

            coordinate_loss = F.smooth_l1_loss(
                prediction,
                target
            )

            # Coordinate supervision is deliberately
            # important.
            loss = (
                coordinate_loss
                +
                2.0 * heatmap_loss
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

        batch_size = (
            reference.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        errors = torch.sqrt(
            torch.sum(
                (
                    prediction
                    - target
                ) ** 2,
                dim=1
            )
        ) * IMAGE_SIZE

        all_errors.append(
            errors.detach().cpu()
        )

    errors = torch.cat(
        all_errors
    )

    return {
        "loss":
            total_loss
            / total_samples,

        "mean_error":
            errors.mean().item(),

        "median_error":
            errors.median().item(),

        "within_1":
            (
                errors <= 1
            ).float().mean().item()
            * 100.0,

        "within_5":
            (
                errors <= 5
            ).float().mean().item()
            * 100.0,

        "within_10":
            (
                errors <= 10
            ).float().mean().item()
            * 100.0
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics
):

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "metrics":
                metrics,

            "model":
                "Model D spatial correlation localization",

            "image_size":
                IMAGE_SIZE,

            "model_image_size":
                MODEL_IMAGE_SIZE
        },
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=2
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("SEMICON / DRIFT-SENSE")
    print("MODEL D - SPATIAL CORRELATION LOCALIZATION")
    print("=" * 72)

    print(
        "Device       :",
        device
    )

    if device.type == "cuda":

        print(
            "GPU          :",
            torch.cuda.get_device_name(0)
        )

    print(
        "Input        : 1000 x 1000"
    )

    print(
        "Internal     : 128 x 128"
    )

    print(
        "Output       : spatial score map"
    )

    print(
        "Decoder      : soft-argmax"
    )

    print(
        "Epochs       :",
        args.epochs
    )

    print(
        "Batch size   :",
        args.batch_size
    )

    print()

    train_dataset = Generator5Dataset(
        TRAIN_CSV,
        REFERENCE_DIR,
        SEARCH_DIR
    )

    val_dataset = Generator5Dataset(
        VAL_CSV,
        REFERENCE_DIR,
        SEARCH_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model = ModelD().to(
        device
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_error = float("inf")

    for epoch in range(
        1,
        args.epochs + 1
    ):

        start = time.time()

        print()
        print("=" * 72)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("=" * 72)

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device
        )

        elapsed = (
            time.time() - start
        )

        print()

        print(
            f"Train loss       : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Train mean error : "
            f"{train_metrics['mean_error']:.2f} px"
        )

        print(
            f"Val loss         : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Val mean error   : "
            f"{val_metrics['mean_error']:.2f} px"
        )

        print(
            f"Val median error : "
            f"{val_metrics['median_error']:.2f} px"
        )

        print(
            f"Val <= 5 px      : "
            f"{val_metrics['within_5']:.2f}%"
        )

        print(
            f"Val <= 10 px     : "
            f"{val_metrics['within_10']:.2f}%"
        )

        print(
            f"Epoch time       : "
            f"{elapsed:.1f} sec"
        )

        if (
            val_metrics["mean_error"]
            < best_error
        ):

            best_error = (
                val_metrics[
                    "mean_error"
                ]
            )

            save_checkpoint(
                OUTPUT_DIR
                / "model_D_best.pt",
                model,
                optimizer,
                epoch,
                val_metrics
            )

            print(
                "Best checkpoint saved."
            )

    save_checkpoint(
        OUTPUT_DIR
        / "model_D_last.pt",
        model,
        optimizer,
        args.epochs,
        val_metrics
    )

    print()
    print("=" * 72)
    print("MODEL D TRAINING COMPLETE")
    print("=" * 72)

    print(
        "Best validation error:",
        f"{best_error:.2f} px"
    )

    print(
        "Best checkpoint:",
        OUTPUT_DIR / "model_D_best.pt"
    )

    print(
        "Last checkpoint:",
        OUTPUT_DIR / "model_D_last.pt"
    )


if __name__ == "__main__":
    main()