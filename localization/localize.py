"""
SEMICON / DRIFT-SENSE
MODEL D SELF-CONTAINED LOCALIZATION INFERENCE

This file contains the exact Model D architecture used for the supplied
checkpoint. It does NOT require train_model_D.py or any training files.

Required files:
    localize.py
    model.pt

Usage:
    python localize.py ^
      --reference "D:\reference.png" ^
      --search "D:\search.png" ^
      --model "D:\model.pt"

Reference and search images must be 1000 x 1000 pixels.
The predicted center is reported in the original 1000 x 1000 coordinate
system.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAW_SIZE = 1000
INTERNAL_SIZE = 128


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




def read_image(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found:\n{path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Could not read image:\n{path}")

    if image.shape != (RAW_SIZE, RAW_SIZE):
        raise ValueError(
            f"Expected a {RAW_SIZE} x {RAW_SIZE} image, "
            f"but received {image.shape}:\n{path}"
        )

    image = cv2.resize(
        image,
        (INTERNAL_SIZE, INTERNAL_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32) / 255.0

    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)


def load_model(model_path, device):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = ModelD().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="SEMICON Model D self-contained localization inference."
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the 1000 x 1000 reference image.",
    )

    parser.add_argument(
        "--search",
        required=True,
        help="Path to the 1000 x 1000 search image.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained Model D checkpoint.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Soft-argmax temperature. Default: 0.05",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("SEMICON / DRIFT-SENSE")
    print("MODEL D LOCALIZATION INFERENCE")
    print("=" * 72)
    print(f"Reference : {args.reference}")
    print(f"Search    : {args.search}")
    print(f"Model     : {args.model}")
    print(f"Input     : {RAW_SIZE} x {RAW_SIZE} px")
    print("Method    : Model D spatial correlation")
    print("Decoder   : Soft-Argmax")
    print()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device    : {device}")
    print()

    try:
        reference = read_image(args.reference).to(device)
        search = read_image(args.search).to(device)

        model = load_model(args.model, device)

        with torch.no_grad():
            score_map = model(reference, search)
            prediction = soft_argmax(
                score_map,
                temperature=args.temperature,
            )

        # soft_argmax returns normalized coordinates in [0, 1].
        # Convert them back to the original 1000 x 1000 pixel coordinates.
        predicted_x = float(prediction[0, 0].cpu()) * (RAW_SIZE - 1)
        predicted_y = float(prediction[0, 1].cpu()) * (RAW_SIZE - 1)

    except Exception as exc:
        print("INFERENCE FAILED")
        print("-" * 72)
        print(str(exc))
        return 1

    print("RESULT")
    print("-" * 72)
    print(f"Predicted center X : {predicted_x:.2f} px")
    print(f"Predicted center Y : {predicted_y:.2f} px")
    print()
    print("=" * 72)
    print("LOCALIZATION COMPLETE")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
