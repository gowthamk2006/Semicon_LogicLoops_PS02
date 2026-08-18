"""
SEMICON / DRIFT-SENSE
MODEL D STANDALONE LOCALIZATION INFERENCE

This inference script imports the EXACT ModelD implementation from
train_model_D.py, then loads models/model.pt.

Expected repository layout:

semicon_submission/
├── models/
│   └── model.pt
├── training/
│   └── train_model_D.py
└── inference/
    └── localize.py

Usage:

python inference\localize.py ^
  --reference "path\to\reference.png" ^
  --search "path\to\search.png"

The reference and search images must be 1000 x 1000 pixels.

The output is the predicted target center in the original
1000 x 1000 image coordinate system.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


RAW_SIZE = 1000
INTERNAL_SIZE = 128

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

DEFAULT_MODEL = (
    REPO_ROOT
    / "models"
    / "model.pt"
)

TRAIN_MODEL_D = (
    REPO_ROOT
    / "training"
    / "train_model_D.py"
)


def load_original_model_class():

    if not TRAIN_MODEL_D.exists():
        raise FileNotFoundError(
            "The exact Model D training file was not found:\n"
            f"{TRAIN_MODEL_D}\n\n"
            "Place train_model_D.py inside the repository's "
            "training folder."
        )

    spec = importlib.util.spec_from_file_location(
        "semicon_original_model_D",
        str(TRAIN_MODEL_D),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load train_model_D.py."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "ModelD"):
        raise RuntimeError(
            "train_model_D.py does not contain a ModelD class."
        )

    return module.ModelD


def read_image(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{path}"
        )

    if image.shape != (
        RAW_SIZE,
        RAW_SIZE,
    ):
        raise ValueError(
            f"Expected a {RAW_SIZE} x {RAW_SIZE} image, "
            f"but received {image.shape}:\n{path}"
        )

    image = cv2.resize(
        image,
        (
            INTERNAL_SIZE,
            INTERNAL_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    return torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)


def load_model(
    model_class,
    model_path,
    device,
):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found:\n"
            f"{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model = model_class().to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


def extract_score_map(output):

    # Different training scripts sometimes return a tuple.
    # The first tensor is the spatial score map.
    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise RuntimeError(
            "Model D did not return a tensor score map."
        )

    if output.ndim == 3:
        output = output.unsqueeze(1)

    if output.ndim != 4:
        raise RuntimeError(
            "Unexpected Model D output shape: "
            f"{tuple(output.shape)}"
        )

    return output


def soft_argmax(
    score_map,
    temperature=0.05,
):

    batch, _, height, width = (
        score_map.shape
    )

    probabilities = F.softmax(
        score_map.reshape(
            batch,
            -1,
        )
        / temperature,
        dim=1,
    )

    yy, xx = torch.meshgrid(
        torch.arange(
            height,
            device=score_map.device,
            dtype=torch.float32,
        ),
        torch.arange(
            width,
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

    # Convert the internal score-map coordinate to the
    # original 1000 x 1000 image coordinate system.
    if width > 1:
        px = (
            px
            / (width - 1)
            * (RAW_SIZE - 1)
        )

    if height > 1:
        py = (
            py
            / (height - 1)
            * (RAW_SIZE - 1)
        )

    return torch.stack(
        [px, py],
        dim=1,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "SEMICON Model D localization inference."
        )
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
        default=str(DEFAULT_MODEL),
        help=(
            "Path to the trained Model D checkpoint. "
            "Default: ../models/model.pt"
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help=(
            "Soft-argmax temperature. "
            "Default: 0.05"
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "SEMICON / DRIFT-SENSE"
    )
    print(
        "MODEL D LOCALIZATION INFERENCE"
    )
    print("=" * 72)

    print(
        f"Reference : {args.reference}"
    )

    print(
        f"Search    : {args.search}"
    )

    print(
        f"Model     : {args.model}"
    )

    print(
        f"Input     : {RAW_SIZE} x {RAW_SIZE} px"
    )

    print(
        "Method    : Model D spatial correlation"
    )

    print(
        "Decoder   : Soft-Argmax"
    )

    print()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device    : {device}"
    )

    print()

    try:

        reference = read_image(
            args.reference
        ).to(device)

        search = read_image(
            args.search
        ).to(device)

        ModelD = load_original_model_class()

        model = load_model(
            ModelD,
            args.model,
            device,
        )

        with torch.no_grad():

            output = model(
                reference,
                search,
            )

            score_map = extract_score_map(
                output
            )

            prediction = soft_argmax(
                score_map,
                temperature=args.temperature,
            )

        predicted_x = float(
            prediction[0, 0].cpu()
        )

        predicted_y = float(
            prediction[0, 1].cpu()
        )

    except Exception as exc:

        print(
            "INFERENCE FAILED"
        )

        print("-" * 72)

        print(
            str(exc)
        )

        return 1

    print(
        "RESULT"
    )

    print("-" * 72)

    print(
        f"Predicted center X : "
        f"{predicted_x:.2f} px"
    )

    print(
        f"Predicted center Y : "
        f"{predicted_y:.2f} px"
    )

    print()

    print("=" * 72)

    print(
        "LOCALIZATION COMPLETE"
    )

    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())