"""
SEMICON / DRIFT-SENSE
GENERATOR 5 DATA AUGMENTATION

Input:
    E:\semicon\generator5_datasets

Expected input:
    reference\
    search\
    visualization\
    metadata.json

Output:
    E:\semicon\generator5_augmented\
        reference\
        search\
        visualization\
        metadata.json
        metadata.csv

Seven augmentation families:
    1. rotation
    2. shear
    3. drift
    4. noise
    5. blur
    6. brightness_contrast
    7. scaling

IMPORTANT:
Generator5 already contains physically generated reference/search pairs.

For each augmentation:
    - the SEARCH image is also augmented
    - the search degradation is slightly stronger
    - the REFERENCE receives its own, slightly weaker capture variation
    - geometric transforms keep the target relationship consistent
    - noise is generated independently for search and reference
    - drift changes the relative landing position
    - the original search/reference files are never overwritten

The resulting data is intended to preserve the Generator5
"physical scene first" character while increasing robustness.
"""

from pathlib import Path
import argparse
import json
import csv
import math
import shutil

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

from PIL import (
    Image,
    ImageFilter,
    ImageEnhance,
    ImageDraw
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = Path(r"E:\semicon\generator5_datasets")
OUTPUT_ROOT = Path(r"E:\semicon\generator5_augmented")

SEARCH_SIZE = 1000
REFERENCE_SIZE = 1000

SEED = 20260816

AUGMENTATIONS = [
    "rotation",
    "shear",
    "drift",
    "noise",
    "blur",
    "brightness_contrast",
    "scaling",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(value, high))


def load_gray(path):
    return Image.open(path).convert("L")


def rotate_image(image, angle):
    fill = int(np.asarray(image).mean())

    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill
    )


def shear_image(image, shear_deg):
    """Pure horizontal shear only. No rotation is applied."""
    shear = math.tan(math.radians(float(shear_deg)))
    width, height = image.size
    fill = int(np.asarray(image).mean())

    # x_out = x_in + shear * (y - cy).
    # Inverse mapping for PIL: x_in = x_out - shear*(y-cy).
    matrix = (
        1.0,
        -shear,
        shear * height / 2.0,
        0.0,
        1.0,
        0.0
    )

    return image.transform(
        (width, height),
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill
    )


def drift_warp(image, dx, dy, smoothness=0.18, seed=0):
    """
    Smooth accumulated scan/stage drift.

    Unlike rotation, the whole frame is not turned. Unlike shear, the
    displacement is not constant/linear. Each scan row is displaced by a
    slowly varying amount, which produces a realistic gradual drift of the
    structures across the field.
    """
    if cv2 is None:
        fill = int(np.asarray(image).mean())
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, -float(dx), 0, 1, -float(dy)),
            resample=Image.Resampling.BICUBIC,
            fillcolor=fill
        )

    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape
    rng = np.random.default_rng(int(seed))

    n = 11
    yp = np.linspace(0, h - 1, n, dtype=np.float32)
    x_ctrl = np.cumsum(rng.normal(0, smoothness, n)).astype(np.float32)
    y_ctrl = np.cumsum(rng.normal(0, smoothness * 0.7, n)).astype(np.float32)
    x_ctrl -= x_ctrl.mean()
    y_ctrl -= y_ctrl.mean()

    # Drift accumulates with scan position. The displacement is therefore
    # different at different rows, rather than being a rigid translation.
    row_x = np.interp(np.arange(h), yp, x_ctrl).astype(np.float32)
    row_y = np.interp(np.arange(h), yp, y_ctrl).astype(np.float32)
    row_x += np.linspace(0.0, float(dx), h, dtype=np.float32)
    row_y += np.linspace(0.0, float(dy), h, dtype=np.float32)

    xx, yy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = xx - row_x[:, None]
    map_y = yy - row_y[:, None]

    fill = float(arr.mean())
    warped = cv2.remap(
        arr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill
    )

    return Image.fromarray(warped.astype(np.uint8), 'L')


def drift_point(x, y, width, height, dx, dy, seed=0, smoothness=0.65):
    """Evaluate the same row-dependent drift field used by drift_warp."""
    rng = np.random.default_rng(int(seed))
    n = 11
    yp = np.linspace(0, height - 1, n, dtype=np.float32)
    xc = np.cumsum(rng.normal(0, smoothness, n)).astype(np.float32)
    yc = np.cumsum(rng.normal(0, smoothness * 0.7, n)).astype(np.float32)
    xc -= xc.mean()
    yc -= yc.mean()
    xd = float(np.interp(y, yp, xc)) + float(dx) * float(y) / max(height-1,1)
    yd = float(np.interp(y, yp, yc)) + float(dy) * float(y) / max(height-1,1)
    return x + xd, y + yd


def transform_box_drift(x, y, w, h, image_w, image_h, dx, dy, seed=0):
    corners = [
        drift_point(x, y, image_w, image_h, dx, dy, seed, 0.65),
        drift_point(x+w, y, image_w, image_h, dx, dy, seed, 0.65),
        drift_point(x, y+h, image_w, image_h, dx, dy, seed, 0.65),
        drift_point(x+w, y+h, image_w, image_h, dx, dy, seed, 0.65),
    ]
    xs=[c[0] for c in corners]; ys=[c[1] for c in corners]
    min_x=max(0.0,min(xs)); min_y=max(0.0,min(ys))
    max_x=min(float(image_w),max(xs)); max_y=min(float(image_h),max(ys))
    return min_x,min_y,max(1.0,max_x-min_x),max(1.0,max_y-min_y)


def scale_image(image, factor):
    width, height = image.size

    new_width = max(
        100,
        int(round(width * factor))
    )

    new_height = max(
        100,
        int(round(height * factor))
    )

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC
    )

    background = int(np.asarray(image).mean())

    output = Image.new(
        "L",
        (width, height),
        background
    )

    if new_width >= width:
        left = (new_width - width) // 2
        top = (new_height - height) // 2

        resized = resized.crop((
            left,
            top,
            left + width,
            top + height
        ))

        output.paste(
            resized,
            (0, 0)
        )

    else:
        left = (width - new_width) // 2
        top = (height - new_height) // 2

        output.paste(
            resized,
            (left, top)
        )

    return output


# ============================================================
# GEOMETRIC COORDINATE TRANSFORM
# ============================================================

def transform_point(
    x,
    y,
    width,
    height,
    rotation_deg=0.0,
    shear_deg=0.0,
    scale_factor=1.0
):
    """
    Transform a point around the image centre.

    Order:
        scale -> shear -> rotation
    """

    cx = width / 2.0
    cy = height / 2.0

    px = x - cx
    py = y - cy

    # Scaling around centre.
    px *= scale_factor
    py *= scale_factor

    # Horizontal shear.
    shear = math.tan(
        math.radians(shear_deg)
    )

    px = px + shear * py

    # Rotation around centre.
    angle = math.radians(rotation_deg)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rx = cos_a * px - sin_a * py
    ry = sin_a * px + cos_a * py

    return (
        rx + cx,
        ry + cy
    )


def transform_box(
    x,
    y,
    width,
    height,
    image_width,
    image_height,
    rotation_deg=0.0,
    shear_deg=0.0,
    scale_factor=1.0
):
    corners = [
        (x, y),
        (x + width, y),
        (x, y + height),
        (x + width, y + height),
    ]

    transformed = [
        transform_point(
            px,
            py,
            image_width,
            image_height,
            rotation_deg,
            shear_deg,
            scale_factor
        )
        for px, py in corners
    ]

    min_x = clamp(
        min(p[0] for p in transformed),
        0,
        image_width
    )

    min_y = clamp(
        min(p[1] for p in transformed),
        0,
        image_height
    )

    max_x = clamp(
        max(p[0] for p in transformed),
        0,
        image_width
    )

    max_y = clamp(
        max(p[1] for p in transformed),
        0,
        image_height
    )

    return (
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y
    )


# ============================================================
# REFERENCE/SOURCE CAPTURE AUGMENTATIONS
# ============================================================

def augment_reference(
    reference,
    augmentation,
    rng
):
    """
    Reference receives slightly weaker capture variation.
    """

    params = {}

    if augmentation == "rotation":
        angle = float(
            rng.uniform(-5.0, 5.0)
        )

        reference = rotate_image(
            reference,
            angle
        )

        params["reference_rotation_deg"] = angle

    elif augmentation == "shear":
        angle = float(
            rng.uniform(-4.0, 4.0)
        )

        reference = shear_image(
            reference,
            angle
        )

        params["reference_shear_deg"] = angle

    elif augmentation == "noise":
        sigma = float(
            rng.uniform(6.0, 12.0)
        )

        image = np.asarray(
            reference,
            dtype=np.float32
        )

        noise = rng.normal(
            0.0,
            sigma,
            image.shape
        )

        reference = Image.fromarray(
            np.clip(
                image + noise,
                0,
                255
            ).astype(np.uint8),
            "L"
        )

        params["reference_noise_sigma"] = sigma

    elif augmentation == "blur":
        sigma = float(
            rng.uniform(0.4, 1.5)
        )

        reference = reference.filter(
            ImageFilter.GaussianBlur(sigma)
        )

        params["reference_blur_sigma"] = sigma

    elif augmentation == "brightness_contrast":
        brightness = float(
            rng.uniform(0.85, 1.15)
        )

        contrast = float(
            rng.uniform(0.75, 1.25)
        )

        reference = ImageEnhance.Brightness(
            reference
        ).enhance(brightness)

        reference = ImageEnhance.Contrast(
            reference
        ).enhance(contrast)

        params["reference_brightness"] = brightness
        params["reference_contrast"] = contrast

    elif augmentation == "scaling":
        factor = float(
            rng.uniform(0.97, 1.03)
        )

        reference = scale_image(
            reference,
            factor
        )

        params["reference_scale_factor"] = factor

    elif augmentation == "drift":
        dx = float(rng.uniform(-18.0, 18.0))
        dy = float(rng.uniform(-18.0, 18.0))
        seed = int(rng.integers(0, 2**32 - 1))
        reference = drift_warp(
            reference, dx, dy, smoothness=0.30, seed=seed
        )
        params["reference_drift_x_px"] = dx
        params["reference_drift_y_px"] = dy
        params["reference_drift_seed"] = seed

    return reference, params


# ============================================================
# SEARCH CAPTURE AUGMENTATIONS
# ============================================================

def augment_search(
    search,
    augmentation,
    rng
):
    """
    Search receives slightly stronger degradation, including stronger smooth drift.

    These are deliberately separate random draws from the
    reference RNG stream.
    """

    params = {}

    rotation = 0.0
    shear = 0.0
    scale = 1.0

    if augmentation == "rotation":

        rotation = float(
            rng.uniform(-6.0, 6.0)
        )

        search = rotate_image(
            search,
            rotation
        )

        params["search_rotation_deg"] = rotation

    elif augmentation == "shear":

        shear = float(
            rng.uniform(-5.0, 5.0)
        )

        search = shear_image(
            search,
            shear
        )

        params["search_shear_deg"] = shear

    elif augmentation == "noise":

        sigma = float(
            rng.uniform(14.0, 28.0)
        )

        image = np.asarray(
            search,
            dtype=np.float32
        )

        # Independent search noise.
        noise = rng.normal(
            0.0,
            sigma,
            image.shape
        )

        search = Image.fromarray(
            np.clip(
                image + noise,
                0,
                255
            ).astype(np.uint8),
            "L"
        )

        params["search_noise_sigma"] = sigma

    elif augmentation == "blur":

        sigma = float(
            rng.uniform(0.7, 2.4)
        )

        search = search.filter(
            ImageFilter.GaussianBlur(sigma)
        )

        params["search_blur_sigma"] = sigma

    elif augmentation == "brightness_contrast":

        brightness = float(
            rng.uniform(0.75, 1.20)
        )

        contrast = float(
            rng.uniform(0.65, 1.35)
        )

        search = ImageEnhance.Brightness(
            search
        ).enhance(brightness)

        search = ImageEnhance.Contrast(
            search
        ).enhance(contrast)

        params["search_brightness"] = brightness
        params["search_contrast"] = contrast

    elif augmentation == "scaling":

        scale = float(
            rng.uniform(0.94, 1.06)
        )

        search = scale_image(
            search,
            scale
        )

        params["search_scale_factor"] = scale

    elif augmentation == "drift":
        # Real drift: smoothly accumulated spatial displacement of the
        # captured search image.  It is deliberately stronger than the
        # reference drift and is NOT implemented as noise or GT-only motion.
        dx = float(rng.uniform(-85.0, 85.0))
        dy = float(rng.uniform(-75.0, 75.0))
        seed = int(rng.integers(0, 2**32 - 1))

        search = drift_warp(
            search, dx, dy, smoothness=0.65, seed=seed
        )

        params["search_drift_x_px"] = dx
        params["search_drift_y_px"] = dy
        params["search_drift_seed"] = seed
        params["search_drift_type"] = "smooth_accumulated_scan_drift"

    return (
        search,
        params,
        rotation,
        shear,
        scale
    )


# ============================================================
# VISUALIZATION
# ============================================================

def make_visualization(
    original_reference,
    augmented_reference,
    original_search,
    augmented_search,
    gt_x,
    gt_y,
    gt_w,
    gt_h,
    pair_id,
    augmentation,
    parameters
):
    canvas = Image.new(
        "RGB",
        (1500, 720),
        (20, 20, 20)
    )

    # --------------------------------------------------------
    # Reference panels.
    # --------------------------------------------------------

    ref_original = original_reference.resize(
        (300, 300),
        Image.Resampling.NEAREST
    ).convert("RGB")

    ref_augmented = augmented_reference.resize(
        (300, 300),
        Image.Resampling.NEAREST
    ).convert("RGB")

    canvas.paste(
        ref_original,
        (20, 100)
    )

    canvas.paste(
        ref_augmented,
        (340, 100)
    )

    # --------------------------------------------------------
    # Search panels.
    # --------------------------------------------------------

    search_original = original_search.resize(
        (500, 500),
        Image.Resampling.NEAREST
    ).convert("RGB")

    search_augmented = augmented_search.resize(
        (500, 500),
        Image.Resampling.NEAREST
    ).convert("RGB")

    canvas.paste(
        search_original,
        (680, 70)
    )

    canvas.paste(
        search_augmented,
        (1000, 70)
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (20, 60),
        "ORIGINAL REFERENCE",
        fill=(255, 255, 255)
    )

    draw.text(
        (340, 60),
        "AUGMENTED REFERENCE",
        fill=(255, 255, 255)
    )

    draw.text(
        (680, 40),
        "ORIGINAL SEARCH",
        fill=(255, 255, 255)
    )

    draw.text(
        (1000, 40),
        "AUGMENTED SEARCH + GT",
        fill=(255, 255, 255)
    )

    # --------------------------------------------------------
    # GT box on augmented search.
    # --------------------------------------------------------

    scale = 500.0 / SEARCH_SIZE

    left = 1000 + gt_x * scale
    top = 70 + gt_y * scale
    right = 1000 + (gt_x + gt_w) * scale
    bottom = 70 + (gt_y + gt_h) * scale

    draw.rectangle(
        (left, top, right, bottom),
        outline=(255, 50, 50),
        width=4
    )

    cx = 1000 + (gt_x + gt_w / 2.0) * scale
    cy = 70 + (gt_y + gt_h / 2.0) * scale

    draw.ellipse(
        (cx - 4, cy - 4, cx + 4, cy + 4),
        fill=(255, 50, 50)
    )

    # --------------------------------------------------------
    # Information.
    # --------------------------------------------------------

    draw.text(
        (20, 450),
        f"PAIR {pair_id:04d}",
        fill=(220, 230, 255)
    )

    draw.text(
        (20, 480),
        f"Augmentation: {augmentation}",
        fill=(255, 255, 255)
    )

    draw.text(
        (20, 510),
        (
            f"GT: "
            f"({gt_x:.1f}, {gt_y:.1f}) "
            f"size=({gt_w:.1f}, {gt_h:.1f})"
        ),
        fill=(255, 255, 255)
    )

    text = []

    for key, value in parameters.items():

        if isinstance(value, float):
            text.append(
                f"{key}={value:.3f}"
            )
        else:
            text.append(
                f"{key}={value}"
            )

    draw.text(
        (340, 450),
        " | ".join(text),
        fill=(255, 220, 80)
    )

    return canvas



# ============================================================
# MULTI-LOCATION TARGET SELECTION
# ============================================================

# Generator5 search scenes are organized as a 4 x 4 set of die/layout
# blocks. Every augmentation variant gets a different block.
GRID_ROWS = 4
GRID_COLS = 4

# In every 7-variant group, these variants deliberately use a junction-
# focused target. The remaining variants use dense interior regions.
JUNCTION_VARIANT_INDICES = {4, 7}


def _block_bounds(row, col, width, height):
    block_w = width / GRID_COLS
    block_h = height / GRID_ROWS
    x0 = int(round(col * block_w))
    y0 = int(round(row * block_h))
    x1 = int(round((col + 1) * block_w))
    y1 = int(round((row + 1) * block_h))
    return x0, y0, x1, y1


def _local_junction_point(image, bounds, rng):
    """
    Locate a bright compact line-intersection/junction candidate.
    """
    x0, y0, x1, y1 = bounds

    margin_x = max(12, int((x1 - x0) * 0.18))
    margin_y = max(12, int((y1 - y0) * 0.18))

    inner = image.crop((
        x0 + margin_x,
        y0 + margin_y,
        x1 - margin_x,
        y1 - margin_y
    ))

    arr = np.asarray(inner, dtype=np.float32)

    base = Image.fromarray(
        np.clip(arr, 0, 255).astype(np.uint8),
        "L"
    )

    small = np.asarray(
        base.filter(ImageFilter.GaussianBlur(1.0)),
        dtype=np.float32
    )

    large = np.asarray(
        base.filter(ImageFilter.GaussianBlur(6.0)),
        dtype=np.float32
    )

    score = small - large

    if score.shape[1] >= 7 and score.shape[0] >= 7:
        horiz = (
            np.roll(small, 3, axis=1)
            + np.roll(small, 2, axis=1)
            + np.roll(small, 1, axis=1)
            + small
            + np.roll(small, -1, axis=1)
            + np.roll(small, -2, axis=1)
            + np.roll(small, -3, axis=1)
        ) / 7.0

        vert = (
            np.roll(small, 3, axis=0)
            + np.roll(small, 2, axis=0)
            + np.roll(small, 1, axis=0)
            + small
            + np.roll(small, -1, axis=0)
            + np.roll(small, -2, axis=0)
            + np.roll(small, -3, axis=0)
        ) / 7.0

        score += 0.35 * (horiz + vert - 2.0 * large)

    score[:5, :] = -np.inf
    score[-5:, :] = -np.inf
    score[:, :5] = -np.inf
    score[:, -5:] = -np.inf

    flat = score.ravel()
    k = min(25, flat.size)
    candidate_indices = np.argpartition(flat, -k)[-k:]

    chosen = int(
        candidate_indices[
            int(rng.integers(0, len(candidate_indices)))
        ]
    )

    yy, xx = np.unravel_index(chosen, score.shape)

    return (
        x0 + margin_x + int(xx),
        y0 + margin_y + int(yy)
    )


def _safe_target_center(
    cx,
    cy,
    target_width,
    target_height,
    image_width,
    image_height
):
    half_w = target_width / 2.0
    half_h = target_height / 2.0

    cx = clamp(
        cx,
        half_w + 2.0,
        image_width - half_w - 2.0
    )

    cy = clamp(
        cy,
        half_h + 2.0,
        image_height - half_h - 2.0
    )

    return float(cx), float(cy)


def select_variant_target(
    search,
    target_width,
    target_height,
    pair_id,
    variant_index,
    selected_block,
    target_seed
):
    """
    Select a different block for each augmentation variant.

    The selected crop becomes the new reference. It is never pasted into
    the search; it is extracted from the already-existing search scene.
    """
    # Block is assigned once per base pair by main().
    # Never reshuffle here.
    row, col = selected_block

    pair_rng = np.random.default_rng(
        int(target_seed)
    )

    bounds = _block_bounds(
        row,
        col,
        search.size[0],
        search.size[1]
    )

    use_junction = variant_index in JUNCTION_VARIANT_INDICES

    if use_junction:
        cx, cy = _local_junction_point(
            search,
            bounds,
            pair_rng
        )
        selection_type = "junction"
    else:
        x0, y0, x1, y1 = bounds

        block_cx = (x0 + x1) / 2.0
        block_cy = (y0 + y1) / 2.0

        max_dx = max(
            0.0,
            (x1 - x0) / 2.0 - target_width / 2.0 - 8.0
        )

        max_dy = max(
            0.0,
            (y1 - y0) / 2.0 - target_height / 2.0 - 8.0
        )

        cx = block_cx + float(
            pair_rng.uniform(-max_dx, max_dx)
        )

        cy = block_cy + float(
            pair_rng.uniform(-max_dy, max_dy)
        )

        selection_type = "interior"

    cx, cy = _safe_target_center(
        cx,
        cy,
        target_width,
        target_height,
        search.size[0],
        search.size[1]
    )

    width = int(round(target_width))
    height = int(round(target_height))

    left = int(round(cx - width / 2.0))
    top = int(round(cy - height / 2.0))

    left = int(
        clamp(
            left,
            0,
            search.size[0] - width
        )
    )

    top = int(
        clamp(
            top,
            0,
            search.size[1] - height
        )
    )

    right = left + width
    bottom = top + height

    cropped = search.crop(
        (left, top, right, bottom)
    )

    return (
        cropped,
        float(left),
        float(top),
        float(width),
        float(height),
        row,
        col,
        selection_type
    )


def build_reference_from_search_crop(
    crop,
    output_size=REFERENCE_SIZE
):
    """
    Build the reference from the newly selected search location.

    The crop is enlarged to the reference canvas, preserving the physical
    target relationship without copying the original reference into search.
    """
    return crop.resize(
        (output_size, output_size),
        Image.Resampling.BICUBIC
    )


# ============================================================
# ONE AUGMENTED PAIR
# ============================================================

def create_variant(
    record,
    augmentation,
    variant_index,
    rng,
    selected_block
):
    reference_path = (
        INPUT_ROOT /
        "reference" /
        Path(record["reference_file"]).name
    )

    search_path = (
        INPUT_ROOT /
        "search" /
        Path(record["search_file"]).name
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference image not found:\n{reference_path}"
        )

    if not search_path.exists():
        raise FileNotFoundError(
            f"Search image not found:\n{search_path}"
        )

    # The complete search scene is the source for the NEW reference.
    original_search = load_gray(search_path)

    # Load only for validation/metadata compatibility. The original
    # reference is deliberately NOT used as the augmented reference source.
    original_reference = load_gray(reference_path)

    search_rng = np.random.default_rng(
        int(rng.integers(0, 2**32 - 1))
    )

    reference_rng = np.random.default_rng(
        int(rng.integers(0, 2**32 - 1))
    )

    drift_rng = np.random.default_rng(
        int(rng.integers(0, 2**32 - 1))
    )

    target_rng = np.random.default_rng(
        int(rng.integers(0, 2**32 - 1))
    )

    # --------------------------------------------------------
    # SELECT A DIFFERENT SEARCH BLOCK FOR THIS VARIANT
    # --------------------------------------------------------
    target_width = float(record["target_width"])
    target_height = float(record["target_height"])

    (
        selected_crop,
        gt_x,
        gt_y,
        gt_w,
        gt_h,
        selected_block_row,
        selected_block_col,
        target_selection
    ) = select_variant_target(
        original_search,
        target_width,
        target_height,
        int(record["pair_id"]),
        variant_index,
        selected_block,
        int(target_rng.integers(0, 2**32 - 1))
    )

    # --------------------------------------------------------
    # THE NEW REFERENCE COMES FROM THAT SELECTED BLOCK
    # --------------------------------------------------------
    original_reference_for_variant = (
        build_reference_from_search_crop(
            selected_crop,
            REFERENCE_SIZE
        )
    )

    # --------------------------------------------------------
    # DRIFT
    # --------------------------------------------------------
    drift_x = 0.0
    drift_y = 0.0
    drift_seed = None

    # --------------------------------------------------------
    # AUGMENT COMPLETE SEARCH
    # --------------------------------------------------------
    (
        augmented_search,
        search_params,
        search_rotation,
        search_shear,
        search_scale
    ) = augment_search(
        original_search,
        augmentation,
        search_rng
    )

    if augmentation == "drift":
        drift_x = float(search_params.get("search_drift_x_px", 0.0))
        drift_y = float(search_params.get("search_drift_y_px", 0.0))
        drift_seed = int(search_params.get("search_drift_seed", 0))

        gt_x, gt_y, gt_w, gt_h = transform_box_drift(
            gt_x, gt_y, gt_w, gt_h,
            SEARCH_SIZE, SEARCH_SIZE,
            drift_x, drift_y, drift_seed
        )

    if augmentation in (
        "rotation",
        "shear",
        "scaling"
    ):
        gt_x, gt_y, gt_w, gt_h = transform_box(
            gt_x,
            gt_y,
            gt_w,
            gt_h,
            SEARCH_SIZE,
            SEARCH_SIZE,
            search_rotation,
            search_shear,
            search_scale
        )

    # --------------------------------------------------------
    # AUGMENT THE SELECTED REFERENCE
    # --------------------------------------------------------
    augmented_reference, reference_params = (
        augment_reference(
            original_reference_for_variant,
            augmentation,
            reference_rng
        )
    )

    parameters = {}
    parameters.update(search_params)
    parameters.update(reference_params)

    parameters.update({
        "drift_x_px": drift_x,
        "drift_y_px": drift_y,
        "selected_block_row": selected_block_row,
        "selected_block_col": selected_block_col,
        "target_selection": target_selection,
        "reference_source": "selected_region_from_original_search",
        "original_reference_file_not_used_as_source": True,
    })

    pair_id = int(record["pair_id"])

    ref_name = (
        f"reference_{pair_id:04d}_"
        f"aug_{variant_index:02d}_"
        f"{augmentation}.png"
    )

    search_name = (
        f"search_{pair_id:04d}_"
        f"aug_{variant_index:02d}_"
        f"{augmentation}.png"
    )

    vis_name = (
        f"pair_{pair_id:04d}_"
        f"aug_{variant_index:02d}_"
        f"{augmentation}.png"
    )

    augmented_reference.save(
        OUTPUT_ROOT / "reference" / ref_name
    )

    augmented_search.save(
        OUTPUT_ROOT / "search" / search_name
    )

    visualization = make_visualization(
        original_reference_for_variant,
        augmented_reference,
        original_search,
        augmented_search,
        gt_x,
        gt_y,
        gt_w,
        gt_h,
        pair_id,
        augmentation,
        parameters
    )

    visualization.save(
        OUTPUT_ROOT / "visualization" / vis_name
    )

    result = {
        "augmented_pair_id": (
            f"{pair_id:04d}_{variant_index:02d}"
        ),

        "original_pair_id": pair_id,

        "augmentation_id": variant_index,

        "augmentation": augmentation,

        "reference_file": ref_name,

        "search_file": search_name,

        "reference_width": REFERENCE_SIZE,

        "reference_height": REFERENCE_SIZE,

        "search_width": SEARCH_SIZE,

        "search_height": SEARCH_SIZE,

        "target_x": round(gt_x, 4),

        "target_y": round(gt_y, 4),

        "target_width": round(gt_w, 4),

        "target_height": round(gt_h, 4),

        "center_x": round(gt_x + gt_w / 2.0, 4),

        "center_y": round(gt_y + gt_h / 2.0, 4),

        "search_augmented": True,

        "reference_augmented": True,

        "search_stronger_than_reference": True,

        "independent_reference_search_noise": True,

        "reference_pasted_into_search": False,

        "reference_selected_from_search": True,

        "selected_block_row": selected_block_row,

        "selected_block_col": selected_block_col,

        "target_selection": target_selection,
    }

    # Preserve all original Generator5 metadata fields.
    for key, value in record.items():
        if key not in result and key not in (
            "reference_file",
            "search_file",
            "target_x",
            "target_y",
            "target_width",
            "target_height",
            "center_x",
            "center_y",
        ):
            result[f"original_{key}"] = value

    result.update(parameters)

    return result


# ============================================================
# OUTPUT PREPARATION
# ============================================================

def prepare_output():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for name in (
        "reference",
        "search",
        "visualization"
    ):

        directory = (
            OUTPUT_ROOT /
            name
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        for item in directory.iterdir():

            if item.is_file() or item.is_symlink():
                item.unlink()

            elif item.is_dir():
                shutil.rmtree(item)


# ============================================================
# MAIN
# ============================================================

def main():

    global OUTPUT_ROOT

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(INPUT_ROOT)
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT)
    )

    parser.add_argument(
        "--variants",
        type=int,
        default=7
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    parser.add_argument(
        "--clean",
        action="store_true"
    )

    args = parser.parse_args()

    input_root = Path(
        args.input
    )

    output_root = Path(
        args.output
    )

    metadata_path = (
        input_root /
        "metadata.json"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Generator5 metadata not found:\n"
            f"{metadata_path}"
        )

    # The functions above use global OUTPUT_ROOT.
    OUTPUT_ROOT = output_root

    prepare_output()

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    master_rng = np.random.default_rng(
        args.seed
    )

    all_results = []

    print("=" * 76)
    print(
        "SEMICON / DRIFT-SENSE"
    )
    print(
        "GENERATOR 5 DATA AUGMENTATION"
    )
    print("=" * 76)

    print(
        f"Input         : {input_root}"
    )

    print(
        f"Output        : {output_root}"
    )

    print(
        f"Original pairs: {len(records)}"
    )

    print(
        f"Variants/pair : {args.variants}"
    )

    print()

    print(
        "7 augmentations:"
    )

    for index, name in enumerate(
        AUGMENTATIONS,
        1
    ):

        print(
            f"  {index}. {name}"
        )

    print()

    print(
        "SEARCH IS ALSO AUGMENTED."
    )

    print(
        "Search degradation is slightly stronger."
    )

    print(
        "Reference/search noise is independently generated."
    )

    print(
        "Original Generator5 files are never modified."
    )

    print()
    print(
        "IMPORTANT: each augmentation selects a DIFFERENT 4x4 scene block."
    )
    print(
        "The reference is cropped from that selected block, not copied from"
    )
    print(
        "the original reference image. Variants 4 and 7 favour junctions."
    )
    print()
    print("AUGMENTATION STRENGTHS")
    print("  Search drift       : +/-85 px X, +/-75 px Y")
    print("  Drift smoothness   : 0.65")
    print("  Search noise       : sigma 14-28")
    print("  Reference noise    : sigma 6-12")
    print("  Search brightness  : 0.75-1.20")
    print("  Search contrast    : 0.65-1.35")
    print("  Reference contrast : 0.75-1.25")
    print()

    print("=" * 76)

    output_counter = 0

    for record_index, record in enumerate(
        records,
        1
    ):

        pair_id = int(
            record["pair_id"]
        )

        print(
            f"[{record_index:03d}/{len(records):03d}] "
            f"pair={pair_id:04d}"
        )

        if args.variants > GRID_ROWS * GRID_COLS:
            raise ValueError(
                "Generator5 target selection supports at most "
                f"{GRID_ROWS * GRID_COLS} variants per base pair because "
                "each variant must use a different scene block."
            )

        # Create the block assignment ONCE for this base pair.
        # Seven variants -> seven different 4x4 scene blocks.
        blocks = [
            (row, col)
            for row in range(GRID_ROWS)
            for col in range(GRID_COLS)
        ]

        pair_plan_rng = np.random.default_rng(
            int(master_rng.integers(0, 2**32 - 1))
        )
        pair_plan_rng.shuffle(blocks)

        selected_blocks = blocks[:args.variants]

        print(
            "    unique target blocks: "
            + ", ".join(
                f"({row},{col})"
                for row, col in selected_blocks
            )
        )

        for variant_index in range(
            1,
            args.variants + 1
        ):

            augmentation = AUGMENTATIONS[
                (variant_index - 1)
                % len(AUGMENTATIONS)
            ]

            selected_block = selected_blocks[
                variant_index - 1
            ]

            result = create_variant(
                record,
                augmentation,
                variant_index,
                master_rng,
                selected_block
            )

            all_results.append(
                result
            )

            print(
                f"    variant={variant_index:02d} "
                f"{augmentation:<20} "
                f"block=({result['selected_block_row']},"
                f"{result['selected_block_col']}) "
                f"type={result['target_selection']}"
            )

            output_counter += 1

    # --------------------------------------------------------
    # Save JSON.
    # --------------------------------------------------------

    output_metadata = (
        output_root /
        "metadata.json"
    )

    with open(
        output_metadata,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save CSV.
    # --------------------------------------------------------

    output_csv = (
        output_root /
        "metadata.csv"
    )

    fields = sorted({
        key
        for row in all_results
        for key in row.keys()
    })

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print()
    print("=" * 76)
    print(
        "GENERATOR 5 AUGMENTATION COMPLETE"
    )
    print("=" * 76)

    print(
        f"Original pairs : {len(records)}"
    )

    print(
        f"Variants/pair  : {args.variants}"
    )

    print(
        f"Augmented pairs : {output_counter}"
    )

    print(
        f"Reference      : "
        f"{output_root / 'reference'}"
    )

    print(
        f"Search         : "
        f"{output_root / 'search'}"
    )

    print(
        f"Visualization  : "
        f"{output_root / 'visualization'}"
    )

    print(
        f"Metadata       : "
        f"{output_metadata}"
    )

    print(
        f"CSV            : "
        f"{output_csv}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()