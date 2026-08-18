"""
generator5.py

SEMICON / DRIFT-SENSE dataset generator.

IMPORTANT:
The search image is NEVER created by pasting the reference image.

For every pair:
1. Generate a complete 10000x10000 physical semiconductor scene.
2. Select a natural 1000x1000 physical crop from that completed scene.
3. Independently capture that crop -> 1000x1000 reference.
4. Independently capture the complete physical scene -> 10000x10000 search capture.
5. Downsample the complete search capture 10x -> 1000x1000 search image.
6. The selected physical crop therefore naturally appears as ~100x100 pixels
   inside the search image.
7. Reference and search receive independent noise.
8. Red GT boxes exist only in visualization images.

Output:
E:\\semicon\\generator5_datasets\\
    reference\\
    search\\
    visualization\\
    metadata.json
    generation_config.json
"""

from pathlib import Path
import argparse
import json
import math
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 20260816
DEFAULT_PAIRS = 300

OUTPUT_ROOT = Path(r"E:\semicon\generator5_datasets")
REFERENCE_DIR = OUTPUT_ROOT / "reference"
SEARCH_DIR = OUTPUT_ROOT / "search"
VIS_DIR = OUTPUT_ROOT / "visualization"

REFERENCE_SIZE = 1000
SEARCH_SIZE = 1000

DOWNSAMPLE_FACTOR = 10
PHYSICAL_SEARCH_SIZE = 10000
PHYSICAL_TARGET_SIZE = 1000

TARGET_SEARCH_SIZE = PHYSICAL_TARGET_SIZE // DOWNSAMPLE_FACTOR

DIE_ROWS = 4
DIE_COLS = 4

BACKGROUND = 78.0


# ============================================================
# BASIC HELPERS
# ============================================================

def to_u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def to_img(a):
    return Image.fromarray(to_u8(a), "L")


def to_array(img):
    return np.asarray(img, dtype=np.float32)


def blur(a, radius):
    if radius <= 0:
        return a.astype(np.float32)
    return to_array(
        to_img(a).filter(
            ImageFilter.GaussianBlur(float(radius))
        )
    )


def resize_array(a, size):
    return to_array(
        to_img(a).resize(
            (size, size),
            Image.Resampling.BOX
        )
    )


# ============================================================
# PHYSICAL STRUCTURE
# ============================================================

def draw_vertical_lines(a, pitch, width, value, phase):
    h, w = a.shape
    x = -pitch + phase
    while x < w + pitch:
        x0 = max(0, int(x - width / 2))
        x1 = min(w, int(x + width / 2))
        if x1 > x0:
            a[:, x0:x1] = value
        x += pitch


def draw_horizontal_lines(a, pitch, width, value, phase):
    h, w = a.shape
    y = -pitch + phase
    while y < h + pitch:
        y0 = max(0, int(y - width / 2))
        y1 = min(h, int(y + width / 2))
        if y1 > y0:
            a[y0:y1, :] = value
        y += pitch


def add_junction_dots(
    a,
    pitch_x,
    pitch_y,
    probability,
    radius,
    value,
    rng,
    phase_x=0.0,
    phase_y=0.0,
):
    """
    Junction/contact dots are part of the underlying physical
    semiconductor structure. They are NOT annotations.

    IMPORTANT:
    The dot centers are generated from the EXACT SAME phase and
    pitch used to draw the vertical and horizontal lines. Therefore
    every dot is located at a true line-line intersection rather
    than being independently placed on a nearby grid.
    """
    h, w = a.shape
    r = max(1, int(round(radius)))
    r2 = r * r

    # draw_vertical_lines() uses:
    #     x = -pitch_x + phase_x
    # and then repeatedly adds pitch_x.
    #
    # draw_horizontal_lines() uses:
    #     y = -pitch_y + phase_y
    # and then repeatedly adds pitch_y.
    #
    # Reproduce those exact coordinates here.
    xs = np.arange(
        -pitch_x + phase_x,
        w + pitch_x,
        pitch_x,
        dtype=np.float64,
    )
    ys = np.arange(
        -pitch_y + phase_y,
        h + pitch_y,
        pitch_y,
        dtype=np.float64,
    )

    for y in ys:
        for x in xs:
            if rng.random() >= probability:
                continue

            # Round only after obtaining the physical intersection.
            cx = int(round(x))
            cy = int(round(y))

            x0 = max(0, cx - r)
            x1 = min(w, cx + r + 1)
            y0 = max(0, cy - r)
            y1 = min(h, cy + r + 1)

            if x1 <= x0 or y1 <= y0:
                continue

            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r2
            a[y0:y1, x0:x1][mask] = value


def make_dram(size, rng, density):
    a = np.full((size, size), BACKGROUND, np.float32)

    ranges = {
        "very_fine": ((35, 55), (35, 55)),
        "fine":       ((55, 85), (55, 85)),
        "medium":     ((90, 135), (90, 135)),
        "coarse":     ((135, 185), (125, 180)),
    }

    (px0, px1), (py0, py1) = ranges[density]

    px = rng.uniform(px0, px1)
    py = rng.uniform(py0, py1)

    # One shared phase for each axis.
    # These phases define the actual physical line locations AND
    # the junction locations.
    phase_x = rng.uniform(0, px)
    phase_y = rng.uniform(0, py)

    draw_vertical_lines(
        a, px, rng.uniform(5, 13),
        rng.uniform(145, 180),
        phase_x
    )
    draw_horizontal_lines(
        a, py, rng.uniform(5, 13),
        rng.uniform(145, 180),
        phase_y
    )

    add_junction_dots(
        a, px, py,
        1.0,
        rng.uniform(5, 11),
        rng.uniform(205, 240),
        rng,
        phase_x,
        phase_y
    )

    return blur(a, rng.uniform(0.5, 1.5)), {
        "architecture": "DRAM",
        "density": density,
        "pitch_x": float(px),
        "pitch_y": float(py),
    }


def make_finfet(size, rng, density):
    a = np.full((size, size), BACKGROUND, np.float32)

    ranges = {
        "very_fine": ((28, 45), (38, 60)),
        "fine":       ((45, 70), (55, 80)),
        "medium":     ((70, 105), (75, 115)),
        "coarse":     ((105, 155), (100, 160)),
    }

    (fx0, fx1), (gy0, gy1) = ranges[density]

    fin_pitch = rng.uniform(fx0, fx1)
    gate_pitch = rng.uniform(gy0, gy1)

    # Shared phases guarantee that contact dots sit exactly on
    # Fin × Gate intersections.
    phase_x = rng.uniform(0, fin_pitch)
    phase_y = rng.uniform(0, gate_pitch)

    draw_vertical_lines(
        a, fin_pitch,
        max(3, fin_pitch * rng.uniform(0.18, 0.30)),
        rng.uniform(145, 180),
        phase_x
    )

    draw_horizontal_lines(
        a, gate_pitch,
        max(4, gate_pitch * rng.uniform(0.16, 0.28)),
        rng.uniform(150, 190),
        phase_y
    )

    add_junction_dots(
        a, fin_pitch, gate_pitch,
        1.0,
        rng.uniform(4, 9),
        rng.uniform(210, 245),
        rng,
        phase_x,
        phase_y
    )

    return blur(a, rng.uniform(0.4, 1.4)), {
        "architecture": "FinFET",
        "density": density,
        "pitch_x": float(fin_pitch),
        "pitch_y": float(gate_pitch),
    }


def make_die(size, rng, architecture_style):
    """Generate one die using the requested architecture style."""
    architecture = architecture_style
    density = rng.choice(
        ["very_fine", "fine", "medium", "coarse"]
    )

    if architecture == "DRAM":
        a, info = make_dram(size, rng, density)
    else:
        a, info = make_finfet(size, rng, density)

    # Mild die-level variation.
    yy = np.linspace(-1, 1, size, dtype=np.float32)
    xx = np.linspace(-1, 1, size, dtype=np.float32)
    Y, X = np.meshgrid(yy, xx)

    a += (
        rng.uniform(-4, 4)
        + rng.uniform(-3, 3) * X
        + rng.uniform(-3, 3) * Y
    )

    return np.clip(a, 0, 255), info


# ============================================================
# COMPLETE PHYSICAL SCENE
# ============================================================

def build_complete_scene(rng, architecture_style):
    """
    Generate the entire 10000x10000 physical scene FIRST.

    No target exists yet at this stage.
    This is the fundamental correction over the previous generator.
    """
    scene = np.full(
        (PHYSICAL_SEARCH_SIZE, PHYSICAL_SEARCH_SIZE),
        BACKGROUND,
        np.float32
    )

    die_w = PHYSICAL_SEARCH_SIZE // DIE_COLS
    die_h = PHYSICAL_SEARCH_SIZE // DIE_ROWS

    dies = []

    for row in range(DIE_ROWS):
        for col in range(DIE_COLS):
            x = col * die_w
            y = row * die_h

            die, info = make_die(
                die_w,
                rng,
                architecture_style
            )

            scene[
                y:y + die_h,
                x:x + die_w
            ] = die

            dies.append({
                "row": row,
                "col": col,
                **info
            })

    # Scribe/die lanes are part of the scene itself.
    img = to_img(scene)
    draw = ImageDraw.Draw(img)

    gap = rng.integers(35, 71)
    gap_value = int(rng.uniform(58, 75))

    for i in range(1, DIE_ROWS):
        y = i * die_h
        draw.rectangle(
            (
                0,
                y - gap // 2,
                PHYSICAL_SEARCH_SIZE,
                y + gap // 2
            ),
            fill=gap_value
        )

    for i in range(1, DIE_COLS):
        x = i * die_w
        draw.rectangle(
            (
                x - gap // 2,
                0,
                x + gap // 2,
                PHYSICAL_SEARCH_SIZE
            ),
            fill=gap_value
        )

    return to_array(img), dies


# ============================================================
# NATURAL TARGET SELECTION
# ============================================================

def select_target(scene, dies, rng):
    """
    Select a 1000x1000 crop from the already-complete scene.

    Nothing is inserted, copied or replaced here.
    """
    die_w = PHYSICAL_SEARCH_SIZE // DIE_COLS
    die_h = PHYSICAL_SEARCH_SIZE // DIE_ROWS

    die_row = rng.randrange(DIE_ROWS)
    die_col = rng.randrange(DIE_COLS)

    margin = 220

    min_x = die_col * die_w + margin
    min_y = die_row * die_h + margin

    max_x = (
        (die_col + 1) * die_w
        - PHYSICAL_TARGET_SIZE
        - margin
    )
    max_y = (
        (die_row + 1) * die_h
        - PHYSICAL_TARGET_SIZE
        - margin
    )

    x = rng.randint(int(min_x), int(max_x))
    y = rng.randint(int(min_y), int(max_y))

    target = scene[
        y:y + PHYSICAL_TARGET_SIZE,
        x:x + PHYSICAL_TARGET_SIZE
    ].copy()

    die_info = dies[
        die_row * DIE_COLS + die_col
    ]

    return target, {
        "physical_target_x": x,
        "physical_target_y": y,
        "physical_target_width": PHYSICAL_TARGET_SIZE,
        "physical_target_height": PHYSICAL_TARGET_SIZE,
        "die_row": die_row,
        "die_col": die_col,
        "target_architecture": die_info["architecture"],
        "target_density": die_info["density"],
    }


# ============================================================
# SEM-STYLE IMAGE FORMATION
# ============================================================

def edge_brighten(a, strength):
    """
    Brighten high-gradient feature edges.
    """
    a = a.astype(np.float32)

    gx = np.zeros_like(a)
    gy = np.zeros_like(a)

    gx[:, 1:-1] = (
        a[:, 2:] - a[:, :-2]
    ) * 0.5

    gy[1:-1, :] = (
        a[2:, :] - a[:-2, :]
    ) * 0.5

    mag = np.sqrt(gx * gx + gy * gy)

    scale = np.percentile(mag, 99)
    if scale < 1e-6:
        return a

    edge = np.clip(mag / scale, 0, 1)

    return np.clip(
        a + strength * 65.0 * edge,
        0,
        255
    )


def low_frequency_field(shape, rng, amplitude):
    h, w = shape

    small_h = max(8, h // 50)
    small_w = max(8, w // 50)

    field = rng.normal(
        0,
        1,
        (small_h, small_w)
    ).astype(np.float32)

    field -= field.min()
    field /= field.max() + 1e-6
    field *= 255

    field = to_array(
        Image.fromarray(
            field.astype(np.uint8),
            "L"
        ).resize(
            (w, h),
            Image.Resampling.BICUBIC
        )
    )

    field -= field.mean()
    field /= field.std() + 1e-6

    return field * amplitude


def capture(a, rng, params):
    """
    Independent physical capture.

    Every call gets a new RNG state, so noise is not shared.
    """
    out = a.copy()

    out = blur(
        out,
        params["blur_sigma"]
    )

    out = edge_brighten(
        out,
        params["edge_strength"]
    )

    out += low_frequency_field(
        out.shape,
        rng,
        params["illumination_amplitude"]
    )

    # Independent sensor noise.
    out += rng.normal(
        0,
        params["noise_sigma"],
        out.shape
    ).astype(np.float32)

    mean = out.mean()

    out = (
        (out - mean)
        * params["contrast"]
        + mean
        + params["brightness"]
    )

    return np.clip(out, 0, 255)


# ============================================================
# SMALL REFERENCE GEOMETRIC VARIATION
# ============================================================

def transform_reference_structure(target, rng):
    """
    Small reference-only acquisition variation.

    This operates on the underlying target structure, not on a
    reference PNG.
    """
    size = target.shape[0]

    scale = rng.uniform(0.99, 1.01)
    angle = rng.uniform(-1.5, 1.5)

    img = to_img(target)

    new_size = max(
        20,
        int(round(size * scale))
    )

    img = img.resize(
        (new_size, new_size),
        Image.Resampling.BICUBIC
    )

    if new_size >= size:
        left = (new_size - size) // 2
        img = img.crop(
            (
                left,
                left,
                left + size,
                left + size
            )
        )
    else:
        canvas = Image.new(
            "L",
            (size, size),
            int(BACKGROUND)
        )

        left = (size - new_size) // 2
        canvas.paste(img, (left, left))
        img = canvas

    img = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=int(BACKGROUND)
    )

    return to_array(img), {
        "rotation_deg": float(angle),
        "scale_factor": float(scale)
    }


# ============================================================
# SEARCH GEOMETRIC VARIATION
# ============================================================

def transform_search_scene(scene, rng):
    """
    Apply small global camera rotation/scale to the COMPLETE
    search scene.

    Because the entire scene is transformed, the GT coordinates
    are transformed mathematically as well.
    """
    size = scene.shape[0]

    scale = rng.uniform(0.995, 1.005)
    angle = rng.uniform(-1.0, 1.0)

    img = to_img(scene)

    new_size = max(
        20,
        int(round(size * scale))
    )

    img = img.resize(
        (new_size, new_size),
        Image.Resampling.BICUBIC
    )

    if new_size >= size:
        left = (new_size - size) // 2
        img = img.crop(
            (
                left,
                left,
                left + size,
                left + size
            )
        )
    else:
        canvas = Image.new(
            "L",
            (size, size),
            int(BACKGROUND)
        )

        left = (size - new_size) // 2
        canvas.paste(img, (left, left))
        img = canvas

    img = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=int(BACKGROUND)
    )

    return to_array(img), {
        "rotation_deg": float(angle),
        "scale_factor": float(scale)
    }


def transform_point(x, y, size, angle_deg, scale):
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0

    dx = (x - cx) * scale
    dy = (y - cy) * scale

    theta = math.radians(angle_deg)

    rx = (
        math.cos(theta) * dx
        - math.sin(theta) * dy
    )

    ry = (
        math.sin(theta) * dx
        + math.cos(theta) * dy
    )

    return rx + cx, ry + cy


# ============================================================
# VISUALIZATION
# ============================================================

def font(size):
    try:
        return ImageFont.truetype(
            "arial.ttf",
            size
        )
    except Exception:
        return ImageFont.load_default()


def make_visualization(
    reference,
    search,
    gt_x,
    gt_y,
    gt_w,
    gt_h,
    pair_id,
    architecture,
):
    """
    GT rectangle is drawn only here.
    It is never saved into search_XXXX.png.
    """
    canvas = Image.new(
        "RGB",
        (1600, 820),
        (12, 15, 22)
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (80, 35),
        "Reference (fixed)",
        fill=(245, 245, 245),
        font=font(34)
    )

    draw.text(
        (810, 35),
        "Search — clean",
        fill=(245, 245, 245),
        font=font(34)
    )

    ref = to_img(reference).convert("RGB")
    sea = to_img(search).convert("RGB")

    ref = ref.resize(
        (640, 640),
        Image.Resampling.BICUBIC
    )

    sea = sea.resize(
        (640, 640),
        Image.Resampling.BICUBIC
    )

    canvas.paste(ref, (50, 130))
    canvas.paste(sea, (760, 130))

    scale = 640.0 / SEARCH_SIZE

    x1 = 760 + int(gt_x * scale)
    y1 = 130 + int(gt_y * scale)

    x2 = 760 + int((gt_x + gt_w) * scale)
    y2 = 130 + int((gt_y + gt_h) * scale)

    draw.rectangle(
        (x1, y1, x2, y2),
        outline=(235, 55, 55),
        width=4
    )

    draw.text(
        (50, 780),
        f"pair={pair_id:04d} | architecture={architecture}",
        fill=(215, 215, 215),
        font=font(22)
    )

    draw.text(
        (760, 780),
        (
            f"GT center="
            f"({gt_x + gt_w / 2:.1f}, "
            f"{gt_y + gt_h / 2:.1f}) px"
        ),
        fill=(215, 215, 215),
        font=font(22)
    )

    return canvas


# ============================================================
# PAIR GENERATION
# ============================================================

def generate_pair(
    pair_id,
    seed,
    architecture_style,
    reference_dir,
    search_dir,
    vis_dir,
):
    """
    THE IMPORTANT PIPELINE:

        complete physical scene
                    |
                    v
        select natural physical crop
             /                 \
            /                   \
    independent reference   independent search
            |                   |
       1000x1000             10000x10000
                                |
                             downsample
                                |
                            1000x1000
    """

    rng = np.random.default_rng(seed)
    py = random.Random(seed)

    # --------------------------------------------------------
    # 1. COMPLETE SCENE FIRST.
    # --------------------------------------------------------
    physical_scene, dies = build_complete_scene(
        rng,
        architecture_style
    )

    # --------------------------------------------------------
    # 2. SELECT NATURAL TARGET FROM EXISTING SCENE.
    # --------------------------------------------------------
    target, target_meta = select_target(
        physical_scene,
        dies,
        py
    )

    physical_x = target_meta["physical_target_x"]
    physical_y = target_meta["physical_target_y"]

    # --------------------------------------------------------
    # 3. REFERENCE: independent capture of target crop.
    # --------------------------------------------------------
    reference_structure, ref_geometry = (
        transform_reference_structure(
            target,
            py
        )
    )

    reference_params = {
        "noise_sigma": py.uniform(3.0, 7.0),
        "blur_sigma": py.uniform(0.4, 1.5),
        "edge_strength": py.uniform(0.20, 0.34),
        "illumination_amplitude": py.uniform(1.0, 3.5),
        "contrast": py.uniform(0.96, 1.05),
        "brightness": py.uniform(-4.0, 4.0),
    }

    # New RNG stream -> independent reference noise.
    ref_rng = np.random.default_rng(seed + 1000003)

    reference = capture(
        reference_structure,
        ref_rng,
        reference_params
    )

    # --------------------------------------------------------
    # 4. SEARCH: independently capture COMPLETE scene.
    # --------------------------------------------------------
    search_scene, search_geometry = (
        transform_search_scene(
            physical_scene,
            py
        )
    )

    search_params = {
        # Search deliberately noisier than reference.
        "noise_sigma": py.uniform(5.0, 12.0),
        "blur_sigma": py.uniform(0.7, 2.4),
        "edge_strength": py.uniform(0.18, 0.32),
        "illumination_amplitude": py.uniform(2.0, 6.0),
        "contrast": py.uniform(0.90, 1.04),
        "brightness": py.uniform(-5.0, 5.0),
    }

    # Completely different RNG stream -> independent search noise.
    search_rng = np.random.default_rng(seed + 2000003)

    search_physical = capture(
        search_scene,
        search_rng,
        search_params
    )

    # --------------------------------------------------------
    # 5. Transform the four physical target corners because
    #    the COMPLETE search scene received global geometry.
    # --------------------------------------------------------
    corners = []

    for x, y in [
        (physical_x, physical_y),
        (
            physical_x + PHYSICAL_TARGET_SIZE,
            physical_y
        ),
        (
            physical_x,
            physical_y + PHYSICAL_TARGET_SIZE
        ),
        (
            physical_x + PHYSICAL_TARGET_SIZE,
            physical_y + PHYSICAL_TARGET_SIZE
        )
    ]:
        corners.append(
            transform_point(
                x,
                y,
                PHYSICAL_SEARCH_SIZE,
                search_geometry["rotation_deg"],
                search_geometry["scale_factor"]
            )
        )

    min_x = max(
        0.0,
        min(p[0] for p in corners)
    )
    min_y = max(
        0.0,
        min(p[1] for p in corners)
    )

    max_x = min(
        float(PHYSICAL_SEARCH_SIZE),
        max(p[0] for p in corners)
    )
    max_y = min(
        float(PHYSICAL_SEARCH_SIZE),
        max(p[1] for p in corners)
    )

    # --------------------------------------------------------
    # 6. COMPLETE SEARCH CAPTURE -> 10x downsample.
    # --------------------------------------------------------
    search = resize_array(
        search_physical,
        SEARCH_SIZE
    )

    gt_x = min_x / DOWNSAMPLE_FACTOR
    gt_y = min_y / DOWNSAMPLE_FACTOR
    gt_w = (max_x - min_x) / DOWNSAMPLE_FACTOR
    gt_h = (max_y - min_y) / DOWNSAMPLE_FACTOR

    center_x = gt_x + gt_w / 2.0
    center_y = gt_y + gt_h / 2.0

    # --------------------------------------------------------
    # 7. SAVE.
    # --------------------------------------------------------
    ref_path = (
        reference_dir
        / f"reference_{pair_id:04d}.png"
    )

    search_path = (
        search_dir
        / f"search_{pair_id:04d}.png"
    )

    vis_path = (
        vis_dir
        / f"pair_{pair_id:04d}.png"
    )

    Image.fromarray(
        to_u8(reference),
        "L"
    ).save(ref_path)

    Image.fromarray(
        to_u8(search),
        "L"
    ).save(search_path)

    visualization = make_visualization(
        reference,
        search,
        gt_x,
        gt_y,
        gt_w,
        gt_h,
        pair_id,
        target_meta["target_architecture"]
    )

    visualization.save(vis_path)

    # --------------------------------------------------------
    # 8. METADATA.
    # --------------------------------------------------------
    return {
        "pair_id": pair_id,

        "architecture": target_meta[
            "target_architecture"
        ],

        "density": target_meta[
            "target_density"
        ],

        "reference_file": ref_path.name,
        "search_file": search_path.name,

        "reference_width": REFERENCE_SIZE,
        "reference_height": REFERENCE_SIZE,

        "search_width": SEARCH_SIZE,
        "search_height": SEARCH_SIZE,

        "physical_search_width": PHYSICAL_SEARCH_SIZE,
        "physical_search_height": PHYSICAL_SEARCH_SIZE,

        "downsample_factor": DOWNSAMPLE_FACTOR,

        "physical_target_x": physical_x,
        "physical_target_y": physical_y,
        "physical_target_width": PHYSICAL_TARGET_SIZE,
        "physical_target_height": PHYSICAL_TARGET_SIZE,

        "target_x": round(gt_x, 4),
        "target_y": round(gt_y, 4),
        "target_width": round(gt_w, 4),
        "target_height": round(gt_h, 4),

        "center_x": round(center_x, 4),
        "center_y": round(center_y, 4),

        "target_die_row": target_meta["die_row"],
        "target_die_col": target_meta["die_col"],

        "reference_geometry": {
            k: round(float(v), 6)
            for k, v in ref_geometry.items()
        },

        "search_geometry": {
            k: round(float(v), 6)
            for k, v in search_geometry.items()
        },

        "reference_capture": {
            k: round(float(v), 6)
            for k, v in reference_params.items()
        },

        "search_capture": {
            k: round(float(v), 6)
            for k, v in search_params.items()
        },

        "independent_noise": True,
        "edge_brightening": True,

        "reference_pasted_into_search": False,

        "generation_rule": (
            "Complete physical scene generated first; target "
            "selected as a natural crop; reference independently "
            "captured from that crop; complete search scene "
            "independently captured and downsampled 10x."
        )
    }


# ============================================================
# OUTPUT
# ============================================================

def prepare_output(output_root):
    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    reference_dir = output_root / "reference"
    search_dir = output_root / "search"
    vis_dir = output_root / "visualization"

    for directory in [
        reference_dir,
        search_dir,
        vis_dir
    ]:
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architecture",
        "--architecture-style",
        dest="architecture",
        choices=["DRAM", "FinFET"],
        required=True,
        help="Semiconductor architecture style to generate: DRAM or FinFET."
    )

    parser.add_argument(
        "--pairs",
        type=int,
        required=True,
        help="Number of reference/search image pairs to generate."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which reference, search, visualization, and metadata are written."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed. Default: {SEED}"
    )

    args = parser.parse_args()

    if args.pairs <= 0:
        raise ValueError(
            "--pairs must be greater than zero"
        )

    output_root = Path(args.output_dir).expanduser().resolve()
    reference_dir = output_root / "reference"
    search_dir = output_root / "search"
    vis_dir = output_root / "visualization"

    print("=" * 76)
    print("SEMICON / DRIFT-SENSE")
    print("GENERATOR5 - PHYSICAL SCENE FIRST")
    print("=" * 76)
    print(f"Architecture      : {args.architecture}")
    print(f"Pairs             : {args.pairs}")
    print(f"Output directory  : {output_root}")
    print(f"Reference         : {REFERENCE_SIZE} x {REFERENCE_SIZE}")
    print(f"Search            : {SEARCH_SIZE} x {SEARCH_SIZE}")
    print(
        f"Physical search   : "
        f"{PHYSICAL_SEARCH_SIZE} x {PHYSICAL_SEARCH_SIZE}"
    )
    print(
        f"Physical target   : "
        f"{PHYSICAL_TARGET_SIZE} x {PHYSICAL_TARGET_SIZE}"
    )
    print(
        f"Downsample        : {DOWNSAMPLE_FACTOR}x"
    )
    print()
    print(
        "NO REFERENCE IMAGE IS EVER PASTED INTO THE SEARCH."
    )
    print(
        "The complete physical scene is generated first."
    )
    print(
        "The target is selected from that already-existing scene."
    )
    print()

    prepare_output(output_root)

    records = []

    for pair_id in range(
        1,
        args.pairs + 1
    ):
        pair_seed = (
            args.seed
            + pair_id * 10007
        )

        record = generate_pair(
            pair_id,
            pair_seed,
            args.architecture,
            reference_dir,
            search_dir,
            vis_dir,
        )

        records.append(record)

        print(
            f"[{pair_id:03d}/{args.pairs:03d}] "
            f"{record['architecture']:6s} | "
            f"{record['density']:10s} | "
            f"GT center = "
            f"({record['center_x']:7.2f}, "
            f"{record['center_y']:7.2f})"
        )

    with open(
        output_root / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f,
            indent=4
        )

    config = {
        "generator": "generator5.py",
        "seed": args.seed,
        "pairs": args.pairs,
        "architecture": args.architecture,
        "output_directory": str(output_root),

        "reference_size": [
            REFERENCE_SIZE,
            REFERENCE_SIZE
        ],

        "search_size": [
            SEARCH_SIZE,
            SEARCH_SIZE
        ],

        "physical_search_size": [
            PHYSICAL_SEARCH_SIZE,
            PHYSICAL_SEARCH_SIZE
        ],

        "physical_target_size": [
            PHYSICAL_TARGET_SIZE,
            PHYSICAL_TARGET_SIZE
        ],

        "downsample_factor": DOWNSAMPLE_FACTOR,

        "architectures": [
            "DRAM",
            "FinFET"
        ],

        "independent_noise": True,
        "edge_brightening": True,
        "blur_variation": True,
        "rotation_variation": True,
        "scale_variation": True,
        "ground_truth_recorded": True,

        "reference_pasted_into_search": False,

        "pipeline": [
            "generate complete 10000x10000 physical scene",
            "select natural 1000x1000 physical target crop",
            "independent reference capture",
            "independent complete-scene search capture",
            "10x downsample search",
            "record ground truth"
        ]
    }

    with open(
        output_root / "generation_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=4
        )

    print()
    print("=" * 76)
    print("GENERATION COMPLETE")
    print("=" * 76)
    print(f"Reference     : {reference_dir}")
    print(f"Search        : {search_dir}")
    print(f"Visualization : {vis_dir}")
    print(
        f"Metadata      : "
        f"{output_root / 'metadata.json'}"
    )
    print()
    print(
        "The red GT box exists ONLY in visualization."
    )
    print(
        "search_XXXX.png contains NO GT box."
    )
    print(
        "The reference PNG is NOT pasted into the search."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()