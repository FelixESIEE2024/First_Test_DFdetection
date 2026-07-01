from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "python_visual_odometry"
DEPTH_ANYTHING_DIR = PROJECT_ROOT / "Depth-Anything"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(DEPTH_ANYTHING_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_DIR))

import camera
import common
import frameData
import params
import pose_estimator_gauss_newton
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize


# Fixed fallback intrinsics.
# They are only used when no calib/ folder is available next to the frames.
USER_FX: float | None = 1310.400005464747
USER_FY: float | None = 1310.400005464747
USER_CX: float | None = 1820.0
USER_CY: float | None = 1024.0

DEPTH_ENCODER = "vitb"
DEPTH_MODEL_NAME = f"LiheYoung/depth_anything_{DEPTH_ENCODER}14"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "implementation" / "output"
DEFAULT_DEPTH_CACHE_DIR = SCRIPT_DIR / "depth_anything_cache" / DEPTH_ENCODER
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEATMAP_VIDEO_FPS = 10
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HISTOGRAM_LABELS = [
    "> 200",
    "150 - 200",
    "100 - 150",
    "50 - 100",
    "30 - 50",
    "10 - 30",
    "0 - 10",
]
HEATMAP_PERCENTILE_LOW = 2.0
HEATMAP_PERCENTILE_HIGH = 98.0
HEATMAP_MIN_VMAX = 0.05
HEATMAP_CMAP_NAME = "magma"
HEATMAP_PANEL_BACKGROUND = (18, 18, 18)
HEATMAP_PANEL_BORDER = (52, 52, 52)
HEATMAP_PANEL_TEXT = (235, 235, 235)
HEATMAP_INVALID_BGR = (28, 28, 28)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the notebook pipeline on every frame from an input folder and save "
            "per-frame outputs plus a global recap_evaluation directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the folder containing the extracted input frames.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where the run folder will be created.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name of the output run folder. Defaults to the input folder name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output run folder first if it already exists.",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def validate_intrinsics(require_complete: bool = True) -> dict[str, float] | None:
    intrinsics = {
        "fx": USER_FX,
        "fy": USER_FY,
        "cx": USER_CX,
        "cy": USER_CY,
    }
    missing = [name for name, value in intrinsics.items() if value is None]
    if missing:
        if not require_complete:
            return None
        missing_names = ", ".join(f"USER_{name.upper()}" for name in missing)
        raise ValueError(
            "Renseigne les intrinseques fixes en haut du script avant execution: "
            f"{missing_names}."
        )

    if USER_FX is None or USER_FY is None:
        if not require_complete:
            return None
        raise ValueError("USER_FX et USER_FY doivent etre definis.")
    if USER_FX <= 0.0 or USER_FY <= 0.0:
        raise ValueError("USER_FX et USER_FY doivent etre strictement positifs.")

    return {key: float(value) for key, value in intrinsics.items() if value is not None}


def load_intrinsics_from_file(calib_path: Path) -> dict[str, float]:
    expected_keys = {
        "USER_FX": "fx",
        "USER_FY": "fy",
        "USER_CX": "cx",
        "USER_CY": "cy",
    }
    parsed_values: dict[str, float] = {}

    for raw_line in calib_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        left, right = line.split("=", 1)
        key = left.strip().upper()
        value = right.strip()
        if key in expected_keys:
            parsed_values[expected_keys[key]] = float(value)

    missing = [mapped_key for mapped_key in expected_keys.values() if mapped_key not in parsed_values]
    if missing:
        raise ValueError(
            f"Fichier calib incomplet: {calib_path}. Champs manquants: {', '.join(missing)}."
        )
    if parsed_values["fx"] <= 0.0 or parsed_values["fy"] <= 0.0:
        raise ValueError(f"Focale invalide dans {calib_path}.")

    return parsed_values


def build_camera_from_intrinsics(
    intrinsics: dict[str, float],
    source_width: int,
    source_height: int,
) -> camera.camera:
    return camera.camera(
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
        source_width,
        source_height,
    )


def build_camera_intrinsics_summary(
    intrinsics_user: dict[str, float],
    sequence_cam: camera.camera,
    source_width: int,
    source_height: int,
    calib_path: Path | None = None,
) -> dict[str, Any]:
    level0 = {
        "width": int(sequence_cam.width[0]),
        "height": int(sequence_cam.height[0]),
        "fx": float(sequence_cam.fx[0]),
        "fy": float(sequence_cam.fy[0]),
        "cx": float(sequence_cam.cx[0]),
        "cy": float(sequence_cam.cy[0]),
    }

    per_level = []
    for lvl in range(len(sequence_cam.fx)):
        per_level.append(
            {
                "level": lvl,
                "width": int(sequence_cam.width[lvl]),
                "height": int(sequence_cam.height[lvl]),
                "fx": float(sequence_cam.fx[lvl]),
                "fy": float(sequence_cam.fy[lvl]),
                "cx": float(sequence_cam.cx[lvl]),
                "cy": float(sequence_cam.cy[lvl]),
            }
        )

    return {
        "calib_path": str(calib_path) if calib_path is not None else None,
        "source_image_size": {
            "width": source_width,
            "height": source_height,
        },
        "user_intrinsics": intrinsics_user,
        "internal_level0_intrinsics": level0,
        "pyramid_intrinsics": per_level,
        "internal_params": {
            "image_width": params.IMAGE_WIDTH,
            "image_height": params.IMAGE_HEIGHT,
            "max_levels": params.MAX_LEVELS,
        },
    }


def resolve_sequence_intrinsics(
    input_dir: Path,
    frame_paths: list[Path],
    default_intrinsics: dict[str, float] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    calib_dir = input_dir / "calib"
    intrinsics_by_frame: dict[str, dict[str, Any]] = {}

    if calib_dir.exists():
        if not calib_dir.is_dir():
            raise NotADirectoryError(f"Le chemin calib existe mais n'est pas un dossier: {calib_dir}")

        for frame_path in frame_paths:
            calib_path = calib_dir / f"{frame_path.stem}.txt"
            if not calib_path.exists():
                raise FileNotFoundError(
                    f"Fichier d'intrinseques manquant pour {frame_path.name}: {calib_path}"
                )
            intrinsics_by_frame[frame_path.name] = {
                "source_intrinsics": load_intrinsics_from_file(calib_path),
                "calib_path": calib_path.resolve(),
            }

        sequence_values = {
            key: [
                intrinsics_by_frame[frame_path.name]["source_intrinsics"][key]
                for frame_path in frame_paths
            ]
            for key in ("fx", "fy", "cx", "cy")
        }

        summary = {
            "mode": "per_frame_calib",
            "calib_dir": str(calib_dir.resolve()),
            "fallback_intrinsics": default_intrinsics,
            "source_intrinsics_statistics": {
                key: {
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "mean": float(np.mean(values)),
                }
                for key, values in sequence_values.items()
            },
        }
        return intrinsics_by_frame, summary

    print("Aucun dossier calib detecte. Repli sur les intrinseques fixes en haut du script.")
    if default_intrinsics is None:
        raise ValueError(
            "Aucun dossier calib detecte et aucun intrinseque fixe complet n'est renseigne en haut du script."
        )
    for frame_path in frame_paths:
        intrinsics_by_frame[frame_path.name] = {
            "source_intrinsics": default_intrinsics,
            "calib_path": None,
        }

    summary = {
        "mode": "fixed_fallback",
        "calib_dir": None,
        "fallback_intrinsics": default_intrinsics,
        "source_intrinsics_statistics": None,
    }
    return intrinsics_by_frame, summary


def ensure_run_directories(output_root: Path, run_name: str, overwrite: bool) -> tuple[Path, Path]:
    run_dir = output_root / run_name
    recap_dir = run_dir / "recap_evaluation"

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Le dossier de sortie existe deja: {run_dir}. "
                "Utilise --overwrite ou change --run-name."
            )
        shutil.rmtree(run_dir)

    recap_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, recap_dir


def collect_frame_paths(input_dir: Path) -> list[Path]:
    def natural_sort_key(path: Path) -> list[Any]:
        parts = re.split(r"(\d+)", path.name.lower())
        key: list[Any] = []
        for part in parts:
            key.append(int(part) if part.isdigit() else part)
        return key

    frame_paths = sorted(
        [
            path.resolve()
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_sort_key,
    )
    if len(frame_paths) < 2:
        raise ValueError(
            f"Il faut au moins 2 frames dans {input_dir}. Trouve: {len(frame_paths)}."
        )
    return frame_paths


def load_gray_image(image_path: str | Path) -> np.ndarray:
    path = resolve_path(image_path)
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image en niveaux de gris: {path}")
    return image


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    path = resolve_path(image_path)
    buffer = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image RGB: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_depth_cache_path(image_path: Path, depth_cache_dir: Path) -> Path:
    safe_name = image_path.stem.replace(" ", "_")
    cache_key = hashlib.md5(str(image_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return depth_cache_dir / f"{safe_name}_{cache_key}_depth.npy"


def get_depth_anything_components() -> tuple[DepthAnything, Compose]:
    if not DEPTH_ANYTHING_DIR.exists():
        raise FileNotFoundError(f"Depth-Anything not found: {DEPTH_ANYTHING_DIR}")

    if not hasattr(get_depth_anything_components, "model"):
        previous_cwd = Path.cwd()
        try:
            os.chdir(DEPTH_ANYTHING_DIR)
            model = DepthAnything.from_pretrained(DEPTH_MODEL_NAME).to(DEVICE).eval()
        finally:
            os.chdir(previous_cwd)

        transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        get_depth_anything_components.model = model
        get_depth_anything_components.transform = transform

        total_params = sum(param.numel() for param in model.parameters())
        print(f"Depth Anything loaded on {DEVICE} ({total_params / 1e6:.2f}M parameters)")

    return get_depth_anything_components.model, get_depth_anything_components.transform


def load_depth_map(
    image_path: Path,
    depth_cache_dir: Path,
    use_cache: bool = True,
    save_cache: bool = True,
) -> np.ndarray:
    depth_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = build_depth_cache_path(image_path, depth_cache_dir)
    if use_cache and cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    image = load_rgb_image(image_path).astype(np.float32) / 255.0
    height, width = image.shape[:2]

    model, transform = get_depth_anything_components()
    image_tensor = transform({"image": image})["image"]
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = model(image_tensor)

    depth = F.interpolate(depth[None], (height, width), mode="bilinear", align_corners=False)[0, 0]
    depth = depth.cpu().numpy().astype(np.float32)

    if save_cache:
        np.save(cache_path, depth)

    return depth


def depth_to_invdepth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv_depth = np.zeros_like(depth, dtype=np.float32)
    valid_mask = depth > 1e-6
    inv_depth[valid_mask] = 1.0 / depth[valid_mask]

    inv_depth_var = np.ones_like(depth, dtype=np.float32)
    inv_depth_var[~valid_mask] = 1e6

    return inv_depth, inv_depth_var, valid_mask


def compute_photometric_error(
    frame: frameData.frameData,
    keyframe: frameData.frameData,
    frame_cam: camera.camera,
    keyframe_cam: camera.camera,
    lvl: int = 0,
) -> dict[str, Any]:
    width = keyframe_cam.width[lvl]
    height = keyframe_cam.height[lvl]
    frame_width = frame_cam.width[lvl]
    frame_height = frame_cam.height[lvl]
    fx = frame_cam.fx[lvl]
    fy = frame_cam.fy[lvl]
    cx = frame_cam.cx[lvl]
    cy = frame_cam.cy[lvl]
    fxinv = keyframe_cam.fxinv[lvl]
    fyinv = keyframe_cam.fyinv[lvl]
    cxinv = keyframe_cam.cxinv[lvl]
    cyinv = keyframe_cam.cyinv[lvl]

    relative_pose = frame.pose.dot(keyframe.pose.inv())

    squared_error_map = np.full((height, width), np.nan, dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            inv_depth = keyframe.invDepth[lvl][y, x]
            if inv_depth <= 0.0:
                continue

            point_keyframe = np.array([fxinv * x + cxinv, fyinv * y + cyinv, 1.0]) / inv_depth
            point_frame = relative_pose.dot(point_keyframe)

            if point_frame[2] <= 0.0:
                continue

            pixel_frame = np.array(
                [
                    fx * point_frame[0] / point_frame[2] + cx,
                    fy * point_frame[1] / point_frame[2] + cy,
                ]
            )
            if (
                pixel_frame[0] < 1.0
                or pixel_frame[0] >= frame_width - 1
                or pixel_frame[1] < 1.0
                or pixel_frame[1] >= frame_height - 1
            ):
                continue

            key_intensity = float(keyframe.image[lvl][y, x])
            observed_intensity = float(common.getSubPixelValue(frame.image[lvl], pixel_frame))
            squared_error = (key_intensity - observed_intensity) ** 2

            squared_error_map[y, x] = squared_error
            valid_mask[y, x] = True

    photometric_error = float(np.nansum(squared_error_map[valid_mask]))

    return {
        "photometric_error": photometric_error,
        "squared_error_map": squared_error_map,
        "valid_mask": valid_mask,
        "level": lvl,
    }


def compute_histogram_counts(valid_abs_errors: np.ndarray) -> list[int]:
    return [
        int(np.sum(valid_abs_errors > 200)),
        int(np.sum((valid_abs_errors >= 150) & (valid_abs_errors <= 200))),
        int(np.sum((valid_abs_errors >= 100) & (valid_abs_errors < 150))),
        int(np.sum((valid_abs_errors >= 50) & (valid_abs_errors < 100))),
        int(np.sum((valid_abs_errors >= 30) & (valid_abs_errors < 50))),
        int(np.sum((valid_abs_errors >= 10) & (valid_abs_errors < 30))),
        int(np.sum((valid_abs_errors >= 0) & (valid_abs_errors < 10))),
    ]


def make_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_serializable(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [make_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(
        json.dumps(make_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_image_file(output_path: Path, image: np.ndarray) -> None:
    suffix = output_path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise IOError(f"Echec de l'encodage de l'image: {output_path}")
    encoded.tofile(str(output_path))


def save_depth_outputs(
    frame_output_dir: Path,
    image_name: str,
    gray_image: np.ndarray,
    depth_map: np.ndarray,
    inv_depth: np.ndarray,
) -> None:
    np.save(frame_output_dir / "depth_map.npy", depth_map.astype(np.float32))

    plt.imsave(frame_output_dir / "depth_map.png", depth_map, cmap="viridis")
    plt.imsave(frame_output_dir / "inverse_depth.png", inv_depth, cmap="magma")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(gray_image, cmap="gray")
    axes[0].set_title(f"Frame: {image_name}")
    axes[1].imshow(depth_map, cmap="viridis")
    axes[1].set_title("Depth map")
    axes[2].imshow(inv_depth, cmap="magma")
    axes[2].set_title("Inverse depth")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(frame_output_dir / "depth_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_histogram_plot(output_path: Path, image_name: str, range_counts: list[int], sequence_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(HISTOGRAM_LABELS, range_counts, color="#4C78A8", edgecolor="black")
    ax.set_title(f"Differences absolues - {sequence_name} - {image_name}")
    ax.set_xlabel("Tranches de difference absolue")
    ax.set_ylabel("Nombre de pixels")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for bar, count in zip(bars, range_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



###########HEATMAP UTILITIES###########

def compute_heatmap_display_range(
    error_map: np.ndarray,
    valid_mask: np.ndarray,
    low_percentile: float = HEATMAP_PERCENTILE_LOW,
    high_percentile: float = HEATMAP_PERCENTILE_HIGH,
    min_vmax: float = HEATMAP_MIN_VMAX,
) -> tuple[float, float]:
    valid_values = np.asarray(error_map, dtype=np.float32)[valid_mask]
    valid_values = valid_values[np.isfinite(valid_values)]

    if valid_values.size == 0:
        return 0.0, max(1.0, min_vmax)

    vmin, vmax = np.percentile(valid_values, [low_percentile, high_percentile])
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.max(valid_values))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    vmax = max(float(vmax), float(min_vmax))

    return float(vmin), float(vmax)


def normalize_error_map_for_display(
    error_map: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    if error_map.shape != valid_mask.shape:
        raise ValueError("La shape de error_map et valid_mask doit etre identique.")

    error_map = np.asarray(error_map, dtype=np.float32)
    vmin, vmax = compute_heatmap_display_range(error_map, valid_mask)

    filled_map = np.nan_to_num(error_map, nan=vmin, posinf=vmax, neginf=vmin)
    normalized_map = np.clip((filled_map - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    normalized_map[~valid_mask] = 0.0

    return normalized_map.astype(np.float32), vmin, vmax


def to_bgr_grayscale(image: np.ndarray) -> np.ndarray:
    gray_image = np.asarray(image)
    if gray_image.ndim != 2:
        raise ValueError("L'image de reference doit etre en niveaux de gris.")
    if gray_image.dtype != np.uint8:
        gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


def colorize_error_map(
    error_map: np.ndarray,
    valid_mask: np.ndarray,
    cmap_name: str = HEATMAP_CMAP_NAME,
) -> np.ndarray:
    normalized_map, _, _ = normalize_error_map_for_display(error_map, valid_mask)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb_heatmap = (cmap(normalized_map)[..., :3] * 255.0).astype(np.uint8)
    bgr_heatmap = cv2.cvtColor(rgb_heatmap, cv2.COLOR_RGB2BGR)
    bgr_heatmap[~valid_mask] = HEATMAP_INVALID_BGR
    return bgr_heatmap


def build_overlay_panel(
    reference_gray_image: np.ndarray,
    heatmap_bgr: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float = 0.68,
) -> np.ndarray:
    reference_bgr = to_bgr_grayscale(reference_gray_image)
    overlay = reference_bgr.copy()
    blended = cv2.addWeighted(reference_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)
    overlay[valid_mask] = blended[valid_mask]
    return overlay


def compose_heatmap_panels(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not panels:
        raise ValueError("Au moins un panneau doit etre fourni pour composer la heatmap.")

    padding = 16
    spacing = 14
    title_band_height = 42
    panel_height = max(image.shape[0] for _, image in panels)
    total_width = (2 * padding) + sum(image.shape[1] for _, image in panels) + spacing * (len(panels) - 1)
    total_height = (2 * padding) + title_band_height + panel_height

    canvas = np.full((total_height, total_width, 3), HEATMAP_PANEL_BACKGROUND, dtype=np.uint8)
    x_offset = padding
    y_image = padding + title_band_height

    for title, image in panels:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Chaque panneau doit etre une image couleur BGR.")

        panel_height_current, panel_width = image.shape[:2]
        y_offset = y_image + (panel_height - panel_height_current) // 2
        canvas[y_offset : y_offset + panel_height_current, x_offset : x_offset + panel_width] = image
        cv2.rectangle(
            canvas,
            (x_offset - 1, y_offset - 1),
            (x_offset + panel_width, y_offset + panel_height_current),
            HEATMAP_PANEL_BORDER,
            1,
        )
        cv2.putText(
            canvas,
            title,
            (x_offset, padding + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            HEATMAP_PANEL_TEXT,
            2,
            cv2.LINE_AA,
        )
        x_offset += panel_width + spacing

    return canvas


def save_squared_error_map_plot(output_path: Path, squared_error_map: np.ndarray, valid_mask: np.ndarray) -> None:
    _, vmin_sq, vmax_sq = normalize_error_map_for_display(squared_error_map, valid_mask)
    plot_map = np.ma.masked_where(~valid_mask, np.asarray(squared_error_map, dtype=np.float32))
    cmap = copy.copy(matplotlib.colormaps.get_cmap(HEATMAP_CMAP_NAME))
    cmap.set_bad(tuple(channel / 255.0 for channel in HEATMAP_INVALID_BGR[::-1]))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(plot_map, cmap=cmap, vmin=vmin_sq, vmax=vmax_sq)
    ax.set_title("Squared photometric error map")
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Squared error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_dynamic_error_heatmap_frame(
    reference_gray_image: np.ndarray,
    squared_error_map: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if squared_error_map.shape != valid_mask.shape:
        raise ValueError("La shape de squared_error_map et valid_mask doit etre identique.")
    if reference_gray_image.shape != squared_error_map.shape:
        # The photometric error map is computed at the solver working resolution.
        reference_gray_image = cv2.resize(
            reference_gray_image,
            (squared_error_map.shape[1], squared_error_map.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    reference_bgr = to_bgr_grayscale(reference_gray_image)
    heatmap_bgr = colorize_error_map(squared_error_map, valid_mask, cmap_name=HEATMAP_CMAP_NAME)
    overlay_bgr = build_overlay_panel(reference_gray_image, heatmap_bgr, valid_mask)

    return compose_heatmap_panels(
        [
            ("Reference keyframe", reference_bgr),
            ("Photometric heatmap", heatmap_bgr),
            ("Overlay", overlay_bgr),
        ]
    )


def save_dynamic_error_heatmap(
    output_path: Path,
    reference_gray_image: np.ndarray,
    squared_error_map: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    heatmap = build_dynamic_error_heatmap_frame(reference_gray_image, squared_error_map, valid_mask)
    write_image_file(output_path, heatmap)
    return heatmap


def save_heatmap_video(output_path: Path, video_frames: list[np.ndarray], fps: int = HEATMAP_VIDEO_FPS) -> None:
    if not video_frames:
        raise ValueError("Aucune frame de heatmap disponible pour construire la video.")

    first_frame = video_frames[0]
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise IOError(f"Impossible d'ouvrir le fichier video pour ecriture: {output_path}")

    try:
        for frame in video_frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("Toutes les frames de la video heatmap doivent avoir la meme taille.")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Chaque frame de la video heatmap doit etre une image couleur BGR.")
            writer.write(frame)
    finally:
        writer.release()


def save_mean_histogram_plot(output_path: Path, mean_histogram_counts: np.ndarray, valid_frame_count: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(HISTOGRAM_LABELS, mean_histogram_counts, color="#59A14F", edgecolor="black")
    ax.set_title(f"Histogramme final moyen sur {valid_frame_count} frames evaluees")
    ax.set_xlabel("Tranches de difference absolue")
    ax.set_ylabel("Nombre moyen de pixels")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for bar, value in zip(bars, mean_histogram_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_photometric_error_curve(output_path: Path, run_name: str, iteration_indices: list[int], photometric_errors: list[float]) -> float:
    mean_photometric_error = float(np.mean(photometric_errors))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iteration_indices, photometric_errors, marker="o", color="#4C78A8", label="Photometric error")
    ax.axhline(
        mean_photometric_error,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_photometric_error:.6f}",
    )
    ax.set_title(f"{run_name} : moyenne de la photometric error sur les iterations valides")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Photometric error")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mean_photometric_error


def create_frame_record_base(
    index: int,
    image_path: Path,
    frame_output_dir: Path,
    depth_valid_ratio: float,
    pose_matrix: np.ndarray,
) -> dict[str, Any]:
    return {
        "index": index,
        "image_name": image_path.name,
        "image_path": str(image_path),
        "output_dir": str(frame_output_dir),
        "depth_valid_ratio": depth_valid_ratio,
        "pose_matrix": pose_matrix.tolist(),
    }


def run_sequence(
    frame_paths: list[Path],
    run_dir: Path,
    recap_dir: Path,
    depth_cache_dir: Path,
    intrinsics_by_frame: dict[str, dict[str, Any]],
    intrinsics_mode_summary: dict[str, Any],
    run_name: str,
) -> None:
    first_image_path = frame_paths[0]
    first_image = load_gray_image(first_image_path)
    sequence_height, sequence_width = first_image.shape[:2]
    keyframe_intrinsics = intrinsics_by_frame[first_image_path.name]["source_intrinsics"]
    keyframe_calib_path = intrinsics_by_frame[first_image_path.name]["calib_path"]
    keyframe_cam = build_camera_from_intrinsics(
        keyframe_intrinsics,
        source_width=sequence_width,
        source_height=sequence_height,
    )
    sequence_pose_solver = pose_estimator_gauss_newton.pose_estimator_gauss_newton(
        keyframe_cam,
        show_debug=False,
        keyframe_camera=keyframe_cam,
        frame_camera=keyframe_cam,
    )

    keyframe_intrinsics_summary = build_camera_intrinsics_summary(
        keyframe_intrinsics,
        keyframe_cam,
        source_width=sequence_width,
        source_height=sequence_height,
        calib_path=keyframe_calib_path,
    )

    print(f"Sequence {run_name}:")
    print(f"  input dir = {frame_paths[0].parent}")
    print(f"  frames = {len(frame_paths)}")
    print(f"  source size = {sequence_width} x {sequence_height}")
    print(f"  intrinsics mode = {intrinsics_mode_summary['mode']}")
    print(f"  working size level-0 = {keyframe_cam.width[0]} x {keyframe_cam.height[0]}")
    print(f"  output run dir = {run_dir}")
    print()

    sequence_keyframe: frameData.frameData | None = None
    sequence_results: list[dict[str, Any]] = []
    heatmap_video_frames: list[np.ndarray] = []

    for step, image_path in enumerate(frame_paths):
        image = load_gray_image(image_path)
        frame_intrinsics_record = intrinsics_by_frame[image_path.name]
        frame_intrinsics = frame_intrinsics_record["source_intrinsics"]
        frame_calib_path = frame_intrinsics_record["calib_path"]
        current_cam = build_camera_from_intrinsics(
            frame_intrinsics,
            source_width=sequence_width,
            source_height=sequence_height,
        )
        current_intrinsics_summary = build_camera_intrinsics_summary(
            frame_intrinsics,
            current_cam,
            source_width=sequence_width,
            source_height=sequence_height,
            calib_path=frame_calib_path,
        )

        if image.shape[:2] != (sequence_height, sequence_width):
            raise ValueError(
                "Toutes les images doivent avoir la meme taille. "
                f"Attendu: {(sequence_height, sequence_width)}, obtenu: {image.shape[:2]} "
                f"pour {image_path.name}."
            )

        depth_map = load_depth_map(image_path, depth_cache_dir=depth_cache_dir)
        inv_depth, inv_depth_var, depth_valid_mask = depth_to_invdepth(depth_map)

        current_frame = frameData.frameData()
        current_frame.setImage(image)
        current_frame.setInvDepth(inv_depth, inv_depth_var)

        frame_output_dir = run_dir / image_path.stem
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        save_depth_outputs(frame_output_dir, image_path.name, image, depth_map, inv_depth)

        if step == 0:
            sequence_keyframe = copy.deepcopy(current_frame)
            pose_matrix = current_frame.pose.as_matrix()
            keyframe_heatmap = save_dynamic_error_heatmap(
                frame_output_dir / "photometric_error_heatmap.png",
                sequence_keyframe.image[0],
                np.zeros((keyframe_cam.height[0], keyframe_cam.width[0]), dtype=np.float32),
                np.zeros((keyframe_cam.height[0], keyframe_cam.width[0]), dtype=bool),
            )
            heatmap_video_frames.append(keyframe_heatmap)

            frame_record = create_frame_record_base(
                index=step,
                image_path=image_path,
                frame_output_dir=frame_output_dir,
                depth_valid_ratio=float(depth_valid_mask.mean()),
                pose_matrix=pose_matrix,
            )
            frame_record.update(
                {
                    "frame_role": "keyframe",
                    "frame_intrinsics": current_intrinsics_summary,
                    "keyframe_intrinsics": current_intrinsics_summary,
                    "photometric_error": None,
                    "photometric_error_mean": None,
                    "photometric_error_cumulative": None,
                    "valid_pixel_count": None,
                    "histogram_counts": None,
                    "optimization_summary": None,
                    "errors_improved_by_level": None,
                    "pose_is_finite": True,
                }
            )

            np.savetxt(frame_output_dir / "pose_matrix.txt", pose_matrix, fmt="%.9f")
            write_json(frame_output_dir / "frame_metrics.json", frame_record)
            sequence_results.append(frame_record)

            print(f"[{run_name}] frame {step} -> keyframe de reference: {image_path.name}")
            continue

        if sequence_keyframe is None:
            raise RuntimeError("Keyframe non initialisee.")

        sequence_pose_solver.set_cameras(current_cam, keyframe_cam)
        initial_error_lvl4, _ = sequence_pose_solver.computeError(current_frame, sequence_keyframe, lvl=4)
        initial_error_lvl3, _ = sequence_pose_solver.computeError(current_frame, sequence_keyframe, lvl=3)
        initial_error_lvl2, _ = sequence_pose_solver.computeError(current_frame, sequence_keyframe, lvl=2)

        sequence_pose_solver.optPose(current_frame, sequence_keyframe)

        final_error_lvl4, _ = sequence_pose_solver.computeError(current_frame, sequence_keyframe, lvl=4)
        final_error_lvl3, _ = sequence_pose_solver.computeError(current_frame, sequence_keyframe, lvl=3)
        final_error_lvl2, _ = sequence_pose_solver.computeError(current_frame, sequence_keyframe, lvl=2)

        evaluation = compute_photometric_error(
            current_frame,
            sequence_keyframe,
            frame_cam=current_cam,
            keyframe_cam=keyframe_cam,
            lvl=0,
        )
        abs_error_map = np.sqrt(evaluation["squared_error_map"])
        valid_mask = evaluation["valid_mask"]
        valid_abs_errors = abs_error_map[valid_mask]

        photometric_error = float(evaluation["photometric_error"])
        photometric_error_mean = float(valid_abs_errors.mean()) if valid_abs_errors.size > 0 else None
        valid_pixel_count = int(valid_abs_errors.size)
        histogram_counts = compute_histogram_counts(valid_abs_errors)

        optimization_summary = {
            "lvl4_initial": float(initial_error_lvl4),
            "lvl4_final": float(final_error_lvl4),
            "lvl3_initial": float(initial_error_lvl3),
            "lvl3_final": float(final_error_lvl3),
            "lvl2_initial": float(initial_error_lvl2),
            "lvl2_final": float(final_error_lvl2),
            "lvl4_improvement": float(initial_error_lvl4 - final_error_lvl4),
            "lvl3_improvement": float(initial_error_lvl3 - final_error_lvl3),
            "lvl2_improvement": float(initial_error_lvl2 - final_error_lvl2),
        }

        errors_improved_by_level = {
            "lvl4": bool(optimization_summary["lvl4_improvement"] > 0.0),
            "lvl3": bool(optimization_summary["lvl3_improvement"] > 0.0),
            "lvl2": bool(optimization_summary["lvl2_improvement"] > 0.0),
        }

        pose_matrix = current_frame.pose.as_matrix()
        pose_is_finite = bool(np.isfinite(pose_matrix).all())
        if not pose_is_finite:
            raise ValueError(f"Pose matrix contains non-finite values for {image_path.name}.")

        save_histogram_plot(
            frame_output_dir / "absolute_difference_histogram.png",
            image_path.name,
            histogram_counts,
            run_name,
        )
        save_squared_error_map_plot(
            frame_output_dir / "squared_photometric_error_map.png",
            evaluation["squared_error_map"],
            valid_mask,
        )
        heatmap_frame = save_dynamic_error_heatmap(
            frame_output_dir / "photometric_error_heatmap.png",
            sequence_keyframe.image[0],
            evaluation["squared_error_map"],
            valid_mask,
        )
        heatmap_video_frames.append(heatmap_frame)

        frame_record = create_frame_record_base(
            index=step,
            image_path=image_path,
            frame_output_dir=frame_output_dir,
            depth_valid_ratio=float(depth_valid_mask.mean()),
            pose_matrix=pose_matrix,
        )
        frame_record.update(
            {
                "frame_role": "frame",
                "frame_intrinsics": current_intrinsics_summary,
                "keyframe_intrinsics": keyframe_intrinsics_summary,
                "photometric_error": photometric_error,
                "photometric_error_mean": photometric_error_mean,
                "photometric_error_cumulative": None,
                "valid_pixel_count": valid_pixel_count,
                "histogram_counts": histogram_counts,
                "optimization_summary": optimization_summary,
                "errors_improved_by_level": errors_improved_by_level,
                "pose_is_finite": pose_is_finite,
            }
        )

        np.savetxt(frame_output_dir / "pose_matrix.txt", pose_matrix, fmt="%.9f")
        write_json(frame_output_dir / "frame_metrics.json", frame_record)
        sequence_results.append(frame_record)

        mean_error_display = f"{photometric_error_mean:.4f}" if photometric_error_mean is not None else "None"
        print(
            f"[{run_name}] frame {step} -> {image_path.name} | "
            f"sum error = {photometric_error:.2f} | "
            f"mean abs error = {mean_error_display} | "
            f"valid pixels = {valid_pixel_count}"
        )

    valid_histogram_results = [
        item for item in sequence_results if item.get("histogram_counts") is not None
    ]
    if not valid_histogram_results:
        raise ValueError("Aucun histogramme valide disponible pour calculer la moyenne finale.")

    histogram_matrix = np.array(
        [item["histogram_counts"] for item in valid_histogram_results],
        dtype=np.float64,
    )
    mean_histogram_counts = histogram_matrix.mean(axis=0)
    save_mean_histogram_plot(
        recap_dir / "mean_histogram.png",
        mean_histogram_counts,
        valid_frame_count=len(valid_histogram_results),
    )

    valid_results = [item for item in sequence_results if item["photometric_error"] is not None]
    if not valid_results:
        raise ValueError(
            "Aucun resultat valide disponible pour calculer les erreurs photometriques globales."
        )

    iteration_indices = [int(item["index"]) for item in valid_results]
    photometric_errors = [float(item["photometric_error"]) for item in valid_results]
    mean_photometric_error = save_photometric_error_curve(
        recap_dir / "photometric_error_curve.png",
        run_name,
        iteration_indices,
        photometric_errors,
    )
    cumulative_photometric_error = float(np.sum(photometric_errors))
    heatmap_video_path = recap_dir / "photometric_error_heatmap_video.mp4"
    save_heatmap_video(heatmap_video_path, heatmap_video_frames, fps=HEATMAP_VIDEO_FPS)

    running_total = 0.0
    for item in sequence_results:
        current_error = item.get("photometric_error")
        if current_error is None:
            item["photometric_error_cumulative"] = None
            continue
        running_total += float(current_error)
        item["photometric_error_cumulative"] = float(running_total)
        frame_metrics_path = Path(item["output_dir"]) / "frame_metrics.json"
        write_json(frame_metrics_path, item)

    summary = {
        "run_name": run_name,
        "project_root": str(PROJECT_ROOT),
        "input_dir": str(frame_paths[0].parent),
        "output_run_dir": str(run_dir),
        "recap_dir": str(recap_dir),
        "frames_processed": len(frame_paths),
        "frames_evaluated": len(valid_results),
        "keyframe_name": frame_paths[0].name,
        "depth_encoder": DEPTH_ENCODER,
        "depth_model_name": DEPTH_MODEL_NAME,
        "device": DEVICE,
        "intrinsics": {
            "mode_summary": intrinsics_mode_summary,
            "keyframe_intrinsics": keyframe_intrinsics_summary,
        },
        "photometric_metrics": {
            "cumulative_photometric_error": cumulative_photometric_error,
            "mean_photometric_error": mean_photometric_error,
            "histogram_labels": HISTOGRAM_LABELS,
            "mean_histogram_counts": mean_histogram_counts.tolist(),
        },
        "output_files": {
            "mean_histogram": str(recap_dir / "mean_histogram.png"),
            "photometric_error_curve": str(recap_dir / "photometric_error_curve.png"),
            "photometric_error_heatmap_video": str(heatmap_video_path),
            "sequence_results_json": str(recap_dir / "sequence_results.json"),
        },
    }

    write_json(recap_dir / "sequence_results.json", {"results": sequence_results})
    write_json(recap_dir / "summary.json", summary)

    print()
    print("Recap evaluation:")
    print(f"  cumulative photometric error = {cumulative_photometric_error:.6f}")
    print(f"  mean photometric error = {mean_photometric_error:.6f}")
    print(f"  recap dir = {recap_dir}")


def main() -> int:
    args = parse_args()

    input_dir = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_root)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    run_name = args.run_name or input_dir.name
    run_dir, recap_dir = ensure_run_directories(output_root, run_name, args.overwrite)
    frame_paths = collect_frame_paths(input_dir)
    has_calib_dir = (input_dir / "calib").is_dir()
    default_intrinsics = validate_intrinsics(require_complete=not has_calib_dir)
    intrinsics_by_frame, intrinsics_mode_summary = resolve_sequence_intrinsics(
        input_dir=input_dir,
        frame_paths=frame_paths,
        default_intrinsics=default_intrinsics,
    )

    run_sequence(
        frame_paths=frame_paths,
        run_dir=run_dir,
        recap_dir=recap_dir,
        depth_cache_dir=DEFAULT_DEPTH_CACHE_DIR,
        intrinsics_by_frame=intrinsics_by_frame,
        intrinsics_mode_summary=intrinsics_mode_summary,
        run_name=run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
