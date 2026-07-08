# Basic run:
# python run_depth_pose_sequence.py --input-dir "C:\path\to\frames" --overwrite

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

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
HEATMAP_MIN_ABS_ERROR_DISPLAY = 10.0
VISIBILITY_REL_DEPTH_EPS = 0.02
HEATMAP_PANEL_BACKGROUND = (18, 18, 18)
HEATMAP_PANEL_BORDER = (52, 52, 52)
HEATMAP_PANEL_TEXT = (235, 235, 235)
HEATMAP_INVALID_BGR = (28, 28, 28)


class Compose:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


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
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. Example: --max-frames 10. "
            "The shorthand --10 / --20 / ... is also accepted."
        ),
    )
    parser.add_argument("--win_sec", type=float, default=3.0, help="Window duration (sec).")
    parser.add_argument("--max_windows", type=int, default=4, help="Max windows per clip.")
    parser.add_argument(
        "--pair-stride",
        "--pair_stride",
        dest="pair_stride",
        type=int,
        default=1,
        help="Stride used to sample target frames inside each window.",
    )
    parser.add_argument(
        "--sans_slice",
        action="store_true",
        help="Inside each window, only compare each source frame with the next frame.",
    )

    args, unknown_args = parser.parse_known_args()

    shorthand_max_frames: int | None = None
    remaining_unknown_args: list[str] = []
    for raw_arg in unknown_args:
        shorthand_match = re.fullmatch(r"--(\d+)", raw_arg)
        if shorthand_match is None:
            remaining_unknown_args.append(raw_arg)
            continue

        parsed_value = int(shorthand_match.group(1))
        if parsed_value <= 0:
            parser.error("Le raccourci --N doit utiliser un entier strictement positif.")
        if shorthand_max_frames is not None and shorthand_max_frames != parsed_value:
            parser.error("Un seul raccourci --N peut etre fourni.")
        shorthand_max_frames = parsed_value

    if remaining_unknown_args:
        parser.error(f"Arguments non reconnus: {' '.join(remaining_unknown_args)}")

    if shorthand_max_frames is not None:
        if args.max_frames is not None and args.max_frames != shorthand_max_frames:
            parser.error("Utilise soit --max-frames N, soit le raccourci --N avec la meme valeur.")
        args.max_frames = shorthand_max_frames

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames doit etre strictement positif.")
    if args.win_sec <= 0.0:
        parser.error("--win_sec doit etre strictement positif.")
    if args.max_windows <= 0:
        parser.error("--max_windows doit etre strictement positif.")
    if args.pair_stride <= 0:
        parser.error("--pair-stride doit etre strictement positif.")

    return args


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


def get_launch_command() -> str:
    original_command = os.environ.get("RUN_DEPTH_POSE_SEQUENCE_COMMAND")
    if original_command:
        return original_command
    return subprocess.list2cmdline([Path(sys.argv[0]).name, *sys.argv[1:]])


def write_launch_command(output_path: Path) -> None:
    launch_command = get_launch_command()
    output_path.write_text(f"{launch_command}\n", encoding="utf-8")


def print_progress(
    step_index: int,
    total_steps: int,
    window_index: int,
    total_windows: int,
    source_name: str,
    target_name: str,
) -> None:
    if total_steps <= 0:
        return

    bar_width = 28
    filled = int(round(bar_width * step_index / total_steps))
    filled = max(0, min(bar_width, filled))
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(
        f"\r[{bar}] {step_index}/{total_steps} | "
        f"window {window_index + 1}/{total_windows} | src={source_name} -> tgt={target_name}"
    )
    sys.stdout.flush()


def build_fixed_len_cover_windows(
    frame_paths: list[Path],
    win_sec: float,
    max_windows: int,
) -> tuple[list[dict[str, Any]], int]:
    frame_count = len(frame_paths)
    if frame_count < 2:
        return [], 0

    requested_window_length = max(2, int(round(win_sec)))
    if requested_window_length >= frame_count:
        return [
            {
                "window_index": 0,
                "start_index": 0,
                "end_index": frame_count - 1,
                "frame_indices": list(range(frame_count)),
            }
        ], frame_count

    required_window_count = int(math.ceil(frame_count / requested_window_length))
    if max_windows > 0 and required_window_count > max_windows:
        effective_window_length = max(2, int(math.ceil(frame_count / max_windows)))
        window_count = max_windows
    else:
        effective_window_length = requested_window_length
        window_count = required_window_count

    if window_count <= 1:
        starts = [0]
    else:
        last_start = max(0, frame_count - effective_window_length)
        starts: list[int] = []
        for offset in range(window_count):
            ideal_start = int(round(last_start * offset / (window_count - 1)))
            min_start = starts[-1] + 1 if starts else 0
            max_start = last_start - (window_count - 1 - offset)
            if max_start < min_start:
                max_start = min_start
            starts.append(min(max(ideal_start, min_start), max_start))

    windows: list[dict[str, Any]] = []
    for window_index, start_index in enumerate(starts):
        frame_indices = list(range(start_index, min(frame_count, start_index + effective_window_length)))
        if len(frame_indices) < 2:
            continue
        windows.append(
            {
                "window_index": window_index,
                "start_index": frame_indices[0],
                "end_index": frame_indices[-1],
                "frame_indices": frame_indices,
            }
        )

    if not windows:
        windows = [
            {
                "window_index": 0,
                "start_index": 0,
                "end_index": frame_count - 1,
                "frame_indices": list(range(frame_count)),
            }
        ]
        effective_window_length = frame_count

    return windows, effective_window_length


def build_source_target_map(
    window_indices: list[int],
    pair_stride: int,
    sans_slice: bool = False,
) -> dict[int, list[int]]:
    if sans_slice:
        source_target_map: dict[int, list[int]] = {}
        for source_position, source_index in enumerate(window_indices):
            if source_position + 1 < len(window_indices):
                source_target_map[source_index] = [window_indices[source_position + 1]]
            else:
                source_target_map[source_index] = []
        return source_target_map

    target_positions = list(range(0, len(window_indices), pair_stride))
    source_target_map: dict[int, list[int]] = {}

    for source_position, source_index in enumerate(window_indices):
        target_indices = [
            window_indices[target_position]
            for target_position in target_positions
            if target_position != source_position
        ]
        if not target_indices:
            fallback_targets = [index for index in window_indices if index != source_index]
            if fallback_targets:
                target_indices = [fallback_targets[0]]
        source_target_map[source_index] = target_indices

    return source_target_map


def collect_frame_paths(input_dir: Path, max_frames: int | None = None) -> list[Path]:
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

    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
        if len(frame_paths) < 2:
            raise ValueError(
                f"--max-frames={max_frames} laisse moins de 2 frames exploitables dans {input_dir}."
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
    z_buffer = np.full((frame_height, frame_width), np.inf, dtype=np.float32)
    projected_candidates: list[tuple[int, int, np.ndarray, float]] = []

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

            projected_candidates.append((x, y, pixel_frame, float(point_frame[2])))

            projected_x = int(round(float(pixel_frame[0])))
            projected_y = int(round(float(pixel_frame[1])))
            projected_x = int(np.clip(projected_x, 0, frame_width - 1))
            projected_y = int(np.clip(projected_y, 0, frame_height - 1))
            z_buffer[projected_y, projected_x] = min(z_buffer[projected_y, projected_x], float(point_frame[2]))

    occluded_pixel_count = 0
    for x, y, pixel_frame, projected_depth in projected_candidates:
        projected_x = int(round(float(pixel_frame[0])))
        projected_y = int(round(float(pixel_frame[1])))
        projected_x = int(np.clip(projected_x, 0, frame_width - 1))
        projected_y = int(np.clip(projected_y, 0, frame_height - 1))

        closest_depth = float(z_buffer[projected_y, projected_x])
        if not np.isfinite(closest_depth):
            continue
        if projected_depth > closest_depth * (1.0 + VISIBILITY_REL_DEPTH_EPS):
            occluded_pixel_count += 1
            continue

        key_intensity = float(keyframe.image[lvl][y, x])
        observed_intensity = float(common.getSubPixelValue(frame.image[lvl], pixel_frame))
        squared_error = (key_intensity - observed_intensity) ** 2

        squared_error_map[y, x] = squared_error
        valid_mask[y, x] = True

    visible_pixel_count = int(np.count_nonzero(valid_mask))
    photometric_error_sum = float(np.nansum(squared_error_map[valid_mask]))
    photometric_error_mean = (
        float(photometric_error_sum / visible_pixel_count) if visible_pixel_count > 0 else None
    )

    return {
        "photometric_error": photometric_error_mean,
        "photometric_error_sum": photometric_error_sum,
        "squared_error_map": squared_error_map,
        "valid_mask": valid_mask,
        "projected_candidate_count": int(len(projected_candidates)),
        "occluded_pixel_count": int(occluded_pixel_count),
        "visible_pixel_count": visible_pixel_count,
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


def summarize_relative_pose(frame_pose: Any, keyframe_pose: Any) -> dict[str, float]:
    relative_pose_matrix = frame_pose.dot(keyframe_pose.inv()).as_matrix()
    translation_norm = float(np.linalg.norm(relative_pose_matrix[:3, 3]))

    rotation_trace = float(np.trace(relative_pose_matrix[:3, :3]))
    rotation_cos = np.clip((rotation_trace - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(rotation_cos)))

    return {
        "translation_norm": translation_norm,
        "rotation_deg": rotation_deg,
    }


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


def build_heatmap_display_mask(
    squared_error_map: np.ndarray,
    valid_mask: np.ndarray,
    min_abs_error: float = HEATMAP_MIN_ABS_ERROR_DISPLAY,
) -> np.ndarray:
    abs_error_map = np.sqrt(np.clip(np.asarray(squared_error_map, dtype=np.float32), 0.0, None))
    return valid_mask & np.isfinite(abs_error_map) & (abs_error_map >= float(min_abs_error))


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

    display_valid_mask = build_heatmap_display_mask(squared_error_map, valid_mask)
    reference_bgr = to_bgr_grayscale(reference_gray_image)
    heatmap_bgr = colorize_error_map(squared_error_map, display_valid_mask, cmap_name=HEATMAP_CMAP_NAME)
    colored_frame_bgr = build_overlay_panel(reference_gray_image, heatmap_bgr, display_valid_mask)

    return compose_heatmap_panels(
        [
            ("Original frame", reference_bgr),
            ("Colored frame", colored_frame_bgr),
        ]
    )


def save_dynamic_error_heatmap(
    output_path: Path | None,
    reference_gray_image: np.ndarray,
    squared_error_map: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    heatmap = build_dynamic_error_heatmap_frame(reference_gray_image, squared_error_map, valid_mask)
    if output_path is not None:
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
    photometric_errors_array = np.asarray(photometric_errors, dtype=np.float64)
    mean_photometric_error = float(np.mean(photometric_errors_array))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        iteration_indices,
        photometric_errors_array,
        marker="o",
        color="#4C78A8",
        label="Mean squared photometric error",
    )
    ax.axhline(
        mean_photometric_error,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_photometric_error:.2f}",
    )
    ax.set_title(
        f"{run_name} : photometric error moyenne par pixel visible"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean squared photometric error")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mean_photometric_error


def create_frame_record_base(
    index: int,
    image_path: Path,
    frame_output_dir: Path | None,
    depth_valid_ratio: float,
    pose_matrix: np.ndarray,
) -> dict[str, Any]:
    return {
        "index": index,
        "image_name": image_path.name,
        "image_path": str(image_path),
        "output_dir": str(frame_output_dir) if frame_output_dir is not None else None,
        "depth_valid_ratio": depth_valid_ratio,
        "pose_matrix": pose_matrix.tolist(),
    }


def prepare_frame_entries(
    frame_paths: list[Path],
    depth_cache_dir: Path,
    intrinsics_by_frame: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    first_image = load_gray_image(frame_paths[0])
    sequence_height, sequence_width = first_image.shape[:2]
    prepared_frames: list[dict[str, Any]] = []

    for frame_index, image_path in enumerate(frame_paths):
        image = first_image if frame_index == 0 else load_gray_image(image_path)
        if image.shape[:2] != (sequence_height, sequence_width):
            raise ValueError(
                "Toutes les images doivent avoir la meme taille. "
                f"Attendu: {(sequence_height, sequence_width)}, obtenu: {image.shape[:2]} "
                f"pour {image_path.name}."
            )

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

        depth_map = load_depth_map(image_path, depth_cache_dir=depth_cache_dir)
        inv_depth, inv_depth_var, depth_valid_mask = depth_to_invdepth(depth_map)

        current_frame = frameData.frameData()
        current_frame.setImage(image)
        current_frame.setInvDepth(inv_depth, inv_depth_var)

        prepared_frames.append(
            {
                "index": frame_index,
                "image_path": image_path,
                "frame_data": current_frame,
                "camera": current_cam,
                "intrinsics_summary": current_intrinsics_summary,
                "depth_valid_ratio": float(depth_valid_mask.mean()),
                "display_image": current_frame.image[0].copy(),
            }
        )

    return prepared_frames


def evaluate_frame_pair(
    source_entry: dict[str, Any],
    target_entry: dict[str, Any],
    window_index: int,
    pair_index: int,
    total_pairs: int,
    total_windows: int,
) -> tuple[dict[str, Any], np.ndarray]:
    source_frame = source_entry["frame_data"]
    target_frame = copy.deepcopy(target_entry["frame_data"])
    source_cam = source_entry["camera"]
    target_cam = target_entry["camera"]

    target_frame.pose = copy.copy(source_frame.pose)

    print_progress(
        step_index=pair_index,
        total_steps=total_pairs,
        window_index=window_index,
        total_windows=total_windows,
        source_name=source_entry["image_path"].name,
        target_name=target_entry["image_path"].name,
    )

    pose_solver = pose_estimator_gauss_newton.pose_estimator_gauss_newton(
        source_cam,
        show_debug=False,
        keyframe_camera=source_cam,
        frame_camera=target_cam,
        verbose=False,
    )
    pose_solver.set_cameras(target_cam, source_cam)

    initial_error_lvl4, _ = pose_solver.computeError(target_frame, source_frame, lvl=4)
    initial_error_lvl3, _ = pose_solver.computeError(target_frame, source_frame, lvl=3)
    initial_error_lvl2, _ = pose_solver.computeError(target_frame, source_frame, lvl=2)

    pose_solver.optPose(target_frame, source_frame)

    final_error_lvl4, _ = pose_solver.computeError(target_frame, source_frame, lvl=4)
    final_error_lvl3, _ = pose_solver.computeError(target_frame, source_frame, lvl=3)
    final_error_lvl2, _ = pose_solver.computeError(target_frame, source_frame, lvl=2)

    evaluation = compute_photometric_error(
        target_frame,
        source_frame,
        frame_cam=target_cam,
        keyframe_cam=source_cam,
        lvl=0,
    )
    abs_error_map = np.sqrt(evaluation["squared_error_map"])
    valid_mask = evaluation["valid_mask"]
    valid_abs_errors = abs_error_map[valid_mask]

    photometric_error = evaluation["photometric_error"]
    photometric_error_sum = float(evaluation["photometric_error_sum"])
    photometric_error_mean_abs = float(valid_abs_errors.mean()) if valid_abs_errors.size > 0 else None
    valid_pixel_count = int(evaluation["visible_pixel_count"])
    projected_candidate_count = int(evaluation["projected_candidate_count"])
    occluded_pixel_count = int(evaluation["occluded_pixel_count"])
    histogram_counts = compute_histogram_counts(valid_abs_errors) if valid_abs_errors.size > 0 else None
    relative_pose_stats = summarize_relative_pose(target_frame.pose, source_frame.pose)

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

    pose_matrix = target_frame.pose.as_matrix()
    if not bool(np.isfinite(pose_matrix).all()):
        raise ValueError(f"Pose matrix contains non-finite values for {target_entry['image_path'].name}.")

    pair_record = {
        "window_index": window_index,
        "source_frame_index": int(source_entry["index"]),
        "target_frame_index": int(target_entry["index"]),
        "source_frame_name": source_entry["image_path"].name,
        "target_frame_name": target_entry["image_path"].name,
        "source_frame_path": str(source_entry["image_path"]),
        "target_frame_path": str(target_entry["image_path"]),
        "source_depth_valid_ratio": float(source_entry["depth_valid_ratio"]),
        "target_depth_valid_ratio": float(target_entry["depth_valid_ratio"]),
        "photometric_error": photometric_error,
        "photometric_error_sum": photometric_error_sum,
        "photometric_error_mean_abs": photometric_error_mean_abs,
        "valid_pixel_count": valid_pixel_count,
        "projected_candidate_count": projected_candidate_count,
        "occluded_pixel_count": occluded_pixel_count,
        "histogram_counts": histogram_counts,
        "optimization_summary": optimization_summary,
        "relative_pose_to_source": relative_pose_stats,
        "estimated_target_pose_matrix": pose_matrix.tolist(),
        "source_intrinsics": copy.deepcopy(source_entry["intrinsics_summary"]),
        "target_intrinsics": copy.deepcopy(target_entry["intrinsics_summary"]),
    }

    heatmap_frame = save_dynamic_error_heatmap(
        None,
        source_entry["display_image"],
        evaluation["squared_error_map"],
        valid_mask,
    )
    return pair_record, heatmap_frame


def aggregate_pair_results(pair_results: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in pair_results if item.get(key) is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_histogram_counts(pair_results: list[dict[str, Any]]) -> list[float] | None:
    histogram_arrays = [
        np.asarray(item["histogram_counts"], dtype=np.float64)
        for item in pair_results
        if item.get("histogram_counts") is not None
    ]
    if not histogram_arrays:
        return None
    return np.mean(np.stack(histogram_arrays, axis=0), axis=0).tolist()


def build_source_result(
    source_result_index: int,
    window_index: int,
    source_entry: dict[str, Any],
    source_pair_results: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_pair_results = [
        pair_result for pair_result in source_pair_results if pair_result.get("photometric_error") is not None
    ]
    histogram_counts = aggregate_histogram_counts(valid_pair_results)

    frame_record = create_frame_record_base(
        index=source_result_index,
        image_path=source_entry["image_path"],
        frame_output_dir=None,
        depth_valid_ratio=float(source_entry["depth_valid_ratio"]),
        pose_matrix=source_entry["frame_data"].pose.as_matrix(),
    )
    frame_record.update(
        {
            "frame_role": "window_source",
            "tracking_mode": "window_pairwise_stride",
            "window_index": window_index,
            "source_frame_index": int(source_entry["index"]),
            "reference_frame_name": None,
            "compared_frame_names": [item["target_frame_name"] for item in source_pair_results],
            "comparison_count": len(source_pair_results),
            "valid_comparison_count": len(valid_pair_results),
            "frame_intrinsics": copy.deepcopy(source_entry["intrinsics_summary"]),
            "photometric_error": aggregate_pair_results(valid_pair_results, "photometric_error"),
            "photometric_error_sum": aggregate_pair_results(valid_pair_results, "photometric_error_sum"),
            "photometric_error_mean": aggregate_pair_results(valid_pair_results, "photometric_error_mean_abs"),
            "photometric_error_mean_abs": aggregate_pair_results(valid_pair_results, "photometric_error_mean_abs"),
            "photometric_error_cumulative": None,
            "valid_pixel_count": aggregate_pair_results(valid_pair_results, "valid_pixel_count"),
            "projected_candidate_count": aggregate_pair_results(valid_pair_results, "projected_candidate_count"),
            "occluded_pixel_count": aggregate_pair_results(valid_pair_results, "occluded_pixel_count"),
            "histogram_counts": histogram_counts,
            "pair_results": source_pair_results,
            "pose_is_finite": True,
        }
    )
    return frame_record


def run_sequence(
    frame_paths: list[Path],
    run_dir: Path,
    recap_dir: Path,
    depth_cache_dir: Path,
    intrinsics_by_frame: dict[str, dict[str, Any]],
    intrinsics_mode_summary: dict[str, Any],
    run_name: str,
    win_sec: float,
    max_windows: int,
    pair_stride: int,
    sans_slice: bool,
) -> None:
    prepared_frames = prepare_frame_entries(
        frame_paths=frame_paths,
        depth_cache_dir=depth_cache_dir,
        intrinsics_by_frame=intrinsics_by_frame,
    )
    windows, effective_window_length = build_fixed_len_cover_windows(
        frame_paths=frame_paths,
        win_sec=win_sec,
        max_windows=max_windows,
    )

    pair_plans: list[dict[str, Any]] = []
    for window in windows:
        source_target_map = build_source_target_map(
            window["frame_indices"],
            pair_stride,
            sans_slice=sans_slice,
        )
        pair_count = sum(len(target_indices) for target_indices in source_target_map.values())
        pair_plans.append(
            {
                "window": window,
                "source_target_map": source_target_map,
                "pair_count": pair_count,
            }
        )

    total_pairs = sum(item["pair_count"] for item in pair_plans)
    if total_pairs <= 0:
        raise ValueError("Aucune paire source/cible exploitable n'a ete generee.")

    source_results: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    window_results: list[dict[str, Any]] = []
    heatmap_video_frames: list[np.ndarray] = []
    source_result_index = 0
    pair_progress_index = 0

    for plan in pair_plans:
        window = plan["window"]
        source_target_map = plan["source_target_map"]
        window_pair_results: list[dict[str, Any]] = []
        window_source_results: list[dict[str, Any]] = []

        print(
            f"Window {window['window_index'] + 1}/{len(windows)} | "
            f"frames {window['start_index']}..{window['end_index']} | "
            f"count={len(window['frame_indices'])}"
        )

        for source_index in window["frame_indices"]:
            source_entry = prepared_frames[source_index]
            source_pair_results: list[dict[str, Any]] = []

            for target_index in source_target_map[source_index]:
                target_entry = prepared_frames[target_index]
                pair_progress_index += 1
                pair_record, heatmap_frame = evaluate_frame_pair(
                    source_entry=source_entry,
                    target_entry=target_entry,
                    window_index=window["window_index"],
                    pair_index=pair_progress_index,
                    total_pairs=total_pairs,
                    total_windows=len(windows),
                )
                source_pair_results.append(pair_record)
                window_pair_results.append(pair_record)
                pair_results.append(pair_record)
                heatmap_video_frames.append(heatmap_frame)

            source_result = build_source_result(
                source_result_index=source_result_index,
                window_index=window["window_index"],
                source_entry=source_entry,
                source_pair_results=source_pair_results,
            )
            source_result_index += 1
            source_results.append(source_result)
            window_source_results.append(source_result)

        valid_window_sources = [
            source_result for source_result in window_source_results if source_result.get("photometric_error") is not None
        ]
        window_histogram_counts = aggregate_histogram_counts(valid_window_sources)
        window_results.append(
            {
                "window_index": window["window_index"],
                "start_index": window["start_index"],
                "end_index": window["end_index"],
                "frame_names": [frame_paths[index].name for index in window["frame_indices"]],
                "pair_count": len(window_pair_results),
                "valid_pair_count": sum(
                    1 for pair_result in window_pair_results if pair_result.get("photometric_error") is not None
                ),
                "source_count": len(window_source_results),
                "valid_source_count": len(valid_window_sources),
                "photometric_error": aggregate_pair_results(valid_window_sources, "photometric_error"),
                "photometric_error_sum": aggregate_pair_results(valid_window_sources, "photometric_error_sum"),
                "photometric_error_mean_abs": aggregate_pair_results(
                    valid_window_sources, "photometric_error_mean_abs"
                ),
                "histogram_counts": window_histogram_counts,
            }
        )
        sys.stdout.write("\n")

    valid_histogram_results = [
        item for item in source_results if item.get("histogram_counts") is not None
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

    valid_results = [item for item in source_results if item["photometric_error"] is not None]
    if not valid_results:
        raise ValueError(
            "Aucun resultat valide disponible pour calculer les erreurs photometriques globales."
        )

    iteration_indices = list(range(len(valid_results)))
    photometric_errors = [float(item["photometric_error"]) for item in valid_results]
    mean_photometric_error = save_photometric_error_curve(
        recap_dir / "photometric_error_curve.png",
        run_name,
        iteration_indices,
        photometric_errors,
    )
    cumulative_photometric_error = float(np.sum(photometric_errors))
    photometric_error_sums = [float(item["photometric_error_sum"]) for item in valid_results if item["photometric_error_sum"] is not None]
    cumulative_photometric_error_sum = float(np.sum(photometric_error_sums))
    mean_photometric_error_sum = float(np.mean(photometric_error_sums))
    heatmap_video_path = recap_dir / "photometric_error_heatmap_video.mp4"
    save_heatmap_video(heatmap_video_path, heatmap_video_frames, fps=HEATMAP_VIDEO_FPS)

    running_total = 0.0
    for item in source_results:
        current_error = item.get("photometric_error")
        if current_error is None:
            item["photometric_error_cumulative"] = None
            continue
        running_total += float(current_error)
        item["photometric_error_cumulative"] = float(running_total)

    summary = {
        "run_name": run_name,
        "tracking_mode": "window_pairwise_stride",
        "project_root": str(PROJECT_ROOT),
        "input_dir": str(frame_paths[0].parent),
        "output_run_dir": str(run_dir),
        "recap_dir": str(recap_dir),
        "frames_processed": len(frame_paths),
        "sources_evaluated": len(source_results),
        "sources_with_valid_score": len(valid_results),
        "pairs_evaluated": len(pair_results),
        "windows_evaluated": len(window_results),
        "per_frame_outputs_saved": False,
        "depth_encoder": DEPTH_ENCODER,
        "depth_model_name": DEPTH_MODEL_NAME,
        "device": DEVICE,
        "windowing": {
            "requested_win_sec": win_sec,
            "effective_window_length_frames": effective_window_length,
            "max_windows": max_windows,
            "pair_stride": pair_stride,
            "sans_slice": bool(sans_slice),
        },
        "intrinsics": {
            "mode_summary": intrinsics_mode_summary,
            "first_frame_intrinsics": copy.deepcopy(prepared_frames[0]["intrinsics_summary"]),
            "last_frame_intrinsics": copy.deepcopy(prepared_frames[-1]["intrinsics_summary"]),
        },
        "photometric_metrics": {
            "cumulative_photometric_error": cumulative_photometric_error,
            "mean_photometric_error": mean_photometric_error,
            "metric_definition": "mean squared photometric error per visible pixel",
            "cumulative_photometric_error_sum": cumulative_photometric_error_sum,
            "mean_photometric_error_sum": mean_photometric_error_sum,
            "histogram_labels": HISTOGRAM_LABELS,
            "mean_histogram_counts": mean_histogram_counts.tolist(),
        },
        "output_files": {
            "command_file": str(run_dir / "commande.txt"),
            "mean_histogram": str(recap_dir / "mean_histogram.png"),
            "photometric_error_curve": str(recap_dir / "photometric_error_curve.png"),
            "photometric_error_heatmap_video": str(heatmap_video_path),
            "sequence_results_json": str(recap_dir / "sequence_results.json"),
        },
    }

    write_json(
        recap_dir / "sequence_results.json",
        {
            "source_results": source_results,
            "pair_results": pair_results,
            "window_results": window_results,
        },
    )
    write_json(recap_dir / "summary.json", summary)

    print(f"Output run dir: {run_dir}")
    print(f"Summary file: {recap_dir / 'summary.json'}")


def main() -> int:
    args = parse_args()

    input_dir = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_root)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    run_name = args.run_name or input_dir.name
    run_dir, recap_dir = ensure_run_directories(output_root, run_name, args.overwrite)
    command_file_path = run_dir / "commande.txt"
    write_launch_command(command_file_path)
    frame_paths = collect_frame_paths(input_dir, max_frames=args.max_frames)
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
        win_sec=args.win_sec,
        max_windows=args.max_windows,
        pair_stride=args.pair_stride,
        sans_slice=args.sans_slice,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
