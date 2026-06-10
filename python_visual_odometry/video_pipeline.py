from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DEPTH_ANYTHING_DIR = PROJECT_ROOT / "Depth-Anything"
LIEGROUPS_DIR = PROJECT_ROOT / "liegroups"

for extra_path in (MODULE_DIR, DEPTH_ANYTHING_DIR, LIEGROUPS_DIR):
    path_str = str(extra_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

import camera
import common
import frameData
import pose_estimator_gauss_newton
from extract_spaced_video_frames import compute_frame_indices, maybe_resize


DEPTH_ENCODER = "vitb"
DEPTH_MODEL_NAME = f"LiheYoung/depth_anything_{DEPTH_ENCODER}14"
DEPTH_CACHE_DIR = MODULE_DIR / "depth_anything_cache" / DEPTH_ENCODER
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_FX = 525.0
CAMERA_FY = 525.0
CAMERA_CX = 319.5
CAMERA_CY = 239.5
DEPTH_SCALE_FACTOR = 5000

@dataclass
class FrameExtractionResult:
    output_dir: Path
    frame_paths: list[Path]
    selected_indices: list[int]
    fps: float
    frame_size: tuple[int, int]


def normalize_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def read_image_gray(image_path: str | os.PathLike[str]) -> np.ndarray:
    path = normalize_path(image_path)
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {path}")
    return image


def read_image_rgb(image_path: str | os.PathLike[str]) -> np.ndarray:
    path = normalize_path(image_path)
    buffer = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_depth_cache_path(image_path: str | os.PathLike[str]) -> Path:
    path = normalize_path(image_path)
    safe_name = path.stem.replace(" ", "_")
    cache_key = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    return DEPTH_CACHE_DIR / f"{safe_name}_{cache_key}_depth.npy"


def get_depth_anything_components():
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


def infer_depth_map(
    image_path: str | os.PathLike[str],
    use_cache: bool = True,
    save_cache: bool = True,
) -> np.ndarray:
    path = normalize_path(image_path)
    cache_path = build_depth_cache_path(path)
    if use_cache and cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    image = read_image_rgb(path).astype(np.float32) / 255.0
    height, width = image.shape[:2]

    model, transform = get_depth_anything_components()
    network_input = transform({"image": image})["image"]
    network_input = torch.from_numpy(network_input).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = model(network_input)

    depth = F.interpolate(depth[None], (height, width), mode="bilinear", align_corners=False)[0, 0]
    depth = depth.cpu().numpy().astype(np.float32)

    if save_cache:
        np.save(cache_path, depth)

    return depth


def depth_to_inverse_depth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv_depth = np.zeros_like(depth, dtype=np.float32)
    valid_mask = depth > 1e-6
    inv_depth[valid_mask] = 1.0 / depth[valid_mask]
    inv_depth_var = np.ones_like(depth, dtype=np.float32)
    inv_depth_var[~valid_mask] = 1e6
    return inv_depth, inv_depth_var, valid_mask


def build_intrinsics(
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    fx = CAMERA_FX
    fy = CAMERA_FY
    cx = CAMERA_CX
    cy = CAMERA_CY
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Les distances focales fx et fy doivent etre strictement positives.")
    if not (0.0 <= cx < image_width) or not (0.0 <= cy < image_height):
        raise ValueError(
            "Le centre principal doit etre dans l'image: "
            f"cx in [0, {image_width}), cy in [0, {image_height})."
        )
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": int(image_width),
        "height": int(image_height),
    }


def extract_frames_from_video(
    video_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    frame_count: int = 10,
    start_frame: int = 0,
    min_gap: int = 10,
    resize_width: int | None = None,
    prefix: str = "scene",
) -> FrameExtractionResult:
    video = normalize_path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video introuvable: {video}")

    destination = (
        normalize_path(output_dir)
        if output_dir
        else video.parent / f"{video.stem}_frames_spaced"
    )
    destination.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    selected_indices = compute_frame_indices(
        total_frames=total_frames,
        count=frame_count,
        start_frame=start_frame,
        min_gap=min_gap,
    )

    frame_paths: list[Path] = []
    extracted_size: tuple[int, int] | None = None
    try:
        for export_id, frame_index in enumerate(selected_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Lecture impossible pour la frame {frame_index}.")

            frame = maybe_resize(frame, resize_width)
            if extracted_size is None:
                extracted_size = (int(frame.shape[1]), int(frame.shape[0]))

            output_path = destination / f"{prefix}_{export_id:03d}.png"
            success, encoded = cv2.imencode(".png", frame)
            if not success:
                raise RuntimeError(f"Impossible d'encoder l'image {output_path.name}.")
            output_path.write_bytes(encoded.tobytes())
            frame_paths.append(output_path)
    finally:
        cap.release()

    if extracted_size is None:
        raise RuntimeError("Aucune frame n'a ete exportee.")

    return FrameExtractionResult(
        output_dir=destination,
        frame_paths=frame_paths,
        selected_indices=selected_indices,
        fps=fps,
        frame_size=extracted_size,
    )


def compute_photometric_maps(frame, keyframe, cam, lvl: int = 0) -> dict[str, np.ndarray]:
    width = cam.width[lvl]
    height = cam.height[lvl]
    fx = cam.fx[lvl]
    fy = cam.fy[lvl]
    cx = cam.cx[lvl]
    cy = cam.cy[lvl]
    fxinv = cam.fxinv[lvl]
    fyinv = cam.fyinv[lvl]
    cxinv = cam.cxinv[lvl]
    cyinv = cam.cyinv[lvl]

    relative_pose = frame.pose.dot(keyframe.pose.inv())

    predicted_map = np.full((height, width), np.nan, dtype=np.float32)
    true_map = np.full((height, width), np.nan, dtype=np.float32)
    signed_diff_map = np.full((height, width), np.nan, dtype=np.float32)
    abs_diff_map = np.full((height, width), np.nan, dtype=np.float32)
    squared_diff_map = np.full((height, width), np.nan, dtype=np.float32)
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
            if pixel_frame[0] < 1.0 or pixel_frame[0] >= width - 1 or pixel_frame[1] < 1.0 or pixel_frame[1] >= height - 1:
                continue

            key_intensity = float(keyframe.image[lvl][y, x])
            observed_intensity = float(common.getSubPixelValue(frame.image[lvl], pixel_frame))

            predicted_map[y, x] = key_intensity
            true_map[y, x] = observed_intensity
            signed_diff_map[y, x] = key_intensity - observed_intensity
            abs_diff_map[y, x] = abs(key_intensity - observed_intensity)
            squared_diff_map[y, x] = (key_intensity - observed_intensity) ** 2
            valid_mask[y, x] = True

    return {
        "predicted_map": predicted_map,
        "true_map": true_map,
        "signed_diff_map": signed_diff_map,
        "abs_diff_map": abs_diff_map,
        "squared_diff_map": squared_diff_map,
        "valid_mask": valid_mask,
    }


def normalize_to_uint8(image: np.ndarray, nan_color: int = 0) -> np.ndarray:
    result = np.zeros(image.shape, dtype=np.uint8)
    valid = np.isfinite(image)
    if not np.any(valid):
        return result

    values = image[valid].astype(np.float32)
    vmin = float(np.nanpercentile(values, 1))
    vmax = float(np.nanpercentile(values, 99))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0

    scaled = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)
    result[valid] = (scaled[valid] * 255.0).astype(np.uint8)
    result[~valid] = nan_color
    return result


def colorize_grayscale_map(image: np.ndarray, colormap: int) -> np.ndarray:
    normalized = normalize_to_uint8(image)
    colored = cv2.applyColorMap(normalized, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def format_pose_matrix(matrix: np.ndarray) -> str:
    lines = ["[" + "  ".join(f"{value: .6f}" for value in row) + "]" for row in matrix]
    return "\n".join(lines)


def build_intrinsics_matrix(intrinsics: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def run_video_pipeline(
    video_path: str,
    output_dir: str | None = None,
    frame_count: int = 10,
    target_frame_index: int = 1,
    min_gap: int = 10,
    start_frame: int = 0,
    resize_width: int | None = None,
    use_cache: bool = True,
    save_cache: bool = True,
) -> dict[str, Any]:
    if target_frame_index <= 0:
        raise ValueError("La frame cible doit etre > 0 pour comparer au keyframe.")
    if frame_count < 2:
        raise ValueError("Il faut extraire au moins 2 frames.")
    if target_frame_index >= frame_count:
        raise ValueError("La frame cible doit etre strictement inferieure au nombre de frames extraites.")

    extraction = extract_frames_from_video(
        video_path=video_path,
        output_dir=output_dir,
        frame_count=frame_count,
        start_frame=start_frame,
        min_gap=min_gap,
        resize_width=resize_width,
    )

    keyframe_path = extraction.frame_paths[0]
    target_path = extraction.frame_paths[target_frame_index]

    keyframe_image = read_image_gray(keyframe_path)
    current_image = read_image_gray(target_path)
    keyframe_depth = infer_depth_map(keyframe_path, use_cache=use_cache, save_cache=save_cache)
    keyframe_inv_depth, keyframe_inv_depth_var, depth_valid_mask = depth_to_inverse_depth(keyframe_depth)

    intrinsics = build_intrinsics(
        image_width=keyframe_image.shape[1],
        image_height=keyframe_image.shape[0],
    )
    cam = camera.camera(
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
        intrinsics["width"],
        intrinsics["height"],
    )

    keyframe = frameData.frameData()
    keyframe.setImage(keyframe_image)
    keyframe.setInvDepth(keyframe_inv_depth, keyframe_inv_depth_var)

    current_frame = frameData.frameData()
    current_frame.setImage(current_image)

    pose_solver = pose_estimator_gauss_newton.pose_estimator_gauss_newton(cam, show_debug=False)

    initial_error_lvl4, _ = pose_solver.computeError(current_frame, keyframe, lvl=4)
    initial_error_lvl3, _ = pose_solver.computeError(current_frame, keyframe, lvl=3)
    initial_error_lvl2, _ = pose_solver.computeError(current_frame, keyframe, lvl=2)

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        pose_solver.optPose(current_frame, keyframe)

    final_error_lvl4, _ = pose_solver.computeError(current_frame, keyframe, lvl=4)
    final_error_lvl3, _ = pose_solver.computeError(current_frame, keyframe, lvl=3)
    final_error_lvl2, residual_error_map_lvl2 = pose_solver.computeError(current_frame, keyframe, lvl=2)

    photometric_maps = compute_photometric_maps(current_frame, keyframe, cam, lvl=0)
    signed_diff_map = photometric_maps["signed_diff_map"]
    abs_diff_map = photometric_maps["abs_diff_map"]
    squared_diff_map = photometric_maps["squared_diff_map"]
    valid_mask = photometric_maps["valid_mask"]

    valid_pixels = int(valid_mask.sum())
    if valid_pixels == 0:
        raise RuntimeError("Aucun pixel valide apres reprojection. Essaie un autre ecart entre les frames.")

    pose_matrix = current_frame.pose.as_matrix()
    intrinsics_matrix = build_intrinsics_matrix(intrinsics)

    metrics = {
        "valid_reprojected_pixels": valid_pixels,
        "depth_valid_ratio": float(depth_valid_mask.mean()),
        "initial_error_lvl4": float(initial_error_lvl4),
        "initial_error_lvl3": float(initial_error_lvl3),
        "initial_error_lvl2": float(initial_error_lvl2),
        "final_error_lvl4": float(final_error_lvl4),
        "final_error_lvl3": float(final_error_lvl3),
        "final_error_lvl2": float(final_error_lvl2),
        "mean_signed_difference": float(np.nanmean(signed_diff_map)),
        "mean_absolute_difference": float(np.nanmean(abs_diff_map)),
        "mean_squared_error": float(np.nanmean(squared_diff_map)),
        "max_absolute_difference": float(np.nanmax(abs_diff_map)),
        "max_squared_difference": float(np.nanmax(squared_diff_map)),
    }

    visuals = {
        "keyframe_image": gray_to_rgb(keyframe_image),
        "target_image": gray_to_rgb(current_image),
        "depth_map": colorize_grayscale_map(keyframe_depth, cv2.COLORMAP_INFERNO),
        "inverse_depth_map": colorize_grayscale_map(keyframe_inv_depth, cv2.COLORMAP_MAGMA),
        "signed_difference_map": colorize_grayscale_map(signed_diff_map, cv2.COLORMAP_TURBO),
        "squared_difference_map": colorize_grayscale_map(squared_diff_map, cv2.COLORMAP_HOT),
        "residual_error_map_lvl2": colorize_grayscale_map(residual_error_map_lvl2, cv2.COLORMAP_HOT),
    }

    extracted_gallery = [(str(path), path.name) for path in extraction.frame_paths]

    return {
        "video_path": str(normalize_path(video_path)),
        "frames_dir": str(extraction.output_dir),
        "frame_paths": [str(path) for path in extraction.frame_paths],
        "selected_video_indices": extraction.selected_indices,
        "selected_timestamps_seconds": [
            float(index / extraction.fps) if extraction.fps > 0 else 0.0
            for index in extraction.selected_indices
        ],
        "keyframe_path": str(keyframe_path),
        "target_path": str(target_path),
        "keyframe_index": 0,
        "target_index": int(target_frame_index),
        "fps": float(extraction.fps),
        "intrinsics": intrinsics,
        "intrinsics_matrix": intrinsics_matrix.tolist(),
        "intrinsics_matrix_text": format_pose_matrix(intrinsics_matrix),
        "pose_matrix": pose_matrix.tolist(),
        "pose_matrix_text": format_pose_matrix(pose_matrix),
        "metrics": metrics,
        "visuals": visuals,
        "extracted_gallery": extracted_gallery,
        "logs": log_buffer.getvalue().strip(),
    }
