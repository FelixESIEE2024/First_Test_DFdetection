from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    return image


def estimate_intrinsics_from_image_size(
    image: np.ndarray,
    horizontal_fov_deg: float = 60.0,
) -> tuple[float, float, float, float, np.ndarray]:
    height, width = image.shape[:2]

    # Heuristique simple :
    # on suppose une camera pinhole avec un champ de vue horizontal donne.
    horizontal_fov_rad = np.deg2rad(horizontal_fov_deg)
    fx = width / (2.0 * np.tan(horizontal_fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    k = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return fx, fy, cx, cy, k


def print_intrinsics(image_path: str | Path, horizontal_fov_deg: float = 60.0) -> np.ndarray:
    image = load_image(image_path)
    height, width = image.shape[:2]
    fx, fy, cx, cy, k = estimate_intrinsics_from_image_size(
        image,
        horizontal_fov_deg=horizontal_fov_deg,
    )

    print(f"Image : {image_path}")
    print(f"Resolution : width={width}, height={height}")
    print(f"Hypothese FOV horizontal : {horizontal_fov_deg:.2f} deg")
    print(f"fx = {fx:.6f}")
    print(f"fy = {fy:.6f}")
    print(f"cx = {cx:.6f}")
    print(f"cy = {cy:.6f}")
    print("Intrinsic matrix K =")
    print(k)

    return k


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estime une matrice intrinsique a partir de la taille de l'image "
            "en supposant un FOV horizontal."
        )
    )
    parser.add_argument("image_path", help="Chemin vers l'image d'entree.")
    parser.add_argument(
        "--hfov",
        type=float,
        default=60.0,
        help="Champ de vue horizontal suppose en degres. Defaut: 60.0",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print_intrinsics(args.image_path, horizontal_fov_deg=args.hfov)
