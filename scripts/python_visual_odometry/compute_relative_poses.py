from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np


POSE_FILE_PATTERN = re.compile(r"scene_(\d{3})\.txt\.new$")


def load_pose_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"La pose {path} doit etre une matrice 4x4, recu: {matrix.shape}")
    return matrix


def save_pose_matrix(path: str | Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.9f")


def compute_relative_pose(reference_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    # Convention alignee sur le notebook:
    # relative_pose = pose_target @ inv(pose_reference)
    return target_pose @ np.linalg.inv(reference_pose)


def list_absolute_pose_files(poses_dir: str | Path) -> list[Path]:
    poses_dir = Path(poses_dir)
    pose_files = []

    for path in sorted(poses_dir.glob("scene_*.txt.new")):
        if POSE_FILE_PATTERN.match(path.name):
            pose_files.append(path)

    if not pose_files:
        raise FileNotFoundError(f"Aucun fichier scene_XXX.txt.new trouve dans {poses_dir}")

    return pose_files


def compute_and_save_relative_poses(poses_dir: str | Path) -> list[Path]:
    poses_dir = Path(poses_dir)
    absolute_pose_files = list_absolute_pose_files(poses_dir)
    written_files: list[Path] = []

    for current_path, next_path in zip(absolute_pose_files, absolute_pose_files[1:]):
        current_match = POSE_FILE_PATTERN.match(current_path.name)
        next_match = POSE_FILE_PATTERN.match(next_path.name)
        if current_match is None or next_match is None:
            continue

        current_index = current_match.group(1)
        next_index = next_match.group(1)

        current_pose = load_pose_matrix(current_path)
        next_pose = load_pose_matrix(next_path)
        relative_pose = compute_relative_pose(current_pose, next_pose)

        output_path = poses_dir / f"scene_{current_index}_vers_{next_index}.txt"
        save_pose_matrix(output_path, relative_pose)
        written_files.append(output_path)

    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcule toutes les poses relatives i -> i+1 a partir des fichiers scene_XXX.txt.new"
    )
    parser.add_argument(
        "poses_dir",
        help="Dossier contenant les poses absolues scene_XXX.txt.new",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_files = compute_and_save_relative_poses(args.poses_dir)

    print(f"{len(written_files)} poses relatives ecrites.")
    for path in written_files[:5]:
        print(path)
    if len(written_files) > 5:
        print("...")


if __name__ == "__main__":
    main()
