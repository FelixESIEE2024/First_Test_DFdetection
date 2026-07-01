from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


MIN_MATCHES_PER_PAIR = 40
MIN_TRACKS_FOR_BA = 25
MAX_TRACKS_FOR_BA = 40
RANSAC_THRESHOLD_PIXELS = 1.0


@dataclass
class PairwiseGeometry:
    points_a: np.ndarray
    points_b: np.ndarray
    keypoints_a_idx: np.ndarray
    keypoints_b_idx: np.ndarray
    fundamental: np.ndarray
    inlier_mask: np.ndarray


def load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    return image


def _ensure_grayscale_uint8(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("Une frame est None.")
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame.copy()
    else:
        raise ValueError("Chaque frame doit etre une image 2D ou 3D.")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _extract_sift_features(frame: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    if descriptors is None or len(keypoints) < MIN_MATCHES_PER_PAIR:
        raise ValueError("Pas assez de points SIFT detectes pour une estimation robuste.")
    return keypoints, descriptors


def _match_descriptors_flann(descriptors_a: np.ndarray, descriptors_b: np.ndarray) -> list[cv2.DMatch]:
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    good_matches: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches


def _estimate_pairwise_geometry(
    keypoints_a: list[cv2.KeyPoint],
    keypoints_b: list[cv2.KeyPoint],
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
) -> PairwiseGeometry:
    matches = _match_descriptors_flann(descriptors_a, descriptors_b)
    if len(matches) < MIN_MATCHES_PER_PAIR:
        raise ValueError(
            f"Pas assez de correspondances apres FLANN + Lowe ratio test: {len(matches)}."
        )

    points_a = np.float64([keypoints_a[m.queryIdx].pt for m in matches])
    points_b = np.float64([keypoints_b[m.trainIdx].pt for m in matches])
    keypoints_a_idx = np.int32([m.queryIdx for m in matches])
    keypoints_b_idx = np.int32([m.trainIdx for m in matches])

    fundamental, mask = cv2.findFundamentalMat(
        points_a,
        points_b,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PIXELS,
        confidence=0.999,
    )

    if fundamental is None or mask is None:
        raise ValueError("Echec de l'estimation de la matrice fondamentale par RANSAC.")

    mask = mask.ravel().astype(bool)
    points_a = points_a[mask]
    points_b = points_b[mask]
    keypoints_a_idx = keypoints_a_idx[mask]
    keypoints_b_idx = keypoints_b_idx[mask]

    if len(points_a) < MIN_MATCHES_PER_PAIR:
        raise ValueError(
            f"Pas assez d'inliers apres RANSAC: {len(points_a)} correspondances."
        )

    return PairwiseGeometry(
        points_a=points_a,
        points_b=points_b,
        keypoints_a_idx=keypoints_a_idx,
        keypoints_b_idx=keypoints_b_idx,
        fundamental=fundamental,
        inlier_mask=mask,
    )


def _build_three_view_tracks(pair_12: PairwiseGeometry, pair_23: PairwiseGeometry) -> np.ndarray:
    frame2_to_frame1 = {
        int(idx2): i for i, idx2 in enumerate(pair_12.keypoints_b_idx.tolist())
    }
    frame2_to_frame3 = {
        int(idx2): i for i, idx2 in enumerate(pair_23.keypoints_a_idx.tolist())
    }

    shared_frame2_indices = sorted(set(frame2_to_frame1) & set(frame2_to_frame3))
    if len(shared_frame2_indices) < MIN_TRACKS_FOR_BA:
        raise ValueError(
            f"Pas assez de tracks visibles dans 3 vues: {len(shared_frame2_indices)}."
        )

    tracks = []
    for frame2_idx in shared_frame2_indices:
        i12 = frame2_to_frame1[frame2_idx]
        i23 = frame2_to_frame3[frame2_idx]
        track = np.array(
            [
                pair_12.points_a[i12],
                pair_12.points_b[i12],
                pair_23.points_b[i23],
            ],
            dtype=np.float64,
        )
        tracks.append(track)
    return np.asarray(tracks, dtype=np.float64)


def _select_tracks_for_bundle_adjustment(tracks: np.ndarray) -> np.ndarray:
    if len(tracks) <= MAX_TRACKS_FOR_BA:
        return tracks

    # On sous-echantillonne regulierement les tracks pour garder une optimisation
    # plus stable et beaucoup plus rapide.
    indices = np.linspace(0, len(tracks) - 1, MAX_TRACKS_FOR_BA, dtype=int)
    return tracks[indices]


def _make_intrinsics(width: int, height: int, focal: float) -> np.ndarray:
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _recover_pose_from_fundamental(
    fundamental: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    width: int,
    height: int,
    focal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = _make_intrinsics(width, height, focal)
    essential = k.T @ fundamental @ k

    # La matrice essentielle definie la geometrie epipolaire entre deux vues
    # normalisees. recoverPose choisit la decomposition [R|t] physiquement coherente.
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        points_a,
        points_b,
        cameraMatrix=k,
    )
    return rotation, translation.reshape(3), pose_mask.ravel().astype(bool)


def _score_focal_candidate(
    focal: float,
    pair_12: PairwiseGeometry,
    pair_23: PairwiseGeometry,
    width: int,
    height: int,
) -> float:
    try:
        r12, t12, mask12 = _recover_pose_from_fundamental(
            pair_12.fundamental,
            pair_12.points_a,
            pair_12.points_b,
            width,
            height,
            focal,
        )
        r23, t23, mask23 = _recover_pose_from_fundamental(
            pair_23.fundamental,
            pair_23.points_a,
            pair_23.points_b,
            width,
            height,
            focal,
        )
    except cv2.error:
        return -np.inf

    inlier_score = float(mask12.sum() + mask23.sum())

    # On penalisera les focales qui generent des poses trop incoherentes.
    t12_norm = np.linalg.norm(t12)
    t23_norm = np.linalg.norm(t23)
    if t12_norm < 1e-8 or t23_norm < 1e-8:
        return -np.inf

    rotation_penalty = abs(np.linalg.det(r12) - 1.0) + abs(np.linalg.det(r23) - 1.0)
    return inlier_score - 1000.0 * rotation_penalty


def _estimate_initial_focal(
    pair_12: PairwiseGeometry,
    pair_23: PairwiseGeometry,
    width: int,
    height: int,
) -> float:
    heuristic_focal = max(width, height) * 1.2

    candidate_focals = np.geomspace(
        max(0.3 * heuristic_focal, 50.0),
        3.0 * heuristic_focal,
        num=25,
    )

    best_score = -np.inf
    best_focal = heuristic_focal
    for focal in candidate_focals:
        score = _score_focal_candidate(focal, pair_12, pair_23, width, height)
        if score > best_score:
            best_score = score
            best_focal = float(focal)

    return best_focal


def _compose_pose(
    rotation_12: np.ndarray,
    translation_12: np.ndarray,
    rotation_23: np.ndarray,
    translation_23: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_13 = rotation_23 @ rotation_12
    translation_13 = rotation_23 @ translation_12 + translation_23
    return rotation_13, translation_13


def _triangulate_points(
    points_1: np.ndarray,
    points_2: np.ndarray,
    focal: float,
    width: int,
    height: int,
    rotation_12: np.ndarray,
    translation_12: np.ndarray,
) -> np.ndarray:
    k = _make_intrinsics(width, height, focal)
    p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k @ np.hstack([rotation_12, translation_12.reshape(3, 1)])

    # Triangulation lineaire: on reconstruit le point 3D qui explique le mieux
    # les observations 2D dans les deux premieres vues.
    points_4d_h = cv2.triangulatePoints(p1, p2, points_1.T, points_2.T)
    points_3d = (points_4d_h[:3] / points_4d_h[3]).T
    return points_3d


def _project_points(
    points_3d: np.ndarray,
    focal: float,
    width: int,
    height: int,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    points_cam = (rotation @ points_3d.T).T + translation.reshape(1, 3)
    z = points_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-8, 1e-8, z)

    cx = width / 2.0
    cy = height / 2.0
    u = focal * (points_cam[:, 0:1] / z) + cx
    v = focal * (points_cam[:, 1:2] / z) + cy
    return np.hstack([u, v])


def _bundle_adjustment_residuals(
    params: np.ndarray,
    observations: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    num_points = observations.shape[0]

    focal = float(params[0])
    rvec_12 = params[1:4]
    tvec_12 = params[4:7]
    rvec_13 = params[7:10]
    tvec_13 = params[10:13]
    points_3d = params[13:].reshape(num_points, 3)

    rotation_12, _ = cv2.Rodrigues(rvec_12)
    rotation_13, _ = cv2.Rodrigues(rvec_13)

    residuals = []

    # Vue 1: pose canonique [I|0].
    reproj_1 = _project_points(
        points_3d,
        focal,
        width,
        height,
        np.eye(3),
        np.zeros(3),
    )
    reproj_2 = _project_points(points_3d, focal, width, height, rotation_12, tvec_12)
    reproj_3 = _project_points(points_3d, focal, width, height, rotation_13, tvec_13)

    residuals.append((reproj_1 - observations[:, 0, :]).ravel())
    residuals.append((reproj_2 - observations[:, 1, :]).ravel())
    residuals.append((reproj_3 - observations[:, 2, :]).ravel())

    # Penalite douce sur les profondeurs negatives pour eviter des solutions non physiques.
    depth_1 = points_3d[:, 2]
    depth_2 = ((rotation_12 @ points_3d.T).T + tvec_12.reshape(1, 3))[:, 2]
    depth_3 = ((rotation_13 @ points_3d.T).T + tvec_13.reshape(1, 3))[:, 2]

    for depth in (depth_1, depth_2, depth_3):
        residuals.append(np.minimum(depth - 1e-3, 0.0) * 100.0)

    return np.concatenate(residuals)


def estimate_camera_intrinsics_from_frames(frames: list) -> np.ndarray:
    """
    Estime la matrice des parametres intrinseques K a partir d'au moins 3 frames
    d'une scene statique via un pipeline d'auto-calibrage base sur:
    - SIFT + FLANN
    - filtrage geometrique par RANSAC
    - estimation initiale de la focale
    - optimisation globale par bundle adjustment

    Parametres
    ----------
    frames:
        Liste de 3 frames ou plus. Chaque frame doit etre un ndarray OpenCV
        representant une image grayscale ou BGR.

    Retour
    ------
    np.ndarray
        Matrice intrinseque 3x3:
        [[f, 0, cx],
         [0, f, cy],
         [0, 0,  1]]

    Notes
    -----
    - On suppose des pixels carres.
    - On suppose le point principal au centre de l'image.
    - La scene doit etre statique, avec variation de point de vue entre les frames.
    """
    if len(frames) < 3:
        raise ValueError("Il faut au moins 3 frames pour estimer les intrinsics.")

    gray_frames = [_ensure_grayscale_uint8(frame) for frame in frames[:3]]
    height, width = gray_frames[0].shape[:2]

    for frame in gray_frames[1:]:
        if frame.shape[:2] != (height, width):
            raise ValueError("Toutes les frames doivent avoir la meme resolution.")

    kp1, desc1 = _extract_sift_features(gray_frames[0])
    kp2, desc2 = _extract_sift_features(gray_frames[1])
    kp3, desc3 = _extract_sift_features(gray_frames[2])

    pair_12 = _estimate_pairwise_geometry(kp1, kp2, desc1, desc2)
    pair_23 = _estimate_pairwise_geometry(kp2, kp3, desc2, desc3)
    tracks_123 = _build_three_view_tracks(pair_12, pair_23)
    tracks_123 = _select_tracks_for_bundle_adjustment(tracks_123)

    initial_focal = _estimate_initial_focal(pair_12, pair_23, width, height)

    rotation_12, translation_12, _ = _recover_pose_from_fundamental(
        pair_12.fundamental,
        tracks_123[:, 0, :],
        tracks_123[:, 1, :],
        width,
        height,
        initial_focal,
    )
    rotation_23, translation_23, _ = _recover_pose_from_fundamental(
        pair_23.fundamental,
        tracks_123[:, 1, :],
        tracks_123[:, 2, :],
        width,
        height,
        initial_focal,
    )
    rotation_13, translation_13 = _compose_pose(
        rotation_12,
        translation_12,
        rotation_23,
        translation_23,
    )

    points_3d_init = _triangulate_points(
        tracks_123[:, 0, :],
        tracks_123[:, 1, :],
        initial_focal,
        width,
        height,
        rotation_12,
        translation_12,
    )

    if not np.all(np.isfinite(points_3d_init)):
        raise ValueError("La triangulation initiale a produit des points non finis.")

    rvec_12, _ = cv2.Rodrigues(rotation_12)
    rvec_13, _ = cv2.Rodrigues(rotation_13)

    initial_params = np.concatenate(
        [
            np.array([initial_focal], dtype=np.float64),
            rvec_12.ravel(),
            translation_12.astype(np.float64).ravel(),
            rvec_13.ravel(),
            translation_13.astype(np.float64).ravel(),
            points_3d_init.astype(np.float64).ravel(),
        ]
    )

    lower_bounds = np.concatenate(
        [
            np.array([max(50.0, 0.1 * max(width, height))]),
            np.full(12, -np.inf),
            np.full(points_3d_init.size, -np.inf),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.array([10.0 * max(width, height)]),
            np.full(12, np.inf),
            np.full(points_3d_init.size, np.inf),
        ]
    )

    # Bundle adjustment:
    # on optimise simultanement la focale, les poses des cameras 2 et 3,
    # et les points 3D pour minimiser l'erreur de reprojection globale.
    result = least_squares(
        _bundle_adjustment_residuals,
        initial_params,
        bounds=(lower_bounds, upper_bounds),
        args=(tracks_123, width, height),
        method="trf",
        loss="huber",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=400,
        verbose=0,
    )

    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("Le bundle adjustment a produit des parametres non finis.")

    if not result.success:
        warnings.warn(
            f"Bundle adjustment non complet, meilleure solution courante retournee: {result.message}",
            RuntimeWarning,
        )

    optimized_focal = float(result.x[0])
    return _make_intrinsics(width, height, optimized_focal)


def print_intrinsics_from_frames(frames: list) -> np.ndarray:
    k = estimate_camera_intrinsics_from_frames(frames)
    print("Intrinsic matrix K =")
    print(k)
    print(f"fx = {k[0, 0]:.6f}")
    print(f"fy = {k[1, 1]:.6f}")
    print(f"cx = {k[0, 2]:.6f}")
    print(f"cy = {k[1, 2]:.6f}")
    return k


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estime une matrice intrinsique a partir d'au moins 3 images d'une "
            "scene statique via SIFT, RANSAC et bundle adjustment."
        )
    )
    parser.add_argument(
        "image_paths",
        nargs="+",
        help="Chemins vers 3 images ou plus. Les 3 premieres sont utilisees.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    frames = [load_image(path) for path in args.image_paths]
    print_intrinsics_from_frames(frames)
