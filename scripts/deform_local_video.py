from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CODEC = "mp4v"
DEFAULT_FPS = 25.0


@dataclass
class LocalDeformer:
    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    sigma_px: float
    amplitude_px: float
    phase_x: float
    phase_y: float
    phase_amp: float
    omega_x: float
    omega_y: float
    omega_amp: float
    lifetime_frames: int
    age_frames: int = 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("La valeur doit etre strictement positive.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("La valeur doit etre strictement positive.")
    return parsed


def ratio_01(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("La valeur doit etre comprise entre 0 et 1.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Applique des deformations locales, legeres et temporellement instables "
            "sur une video afin de creer des incoherences subtiles entre les frames."
        )
    )
    parser.add_argument("--input", required=True, help="Chemin de la video d'entree.")
    parser.add_argument("--output", required=True, help="Chemin de la video deformee de sortie.")
    parser.add_argument(
        "--regions",
        type=positive_int,
        default=5,
        help="Nombre de zones locales de deformation actives en meme temps.",
    )
    parser.add_argument(
        "--strength-px",
        type=positive_float,
        default=6.0,
        help="Amplitude moyenne des deformations locales, en pixels.",
    )
    parser.add_argument(
        "--radius-min-ratio",
        type=positive_float,
        default=0.07,
        help="Rayon minimal d'une zone deformee, ratio de la plus petite dimension.",
    )
    parser.add_argument(
        "--radius-max-ratio",
        type=positive_float,
        default=0.16,
        help="Rayon maximal d'une zone deformee, ratio de la plus petite dimension.",
    )
    parser.add_argument(
        "--speed-px",
        type=positive_float,
        default=1.20,
        help="Vitesse maximale de derive du centre d'une zone, en pixels par frame.",
    )
    parser.add_argument(
        "--drift-px",
        type=positive_float,
        default=0.28,
        help="Petite derive aleatoire ajoutee a chaque frame.",
    )
    parser.add_argument(
        "--jitter-ratio",
        type=ratio_01,
        default=0.35,
        dest="jitter_ratio",
        help=(
            "Jitter relatif applique a l'intensite instantanee de deformation. "
            "0 = tres stable, 1 = plus nerveux."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="Optionnel: limite le traitement aux N premieres frames.",
    )
    parser.add_argument(
        "--deform-start-frame",
        type=positive_int,
        default=1,
        help=(
            "Premiere frame deformee, en index 1-based. "
            "Exemple: 20 signifie que la 20eme frame est la premiere deformee."
        ),
    )
    parser.add_argument(
        "--deform-end-frame",
        type=positive_int,
        default=None,
        help=(
            "Derniere frame deformee, en index 1-based et incluse. "
            "Exemple: 35 signifie que la 35eme frame est encore deformee."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed aleatoire pour reproduire exactement la meme deformation.",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        help="Codec OpenCV/FourCC de sortie. Par defaut: mp4v.",
    )
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=25,
        help="Affiche une ligne de progression toutes les N frames.",
    )
    return parser


def clamp_margin(length: int, requested_margin: float) -> float:
    if length <= 2:
        return 0.0
    return min(requested_margin, max(0.0, (length - 1.0) * 0.45))


def spawn_deformer(
    width: int,
    height: int,
    radius_range_px: tuple[float, float],
    strength_px: float,
    speed_px: float,
    rng: np.random.Generator,
) -> LocalDeformer:
    sigma_px = float(rng.uniform(radius_range_px[0], radius_range_px[1]))
    margin_x = clamp_margin(width, max(6.0, sigma_px * 1.15))
    margin_y = clamp_margin(height, max(6.0, sigma_px * 1.15))
    center_x = float(rng.uniform(margin_x, max(margin_x + 1e-3, width - 1.0 - margin_x)))
    center_y = float(rng.uniform(margin_y, max(margin_y + 1e-3, height - 1.0 - margin_y)))

    base_speed = float(rng.uniform(0.15, speed_px))
    direction = float(rng.uniform(0.0, 2.0 * math.pi))
    velocity_x = base_speed * math.cos(direction)
    velocity_y = base_speed * math.sin(direction)

    amplitude_px = float(strength_px * rng.uniform(0.65, 1.25))
    lifetime_frames = int(rng.integers(24, 96))

    return LocalDeformer(
        center_x=center_x,
        center_y=center_y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        sigma_px=sigma_px,
        amplitude_px=amplitude_px,
        phase_x=float(rng.uniform(0.0, 2.0 * math.pi)),
        phase_y=float(rng.uniform(0.0, 2.0 * math.pi)),
        phase_amp=float(rng.uniform(0.0, 2.0 * math.pi)),
        omega_x=float(rng.uniform(0.045, 0.140)),
        omega_y=float(rng.uniform(0.045, 0.140)),
        omega_amp=float(rng.uniform(0.025, 0.090)),
        lifetime_frames=lifetime_frames,
    )


def update_deformer(
    deformer: LocalDeformer,
    width: int,
    height: int,
    radius_range_px: tuple[float, float],
    strength_px: float,
    speed_px: float,
    drift_px: float,
    rng: np.random.Generator,
) -> LocalDeformer:
    deformer.age_frames += 1
    if deformer.age_frames >= deformer.lifetime_frames:
        return spawn_deformer(width, height, radius_range_px, strength_px, speed_px, rng)

    deformer.velocity_x += float(rng.normal(0.0, drift_px))
    deformer.velocity_y += float(rng.normal(0.0, drift_px))

    current_speed = math.hypot(deformer.velocity_x, deformer.velocity_y)
    if current_speed > speed_px:
        scale = speed_px / current_speed
        deformer.velocity_x *= scale
        deformer.velocity_y *= scale

    deformer.center_x += deformer.velocity_x
    deformer.center_y += deformer.velocity_y

    margin_x = clamp_margin(width, max(6.0, deformer.sigma_px * 1.05))
    margin_y = clamp_margin(height, max(6.0, deformer.sigma_px * 1.05))
    max_x = max(margin_x + 1e-3, width - 1.0 - margin_x)
    max_y = max(margin_y + 1e-3, height - 1.0 - margin_y)

    if deformer.center_x < margin_x or deformer.center_x > max_x:
        deformer.center_x = float(np.clip(deformer.center_x, margin_x, max_x))
        deformer.velocity_x *= -1.0

    if deformer.center_y < margin_y or deformer.center_y > max_y:
        deformer.center_y = float(np.clip(deformer.center_y, margin_y, max_y))
        deformer.velocity_y *= -1.0

    return deformer


def build_warp_maps(
    width: int,
    height: int,
    frame_index: int,
    deformers: list[LocalDeformer],
    jitter_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    for deformer in deformers:
        dx = grid_x - deformer.center_x
        dy = grid_y - deformer.center_y
        inv_sigma_sq = -1.0 / (2.0 * deformer.sigma_px * deformer.sigma_px)
        weight = np.exp((dx * dx + dy * dy) * inv_sigma_sq).astype(np.float32)

        envelope = 0.55 + 0.45 * math.sin(deformer.phase_amp + frame_index * deformer.omega_amp)
        jitter_scale = 1.0 + float(rng.normal(0.0, jitter_ratio * 0.20))
        jitter_scale = max(0.35, jitter_scale)
        amplitude = deformer.amplitude_px * envelope * jitter_scale

        shift_x = amplitude * math.sin(deformer.phase_x + frame_index * deformer.omega_x)
        shift_y = amplitude * math.cos(deformer.phase_y + frame_index * deformer.omega_y)

        map_x += weight * shift_x
        map_y += weight * shift_y

    np.clip(map_x, 0.0, width - 1.0, out=map_x)
    np.clip(map_y, 0.0, height - 1.0, out=map_y)
    return map_x, map_y


def deform_frame(frame: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def process_video(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if input_path == output_path:
        raise ValueError("Le fichier d'entree et le fichier de sortie doivent etre differents.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Video d'entree introuvable: {input_path}")
    if args.radius_min_ratio >= args.radius_max_ratio:
        raise ValueError("--radius-min-ratio doit etre strictement inferieur a --radius-max-ratio.")
    if args.deform_end_frame is not None and args.deform_end_frame < args.deform_start_frame:
        raise ValueError("--deform-end-frame doit etre >= --deform-start-frame.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video d'entree: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Dimensions video invalides. Impossible de lire largeur/hauteur.")
    if fps <= 0.0:
        fps = DEFAULT_FPS

    input_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if input_frame_count <= 0 and args.max_frames is None:
        frames_to_process = None
    elif args.max_frames is None:
        frames_to_process = input_frame_count
    elif input_frame_count <= 0:
        frames_to_process = args.max_frames
    else:
        frames_to_process = min(input_frame_count, args.max_frames)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Impossible d'ouvrir la video de sortie: {output_path}. "
            f"Essaie par exemple --codec {DEFAULT_CODEC}."
        )

    min_side = float(min(width, height))
    radius_range_px = (
        min_side * args.radius_min_ratio,
        min_side * args.radius_max_ratio,
    )
    rng = np.random.default_rng(args.seed)
    deformers = [
        spawn_deformer(width, height, radius_range_px, args.strength_px, args.speed_px, rng)
        for _ in range(args.regions)
    ]

    print("Local video deformation")
    print(f"  input  = {input_path}")
    print(f"  output = {output_path}")
    print(f"  size   = {width} x {height}")
    print(f"  fps    = {fps:.3f}")
    print(f"  regions= {args.regions}")
    print(f"  strength_px = {args.strength_px:.2f}")
    print(f"  seed   = {args.seed}")
    if args.deform_end_frame is None:
        print(f"  deformed frames = [{args.deform_start_frame}, fin]")
    else:
        print(f"  deformed frames = [{args.deform_start_frame}, {args.deform_end_frame}]")
    if frames_to_process:
        print(f"  frames = {frames_to_process}")
    elif input_frame_count > 0:
        print(f"  frames = {input_frame_count}")
    else:
        print("  frames = inconnu (lecture jusqu'a la fin du flux)")

    frame_index = 0
    written_frames = 0

    try:
        while True:
            if frames_to_process is not None and frame_index >= frames_to_process:
                break

            ok, frame = capture.read()
            if not ok:
                break

            human_frame_index = frame_index + 1
            should_deform = human_frame_index >= args.deform_start_frame and (
                args.deform_end_frame is None or human_frame_index <= args.deform_end_frame
            )

            if should_deform:
                map_x, map_y = build_warp_maps(
                    width=width,
                    height=height,
                    frame_index=frame_index,
                    deformers=deformers,
                    jitter_ratio=args.jitter_ratio,
                    rng=rng,
                )
                output_frame = deform_frame(frame, map_x, map_y)
            else:
                output_frame = frame

            writer.write(output_frame)

            written_frames += 1
            frame_index += 1

            if should_deform:
                deformers = [
                    update_deformer(
                        deformer=deformer,
                        width=width,
                        height=height,
                        radius_range_px=radius_range_px,
                        strength_px=args.strength_px,
                        speed_px=args.speed_px,
                        drift_px=args.drift_px,
                        rng=rng,
                    )
                    for deformer in deformers
                ]

            if written_frames == 1 or written_frames % args.progress_every == 0:
                print(f"  processed frames: {written_frames}")
    finally:
        capture.release()
        writer.release()

    if written_frames == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Aucune frame n'a ete ecrite. Verifie la video d'entree.")

    print(f"Done. Output video written to: {output_path}")
    return output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    process_video(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
