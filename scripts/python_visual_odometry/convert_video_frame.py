import os
from pathlib import Path

import cv2
import numpy as np


def save_frame(frame_path, frame):
    suffix = Path(frame_path).suffix or ".png"
    success, encoded = cv2.imencode(suffix, frame)
    if not success:
        return False

    try:
        encoded.tofile(frame_path)
    except OSError:
        return False
    return True


def load_frame(frame_path, flags=cv2.IMREAD_COLOR):
    try:
        buffer = np.fromfile(os.fspath(frame_path), dtype=np.uint8)
    except OSError:
        return None

    if buffer.size == 0:
        return None

    return cv2.imdecode(buffer, flags)


def video_to_frames_and_save(video_path, output_dir, gray=True, step=1):
    if step < 1:
        raise ValueError("Le parametre 'step' doit etre superieur ou egal a 1.")

    video_path = os.fspath(video_path)
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video : {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = 0
    saved_idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS : {fps}")
    print(f"Nombre total de trames : {frame_count}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % step == 0:
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_name = f"frame_{saved_idx:05d}.png"
            frame_path = os.path.join(output_dir, frame_name)

            if not save_frame(frame_path, frame):
                cap.release()
                raise IOError(f"Echec de l'ecriture de l'image : {frame_path}")

            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"{saved_idx} frames sauvegardees dans : {output_dir}")


def frames_to_video_in_same_folder(
    frames_dir,
    output_name="output.mp4",
    fps=30,
    pattern="*.png",
):
    if fps <= 0:
        raise ValueError("Le parametre 'fps' doit etre strictement positif.")

    frames_dir = Path(frames_dir)
    if not frames_dir.exists() or not frames_dir.is_dir():
        raise ValueError(f"Dossier de frames introuvable : {frames_dir}")

    frame_paths = sorted(frames_dir.glob(pattern))
    if not frame_paths:
        raise ValueError(
            f"Aucune frame correspondant a '{pattern}' n'a ete trouvee dans : {frames_dir}"
        )

    first_frame = load_frame(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Impossible de lire la premiere frame : {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    output_path = frames_dir / output_name

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError(f"Impossible de creer la video : {output_path}")

    written_count = 0

    try:
        for frame_path in frame_paths:
            frame = load_frame(frame_path)
            if frame is None:
                raise ValueError(f"Impossible de lire la frame : {frame_path}")

            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
            written_count += 1
    finally:
        writer.release()

    print(f"Video creee : {output_path}")
    print(f"Nombre de frames ajoutees : {written_count}")
    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    video_path = project_root / "dataset" / "ai" / "TUM_fake.mp4"
    output_dir = project_root / "dataset" / "ai" / "frames"
    video_to_frames_and_save(video_path, output_dir)
