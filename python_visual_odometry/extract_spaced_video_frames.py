from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a small set of spaced frames from a video so the images "
            "have enough motion for visual odometry."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input .mp4 or .mov video.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the extracted frames will be saved.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of frames to export.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame eligible for extraction.",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=10,
        help=(
            "Preferred minimum gap in frames between two extracted images. "
            "If the video is too short, the script falls back to uniform spacing."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="scene",
        help="Filename prefix for extracted frames.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional output width. Height is adjusted to preserve aspect ratio.",
    )
    return parser.parse_args()


def compute_frame_indices(
    total_frames: int,
    count: int,
    start_frame: int,
    min_gap: int,
) -> list[int]:
    if total_frames <= 0:
        raise ValueError("The video does not contain any readable frames.")
    if count <= 0:
        raise ValueError("--count must be strictly positive.")
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0.")
    if min_gap <= 0:
        raise ValueError("--min-gap must be strictly positive.")
    if start_frame >= total_frames:
        raise ValueError(
            f"--start-frame ({start_frame}) is beyond the video length ({total_frames} frames)."
        )

    remaining = total_frames - start_frame
    if count == 1:
        return [start_frame]

    preferred_span = 1 + (count - 1) * min_gap
    if remaining >= preferred_span:
        return [start_frame + i * min_gap for i in range(count)]

    last_frame = total_frames - 1
    indices = np.linspace(start_frame, last_frame, count, dtype=int).tolist()

    deduped: list[int] = []
    for index in indices:
        if not deduped or index != deduped[-1]:
            deduped.append(index)

    if len(deduped) < count:
        raise ValueError(
            f"Unable to extract {count} distinct frames from a video with only {remaining} "
            f"available frames after frame {start_frame}."
        )

    return deduped[:count]


def maybe_resize(frame: np.ndarray, resize_width: int | None) -> np.ndarray:
    if resize_width is None:
        return frame

    height, width = frame.shape[:2]
    if width == resize_width:
        return frame

    scale = resize_width / float(width)
    resize_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else video_path.parent / f"{video_path.stem}_frames_spaced"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    indices = compute_frame_indices(
        total_frames=total_frames,
        count=args.count,
        start_frame=args.start_frame,
        min_gap=args.min_gap,
    )

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.3f}")
    print(f"Selected frame indices: {indices}")
    print(f"Output directory: {output_dir}")

    for export_id, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

        frame = maybe_resize(frame, args.resize_width)
        output_path = output_dir / f"{args.prefix}_{export_id:03d}.png"
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write image: {output_path}")

        timestamp = frame_index / fps if fps > 0 else 0.0
        print(
            f"Saved {output_path.name} from frame {frame_index} "
            f"(t={timestamp:.3f}s)"
        )

    cap.release()
    print("Extraction completed successfully.")


if __name__ == "__main__":
    main()
