from pathlib import Path
import argparse

from convert_video_frame import frames_to_video_in_same_folder


def main():
    parser = argparse.ArgumentParser(
        description="Convertit un dossier de frames PNG en video MP4."
    )
    parser.add_argument("frames_dir", help="Dossier contenant les frames .png")
    parser.add_argument(
        "--output-name",
        default="output.mp4",
        help="Nom du fichier video de sortie",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Nombre d'images par seconde",
    )
    args = parser.parse_args()

    output_path = frames_to_video_in_same_folder(
        frames_dir=Path(args.frames_dir),
        output_name=args.output_name,
        fps=args.fps,
    )
    print(output_path)


if __name__ == "__main__":
    main()
