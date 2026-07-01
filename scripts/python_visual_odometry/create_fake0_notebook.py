from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_NOTEBOOK = PROJECT_ROOT / "notbook" / "Detection_DF_Emanuel_fake.ipynb"


def set_cell_source(notebook: dict, cell_index: int, source: str) -> None:
    notebook["cells"][cell_index]["source"] = source.splitlines(keepends=True)


def clear_code_outputs(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def fill_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_notebook(source_notebook: Path, replacements: dict[str, str]) -> dict:
    notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
    clear_code_outputs(notebook)

    set_cell_source(
        notebook,
        9,
        fill_template(
            """import convert_video_frame
VIDEO_PATH = Path(r"__VIDEO_PATH__")
FRAMES_OUTPUT_DIR = Path(r"__FRAMES_OUTPUT_DIR__")
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "image"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if FRAMES_OUTPUT_DIR.exists():
    for existing_file in FRAMES_OUTPUT_DIR.glob("*.png"):
        existing_file.unlink()
else:
    FRAMES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise ValueError(f"Impossible d'ouvrir la video : {VIDEO_PATH}")

fps = float(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
max_frames = __MAX_FRAMES__
saved_idx = 0

print(f"Video source : {VIDEO_PATH}")
print(f"FPS : {fps}")
print(f"Nombre total de trames dans la video : {frame_count}")
print(f"Extraction des {max_frames} premieres frames vers : {FRAMES_OUTPUT_DIR}")

while saved_idx < max_frames:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_path = FRAMES_OUTPUT_DIR / f"frame_{saved_idx:05d}.png"
    if not convert_video_frame.save_frame(frame_path, gray_frame):
        cap.release()
        raise IOError(f"Echec de l'ecriture de l'image : {frame_path}")

    saved_idx += 1

cap.release()
print(f"{saved_idx} frames sauvegardees dans : {FRAMES_OUTPUT_DIR}")

if saved_idx < 3:
    raise ValueError("Il faut au moins 3 frames pour estimer les intrinseques.")
""",
            replacements,
        ),
    )

    set_cell_source(notebook, 10, """dossier_fake = FRAMES_OUTPUT_DIR""")

    set_cell_source(
        notebook,
        11,
        fill_template(
            """fake_frame_paths = sorted(
    str(f.resolve())
    for f in dossier_fake.rglob("*.png")
    if f.is_file()
)

print(f"{len(fake_frame_paths)} fichiers trouves dans {dossier_fake}")

if len(fake_frame_paths) != __MAX_FRAMES__:
    raise ValueError(f"Le dossier de travail doit contenir exactement __MAX_FRAMES__ frames, obtenu: {len(fake_frame_paths)}")
""",
            replacements,
        ),
    )

    notebook["cells"][19]["source"] = [
        fill_template("### **Intrinsics estimated from __VIDEO_FILENAME__**\n", replacements),
    ]
    notebook["cells"][20]["source"] = [
        fill_template(
            "Les intrinseques sont estimees automatiquement une seule fois a partir de 3 frames reparties sur les **__MAX_FRAMES__ premieres images** de `__VIDEO_FILENAME__`.\n",
            replacements,
        ),
        "Le resultat alimente ensuite **tout le notebook** et la boucle sequence complete.\n",
    ]

    set_cell_source(
        notebook,
        22,
        fill_template(
            """# ==================================================
# Intrinseques estimees UNE FOIS au debut
# ==================================================
from estimate_intrinsics import estimate_camera_intrinsics_from_frames

loaded_height, loaded_width = keyframe_image.shape[:2]
frames_for_intrinsics = fake_frame_paths[:__MAX_FRAMES__]

candidate_index_sets = []
if len(frames_for_intrinsics) >= 3:
    last_index = len(frames_for_intrinsics) - 1
    candidate_index_sets.extend(
        [
            [0, last_index // 2, last_index],
            [last_index // 6, last_index // 2, (5 * last_index) // 6],
            [0, last_index // 3, (2 * last_index) // 3],
            [0, 1, 2],
        ]
    )

candidate_index_sets.append(np.linspace(0, len(frames_for_intrinsics) - 1, 3, dtype=int).tolist())

estimated_K = None
used_intrinsic_paths = None
last_error = None

for raw_indices in candidate_index_sets:
    candidate_indices = [int(min(max(index, 0), len(frames_for_intrinsics) - 1)) for index in raw_indices]
    if len(set(candidate_indices)) < 3:
        continue

    sample_paths = [normalize_image_path(frames_for_intrinsics[index]) for index in candidate_indices]
    sample_frames = [load_gray_image(path) for path in sample_paths]

    try:
        estimated_K = estimate_camera_intrinsics_from_frames(sample_frames)
        used_intrinsic_paths = sample_paths
        break
    except Exception as exc:
        print(f"Echec estimation intrinseques avec indices {candidate_indices}: {exc}")
        last_error = exc

if estimated_K is None:
    raise RuntimeError(f"Impossible d'estimer les intrinseques pour __VIDEO_FILENAME__. Derniere erreur: {last_error}")

USER_WIDTH = loaded_width
USER_HEIGHT = loaded_height
USER_FX = float(estimated_K[0, 0])
USER_FY = float(estimated_K[1, 1])
USER_CX = float(estimated_K[0, 2])
USER_CY = float(estimated_K[1, 2])

intrinsics_user = {
    "width": USER_WIDTH,
    "height": USER_HEIGHT,
    "fx": USER_FX,
    "fy": USER_FY,
    "cx": USER_CX,
    "cy": USER_CY,
}

K_user = np.array(
    [
        [USER_FX, 0.0, USER_CX],
        [0.0, USER_FY, USER_CY],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

cam = camera.camera(
    USER_FX,
    USER_FY,
    USER_CX,
    USER_CY,
    USER_WIDTH,
    USER_HEIGHT,
)

intrinsics_used_lvl0 = {
    "width": cam.width[0],
    "height": cam.height[0],
    "fx": cam.fx[0],
    "fy": cam.fy[0],
    "cx": cam.cx[0],
    "cy": cam.cy[0],
}

K_used_lvl0 = np.array(
    [
        [cam.fx[0], 0.0, cam.cx[0]],
        [0.0, cam.fy[0], cam.cy[0]],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

print("Loaded keyframe size:")
print("  width  =", loaded_width)
print("  height =", loaded_height)
print()
print("Frames utilisees pour estimer les intrinseques:")
for path in used_intrinsic_paths:
    print(" ", path)
print()
print("Intrinseques estimees au debut du notebook:")
print(intrinsics_user)
print()
print("K_user =")
print(K_user)
print()
print("Internal working resolution used by the pipeline (level 0):")
print("  width  =", intrinsics_used_lvl0["width"])
print("  height =", intrinsics_used_lvl0["height"])
print()
print("K_used_lvl0 =")
print(K_used_lvl0)
""",
            replacements,
        ),
    )

    set_cell_source(
        notebook,
        37,
        """optimization_summary = {
    "lvl4_initial": float(initial_error_lvl4),
    "lvl4_final": float(final_error_lvl4),
    "lvl3_initial": float(initial_error_lvl3),
    "lvl3_final": float(final_error_lvl3),
    "lvl2_initial": float(initial_error_lvl2),
    "lvl2_final": float(final_error_lvl2),
}

error_improvement = {
    "lvl4": float(initial_error_lvl4 - final_error_lvl4),
    "lvl3": float(initial_error_lvl3 - final_error_lvl3),
    "lvl2": float(initial_error_lvl2 - final_error_lvl2),
}

print("Optimization summary:")
for lvl in [4, 3, 2]:
    initial_value = optimization_summary[f"lvl{lvl}_initial"]
    final_value = optimization_summary[f"lvl{lvl}_final"]
    improvement = error_improvement[f"lvl{lvl}"]
    print(f"  level {lvl}: initial={initial_value:.6f}, final={final_value:.6f}, improvement={improvement:.6f}")

pose_matrix_now = current_frame.pose.as_matrix()
pose_is_finite = bool(np.isfinite(pose_matrix_now).all())
errors_improved = {lvl: (error_improvement[f"lvl{lvl}"] > 0.0) for lvl in [4, 3, 2]}

print()
print("Pose matrix finite ->", pose_is_finite)
print("Errors improved by level ->", errors_improved)

assert pose_is_finite, "Pose matrix contains non-finite values."

if all(errors_improved.values()):
    print()
    print("Sanity check passed: intrinsics are coherent and pose optimization improves the photometric error at all checked levels.")
else:
    print()
    print("Warning: at least one pyramid level did not improve, but the pose matrix remains finite. Execution continues for sequence-level analysis.")
""",
    )

    set_cell_source(
        notebook,
        55,
        fill_template(
            """# Boucle simple sur toute la sequence: depth estimation puis pose estimation par frame.
# Cette version reconstruit une camera coherente pour la sequence traitee
# et reinitialise le solveur pour eviter de reutiliser un etat precedent.
frame_paths = fake_frame_paths[:__MAX_FRAMES__]

if len(frame_paths) < 2:
    raise ValueError("Il faut au moins 2 frames pour lancer le pipeline.")

sequence_name = "__SEQUENCE_NAME__"
sequence_keyframe = None
sequence_results = []

first_image_path = normalize_image_path(frame_paths[0])
first_image = load_gray_image(first_image_path)
sequence_height, sequence_width = first_image.shape[:2]

sequence_cam = camera.camera(
    USER_FX,
    USER_FY,
    USER_CX,
    USER_CY,
    sequence_width,
    sequence_height,
)
sequence_pose_solver = pose_estimator_gauss_newton.pose_estimator_gauss_newton(sequence_cam, show_debug=False)

print(f"Sequence {sequence_name}:")
print(f"  source size = {sequence_width} x {sequence_height}")
print(f"  working size level-0 = {sequence_cam.width[0]} x {sequence_cam.height[0]}")
print(f"  fx level-0 = {sequence_cam.fx[0]}")
print(f"  fy level-0 = {sequence_cam.fy[0]}")
print()

for step, image_path in enumerate(frame_paths):
    image_path = normalize_image_path(image_path)
    image = load_gray_image(image_path)

    if image.shape[:2] != (sequence_height, sequence_width):
        raise ValueError(
            f"Toutes les images de la sequence {sequence_name} doivent avoir la meme taille. "
            f"Attendu: {(sequence_height, sequence_width)}, obtenu: {image.shape[:2]} pour {image_path.name}"
        )

    depth_map = load_depth_map(image_path)
    inv_depth, inv_depth_var, depth_valid_mask = depth_to_invdepth(depth_map)

    current_frame = frameData.frameData()
    current_frame.setImage(image)
    current_frame.setInvDepth(inv_depth, inv_depth_var)

    if step == 0:
        sequence_keyframe = current_frame
        print(f"[{sequence_name}] frame {step} -> keyframe de reference: {image_path.name}")
        sequence_results.append(
            {
                "index": step,
                "image_path": str(image_path),
                "depth_valid_ratio": float(depth_valid_mask.mean()),
                "pose_matrix": current_frame.pose.as_matrix(),
                "photometric_error": None,
                "photometric_error_mean": None,
                "valid_pixel_count": None,
                "histogram_counts": None,
            }
        )
        continue

    sequence_pose_solver.optPose(current_frame, sequence_keyframe)
    evaluation = compute_photometric_error(current_frame, sequence_keyframe, sequence_cam, lvl=0)

    abs_error_map = np.sqrt(evaluation["squared_error_map"])
    valid_mask = evaluation["valid_mask"]
    valid_abs_errors = abs_error_map[valid_mask]
    valid_pixel_count = int(valid_abs_errors.size)
    photometric_error = float(evaluation["photometric_error"])
    photometric_error_mean = float(valid_abs_errors.mean()) if valid_pixel_count > 0 else None
    mean_error_display = f"{photometric_error_mean:.4f}" if photometric_error_mean is not None else "None"

    print(
        f"[{sequence_name}] frame {step} -> {image_path.name} | "
        f"sum error = {photometric_error:.2f} | "
        f"mean abs error = {mean_error_display} | "
        f"valid pixels = {valid_pixel_count}"
    )

    num_gt_200 = int(np.sum(valid_abs_errors > 200))
    num_150_200 = int(np.sum((valid_abs_errors >= 150) & (valid_abs_errors <= 200)))
    num_100_150 = int(np.sum((valid_abs_errors >= 100) & (valid_abs_errors < 150)))
    num_50_100 = int(np.sum((valid_abs_errors >= 50) & (valid_abs_errors < 100)))
    num_30_50 = int(np.sum((valid_abs_errors >= 30) & (valid_abs_errors < 50)))
    num_10_30 = int(np.sum((valid_abs_errors >= 10) & (valid_abs_errors < 30)))
    num_0_10 = int(np.sum((valid_abs_errors >= 0) & (valid_abs_errors < 10)))

    range_labels = [
        "> 200",
        "150 - 200",
        "100 - 150",
        "50 - 100",
        "30 - 50",
        "10 - 30",
        "0 - 10",
    ]

    range_counts = [
        num_gt_200,
        num_150_200,
        num_100_150,
        num_50_100,
        num_30_50,
        num_10_30,
        num_0_10,
    ]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range_labels, range_counts, color="#4C78A8", edgecolor="black")
    plt.title(f"Differences absolues - {sequence_name} - {image_path.name}")
    plt.xlabel("Tranches de difference absolue")
    plt.ylabel("Nombre de pixels")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")

    for bar, count in zip(bars, range_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    sequence_results.append(
        {
            "index": step,
            "image_path": str(image_path),
            "depth_valid_ratio": float(depth_valid_mask.mean()),
            "pose_matrix": current_frame.pose.as_matrix(),
            "photometric_error": photometric_error,
            "photometric_error_mean": photometric_error_mean,
            "valid_pixel_count": valid_pixel_count,
            "histogram_counts": range_counts,
        }
    )
""",
            replacements,
        ),
    )

    set_cell_source(
        notebook,
        56,
        fill_template(
            """valid_histogram_results = [
    item for item in sequence_results
    if item.get("histogram_counts") is not None
]

if not valid_histogram_results:
    raise ValueError("Aucun histogramme valide disponible pour calculer la moyenne finale.")

histogram_matrix = np.array([item["histogram_counts"] for item in valid_histogram_results], dtype=np.float64)
mean_histogram_counts = histogram_matrix.mean(axis=0)

final_range_labels = [
    "> 200",
    "150 - 200",
    "100 - 150",
    "50 - 100",
    "30 - 50",
    "10 - 30",
    "0 - 10",
]

plt.figure(figsize=(10, 5))
bars = plt.bar(final_range_labels, mean_histogram_counts, color="#59A14F", edgecolor="black")
plt.title("Histogramme final moyen sur les __MAX_FRAMES__ premieres frames traitees")
plt.xlabel("Tranches de difference absolue")
plt.ylabel("Nombre moyen de pixels")
plt.grid(axis="y", alpha=0.25)
plt.xticks(rotation=30, ha="right")

for bar, value in zip(bars, mean_histogram_counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
final_histogram_path = IMAGE_OUTPUT_DIR / "__OUTPUT_PREFIX___mean_histogram.png"
plt.savefig(final_histogram_path, dpi=150, bbox_inches="tight")
plt.show()

print("Histogramme final moyen:")
for label, value in zip(final_range_labels, mean_histogram_counts):
    print(f"  {label:>8} -> {value:.2f}")
print(f"Image sauvegardee : {final_histogram_path}")
""",
            replacements,
        ),
    )

    set_cell_source(
        notebook,
        57,
        fill_template(
            """valid_results = [
    item for item in sequence_results
    if item["photometric_error"] is not None
]

if not valid_results:
    raise ValueError("Aucun resultat valide disponible pour calculer la moyenne des erreurs photometriques.")

iteration_indices = [item["index"] for item in valid_results]
photometric_errors = [item["photometric_error"] for item in valid_results]
mean_photometric_error = float(np.mean(photometric_errors))

print(mean_photometric_error)

plt.figure(figsize=(10, 5))
plt.plot(iteration_indices, photometric_errors, marker="o", color="#4C78A8", label="Photometric error")
plt.axhline(
    mean_photometric_error,
    color="#E45756",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_photometric_error:.6f}"
)
plt.title("__PLOT_LABEL__ : moyenne de la photometric error sur les iterations valides")
plt.xlabel("Iteration")
plt.ylabel("Photometric error")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
final_error_curve_path = IMAGE_OUTPUT_DIR / "__OUTPUT_PREFIX___photometric_error.png"
plt.savefig(final_error_curve_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Image sauvegardee : {final_error_curve_path}")
""",
            replacements,
        ),
    )

    return notebook


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cree un notebook d'analyse de sequence video avec estimation automatique des intrinseques."
    )
    parser.add_argument(
        "--source-notebook",
        type=Path,
        default=DEFAULT_SOURCE_NOTEBOOK,
        help="Notebook source servant de base.",
    )
    parser.add_argument(
        "--target-notebook",
        type=Path,
        default=PROJECT_ROOT / "notbook" / "Detection_DF_fake0_first30_estimated_intrinsics.ipynb",
        help="Notebook cible a ecrire.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Chemin absolu ou relatif vers la video source. Si absent, construit dataset/<video-subdir>/<video-filename>.",
    )
    parser.add_argument(
        "--video-subdir",
        default="ai",
        help="Sous-dossier dans dataset contenant la video.",
    )
    parser.add_argument(
        "--frames-output-dir",
        type=Path,
        default=None,
        help="Chemin absolu ou relatif du dossier de frames a creer. Si absent, utilise dataset/<video-subdir>/<frames-dirname>.",
    )
    parser.add_argument(
        "--video-filename",
        default="fake0.mp4",
        help="Nom du fichier video source.",
    )
    parser.add_argument(
        "--frames-dirname",
        default="fake0_frames_first30",
        help="Nom du dossier de frames de travail a creer dans dataset/<video-subdir>.",
    )
    parser.add_argument(
        "--sequence-name",
        default="fake0_first30",
        help="Nom logique de la sequence pour les sorties console et le notebook.",
    )
    parser.add_argument(
        "--output-prefix",
        default="fake0_first30",
        help="Prefixe des images finales sauvegardees dans le dossier image.",
    )
    parser.add_argument(
        "--plot-label",
        default="FAKE0",
        help="Libelle court utilise dans le titre du graphe final d'erreur.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Nombre maximal de frames a extraire et traiter.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    video_path = args.video_path
    if video_path is None:
        video_path = PROJECT_ROOT / "dataset" / args.video_subdir / args.video_filename
    elif not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()
    else:
        video_path = video_path.resolve()

    frames_output_dir = args.frames_output_dir
    if frames_output_dir is None:
        frames_output_dir = PROJECT_ROOT / "dataset" / args.video_subdir / args.frames_dirname
    elif not frames_output_dir.is_absolute():
        frames_output_dir = (PROJECT_ROOT / frames_output_dir).resolve()
    else:
        frames_output_dir = frames_output_dir.resolve()

    replacements = {
        "__VIDEO_PATH__": str(video_path),
        "__VIDEO_FILENAME__": video_path.name,
        "__FRAMES_OUTPUT_DIR__": str(frames_output_dir),
        "__SEQUENCE_NAME__": args.sequence_name,
        "__OUTPUT_PREFIX__": args.output_prefix,
        "__PLOT_LABEL__": args.plot_label,
        "__MAX_FRAMES__": str(args.max_frames),
    }

    notebook = build_notebook(args.source_notebook, replacements)
    args.target_notebook.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Notebook cree : {args.target_notebook}")


if __name__ == "__main__":
    main()
