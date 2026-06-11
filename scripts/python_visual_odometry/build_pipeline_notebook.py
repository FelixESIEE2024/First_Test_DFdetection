from pathlib import Path
import textwrap

import nbformat as nbf


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notbook"
NOTEBOOK_PATH = NOTEBOOK_DIR / "visual_odometry_pipeline_walkthrough.ipynb"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

nb.cells = [
    md(
        """
        # Pipeline explicite de `python_visual_odometry`

        Ce notebook reconstruit pas a pas le pipeline actuel du projet.

        Objectif :
        - charger un **keyframe**
        - charger sa **depth** calculee a l'avance
        - convertir cette depth en **inverse depth**
        - charger une image courante
        - estimer la **pose relative** entre les deux
        - afficher a la fin la **matrice extrinseque 4x4**

        L'idee est que tu puisses lire le code comme un cours pratique, pas seulement comme un script qui tourne.
        """
    ),
    md(
        """
        ## 1. Les fichiers utilises

        Le pipeline s'appuie surtout sur :

        - `camera.py` : construit les intrinseques pour chaque niveau de pyramide
        - `frameData.py` : stocke image, derivees, inverse depth et pose
        - `pose_estimator_gauss_newton.py` : optimise la pose
        - `common.py` : interpolation bilineaire
        - `dataset/desktop_dataset/images` : les images
        - `dataset/desktop_dataset/depth` : les depth maps `.npy`

        Dans ce notebook, tout ce que tu voudras changer sera regroupe dans une seule cellule
        de parametres juste en dessous :
        - dossier dataset
        - index du keyframe
        - index de l'image courante
        - intrinseques camera `fx`, `fy`, `cx`, `cy`
        - activation ou non des fenetres debug du solveur
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        import cv2
        import matplotlib.pyplot as plt
        import numpy as np

        ROOT = Path.cwd()
        if not (ROOT / "scripts" / "python_visual_odometry").exists() and (ROOT.parent / "scripts" / "python_visual_odometry").exists():
            ROOT = ROOT.parent

        SCRIPT_DIR = ROOT / "scripts" / "python_visual_odometry"
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))

        import camera
        import common
        import frameData
        import pose_estimator_gauss_newton

        plt.rcParams["figure.figsize"] = (8, 5)
        np.set_printoptions(precision=6, suppress=True)

        # =========================
        # Parametres a modifier
        # =========================
        DATASET_DIR = ROOT / "dataset" / "desktop_dataset"
        IMAGE_DIR = DATASET_DIR / "images"
        DEPTH_DIR = DATASET_DIR / "depth"

        KEYFRAME_INDEX = 0
        TARGET_INDEX = 1
        SHOW_SOLVER_DEBUG = False

        CAMERA_FX = 525.0
        CAMERA_FY = 525.0
        CAMERA_CX = 319.5
        CAMERA_CY = 239.5

        print("ROOT =", ROOT)
        print("DATASET_DIR =", DATASET_DIR)
        print("IMAGE_DIR =", IMAGE_DIR)
        print("DEPTH_DIR =", DEPTH_DIR)
        print("KEYFRAME_INDEX =", KEYFRAME_INDEX)
        print("TARGET_INDEX =", TARGET_INDEX)
        print("SHOW_SOLVER_DEBUG =", SHOW_SOLVER_DEBUG)
        print("CAMERA_FX =", CAMERA_FX)
        print("CAMERA_FY =", CAMERA_FY)
        print("CAMERA_CX =", CAMERA_CX)
        print("CAMERA_CY =", CAMERA_CY)
        """
    ),
    md(
        """
        ## 2. Fonctions utilitaires du notebook

        On re-ecrit ici quelques petites fonctions pour rendre le pipeline explicite :

        - charger une image en niveaux de gris
        - charger une depth map
        - convertir depth -> inverse depth
        - afficher rapidement une matrice 4x4

        La conversion importante est :

        $$
        \\rho = \\frac{1}{d}
        $$

        ou `d` est la depth et `rho` l'inverse depth attendue par le solveur.
        """
    ),
    code(
        """
        def load_gray_image(index: int) -> np.ndarray:
            path = IMAGE_DIR / f"scene_{index:03d}.png"
            buffer = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Image introuvable: {path}")
            return image


        def load_depth_map(index: int) -> np.ndarray:
            path = DEPTH_DIR / f"scene_{index:03d}_depth.npy"
            if not path.exists():
                raise FileNotFoundError(f"Depth introuvable: {path}")
            return np.load(path).astype(np.float32)


        def depth_to_invdepth(depth: np.ndarray):
            inv_depth = np.zeros_like(depth, dtype=np.float32)
            valid_mask = depth > 1e-6
            inv_depth[valid_mask] = 1.0 / depth[valid_mask]
            inv_depth_var = np.ones_like(depth, dtype=np.float32)
            inv_depth_var[~valid_mask] = 1e6
            return inv_depth, inv_depth_var, valid_mask


        def show_pose_matrix(title: str, pose):
            matrix = pose.as_matrix()
            print(title)
            print(matrix)
            return matrix
        """
    ),
    md(
        """
        ## 3. Chargement du keyframe et de sa depth

        Ici on charge :

        - l'image de reference `scene_000`
        - la depth map `scene_000_depth.npy`

        C'est cette image qui fournit la geometrie 3D de base.
        """
    ),
    code(
        """
        keyframe_image = load_gray_image(KEYFRAME_INDEX)
        keyframe_depth = load_depth_map(KEYFRAME_INDEX)
        keyframe_inv_depth, keyframe_inv_depth_var, valid_mask = depth_to_invdepth(keyframe_depth)

        print("keyframe_image shape =", keyframe_image.shape, "dtype =", keyframe_image.dtype)
        print("keyframe_depth shape =", keyframe_depth.shape, "dtype =", keyframe_depth.dtype)
        print("depth min/max =", float(keyframe_depth.min()), float(keyframe_depth.max()))
        print("valid depth ratio =", float(valid_mask.mean()))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(keyframe_image, cmap="gray")
        axes[0].set_title("Keyframe image")
        axes[1].imshow(keyframe_depth, cmap="viridis")
        axes[1].set_title("Depth map")
        axes[2].imshow(keyframe_inv_depth, cmap="magma")
        axes[2].set_title("Inverse depth")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 4. Construction du keyframe `frameData`

        `frameData` cree une pyramide multi-echelle :

        - image niveau 0, 1, 2, 3, 4
        - derivees spatiales de chaque niveau
        - inverse depth et variance pour chaque niveau

        En pratique :
        - `setImage(...)` construit la pyramide de l'image
        - `setInvDepth(...)` redimensionne aussi l'inverse depth sur chaque niveau

        Ici, les intrinseques viennent directement de la cellule de parametres du notebook.
        Il n'y a pas d'autre valeur cachee ailleurs dans cette version.
        """
    ),
    code(
        """
        height, width = keyframe_image.shape

        cam = camera.camera(
            CAMERA_FX,
            CAMERA_FY,
            CAMERA_CX,
            CAMERA_CY,
            width,
            height,
        )

        print("Camera intrinsics used by pose_solver:")
        print("fx =", CAMERA_FX)
        print("fy =", CAMERA_FY)
        print("cx =", CAMERA_CX)
        print("cy =", CAMERA_CY)

        keyframe = frameData.frameData()
        keyframe.setImage(keyframe_image)
        keyframe.setInvDepth(keyframe_inv_depth, keyframe_inv_depth_var)

        print("Pyramid levels =", len(keyframe.image))
        for lvl in range(len(keyframe.image)):
            print(
                f"level {lvl}: image={keyframe.image[lvl].shape}, "
                f"invDepth={keyframe.invDepth[lvl].shape}, "
                f"grad={keyframe.imageDerivative[lvl].shape}"
            )
        """
    ),
    md(
        """
        ## 5. Chargement de l'image courante

        Maintenant on prend une deuxieme image.

        C'est elle dont on veut estimer la pose relative par rapport au keyframe.
        Au debut, sa pose est l'identite, puis le solveur va la mettre a jour.
        """
    ),
    code(
        """
        current_image = load_gray_image(TARGET_INDEX)

        current_frame = frameData.frameData()
        current_frame.setImage(current_image)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(keyframe_image, cmap="gray")
        axes[0].set_title(f"Keyframe scene_{KEYFRAME_INDEX:03d}")
        axes[1].imshow(current_image, cmap="gray")
        axes[1].set_title(f"Current frame scene_{TARGET_INDEX:03d}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 6. Que veut dire `estimer la pose` ici ?

        Pour chaque pixel du keyframe :

        1. on utilise l'inverse depth pour remonter au point 3D
        2. on applique une pose candidate
        3. on reprojette ce point dans l'image courante
        4. on compare l'intensite predite et l'intensite observee

        Le solveur essaye donc de minimiser une erreur photometrique.

        Si la pose est correcte, les points reprojectes tombent aux bons endroits dans l'image courante.
        """
    ),
    code(
        """
        pose_solver = pose_estimator_gauss_newton.pose_estimator_gauss_newton(
            cam,
            show_debug=SHOW_SOLVER_DEBUG,
        )

        initial_error_lvl4, _ = pose_solver.computeError(current_frame, keyframe, lvl=4)
        initial_error_lvl3, _ = pose_solver.computeError(current_frame, keyframe, lvl=3)
        initial_error_lvl2, _ = pose_solver.computeError(current_frame, keyframe, lvl=2)

        print("Initial photometric errors before optimization:")
        print("  level 4 =", initial_error_lvl4)
        print("  level 3 =", initial_error_lvl3)
        print("  level 2 =", initial_error_lvl2)
        """
    ),
    md(
        """
        ## 7. Optimisation de pose

        C'est ici que la pose est reellement calculee.

        `optPose(...)` fait plusieurs mises a jour :

        - il calcule une erreur
        - il linearise localement le probleme
        - il calcule un increment de pose
        - il accepte ou rejette cette mise a jour selon l'amelioration de l'erreur

        A la fin, `current_frame.pose` contient la matrice extrinseque estimee du frame courant par rapport au keyframe.
        """
    ),
    code(
        """
        pose_solver.optPose(current_frame, keyframe)

        final_error_lvl4, _ = pose_solver.computeError(current_frame, keyframe, lvl=4)
        final_error_lvl3, _ = pose_solver.computeError(current_frame, keyframe, lvl=3)
        final_error_lvl2, _ = pose_solver.computeError(current_frame, keyframe, lvl=2)

        print("Final photometric errors after optimization:")
        print("  level 4 =", final_error_lvl4)
        print("  level 3 =", final_error_lvl3)
        print("  level 2 =", final_error_lvl2)
        """
    ),
    md(
        """
        ## 8. La matrice extrinseque finale

        C'est la sortie la plus importante du notebook.

        Cette matrice 4x4 represente la pose relative estimee de l'image courante par rapport au keyframe.

        Sous forme bloc :

        $$
        T =
        \\begin{bmatrix}
        R & t \\\\
        0 & 1
        \\end{bmatrix}
        $$

        avec :
        - `R` = rotation `3x3`
        - `t` = translation `3x1`
        """
    ),
    code(
        """
        extrinsic_matrix = show_pose_matrix(
            f"Extrinsic matrix for scene_{TARGET_INDEX:03d} relative to scene_{KEYFRAME_INDEX:03d}:",
            current_frame.pose,
        )

        rotation_matrix = extrinsic_matrix[:3, :3]
        translation_vector = extrinsic_matrix[:3, 3]

        print("\\nRotation matrix:")
        print(rotation_matrix)

        print("\\nTranslation vector:")
        print(translation_vector)
        """
    ),
    md(
        """
        ## 9. Calcul explicite de la difference d'intensite pixel par pixel

        Maintenant que la pose est estimee, on peut faire la comparaison la plus concrete possible :

        - pour chaque pixel du keyframe
        - on le reprojette dans l'image courante avec la pose finale
        - on compare l'intensite du keyframe et l'intensite observee dans l'image 2

        Cela nous donne une **carte de differences** directement interpretable.
        """
    ),
    code(
        """
        def compute_photometric_maps(frame, keyframe, cam, lvl=0):
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

            relativePose = frame.pose.dot(keyframe.pose.inv())

            predicted_map = np.full((height, width), np.nan, dtype=np.float32)
            true_map = np.full((height, width), np.nan, dtype=np.float32)
            signed_diff_map = np.full((height, width), np.nan, dtype=np.float32)
            abs_diff_map = np.full((height, width), np.nan, dtype=np.float32)
            squared_diff_map = np.full((height, width), np.nan, dtype=np.float32)
            valid_mask = np.zeros((height, width), dtype=bool)

            for y in range(height):
                for x in range(width):
                    invDepth = keyframe.invDepth[lvl][y, x]
                    if invDepth <= 0.0:
                        continue

                    pointKeyframe = np.array([fxinv * x + cxinv, fyinv * y + cyinv, 1.0]) / invDepth
                    pointFrame = relativePose.dot(pointKeyframe)
                    if pointFrame[2] <= 0.0:
                        continue

                    pixelFrame = np.array([
                        fx * pointFrame[0] / pointFrame[2] + cx,
                        fy * pointFrame[1] / pointFrame[2] + cy,
                    ])
                    if pixelFrame[0] < 1.0 or pixelFrame[0] >= width - 1 or pixelFrame[1] < 1.0 or pixelFrame[1] >= height - 1:
                        continue

                    key_intensity = float(keyframe.image[lvl][y, x])
                    observed_intensity = float(common.getSubPixelValue(frame.image[lvl], pixelFrame))

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


        photometric_maps = compute_photometric_maps(current_frame, keyframe, cam, lvl=0)
        valid_pixels = photometric_maps["valid_mask"].sum()
        print("Number of valid reprojected pixels =", int(valid_pixels))
        """
    ),
    md(
        """
        ## 10. Visualiser la difference d'intensite et son carre

        Voici les cartes utiles :

        - **difference signee** : positive ou negative selon que le keyframe est plus clair ou plus sombre
        - **difference absolue** : plus c'est rouge, plus l'erreur d'intensite est forte
        - **difference au carre** : penalise encore plus les grosses erreurs
        """
    ),
    code(
        """
        signed_diff_map = photometric_maps["signed_diff_map"]
        abs_diff_map = photometric_maps["abs_diff_map"]
        squared_diff_map = photometric_maps["squared_diff_map"]
        valid_mask = photometric_maps["valid_mask"]

        valid_signed = signed_diff_map[valid_mask]
        vmax_signed = float(np.nanpercentile(np.abs(valid_signed), 99))
        vmax_abs = float(np.nanpercentile(abs_diff_map[valid_mask], 99))
        vmax_sq = float(np.nanpercentile(squared_diff_map[valid_mask], 99))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(signed_diff_map, cmap="bwr", vmin=-vmax_signed, vmax=vmax_signed)
        axes[0].set_title("Signed intensity difference")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(abs_diff_map, cmap="Reds", vmin=0, vmax=vmax_abs)
        axes[1].set_title("Absolute intensity difference")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(squared_diff_map, cmap="Reds", vmin=0, vmax=vmax_sq)
        axes[2].set_title("Squared intensity difference")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        print("Difference metrics on valid pixels:")
        print("  mean signed difference   =", float(np.nanmean(signed_diff_map)))
        print("  mean absolute difference =", float(np.nanmean(abs_diff_map)))
        print("  mean squared difference  =", float(np.nanmean(squared_diff_map)))
        print("  max absolute difference  =", float(np.nanmax(abs_diff_map)))
        print("  max squared difference   =", float(np.nanmax(squared_diff_map)))
        """
    ),
    md(
        """
        ## 11. Que faire de l'erreur apres optimisation ?

        Une question importante est la suivante :

        > si j'utilise deja l'image 2 pour optimiser la pose, ai-je encore le droit
        > d'utiliser la meme erreur de reprojection comme signal de detection ?

        La bonne reponse est plus subtile :

        - oui, l'optimisation reduit les residus autant que possible
        - mais elle ne peut reduire que les erreurs **explicables par un changement de pose**

        Donc si une paire d'images contient des incoherences geometriques que meme la meilleure
        pose ne peut pas expliquer, ces incoherences resteront visibles dans les residus finaux.

        Ce que l'on veut analyser n'est donc pas l'erreur "avant optimisation", mais bien :

        > **ce qui reste apres avoir donne sa meilleure chance a l'hypothese geometrique**

        C'est ce residu final, ou plus precisement sa magnitude et sa structure spatiale,
        qui peut devenir un signal utile pour la detection.
        """
    ),
    code(
        """
        residual_error_lvl2, residual_error_map_lvl2 = pose_solver.computeError(current_frame, keyframe, lvl=2)

        print("Residual error after pose optimization (lvl 2) =", residual_error_lvl2)

        plt.figure(figsize=(6, 5))
        plt.imshow(residual_error_map_lvl2, cmap="hot")
        plt.title("Residual reprojection error map after optimization (lvl 2)")
        plt.axis("off")
        plt.colorbar()
        plt.show()
        """
    ),
    md(
        """
        ## 12. Pourquoi ce residu reste informatif

        Ce residu final est deja "biaise vers le bas" parce que la pose a ete optimisee.
        Cette intuition est juste.

        Mais ce biais n'efface pas tout :

        - si la scene est coherente avec un vrai mouvement de camera, la meilleure pose peut expliquer une grande partie des differences
        - si certaines differences ne peuvent pas etre expliquees par une simple pose 3D, elles resteront visibles

        Ce que l'on veut donc regarder, ce n'est pas seulement "combien l'erreur est grande",
        mais aussi **comment elle est organisee spatialement**.
        """
    ),
    code(
        """
        nonzero_residuals = residual_error_map_lvl2[residual_error_map_lvl2 > 0]

        print("Number of valid residual pixels =", nonzero_residuals.size)
        print("Mean residual =", float(nonzero_residuals.mean()))
        print("Median residual =", float(np.median(nonzero_residuals)))
        print("90th percentile residual =", float(np.quantile(nonzero_residuals, 0.90)))
        print("99th percentile residual =", float(np.quantile(nonzero_residuals, 0.99)))
        """
    ),
    md(
        """
        ## 13. Mesurer la structure spatiale du residu

        Deux videos peuvent avoir une erreur moyenne proche, mais une structure tres differente :

        - une video reelle peut avoir des petites erreurs diffuses
        - une video modifiee peut laisser des paquets d'erreurs localises, plus structures

        On calcule donc aussi quelques indicateurs spatiaux simples sur la carte de residus.
        """
    ),
    code(
        """
        residual_binary = residual_error_map_lvl2 > np.quantile(nonzero_residuals, 0.90)
        residual_binary_u8 = (residual_binary.astype(np.uint8) * 255)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(residual_binary_u8, connectivity=8)
        component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])

        largest_component = int(component_areas.max()) if component_areas.size > 0 else 0
        num_components = int(component_areas.size)

        print("Connected high-error components =", num_components)
        print("Largest high-error component area =", largest_component)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(residual_error_map_lvl2, cmap="hot")
        axes[0].set_title("Residual error map")
        axes[1].imshow(residual_binary, cmap="gray")
        axes[1].set_title("Top 10% residual mask")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 14. Construire un score de suspicion simple

        Pour finir le projet au niveau prototype, on peut definir un score tres simple :

        - plus le residu photometrique final est grand,
        - plus les zones de forte erreur sont structurees,
        - plus la paire de frames parait incoherente avec une simple hypothese de pose camera

        Attention :
        - ce score ne permet pas a lui seul de dire "deepfake" ou "pas deepfake"
        - il faut calibrer un seuil sur un jeu de videos reelles et de videos falsifiees

        Mais cette cellule te donne deja la bonne sortie experimentale :
        un **score de residu geometrique apres optimisation**.
        """
    ),
    code(
        """
        suspicion_score = float(
            residual_error_lvl2
            + 0.001 * largest_component
            + 0.01 * num_components
        )

        print("Suspicion score (higher means less geometric consistency) =", suspicion_score)
        print("Base residual term =", float(residual_error_lvl2))
        print("Largest component contribution =", 0.001 * largest_component)
        print("Component count contribution =", 0.01 * num_components)

        print("\\nInterpretation:")
        print("- low score  -> the best pose explains the image pair fairly well")
        print("- high score -> residual inconsistency remains even after optimization")
        """
    ),
    md(
        """
        ## 15. Option bonus : calculer plusieurs poses d'un coup

        La cellule suivante permet de lancer le meme pipeline sur plusieurs frames successives.

        Elle n'est pas necessaire pour comprendre le principe, mais elle aide a voir que le keyframe et sa depth peuvent servir de reference pour plusieurs images.
        """
    ),
    code(
        """
        results = []
        sequence_solver = pose_estimator_gauss_newton.pose_estimator_gauss_newton(
            cam,
            show_debug=SHOW_SOLVER_DEBUG,
        )

        for idx in range(1, 4):
            frame = frameData.frameData()
            frame.setImage(load_gray_image(idx))
            sequence_solver.optPose(frame, keyframe)
            results.append((idx, frame.pose.as_matrix()))

        print("Computed poses relative to the keyframe:")
        for idx, matrix in results:
            print(f"scene_{idx:03d}")
            print(matrix)
            print()
        """
    ),
    md(
        """
        ## 16. Ce qu'il faut retenir

        Le pipeline complet est :

        1. image du keyframe
        2. depth du keyframe
        3. conversion en inverse depth
        4. reconstruction implicite de points 3D
        5. reprojection dans l'image courante
        6. optimisation de la meilleure pose possible
        7. calcul du residu qui reste apres optimisation
        8. analyse de la magnitude et de la structure de ce residu
        9. matrice extrinseque finale

        Si tu veux approfondir, le meilleur enchainement est souvent :

        - relire cette cellule finale
        - remonter ensuite a la cellule d'optimisation
        - puis regarder le code de `pose_estimator_gauss_newton.py`
        """
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(f"Notebook written to {NOTEBOOK_PATH}")
