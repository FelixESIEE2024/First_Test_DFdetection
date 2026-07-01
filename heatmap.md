# Reproduire Le Rendu Des Heatmaps Geometriques

Ce document explique comment le repo `GeCo` produit ses heatmaps d'erreurs geometriques avec un rendu propre, lisible et stable. Le but est de donner une recette directement reutilisable dans un autre projet.

## Idee generale

Le beau rendu ne vient pas d'un effet graphique complexe. Il vient surtout de quatre decisions:

1. calculer une erreur geometrique propre et physiquement coherente
2. masquer les pixels peu fiables
3. moyenner l'erreur sur plusieurs vues voisines pour reduire le bruit
4. coloriser avec une normalisation robuste avant de composer une figure propre

Autrement dit: la qualite visuelle est surtout un resultat de la qualite du signal avant affichage.

## Pipeline global

Le pipeline observe dans le repo est le suivant:

1. prendre une image source
2. comparer cette image a plusieurs frames voisines
3. calculer une erreur de mouvement 2D
4. calculer une erreur de reprojection / profondeur
5. fusionner ces erreurs selon la visibilite et les occlusions
6. moyenner les cartes sur la fenetre temporelle
7. appliquer une colormap avec une plage dynamique robuste
8. composer une grille avec l'image source et les heatmaps

## 1. Erreur de mouvement

Le repo compare:

- un flow observe entre source et target
- un flow rigide predit a partir de la profondeur et du mouvement camera

La carte d'erreur mouvement est alors le residu entre les deux:

```text
residual_flow = observed_flow - rigid_flow
```

Puis le residu est normalise par la focale pour obtenir une erreur plus stable entre images:

```text
angular_like_error = residual_flow / [fx, fy]
motion_error = norme_L2(angular_like_error)
```

Pourquoi c'est important:

- une erreur en pixels purs depend trop de la resolution et de l'intrinsique
- la normalisation par `fx, fy` donne une mesure plus transferable

## 2. Erreur de structure / reprojection

Le repo reprojette la geometrie source dans la vue cible:

1. backprojection des pixels source en 3D a partir de la profondeur
2. transformation 3D avec la pose relative source -> cible
3. reprojection dans l'image cible
4. echantillonnage de la profondeur cible au point reprojete
5. comparaison entre profondeur predite et profondeur observee

La forme de l'erreur est:

```text
depth_error = abs(depth_target_sampled - depth_reprojected) / depth_source
```

Le fait de diviser par la profondeur source rend la mesure relative, donc plus stable entre objets proches et lointains.

## 3. Gestion des zones fiables

Le rendu est joli parce qu'ils n'affichent pas tout.

Ils eliminent plusieurs types de pixels:

- pixels hors champ apres reprojection
- profondeurs invalides ou non finies
- zones de faible confiance sur la profondeur
- zones non covisibles selon le modele de flow
- zones incoherentes avec l'occlusion

Ils construisent notamment un masque de confiance a partir:

- d'un seuil percentile sur la confiance
- d'un seuil absolu minimum

Recette:

```text
confidence_mask = (conf >= percentile(conf, p)) and (conf > conf_min)
```

Pourquoi c'est important:

- les heatmaps deviennent beaucoup moins salees
- les bords faux et les trous de profondeur n'ecrasent plus le contraste

## 4. Gestion des occlusions

Le repo ne fusionne pas naivement mouvement et profondeur.

Il distingue essentiellement:

1. pixels valides en mouvement et en structure
2. pixels ou le mouvement est valide mais la profondeur ne l'est pas
3. pixels interpretes comme occlusions incorrectes

La logique de fusion est:

- si mouvement et structure sont valides: moyenne des deux erreurs
- si l'occlusion estimee par le flow semble fausse: utiliser plutot l'erreur de profondeur
- si seule l'erreur mouvement est fiable: garder l'erreur mouvement

Pseudo-code:

```python
fused = zeros_like(motion_error)

if non_occluded_and_valid:
    fused = (motion_error + depth_error) / 2

if wrong_occlusion:
    fused = depth_error

if motion_valid_but_depth_invalid:
    fused = motion_error
```

Pourquoi c'est important:

- la carte finale reste interpretable
- les zones d'occlusion ne polluent pas tout le rendu

## 5. Moyennage temporel

Chaque frame source est comparee a plusieurs voisines dans une fenetre glissante. Les cartes sont accumulees puis moyennees pixel par pixel en tenant compte du masque de validite.

Recette:

```text
sum_map += error * valid_mask
count_map += valid_mask
avg_map = sum_map / max(count_map, 1)
```

Pourquoi c'est important:

- le bruit diminue
- les motifs geometriques coherents ressortent mieux
- le rendu devient plus "dense" et plus stable visuellement

## 6. Le vrai secret du beau rendu: la normalisation robuste

La partie la plus importante pour l'esthetique est la colorisation.

Le repo n'utilise pas un min/max brut. Il calcule une plage robuste avec des percentiles:

```text
vmin = percentile(values, 2)
vmax = percentile(values, 98)
```

Puis:

- clamp des valeurs dans `[0, 1]`
- garde un `vmax` minimum pour eviter une image plate
- ignore les NaN / inf

Pseudo-code:

```python
vals = arr[isfinite(arr)]
vmin = percentile(vals, 2)
vmax = percentile(vals, 98)
vmax = max(vmax, min_vmax)
norm = clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
```

Pourquoi c'est crucial:

- un seul outlier ne casse pas tout le contraste
- les structures fines deviennent visibles
- le rendu reste stable d'une frame a l'autre

## 7. Choix des colormaps

Le repo utilise trois colormaps differentes:

- `magma` pour la motion map
- `viridis` pour la structure map
- `plasma` pour la fused map

Pourquoi ca marche bien:

- ces colormaps sont perceptuellement agreables
- elles gardent du contraste dans les basses valeurs
- elles evitent en grande partie les artefacts visuels de `jet`

Conseil pratique:

- si tu veux un rendu proche, garde exactement ces colormaps
- si tu veux une version encore plus "scientifique", `viridis` partout marche bien aussi

## 8. Composition de l'image finale

Le rendu final n'est pas une simple heatmap seule. Ils composent une grille 1x4:

1. image source
2. motion map
3. structure map
4. fused map

Choix de presentation:

- axes caches
- un titre simple par panneau
- `tight_layout`
- export direct du canvas en RGB

Pourquoi ca aide:

- on compare instantanement la scene et les cartes
- la lecture est beaucoup plus convaincante pour une demo ou une revue

## Recette minimale a repliquer

Si ton autre projet a deja une carte d'erreur 2D float, le minimum pour retrouver un rendu similaire est:

1. nettoyer la carte: NaN, inf, pixels invalides
2. calculer `vmin/vmax` avec percentiles 2/98
3. normaliser dans `[0, 1]`
4. appliquer une colormap `magma`, `viridis` ou `plasma`
5. convertir en RGB 8-bit
6. afficher dans une grille avec l'image source

## Reference de pseudo-code reproductible

```python
import numpy as np
import matplotlib


def robust_vrange(arr, lo=2, hi=98, fallback=(0.0, 1.0), floor_val=0.05):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return fallback
    vmin, vmax = np.percentile(vals, [lo, hi])
    if not np.isfinite(vmin):
        vmin = fallback[0]
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    vmax = max(vmax, floor_val)
    return float(vmin), float(vmax)


def colorize_error_map(error_map, cmap_name="plasma", min_vmax=0.05):
    error_map = np.asarray(error_map, dtype=np.float32)
    vmin, vmax = robust_vrange(error_map, floor_val=min_vmax)
    norm = (error_map - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = cmap(norm)[..., :3]
    return (rgb * 255).astype(np.uint8)
```

## Version complete type "GeCo"

Pour reproduire encore plus fidelement le rendu, il faut aussi reprendre les ingredients suivants:

- erreur mouvement = difference entre flow observe et flow rigide
- erreur structure = erreur relative de profondeur apres reprojection
- masques de confiance
- logique d'occlusion
- moyenne sur plusieurs frames voisines

Sans ces etapes, tu peux obtenir une jolie heatmap. Avec elles, tu obtiens une heatmap jolie et semantiquement utile.

## Parametres recommandes

Valeurs proches de ce repo:

- percentiles d'affichage: `2` et `98`
- plancher pour `vmax`: `0.05`
- masque confiance par percentile: `20`
- confiance minimale absolue: `0.2`
- seuil de covisibilite flow: `0.5`
- seuil relatif d'occlusion profondeur: `0.02`

Ces valeurs sont de bons points de depart, mais elles dependent du niveau de bruit de ton pipeline.

## Ce qu'il faut demander au dev

Si l'objectif est "faire pareil", demande explicitement:

1. une carte d'erreur float par pixel avant visualisation
2. un masque de validite par pixel
3. une normalisation robuste par percentiles
4. une colorisation Matplotlib avec `magma` / `viridis` / `plasma`
5. une figure finale avec source + cartes
6. idealement un lissage temporel par moyennage sur plusieurs vues

## Conclusion

La signature visuelle du rendu vient principalement de:

- une erreur geometrique bien definie
- des masques de validite stricts
- une fusion intelligente des cas d'occlusion
- un moyennage temporel
- une normalisation robuste avant colormap

Si tu ne devais copier qu'une seule chose, ce serait la normalisation robuste par percentiles avant application de la colormap. C'est le levier le plus simple et le plus rentable pour obtenir immediatement un rendu nettement plus propre.
