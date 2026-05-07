# Croppers

This module provides cropping functions for 3D face meshes. The goal is to reduce the mesh to a subset of vertices centered around a meaningful facial landmark, typically to limit computation or focus analysis on relevant regions.

## Function: `point_based_crop`

This function selects a subset of the 3D mesh by keeping only the vertices that lie within a distance threshold from a reference landmark. The threshold is computed based on the **interpupillary distance (IPD)**, making the method scale-invariant across different face sizes.

```python
def point_based_crop(
    X: np.ndarray,
    Xlmks: np.ndarray,
    dist_threshold_ratio: float = 1.0,
    ref_lmk_index: int = 28,
    leyec_index: int = 36,
    reyec_index: int = 45,
) -> np.ndarray:
```

## Parameters

| Name               | Type          | Default | Description                                                                 |
|--------------------|---------------|---------|-----------------------------------------------------------------------------|
| `X`                | `(N, 3) ndarray` | –       | Source mesh vertices.                                                      |
| `Xlmks`            | `(L, 3) ndarray` | –       | Landmark coordinates on the source mesh.                                   |
| `dist_threshold_ratio` | `float`     | `1.0`   | Distance multiplier applied to the interpupillary distance. Controls how far from the reference landmark vertices are retained. |
| `ref_lmk_index`    | `int`         | `28`    | Index of the reference landmark. Typically the nose tip.                   |
| `leyec_index`      | `int`         | `36`    | Index of the left eye corner landmark.                                     |
| `reyec_index`      | `int`         | `45`    | Index of the right eye corner landmark.                                    |

## Returns

- **`(K, 3) ndarray`**: The cropped subset of vertices from `X` that fall within the threshold distance from the reference landmark.

## How it works

The cropping operation proceeds in three steps:

1. **Interpupillary Distance Calculation**
   It calculates the Euclidean distance between the two eye corner landmarks (`leyec_index` and `reyec_index`) to estimate the scale of the face.

2. **Distance from Reference Point**
   For each vertex in the mesh, it computes the distance to the reference landmark (typically the nose).

3. **Thresholding**
   It retains only the vertices whose distance from the reference point is less than `dist_threshold_ratio × interpupillary_distance`.

This method is useful in contexts where only a central region of the face is of interest, for instance, when aligning or comparing facial reconstructions.
