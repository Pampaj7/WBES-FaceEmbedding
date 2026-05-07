

This section provides a detailed overview of the non-rigid alignment functions available in the FaceBench library. These functions are designed to perform non-rigid alignment of 3D meshes, particularly in the context of facial reconstructions.
## Landmark Elastic Alignment
This function performs non-rigid alignment of a source mesh to a target mesh using a set of corresponding landmarks. The method is based on elastic deformation, where the source mesh is deformed to match the target mesh while preserving smoothness and regularity.

```python
def landmark_elastic_align(
    R: np.ndarray,
    G: np.ndarray,
    Glmks: np.ndarray,
    lmk_indices: List[int],
    gamma: float = 1.0,
    sel_lmk_ids: List[int] = list(range(51))
) -> np.ndarray:
```


### Parameters

| Name            | Type               | Default              | Description |
|-----------------|--------------------|----------------------|-------------|
| `R`             | `(N, 3) ndarray`   | –                    | Source mesh (to be aligned). |
| `G`             | `(M, 3) ndarray`   | –                    | Target mesh (reference geometry). |
| `Glmks`         | `(L, 3) ndarray`   | –                    | Landmarks on the target mesh. |
| `lmk_indices`   | `List[int]`        | –                    | Indices of the landmarks on the source mesh corresponding to `Glmks`. |
| `gamma`         | `float`            | `1.0`                | Weighting exponent that controls the influence radius of each landmark during deformation. Higher values make the influence more localized. |
| `sel_lmk_ids`   | `List[int]`        | `list(range(51))`    | Subset of landmark indices (referring to `lmk_indices`) used to define alignment constraints. |

### Returns

| Type              | Description |
|-------------------|-------------|
| `(N, 3) ndarray`  | A new set of vertices representing the **non-rigidly aligned** version of the source mesh `R`. The deformation is smooth and regularized, with displacements constrained by the selected landmarks. |

### Internals

The algorithm follows these main steps:

1. **Distance matrix computation**
   For each landmark in `lmk_indices`, it computes the distance to every other point in the source mesh using a KD-tree. This produces a matrix `D` of shape `(N, L)` (N = points, L = landmarks), where each column holds the distances from a given landmark.

2. **Weight normalization**
   Each column of the distance matrix is normalized to the range `[0, 1]` and flipped (`1 - d`) so that closer points get higher weights. The weights are then raised to the power `gamma`.

3. **Constraint construction**
   The displacement from each source landmark to the corresponding target landmark is computed in all three axes (`x`, `y`, `z`) using only the subset `sel_lmk_ids`. These form the constraint vector `b`.

4. **Optimization**
   For each axis, the method solves a constrained least-squares problem using CVXPY:

    ??? example "Optimization objective"

        ```python
            minimize ‖D_lmk_subset × w - b‖²
            subject to ‖D_sub × w‖_∞ ≤ max(|b|)
        ```

    This regularizes the deformation globally while fitting local landmark displacements.

5. **Reconstruction**
   The computed deformation weights `w` are then applied to the full mesh to produce the final deformed (aligned) output.

!!! note

    - The function is robust to moderately noisy landmarks due to the global regularization.
      - This method is especially effective when the number of landmarks is large but uniformly distributed.
      - The optimization fallback from Clarabel to SCS ensures compatibility in case a solver fails.

---

## Non-rigid ICP Alignment

This section provides a non-rigid Iterative Closest Point (ICP) algorithm that deforms a source mesh to match a target mesh by minimizing a weighted distance term and a regularization constraint. Optionally, it can perform a preliminary elastic alignment based on landmarks.


```python
def nonrigid_icp_align(
    source_points: np.ndarray,
    target_points: np.ndarray,
    gamma: float = 1.0,
    alpha: float = 50.0,
    epsilon: float = 1.0,
    source_point_lmks: Optional[np.ndarray] = None,
    lmk_indices: Optional[np.ndarray] = None,
    prealign: bool = False,
) -> np.ndarray:
```

### Parameters

| Name                 | Type                   | Default   | Description |
|----------------------|------------------------|-----------|-------------|
| `source_points`      | `(N, 3) ndarray`        | –         | Source mesh vertices to be aligned. |
| `target_points`      | `(M, 3) ndarray`        | –         | Target mesh vertices used as alignment reference. |
| `gamma`              | `float`                | `1.0`     | Elastic regularization factor that controls smoothness of deformation. |
| `alpha`              | `float`                | `50.0`    | Weight of the regularization term relative to the data term. |
| `epsilon`            | `float`                | `1.0`     | Convergence threshold. Iterations stop when changes fall below this value. |
| `source_point_lmks`  | `(L, 3) ndarray`        | `None`    | Optional landmarks on the source mesh, used for pre-alignment. |
| `lmk_indices`        | `ndarray`              | `None`    | Indices of the source mesh vertices corresponding to landmarks. |
| `prealign`           | `bool`                 | `False`   | Whether to apply a preliminary elastic alignment based on landmarks. |

### Returns

| Type              | Description |
|-------------------|-------------|
| `(N, 3) ndarray`  | Aligned version of the source mesh, after applying non-rigid deformation to match the target mesh. |

### Internals

The algorithm works by alternating between **correspondence estimation** and **non-rigid deformation solving**, and includes optional pre-alignment:

1. **Landmark-based pre-alignment (optional)**
   If `prealign=True`, the function performs a smooth elastic alignment based on facial landmarks using `landmark_elastic_align`.

2. **Triangulation and adjacency matrix**
   A 2D Delaunay triangulation of the source mesh is computed. The edge-to-vertex incidence matrix is transformed into a regularization matrix `A1`.

3. **Initialization**
   All data is converted to homogeneous coordinates and placed into sparse matrices to allow efficient solving.

4. **Iterative alignment process** for each iteration block (3 times), the algorithm:
      - Computes nearest neighbors from the current deformed source to the target mesh.
      - Builds a sparse system of equations combining regularization (`A1`) and data terms (`A2`, `B2`).
      - Solves the system using sparse LU factorization and applies the update.

5. **Matrix solving details**
       - The solver uses permutation matrices and factorizes the normal equations to solve `AᵀA x = Aᵀb` robustly even for large sparse systems.
    - It supports multiple decay levels to progressively refine the deformation with decreasing tolerance.


!!!Note

    - The convergence threshold is adaptive across decay levels, making the deformation increasingly rigid.
      - The combination of spatial regularization and iterative fitting makes it robust to noisy or incomplete target meshes.
      - Best used when the source and target are roughly in the same coordinate system (e.g., post-rigid alignment or prealigned landmarks).

---
