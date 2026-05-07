
Rigid alignment is the first step in bringing reconstructed meshes (`R`) into correspondence with ground-truth
meshes (`G`). It ensures that the comparison between shapes is meaningful and not affected by global rotation or
translation.

### ICP Alignment
FaceBench provides a unified function for rigid alignment using **Iterative Closest Point (ICP)** with multiple optional
**pre-alignment strategies**.

---

```python
  R_aligned, Rlmks_aligned = fb.icp_align(
    R, G,
    prealign="landmark",
    source_lmks=Rlmks,
    target_lmks=Glmks,
    ref_lmk_indices=[13, 19, 28, 31, 37]
)
```

This function performs a **rigid ICP alignment**, optionally preceded by a pre-alignment step that improves robustness
and convergence. You can control how pre-alignment is performed using the `prealign` argument:

- `'landmark'` → Performs Procrustes alignment using landmarks (most accurate).
- `'bbox'` → Roughly aligns the meshes by scaling and centering their bounding boxes.
- `'none'` → Skips pre-alignment. ICP starts from identity (use with caution).

---

##### Parameters

| Name              | Type                | Description                                                   |
|-------------------|---------------------|---------------------------------------------------------------|
| `source_points`   | `(N, 3) np.ndarray` | Reconstructed mesh (`R`)                                      |
| `target_points`   | `(M, 3) np.ndarray` | Ground-truth mesh (`G`)                                       |
| `prealign`        | `str`               | `"landmark"`, `"bbox"` or `"none"`                            |
| `source_lmks`     | `(L, 3) np.ndarray` | Optional landmarks on `R`, required if `prealign="landmark"`  |
| `target_lmks`     | `(L, 3) np.ndarray` | Optional landmarks on `G`, required if `prealign="landmark"`  |
| `ref_lmk_indices` | `List[int]`         | Landmark subset to use during alignment (default: eyes+mouth) |
| `icp_threshold`   | `float`             | Distance threshold for ICP correspondences                    |

---

##### Returns

- `aligned_points`: Aligned source points (after ICP).
- `aligned_landmarks`: Landmarks transformed with the same rigid transform, if provided.

---

#### Internals

This function internally:

1. Applies a **pre-alignment** (if enabled) using either bounding box scaling or landmark-based Procrustes.
2. Converts source and target to Open3D `PointCloud` objects.
3. Runs Open3D’s built-in **Point-to-Point ICP**.
4. Applies the resulting transformation to both the mesh and the landmarks.

This makes the function suitable both for standalone use and for integration in the full FaceBench pipeline.

!!! warning "Landmarks"

    ⚠If neither `prealign` nor landmarks are provided, ICP may fail to converge or produce meaningless results. Always
    prefer at least `prealign='bbox'`.
---


### Landmark-based Alignment

This function performs rigid **Procrustes alignment** between two 3D shapes using a set of corresponding landmarks. It
returns both the aligned mesh and aligned landmarks, transformed using a similarity transformation (scale, rotation,
translation).

---

```python
  R_aligned, Rlmks_aligned = fb.landmark_based_align(
    R, G,
    Rlmks, Glmks,
    ref_lmk_indices=[13, 19, 28, 31, 37]
)
```

---

- Computes the **optimal similarity transform** to align `Rlmks` to `Glmks`.
- Applies the same transform to the full mesh `R`, aligning it to `G`.
- Ideal for initial alignment before applying ICP or non-rigid deformation.

This is the most robust alignment method when corresponding landmarks are available on both shapes.

---

#### Parameters

| Name              | Type                | Description                                 |
|-------------------|---------------------|---------------------------------------------|
| `X`               | `(N, 3) np.ndarray` | Source mesh (e.g., reconstruction).         |
| `Y`               | `(M, 3) np.ndarray` | Target mesh (e.g., ground-truth).           |
| `Xlmks`           | `(L, 3) np.ndarray` | Landmarks on source mesh.                   |
| `Ylmks`           | `(L, 3) np.ndarray` | Landmarks on target mesh.                   |
| `ref_lmk_indices` | `List[int]`         | Indices of reference landmarks to align on. |

---

#### Returns

- `X_aligned` → Source mesh aligned to target.
- `Xlmks_aligned` → Source landmarks after alignment.

---

#### Internals

Internally, this uses a standard Procrustes alignment procedure:

1. **Centers** both landmark sets.
2. **Normalizes** them by scale (L2 norm).
3. Computes the best **rotation** using SVD.
4. **Scales** and **translates** to align the two sets.
5. Applies the transformation to both the full mesh and its landmarks.

---

!!! note

    - Make sure landmarks are in the **same order and correspondence** across meshes.
    - You can choose a subset of stable landmarks via `ref_lmk_indices` (e.g., outer eyes, nose, mouth).
    - Common default: `[13, 19, 28, 31, 37]` (eyes and mouth corners in most facial templates).

---
