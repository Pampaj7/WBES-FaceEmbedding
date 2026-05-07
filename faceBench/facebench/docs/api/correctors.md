
Correctors are post-processing modules that adjust reconstructed meshes to improve **topological or geometric consistency** with a reference. This is especially useful when comparing reconstructed meshes with parametric models or when distortions remain after alignment.

This module currently includes a **topology-aware corrector** that leverages landmark-based spatial weighting and Laplacian-like smoothing techniques.

## Topology-aware corrector

```python
def topology_consistency_corrector(
    X: np.ndarray,
    G: np.ndarray,
    corr: np.ndarray,
    mm: Dict,
    correction_strategy: Literal["pair", "trace"] = "pair",
    weight_power: Literal["sqrt", "square"] = "sqrt",
    weight_strategy: Literal["mixed", "min", "mean"] = "mixed"
) -> np.ndarray:
```

### Parameters

| Name                 | Type                   | Default | Description |
|----------------------|------------------------|---------|-------------|
| `X`                  | `(N, 3) ndarray`        | –       | Reconstructed mesh after alignment. Used as the reference. |
| `G`                  | `(M, 3) ndarray`        | –       | Ground-truth mesh to be corrected. |
| `corr`               | `(N,) ndarray`          | –       | Correspondences mapping each point in `X` to a point in `G`. |
| `mm`                 | `dict`                 | –       | Morphable model metadata, including landmark indices and shape. |
| `correction_strategy`| `str` (`"pair"` / `"trace"`) | `"pair"` | Strategy for correction system construction. `"pair"` is local and sparse; `"trace"` is global and dense. |
| `weight_power`       | `str` (`"sqrt"` / `"square"`) | `"sqrt"` | Transformation to apply to the landmark-based weights. |
| `weight_strategy`    | `str` (`"mixed"`, `"min"`, `"mean"`) | `"mixed"` | Strategy for aggregating landmark distance weights. |

### Returns

| Type               | Description |
|--------------------|-------------|
| `(M, 3) ndarray`   | Corrected mesh, same shape as `G`, with deformations applied only on vertices selected by `corr`. |

---

### Internals
The corrector operates in three main steps:

#### 1. Landmark-based per-vertex weights

The corrector uses a morphable model to generate per-vertex weights depending on landmark proximity and overall structure. This is done via:

```python
compute_landmark_base_vertex_weights(mm, weight_power, weight_strategy)
```

Weights ensure that more reliable or structurally important vertices (e.g., around eyes or nose) receive more influence during correction.

---

#### 2. Correction strategies

Two different correction systems are available:

🔹 `"pair"` (default)
- Constructs a sparse Laplacian system based on **neighbor differences**.
- Uses a symmetric tridiagonal matrix.
- Efficient and local: good for smooth, fine-scale adjustment.

🔹 `"trace"`
- Based on a **global residual model**: the overall energy of the deformation is redistributed across the trace.
- Solves a dense system using custom Cholesky decomposition.
- Slightly more aggressive; better for large or distributed bias errors.

---

#### 3. Correction flow

For each axis (`x`, `y`, `z`):
- Extracts reference values from `X`, observed values from `G[corr]`
- Solves a weighted linear system (via LU or custom decomposition)
- Applies the computed correction vector to `G[corr]`

Final result is a **corrected version of `G`**, where distortions introduced by topology mismatches or landmark sparsity are mitigated.

---

!!! note

    - The correction operates **only on a subset of vertices**, preserving the majority of the original structure.
      - The system is compatible with any morphable model containing:
        - `mean_face_shape`
        - `lmk_indices`
        - `leye_oc_rel_index`, `reye_oc_rel_index`
      - The system is especially useful for **synthesized datasets**, where minor offsets can dominate global error metrics.

---
