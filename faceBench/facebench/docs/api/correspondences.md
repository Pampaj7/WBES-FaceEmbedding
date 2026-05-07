
This module provides methods to compute point correspondences between two 3D point clouds or meshes. Correspondences are a key step in evaluating or aligning 3D reconstructions, as they determine which vertices in the source and target meshes are compared.

## Chamfer Correspondence

```python
def chamfer_correspondence(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
```

### Parameters

| Name  | Type             | Description |
|--------|------------------|-------------|
| `X`    | `(N, 3) ndarray` | Source point cloud (e.g., reconstructed mesh). |
| `Y`    | `(M, 3) ndarray` | Target point cloud (e.g., ground-truth mesh).   |

### Returns

| Type             | Description |
|------------------|-------------|
| `(N,) ndarray`   | For each point in `X`, the index of its nearest neighbor in `Y`. |

### Internal details

- A **KD-tree** is constructed on `Y` for efficient nearest-neighbor queries.
- For each point in `X`, the algorithm finds the closest point in `Y` (Euclidean distance, `k=1`).
- This corresponds to the **Chamfer matching** direction: `X → Y`.

This function is used when `X` and `Y` may have different numbers of points or are not trivially aligned. It does not enforce symmetric matching unless called twice in both directions.

---

## Identity Correspondence

```python
def identity_correspondence(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
```

### Parameters

| Name  | Type             | Description |
|--------|------------------|-------------|
| `X`    | `(N, 3) ndarray` | Source mesh. |
| `Y`    | `(N, 3) ndarray` | Target mesh, assumed to be aligned and in one-to-one correspondence with `X`. |

### Returns

| Type             | Description |
|------------------|-------------|
| `(N,) ndarray`   | The identity correspondence: `[0, 1, ..., N-1]`. |

### Internal details

- Assumes that `X` and `Y` are in **perfect point-wise correspondence**.
- Performs a shape check and raises a `ValueError` if `X` and `Y` do not have the same number of points.

This is ideal for synthetic datasets or when you know that both meshes share the same topology or vertex ordering (e.g., aligned parametric models like FLAME or BFM).

---
