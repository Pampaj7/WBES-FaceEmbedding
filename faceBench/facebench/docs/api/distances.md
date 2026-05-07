
This module includes metrics to compute geometric distances between two 3D meshes, either using **landmarks**, **point correspondences**, or **triangular surfaces**. These metrics are commonly used in face reconstruction benchmarks to evaluate how closely the reconstructed mesh matches the ground truth.

## Landmark Distance

```python
def landmark_distance(Xlmks: np.ndarray, Ylmks: np.ndarray) -> np.ndarray:
```

### Parameters

| Name     | Type               | Description |
|----------|--------------------|-------------|
| `Xlmks`  | `(L, 3) ndarray`    | Landmark points from the source mesh. |
| `Ylmks`  | `(L, 3) ndarray`    | Corresponding landmark points from the target mesh. |

### Returns

| Type               | Description |
|--------------------|-------------|
| `(L,) ndarray`     | Per-landmark Euclidean distances. |

### Internals

- Verifies shape compatibility between `Xlmks` and `Ylmks`.
- Computes the Euclidean distance for each pair of corresponding landmark points.
- Useful to evaluate alignment error when landmark ground truth is available.

---

## Point-to-Point Distance

```python
def p2p_distance(X: np.ndarray, Y: np.ndarray, pidx: np.ndarray) -> np.ndarray:
```

### Parameters

| Name     | Type               | Description |
|----------|--------------------|-------------|
| `X`      | `(N, 3) ndarray`    | Source mesh. |
| `Y`      | `(M, 3) ndarray`    | Target mesh. |
| `pidx`   | `(N,) ndarray`      | Indices mapping each point in `X` to a point in `Y`. |

### Returns

| Type               | Description |
|--------------------|-------------|
| `(N,) ndarray`     | Per-vertex Euclidean distances between `X[i]` and `Y[pidx[i]]`. |

### Internals

- Efficient way to compute vertex-wise error when correspondences are known (e.g., via Chamfer or identity).
- Raises error if index mismatch occurs or `pidx` is invalid.

---

## Point-to-Triangle Distance

```python
def p2tri_distance(X: np.ndarray, Y: np.ndarray, pidx: np.ndarray) -> np.ndarray:
```

### Parameters

| Name     | Type               | Description |
|----------|--------------------|-------------|
| `X`      | `(N, 3) ndarray`    | Source mesh points. |
| `Y`      | `(M, 3) ndarray`    | Target mesh points. |
| `pidx`   | `(N,) ndarray`      | Indices mapping each point in `X` to a subset of `Y` for triangle construction. |

### Returns

| Type               | Description |
|--------------------|-------------|
| `(N,) ndarray`     | Distance from each point in `X` to a triangle in `Y` defined by its 3 nearest neighbors. |

### Internals

- For each point in `X`, finds its 3 nearest neighbors in `Y[pidx]`.
- Constructs a triangle using these 3 points.
- Computes the minimum distance between the source point and the triangle using:
  - Projection onto the triangle plane
  - Barycentric coordinates
  - Segment fallback for degenerate cases
- More accurate and robust than simple point-to-point distance in surface evaluation contexts.

---

!!! notes

    - `landmark_distance` is typically used when evaluating alignment accuracy.
      - `p2p_distance` is fast and simple, ideal when dense correspondences are known.
      - `p2tri_distance` is computationally heavier but captures geometric errors more realistically, especially on non-isomorphic surfaces.

---
