
This page documents the core execution logic of the FaceBench pipeline, focusing on how `run_pipeline_batch()` orchestrates all evaluation steps across subjects and methods.

## How it works

The high-level entry point for running a batch experiment is:

??? example "Basic usage"

    ```python
    from facebench import run_pipeline_batch

    errors, vertices, subject_ids, methods = run_pipeline_batch(
        rec_methods=["method1", "method2"],
        g_path="Gmeshes",
        g_lmks_path="Glandmarks",
        config=PipelineConfig(...),
        mm_data=mm,
        base_r_path="Rmeshes"
    )
    ```

This will:

- Detect the subject IDs from the mesh filenames (`idXXXX.txt`)
- Loop over all reconstruction methods
- For each subject-method pair:
  - Load the reconstructed mesh, ground truth mesh, and ground truth landmarks
  - Run the full evaluation pipeline
- Return a **stacked array** of per-vertex errors and aligned vertices

---

## Output structure

The function returns two aligned tensors:

| Name              | Shape                                       | Description                                      |
|-------------------|---------------------------------------------|--------------------------------------------------|
| `errors_stacked`  | `(num_subjects, num_methods, num_vertices)` | Euclidean error per vertex (in meters).         |
| `vertices_stacked`| `(num_subjects, num_methods, num_vertices, 3)` | Aligned coordinates of reconstructed meshes.  |

??? example "Accessing data"

    ```python
    errors = errors_stacked[0, 1]        # errors for subject 0, method 1
    coords = vertices_stacked[0, 1]      # aligned vertices for the same
    ```

---

## Internal pipeline stages

Each subject is processed using the lower-level function `run_pipeline()`.

The main stages are:

### 1. Mesh cropping (optional)
Focuses evaluation on a region (e.g., center of the face).

??? example
    ```python
    G = point_based_crop(G, Glmks, dist_threshold_ratio=1.0, ...)
    ```

---

### 2. Rigid alignment (optional)
Aligns the reconstructed mesh to the ground truth via:

- `landmark_based_align`
- `icp_align` with optional pre-alignment

??? example
    ```python
    R, Rlmks = icp_align(R, G, prealign=True, ...)
    ```

---

### 3. Non-rigid alignment (optional)
Applies a deformation using:

- `landmark_elastic_align` (based on landmarks)
- `nonrigid_icp_align` (dense iterative deformation)

??? example
    ```python
    Rref = nonrigid_icp_align(R.copy(), G, prealign=True, ...)
    ```

---

### 4. Correspondence estimation
Maps each point in `Rref` to a point in `G`, using:

- `chamfer_correspondence`
- `identity_correspondence`

??? example
    ```python
    pidx = chamfer_correspondence(Rref, G)
    ```

---

### 5. Topology correction (optional)
Applies a correction to `G` based on `Rref`, using landmark-informed weights.

??? example
    ```python
    G = topology_consistency_corrector(Rref, G, pidx, mm, strategy="pair", ...)
    ```

---

### 6. Distance computation
Computes the final reconstruction error between aligned mesh and ground truth.

- `p2p_distance`
- `p2tri_distance`

??? example
    ```python
    error = p2tri_distance(R, G, pidx)
    ```

---

## Parallel execution

The batch function uses Python's `multiprocessing.Pool` to parallelize subject-level evaluation across CPU cores.

You can control the number of workers with `max_cores`:

??? example
    ```python
    run_pipeline_batch(..., max_cores=8)
    ```

By default, it uses all available cores.

---

## Summary

- ✅ Modular: every stage is pluggable
- ✅ Parallelized: subject-level parallelism by default
- ✅ Compatible with multiple methods
- ✅ Designed for scalability and easy analysis
'''