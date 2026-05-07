# Configuration

FaceBench uses **structured dataclasses** to define each step of the evaluation pipeline.
This ensures that the configuration is **modular**, **type-safe**, and **autocompletable** in modern editors.

Each pipeline component (e.g., alignment, cropping, correction...) has its own config class and can be enabled or skipped independently.

---

## PipelineConfig

The master configuration object passed to `run_pipeline_batch()`.

```python
fb.PipelineConfig(
    mesh_cropper=...,
    rigid_aligner=...,
    nonrigid_aligner=...,
    corr_establisher=...,
    corrector=...,
    distance_computer=...
)
```

All fields are optional — only include what you need.

---

## MeshCropperConfig

The mesh cropper trims the mesh around a central region based on interocular distance.
It is useful to discard noisy or irrelevant regions like the neck, ears, or shoulders.

=== "Minimal"

    ```python
    fb.MeshCropperConfig(
        method=fb.MeshCropperType.POINT_BASED
    )
    ```

    ✅ This uses the default values:

    - `dist_threshold_ratio = 1.0`
    - `ref_lmk_index = 28` (typically nose tip)
    - `leyec_index = 36`, `reyec_index = 45` (eye corners)


=== "Custom Parameters"

    ```python
    fb.MeshCropperConfig(
        method=fb.MeshCropperType.POINT_BASED,
        dist_threshold_ratio=0.6,
        ref_lmk_index=30,
        leyec_index=36,
        reyec_index=45
    )
    ```

    ✅ You can tune `dist_threshold_ratio` to crop more or less of the face.

    - Lower values → tighter crops
    - Higher values → larger region


!!! tip "Face-only cropping"
    Reduces noise by discarding irrelevant mesh parts (e.g., ears, neck, shoulders).

---

## RigidAlignerConfig

Configure rigid alignment with either **ICP** or **landmark-based Procrustes**.

=== "Prealign: LANDMARK"

    ```python
    fb.RigidAlignerConfig(
        type=fb.RigidAlignerType.ICP,
        prealign=fb.PrealignMethod.LANDMARK,
        ref_lmk_indices=[13, 19, 28, 31, 37],
    )
    ```

=== "Prealign: BBOX"

    ```python
    fb.RigidAlignerConfig(
        type=fb.RigidAlignerType.ICP,
        prealign=fb.PrealignMethod.BBOX,
    )
    ```

=== "Prealign: NONE"

    ```python
    fb.RigidAlignerConfig(
        type=fb.RigidAlignerType.ICP,
        prealign=fb.PrealignMethod.NONE,
    )
    ```

Available `prealign` values:

- `"landmark"`: Best accuracy, uses 3D landmarks
- `"bbox"`: Coarse alignment using bounding box
- `"none"`: Starts ICP from identity

!!! tip "Best practice"
    Use `"landmark"` when accurate landmarks are available.

---

## NonRigidAlignerConfig

Applies non-rigid deformation after rigid alignment.
Supports two modes: `ELASTIC` (landmark-based) and `NICP` (dense).

=== "Elastic - Minimal"

    ```python
    fb.NonRigidAlignerConfig(
        type=fb.NonRigidAlignerType.ELASTIC
    )
    ```

    ✅ Uses default values:
    - `gamma=1.0`
      - uses all landmarks (if provided in the pipeline)

=== "Elastic - Custom"

    ```python
    fb.NonRigidAlignerConfig(
        type=fb.NonRigidAlignerType.ELASTIC,
        gamma=2.0,
        ref_lmk_indices=mm["lmk_indices"],
        sel_lmk_ids=list(range(51))
    )
    ```

    ✅ Allows you to:
    - Control deformation stiffness with `gamma`
      - Specify which landmarks to use via `sel_lmk_ids`
      - Provide landmark locations via `ref_lmk_indices`

---

=== "NICP - Minimal"

    ```python
    fb.NonRigidAlignerConfig(
        type=fb.NonRigidAlignerType.NICP
    )
    ```

    ✅ Uses default values:
    - `gamma=1.0`, `alpha=50.0`, `epsilon=1.0`, `prealign=False`

    > 🔗 Landmark data is **not required** for NICP
    > It aligns dense vertex clouds directly.

=== "NICP - Custom"

    ```python
    fb.NonRigidAlignerConfig(
        type=fb.NonRigidAlignerType.NICP,
        gamma=0.5,
        alpha=75.0,
        epsilon=0.5,
        prealign=True
    )
    ```

    ✅ Fine-tune:
    - `gamma`: elastic smoothing factor
      - `alpha`: regularization weight
      - `epsilon`: convergence threshold
      - `prealign=True`: optionally apply elastic landmark alignment before starting

!!! tip "Which one to use?"
    - Use **Elastic** if you trust your landmark annotations and want smooth, localized corrections
    - Use **NICP** if you need to recover from large shape mismatches without relying on landmarks

---

## CorrEstablisherConfig

Defines how correspondence is established between the reconstructed mesh `R` and the ground-truth mesh `G`.

In most cases, you'll want to use the **Chamfer** method — it automatically finds the nearest neighbor in `G` for each point in `R`, even if the two meshes have different topology or number of vertices.

=== "Chamfer (Recommended)"

    ```python
        fb.CorrEstablisherConfig(
            type=fb.CorrEstablisherType.CHAMFER
        )
    ```

    ✅ Uses a KD-tree to find the closest point in `G` for each vertex in `R`.
    ✅ Robust to different vertex count or sampling.

=== "Identity (Advanced)"

    ```python
    fb.CorrEstablisherConfig(
        type=fb.CorrEstablisherType.IDENTITY
    )
    ```

    ✅ Fastest option — **but only valid** when:

    - `R.shape == G.shape`
      - Each vertex `R[i]` corresponds exactly to `G[i]`
      - The two meshes are already aligned and have identical topology

!!! tip "When to use IDENTITY"
    Use `IDENTITY` **only** when you know that `R` and `G` are in perfect correspondence.
    Otherwise, prefer `CHAMFER`, which is safer and more flexible.

---

## CorrectorConfig

Applies a **topology-aware correction** to the ground-truth mesh `G` to better match the aligned reconstructed mesh `R`.

This is particularly useful when:
- `G` has slight inconsistencies, noise, or different sampling
- You want to **fairly evaluate** per-vertex distances

=== "Minimal (Recommended)"

    ```python
    fb.CorrectorConfig()
    ```

    ✅ Applies the default correction strategy: `"pair"`
    ✅ Uses a mixed landmark weighting scheme and `sqrt` falloff

=== "Fully Custom"

    ```python
    fb.CorrectorConfig(
        type=fb.CorrectorType.TOPOLOGY_CONSISTENCY,
        correction_strategy=fb.CorrectionStrategy.PAIR,
        weight_power=fb.WeightPower.SQRT,
        weight_strategy=fb.WeightStrategy.MIXED
    )
    ```

## Available Options

- `correction_strategy`:
  `"pair"` (default): pairwise smoothness
  `"trace"`: trace-based regularization
- `weight_power`:
  `"sqrt"` (default): moderate falloff
  `"square"`: sharper influence from landmarks
- `weight_strategy`:
  `"mixed"` (default): combines strategies
  `"min"`: conservative
  `"mean"`: smoother but possibly oversmoothed

!!! tip "Keep it simple"
    The default configuration (`PAIR + SQRT + MIXED`) works well in most cases.
    You only need to customize this if you're experimenting with correction behavior or analyzing edge cases.
---

## DistanceComputerConfig

Defines **how the final geometric error is computed** between the reconstructed and ground-truth meshes.

This is the last step in the pipeline and determines how accuracy is measured.

=== "Point-to-Triangle"

    ```python
    fb.DistanceComputerConfig(
        type=fb.DistanceComputerType.P2TRI
    )
    ```

=== "Point-to-Point"

    ```python
    fb.DistanceComputerConfig(
        type=fb.DistanceComputerType.P2P
    )
    ```

## Options

- `P2P` — Point-to-Point:
  Measures Euclidean distance between corresponding points.

- `P2TRI` — Point-to-Triangle:
  Measures shortest distance from each reconstructed point to the **surface** of the ground-truth mesh.

!!! tip "Best Practice"
    Use `P2TRI` when reconstructions may have minor **topological inconsistencies** (e.g., slight misalignments, different sampling).
    It provides a more stable and meaningful evaluation, especially for surface-aware analysis.
---