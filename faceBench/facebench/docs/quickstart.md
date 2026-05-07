# ⚡ Quickstart

This guide shows how to get started with **FaceBench** in minutes.
You’ll load meshes and compute evaluation errors — all in a few lines of code.
(TODO need to change when the library will be published)
---

## Install

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or create a new environment:

```bash
conda create -n facebench python=3.8
conda activate facebench
pip install -r requirements.txt
```


---

## Minimal Setup

To run the minimal example, you only need **4 essential files**:

### Required Files

- ✅ `G.txt`: Ground-truth mesh
- ✅ `R.txt`: Reconstructed mesh
- ✅ `G.lmks`: Ground-truth landmarks
- ✅ `BFM-p23470.json`: Morphable model info file (containing `lmk_indices`)

These files are loaded individually and passed directly to FaceBench functions, without any assumption on folder
structure or naming conventions.

---

### Example Layout

```
your_project/
├── id0000.txt          ← Reconstructed mesh (R)
├── id0000_gt.txt       ← Ground-truth mesh (G)
├── id0000.lmks         ← Ground-truth landmarks
└── BFM-p23470.json     ← Morphable model metadata
```

> 🧠 Note: These files can be located anywhere — you just need to pass the correct paths to `np.loadtxt()`
> and `json.load()` in your script.


---

## Minimal Working Example

```python
##(step-by-step)

import facebench as fb
import numpy as np
import json

r_path = "..your_path"  # .txt file with reconstructed mesh
g_path = "..your_path"  # .txt file with ground-truth mesh
g_lmk_path = "..your_path"  # .lmks file with ground-truth landmarks

R = np.loadtxt(r_path)
G = np.loadtxt(g_path)
Glmks = np.loadtxt(g_lmk_path)

with open("..your_path") as f:
    mm = json.load(f)

Rlmks = R[mm["lmk_indices"]]
```

To compute alignment, you must extract **landmark coordinates from your reconstructed mesh `R`**.
This is done using a list of landmark indices — either manually or by loading them from a `.json` file.

✅ If you have a `.json` metadata file from a morphable model, it likely includes something like:

```python
Rlmks = R[mm["lmk_indices"]]
```

🧠 If you **don't have a JSON**, that's also fine — just select the landmark vertices manually:

```python
Rlmks = R[[13, 19, 28, 31, 37]]  # Example
```

✅ The only requirement is:

- The landmark coordinates in `Rlmks` must **semantically correspond** to those in `Glmks`
- The library doesn't care *how* you extract them — just that they match

!!!note
    For more information about the data format, see the [Data Format](data_format.md) section.

---

### Run the Evaluation

Once you've loaded these files, you can evaluate reconstruction quality by chaining together:

- Mesh cropping
- Rigid alignment
- Non-rigid alignment
- Correspondence
- Topology correction
- Distance computation

---

## Mesh Cropping

Crop the mesh to focus only on the central facial region.

- `R`: reconstructed mesh
- `Rlmks`: landmarks on the mesh
- A reference landmark (e.g. the eye corners)
- A `dist_threshold_ratio` to control how much of the face is kept

This removes vertices that are too far from the reference landmark — useful to discard ears, neck, etc.

```python
# Optional: Crop the face around the central region
R_cropped = fb.point_based_crop(
    R, Rlmks,
    dist_threshold_ratio=0.6,
    ref_lmk_index=30,
    leyec_index=36,
    reyec_index=45
)
```
!!! warning "️ Avoid double-cropping"
    Cropping is meant to **remove peripheral regions** like ears or neck.
    However, if your input mesh is **already cropped or focused on the face**, applying an additional crop can:

    - Remove valid vertices
    - Introduce **artificial alignment errors**
    - Reduce evaluation accuracy

    ✅ Always **check your data** before applying mesh cropping.

## Rigid Alignment

FaceBench provides two flexible functions to perform rigid alignment between the reconstructed mesh (`R`)
and the ground-truth (`G`).

### ICP
The library provides **Iterative Closest Point (ICP)** and supports optional **pre-alignment strategies** to
improve convergence.

!!! note "🧭 Prealignment Options"
    You can pass one of the following values to the `prealign` parameter:

    - `"landmark"`: Uses Procrustes alignment based on landmarks (requires `source_lmks` and `target_lmks`)
    - `"bbox"`: Centers and scales meshes based on bounding box
    - `None`: No prealignment; ICP starts from identity

Here are a few working examples:

=== "Landmark-based pre-alignment"
    ```python
    R_aligned, Rlmks_aligned = fb.icp_align(
        R, G,
        prealign="landmark",
        source_lmks=Rlmks,
        target_lmks=Glmks
    )
    ```
    !!! warning "️ Landmark pre-alignment"
        This requires that `Rlmks` and `Glmks` are semantically aligned (e.g., same order and meaning).

=== "Bounding-box based pre-alignment"
    ```python
    R_aligned, _ = fb.icp_align(R, G, prealign="bbox")
    ```
=== "No pre-alignment (identity start)"
    ```python
    R_aligned, _ = fb.icp_align(R, G)
    ```
    !!! warning "️ No pre-alignment"
        This may lead to suboptimal results if the meshes are far apart.

### Landmark-Based Alignment (Procrustes)

If you prefer to perform **deterministic alignment** based solely on landmarks, you can use:

```python
R_aligned, Rlmks_aligned = fb.landmark_based_align(R, G, Rlmks, Glmks)
```

This uses **Procrustes alignment**, which estimates the best similarity transformation (scale + rotation + translation) between `Rlmks` and `Glmks`, and then applies it to the full mesh `R`.

Example:
```python
R_aligned, Rlmks_aligned = fb.landmark_based_align(
    R, G, Glmks, Rlmks,
    ref_lmk_indices=[13, 19, 28, 31, 37]  # default
)
```
!!! info "🔢 Reference Landmark Indices"
    You can customize which landmarks are used for alignment by passing a list of indices to `ref_lmk_indices`.


These indices select a subset of landmarks (typically central and stable points) to drive the alignment.

This method is useful if:

- You want **repeatable alignment**
- You have **consistent landmarks**
- You don’t want to rely on iterative methods like ICP


## Non-Rigid Alignment

In some cases, rigid alignment might not be sufficient to accurately superimpose the reconstructed mesh `R` onto the ground-truth mesh `G`, especially when dealing with **non-rigid deformations**. FaceBench offers two methods to handle such scenarios: **Non-Rigid ICP** and **Elastic Alignment**.

---

### Non-Rigid ICP

The **Non-Rigid ICP** algorithm allows local deformations while aligning the reconstructed mesh `R` to the ground-truth mesh `G`. It's particularly useful when rigid alignment fails due to shape mismatch caused by expressions or modeling variations.

```python
R_deformed = fb.nonrigid_icp_align(R, G)
```

🧠 **Parameters**:

- `R`: Reconstructed mesh (source)
- `G`: Ground-truth mesh (target)
✅ This function returns a non-rigidly deformed version of `R` that better matches `G`, especially useful when working with facial expressions or non-neutral scans.

!!! warning "Execution time"
    The non-rigid ICP algorithm can be computationally intensive, especially for large meshes.

---

### Landmark Elastic Alignment

This method performs **non-rigid alignment** by smoothly deforming the reconstructed mesh `R` to match the ground-truth mesh `G`, guided by a subset of landmark constraints.

```python
R_deformed = fb.landmark_elastic_align(
    R, G, Glmks,
    lmk_indices=mm["lmk_indices"],
    sel_lmk_ids=list(range(51))
)
```

🧠 **Parameters**:

- `R`: Reconstructed mesh (source)
- `G`: Ground-truth mesh (target)
- `Glmks`: Ground-truth landmarks (in the same format as the ones used in alignment)
- `lmk_indices`: Indices in `R` that correspond to each landmark
- `sel_lmk_ids`: Indices (relative to `lmk_indices`) of the landmarks used to drive the deformation

✅ This method is typically used **after rigid alignment** to fine-tune the shape and handle localized deformations — like expressions or noise in reconstruction.

💡 `sel_lmk_ids` defaults to the first 51 landmarks, which are usually those around the face contour, eyes, nose, and mouth.

## Correspondence

Once the meshes are aligned, we need to establish **point-wise correspondence** between the reconstructed mesh `R` and the ground-truth mesh `G`.
This determines **which points to compare** when computing distances.

---

### Chamfer Correspondence

This is the most commonly used method. It finds the **nearest neighbor in `G` for each point in `R`** using a KD-tree.

```python
corr = fb.chamfer_correspondence(R, G)
```

✅ Works even if the two meshes have **different number of vertices**.
✅ Robust to slight misalignments or sampling differences.

---

### Identity Correspondence

This assumes that `R` and `G` already have **exactly the same topology and number of vertices**, and that points are ordered consistently.

```python
corr = fb.identity_correspondence(R, G)
```

!!! warning "️Use with caution"
    Use this **only if you're sure** that `R[i]` corresponds exactly to `G[i]`.
    Otherwise, the error computation will be meaningless or misleading.



---

## Correction

After correspondence, it's possible that some **topological mismatches** still exist between the aligned reconstruction `R` and the ground-truth mesh `G`.

To reduce local inconsistency, FaceBench offers a **topology-aware correction** mechanism.

---

This function adjusts the ground-truth mesh `G` **at the corresponding points**, based on its local deviation from `R`.

```python
G_corrected = fb.topology_consistency_corrector(R_aligned, G, corr, mm)
```

✅ The result is a smoother and more locally coherent shape for comparison.

---

⚙️ Parameters

- `R_aligned`: the aligned reconstructed mesh (typically the output of alignment)
- `G`: the ground-truth mesh
- `corr`: correspondence indices between `R_aligned` and `G`
- `mm`: morphable model info, including landmark weights
- `correction_strategy`:  `"pair"` (default) or `"trace"` for the type of smoothness applied
- `weight_power`: `"sqrt"` (default) or `"square"`  defines how strongly landmark proximity affects correction
- `weight_strategy`: `mixed`(dafault) or `min` or `mean` (`"mixed"` is usually best)

---

🧠 When to Use

- Useful when `G` contains noise, is incomplete, or has sampling differences
- Helps ensure fair per-vertex distance comparisons with `R`
- Required for high-precision benchmarks or visualizations

---

## Distance Computation

Once the meshes are aligned and corrected, FaceBench computes the **geometric error** between them.

You can choose between three distance types, depending on the application:

- **Landmark-based** distances (between sparse keypoints)
- **Point-to-point** distances (between corresponding vertices)
- **Point-to-triangle** distances (from vertex to local surface region)

---

### Landmark Distance

Use this when you're interested only in error at landmark locations.

```python
errors = fb.landmark_distance(Rlmks, Glmks)
```

This computes per-landmark **Euclidean distance** between predicted and ground-truth landmarks.

✅ Output: one scalar distance for each landmark
✅ Ideal for comparison with other face fitting papers

---

### Point-to-Point Distance

This is the most direct mesh-to-mesh comparison metric.

```python
errors = fb.p2p_distance(R_aligned, G_corrected, corr)
```

- `R_aligned`: aligned reconstructed mesh
- `G_corrected`: corrected ground-truth mesh
- `corr`: correspondence indices (from `chamfer_correspondence` or similar)

✅ Output: one scalar error for each vertex in `R_aligned`
✅ Useful for evaluating global reconstruction quality

---

### Point-to-Triangle Distance

Instead of comparing each point to a single vertex, this method compares it to the **local surface triangle**.

```python
errors = fb.p2tri_distance(R_aligned, G_corrected, corr)
```

- For each point in `R_aligned`, it constructs a triangle around the corresponding point in `G_corrected`
- Computes the shortest distance to that triangle surface

✅ Output: smoother and more surface-aware error
✅ Recommended when reconstructions have local vertex misalignments

---
