## 2D Error Heatmap

This function visualizes **per-vertex errors** by projecting the 3D face onto a 2D plane and coloring it using a
heatmap.

✅ Ideal for publishing results and highlighting **error concentration areas** across the face.

---

### What it does

- Projects vertices to a 2D grid (XY plane)
- Interpolates missing points
- Optionally normalizes scale using inter-ocular distance (IOD)
- Masks out regions outside the convex hull (e.g. background)
- Renders the result using `matplotlib`

---

### Parameters

| Argument           | Type       | Description                                                                 |
|--------------------|------------|-----------------------------------------------------------------------------|
| `vertices`         | (N, 3)     | 3D points of the reconstructed mesh                                         |
| `errors`           | (N,)       | Per-vertex scalar error values                                              |
| `landmarks`        | (L, 3)     | Optional: 3D landmarks used for IOD normalization                           |
| `landmark_indices` | (int, int) | Optional: indices of left and right eye landmarks                           |
| `coef`             | int        | Grid resolution (default: 300)                                              |
| `vmax`             | float      | Max value for colormap scale (automatically set to 95th percentile if None) |
| `cmap`             | str        | Colormap name (default: `"jet"`)                                            |
| `title`            | str        | Optional title for the plot                                                 |
| `save_path`        | str        | If provided, the image will be saved instead of shown interactively         |

---

### Example Usage

=== "Basic (no landmarks)"

    ```python
    fb.plot_face_heatmap(
        vertices=R,
        errors=err
    )
    ```

=== "With normalization + title"

    ```python
    fb.plot_face_heatmap(
        vertices=R,
        errors=err,
        landmarks=Rlmks,
        landmark_indices=(36, 45),
        title="Heatmap Error on XY Plane"
    )
    ```

=== "Save to file"

    ```python
    fb.plot_face_heatmap(
        vertices=R,
        errors=err,
        save_path="error_map.png"
    )
    ```

---

!!! tip "Recommended setup"
    Always normalize using landmarks if available.
    Set `landmark_indices=(36, 45)` for IOD-based scaling.

---

### Expected Output

> You should see a 2D color heatmap of the reconstruction error projected on the face, like this:


![2D Heatmap](assets/error_heatmap.png)


---

📎 Used when you want to visualize:

- **Where** the reconstruction failed
- Differences between methods (use same scale `vmax`)
- Errors in a visually meaningful way (for papers, debugging, etc.)

---

# 3D Visualizations

FaceBench provides multiple 3D error visualizations for qualitative evaluation and presentation.
These allow you to explore the distribution of reconstruction errors in 3D space, with both static and interactive
options.

---

## 3D errors

Visualizes per-vertex errors directly on the 3D mesh using **matplotlib**. Changing `elev` and `azim` rotates the view.

### Usage

```python
fb.plot_face_3d_error(
    vertices=R_aligned,
    errors=errors,
    title="Reconstruction Error",
    cmap="jet",
    elev=30,
    azim=-75
)
```

### Parameters

| Parameter      | Description                                      |
|----------------|--------------------------------------------------|
| `vertices`     | 3D coordinates of the mesh                       |
| `errors`       | Per-vertex scalar error values (same length)     |
| `title`        | Optional plot title                              |
| `cmap`         | Colormap used (default: `"jet"`)                 |
| `elev`, `azim` | Viewpoint angles for static visualization        |
| `figsize`      | Output figure size                               |
| `vmax`         | Max colorbar value (auto set to 95th percentile) |
| `save_path`    | If set, saves the figure instead of displaying   |

### Output

Produces a **static 3D scatter plot** with color-coded error values.

![3D Heatmap](assets/error_3d.png)

---

## 3D interactive


Renders a fully **interactive 3D plot** using Plotly, allowing zoom/rotate and tooltips.

### Usage

```python
fb.plot_face_3d_error_interactive(
    vertices=R_aligned,
    errors=errors,
    title="3D Error Map (Interactive)"
)
```

### Parameters

| Parameter    | Description                                     |
|--------------|-------------------------------------------------|
| `vertices`   | Reconstructed mesh vertices                     |
| `errors`     | Per-vertex errors (same size as `vertices`)     |
| `title`      | Plot title                                      |
| `colormap`   | `"Jet"`, `"Viridis"`, etc.                      |
| `point_size` | Point size in 3D scatter                        |
| `vmax`       | Max error for color normalization               |
| `save_html`  | If set, saves output as HTML instead of showing |

### When to Use

- For interactive presentations or reports
- When exploring per-vertex errors manually

![3D Heatmap_inter](assets/error_3dinter.png)

???+ note "What it does"

    - Uses Plotly to create an interactive 3D view
    - Colors each vertex based on its error magnitude
    - Supports zoom, rotation, and hover tooltips

---

## 3D interactive + ground truth


Displays **both the reconstructed mesh and ground-truth mesh** in the same 3D scene.

- `R`: rendered with errors as color
- `G`: rendered in black, semi-transparent

### Usage

```python
fb.plot_point_clouds_with_error_interactive(
    R, G, errors,
    title="Reconstruction vs Ground Truth"
)
```

### Parameters

| Parameter   | Description                   |
|-------------|-------------------------------|
| `R`         | Reconstructed mesh (aligned)  |
| `G`         | Ground-truth mesh             |
| `errors`    | Per-vertex errors (for `R`)   |
| `title`     | Plot title                    |
| `colormap`  | `"Jet"`, `"Viridis"`, etc.    |
| `size`      | Point size                    |
| `vmax`      | Max value for normalization   |
| `save_path` | If set, saves to `.html` file |

### Use case

- Ideal for **side-by-side visual comparisons**
- Good for paper figures or supplementary videos

![3D Heatmap_inter](assets/error_3d_ground.png)

???+ note "What it shows"

    - Reconstructed mesh is colored by error magnitude
    - Ground-truth mesh is shown as a semi-transparent reference
    - Helps assess where and how much deviation occurs

---

# Results table

Generates clean, readable tables summarizing the reconstruction error statistics.

```python
fb.generate_results_table(
    errors,
    method_names=rec_methods  # Optional if using single method
)
```

???+ note "What it does"
    - If `errors` is a 3D array from `run_pipeline_batch`: computes per-subject and overall stats
    - If `errors` is a list or 1D array: computes a summary for a single mesh
    - Returns:
    - `summary_df`: global stats (mean, std, median, min, max)
    - `subject_df`: per-subject average error (if applicable)

---

## Example: Single Method

If you're evaluating just one reconstruction:

```python
fb.generate_results_table(errors)
```

**Output:**

```
📊 Summary Statistics Table:
         mean    std  median    min     max
method  1.839  1.243   1.547  0.036  11.205
```

---

### Example: From `run_pipeline_batch`

When evaluating multiple methods and subjects using the high-level pipeline:

```python
errors, aligned_vertices, subject_ids, methods = fb.run_pipeline_batch(
    rec_methods=rec_methods,
    g_path=g_path,
    g_lmks_path=g_lmks_path,
    config=config,
    mm_data=mm_data,
    base_r_path="../data/BFMsynth/Rmeshes/BFM/p23470/"
)

fb.generate_results_table(errors, method_names=rec_methods)
```

**Output:**

```
📋 Per-subject Mean Errors Table:
            Deep3DFace-m  3DDFAv2-m  3DIv2-m  INORig-m
subject_id
0                  1.099      1.077    1.144     0.997
1                  0.972      1.262    1.071     1.033
...

📊 Summary Statistics Table:
               mean    std  median    min    max
method
Deep3DFace-m  1.299  0.292   1.262  0.972  1.713
3DDFAv2-m     1.387  0.247   1.368  1.077  1.735
...
```

???+ note "Shape of `errors`"
    When returned by `run_pipeline_batch`, the `errors` array has shape:
    **(subjects, methods, vertices)** — enabling per-subject and per-method analysis.

---

## When to use

- After evaluating multiple subjects and methods with `run_pipeline_batch`
- To produce tables for reports or papers
- To get a quick overview of reconstruction performance

---
