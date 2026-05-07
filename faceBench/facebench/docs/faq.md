
This page addresses common issues, usage questions, and deep implementation details about FaceBench.

---

## 🧩 General Usage


### How do I evaluate multiple reconstruction methods?

Use `run_pipeline_batch()` with a list of method subfolders:

??? example
    ```python
    from facebench import run_pipeline_batch
    errors, vertices, ids, methods = run_pipeline_batch(
        rec_methods=["m1", "m2"],
        g_path="Gmeshes",
        g_lmks_path="Glandmarks",
        base_r_path="Rmeshes",
        config=cfg,
        mm_data=mm
    )
    ```

---

## ⚙️ Configuration

### How do I disable non-rigid alignment?

Set:

??? example
    ```python
    config.nonrigid_aligner = None
    ```

Same logic applies to `mesh_cropper`, `corrector`, etc.

---

### Can I combine ICP and Elastic alignment?

Yes. Use ICP as a rigid pre-aligner and Elastic as a non-rigid aligner.

??? info
    Use `rigid_aligner.type = "icp"` and `nonrigid_aligner.type = "elastic"` in your config.

---

## 🧠 Deep Questions

### Why do you use `Rref = R` even after ICP?

Because ICP modifies `R` in-place (rigid). Non-rigid aligners (Elastic, NICP) are applied on a copy (`R.copy()`) to preserve reference integrity.

---

### Why separate `error` and `vertices` in the output?

This is intentional. Keeping `errors_stacked` and `vertices_stacked` as **parallel tensors** allows:

- Efficient slicing for plotting
- Fast aggregation across subjects/methods
- Compatibility with tools like NumPy, Plotly, Vedo

---

### How is parallelism handled?

Subject-level parallelism is handled via `multiprocessing.Pool`, not threading.

- Fully CPU-bound
- Max core count can be set via `max_cores=...`

---

### Why use per-subject batching instead of method batching?

Because subjects are the **unit of independence**:
- Each subject can be processed in isolation
- Method comparison happens post-evaluation

This allows maximal parallelization and avoids shared state.

---

### Why are some errors zero or NaN?

Possible reasons:

- R and G are identical (e.g., identity pipeline)
- Meshes are misaligned due to bad config
- Some points fall outside valid correspondence range

---

## 📂 Files & Structure

### What filenames does FaceBench expect?

- Reconstructed meshes: `idXXXX.txt` (e.g. `id0000.txt`)
- Landmarks: `idXXXX.lmks`
- One subfolder per method inside `Rmeshes/`

---

### What file formats are supported?

Currently only `.txt` (ASCII) with shape `(N, 3)`. For `.ply` or `.obj`, convert externally.

---

### Can I use synthetic meshes?

Yes — FaceBench is mesh-format agnostic. Just ensure landmark indices and alignment assumptions are coherent.

---

## 🧪 Evaluation Logic

### What is the difference between `p2p` and `p2tri` distances?

| Metric  | Description                          | Accuracy | Cost |
|---------|--------------------------------------|----------|------|
| `p2p`   | Vertex-to-vertex Euclidean error     | Fast     | Low  |
| `p2tri` | Vertex-to-triangle projection error  | More accurate | Higher |

Use `p2tri` if you expect large topological mismatches or sparse reconstructions.

---


### How are landmark weights computed in the corrector?

The function `compute_landmark_base_vertex_weights` uses:

1. Distance to key landmarks
2. Deviation from average landmark distances
3. Combination via `mean`, `min`, or `mixed` strategies
4. Optional transformation (`sqrt`, `square`)

---

## 🧩 Contributing

### Can I plug in my own aligner or distance metric?

Absolutely.

- Write a function with the correct signature
- Register it in `config.py` or pass it directly in a config object

---

Still stuck? Open an [issue on GitHub](https://github.com/Pampaj7/facebench/issues).
