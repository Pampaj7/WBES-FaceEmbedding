# 🧠 What is FaceBench?

FaceBench is a **modular Python library** for evaluating 3D face reconstruction algorithms.

It provides a clean pipeline to align, compare, and visualize facial meshes — both on a single-subject scale and over entire datasets.

---

## 🔧 What does it do?

FaceBench evaluates reconstruction quality through these main stages:

1. ✂️ **Cropping**: focuses on the central face region
2. 🔄 **Rigid alignment**: aligns predicted mesh to ground-truth
3. 🛠 **Non-rigid alignment**: optional deformation for better matching
4. 🧭 **Correspondence**: finds point-wise pairs between meshes
5. 🛡 **Correction**: topology-aware adjustment of ground-truth
6. 📏 **Distance computation**: measures geometric errors
7. 📊 **Visualization & reporting**

Each step is fully customizable through structured configuration objects.

---

## 🧼 Why was it refactored?

The original codebase was a collection of experimental scripts.

We restructured it into a **fully reusable Python library**, with:

- ✅ Clean functional interface
- ✅ No legacy cache or state
- ✅ performance-optimized code
- ✅ Fully typed, testable modules
- ✅ Configurable with pipeline
- ✅ multiprocessing support for IDs
- ✅ Clear module separation (`aligners`, `distances`, `correctors`, etc.)
- ✅ Rich visualizations with **Plotly** and **Matplotlib**
- ✅ Modern documentation with **MkDocs + Material**

---

## 🧩 How is it organized?

FaceBench is divided into intuitive modules:

- `aligners/` → rigid & non-rigid alignment
- `distances/` → p2p, p2tri, landmark distances
- `correctors/` → local topology refinement
- `correspondences/` → chamfer/identity matching
- `mesh_croppers/` → distance-based cropping
- `visualization/` → plotting, 3D viewers, heatmaps
- `config.py` → dataclass configs for the pipeline
- `run_pipeline_batch()` → the full high-level evaluator

You can run everything with just a few lines of code, or build your own custom evaluation pipeline step-by-step.

---

## 📊 Example Workflow

```python
import facebench as fb

# Load meshes and landmarks
R = ...
G = ...
Rlmks = ...
Glmks = ...

# Align
R_aligned, _ = fb.icp_align(R, G, prealign="landmark", source_lmks=Rlmks, target_lmks=Glmks)

# Correct and compare
corr = fb.chamfer_correspondence(R_aligned, G)
Gcorr = fb.topology_consistency_corrector(R_aligned, G, corr, mm)
errors = fb.p2tri_distance(R_aligned, Gcorr, corr)

# Visualize
fb.plot_face_3d_error_interactive(R_aligned, errors)
```

---

## 📚 Start Exploring

👉 Head over to the [Quickstart guide](quickstart.md) to run your first experiment.
📁 Or check out the [Data Format](data_format.md) section to prepare your dataset.