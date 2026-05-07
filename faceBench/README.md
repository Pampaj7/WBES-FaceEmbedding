# 🎭 FaceBench

A modular and extensible framework for benchmarking **3D face reconstruction** methods.
FaceBench provides a clean, fully-documented API and supports rich visualizations, flexible evaluation pipelines, and reproducible comparisons.

![pipeline](info/framework.jpg)

---

## 🚀 Key Features

- ✅ **Modular pipeline**: Mesh cropping, alignment, correspondence, correction, and distance — all pluggable
- 🧱 **Configurable** with typed Python dataclasses — no more fragile JSONs
- ⚡ **Parallelized**: Fast evaluation across subjects with `multiprocessing`
- 📈 **Visualization-ready**: Heatmaps, 3D plots, interactive inspection
- 📊 **Flexible Output**: Vertex-level errors for each subject and method

---

## 📖 Documentation

Full documentation is available at:
📚 **https://your-org.github.io/facebench**

Includes:
- Quickstart guide
- Pipeline configuration
- Visualization utilities
- Data format specification
- API reference

---

## 🧪 Minimal Example

```python
import facebench as fb
import numpy as np
import json

# Load input data
R = np.loadtxt("id0000.txt")
G = np.loadtxt("id0000_gt.txt")
Glmks = np.loadtxt("id0000.lmks")

with open("BFM-p23470.json") as f:
    mm = json.load(f)

Rlmks = R[mm["lmk_indices"]]

# Run basic evaluation
R_aligned, _ = fb.icp_align(R, G, prealign="landmark", source_lmks=Rlmks, target_lmks=Glmks)
corr = fb.chamfer_correspondence(R_aligned, G)
G_corr = fb.topology_consistency_corrector(R_aligned, G, corr, mm)
errors = fb.p2tri_distance(R_aligned, G_corr, corr)

fb.plot_face_3d_error_interactive(R_aligned, errors)
fb.generate_results_table(errors)
```

---

## 🧠 Pipeline Example

Evaluate multiple methods in one line using `run_pipeline_batch`:

```python
errors, aligned_vertices, subject_ids, methods = fb.run_pipeline_batch(
    rec_methods=["Deep3DFace-m", "3DDFAv2-m"],
    g_path="data/Gmeshes",
    g_lmks_path="data/Gmeshes",
    base_r_path="data/Rmeshes/BFM/p23470/",
    config=your_config,
    mm_data=your_mm_json
)
```

See full examples in [📦 pipeline](https://your-org.github.io/facebench/pipeline)

---

## 📁 Dataset Format

FaceBench assumes `.txt` meshes and `.lmks` landmarks.
Filenames must follow this pattern:

```
Gmeshes/
├── id0000.txt     ← GT mesh
├── id0000.lmks    ← GT landmarks

Rmeshes/
├── Deep3DFace-m/
│   └── id0000.txt
```

More info in [📁 Data Format](https://your-org.github.io/facebench/data_format)

---

## 📦 Install

```bash
pip install -r requirements.txt
```

Or with Conda:

```bash
conda create -n facebench python=3.8
conda activate facebench
pip install -r requirements.txt
```

---

## 🤝 Citing & Credits

This framework builds upon M3D-FB.
Refactored, documented, and redesigned by CHOP team for publication and reusability.

> 🔬 If you use this in your research, please cite the original paper and link to this repository.

---

## 🧠 License

MIT © CHOP Research Team, 2025