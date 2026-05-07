# Data Format

To run FaceBench, you need to provide a minimal set of files:

---

## Required Inputs

- `R.txt`: Reconstructed mesh (as `.txt`, shape `(N, 3)`)
- `G.txt`: Ground-truth mesh (as `.txt`, shape `(M, 3)`)
- `G.lmks`: Ground-truth landmarks (as `.lmks`, shape `(L, 3)`)
- `BFM-p23470.json`: (optional) Morphable model info containing landmark indices used to extract `Rlmks`

---

## Mesh Format

Mesh files must be plain `.txt` files, where each line represents a vertex (x, y, z).
No header, no metadata — just three floats per line, separated by whitespace.

Example:

```
-5.522209391879443865e+04 -4.235516904937755316e+04 8.888917400567166624e+03
-5.517882928701071069e+04 -4.204418163471164735e+04 8.849323899468261516e+03
-5.512091975756881584e+04 -4.173335823981046997e+04 8.822460391894433997e+03
...
```

!!! note "Scale"
    Pay **attention** to scale. If your file is in millimeters or meters divide by the right scale.

---

## Landmark Format

Landmarks are stored similarly to meshes: one `(x, y, z)` coordinate per line.
These are used for alignment and evaluation.

Example:

```
-1.234e+02  5.678e+01  4.321e+01
...
```

🧠 The number of landmarks (`L`) is usually 51 for facial landmarks, but can vary.

---

## Landmark Indices

To extract the corresponding landmark coordinates from your mesh `R`, you need to know which **indices** they refer to.

This information is typically stored in a `.json` file like:

```json
{
  "lmk_indices": [19106, 19413, 19656, 19814, 19981, ..., 7508],
  "leye_oc_rel_index": 28,
  "reye_oc_rel_index": 19
}
```

You can then load and use it like:

```python
with open("BFM-p23470.json") as f:
    mm = json.load(f)

Rlmks = R[mm["lmk_indices"]]
```

---

### Custom formats are supported

You're **not required** to use this exact structure.
You just need to extract `Rlmks` such that:

- Each point in `Rlmks[i]` **corresponds** semantically to `Glmks[i]`
- The values **exist** in the mesh (i.e., valid indices)

✅ Valid alternative:

```python
Rlmks = R[[13, 19, 28, 31, 37]]
```

---

## Example Folder Structure

FaceBench doesn't enforce any strict folder layout, but here's a common one:

```
project/
├── Gmeshes/
│   ├── id0000.txt         ← G
│   ├── id0000.lmks        ← Glmks
├── Rmeshes/
│   └── Deep3DFace-m/
│       └── id0000.txt     ← R
├── info/
│   └── BFM-p23470.json    ← mm info
```

Just make sure to pass the correct paths to `np.loadtxt()` and `json.load()` — FaceBench does not assume or scan folders by default.

---

## Download Sample Files

You can download the files below to test FaceBench on a minimal example:

| File              | Description                       | Download Link                           |
|-------------------|-----------------------------------|-----------------------------------------|
| `id0000.txt`      | Reconstructed mesh (R)            | [Download](assets/data/id0000.txt)      |
| `id0000_gt.txt`   | Ground-truth mesh (G)             | [Download](assets/data/id0000_gt.txt)   |
| `id0000.lmks`     | Ground-truth landmarks (Glmks)    | [Download](assets/data/id0000.lmks)     |
| `BFM-p23470.json` | Morphable model landmark metadata | [Download](assets/data/BFM-p23470.json) |

📁 Place them all in the same directory, for example:

```
example_data/
├── id0000.txt
├── id0000_gt.txt
├── id0000.lmks
└── BFM-p23470.json
```

---

## Full Example Using Sample Data

```python
import facebench as fb
import numpy as np
import json

# === Load data
R = np.loadtxt("example_data/id0000.txt")
G = np.loadtxt("example_data/id0000_gt.txt") / 1e6
Glmks = np.loadtxt("example_data/id0000.lmks") / 1e6

with open("example_data/BFM-p23470.json") as f:
    mm = json.load(f)

Rlmks = R[mm["lmk_indices"]]

# === Align, correct and evaluate
R_aligned, _ = fb.icp_align(R, G, prealign="landmark", source_lmks=Rlmks, target_lmks=Glmks)
R_elastic = fb.landmark_elastic_align(R_aligned, G, Glmks, lmk_indices=mm["lmk_indices"])
corr = fb.chamfer_correspondence(R_elastic, G)
G_corr = fb.topology_consistency_corrector(R_elastic, G, corr, mm)
errors = fb.p2p_distance(R_elastic, G_corr, corr)

# === View results
fb.generate_results_table(errors)
fb.plot_face_3d_error_interactive(R_elastic, errors)

```

---

📦 With these sample files, you can test all core functionalities of the FaceBench library in under 60 seconds.

Want to benchmark full datasets? Check the [Pipeline](pipeline.md) section for multi-method evaluation.
