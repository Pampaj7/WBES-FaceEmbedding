# Full Pipeline

FaceBench provides a high-level utility to evaluate **multiple reconstruction methods** over a dataset of subjects,
in a single function call. This is done via:

```python
fb.run_pipeline_batch()
```

It performs all the required steps internally: cropping, alignment, correspondence, correction, and distance computation — fully parallelized across subjects.

---

## Required Files

To use `run_pipeline_batch`, FaceBench expects **organized folders** containing:

- ✅ Reconstructed meshes (one folder per method)
- ✅ Ground-truth meshes
- ✅ Ground-truth landmarks
- ✅ Morphable model metadata (e.g., `BFM-p23470.json`)

Each **reconstructed mesh** must correspond to a ground-truth mesh and landmark file with the **same subject ID** (e.g., `id0000.txt`).

---

## Example Structure

```
    your_project/
    ├── data/
    │   └── BFMsynth/
    │       ├── Gmeshes/
    │       │   ├── id0000.txt       ← Ground-truth mesh
    │       │   ├── id0000.lmks      ← Ground-truth landmarks
    │       │   ├── id0001.txt
    │       │   └── ...
    │       └── Rmeshes/
    │           ├── Deep3DFace-m/
    │           │   ├── id0000.txt   ← Reconstructed mesh (method 1)
    │           │   └── ...
    │           ├── 3DDFAv2-m/
    │           ├── 3DIv2-m/
    │           └── INORig-m/
    ├── info/
    │   └── BFM-p23470.json          ← Morphable model info (contains `lmk_indices`)
```

---

## File Naming Convention

FaceBench expects **consistent IDs** across all methods:

```text
id0000.txt      → reconstructed mesh
id0000.txt      → ground-truth mesh
id0000.lmks     → ground-truth landmarks
```

Each ID is used to match reconstructed and ground-truth data **across all folders**.

> 📌 The internal pipeline will automatically parse these filenames and pair the files correctly.

---

## What's Customizable?

✅ You can freely change:
- Folder names
- Root paths
- Number of methods
- Number of subjects

✅ Just make sure:
- File names follow the `idXXXX.txt` / `idXXXX.lmks` convention
- You pass the correct paths to:

```python
fb.run_pipeline_batch(
    g_path=...,
    g_lmks_path=...,
    base_r_path=...,
    ...
)
```

---

## Load Your Data

Before calling the pipeline, it's your responsibility to prepare and load:

- A list of reconstruction method names (used as folder names)
- Paths to:
  - Ground-truth meshes
  - Ground-truth landmarks
  - Morphable model metadata file (JSON)

Example setup:

```python
import json

# Load Morphable Model metadata
with open("../info/BFM-p23470.json") as f:
    mm_data = json.load(f)

# Landmark indices used to extract Rlmks
mm_data["lmk_indices"] = mm_data.get("lmk_indices", [])

# List of methods to evaluate (each should match a subfolder in Rmeshes/)
rec_methods = ["Deep3DFace-m", "3DDFAv2-m", "3DIv2-m", "INORig-m"]

# Folder containing GT meshes and GT landmarks
g_path = "../data/BFMsynth/Gmeshes"
g_lmks_path = g_path  # usually same as g_path
```

---

✅ This setup must be done manually before running the pipeline.
✅ `mm_data["lmk_indices"]` is used to extract the correct landmark points from reconstructed meshes (`R`).

> 💡 FaceBench does **not assume** where the files are — you have full control over the paths.
> You simply need to pass the correct arguments to `run_pipeline_batch`.

---

## Configure the Pipeline

FaceBench uses Python **dataclasses** to configure each step of the pipeline in a clean, safe, and modular way.

Instead of relying on fragile strings like `"p2p"` or `"landmark"`, we use explicit structured configs such as:

```python
fb.DistanceComputerConfig(type=fb.DistanceComputerType.P2TRI)
```

This makes your code easier to read, validate, and debug — and prevents silent misconfigurations.

---

### PipelineConfig

All components are wrapped inside a single master configuration:

```python
@dataclass
class PipelineConfig:
    mesh_cropper: Optional[MeshCropperConfig] = None
    rigid_aligner: Optional[RigidAlignerConfig] = None
    nonrigid_aligner: Optional[NonRigidAlignerConfig] = None
    corr_establisher: Optional[CorrEstablisherConfig] = None
    corrector: Optional[CorrectorConfig] = None
    distance_computer: Optional[DistanceComputerConfig] = None
```
Each step is optional — but **you are responsible** for building a valid and meaningful configuration.

---

### Example

Here is a complete example used in this tutorial:

```python
config = fb.PipelineConfig(
    mesh_cropper=fb.MeshCropperConfig(
        method=fb.MeshCropperType.POINT_BASED
    ),
    rigid_aligner=fb.RigidAlignerConfig(
        type=fb.RigidAlignerType.ICP,
        prealign=fb.PrealignMethod.LANDMARK,
    ),
    nonrigid_aligner=fb.NonRigidAlignerConfig(
        type=fb.NonRigidAlignerType.ELASTIC,
        ref_lmk_indices=mm_data["lmk_indices"]
    ),
    corr_establisher=fb.CorrEstablisherConfig(
        type=fb.CorrEstablisherType.CHAMFER
    ),
    corrector=fb.CorrectorConfig(),
    distance_computer=fb.DistanceComputerConfig(
        type=fb.DistanceComputerType.P2TRI
    )
)
```

!!!note
    for more details on each step, see the [Configuration Reference](configuration.md) section.


---
## Run the Pipeline

Once your configuration is ready, you can evaluate all reconstruction methods **in one call** using:

```python
errors, aligned_vertices, subject_ids, methods = fb.run_pipeline_batch(
    rec_methods=rec_methods,
    g_path=g_path,
    g_lmks_path=g_lmks_path,
    config=config,
    mm_data=mm_data,
    base_r_path="../data/BFMsynth/Rmeshes/BFM/p23470/"
)
```

---

### Inputs

- `rec_methods`: List of reconstruction method names (folder names under `base_r_path`)
- `g_path`: Path to ground-truth meshes (`.txt` files like `id0000.txt`)
- `g_lmks_path`: Path to ground-truth landmarks (`.lmks` files like `id0000.lmks`)
- `config`: A `PipelineConfig` instance specifying the pipeline steps
- `mm_data`: Morphable model info dictionary (must contain `"lmk_indices"`)
- `base_r_path`: Path to the folder containing subfolders for each method (e.g. `Rmeshes/MethodName/id0000.txt`)

---

### Output Structure

The `run_pipeline_batch()` function returns structured NumPy arrays for maximum flexibility.

You get:

```python
errors.shape             == (num_subjects, num_methods, num_vertices)
aligned_vertices.shape   == (num_subjects, num_methods, num_vertices, 3)
```

---

Each output is indexed as:

- `s` → subject index
- `m` → reconstruction method index
- `v` → vertex index

This means that:

- `errors[s, m, v]` is the **error** for vertex `v` of subject `s` using method `m`
- `aligned_vertices[s, m, v, :]` gives the **3D coordinate** of that vertex after alignment

---

### Why This Format?

FaceBench uses **parallel arrays** instead of fusing error and coordinates into a single object:

- 💡 More efficient for NumPy slicing and broadcasting
- 🎯 Works seamlessly with visualization tools like **Plotly**, **Vedo**, or **Open3D**
- 📈 Lets you compute per-method or per-subject stats easily (e.g., mean error, heatmaps, etc.)

---

### Example Usage

You can extract results like this:

```python
# Errors for subject 0, method 1
errors_subject0_method1 = errors[0, 1]  # shape: (num_vertices,)

# Aligned mesh for the same
mesh_coords = aligned_vertices[0, 1]    # shape: (num_vertices, 3)

# Print each vertex and its error
for err, pt in zip(errors_subject0_method1, mesh_coords):
    print(f"Vertex {pt} → error: {err}")
```

---

### Now You Can...

Once you have these arrays, you can:

- Plot errors as heatmaps or 3D color maps
- Compute per-region statistics (e.g., eyes, mouth)
- Compare multiple methods on the same subject
- Export data for external tools or benchmarks

---

### Behind the Scenes

This function:

- ✅ Loads the correct `.txt` and `.lmks` files for each subject
- ✅ Applies all pipeline steps in the right order
- ✅ Runs everything **in parallel** using `multiprocessing`
- ✅ Returns a clean structure ready for analysis or visualization

---

## Full Example

Here is a complete working example that:

- Loads the Morphable Model metadata
- Defines a list of reconstruction methods
- Configures the FaceBench pipeline
- Runs evaluation across all subjects and methods
- Generates a visualization and summary table


```python
import facebench as fb
import time
import json

if __name__ == "__main__":

    # Load Morphable Model Info
    with open("../info/BFM-p23470.json") as f:
        mm_data = json.load(f)
    mm_data["lmk_indices"] = mm_data.get("lmk_indices", [])

    # Define reconstruction methods (must match folder names)
    rec_methods = ["Deep3DFace-m", "3DDFAv2-m", "3DIv2-m", "INORig-m"]

    # Define paths to GT meshes and landmarks
    g_path = "../data/BFMsynth/Gmeshes"
    g_lmks_path = g_path

    # Configure the pipeline
    config = fb.PipelineConfig(
        rigid_aligner=fb.RigidAlignerConfig(
            type=fb.RigidAlignerType.ICP,
            prealign=fb.PrealignMethod.LANDMARK,
        ),
        nonrigid_aligner=fb.NonRigidAlignerConfig(
            type=fb.NonRigidAlignerType.ELASTIC,
            ref_lmk_indices=mm_data["lmk_indices"]
        ),
        corr_establisher=fb.CorrEstablisherConfig(
            type=fb.CorrEstablisherType.CHAMFER
        ),
        corrector=fb.CorrectorConfig(),
        distance_computer=fb.DistanceComputerConfig(
            type=fb.DistanceComputerType.P2TRI
        )
    )

    # Run evaluation
    errors, aligned_vertices, subject_ids, methods = fb.run_pipeline_batch(
        rec_methods=rec_methods,
        g_path=g_path,
        g_lmks_path=g_lmks_path,
        config=config,
        mm_data=mm_data,
        base_r_path="../data/BFMsynth/Rmeshes/BFM/p23470/"
    )

    # Visualize and summarize results
    fb.plot_face_3d_error_interactive(aligned_vertices[1, 2], errors[1, 2])
    fb.generate_results_table(errors, method_names=rec_methods)
```

For per-subject metrics and visualizations, see:

📊 [📈 Visualizations](visualizations.md)

📄 [⚙️ Configuration Reference](configuration.md)
