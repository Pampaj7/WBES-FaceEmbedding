🧩 Dataset Module — Overview

This folder contains all components related to data loading, mesh preprocessing, and operator caching for DiffusionNet-based models.

📁 Files

mesh_dataset.py
Scans directories containing .obj meshes and extracts subject_id values from filenames.
Each mesh is associated with a subject to enable supervised pairing across topologies.

operators_cache.py
Computes or loads precomputed DiffusionNet geometric operators
(mass matrix, Laplacian, eigenvalues/eigenvectors, gradient operators).
Stores them as .npz files inside a cache directory to avoid recomputation.

sampler.py
Creates paired samples between two topologies (e.g., BFM ↔ FLAME)
for the same subject. Also supports negative sampling for triplet loss training.

🔄 Data Flow
Mesh (.obj)
   ↓
normalize_positions()
   ↓
compute_operators()
   ↓
Saved as cache (.npz)
   ↓
Loaded by train_cross_topo.py
   ↓
Fed into DiffusionBackbone

🧠 Key Concepts

Each mesh is linked to a subject ID extracted from its filename
(e.g., id0001_neutral.obj → subject_id = id0001).

The same subject can appear in two different topologies (e.g., BFM and FLAME).
These meshes are treated as positive pairs during training.

Operator caching ensures:

⚡ Speed: no repeated computation across epochs

🔁 Reproducibility: deterministic operator reuse

📦 Portability: cache can be reused across experiments and systems

⚙️ Typical Usage
from dataset.operators_cache import OperatorCache

# Initialize the cache directory
cache = OperatorCache("./cache_ops")

# Load precomputed (or compute and save) operators
(mass, L, evals, evecs, gradX, gradY), verts, faces = cache.get_with_geo(
    topo_name="BFM",
    mesh_path="/path/to/mesh.obj",
    k_eig=128
)

🧩 Integration Summary
MeshDataset → scans subjects and mesh paths  
OperatorCache → loads or computes geometric operators  
Sampler → builds (BFM, FLAME) subject pairs  
Train script → feeds both meshes to the shared DiffusionBackbone  