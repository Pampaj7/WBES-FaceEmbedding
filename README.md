# WBES Face Embedding

<p align="center">
  <img alt="Research code" src="https://img.shields.io/badge/status-research_code-8a5a44">
  <img alt="Domain" src="https://img.shields.io/badge/domain-3D_face_analysis-1f6f8b">
  <img alt="Focus" src="https://img.shields.io/badge/focus-identity_aware_evaluation-cb6d51">
  <img alt="Backbone" src="https://img.shields.io/badge/backbone-DiffusionNet-2f855a">
</p>

<p align="center">
  Research code for identity-aware evaluation and representation learning on 3D face meshes.
</p>

<p align="center">
  <img src="WBES/plots/wbes_density_grid.png" alt="WBES density grid" width="84%">
</p>

## Overview

This repository studies a simple but important question:

> Can a 3D face reconstruction be geometrically accurate and still fail to preserve identity?

The codebase tackles that question from two complementary directions:

- `WBES`: a statistical evaluation pipeline that measures identity separability using within-subject and between-subject distances.
- `face_embedding`: DiffusionNet-based models that learn identity-sensitive embeddings directly from 3D meshes.

The repository also contains topology-robustness experiments, remeshing utilities, operator precomputation, and a vendored `BFM_to_FLAME` conversion utility.

## Visual Snapshot

<table>
  <tr>
    <td align="center">
      <img src="WBES/plots/wbse_vs_f_lineplot.png" alt="WBES vs F" width="100%">
      <br>
      <sub>WBES trends as more frames are averaged</sub>
    </td>
    <td align="center">
      <img src="WBES/utils/landmarks_on_mesh.png" alt="Landmarks on mesh" width="100%">
      <br>
      <sub>Landmark-based topology alignment support</sub>
    </td>
  </tr>
</table>

## What Is In This Repo

### 1. WBES: identity-aware evaluation

WBES stands for Within- and Between-subject Effect Size.

The idea is:

- low within-subject variability is good
- high between-subject variability is good
- the larger the gap, the better identity is preserved

This branch contains:

- mesh-level WBES
- landmark-level WBES
- geometry-vs-identity correlation analyses
- plotting utilities for result interpretation

Main area:

- [`WBES/`](WBES)

### 2. 3D face embedding on meshes

This branch explores learned mesh representations using DiffusionNet-style encoders.

It includes:

- a baseline autoencoder
- encoder-only variants
- intrinsic descriptor variants based on HKS/WKS
- spectrum-aware encoders
- latent analysis and ranking scripts

Main area:

- [`face_embedding/gt_encdec/`](face_embedding/gt_encdec)

### 3. Topology robustness

The repository does not only ask whether identity is preserved across subjects.
It also asks whether identity structure survives:

- remeshing
- decimation
- cropping
- noisy perturbations
- topology changes

This is the role of the REMESH dataset generation scripts and the `intrinsic/robustness` package.

## Why This Repo Is Interesting

Many 3D face pipelines optimize geometry first and treat identity as a side effect.
This repository flips that perspective.

It is built around the idea that:

- geometry alone is not enough,
- identity should be measured explicitly,
- and learned representations should be tested under topology variation, not only clean canonical meshes.

That makes the project relevant if you work on:

- 3D face reconstruction
- identity-preserving generation
- geometric deep learning
- spectral mesh learning
- robust representation learning

## Repository Map

```text
WBES-FaceEmbedding/
├── WBES/
│   ├── code/                 # WBES pipelines, plots, correlation scripts
│   ├── utils/                # landmark indices, topology indices, helpers
│   ├── results/              # saved WBES outputs
│   ├── results_landmarks/    # landmark-only WBES outputs
│   └── plots/                # generated figures
├── datasets/
│   ├── GT_ready/             # aligned/canonical mesh assets
│   ├── REMESH/               # topology-variant dataset assets
│   ├── FaceVerse/            # FaceVerse-specific utilities
│   └── *.py                  # conversion, cropping, remeshing, preview tools
├── face_embedding/
│   └── gt_encdec/
│       ├── alignment/        # GT-ready mesh preparation
│       ├── autoencoder/      # baseline model zoo + latent analysis
│       ├── mse/              # pairwise geometric baselines
│       └── remeshing/        # cross-topology, intrinsic, voxel experiments
├── BFM_to_FLAME/             # bundled external conversion utility
├── CODEBASE_GUIDE.md         # detailed codebase documentation
└── tmp/                      # local exploratory artifacts
```

## Key Files

If you only read a handful of files, start here:

- [`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md)
- [`WBES/utils/WBES_helper.py`](WBES/utils/WBES_helper.py)
- [`WBES/code/WBES_pipeline.py`](WBES/code/WBES_pipeline.py)
- [`datasets/render_mesh_preview.py`](datasets/render_mesh_preview.py)
- [`datasets/expand_remesh_topologies.py`](datasets/expand_remesh_topologies.py)
- [`face_embedding/gt_encdec/autoencoder/dataset_gtready.py`](face_embedding/gt_encdec/autoencoder/dataset_gtready.py)
- [`face_embedding/gt_encdec/autoencoder/diffusion_autoencoder.py`](face_embedding/gt_encdec/autoencoder/diffusion_autoencoder.py)
- [`face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py`](face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py)
- [`face_embedding/gt_encdec/remeshing/intrinsic/train_twotower_dn_spec_robust.py`](face_embedding/gt_encdec/remeshing/intrinsic/train_twotower_dn_spec_robust.py)
- [`face_embedding/gt_encdec/remeshing/intrinsic/robustness/train_runner.py`](face_embedding/gt_encdec/remeshing/intrinsic/robustness/train_runner.py)

## Current Code Reality

This is research code, not a packaged library.

That means:

- some scripts are stable, others are exploratory
- data and experiment outputs often live next to source code
- several scripts assume local absolute paths
- large datasets are intentionally not versioned in git
- there is no single root-level environment file yet

The repository is still very useful, but it should be approached as an active research workspace.

## Results Already Present In The Repo

The current checkout already contains:

- stored WBES CSVs and plots
- landmark WBES outputs
- geometry-vs-WBES correlation summaries
- baseline autoencoder evaluation artifacts
- saved intrinsic robustness runs and logs

Some concrete examples extracted from local artifacts:

- mesh-level WBES improves strongly from `F=1` to larger frame groups for methods such as `3DDFAV3`, `Deep3DFace`, `FaceVerse`, `Smirk`, and `SynergyNet`
- geometric error changes are much smaller than WBES changes, which supports the core thesis of the project
- the stored baseline autoencoder evaluation reports very high rank preservation and no obvious mean-face collapse
- a saved intrinsic `xyz_dn` robustness run stays very stable across the tested perturbation grid

For the detailed artifact-backed summary, see:

- [`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md)

## Quick Start

There is no single command that runs the whole repository.
Instead, the practical workflow is:

1. Prepare aligned GT-ready meshes
2. Convert meshes to NPZ
3. Precompute DiffusionNet operators
4. Run either:
   - WBES evaluation
   - baseline embedding training
   - topology robustness experiments

Good entry points:

- mesh preview and QA:
  - [`datasets/render_mesh_preview.py`](datasets/render_mesh_preview.py)
- topology generation:
  - [`datasets/remesh.py`](datasets/remesh.py)
  - [`datasets/expand_remesh_topologies.py`](datasets/expand_remesh_topologies.py)
- operator precomputation:
  - [`face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py`](face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py)
- WBES:
  - [`WBES/code/WBES_pipeline.py`](WBES/code/WBES_pipeline.py)
  - [`WBES/code/WBES_pipeline_landmarks.py`](WBES/code/WBES_pipeline_landmarks.py)
- robustness training:
  - [`face_embedding/gt_encdec/remeshing/intrinsic/robustness/train_runner.py`](face_embedding/gt_encdec/remeshing/intrinsic/robustness/train_runner.py)

## Documentation

If you want the full technical walkthrough, read:

- [`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md)

That guide includes:

- actual dataset/output counts from the current checkout
- file-format conventions
- directory-by-directory explanations
- stored-results snapshots
- code-level notes for the most important modules
- a deep dive on:
  - [`diffusion_autoencoder.py`](face_embedding/gt_encdec/autoencoder/diffusion_autoencoder.py)
  - [`train_twotower_dn_spec_robust.py`](face_embedding/gt_encdec/remeshing/intrinsic/train_twotower_dn_spec_robust.py)

## External Context

Part of the conceptual background for this repo comes from FaceBench, an internal end-to-end evaluation pipeline developed during work at CHOP.

Important note:

- FaceBench itself is not included here
- this repository contains the WBES and mesh-embedding research code built around that broader evaluation perspective

## Author

Leonardo Pampaloni  
AI Engineer and Researcher  
University of Florence (MICC)  
Former AI Research Intern at CHOP

## Citation

If you use ideas, code, or results from this repository in academic work, cite the relevant thesis/paper if available, or contact the author for the correct reference.
