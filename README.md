# WBES Face Embedding

**WBES Face Embedding** is a research-oriented framework for **identity-aware evaluation and representation of 3D face reconstructions**.
The project combines **geometric evaluation**, **statistical identity metrics**, and **learning-based face embeddings** to study how identity information emerges from single- and multi-frame 3D face data.

This repository currently contains:

* a **validated application of the WBES metric** (used in a Master’s thesis),
* an **ongoing Face Embedding pipeline** based on geometric deep learning.

> ⚠️ **Status**: under active development.
> Some components are research prototypes and subject to change.

---

## FaceBench — Full Evaluation Pipeline (Private, CHOP Internal)

In addition to the WBES and Face Embedding work, this project builds on **FaceBench**: a complete, end-to-end **3D face reconstruction evaluation pipeline** developed during my work at **CHOP (Children’s Hospital of Philadelphia)**.

> 🔒 **Note**: FaceBench is currently **private/proprietary** and used internally at CHOP. This repository references its concepts and outputs, but does not include the full codebase.

### Why FaceBench exists

Existing evaluation scripts and toolboxes often suffer from recurring problems:

* tight coupling to specific data formats or frameworks,
* hidden dependencies / cache artifacts ("it works on my machine"),
* manual and fragile parallelization for large-scale runs,
* inconsistent metric parameterization across experiments.

FaceBench was designed to fix this by prioritizing:

* **Modularity**: swappable components (cropping, alignment, correspondence, metrics, correction).
* **Transparency**: no implicit state or opaque caching.
* **Reproducibility**: explicit, versionable configurations.
* **Scalability**: automated parallel execution across subjects with available hardware.

### Architectural highlights

* **Pure-function components** (stateless functions; easy to test and compose)
* **Dataclass-based configuration** (typed configs, explicit parameters)
* **Automatic parallel execution** (subject-level evaluation without user-written multiprocessing)
* **Extensibility** (new modules can be plugged in without rewriting the core)

### What FaceBench evaluates

A typical evaluation pipeline can include (optionally):

1. **Cropping** (e.g., inter-ocular normalized point-based crop)
2. **Rigid alignment** (landmark-based Procrustes or ICP, with optional pre-alignment)
3. **Non-rigid warping** (elastic landmark deformation or non-rigid ICP)
4. **Correspondence** (e.g., Chamfer NN or identity mapping when topology matches)
5. **Distance computation** (P2P, P2Tri, landmark error, etc.)
6. **Optional topology-aware correction**
7. **Visualization & reporting** (error maps, interactive 3D views, result tables)

FaceBench is the evaluation backbone used to generate the geometric signals that WBES and the embedding work build upon.

---

## Motivation

Evaluating 3D face reconstruction methods is traditionally done using **purely geometric metrics** (e.g. Chamfer distance).
However, low geometric error does **not guarantee correct identity preservation**.

This project addresses a core question:

> *Can we quantify how well a reconstruction preserves subject identity, independently of pure geometry?*

To answer this, we introduce **WBES (Within- and Between-subject Effect Size)** and extend it towards **learned face embeddings** operating directly on 3D meshes.

---

## Part I — WBES: Identity-Aware Evaluation (Stable)

### What is WBES?

**WBES (Within- and Between-subject Effect Size)** is a statistical metric designed to measure **identity separability** in 3D face data.

It compares:

* **within-subject variability** (same identity, different frames),
* **between-subject variability** (different identities),

and quantifies how well identities are separated as the number of frames increases.

Intuitively:

* good reconstructions → **low within**, **high between**
* poor identity preservation → overlap between the two

---

### WBES Pipeline (PCD Application)

The WBES pipeline operates on **per-subject averaged meshes**, computed from multiple reconstructed frames.

High-level steps:

1. Multi-frame 3D reconstructions per subject
2. Mesh alignment, cropping, and normalization
3. Per-subject mesh averaging
4. Pairwise distance computation
5. WBES evaluation and analysis

This pipeline has been **fully implemented, tested, and used** in the author’s Master’s thesis.

---

### Key Properties

* Works **without ground-truth identity labels**
* Scales with the number of frames
* Independent of reconstruction method
* Compatible with different mesh topologies (with proper alignment)

---

### Results (Summary)

WBES experiments show that:

* identity separability **improves with more frames**, even without ground truth,
* geometric error alone is insufficient to explain identity preservation,
* some methods with low Chamfer error exhibit **poor identity consistency**.

These results motivate the second part of the project.

---

## Part II — Face Embedding on 3D Meshes (Ongoing)

### Goal

Move from *identity evaluation* to **identity representation**.

The objective is to learn a **compact embedding space** where:

* samples of the same subject cluster together,
* different identities are well separated,
* the structure reflects underlying geometric morphology.

---

### Approach

The current approach uses **DiffusionNet-based autoencoders** operating directly on meshes:

* **Encoder**
  Produces:

  * per-vertex latent fields (local geometry),
  * global latent vectors (identity descriptors).

* **Decoder**
  Reconstructs the original mesh geometry.

* **Losses**
  Combine:

  * geometric reconstruction losses,
  * latent regularization losses (e.g. stress, scale, structure preservation).

This setup allows disentangling:

* *geometric fidelity* vs
* *identity structure in latent space*.

---

### Current Research Questions

* How much identity structure emerges **without supervision**?
* What is the trade-off between perfect reconstruction and meaningful embeddings?
* How stable are identity embeddings across different topologies (BFM / FLAME)?
* Can latent distances correlate with WBES and geometric metrics?

These questions are **actively investigated** in this repository.

---

## Repository Structure (High-Level)

```
wbes-face-embedding/
│
├── wbes_pipeline/        # WBES evaluation pipeline (stable)
├── face_embedding/       # DiffusionNet-based embedding models (research)
├── latent_analysis/      # Correlation & embedding analysis
├── visualization/       # 3D and statistical visualizations
├── datasets/             # (ignored) large mesh datasets
└── results/              # experiment outputs
```

Large datasets and trained models are intentionally **not included**.

---

## Status & Disclaimer

* WBES pipeline: **stable and validated**
* Face Embedding models: **experimental**
* Code is research-driven, not packaged as a production library
* APIs may change as experiments evolve

---

## Author

**Leonardo Pampaloni**
AI Engineer & Researcher
University of Florence (MICC)
Former AI Research Intern @ CHOP (USA)

Research interests:

* 3D face reconstruction
* geometric deep learning
* identity-aware metrics
* Green AI & efficiency

---

## Citation

If you use ideas or parts of this project, please cite the corresponding thesis or contact the author.
