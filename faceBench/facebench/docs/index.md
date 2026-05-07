# FaceBench

> **FaceBench** is a modular benchmarking framework for evaluating 3D face reconstruction methods.

It provides a clean and extensible pipeline for:

    - 🧩 Mesh alignment (rigid + non-rigid)
    - 📏 Dense error computation
    - 📊 Result visualization
    - 🧪 Multi-method evaluation over datasets

FaceBench is designed for **researchers, engineers, and developers** working on:

- 3D face reconstruction
- Morphable models
- Landmark fitting
- Facial geometry analysis



---

## Features

- ✅ Pure Python + NumPy + Open3D implementation
- ✅ Supports both single-subject and full-dataset evaluation
- ✅ Full pipeline modularity via dataclass configuration
- ✅ Built-in support for:
    - ICP / Procrustes rigid alignment
      - Non-rigid alignment (Elastic / NICP)
      - Landmark & dense distances
      - Topology-aware correction
- ✅ Visual diagnostics: 2D heatmaps, 3D errors, interactive viewers
- ✅ One-liner benchmarking across multiple methods

---

## Getting Started

- [⚡ Quickstart](quickstart.md): minimal working example
- [📦 Configuration](configuration.md): available options
- [📁 Data Format](data_format.md): input files and structure
- [🧪 Pipeline](pipeline.md): full multi-method benchmark
- [📊 Visualizations](visualizations.md): how to display results
- [🧠 API Reference](api/croppers.md): documentation by module

---

## Contributors

This library was developed at the **Children’s Hospital of Philadelphia** (CHOP)
Originally authored and refactored by:

- Leonardo Pampaloni `(@Pampaj7)`
- Evangelos Sariyanidi
- Claudio Ferrari
- Federico Nocentini

---

## Links

- 📂 [GitHub Repository](https://github.com/your-org/facebench)
- 📄 [License](https://github.com/your-org/facebench/blob/main/LICENSE)
