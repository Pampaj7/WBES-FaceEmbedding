# 🔄 Aligners

FaceBench supports both **rigid** and **non-rigid** alignment of 3D face meshes.

Before comparing reconstructed and ground-truth face meshes, it is essential to **bring them into alignment**—both
spatially and semantically. FaceBench provides robust tools for this task by supporting two main alignment strategies:

- ✅ **Rigid alignment** – Applies a global rigid transformation (rotation + translation, optionally scaling) to roughly
  align the meshes.
- 🧠 **Non-rigid alignment** – Refines the alignment by locally deforming the mesh to better match shape-specific
  variations (e.g., expressions, wrinkles).

> 📌 These alignment steps operate on the **reconstructed mesh (`R`)** and align it to the **ground-truth mesh (`G`)**,
> ensuring that any error measurement reflects **only shape reconstruction accuracy**, not misalignment.

Alignment is a **key component** of the evaluation pipeline and must be chosen carefully depending on the reconstruction
quality and the evaluation goals.

---
