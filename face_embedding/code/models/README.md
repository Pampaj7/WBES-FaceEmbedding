🧠 Models Module — Overview

This folder defines all neural network components and losses used for cross-topology embedding learning.

📁 Files

backbone.py
Implements the shared DiffusionBackbone, a DiffusionNet-based encoder that
converts a mesh (with precomputed operators) into a global embedding vector.
The embedding is L2-normalized and topology-invariant.

losses.py
Contains the training objectives used for metric learning:

Triplet loss for enforcing intra-subject similarity and inter-subject separation

Cosine loss or MSE loss for cross-topology alignment between BFM and FLAME embeddings.

🧩 Model Design
 ┌────────────────────────────────────┐
 │          DiffusionBackbone         │
 │------------------------------------│
 │ Input: (verts, faces, operators)   │
 │ - DiffusionNet (C_in→C_out)        │
 │ - Global pooling (mean or mass)    │
 │ - MLP projection → emb_dim         │
 │ Output: normalized embedding ℝᵈ    │
 └────────────────────────────────────┘

⚖️ Training Objective

The shared backbone is used for both topologies (BFM, FLAME) with identical weights.
The only difference lies in their geometric operators.

Training enforces that embeddings of the same subject — even from different topologies —
lie close in latent space, while embeddings of different subjects stay apart:


embedding_BFM(subject_i) ≈ embedding_FLAME(subject_i)
embedding_BFM(subject_i) ≠ embedding_FLAME(subject_j)


This is achieved through a combination of:

Alignment loss: MSE(z_bfm, z_flame)

Triplet loss: enforces separation between different subjects
(anchor = BFM_i, positive = FLAME_i, negative = BFM_j or FLAME_j)

🧠 Key Concepts

The DiffusionBackbone is a general geometric encoder, not tied to any specific topology.
Only the Laplacian and gradient operators determine its receptive field.

The embedding space learned is cross-topology consistent:
it maps equivalent shapes (same identity, different topology) to nearby points.

⚙️ Typical Usage
# Forward pass on BFM and FLAME meshes (shared backbone)
z_bfm = model(verts_bfm, mass_bfm, L_bfm, evals_bfm, evecs_bfm, gradX_bfm, gradY_bfm, faces_bfm)
z_flame = model(verts_flame, mass_flame, L_flame, evals_flame, evecs_flame, gradX_flame, gradY_flame, faces_flame)

# Compute alignment and triplet losses
loss_align = torch.nn.functional.mse_loss(z_bfm, z_flame)
loss_trip = triplet_loss(z_bfm, z_flame, z_neg)

🔁 Summary
Shared weights → DiffusionBackbone  
Different geometry → (BFM vs FLAME operators)  
Loss supervision → Triplet + Alignment  
Goal → topology-invariant face embedding space
