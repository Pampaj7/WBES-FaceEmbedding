#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================
# DiffusionNet — Minimal example on a single .OBJ mesh
#
# Run command:
#   PYTHONPATH=/Users/pampaj/PycharmProjects/diffusion-net/src \
#   python3 /Users/pampaj/PycharmProjects/data/face_embedding/code/diffusion_obj.py
#
# This script loads a 3D mesh, computes its geometric Laplacian operators,
# and performs a single forward pass through DiffusionNet to produce
# per-vertex embeddings (or class scores).
#
# Author: Leonardo Pampaloni (2025)
# ==============================================================

import numpy as np
import torch
import diffusion_net
import trimesh
from diffusion_net.geometry import normalize_positions, compute_operators

# ==============================================================
# === CONFIGURATION SECTION ====================================
# ==============================================================

# Path alla mesh OBJ di test (qui usiamo una mesh facciale canonica)
OBJ_PATH = "/Users/pampaj/PycharmProjects/data/face_embedding/data/canonical_face_model.obj"

# Numero di autovettori (eigenvectors) del Laplaciano da calcolare.
# Governa la risoluzione spettrale — più alto = più dettagli geometrici ma più lento.
K_EIG = 64

# Dimensione interna (numero di canali "latenti") del modello DiffusionNet.
# Controlla la capacità di apprendimento: 32 = leggero, 128 = bilanciato, 256+ = pesante.
C_WIDTH = 128

# Numero di classi (o dimensione di embedding di output). Nella guida ufficiale usano 10.
N_CLASS = 10

# ==============================================================
# === MODEL INITIALIZATION =====================================
# ==============================================================

# Inizializza il modello DiffusionNet
# - C_in = 3 → ogni vertice ha coordinate (x,y,z)
# - C_out = N_CLASS → numero di output per vertice
# - C_width → numero di canali interni per layer
# - last_activation → funzione finale (softmax per classificazione)
# - outputs_at='vertices' → produce output a livello di vertice
model = diffusion_net.layers.DiffusionNet(
    C_in=3,
    C_out=N_CLASS,
    C_width=C_WIDTH,
    last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
    outputs_at="vertices"
)

# ==============================================================
# === LOAD MESH (via trimesh) ==================================
# ==============================================================

# Carichiamo la mesh (senza auto-riparazione)
mesh = trimesh.load_mesh(OBJ_PATH, process=False)
verts = torch.tensor(mesh.vertices, dtype=torch.float32)  # [V, 3]
faces = torch.tensor(mesh.faces, dtype=torch.long)        # [F, 3]

print(f"✅ Loaded mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

# ==============================================================
# === NORMALIZATION =============================================
# ==============================================================

# Normalizza la mesh: centroide = 0, scala = 1
# Questo step è cruciale per evitare problemi numerici durante la diffusione
verts = normalize_positions(verts)

# ==============================================================
# === GEOMETRIC OPERATORS =======================================
# ==============================================================

# Calcola i principali operatori geometrici (una sola volta per mesh):
# - frames: basi locali
# - mass: matrice di massa (peso per vertice)
# - L: Laplaciano discreto
# - evals: autovalori (frequenze)
# - evecs: autovettori (basi spettrali)
# - gradX, gradY: operatori di gradiente tangenziale
frames, mass, L, evals, evecs, gradX, gradY = compute_operators(
    verts, faces, k_eig=K_EIG)
print("✅ Operators computed")

# ==============================================================
# === PREPARE INPUTS FOR DIFFUSIONNET ===========================
# ==============================================================

# DiffusionNet accetta batch come [B, V, C], quindi aggiungiamo la dimensione batch = 1
features = verts.unsqueeze(0)  # [1, V, 3] — feature input (coordinate)
mass = mass.unsqueeze(0)       # [1, V]
L = L.unsqueeze(0)             # [1, V, V]
evals = evals.unsqueeze(0)     # [1, K]
evecs = evecs.unsqueeze(0)     # [1, V, K]
gradX = gradX.unsqueeze(0)     # [1, V, K]
gradY = gradY.unsqueeze(0)     # [1, V, K]
faces = faces.unsqueeze(0)     # [1, F, 3]

# ==============================================================
# === FORWARD PASS ==============================================
# ==============================================================

# Disattiva il gradiente (solo inference)
with torch.no_grad():
    outputs = model(
        features,
        mass,
        L=L,
        evals=evals,
        evecs=evecs,
        gradX=gradX,
        gradY=gradY,
        faces=faces
    )

# ==============================================================
# === RESULTS ===================================================
# ==============================================================

# L’output ha dimensione [B, V, C_out]
print("✅ Model forward successful!")
print("Output shape:", outputs.shape)
