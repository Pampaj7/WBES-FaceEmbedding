#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import trimesh

# IMPORTANTE: serve avere il repo diffusion-net nel PYTHONPATH
# es: PYTHONPATH=/path/to/diffusion-net/src python3 train_cross_topology.py
import diffusion_net
from diffusion_net.geometry import normalize_positions, compute_operators

# ==============
# CONFIG
# ==============

CFG = {
    "bfm_dir": "/path/to/BFM_meshes",      # cartella .obj BFM
    "flame_dir": "/path/to/FLAME_meshes",  # cartella .obj FLAME
    "cache_dir": "./op_cache",             # dove salvare operatori precomputati
    "k_eig": 128,
    "c_width": 128,
    "emb_dim": 64,      # dimensione embedding globale
    # ogni batch contiene 2 soggetti diversi, ciascuno con (BFM, FLAME)
    "batch_size": 2,
    "num_workers": 0,
    "lr": 1e-4,
    "max_epochs": 20,
    "triplet_margin": 0.2,
    "lambda_align": 1.0,  # peso per loss di allineamento cross-topology
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# =========================
# UTILS: subject id parsing
# =========================


def parse_subject_id(path):
    # Adatta questa funzione al tuo naming (es: subject_0001_something.obj)
    # Qui prendo solo la prima parte prima di '_' come id: es. '0001'
    stem = Path(path).stem
    # Esempi possibili:
    #   0001_neutral.obj -> '0001'
    #   subj-23_expr1.obj -> 'subj-23'
    sid = stem.split('_')[0]
    return sid

# ======================================
# DATASET: indicizza .obj e subject ids
# ======================================


class MeshDataset(Dataset):
    """
    Indice di mesh per topologia (folder).
    Ritorna (path, subject_id).
    """

    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.obj")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"Nessun .obj in {root_dir}")
        self.subjects = [parse_subject_id(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {
            "path": self.files[idx],
            "sid": self.subjects[idx],
        }

# =========================================================
# MATCHER: batch di coppie (BFM, FLAME) per gli stessi sID
# =========================================================


class CrossTopoPairedSampler:
    """
    Costruisce mini-batch di size=CFG['batch_size'] soggetti:
      per ciascun soggetto -> una mesh BFM e una mesh FLAME
    Inoltre fornisce NEGATIVI intra-topology per triplet.
    """

    def __init__(self, ds_bfm: MeshDataset, ds_flame: MeshDataset, batch_size=2):
        # mappa subject_id -> lista paths
        self.map_bfm = {}
        for p, s in zip(ds_bfm.files, ds_bfm.subjects):
            self.map_bfm.setdefault(s, []).append(p)
        self.map_flame = {}
        for p, s in zip(ds_flame.files, ds_flame.subjects):
            self.map_flame.setdefault(s, []).append(p)

        # soggetti comuni a entrambe le topologie
        self.common = sorted(list(set(self.map_bfm.keys())
                             & set(self.map_flame.keys())))
        if len(self.common) == 0:
            raise RuntimeError("Nessun subject_id in comune tra BFM e FLAME.")

        self.batch_size = batch_size

    def __iter__(self):
        # shuffle soggetti
        ids = self.common.copy()
        random.shuffle(ids)
        # crea batch di soggetti
        for i in range(0, len(ids), self.batch_size):
            chunk = ids[i:i+self.batch_size]
            if len(chunk) < self.batch_size:
                continue
            # per ogni subject scegli una mesh random in BFM e una in FLAME
            pairs = []
            for sid in chunk:
                p_bfm = random.choice(self.map_bfm[sid])
                p_flm = random.choice(self.map_flame[sid])
                pairs.append((sid, p_bfm, p_flm))
            yield pairs

    def __len__(self):
        return len(self.common) // self.batch_size

# =========================================
# OP-CACHE: precomputo e salvataggio operatori
# =========================================


class OperatorCache:
    """
    Cache su disco per frames, mass, L, evals, evecs, gradX, gradY.
    Chiave: (toponame, path)
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key2path(self, topo_name, mesh_path):
        base = f"{topo_name}__{Path(mesh_path).stem}.npz"
        return self.cache_dir / base

    def get(self, topo_name, mesh_path, k_eig):
        out = self._key2path(topo_name, mesh_path)
        if out.exists():
            data = np.load(out, allow_pickle=True)
            return [torch.from_numpy(data[k]).float() for k in ["mass", "L", "evals", "evecs", "gradX", "gradY"]]
        # compute
        mesh = trimesh.load_mesh(mesh_path, process=False)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        verts = normalize_positions(verts)
        frames, mass, L, evals, evecs, gradX, gradY = compute_operators(
            verts, faces, k_eig=k_eig)
        np.savez(out,
                 mass=mass.numpy(),
                 L=L.numpy(),
                 evals=evals.numpy(),
                 evecs=evecs.numpy(),
                 gradX=gradX.numpy(),
                 gradY=gradY.numpy())
        return [mass, L, evals, evecs, gradX, gradY], verts, faces

    def get_with_geo(self, topo_name, mesh_path, k_eig):
        """Ritorna anche verts/faces (battezzati) per features."""
        out = self._key2path(topo_name, mesh_path)
        if out.exists():
            # Se già cache, dobbiamo ricaricare anche verts/faces (non salvati): ricalcolo leggero
            mesh = trimesh.load_mesh(mesh_path, process=False)
            verts = torch.tensor(mesh.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh.faces, dtype=torch.long)
            verts = normalize_positions(verts)
            data = np.load(out, allow_pickle=True)
            mass, L, evals, evecs, gradX, gradY = [torch.from_numpy(data[k]).float()
                                                   for k in ["mass", "L", "evals", "evecs", "gradX", "gradY"]]
            return (mass, L, evals, evecs, gradX, gradY), verts, faces
        else:
            return self.get(topo_name, mesh_path, k_eig)

# =========================================
# MODEL: shared-backbone DiffusionNet
# =========================================


class DiffusionBackbone(nn.Module):
    """
    Backbone DiffusionNet condivisa (stessi pesi per tutte le topologie).
    Proietta per-vertex features -> per-vertex latent (C_width) -> linear head -> (C_out per-vertex).
    Poi facciamo un global pooling + MLP per embedding globale (emb_dim).
    """

    def __init__(self, c_in=3, c_width=128, c_out=64, emb_dim=64):
        super().__init__()
        self.encoder = diffusion_net.layers.DiffusionNet(
            C_in=c_in,
            C_out=c_out,
            C_width=c_width,
            N_block=4,
            outputs_at='vertices',
            last_activation=None,
        )
        # Global head: media sui vertici -> MLP + L2 norm
        self.proj = nn.Sequential(
            nn.Linear(c_out, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, verts, mass, L, evals, evecs, gradX, gradY, faces):
        """
        verts: [N, 3]  (già normalizzati)
        operatori: mass [N], L [N,N], evals [K], evecs [N,K], gradX/gradY [N,K]
        faces: [F,3]
        """
        # batchify
        features = verts.unsqueeze(0)          # [1, N, 3]
        mass = mass.unsqueeze(0)           # [1, N]
        L = L.unsqueeze(0)              # [1, N, N]
        evals = evals.unsqueeze(0)          # [1, K]
        evecs = evecs.unsqueeze(0)          # [1, N, K]
        gradX = gradX.unsqueeze(0)          # [1, N, K]
        gradY = gradY.unsqueeze(0)          # [1, N, K]
        faces = faces.unsqueeze(0)          # [1, F, 3]

        with torch.no_grad():  # opzionale togli grad per verts se non vuoi backprop su posizioni
            pass

        per_vertex = self.encoder(
            features, mass, L=L, evals=evals, evecs=evecs,
            gradX=gradX, gradY=gradY, faces=faces
        )  # [1, N, c_out]

        # Global average pooling sui vertici
        per_vertex = per_vertex.squeeze(0)      # [N, c_out]
        global_feat = per_vertex.mean(dim=0)    # [c_out]

        emb = self.proj(global_feat)            # [emb_dim]
        emb = nn.functional.normalize(emb, p=2, dim=0)  # L2-normalized
        return emb  # [emb_dim]

# =========================================
# LOSSES
# =========================================


def cosine_distance(a, b):
    # a,b: [D]
    sim = torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)
    return 1.0 - sim  # distanza = 1 - cos


def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = cosine_distance(anchor, positive)
    d_an = cosine_distance(anchor, negative)
    return torch.relu(d_ap - d_an + margin)

# =========================================
# TRAIN LOOP (shared backbone)
# =========================================


def train():
    device = CFG["device"]
    os.makedirs(CFG["cache_dir"], exist_ok=True)

    ds_bfm = MeshDataset(CFG["bfm_dir"])
    ds_flm = MeshDataset(CFG["flame_dir"])
    sampler = CrossTopoPairedSampler(
        ds_bfm, ds_flm, batch_size=CFG["batch_size"])
    op_cache = OperatorCache(CFG["cache_dir"])

    model = DiffusionBackbone(
        c_in=3, c_width=CFG["c_width"], c_out=CFG["c_width"], emb_dim=CFG["emb_dim"]
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    for epoch in range(1, CFG["max_epochs"]+1):
        model.train()
        running = []
        for pairs in sampler:
            # pairs: lista di length=batch_size di tuple (sid, path_bfm, path_flm)
            optim.zero_grad()
            loss_batch = 0.0

            # Per negative sampling intra-topology, prendo i vicini di batch come negativi
            # (semplice ma efficace per partire)
            for idx, (sid, pb, pf) in enumerate(pairs):
                # BFM
                (mass_b, L_b, ev_b, U_b, gx_b, gy_b), V_b, F_b = op_cache.get_with_geo(
                    "BFM", pb, CFG["k_eig"])
                # FLAME
                (mass_f, L_f, ev_f, U_f, gx_f, gy_f), V_f, F_f = op_cache.get_with_geo(
                    "FLAME", pf, CFG["k_eig"])

                # to device
                V_b, F_b = V_b.to(device), F_b.to(device)
                V_f, F_f = V_f.to(device), F_f.to(device)
                mass_b, L_b, ev_b, U_b, gx_b, gy_b = [
                    t.to(device) for t in [mass_b, L_b, ev_b, U_b, gx_b, gy_b]]
                mass_f, L_f, ev_f, U_f, gx_f, gy_f = [
                    t.to(device) for t in [mass_f, L_f, ev_f, U_f, gx_f, gy_f]]

                # forward -> embedding normalizzati
                z_b = model(V_b, mass_b, L_b, ev_b,
                            U_b, gx_b, gy_b, F_b)  # [D]
                z_f = model(V_f, mass_f, L_f, ev_f,
                            U_f, gx_f, gy_f, F_f)  # [D]

                # 1) allineamento cross-topology (stesso soggetto)
                loss_align = torch.nn.functional.mse_loss(z_b, z_f)

                # 2) triplet intra-topology (negativo = altro elemento nel batch)
                # pick a negative index (ciclo semplice)
                neg_idx = (idx + 1) % len(pairs)
                _, pb_neg, pf_neg = pairs[neg_idx]

                # negativo BFM
                (mb_n, Lb_n, evb_n, Ub_n, gxb_n, gyb_n), Vb_n, Fb_n = op_cache.get_with_geo(
                    "BFM", pb_neg, CFG["k_eig"])
                Vb_n, Fb_n = Vb_n.to(device), Fb_n.to(device)
                mb_n, Lb_n, evb_n, Ub_n, gxb_n, gyb_n = [
                    t.to(device) for t in [mb_n, Lb_n, evb_n, Ub_n, gxb_n, gyb_n]]
                z_b_neg = model(Vb_n, mb_n, Lb_n, evb_n,
                                Ub_n, gxb_n, gyb_n, Fb_n)

                # negativo FLAME
                (mf_n, Lf_n, evf_n, Uf_n, gxf_n, gyf_n), Vf_n, Ff_n = op_cache.get_with_geo(
                    "FLAME", pf_neg, CFG["k_eig"])
                Vf_n, Ff_n = Vf_n.to(device), Ff_n.to(device)
                mf_n, Lf_n, evf_n, Uf_n, gxf_n, gyf_n = [
                    t.to(device) for t in [mf_n, Lf_n, evf_n, Uf_n, gxf_n, gyf_n]]
                z_f_neg = model(Vf_n, mf_n, Lf_n, evf_n,
                                Uf_n, gxf_n, gyf_n, Ff_n)

                # Triplet su entrambi i rami
                loss_trip_b = triplet_loss(
                    z_b, z_f, z_b_neg, margin=CFG["triplet_margin"])
                loss_trip_f = triplet_loss(
                    z_f, z_b, z_f_neg, margin=CFG["triplet_margin"])

                loss = CFG["lambda_align"] * loss_align + \
                    (loss_trip_b + loss_trip_f) * 0.5
                loss_batch += loss

            loss_batch = loss_batch / len(pairs)
            loss_batch.backward()
            optim.step()
            running.append(loss_batch.item())

        print(f"[Epoch {epoch:03d}] loss={np.mean(running):.4f}")

    # salva pesi
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(),
               "checkpoints/diffusionnet_shared_backbone.pt")
    print("✅ Saved checkpoints/diffusionnet_shared_backbone.pt")


if __name__ == "__main__":
    # Consiglio: fissa il seed per riproducibilità
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    train()
