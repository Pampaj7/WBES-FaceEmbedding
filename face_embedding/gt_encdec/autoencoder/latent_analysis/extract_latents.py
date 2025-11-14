import os, sys, torch, numpy as np, random

# === Add DiffusionNet path ===
for p in [
    "/equilibrium/lpampaloni/diffusion-net/src",
    "/home/pampaj/diffusion-net/src",
    "/seidenas/users/lpampaloni/diffusion-net/src",
]:
    if p not in sys.path:
        sys.path.append(p)

# aggiungi la cartella dell'autoencoder al path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOENC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # una directory sopra

if AUTOENC_DIR not in sys.path:
    sys.path.append(AUTOENC_DIR)

from diffusion_autoencoder import DiffusionAutoencoder

# === CONFIG ===
CKPT = "../results_diffusionAE_latentaware/latentaware_epoch50.pth"
NPZ_DIR = "../../../../datasets/GT_ready/npz_data"
OUT_DIR = "../results_diffusionAE_latentaware/latents_full"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionAutoencoder(latent_dim=256, width=128, n_blocks=4, k_spec=16).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


import traceback

def safe_row_normalize(X, eps=1e-9):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / (norms + eps)


# === Utility per caricare i file NPZ ===
def load_npz(path):
    d = np.load(path)

    def make_sparse(prefix):
        idx = torch.tensor(d[f"{prefix}_indices"], dtype=torch.long, device=device)
        if idx.shape[0] == 2:
            pass
        elif idx.shape[1] == 2:
            idx = idx.T
        else:
            raise ValueError(f"Shape indici non riconosciuta per {prefix}: {idx.shape}")
        vals = torch.tensor(d[f"{prefix}_values"], dtype=torch.float32, device=device)
        shape = tuple(int(x) for x in d[f"{prefix}_shape"])
        return torch.sparse_coo_tensor(idx, vals, size=shape)

    return {
        "verts": torch.tensor(d["verts"], dtype=torch.float32, device=device),
        "faces": torch.tensor(d["faces"], dtype=torch.long, device=device),
        "mass":  torch.tensor(d["mass"], dtype=torch.float32, device=device),
        "evals": torch.tensor(d["evals"], dtype=torch.float32, device=device),
        "evecs": torch.tensor(d["evecs"], dtype=torch.float32, device=device),
        "L":     make_sparse("L"),
        "gradX": make_sparse("gradX"),
        "gradY": make_sparse("gradY"),
    }

# === Subset random di mesh da processare ===
MAX_FILES = 200
all_files = [f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")]
subset = sorted(random.sample(all_files, min(MAX_FILES, len(all_files))))
print(f"📦 Processing subset of {len(subset)} / {len(all_files)} meshes")

# === Estrazione latenti ===
with torch.no_grad():
    for fn in subset:
        sid = os.path.splitext(fn)[0]
        try:
            D = load_npz(os.path.join(NPZ_DIR, fn))

            # === ENCODER SOLO ===
            Z_per_vertex = model.encoder(
                D["verts"], D["mass"], D["L"], D["evals"], D["evecs"],
                faces=D["faces"], gradX=D["gradX"], gradY=D["gradY"]
            )
            Z_per_vertex = model.vertex_bottleneck(Z_per_vertex)

            # RAW
            z_field_raw = Z_per_vertex.cpu().numpy().astype(np.float32)   # (V, D)
            z_global_raw = z_field_raw.mean(axis=0)                      # (D,)

            # normalizzazioni (opzionali ma salvate)
            z_field_norm = safe_row_normalize(z_field_raw)
            z_global_norm = z_global_raw / (np.linalg.norm(z_global_raw) + 1e-9)

            np.savez_compressed(
                os.path.join(OUT_DIR, f"{sid}.npz"),
                Z_field_raw=z_field_raw,
                Z_global_raw=z_global_raw,
                Z_field_norm=z_field_norm,
                Z_global_norm=z_global_norm
            )

            print(f"💾 {sid} done ({z_field_raw.shape[0]} verts)")

        except Exception as e:
            print(f"[WARN] Skipping {sid}: {e}")
            print(traceback.format_exc())


print(f"\n✅ Latenti globali + per-vertex salvati in: {OUT_DIR}")
