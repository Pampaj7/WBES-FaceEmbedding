import torch
import torch.nn as nn
import sys

# ... (try/except per importare DiffusionNet) ...
try:
    import diffusion_net
    DiffusionNet = diffusion_net.layers.DiffusionNet 
except Exception as e1:
    try:
        from diffusion_net import DiffusionNet
    except Exception as e2:
        print(f"Errore: diffusion_net non trovato.")
        print(f"Errore 1: {e1}")
        print(f"Errore 2: {e2}")
        sys.exit(1)


class DiffusionAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, width=128, n_blocks=4):
        super().__init__()
        
        self.latent_dim = latent_dim
        print(f"🧬 Creazione DiffusionAutoencoder [GLOBALE-PULITO]: Latent={latent_dim}, Width={width}, N_block={n_blocks}")

        # 🌟 CORREZIONE: Abilita le feature di gradiente
        self.encoder = DiffusionNet(
            C_in=3,
            C_out=latent_dim,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True, # 
            dropout=0.0,
        )

        self.decoder = DiffusionNet(
            C_in=latent_dim, 
            C_out=3,
            C_width=width,
            N_block=n_blocks,
            with_gradient_features=True, # 
            dropout=0.0,
        )
        
        self.tanh_out = nn.Tanh()
        
    # 🌟 CORREZIONE: Aggiunto 'gradX' e 'gradY'
    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY):
        
        # === 1. ENCODER ===
        # 🌟 CORREZIONE: Passa 'gradX' e 'gradY'
        Z_per_vertex = self.encoder(V, mass, L, evals, evecs, 
                                    faces=faces, gradX=gradX, gradY=gradY)

        # === 2. POOLING ===
        # Pooling robusto con massa (questo è fatto bene!)
        if mass.sum() < 1e-6:
             Z_global = Z_per_vertex.mean(dim=0, keepdim=True)
        else:
             Z_global = (Z_per_vertex * mass.unsqueeze(1)).sum(dim=0, keepdim=True) / mass.sum()

        # === 3. DECODER ===
        N_verts = V.shape[0]
        Z_broadcast = Z_global.expand(N_verts, -1) 
        
        # 🌟 CORREZIONE: Passa 'gradX' e 'gradY'
        V_rec = self.decoder(Z_broadcast, mass, L, evals, evecs, 
                             faces=faces, gradX=gradX, gradY=gradY)

        # L'output è Tanh, coerente con la normalizzazione [-1, 1] dei vertici
        V_rec = self.tanh_out(V_rec) 

        return V_rec, Z_global