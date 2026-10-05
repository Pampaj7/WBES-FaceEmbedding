#!/usr/bin/env python3
"""DiffusionNet a due rami di operatori sullo stesso input: standard e col pozzo di potenziale.

Pilota pot (aau/runs/pilot_pot/). Il braccio col solo pozzo sostituisce gli operatori: guadagna
(forse) invarianza al bordo e perde la superficie che il pozzo sopprime (STATUS.md 2026-08-18,
crop 0.7072 -> 0.7012). Qui non si sceglie: ogni blocco diffonde le stesse feature con
ENTRAMBE le basi spettrali e la MLP del blocco vede le due diffusioni affiancate.

Blocco (stessa struttura di diffusion_net.layers.DiffusionNetBlock, raddoppiata dove dipende
dagli operatori):

    x_s = diffusione(x; mass, evals, evecs)              tempi appresi propri
    g_s = SpatialGradientFeatures(gradX x_s, gradY x_s)  rotazioni apprese proprie
    x_p, g_p = idem con gli operatori del pozzo
    out = x + MLP([x, x_s, g_s, x_p, g_p])               5C -> C -> C -> C (era 3C -> ...)

Concatenazione, non somma: con la somma la MLP non potrebbe pesare i due rami in modo diverso
per canale. Costo in parametri per blocco 11C^2 contro 7C^2, quindi a pari parametri totali la
width scende (matched_width; 128 -> 103, +0.2%, vedi __main__).

Per ramo, gli operatori sono quelli che il loader congelato (GTReadyDatasetNPZ) produce dal
proprio npz: ognuno con la SUA normalizzazione (evals / lambda_max, gradienti / sqrt(lambda_max)),
cioe' il ramo pozzo vede esattamente quello che vede il braccio col solo pozzo. La massa e' la
stessa (il pozzo e' un termine di ordine zero, aau/models/check_pot_ops.py lo verifica). L non
serve: la diffusione spettrale non lo legge.

Bottleneck per vertice, rumore, pooling mean+max e proiezione sono quelli di
DiffusionEncoderOnly, riga per riga.

Memoria: senza accorgimenti il modello non sta su un A10 (smoke 1055004: OOM a 22 GB nel primo
batch; il braccio a un ramo, width 128, arriva a ~19 GB). In training ogni blocco passa da
torch.utils.checkpoint: le attivazioni interne si ricalcolano nel backward invece di restare
in memoria. Stessi conti, stesso risultato (nessuna casualita' nell'encoder: dropout spento),
piu' tempo per epoca.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "diffusion-net/src"))

import torch.utils.checkpoint  # noqa: E402
from diffusion_net.layers import LearnedTimeDiffusion, MiniMLP, SpatialGradientFeatures  # noqa: E402

# Chiavi del campione con gli operatori del secondo ramo (aau/models/pilot_hooks.py le attacca).
POT_KEYS = ("pot_mass", "pot_evals", "pot_evecs", "pot_gradX", "pot_gradY")


class DualOpsBlock(nn.Module):
    def __init__(self, C_width: int, mlp_hidden_dims: list[int], dropout: bool = False,
                 with_gradient_rotations: bool = True):
        super().__init__()
        self.C_width = C_width
        self.diffusion_std = LearnedTimeDiffusion(C_width, method="spectral")
        self.diffusion_pot = LearnedTimeDiffusion(C_width, method="spectral")
        self.gradient_std = SpatialGradientFeatures(C_width, with_gradient_rotations=with_gradient_rotations)
        self.gradient_pot = SpatialGradientFeatures(C_width, with_gradient_rotations=with_gradient_rotations)
        self.mlp = MiniMLP([5 * C_width] + mlp_hidden_dims + [C_width], dropout=dropout)

    @staticmethod
    def _branch(x_in, diffusion, gradient, mass, evals, evecs, gradX, gradY):
        x_diffuse = diffusion(x_in, None, mass, evals, evecs)
        # come DiffusionNetBlock: torch.mm non fa batch, ciclo esplicito
        x_grads = []
        for b in range(x_in.shape[0]):
            x_gradX = torch.mm(gradX[b, ...], x_diffuse[b, ...])
            x_gradY = torch.mm(gradY[b, ...], x_diffuse[b, ...])
            x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
        return x_diffuse, gradient(torch.stack(x_grads, dim=0))

    def forward(self, x_in, ops_std, ops_pot):
        if x_in.shape[-1] != self.C_width:
            raise ValueError(f"last dim {x_in.shape[-1]} != C_width {self.C_width}")
        x_s, g_s = self._branch(x_in, self.diffusion_std, self.gradient_std, *ops_std)
        x_p, g_p = self._branch(x_in, self.diffusion_pot, self.gradient_pot, *ops_pot)
        return self.mlp(torch.cat((x_in, x_s, g_s, x_p, g_p), dim=-1)) + x_in


class DualOpsDiffusionNet(nn.Module):
    """DiffusionNet(outputs_at='vertices') con DualOpsBlock al posto di DiffusionNetBlock."""

    def __init__(self, C_in: int, C_out: int, C_width: int = 128, N_block: int = 4,
                 dropout: bool = False, checkpoint_blocks: bool = True):
        super().__init__()
        self.C_in, self.C_out, self.C_width = C_in, C_out, C_width
        self.checkpoint_blocks = checkpoint_blocks
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
        self.blocks = nn.ModuleList(
            DualOpsBlock(C_width, [C_width, C_width], dropout=dropout) for _ in range(N_block)
        )

    def forward(self, x_in, ops_std, ops_pot):
        """x_in (N, C_in); ops_* = (mass, evals, evecs, gradX, gradY) senza dimensione di batch."""
        if x_in.dim() != 2:
            raise ValueError("DualOpsDiffusionNet vuole x_in (N, C), un campione alla volta")
        x = self.first_lin(x_in.unsqueeze(0))
        ops_std = tuple(t.unsqueeze(0) for t in ops_std)
        ops_pot = tuple(t.unsqueeze(0) for t in ops_pot)
        for b in self.blocks:
            if self.checkpoint_blocks and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(b, x, ops_std, ops_pot, use_reentrant=False)
            else:
                x = b(x, ops_std, ops_pot)
        return self.last_lin(x).squeeze(0)


class DiffusionEncoderDualOps(nn.Module):
    """DiffusionEncoderOnly con l'encoder a due rami di operatori."""

    def __init__(self, latent_dim=256, width=103, n_blocks=4, dropout=0.1, pool_mode="mean"):
        super().__init__()
        self.latent_dim = latent_dim
        self.pool_mode = str(pool_mode)
        print(f"DiffusionEncoderDualOps | Z={latent_dim}, width={width}, blocks={n_blocks}, "
              f"pool={self.pool_mode}", flush=True)

        # dropout=0.0 nell'encoder, come DiffusionEncoderOnly (il dropout sta nel bottleneck)
        self.encoder = DualOpsDiffusionNet(C_in=3, C_out=latent_dim, C_width=width,
                                           N_block=n_blocks, dropout=False)
        self.vertex_bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim),
        )
        if self.pool_mode == "meanmax":
            self.pool_proj = nn.Linear(2 * latent_dim, latent_dim)
        elif self.pool_mode == "mean":
            self.pool_proj = nn.Identity()
        else:
            raise ValueError("pool_mode must be 'mean' or 'meanmax'")

    def forward(self, V, mass, evals, evecs, gradX, gradY, pot_ops,
                return_per_vertex: bool = False, add_noise: bool = True):
        Z_per_vertex = self.encoder(V, (mass, evals, evecs, gradX, gradY), tuple(pot_ops))
        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)
        if add_noise:
            Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)

        Z_mean = Z_per_vertex.mean(dim=0, keepdim=True)
        if self.pool_mode == "meanmax":
            Z_max = Z_per_vertex.max(dim=0, keepdim=True).values
            Z_global = self.pool_proj(torch.cat([Z_mean, Z_max], dim=1))
        else:
            Z_global = self.pool_proj(Z_mean)

        if return_per_vertex:
            return Z_per_vertex, Z_global
        return Z_global


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def reference_params(latent_dim=256, width=128, n_blocks=4, pool_mode="meanmax") -> int:
    """Parametri di xyz_dn (DiffusionEncoderOnly) alla ricetta v1."""
    sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/autoencoder"))
    from diffusion_autoencoder import DiffusionEncoderOnly
    return n_params(DiffusionEncoderOnly(latent_dim=latent_dim, width=width, n_blocks=n_blocks,
                                         dropout=0.1, pool_mode=pool_mode))


def matched_width(target: int, latent_dim=256, n_blocks=4, pool_mode="meanmax") -> int:
    """Width del modello a due rami col numero di parametri piu' vicino a `target`."""
    best = min(range(32, 129), key=lambda w: abs(n_params(DiffusionEncoderDualOps(
        latent_dim=latent_dim, width=w, n_blocks=n_blocks, pool_mode=pool_mode)) - target))
    return best


def _demo() -> None:
    """Parametri a confronto e un forward su una mesh finta: i due rami devono contare entrambi."""
    import contextlib
    import io

    ref = reference_params()
    with contextlib.redirect_stdout(io.StringIO()):
        w = matched_width(ref)
        m = DiffusionEncoderDualOps(latent_dim=256, width=w, n_blocks=4, pool_mode="meanmax")
    n = n_params(m)
    print(f"xyz_dn width 128: {ref} parametri | due rami width {w}: {n} ({100 * (n - ref) / ref:+.1f}%)")

    torch.manual_seed(0)
    N, K = 200, 16
    V = torch.randn(N, 3)
    mass = torch.rand(N) + 0.1
    idx = torch.stack([torch.arange(N), torch.randint(0, N, (N,))])
    grad = lambda: torch.sparse_coo_tensor(idx, torch.randn(N), (N, N)).coalesce()  # noqa: E731
    std = (mass, torch.linspace(0, 1, K), torch.randn(N, K), grad(), grad())
    pot = (mass, torch.linspace(0.2, 1, K), torch.randn(N, K), grad(), grad())
    m.eval()
    with torch.no_grad():
        z = m(V, *std, pot_ops=pot, add_noise=False)
        z_other = m(V, *std, pot_ops=(mass, torch.linspace(0.5, 1, K), torch.randn(N, K), grad(), grad()),
                    add_noise=False)
    assert z.shape == (1, 256), z.shape
    # il checkpoint non deve cambiare ne' l'uscita ne' i gradienti
    grads = []
    for ck in (False, True):
        m.encoder.checkpoint_blocks = ck
        m.zero_grad()
        out = m(V, *std, pot_ops=pot, add_noise=False)
        out.square().sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-5, atol=1e-7), "checkpoint cambia i gradienti"
    assert not torch.allclose(z, z_other), "il ramo pozzo non cambia l'embedding"
    print("demo OK: forward (1, 256), il ramo pozzo entra nell'embedding, checkpoint = stessi gradienti")


if __name__ == "__main__":
    _demo()
