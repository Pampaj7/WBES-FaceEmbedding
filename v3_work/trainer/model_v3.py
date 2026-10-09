"""Modello del trainer v3 e i due modi di calcolare gli embedding di un passo.

Pooling (``--pooling``):
  * ``meanmax`` (default, v2): media per vertice + max, poi pool_proj. Il modello e' la classe v1
    DiffusionEncoderOnly cosi' com'e': stessi pesi iniziali, stesso checkpoint, stessi tool di eval.
  * ``area_meanmax``: media pesata per massa (area dei vertici) + max. Cambia solo il termine che dipende
    dalla densita' della mesh.
  * ``area_attn``: media pesata per massa + attenzione pesata per massa (softmax di log(massa) + punteggio
    appreso): entrambe approssimano integrali sulla superficie, quindi non dipendono dalla densita'. Il
    punteggio parte da zero, quindi all'inizio l'attenzione e' la media pesata.
  Le varianti sono sottoclassi di DiffusionEncoderOnly con lo stesso forward: l'eval congelato v1
  (forward_model fa isinstance) le usa senza modifiche. I moduli nuovi sono creati dentro fork_rng: il
  generatore CPU non avanza, quindi encoder e proiezione partono IDENTICI al controllo a parita' di seme.

Embedding di un passo (``--forward``):
  * ``sequential`` (default, v2): una forward per mesh, perturbazione e forward intrecciati nello stesso
    ordine di train_runner._train_epoch_mixed (anche l'ordine delle estrazioni dal generatore CUDA).
  * ``groups``: il percorso di v2_work/fastio/batched.py (copiato e esteso al training e ai pooling nuovi):
    gruppi di mesh di taglia quasi uguale (max/min <= 1 + ``pad_slack``, default 0.05), una forward [B,N,C]
    per gruppo con padding mascherato. Misurato da E9 (aau/runs/evidence/e9/tables.md) come il migliore
    con V <= 10k (170 contro 93 mesh/s sul passo da 64 mesh); con V da 2k a 60k nessun guadagno.
  * ``packed``: tutte le mesh del passo concatenate in UNA forward. gradX/gradY diventano una matrice
    sparsa a blocchi diagonali (un solo spmm al posto di uno per mesh); la diffusione spettrale, che
    vuole una base per mesh, gira per bucket di taglia (max/min <= ``bucket_ratio``) con padding a zero
    (righe di evecs e massa nulle: il padding non contribuisce, vedi v2_work/fastio/batched.py). Stessa
    matematica del sequenziale; dropout e rumore latente estraggono numeri casuali in un altro ordine.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

import common  # noqa: F401  (path del pacchetto congelato)

from diffusion_autoencoder import DiffusionEncoderOnly  # noqa: E402
from robustness.data_utils import sample_to_device  # noqa: E402
from robustness.model_helpers import build_model as build_model_v1  # noqa: E402
from robustness.model_helpers import forward_model  # noqa: E402
from robustness.noise import apply_xyz_perturbation_with_params  # noqa: E402

import area_v3  # noqa: E402

POOLINGS = ("meanmax", "area_meanmax", "area_attn")


def _area_weights(mass: torch.Tensor) -> torch.Tensor:
    w = mass.reshape(-1).clamp_min(0)
    return w / w.sum().clamp_min(1e-30)


class EncoderV3(DiffusionEncoderOnly):
    """DiffusionEncoderOnly con pooling pesato per area (``area_meanmax`` o ``area_attn``)."""

    def __init__(self, latent_dim=256, width=128, n_blocks=4, dropout=0.1, pooling="area_attn",
                 attn_heads=1, attn_hidden=64, area_weights="mass"):
        if pooling not in ("area_meanmax", "area_attn"):
            raise ValueError(f"EncoderV3: pooling {pooling!r}")
        if area_weights not in area_v3.AREA_WEIGHTS:
            raise ValueError(f"EncoderV3: area_weights {area_weights!r}")
        super().__init__(latent_dim=latent_dim, width=width, n_blocks=n_blocks, dropout=dropout, pool_mode="meanmax")
        self.pooling = str(pooling)
        self.area_weights = str(area_weights)   # pesi del pooling: mass (massa degli operatori), smooth, winsor
        self.attn_heads = int(attn_heads)
        if pooling == "area_attn":
            with torch.random.fork_rng(devices=[]):
                self.attn = nn.Sequential(nn.Linear(latent_dim, attn_hidden), nn.Tanh(),
                                          nn.Linear(attn_hidden, self.attn_heads))
                nn.init.zeros_(self.attn[-1].weight)
                nn.init.zeros_(self.attn[-1].bias)
                if self.attn_heads != 1:
                    self.pool_proj = nn.Linear((1 + self.attn_heads) * latent_dim, latent_dim)

    def pool(self, Z: torch.Tensor, mass: torch.Tensor, w_raw: torch.Tensor | None = None) -> torch.Tensor:
        """Z [N, D], massa [N] (o pesi gia' calcolati ``w_raw``) -> [1, D]."""
        w = _area_weights(mass if w_raw is None else w_raw).to(Z.dtype)
        z_mean = (w.unsqueeze(1) * Z).sum(dim=0, keepdim=True)
        if self.pooling == "area_meanmax":
            return self.pool_proj(torch.cat([z_mean, Z.max(dim=0, keepdim=True).values], dim=1))
        logits = self.attn(Z) + torch.log(w.clamp_min(1e-30)).unsqueeze(1)     # [N, H]
        alpha = torch.softmax(logits, dim=0)
        z_att = (alpha.transpose(0, 1) @ Z).reshape(1, -1)                    # [1, H*D]
        return self.pool_proj(torch.cat([z_mean, z_att], dim=1))

    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY, return_per_vertex: bool = False,
                add_noise: bool = True):
        Z = self.encoder(V, mass, L, evals, evecs, faces=faces, gradX=gradX, gradY=gradY)
        Z = self.vertex_bottleneck(Z)
        if add_noise:
            Z = Z + 0.01 * torch.randn_like(Z)
        with torch.no_grad():
            w = area_v3.pool_weights(self.area_weights, V.detach(), faces, mass, evecs)
        Z_global = self.pool(Z, mass, w)
        if return_per_vertex:
            return Z, Z_global
        return Z_global


def build_model_v3(args, device: torch.device) -> nn.Module:
    """``meanmax``: la build v1 (DiffusionEncoderOnly). Altrimenti EncoderV3 con la stessa larghezza.
    ``--head factorized``: factorized_v3.FactorizedEncoderV3 (EncoderV3 + testa della dimensione)."""
    if str(args.model) != "xyz_dn":
        raise SystemExit("il trainer v3 supporta solo --model xyz_dn")
    pooling = str(getattr(args, "pooling", "meanmax"))
    if str(getattr(args, "head", "embed")) in ("factorized", "factorized2"):
        if pooling == "meanmax":
            raise SystemExit("--head factorized richiede un pooling per area (--area on|robust o --pooling area_*)")
        import factorized_v3
        return factorized_v3.build(args, device)
    if pooling == "meanmax":
        if str(args.pool_mode) != "meanmax":
            raise SystemExit("--pooling meanmax richiede --pool_mode meanmax (il controllo v2)")
        return build_model_v1(args, device)
    m = EncoderV3(latent_dim=args.latent_dim, width=args.width, n_blocks=args.n_blocks, dropout=args.dropout,
                  pooling=pooling, attn_heads=int(getattr(args, "attn_heads", 1)),
                  area_weights=str(getattr(args, "area_weights", "mass")))
    return m.to(device)


def model_from_checkpoint_args(args, device) -> nn.Module:
    """Per l'eval: ricostruisce il modello di un checkpoint v3 (o v1) dai suoi ``args``."""
    return build_model_v3(args, device)


# --- embedding di un passo ---------------------------------------------------------------------------

def _perturb(V: torch.Tensor, sigma: float, mode: str, perturbation) -> torch.Tensor:
    if sigma > 0.0:
        return apply_xyz_perturbation_with_params(V=V, mode=mode, sigma=sigma, params=perturbation)
    return V


def to_device(sample: Dict, device, fast: bool, need_faces: bool = False) -> Dict:
    """sample_to_device di v1 (8 tensori), oppure con ``fast`` solo i 6 che il forward spettrale legge: L e le
    facce restano None (DiffusionNet li ignora con diffusione spettrale e uscite ai vertici). Le facce si
    copiano se il pooling le usa (aree 'smooth')."""
    if not fast:
        return sample_to_device(sample, device=device)
    out = {k: sample[k].to(device) for k in ("verts", "mass", "evals", "evecs", "gradX", "gradY")}
    out["L"] = None
    out["faces"] = sample["faces"].to(device) if need_faces else None
    return out


class StepEmbedder(nn.Module):
    """Embedding di tutte le mesh di un passo in UNA chiamata di modulo: e' cio' che DDP avvolge
    (una forward e una backward per passo, qualunque sia il numero di mesh)."""

    def __init__(self, model: nn.Module, forward_mode: str = "sequential", bucket_ratio: float = 1.5,
                 pad_slack: float = 0.05) -> None:
        super().__init__()
        self.pad_slack = float(pad_slack)
        self.fast_data = False
        if forward_mode not in ("sequential", "groups", "packed"):
            raise ValueError(f"forward {forward_mode!r}")
        self.model = model
        self.forward_mode = forward_mode
        self.bucket_ratio = float(bucket_ratio)
        # oggetti Python, non stato del modulo
        self.__dict__["dataset"] = None
        self.__dict__["perturbation"] = None
        self.__dict__["log_scales"] = None    # --scale-aug: log a per mesh del passo (None = spenta, come v2)

    def bind(self, dataset, perturbation) -> None:
        self.__dict__["dataset"] = dataset
        self.__dict__["perturbation"] = perturbation

    def forward(self, entries: List[tuple], sigma: float, add_noise: bool) -> torch.Tensor:
        device = next(self.model.parameters()).device
        log_a = self.__dict__.get("log_scales")
        if self.forward_mode == "sequential":
            zs = []
            for k, (_sid, idx, _topo, mode) in enumerate(entries):
                sample_d = to_device(self.dataset[int(idx)], device, self.fast_data,
                                     getattr(self.model, "area_weights", "mass") == "smooth")
                V0 = sample_d["verts"] if log_a is None else sample_d["verts"] * math.exp(float(log_a[k]))
                V_in = _perturb(V0, sigma, mode, self.perturbation)
                z, _ = forward_model(model=self.model, sample_dict=sample_d, V_in=V_in,
                                     return_gate_info=False, add_noise=add_noise)
                zs.append(z.squeeze(0))
            return torch.stack(zs, dim=0)
        items = []
        for k, (_sid, idx, _topo, mode) in enumerate(entries):
            s = self.dataset[int(idx)]
            V = s["verts"].to(device) if log_a is None else s["verts"].to(device) * math.exp(float(log_a[k]))
            it = {"verts": _perturb(V, sigma, mode, self.perturbation), "mass": s["mass"].to(device),
                  "evals": s["evals"].to(device), "evecs": s["evecs"].to(device),
                  "gradX": s["gradX"], "gradY": s["gradY"]}
            aw = getattr(self.model, "area_weights", None)
            if aw is not None:   # pesi del pooling per area, come in EncoderV3.forward
                with torch.no_grad():
                    it["pool_w"] = area_v3.pool_weights(aw, it["verts"], s["faces"].to(device) if aw == "smooth"
                                                        else None, it["mass"], it["evecs"])
            items.append(it)
        if self.forward_mode == "groups":
            return embed_groups(self.model, items, device, add_noise=add_noise, pad_slack=self.pad_slack)
        return embed_packed(self.model, items, device, add_noise=add_noise, bucket_ratio=self.bucket_ratio)


# --- forward a gruppi (v2_work/fastio/batched.py) ------------------------------------------------------

class _SparseBatch:
    """L'operando sparso [B,N,N] come lo usa DiffusionNetBlock (solo ``gradX[b,...]``): copia di batched.py.
    Il padding si esprime ridichiarando ogni matrice [n_pad, n_pad], senza aggiungere voci."""

    __slots__ = ("mats",)

    def __init__(self, mats, n_pad: int, device) -> None:
        self.mats = []
        for m in mats:
            m = m.to(device).coalesce()
            if m.shape[-1] != n_pad:
                m = torch.sparse_coo_tensor(m.indices(), m.values(), (n_pad, n_pad)).coalesce()
            self.mats.append(m)

    def __getitem__(self, key):
        return self.mats[key[0] if isinstance(key, tuple) else key]


def size_groups(ns: Sequence[int], pad_slack: float) -> List[List[int]]:
    """Indici raggruppati per taglia, max(N) <= (1 + pad_slack) * min(N) nel gruppo (batched.size_groups)."""
    order = sorted(range(len(ns)), key=lambda i: ns[i])
    groups: List[List[int]] = []
    cur: List[int] = []
    for i in order:
        if cur and ns[i] > (1.0 + pad_slack) * ns[cur[0]]:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def _pool_masked(model, x: torch.Tensor, mask: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Pooling del modello su [B,N,D] con le righe di padding escluse (mask False, massa 0)."""
    m = mask.unsqueeze(-1)
    pooling = getattr(model, "pooling", "meanmax")
    if pooling == "meanmax":
        z_mean = (x * m).sum(dim=-2) / mask.sum(dim=-1, keepdim=True)
    else:
        w = mass.clamp_min(0) * mask
        w = (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-30)).to(x.dtype)
        z_mean = (w.unsqueeze(-1) * x).sum(dim=-2)
    if pooling in ("meanmax", "area_meanmax"):
        if getattr(model, "pool_mode", "meanmax") != "meanmax":
            return model.pool_proj(z_mean)
        z_max = x.masked_fill(~m, float("-inf")).max(dim=-2).values
        return model.pool_proj(torch.cat([z_mean, z_max], dim=-1))
    logits = (model.attn(x) + torch.log(w.clamp_min(1e-30)).unsqueeze(-1)).masked_fill(~m, float("-inf"))
    alpha = torch.softmax(logits, dim=-2)                                   # [B, N, H]
    z_att = torch.einsum("bnh,bnd->bhd", alpha, x).reshape(x.shape[0], -1)  # [B, H*D]
    return model.pool_proj(torch.cat([z_mean, z_att], dim=-1))


def embed_groups(model, items: Sequence[Dict], device, add_noise: bool, pad_slack: float = 0.05) -> torch.Tensor:
    """Embedding [M, latent] per gruppi di taglia, nell'ordine d'ingresso (verts gia' perturbati)."""
    enc = model.encoder
    if enc.diffusion_method != "spectral" or enc.outputs_at != "vertices":
        raise ValueError("forward a gruppi: serve diffusione spettrale e uscite ai vertici")
    device = torch.device(device)
    ns = [int(it["verts"].shape[0]) for it in items]
    K = int(items[0]["evals"].numel())
    outs, order = [], []
    for g in size_groups(ns, pad_slack):
        B, N = len(g), max(ns[i] for i in g)
        verts = torch.zeros(B, N, 3, device=device)
        mass = torch.zeros(B, N, device=device)
        evecs = torch.zeros(B, N, K, device=device)
        evals = torch.zeros(B, K, device=device)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        poolw = torch.zeros(B, N, device=device)
        for b, i in enumerate(g):
            n = ns[i]
            verts[b, :n] = items[i]["verts"]
            mass[b, :n] = items[i]["mass"].reshape(-1)
            poolw[b, :n] = items[i].get("pool_w", items[i]["mass"]).reshape(-1)
            evecs[b, :n] = items[i]["evecs"]
            evals[b] = items[i]["evals"].reshape(-1)
            mask[b, :n] = True
        gx = _SparseBatch([items[i]["gradX"] for i in g], N, device)
        gy = _SparseBatch([items[i]["gradY"] for i in g], N, device)
        x = enc(verts, mass, None, evals, evecs, gradX=gx, gradY=gy)
        x = model.vertex_bottleneck(x)
        if add_noise:
            x = x + 0.01 * torch.randn_like(x)
        outs.append(_pool_masked(model, x, mask, poolw))
        order += g
    z = torch.cat(outs, dim=0)
    inv = torch.empty(len(order), dtype=torch.long, device=z.device)
    inv[torch.as_tensor(order, device=z.device)] = torch.arange(len(order), device=z.device)
    return z.index_select(0, inv)


# --- forward packed -----------------------------------------------------------------------------------

class _Packed:
    """Layout concatenato di M mesh ordinate per taglia, operatori a blocchi e bucket spettrali."""

    def __init__(self, items: Sequence[Dict], device: torch.device, bucket_ratio: float) -> None:
        ns = [int(it["verts"].shape[0]) for it in items]
        self.order = sorted(range(len(items)), key=lambda i: ns[i])
        n_sorted = [ns[i] for i in self.order]
        self.M = len(items)
        offs = [0]
        for n in n_sorted:
            offs.append(offs[-1] + n)
        self.N = offs[-1]
        self.verts = torch.cat([items[i]["verts"] for i in self.order], dim=0)
        self.mass = torch.cat([items[i]["mass"].reshape(-1) for i in self.order], dim=0)
        self.poolw = torch.cat([items[i].get("pool_w", items[i]["mass"]).reshape(-1) for i in self.order], dim=0)
        self.seg = torch.repeat_interleave(torch.arange(self.M, device=device),
                                           torch.as_tensor(n_sorted, device=device))
        self.gradX = self._block_diag([items[i]["gradX"] for i in self.order], offs, device)
        self.gradY = self._block_diag([items[i]["gradY"] for i in self.order], offs, device)
        # bucket di taglia consecutivi nell'ordine per taglia
        self.buckets = []
        start = 0
        K = int(items[self.order[0]]["evals"].numel())
        for j in range(1, self.M + 1):
            if j == self.M or n_sorted[j] > bucket_ratio * n_sorted[start]:
                members = list(range(start, j))
                Ng = max(n_sorted[m] for m in members)
                pad_index = torch.cat([torch.arange(n_sorted[m], device=device) + b * Ng
                                       for b, m in enumerate(members)])
                Bg = len(members)
                evecs = torch.cat([items[self.order[m]]["evecs"] for m in members], dim=0)
                if evecs.shape[-1] != K:
                    raise ValueError(f"k_eig diversi nel passo: {evecs.shape[-1]} contro {K}")
                evecs_pad = evecs.new_zeros(Bg * Ng, K).index_copy(0, pad_index, evecs).view(Bg, Ng, K)
                mass_pad = self.mass.new_zeros(Bg * Ng).index_copy(
                    0, pad_index, self.mass[offs[start]:offs[j]]).view(Bg, Ng)
                evals = torch.stack([items[self.order[m]]["evals"].reshape(-1) for m in members], dim=0)
                self.buckets.append({"a": offs[start], "b": offs[j], "Bg": Bg, "Ng": Ng, "pad_index": pad_index,
                                     "evecs": evecs_pad, "mass": mass_pad, "evals": evals})
                start = j

    @staticmethod
    def _block_diag(mats, offs, device) -> torch.Tensor:
        idx, val = [], []
        for m, o in zip(mats, offs):
            c = m.coalesce()
            idx.append(c.indices().to(device) + o)
            val.append(c.values().to(device))
        n = offs[-1]
        return torch.sparse_coo_tensor(torch.cat(idx, dim=1), torch.cat(val), (n, n)).coalesce()

    def spectral_diffuse(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """LearnedTimeDiffusion 'spectral' per mesh: evecs (coef * evecs^T (massa * x))."""
        outs = []
        for bk in self.buckets:
            C = x.shape[-1]
            Xp = x.new_zeros(bk["Bg"] * bk["Ng"], C).index_copy(0, bk["pad_index"], x[bk["a"]:bk["b"]])
            Xp = Xp.view(bk["Bg"], bk["Ng"], C)
            x_spec = torch.matmul(bk["evecs"].transpose(-2, -1), Xp * bk["mass"].unsqueeze(-1))
            coefs = torch.exp(-bk["evals"].unsqueeze(-1) * time.unsqueeze(0))
            xd = torch.matmul(bk["evecs"], coefs * x_spec)
            outs.append(xd.reshape(bk["Bg"] * bk["Ng"], C).index_select(0, bk["pad_index"]))
        return torch.cat(outs, dim=0)


def _packed_block(blk, x_in: torch.Tensor, P: _Packed) -> torch.Tensor:
    """DiffusionNetBlock.forward (diffusion-net/src/diffusion_net/layers.py) sul layout concatenato."""
    if x_in.shape[-1] != blk.C_width:
        raise ValueError(f"canali {x_in.shape[-1]} != {blk.C_width}")
    diff = blk.diffusion
    if diff.method != "spectral":
        raise ValueError("forward packed: solo diffusione 'spectral'")
    with torch.no_grad():
        diff.diffusion_time.data = torch.clamp(diff.diffusion_time, min=1e-8)
    x_diffuse = P.spectral_diffuse(x_in, diff.diffusion_time)
    if blk.with_gradient_features:
        x_grad = torch.stack((torch.mm(P.gradX, x_diffuse), torch.mm(P.gradY, x_diffuse)), dim=-1)
        feature_combined = torch.cat((x_in, x_diffuse, blk.gradient_features(x_grad)), dim=-1)
    else:
        feature_combined = torch.cat((x_in, x_diffuse), dim=-1)
    return blk.mlp(feature_combined) + x_in


def _segment_softmax_pool(logits: torch.Tensor, x: torch.Tensor, seg: torch.Tensor, M: int) -> torch.Tensor:
    """Somma per segmento di softmax_segmento(logits)[:, h] * x -> [M, H*D]."""
    H, D = logits.shape[1], x.shape[1]
    seg_max = torch.full((M, H), -math.inf, device=x.device, dtype=x.dtype).scatter_reduce(
        0, seg.unsqueeze(1).expand(-1, H), logits.detach(), reduce="amax", include_self=True)
    e = torch.exp(logits - seg_max[seg])
    den = x.new_zeros(M, H).index_add(0, seg, e)
    alpha = e / den[seg]
    att = x.new_zeros(M, H, D).index_add(0, seg, alpha.unsqueeze(-1) * x.unsqueeze(1))
    return att.reshape(M, H * D)


def _packed_pool(model, x: torch.Tensor, P: _Packed) -> torch.Tensor:
    M, D = P.M, x.shape[1]
    seg = P.seg
    pooling = getattr(model, "pooling", "meanmax")
    if pooling == "meanmax":
        cnt = torch.bincount(seg, minlength=M).to(x.dtype)
        z_mean = x.new_zeros(M, D).index_add(0, seg, x) / cnt.unsqueeze(1)
    else:
        wsum = P.poolw.new_zeros(M).index_add(0, seg, P.poolw.clamp_min(0))
        w = (P.poolw.clamp_min(0) / wsum.clamp_min(1e-30)[seg]).to(x.dtype)
        z_mean = x.new_zeros(M, D).index_add(0, seg, w.unsqueeze(1) * x)
    if pooling in ("meanmax", "area_meanmax"):
        if getattr(model, "pool_mode", "meanmax") != "meanmax":
            return model.pool_proj(z_mean)
        z_max = torch.full((M, D), -math.inf, device=x.device, dtype=x.dtype).scatter_reduce(
            0, seg.unsqueeze(1).expand(-1, D), x, reduce="amax", include_self=True)
        return model.pool_proj(torch.cat([z_mean, z_max], dim=1))
    logits = model.attn(x) + torch.log(w.clamp_min(1e-30)).unsqueeze(1)
    return model.pool_proj(torch.cat([z_mean, _segment_softmax_pool(logits, x, seg, M)], dim=1))


def embed_packed(model, items: Sequence[Dict], device, add_noise: bool, bucket_ratio: float = 1.5) -> torch.Tensor:
    """Embedding [M, latent] di M campioni (verts gia' perturbati) in una forward, nell'ordine d'ingresso."""
    enc = model.encoder
    if enc.outputs_at != "vertices" or enc.last_activation is not None:
        raise ValueError("forward packed: serve outputs_at='vertices' senza last_activation")
    P = _Packed(items, torch.device(device), bucket_ratio)
    x = enc.first_lin(P.verts)
    for blk in enc.blocks:
        x = _packed_block(blk, x, P)
    x = enc.last_lin(x)
    x = model.vertex_bottleneck(x)
    if add_noise:
        x = x + 0.01 * torch.randn_like(x)
    z_sorted = _packed_pool(model, x, P)
    inv = torch.empty(P.M, dtype=torch.long, device=z_sorted.device)
    inv[torch.as_tensor(P.order, device=z_sorted.device)] = torch.arange(P.M, device=z_sorted.device)
    return z_sorted.index_select(0, inv)
