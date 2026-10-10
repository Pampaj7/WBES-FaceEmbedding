"""Embedding fattorizzato dimensione + forma (``--head factorized``): z = (s, u).

  * s: scalare, log della centroid size (mm) prevista della regione del volto, da una testa piccola sulle stesse
    feature aggregate di u (media pesata per area e max, o attenzione);
  * u: embedding di forma (la proiezione di sempre, ``pool_proj``).
Il modello restituisce [s, u] (1 + latent_dim) nel training; con ``output(model, "u")`` solo u, che e' cio' che
leggono l'eval online e gli script di eval (GT di forma, maxabs o shape).

Loss (``factorized_loss``): la loss di forma scelta da ``--loss`` (v2: stress + rank + id) su u con la GT "shape"
(tools/build_factorized_targets.py: kappa * dP, dP = corda fra pre-forme a centroid size unitaria), piu'
``--lambda-size`` * MSE fra s e log S_i + log a (S_i centroid size vera dell'identita' sulla regione comune, la
stessa per ogni mesh, crop ed espressioni compresi; a = fattore dell'augmentation di scala della mesh).

Augmentation di scala (``--scale-aug lo,hi``): ogni mesh del passo e' moltiplicata per a, log-uniforme in [lo, hi],
PRIMA del rumore (il rumore resta in unita' assolute); bersaglio s + log a, u invariato. Le estrazioni dipendono
solo da (seme, rank, passo): la ripresa da checkpoint le riproduce.

Distanza form (Dryden & Mardia 2016, distanza riemanniana size-and-shape; ``ssriemdist`` del pacchetto R shapes):
    d_F^2 = (S_i - S_j)^2 + S_i S_j dP_ij^2,   S = exp(s),   dP = 2 sin(rho / 2) = ||u_i - u_j|| * dp_per_unit
equivalente a S_i^2 + S_j^2 - 2 S_i S_j cos(rho). Con dP calcolata SENZA rotazione per identita' (la GT shape di
questo trainer, nel frame canonico) d_F coincide con la GT ``F_centered`` di E12; con la rotazione
ottimizzata per coppia e' la distanza size-and-shape di Procrustes (controlli in build_factorized_targets.py).
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from common import domain_of
from losses_v3 import StepBatch, compute_loss
from model_v3 import EncoderV3

HEADS = ("embed", "factorized", "factorized2", "dual")
FACTORIZED = ("factorized", "factorized2")


class FactorizedEncoderV3(EncoderV3):
    """EncoderV3 (pooling per area) con la testa della dimensione: forward -> [s, u] o u (``out_mode``)."""

    def __init__(self, latent_dim=256, width=128, n_blocks=4, dropout=0.1, pooling="area_meanmax", attn_heads=1,
                 attn_hidden=64, area_weights="mass", size_hidden=64, size_init=0.0):
        super().__init__(latent_dim=latent_dim, width=width, n_blocks=n_blocks, dropout=dropout, pooling=pooling,
                         attn_heads=attn_heads, attn_hidden=attn_hidden, area_weights=area_weights)
        n_in = (2 if pooling == "area_meanmax" else 1 + self.attn_heads) * latent_dim
        # dentro fork_rng: encoder e pool_proj partono identici a EncoderV3 con lo stesso seme
        with torch.random.fork_rng(devices=[]):
            self.size_head = nn.Sequential(nn.Linear(n_in, size_hidden), nn.SiLU(), nn.Linear(size_hidden, 1))
            nn.init.zeros_(self.size_head[-1].weight)
            nn.init.constant_(self.size_head[-1].bias, float(size_init))    # all'inizio s = media del training
        self.out_mode = "full"

    def pool(self, Z: torch.Tensor, mass: torch.Tensor, w_raw: torch.Tensor | None = None) -> torch.Tensor:
        """Come EncoderV3.pool, con s dalle stesse feature aggregate."""
        w = (mass if w_raw is None else w_raw).reshape(-1).clamp_min(0)
        w = (w / w.sum().clamp_min(1e-30)).to(Z.dtype)
        z_mean = (w.unsqueeze(1) * Z).sum(dim=0, keepdim=True)
        if self.pooling == "area_meanmax":
            feat = torch.cat([z_mean, Z.max(dim=0, keepdim=True).values], dim=1)
        else:
            logits = self.attn(Z) + torch.log(w.clamp_min(1e-30)).unsqueeze(1)
            alpha = torch.softmax(logits, dim=0)
            feat = torch.cat([z_mean, (alpha.transpose(0, 1) @ Z).reshape(1, -1)], dim=1)
        u = self.pool_proj(feat)
        if self.out_mode == "u":
            return u
        return torch.cat([self.size_head(feat), u], dim=1)


class FactorizedNormEncoderV3(FactorizedEncoderV3):
    """``--head factorized2``: fattorizzazione per COSTRUZIONE. Dall'ingresso globale X: c, R = baricentro e centroid
    size pesati con i pesi del pooling (aree robuste, ``smooth`` con --area robust), Xn = (X - c) / R;
        u = pool_proj(feature(Xn))                       invariante esatta a traslazione e scala di X;
        s = log(R x L0) + delta(feature(Xn))             s(aX) = s(X) + log a esatta;
    delta (la testa della dimensione, ultimo strato a zero) corregge il supporto (crop, testa intera) rispetto alla
    regione del volto. Pesi, c e R senza gradiente, in float64."""

    def __init__(self, *a, unit_mm: float = 100.0, **kw):
        super().__init__(*a, **kw)
        self.unit_mm = float(unit_mm)
        with torch.no_grad():
            self.size_head[-1].bias.zero_()

    def group_normalize(self, it: dict, faces: torch.Tensor) -> dict:
        """Per ``--forward groups`` (model_v3.embed_groups): la normalizzazione di ``forward`` su una mesh, gia'
        perturbata: verts -> Xn, pesi del pooling, ``s_shift`` = log(R L0) da sommare a s dopo il pooling."""
        import area_v3
        V = it["verts"]
        with torch.no_grad():
            w = area_v3.pool_weights(self.area_weights, V.detach().double(), faces, it["mass"].double(),
                                     it["evecs"].double())
            wn = w / w.sum()
            c = (wn.unsqueeze(1) * V.double()).sum(0, keepdim=True)
            R = torch.sqrt((wn * ((V.double() - c) ** 2).sum(1)).sum())
        out = dict(it)
        out["verts"] = ((V.double() - c) / R).to(V.dtype)
        out["pool_w"] = w.to(V.dtype)
        out["s_shift"] = float(torch.log(R * self.unit_mm))
        return out

    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY, return_per_vertex: bool = False,
                add_noise: bool = True):
        import area_v3
        with torch.no_grad():
            w = area_v3.pool_weights(self.area_weights, V.detach().double(), faces, mass.double(), evecs.double())
            wn = w / w.sum()
            c = (wn.unsqueeze(1) * V.double()).sum(0, keepdim=True)
            R = torch.sqrt((wn * ((V.double() - c) ** 2).sum(1)).sum())
        Vn = ((V.double() - c) / R).to(V.dtype)
        Z = self.encoder(Vn, mass, L, evals, evecs, faces=faces, gradX=gradX, gradY=gradY)
        Z = self.vertex_bottleneck(Z)
        if add_noise:
            Z = Z + 0.01 * torch.randn_like(Z)
        out = self.pool(Z, mass, w.to(Z.dtype))
        if self.out_mode != "u":
            out = torch.cat([out[:, :1] + float(torch.log(R * self.unit_mm)), out[:, 1:]], dim=1)
        if return_per_vertex:
            return Z, out
        return out


class DualEncoderV3(EncoderV3):
    """``--head dual``: backbone comune, due proiezioni dalle stesse feature aggregate: z_F (``pool_proj``, loss di
    forma su GT-FR, come ctrlfr) e u (``pool_proj_u``, loss di forma su GT-SR, come factorized). forward -> [z_F, u]
    (``full``), z_F (``zf``) o u (``u``). pool_proj_u nasce in fork_rng: encoder e pool_proj identici a EncoderV3."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        with torch.random.fork_rng(devices=[]):
            self.pool_proj_u = nn.Linear(self.pool_proj.in_features, self.pool_proj.out_features)
        self.out_mode = "full"

    def pool(self, Z: torch.Tensor, mass: torch.Tensor, w_raw: torch.Tensor | None = None) -> torch.Tensor:
        w = (mass if w_raw is None else w_raw).reshape(-1).clamp_min(0)
        w = (w / w.sum().clamp_min(1e-30)).to(Z.dtype)
        z_mean = (w.unsqueeze(1) * Z).sum(dim=0, keepdim=True)
        if self.pooling == "area_meanmax":
            feat = torch.cat([z_mean, Z.max(dim=0, keepdim=True).values], dim=1)
        else:
            logits = self.attn(Z) + torch.log(w.clamp_min(1e-30)).unsqueeze(1)
            alpha = torch.softmax(logits, dim=0)
            feat = torch.cat([z_mean, (alpha.transpose(0, 1) @ Z).reshape(1, -1)], dim=1)
        if self.out_mode == "zf":
            return self.pool_proj(feat)
        if self.out_mode == "u":
            return self.pool_proj_u(feat)
        return torch.cat([self.pool_proj(feat), self.pool_proj_u(feat)], dim=1)


def build(args, device: torch.device) -> nn.Module:
    if str(getattr(args, "head", "factorized")) == "dual":
        md = DualEncoderV3(latent_dim=args.latent_dim, width=args.width, n_blocks=args.n_blocks, dropout=args.dropout,
                           pooling=str(args.pooling), attn_heads=int(getattr(args, "attn_heads", 1)),
                           area_weights=str(getattr(args, "area_weights", "mass")))
        return md.to(device)
    if str(getattr(args, "head", "factorized")) == "factorized2":
        m2 = FactorizedNormEncoderV3(latent_dim=args.latent_dim, width=args.width, n_blocks=args.n_blocks,
                                     dropout=args.dropout, pooling=str(args.pooling),
                                     attn_heads=int(getattr(args, "attn_heads", 1)),
                                     area_weights=str(getattr(args, "area_weights", "mass")),
                                     size_hidden=int(getattr(args, "size_hidden", 64)),
                                     unit_mm=float(getattr(args, "global_unit_mm", 100.0)))
        return m2.to(device)
    m = FactorizedEncoderV3(latent_dim=args.latent_dim, width=args.width, n_blocks=args.n_blocks, dropout=args.dropout,
                            pooling=str(args.pooling), attn_heads=int(getattr(args, "attn_heads", 1)),
                            area_weights=str(getattr(args, "area_weights", "mass")),
                            size_hidden=int(getattr(args, "size_hidden", 64)),
                            size_init=float(getattr(args, "size_init", 0.0)))
    return m.to(device)


def set_output(model: nn.Module, mode: str) -> None:
    """``full``, ``u`` (forma); ``eval`` = l'uscita dell'eval online (u per i fattorizzati, z_F per dual); ``zf`` solo
    dual. Nessun effetto sui modelli senza teste multiple."""
    if mode not in ("full", "u", "zf", "eval"):
        raise ValueError(f"uscita {mode!r} (full|u|zf|eval)")
    if isinstance(model, DualEncoderV3):
        model.out_mode = "zf" if mode == "eval" else mode
    elif isinstance(model, FactorizedEncoderV3):
        if mode == "zf":
            raise ValueError("uscita zf solo con --head dual")
        model.out_mode = "u" if mode == "eval" else mode


@contextlib.contextmanager
def output(model: nn.Module, mode: str):
    """Uscita temporanea (eval online: solo u); nessun effetto sui modelli non fattorizzati."""
    prev = getattr(model, "out_mode", None)
    set_output(model, mode)
    try:
        yield model
    finally:
        if prev is not None:
            model.out_mode = prev


# --- training ----------------------------------------------------------------------------------------

def load_log_cs(path) -> Dict[str, float]:
    """id del soggetto -> log S (mm). Formati: size.npz di build_factorized_targets (``names``, ``log_cs_mm``) o
    centroid_size_*.npz di E12 (``names`` come la GT, ``S`` in mm)."""
    import re
    with np.load(path, allow_pickle=True) as z:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["names"]]
        v = np.asarray(z["log_cs_mm"], np.float64) if "log_cs_mm" in z.files else np.log(np.asarray(z["S"], np.float64))
    ids = [re.search(r"(id\d+)", n, re.IGNORECASE).group(1).lower() for n in names]
    if not np.isfinite(v).all():
        raise ValueError(f"{path}: centroid size non finite")
    return dict(zip(ids, v.tolist()))


def parse_scale_aug(spec: str) -> Tuple[float, float] | None:
    if not spec:
        return None
    lo, hi = (float(x) for x in spec.split(","))
    if not 0 < lo <= 1 <= hi:
        raise SystemExit(f"--scale-aug {spec}: serve 0 < lo <= 1 <= hi")
    return lo, hi


def draw_log_scales(spec: str, n: int, seed: int, rank: int, step: int) -> np.ndarray | None:
    """log a per le n mesh del passo, log-uniforme in [log lo, log hi]; None se l'augmentation e' spenta."""
    r = parse_scale_aug(spec)
    if r is None:
        return None
    rng = np.random.default_rng([int(seed), 7_103, int(rank), int(step)])
    return rng.uniform(math.log(r[0]), math.log(r[1]), size=n)


def factorized_loss(args, batch: StepBatch, log_cs: Dict[str, float], log_a: np.ndarray | None):
    """Loss di forma su u + lambda_size * MSE(s, log S + log a)."""
    Z = batch.Z
    s, u = Z[:, 0], Z[:, 1:]
    loss, terms = compute_loss(args.loss, dataclasses.replace(batch, Z=u), args)
    t = np.asarray([log_cs[sid] for sid in batch.mesh_subjects], dtype=np.float64)
    if log_a is not None:
        t = t + log_a
    target = torch.as_tensor(t, dtype=Z.dtype, device=Z.device)
    err = s - target
    # --size-mask-domains: identita' con taglia inaffidabile (BFM REMESH, normalizzate per similarita' una per una)
    # fuori dalla MSE; s resta nel grafo (peso 0), u non cambia
    masked = {d for d in str(getattr(args, "size_mask_domains", "")).split(",") if d}
    w = torch.as_tensor([0.0 if domain_of(sid) in masked else 1.0 for sid in batch.mesh_subjects],
                        dtype=Z.dtype, device=Z.device)
    l_size = (w * err ** 2).sum() / w.sum().clamp_min(1.0)
    loss = loss + float(args.lambda_size) * l_size
    terms = dict(terms)
    terms["size_mse"] = float(l_size.item())
    terms["size_mae"] = float(((w * err.abs()).sum() / w.sum().clamp_min(1.0)).item())
    return loss, terms


def dual_loss(args, batch: StepBatch, gt_shape, n2i_shape):
    """lambda_form x loss di forma su z_F (GT di --dist_npz, FR) + lambda_shape x loss su u (GT-SR, --dist-npz-shape)."""
    L = batch.Z.shape[1] // 2
    lf, tf = compute_loss(args.loss, dataclasses.replace(batch, Z=batch.Z[:, :L]), args)
    ls, ts = compute_loss(args.loss, dataclasses.replace(batch, Z=batch.Z[:, L:], gt=gt_shape, name_to_idx=n2i_shape), args)
    terms = dict(tf)
    terms.update({f"shape_{k}": v for k, v in ts.items() if k in ("stress", "rank", "id")})
    terms["form_loss"], terms["shape_loss"] = float(lf.item()), float(ls.item())
    return float(args.lambda_form) * lf + float(args.lambda_shape) * ls, terms


def model_distances(Z, i, j, head: str, dp_per_unit: float = float("nan")) -> Dict[str, np.ndarray]:
    """Distanze del modello per le coppie (i, j) di righe di Z, secondo la testa: fattorizzati -> form (d_F, mm) e
    shape (d_P); dual -> zf e u; altrimenti z."""
    Z = np.asarray(Z, dtype=np.float64)
    if head in FACTORIZED:
        d = pair_distances(Z, i, j, dp_per_unit)
        return {"form": d["form_mm"], "shape": d["dP"]}
    if head == "dual":
        L = Z.shape[1] // 2
        return {"zf": np.linalg.norm(Z[i, :L] - Z[j, :L], axis=1), "u": np.linalg.norm(Z[i, L:] - Z[j, L:], axis=1)}
    return {"z": np.linalg.norm(Z[i] - Z[j], axis=1)}


# --- valutazione ---------------------------------------------------------------------------------------

def split(Z: np.ndarray | torch.Tensor):
    """z = (s, u): s [n], u [n, d]."""
    return Z[:, 0], Z[:, 1:]


def form_distance(S_i, S_j, dP):
    """d_F = sqrt((S_i - S_j)^2 + S_i S_j dP^2), nelle unita' di S (mm)."""
    return np.sqrt((np.asarray(S_i) - np.asarray(S_j)) ** 2 + np.asarray(S_i) * np.asarray(S_j) * np.asarray(dP) ** 2)


def pair_distances(Z: np.ndarray, i: np.ndarray, j: np.ndarray, dp_per_unit: float) -> Dict[str, np.ndarray]:
    """Per le coppie (i, j) di righe di Z = [s, u]: form (mm), dP, rho (rad), |delta s|, ||delta u||."""
    Z = np.asarray(Z, dtype=np.float64)
    s, u = split(Z)
    du = np.linalg.norm(u[i] - u[j], axis=1)
    dP = du * float(dp_per_unit)
    S = np.exp(s)
    rho = 2.0 * np.arcsin(np.clip(dP / 2.0, 0.0, 1.0))
    return {"form_mm": form_distance(S[i], S[j], dP), "dP": dP, "rho_rad": rho, "size_abs": np.abs(s[i] - s[j]),
            "u": du}
