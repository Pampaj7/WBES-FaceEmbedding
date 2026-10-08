"""Loss del trainer v3, una funzione per variante (``--loss``), tutte con la stessa firma.

``v2`` (default): la loss di train_runner._train_epoch_mixed riga per riga, con le funzioni congelate v1:
    lambda_subject (S_subj + lambda_rank R_subj) + lambda_mesh (S_mesh + lambda_rank R_mesh) + lambda_id L_id
(ricetta: 1, 0.5, 1, 0.25). Stesso ordine dei termini, quindi stesso ordine delle estrazioni CUDA dei due
rank (prima quello dei soggetti, poi quello delle mesh). I termini smooth e teacher di v1 valgono 0.0 e sono
omessi: x + 0.0 == x.

Varianti log (PLAN_MASSIVE §6 con le correzioni del critic dell'8 ottobre). Notazione: d distanza latente fra
mesh, g la GT fra le loro identita', coppie "diverse" = identita' diverse, "stesse" = stessa identita',
mesh diverse.
  * L_grad = Huber_delta(log(d+eps) - log(g+eps_g) - b) sulle coppie diverse, b = mediana dei residui
    CON gradiente: la loss e' esattamente invariante alla scala globale degli embedding (nessuna spinta a
    collassare, ma nemmeno un'ancora). Coppie fra domini (batch misti) pesate ``w_cross``.
  * L_scale = (log mean_diverse(d) - log s0)^2: ANCORA DI SCALA esplicita. Senza, con b e i denominatori
    senza gradiente la distanza mediana collassa (verificato dal critic su un modello giocattolo).
  * L_inv = mean_stesse(d^2) / mean_diverse(d^2), denominatore CON gradiente (invariante alla scala).
  * L_nbr (spento di default): KL(P_m || Q_m), P_m(n) ~ exp(-g^2/2 sigma_m^2), Q_m(n) ~ exp(-d^2/2 tau_m^2),
    tau_m = e^b sigma_m, sigma_m = GT dal terzo vicino fra le identita' DIVERSE (non le viste sorelle, che
    hanno g = 0 e lo renderebbero degenere).
    log       = L_grad + l_scale L_scale
    log+inv   = log + l_inv L_inv
    log+inv+nbr = log+inv + l_nbr L_nbr
Ogni variante log va accompagnata dal test anti-collasso (tests/anti_collapse.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import common  # noqa: F401

from common import domain_of
from intrinsic_utils import pairwise_rank_loss  # noqa: E402
from latent_loss import stress_loss  # noqa: E402
from robustness.train_runner import (  # noqa: E402
    _masked_stress_loss_from_pair_mask,
    _pair_mask_from_metadata,
    _pair_observation_rank_loss,
)

LOSSES = ("v2", "log", "log+inv", "log+inv+nbr")


@dataclass
class StepBatch:
    """Gli embedding di un passo e i loro metadati, nell'ordine in cui v2 li accumulava."""
    Z: torch.Tensor                 # [M, D]
    mesh_subjects: List[str]        # soggetto di ogni mesh
    mesh_topos: List[str]           # etichetta di topologia di ogni mesh
    batch_subjects: List[str]       # ordine dei soggetti del batch (plan.subjects)
    gt: object                      # matrice (NanGuardedMatrix) o VectorGT
    name_to_idx: Dict[str, int]

    def subject_rows(self) -> Dict[str, List[int]]:
        """soggetto -> righe di Z, nell'ordine d'inserimento di per_subject_latents di v2."""
        out: Dict[str, List[int]] = {}
        for i, s in enumerate(self.mesh_subjects):
            out.setdefault(s, []).append(i)
        return out

    def gt_block(self, ids: Sequence[str], dtype, device) -> torch.Tensor:
        idx = np.asarray([self.name_to_idx[s] for s in ids], dtype=int)
        return torch.tensor(self.gt[np.ix_(idx, idx)], device=device, dtype=dtype)


def loss_v2(b: StepBatch, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    Z, device = b.Z, b.Z.device
    rows = b.subject_rows()
    per_subject = {s: Z.index_select(0, torch.as_tensor(r, device=device)) for s, r in rows.items()}
    subj_means, subj_ids = [], []
    for sid in b.batch_subjects:
        Zs = per_subject.get(str(sid))
        if Zs is None or b.name_to_idx.get(str(sid)) is None:
            continue
        subj_means.append(Zs.mean(dim=0))
        subj_ids.append(str(sid))
    if len(subj_means) < 2:
        raise RuntimeError("batch con meno di 2 soggetti validi: il piano doveva saltarlo")
    lam_r = float(args.lambda_rank)

    Z_subject = torch.stack(subj_means, dim=0)
    D_subject = b.gt_block(subj_ids, Z_subject.dtype, device)
    s_subj = stress_loss(Z_subject, D_subject)
    if lam_r > 0.0:
        r_subj = pairwise_rank_loss(torch.cdist(Z_subject, Z_subject, p=2), D_subject, n_pairs=int(args.rank_pairs),
                                    margin=float(args.rank_margin), tau=float(args.rank_tau),
                                    hard_frac=float(args.rank_hard_frac))
    else:
        r_subj = torch.tensor(0.0, device=device)

    D_mesh = b.gt_block(b.mesh_subjects, Z.dtype, device)
    D_mesh_lat = torch.cdist(Z, Z, p=2)
    pair_mask = _pair_mask_from_metadata(subject_ids=b.mesh_subjects, topology_labels=b.mesh_topos,
                                         device=device, pair_mode=str(args.train_pair_mode))
    s_mesh = _masked_stress_loss_from_pair_mask(D_lat=D_mesh_lat, D_gt=D_mesh, pair_mask=pair_mask)
    if lam_r > 0.0:
        tri = torch.triu(pair_mask, diagonal=1)
        r_mesh = _pair_observation_rank_loss(lat_values=D_mesh_lat[tri], gt_values=D_mesh[tri],
                                             n_pairs=int(args.rank_pairs), margin=float(args.rank_margin),
                                             tau=float(args.rank_tau), hard_frac=float(args.rank_hard_frac))
    else:
        r_mesh = torch.tensor(0.0, device=device)

    if args.use_id_loss:
        id_terms, fallback = [], []
        for s, Zs in per_subject.items():
            if Zs.shape[0] < 2:
                continue
            term = ((Zs - Zs.mean(dim=0).unsqueeze(0)) ** 2).mean()
            topos = {b.mesh_topos[i] for i in rows[s]}
            (id_terms if len(topos) >= 2 else fallback).append(term)
        chosen = id_terms if id_terms else fallback
        l_id = torch.stack(chosen).mean() if chosen else torch.tensor(0.0, device=device)
    else:
        l_id = torch.tensor(0.0, device=device)

    lam_s, lam_m = float(args.lambda_subject), float(args.lambda_mesh)
    loss = lam_s * (s_subj + args.lambda_rank * r_subj) + lam_m * (s_mesh + args.lambda_rank * r_mesh) \
        + args.lambda_id * l_id
    terms = {"subject_stress": float(s_subj.item()), "subject_rank": float(r_subj.item()),
             "mesh_stress": float(s_mesh.item()), "mesh_rank": float(r_mesh.item()), "id": float(l_id.item())}
    terms["stress"] = lam_s * terms["subject_stress"] + lam_m * terms["mesh_stress"]
    terms["rank"] = lam_s * terms["subject_rank"] + lam_m * terms["mesh_rank"]
    return loss, terms


def _pair_masks(b: StepBatch, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    subj = np.asarray(b.mesh_subjects, dtype=object)
    same = subj[:, None] == subj[None, :]
    off = ~np.eye(len(subj), dtype=bool)
    dom = np.asarray([domain_of(s) for s in b.mesh_subjects], dtype=object)
    cross_dom = dom[:, None] != dom[None, :]
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return t(off & ~same), t(off & same), t(cross_dom)


def loss_log(b: StepBatch, args, inv: bool, nbr: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    Z, device = b.Z, b.Z.device
    diff_m, same_m, xdom_m = _pair_masks(b, device)
    if args.log_pair_mode == "cross_topology":
        topo = np.asarray(b.mesh_topos, dtype=object)
        diff_m = diff_m & torch.as_tensor(topo[:, None] != topo[None, :], device=device)
    tri = torch.triu(torch.ones_like(diff_m), diagonal=1)
    diff_t = diff_m & tri
    if int(diff_t.sum()) < 2:
        raise RuntimeError("meno di 2 coppie di identita' diverse nel batch")
    sq = ((Z.unsqueeze(1) - Z.unsqueeze(0)) ** 2).sum(-1)
    d = torch.sqrt(sq + 1e-12)
    G = b.gt_block(b.mesh_subjects, Z.dtype, device)
    eps, eps_g = float(args.log_eps), float(args.log_eps_gt)

    r = torch.log(d[diff_t] + eps) - torch.log(G[diff_t] + eps_g)
    off_b = torch.median(r)                               # con gradiente: invarianza di scala esatta
    w = torch.where(xdom_m[diff_t], float(args.w_cross), 1.0).to(Z.dtype)
    l_grad = (w * F.huber_loss(r - off_b, torch.zeros_like(r), reduction="none",
                               delta=float(args.log_huber_delta))).sum() / w.sum()
    mean_d = d[diff_t].mean()
    l_scale = (torch.log(mean_d) - float(np.log(args.scale_target))) ** 2
    loss = l_grad + float(args.lambda_scale) * l_scale
    terms = {"grad": float(l_grad.item()), "scale": float(l_scale.item()), "b": float(off_b.item()),
             "d_mean": float(mean_d.item())}

    if inv:
        same_t = same_m & tri
        if bool(same_t.any()):
            l_inv = sq[same_t].mean() / sq[diff_t].mean()
        else:
            l_inv = torch.zeros((), device=device)
        loss = loss + float(args.lambda_inv) * l_inv
        terms["inv"] = float(l_inv.item())

    if nbr:
        l_nbr = _nbr_kl(b, sq, G, off_b, diff_m, device)
        loss = loss + float(args.lambda_nbr) * l_nbr
        terms["nbr"] = float(l_nbr.item())
    return loss, terms


def _nbr_kl(b: StepBatch, sq: torch.Tensor, G: torch.Tensor, off_b: torch.Tensor, diff_m: torch.Tensor,
            device) -> torch.Tensor:
    """KL(P_m||Q_m) media sulle ancore; sigma_m dal terzo vicino fra le identita' DIVERSE."""
    ids = list(dict.fromkeys(b.mesh_subjects))
    Gs = b.gt_block(ids, G.dtype, device)                       # [S, S] fra identita'
    S = len(ids)
    k = min(3, S - 1)
    off = ~torch.eye(S, dtype=torch.bool, device=device)
    Gs_masked = torch.where(off, Gs, torch.full_like(Gs, float("inf")))
    sigma_s = torch.sort(Gs_masked, dim=1).values[:, k - 1].clamp_min(1e-6)   # [S]
    pos = {s: i for i, s in enumerate(ids)}
    sigma = sigma_s[torch.as_tensor([pos[s] for s in b.mesh_subjects], device=device)]   # [M]
    tau = torch.exp(off_b) * sigma
    M = sq.shape[0]
    eye = torch.eye(M, dtype=torch.bool, device=device)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=sq.dtype)
    logit_p = torch.where(eye, neg_inf, -(G ** 2) / (2 * sigma.unsqueeze(1) ** 2))
    logit_q = torch.where(eye, neg_inf, -sq / (2 * tau.unsqueeze(1) ** 2))
    logp = torch.log_softmax(logit_p, dim=1)
    logq = torch.log_softmax(logit_q, dim=1)
    p = logp.exp()
    kl = torch.where(eye, torch.zeros_like(p), p * (logp - logq)).sum(dim=1)
    return kl.mean()


def compute_loss(name: str, b: StepBatch, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    if name == "v2":
        return loss_v2(b, args)
    if name == "log":
        return loss_log(b, args, inv=False, nbr=False)
    if name == "log+inv":
        return loss_log(b, args, inv=True, nbr=False)
    if name == "log+inv+nbr":
        return loss_log(b, args, inv=True, nbr=True)
    raise ValueError(f"--loss {name!r} non in {LOSSES}")
