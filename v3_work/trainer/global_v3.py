"""Ingresso a scala globale (``--input-norm global``): vertici in mm nel frame canonico, divisi per UNA costante.

Perche': maxabs e sqrt(area) normalizzano ogni mesh per la propria taglia, quindi il modello non vede la dimensione
del volto, che nei dati metrici e' un tratto identitario. Qui la taglia resta nell'ingresso:

    X = ((V - c) * f) @ R_d^T / L0
  * V: vertici serviti dal loader congelato (centro dei vertici e maxabs per mesh, come sempre);
  * f = sqrt(area_mm2 / area(V)): riporta V in mm (area_mm2 = s_d^2 * area della geometria grezza, dalla tabella
    di tools/build_scale_table.py; l'area non dipende da centro e rotazione, quindi f non dipende da come il loader
    ha centrato e scalato);
  * c: baricentro pesato con i pesi di ``--area-weights`` (``smooth`` con --area robust: aree della geometria
    passa-basso, come arearobust; ``mass``: la massa degli operatori);
  * R_d: rotazione del dominio verso il frame canonico (FLAME: +y alto, +z fuori dal volto); la conversione delle
    unita' u_d e' gia' in area_mm2. Entrambe dal frame di GT-F di E12 (aau/runs/evidence/e12/frames.json): u_d = unita'
    fisica (o 63 mm / IPD della media per HIFI3D e FaceVerse), R_d = la rotazione di canonical_transforms.json; NON
    la scala s_d del json, che porterebbe la media di ogni dominio sulla taglia della media FLAME;
  * L0 = ``--global-unit-mm`` (100): la stessa per tutte le mesh, quindi una faccia piu' grande ha coordinate piu'
    grandi. Le perturbazioni del training arrivano dopo, con le stesse sigma: sono in unita' di L0 (0.05-2 mm).

Operatori (``--global-ops``): il loader congelato divide gli autovalori per lambda_max e i gradienti per
sqrt(lambda_max), e gli operatori dello store sono calcolati ad area 1: diffusione spettrale, gradienti e pesi del
pooling sono gia' adimensionali, l'unica grandezza con la scala e' xyz.
  * ``areanorm`` (default): operatori come serviti;
  * ``mm``: massa e autovettori riportati alle unita' di X (massa * k, autovettori / sqrt(k), k = area_mm2 / L0^2 /
    somma della massa: resta l'M-ortonormalita'). In aritmetica esatta l'embedding e' lo stesso (tests/
    test_factorized.py lo misura): il flag documenta l'invarianza, non cambia il modello.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

import area_v3
from common import REPO_ROOT

FRAMES = REPO_ROOT / "aau/runs/evidence/e12/frames.json"   # u_d, R_d, t_d di GT-F (E12)
GLOBAL_OPS = ("areanorm", "mm")
# patch di valutazione gia' nel frame del template T7 di NoW (aau/recon/now_prepare_meshes.py: mm, +y alto, +z naso,
# il frame canonico): nessuna rotazione; la scala e' nelle tabelle (FaMoS metrica, NoW quella del template)
EXTRA_FRAMES = {"famos": {"u": 1.0, "R": np.eye(3).tolist(), "unit_source": "patch T7 metriche (scala annullata)"},
                "now": {"u": 1.0, "R": np.eye(3).tolist(), "unit_source": "patch T7 alla taglia del template"}}


def rotations() -> Dict[str, np.ndarray]:
    """dominio -> R_d (3x3, float64)."""
    doms = {**json.loads(FRAMES.read_text())["domains"], **EXTRA_FRAMES}
    return {d: np.asarray(v["R"], dtype=np.float64) for d, v in doms.items()}


class ScaleTable:
    """Nome del file della mesh -> (area in mm^2 nel frame canonico, dominio); una o piu' tabelle npz."""

    def __init__(self, paths: Iterable[str | Path]) -> None:
        self.area: Dict[str, float] = {}
        self.domain: Dict[str, str] = {}
        self.paths = [str(p) for p in paths]
        for p in self.paths:
            with np.load(p) as z:
                for n, a, d in zip(z["names"], z["area_mm2"], z["domain"]):
                    n = str(n)
                    if n in self.area and (self.area[n] != float(a) or self.domain[n] != str(d)):
                        raise ValueError(f"{n}: valori diversi in due tabelle di scala")
                    self.area[n], self.domain[n] = float(a), str(d)
        if not self.area:
            raise ValueError(f"tabelle di scala vuote: {self.paths}")

    def __contains__(self, name: str) -> bool:
        return name in self.area

    def lookup(self, name: str, group: str = "") -> tuple[float, str]:
        """Prima ``<group>/<name>`` (nomi ripetuti fra cartelle, es. i metodi di NoW), poi ``name``."""
        if group and f"{group}/{name}" in self.area:
            name = f"{group}/{name}"
        if name not in self.area:
            raise KeyError(f"{name}: assente dalle tabelle di scala {self.paths} (tools/build_scale_table.py)")
        return self.area[name], self.domain[name]


def center_weights(mode: str, V: torch.Tensor, F: torch.Tensor, mass: torch.Tensor, evecs: torch.Tensor) -> torch.Tensor:
    """Pesi del centro: ``mass`` = la massa degli operatori (come reframe_sqrt_area), altrimenti area_v3."""
    if mode == "mass":
        return mass.reshape(-1).to(V.dtype).clamp_min(0)
    return area_v3.area_weights(mode, V, F, mass, evecs)


def frame_params(V: torch.Tensor, F: torch.Tensor, mass: torch.Tensor, evecs: torch.Tensor, area_mm2: float,
                 weights: str) -> tuple[torch.Tensor, float]:
    """(centro [1, 3] float64, fattore mm per unita' di V) dalla mesh pulita servita."""
    Vd = V.detach().double()
    w = center_weights(weights, Vd, F, mass.double(), evecs.double())
    tot = w.sum()
    if not torch.isfinite(tot) or float(tot) <= 0:
        raise ValueError("pesi degeneri: impossibile il centro per area")
    c = (w.unsqueeze(1) * Vd).sum(0, keepdim=True) / tot
    tri = Vd[F.long()]
    area = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1).sum()
    if not float(area) > 0 or not np.isfinite(area_mm2) or area_mm2 <= 0:
        raise ValueError(f"area degenere: servita {float(area)}, tabella {area_mm2}")
    return c, float(np.sqrt(area_mm2 / float(area)))


def apply(V: torch.Tensor, c: torch.Tensor, f: float, R: np.ndarray, unit_mm: float) -> torch.Tensor:
    """X = ((V - c) f) R^T / L0, calcolato in float64 e restituito nel dtype di V."""
    Rt = torch.as_tensor(R.T, dtype=torch.float64, device=V.device)
    X = ((V.double() - c.to(V.device)) * (f / float(unit_mm))) @ Rt
    return X.to(V.dtype)


def ops_to_mm(mass: torch.Tensor, evecs: torch.Tensor, area_mm2: float, unit_mm: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Massa e autovettori nelle unita' di X (M-ortonormalita' conservata)."""
    k = float(area_mm2) / float(unit_mm) ** 2 / float(mass.double().sum())
    return mass * k, evecs / float(np.sqrt(k))


class GlobalFrame:
    """La trasformazione al servizio (data_v3.ServeTransform), con centro e fattore memorizzati per mesh."""

    def __init__(self, table: ScaleTable, unit_mm: float, weights: str, ops: str = "areanorm") -> None:
        if ops not in GLOBAL_OPS:
            raise ValueError(f"--global-ops {ops!r}")
        self.table, self.unit_mm, self.weights, self.ops = table, float(unit_mm), str(weights), ops
        self.R = rotations()
        self._cache: dict = {}

    def __call__(self, name: str, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        area_mm2, dom = self.table.lookup(name)
        fr = self._cache.get(name)
        if fr is None:
            fr = frame_params(sample["verts"], sample["faces"], sample["mass"], sample["evecs"], area_mm2, self.weights)
            self._cache[name] = fr
        sample["verts"] = apply(sample["verts"], fr[0], fr[1], self.R[dom], self.unit_mm)
        if self.ops == "mm":
            sample["mass"], sample["evecs"] = ops_to_mm(sample["mass"], sample["evecs"], area_mm2, self.unit_mm)
        return sample
