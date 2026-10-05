#!/usr/bin/env python3
"""Aggancia al percorso congelato v1 i due bracci del pilota pot, senza toccarne il codice.

Due difetti dell'implementazione esistente del pooling mascherato (v2_work/fastio + v2_work/masked),
trovati preparando il braccio pot_m55, che non e' mai girato. Tutti e due SILENZIOSI: il braccio
mascherato sarebbe stato il braccio col solo pozzo, con un altro nome.

1. `robustness.data_utils.sample_to_device` costruisce un dict NUOVO con le 8 chiavi fisse
   (verts, mass, L, evals, evecs, faces, gradX, gradY): `roi_mask` non arriva mai a
   forward_model, `set_roi_mask(None)` e DiffusionEncoderOnlyMasked torna al pooling pieno.
2. `fast_data.CachedDataset` chiama `_with_roi(sample, self.files[i])` con il NOME del file,
   non il percorso: `np.load` relativo alla cwd fallisce, l'OSError e' tollerato apposta, e
   la maschera non viene attaccata (il campione sonda, poi, non passa nemmeno da _with_roi).

Qui: `sample_to_device` porta anche le chiavi extra, il dataset attacca `roi_mask` col percorso
completo, e forward_model FALLISCE se un modello mascherato riceve un campione senza maschera o
il modello a due rami un campione senza operatori del pozzo, invece di proseguire.

Lo stesso dataset attacca gli operatori del secondo ramo (POT_KEYS) letti da una seconda
cartella con lo stesso loader congelato, quindi con la stessa normalizzazione che vedrebbe un
braccio addestrato solo su quella cartella.

    from pilot_hooks import install
    install(pot_dir=..., dual=True)        # prima di importare/lanciare trainer o script di eval
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in ("face_embedding/gt_encdec/remeshing/intrinsic", "face_embedding/gt_encdec/autoencoder",
           "diffusion-net/src", "v2_work/fastio"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))
sys.path.insert(0, str(THIS_DIR))

from dataset_gtready import GTReadyDatasetNPZ  # noqa: E402
from dn_dual_ops import POT_KEYS, DiffusionEncoderDualOps  # noqa: E402

POT_SPARSE = ("pot_gradX", "pot_gradY")
EXTRA_KEYS = ("roi_mask",) + POT_KEYS

_STATE = {"pot_dir": None, "dual": False, "require_roi": False, "logged": set()}


class PilotDataset:
    """GTReadyDatasetNPZ di `data_dir` + roi_mask (se c'e' nell'npz) + operatori di `pot_dir`.

    Stessa superficie che il trainer e gli script di eval toccano: `files`, `__len__`,
    `__getitem__`, `data_dir`.
    """

    def __init__(self, data_dir, pot_dir=None, verbose=False):
        self._base = GTReadyDatasetNPZ(str(data_dir), verbose=verbose)
        self.files = self._base.files
        self.data_dir = str(data_dir)
        self._pot = None
        if pot_dir is not None:
            self._pot = GTReadyDatasetNPZ(str(pot_dir), verbose=verbose)
            pot_idx = {f: i for i, f in enumerate(self._pot.files)}
            missing = [f for f in self.files if f not in pot_idx]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)}/{len(self.files)} file di {data_dir} senza operatori del pozzo "
                    f"in {pot_dir} (primo: {missing[0]})")
            self._pot_idx = pot_idx
            print(f"[pilot] dataset a due operatori: {data_dir} + {pot_dir}", flush=True)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[int(idx)]
        s = dict(self._base[int(idx)])
        with np.load(os.path.join(self.data_dir, name), allow_pickle=False) as z:
            if "roi_mask" in z.files:
                s["roi_mask"] = torch.as_tensor(z["roi_mask"], dtype=torch.float32)
        if self._pot is not None:
            p = self._pot[self._pot_idx[name]]
            # Stessi vertici (stessa mesh, stessa normalizzazione del loader): altrimenti i due
            # rami parlerebbero di vertici diversi.
            if p["verts"].shape != s["verts"].shape or not torch.equal(p["verts"], s["verts"]):
                raise ValueError(f"{name}: vertici diversi fra {self.data_dir} e la cartella del pozzo")
            for k in POT_KEYS:
                s[k] = p[k[len("pot_"):]]
        return s


def sample_to_device_factory(orig):
    def sample_to_device(sample: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
        out = orig(sample, device=device)
        for k in EXTRA_KEYS:
            t = sample.get(k)
            if torch.is_tensor(t):
                out[k] = t.to(device)
        return out
    sample_to_device._pilot = True
    return sample_to_device


def _log_once(key: str, msg: str) -> None:
    if key not in _STATE["logged"]:
        _STATE["logged"].add(key)
        print(msg, flush=True)


def _rebind(name: str, obj) -> None:
    """Rebind `name` in ogni modulo robustness/perturbated gia' importato che lo aveva copiato."""
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("robustness") or mod_name.startswith("compare_model_vs_chamfer")
                or mod_name in ("eval_by_topology", "__main__")):
            continue
        if hasattr(mod, name):
            setattr(mod, name, obj)


def install(pot_dir: str | Path | None = None, dual: bool = False, require_roi: bool = False) -> None:
    """Installa dataset, sample_to_device e (se dual) build/forward del modello a due rami.

    Va chiamata PRIMA che gli script importino i nomi (`from robustness.data_utils import ...`
    li copia); per i moduli gia' importati i nomi vengono rebindati qui.
    """
    import fast_data as fd
    import robustness.data_utils as du
    import robustness.model_helpers as mh

    if dual and pot_dir is None:
        raise ValueError("il modello a due rami vuole --dual-ops-dir")
    _STATE.update(pot_dir=None if pot_dir is None else str(pot_dir), dual=bool(dual),
                  require_roi=bool(require_roi))

    factory = lambda data_dir, *a, **k: PilotDataset(data_dir, pot_dir=_STATE["pot_dir"])  # noqa: E731
    du.GTReadyDataset = factory
    _rebind("GTReadyDataset", factory)
    fd.GTReadyDatasetNPZ = factory          # base del CachedDataset del training
    # Le chiavi extra devono anche essere pinnate/contate e, se sparse, ricostruite sotto
    # inference_mode come quelle standard (docstring di fast_data._rebuild_sparse).
    fd.TENSOR_KEYS = tuple(fd.TENSOR_KEYS) + tuple(k for k in EXTRA_KEYS if k not in fd.TENSOR_KEYS)
    fd.SPARSE_KEYS = tuple(fd.SPARSE_KEYS) + tuple(k for k in POT_SPARSE if k not in fd.SPARSE_KEYS)

    if not getattr(du.sample_to_device, "_pilot", False):
        stod = sample_to_device_factory(du.sample_to_device)
        du.sample_to_device = stod
        _rebind("sample_to_device", stod)

    orig_build, orig_forward = mh.build_model, mh.forward_model

    def build_model(args, device):
        if not _STATE["dual"]:
            return orig_build(args, device)
        if getattr(args, "model", None) != "xyz_dn":
            raise ValueError("il modello a due rami e' agganciato solo a --model xyz_dn")
        if getattr(args, "use_smooth_loss", False):
            raise ValueError("smooth loss non agganciata al modello a due rami")
        m = DiffusionEncoderDualOps(latent_dim=args.latent_dim, width=args.width,
                                    n_blocks=args.n_blocks, dropout=args.dropout,
                                    pool_mode=args.pool_mode).to(device)
        print(f"[pilot] DiffusionEncoderDualOps: {sum(p.numel() for p in m.parameters())} parametri, "
              f"width {args.width}", flush=True)
        return m

    def forward_model(model, sample_dict, V_in, return_gate_info, add_noise):
        if isinstance(model, DiffusionEncoderDualOps):
            missing = [k for k in POT_KEYS if k not in sample_dict]
            if missing:
                raise RuntimeError(f"modello a due rami senza operatori del pozzo: mancano {missing}")
            _log_once("dual", f"[pilot] forward a due rami attivo: {sample_dict['pot_evecs'].shape[0]} "
                              f"vertici, evals pozzo [{float(sample_dict['pot_evals'][1]):.3g} .. 1]")
            z = model(V_in, sample_dict["mass"], sample_dict["evals"], sample_dict["evecs"],
                      sample_dict["gradX"], sample_dict["gradY"],
                      pot_ops=tuple(sample_dict[k] for k in POT_KEYS),
                      return_per_vertex=False, add_noise=add_noise)
            return z, mh._default_gate_info(z)
        if _STATE["require_roi"]:
            # Chiamata da dentro il forward_model di install_masked_pooling, dopo set_roi_mask.
            roi = getattr(model, "_roi_mask", None)
            if roi is None:
                raise RuntimeError("pooling mascherato senza roi_mask nel campione: il braccio "
                                   "sarebbe il solo pozzo. Gli operatori in --data_dir hanno roi_mask?")
            keep = int((roi > model.roi_threshold).sum())
            _log_once("roi", f"[pilot] pooling mascherato attivo: {keep}/{roi.numel()} vertici nella ROI")
        return orig_forward(model, sample_dict, V_in, return_gate_info, add_noise)

    mh.build_model, mh.forward_model = build_model, forward_model
    _rebind("build_model", build_model)
    _rebind("forward_model", forward_model)
    print(f"[pilot] hook installati: pot_dir={_STATE['pot_dir']} dual={dual} "
          f"require_roi={require_roi}", flush=True)
