#!/usr/bin/env python3
"""Aggancia al percorso congelato v1 il token di taglia (ablazioni B, E) e il frame d'ingresso in
eval, senza toccarne il codice. Stesso schema di aau/models/pilot_hooks.py.

Token di taglia. Il dataset attacca a ogni campione `size_token`, shape (1,): il log del raggio
rms della mesh GREZZA, standardizzato con media e std del training (aau/models/size_token.py, una
tabella JSON per collezione, chiave = nome del file senza .npz). Il modello e'
DiffusionEncoderSizeToken: DiffusionEncoderOnly riga per riga, ma la proiezione dopo il pooling
mean+max riceve [Z_mean, Z_max, token], 2*latent+1 -> latent. Il token arriva al modello senza
perturbazioni: il rumore del trainer tocca solo i vertici.

Tre punti da cui il token sparirebbe in silenzio, e cosa fa qui ciascuno:
1. `robustness.data_utils.sample_to_device` ricostruisce il dict con 8 chiavi fisse: la versione
   agganciata porta anche `size_token`.
2. `fast_data.CachedDataset` legge da `fast_data.GTReadyDatasetNPZ`: rebindato al dataset col
   token, e `size_token` aggiunto a TENSOR_KEYS (pinnato con gli altri).
3. forward_model FALLISCE se il modello col token riceve un campione senza token, e il dataset
   fallisce se un file non e' nella tabella: niente token di default.

Frame in eval. In training il frame lo applica train_fast.install_frame sulla cache. Gli script
di eval del repo (compare_model_vs_chamfer_*) danno al modello i vertici del loader (maxabs):
con `eval_frame` il forward_model agganciato ri-inquadra V_in prima del modello. Il ri-inquadramento
e' idempotente (frames.py: il risultato non dipende dal frame d'ingresso), quindi passare anche
--frame rms a eval_by_topology non cambia niente. SOLO per input puliti: su vertici perturbati
annullerebbe traslazione e scala della perturbazione (eval_ablation.py rifiuta scenari non clean).

    from ablation_hooks import install
    install(size_token_json=..., eval_frame="rms")   # prima di importare/lanciare trainer o eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in ("face_embedding/gt_encdec/remeshing/intrinsic", "face_embedding/gt_encdec/autoencoder",
           "diffusion-net/src", "v2_work/fastio", "v2_work/pointnet"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from dataset_gtready import GTReadyDatasetNPZ  # noqa: E402
from diffusion_autoencoder import DiffusionEncoderOnly  # noqa: E402

SIZE_KEY = "size_token"

_STATE = {"table": None, "eval_frame": None, "logged": set()}


class SizeTokenTable:
    """log r per nome di file e standardizzazione del training, da size_token.py."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = json.loads(self.path.read_text())
        self.log_r: Dict[str, float] = payload["log_r"]
        self.mean = float(payload["train"]["mean"])
        self.std = float(payload["train"]["std"])
        self.seed = payload["train"].get("seed")
        self.collection = payload["collection"]
        if not self.std > 0:
            raise ValueError(f"{path}: std del training non positiva ({self.std})")

    def token(self, name: str) -> float:
        try:
            v = self.log_r[name]
        except KeyError:
            raise KeyError(f"{name} non e' nella tabella dei token {self.path}") from None
        return (float(v) - self.mean) / self.std


class SizeTokenDataset:
    """GTReadyDatasetNPZ di `data_dir` + `size_token`. Superficie: files, __len__, __getitem__, data_dir."""

    def __init__(self, data_dir, table: SizeTokenTable, verbose=False):
        self._base = GTReadyDatasetNPZ(str(data_dir), verbose=verbose)
        self.files = self._base.files
        self.data_dir = str(data_dir)
        self._table = table
        missing = [f for f in self.files if f[:-len(".npz")] not in table.log_r]
        if missing:
            raise KeyError(f"{len(missing)}/{len(self.files)} file di {data_dir} senza token in "
                           f"{table.path} (primo: {missing[0]})")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        s = dict(self._base[int(idx)])
        s[SIZE_KEY] = torch.tensor([self._table.token(self.files[int(idx)][:-len(".npz")])],
                                   dtype=torch.float32)
        return s


class DiffusionEncoderSizeToken(DiffusionEncoderOnly):
    """DiffusionEncoderOnly con il token di taglia concatenato dopo il pooling mean+max."""

    def __init__(self, latent_dim=256, width=128, n_blocks=4, dropout=0.1, pool_mode="meanmax"):
        if str(pool_mode) != "meanmax":
            raise ValueError("il token di taglia e' agganciato solo a pool_mode meanmax")
        super().__init__(latent_dim=latent_dim, width=width, n_blocks=n_blocks, dropout=dropout,
                         pool_mode=pool_mode)
        # Partenza IDENTICA al controllo: stessi pesi della proiezione mean+max e colonna del token a
        # zero, e il generatore CPU non avanza (fork_rng), quindi tutto quello che il trainer estrae
        # dopo e' lo stesso. Con l'init di default la colonna del token sposta z di ~|w|*2.5 fra crop e
        # original dello stesso soggetto gia' al passo zero (crop = -2.5 std di token): smoke 1055097,
        # loss all'epoca 1 0.266 contro 0.080 del controllo. Da zero il token entra solo se il gradiente
        # lo chiede.
        old = self.pool_proj
        with torch.random.fork_rng(devices=[]):
            new = nn.Linear(2 * latent_dim + 1, latent_dim)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, : 2 * latent_dim].copy_(old.weight)
            new.bias.copy_(old.bias)
        self.pool_proj = new

    def forward(self, V, mass, L, evals, evecs, faces, gradX, gradY, size_token,
                return_per_vertex: bool = False, add_noise: bool = True):
        Z_per_vertex = self.encoder(V, mass, L, evals, evecs, faces=faces, gradX=gradX, gradY=gradY)
        Z_per_vertex = self.vertex_bottleneck(Z_per_vertex)
        if add_noise:
            Z_per_vertex = Z_per_vertex + 0.01 * torch.randn_like(Z_per_vertex)
        Z_mean = Z_per_vertex.mean(dim=0, keepdim=True)
        Z_max = Z_per_vertex.max(dim=0, keepdim=True).values
        tok = size_token.reshape(1, 1).to(dtype=Z_mean.dtype)
        Z_global = self.pool_proj(torch.cat([Z_mean, Z_max, tok], dim=1))
        if return_per_vertex:
            return Z_per_vertex, Z_global
        return Z_global


def sample_to_device_factory(orig):
    def sample_to_device(sample: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
        out = orig(sample, device=device)
        t = sample.get(SIZE_KEY)
        if torch.is_tensor(t):
            out[SIZE_KEY] = t.to(device)
        return out
    sample_to_device._ablation = True
    return sample_to_device


def _log_once(key: str, msg: str) -> None:
    if key not in _STATE["logged"]:
        _STATE["logged"].add(key)
        print(msg, flush=True)


def _rebind(name: str, obj) -> None:
    """Rebind `name` in ogni modulo robustness/perturbated/eval gia' importato che lo aveva copiato."""
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("robustness") or mod_name.startswith("compare_model_vs_chamfer")
                or mod_name in ("eval_by_topology", "eval_cells", "__main__")):
            continue
        if hasattr(mod, name):
            setattr(mod, name, obj)


def install(size_token_json: str | Path | None = None, eval_frame: str | None = None) -> None:
    """Installa dataset col token, sample_to_device, build/forward del modello col token.

    Va chiamata PRIMA che gli script importino i nomi; per i moduli gia' importati li rebinda.
    """
    import fast_data as fd
    import robustness.data_utils as du
    import robustness.model_helpers as mh
    from frames import FRAMES, reframe

    if eval_frame is not None and eval_frame not in FRAMES:
        raise ValueError(f"eval_frame {eval_frame!r} non in {FRAMES}")
    table = None if size_token_json is None else SizeTokenTable(size_token_json)
    _STATE.update(table=table, eval_frame=None if eval_frame in (None, "current") else eval_frame)

    if table is not None:
        factory = lambda data_dir, *a, **k: SizeTokenDataset(data_dir, _STATE["table"])  # noqa: E731
        du.GTReadyDataset = factory
        _rebind("GTReadyDataset", factory)
        fd.GTReadyDatasetNPZ = factory      # base del CachedDataset del training
        if SIZE_KEY not in fd.TENSOR_KEYS:
            fd.TENSOR_KEYS = tuple(fd.TENSOR_KEYS) + (SIZE_KEY,)
        if not getattr(du.sample_to_device, "_ablation", False):
            stod = sample_to_device_factory(du.sample_to_device)
            du.sample_to_device = stod
            _rebind("sample_to_device", stod)

    orig_build, orig_forward = mh.build_model, mh.forward_model

    def build_model(args, device):
        if _STATE["table"] is None:
            return orig_build(args, device)
        if getattr(args, "model", None) != "xyz_dn":
            raise ValueError("il token di taglia e' agganciato solo a --model xyz_dn")
        if getattr(args, "use_smooth_loss", False):
            raise ValueError("smooth loss non agganciata al modello col token")
        m = DiffusionEncoderSizeToken(latent_dim=args.latent_dim, width=args.width,
                                      n_blocks=args.n_blocks, dropout=args.dropout,
                                      pool_mode=args.pool_mode).to(device)
        print(f"[ablation] DiffusionEncoderSizeToken: {sum(p.numel() for p in m.parameters())} "
              f"parametri, token da {_STATE['table'].path}", flush=True)
        return m

    def forward_model(model, sample_dict, V_in, return_gate_info, add_noise):
        if _STATE["eval_frame"] is not None:
            V_in = reframe(V_in, sample_dict["mass"], sample_dict["faces"], _STATE["eval_frame"])
            _log_once("frame", f"[ablation] eval: V_in ri-inquadrato nel frame {_STATE['eval_frame']}")
        if isinstance(model, DiffusionEncoderSizeToken):
            tok = sample_dict.get(SIZE_KEY)
            if tok is None:
                raise RuntimeError("modello col token di taglia ma campione senza size_token: "
                                   "il dataset non e' quello agganciato")
            _log_once("token", f"[ablation] forward col token attivo (primo campione: {float(tok.reshape(-1)[0]):+.3f})")
            z = model(V_in, sample_dict["mass"], sample_dict["L"], sample_dict["evals"],
                      sample_dict["evecs"], sample_dict["faces"], sample_dict["gradX"],
                      sample_dict["gradY"], size_token=tok, return_per_vertex=False,
                      add_noise=add_noise)
            return z, mh._default_gate_info(z)
        return orig_forward(model, sample_dict, V_in, return_gate_info, add_noise)

    mh.build_model, mh.forward_model = build_model, forward_model
    _rebind("build_model", build_model)
    _rebind("forward_model", forward_model)
    print(f"[ablation] hook installati: size_token={None if table is None else table.path} "
          f"(collezione {None if table is None else table.collection}) eval_frame={eval_frame}",
          flush=True)
