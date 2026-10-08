"""Libreria uniforme dei 3DMM per il training massivo (D1, paper/PLAN_MASSIVE.md sez. 2-4).

    import sys; sys.path.insert(0, "<repo>")
    from v3_work.mm import load_model, training_models

    m = load_model("flame2020")
    rng = np.random.default_rng(1234)
    z = m.sample_identity(rng)                 # 80% N(0,1) + 20% N(0,2^2) per coefficiente, |z| <= 3.5
    e = m.sample_expression(rng)               # prior d'espressione del modello
    V = m.mesh(z, e)                           # patch del volto, unita' e frame nativi
    V = m.local_rbf_deform(V, rng)             # bump 1-3 mm, raggio 10-30 mm
    Vc, Fc = m.canonical_transform(V, m.faces) # frame FLAME in mm, normali uscenti

Interfaccia in ``model.py``, un caricatore per modello in ``loaders.py`` (tabella dei modelli, delle
regioni e dei frame nel suo docstring), prove in ``selftest.py``.

Ruoli (PLAN_MASSIVE sez. 9 e 14.5): ``train`` BFM 2019 (il BFM del training; varianti face12 e
fullHead per il supporto), ICT, GNM, FLAME 2020/2023, e BFM 3DDFA (40 modi, solo per confronto);
``dev`` FaceScape (300 e 50 modi), mai in training; ``test`` HIFI3D e FaceVerse. I campionatori chiamati
con ``purpose="train"`` (il default) sollevano ``RoleError`` sui modelli dev e test.

LICENZE: nessun file dei modelli sta nel repo, ne' ci finisce. I percorsi sono quelli di
``loaders.DEFAULT_PATHS`` (FLAME in v2_work/genflame/official/ e BFM 2019 in external_data/bfm2019/,
ignorate da git; FaceScape, HIFI3D, FaceVerse e GNM in ~/data). BFM 2019 si legge dall'npz di
``bfm2019_convert.py`` (accanto all'.h5): il venv non ha h5py. La cache U_exp di FaceScape 50 sta in
~/.cache/wbes_mm.
"""

from __future__ import annotations

from functools import lru_cache

from .loaders import ALIASES, LOADERS, ROLES
from .model import (BilinearModel, ExpressionSpace, Frame, MorphableModel, RoleError, assert_trainable,
                    rbf_bump, sample_coefficients, sample_expression_coefficients)

MODELS = tuple(LOADERS)


def _canonical_name(name: str) -> str:
    if name in LOADERS:
        return name
    for k, al in ALIASES.items():
        if name in al:
            return k
    raise KeyError(f"modello sconosciuto: {name!r} (noti: {', '.join(MODELS)})")


@lru_cache(maxsize=None)
def load_model(name: str) -> MorphableModel:
    """Il modello ``name`` (o un suo alias), caricato una volta per processo."""
    key = _canonical_name(name)
    return LOADERS[key](key)


def models_with_role(role: str) -> list[str]:
    return [n for n in MODELS if ROLES[n] == role]


def training_models() -> list[str]:
    return models_with_role("train")


def load_for_training(name: str) -> MorphableModel:
    """``load_model`` piu' ``assert_trainable``: l'ingresso da usare nei generatori di training."""
    m = load_model(name)
    assert_trainable(m)
    return m


__all__ = ["MODELS", "ROLES", "BilinearModel", "ExpressionSpace", "Frame", "MorphableModel", "RoleError",
           "assert_trainable", "load_for_training", "load_model", "models_with_role", "rbf_bump",
           "sample_coefficients", "sample_expression_coefficients", "training_models"]
