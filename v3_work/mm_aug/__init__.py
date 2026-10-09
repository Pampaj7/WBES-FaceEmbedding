"""Moltiplicatori di dati per il run massivo (PLAN_MASSIVE sez. 17-21): identita' ed espressioni NUOVE, fuori dai
sottospazi dei singoli 3DMM, generate al volo.

    import sys; sys.path.insert(0, "<repo>")
    from v3_work.mm_aug import AugConfig, sample_view_spec, rebuild_view_spec
    spec = sample_view_spec(np.random.default_rng(0), AugConfig())

``aug.py`` (API, provenienza, licenze), ``transfer.py`` (trasporto con raccordo biarmonico, controlli di validita'),
``selftest.py`` (prove), ``stats.py`` (soglie, statistiche nello spazio FR, render). Note e numeri in
``aau/runs/evidence/mm_aug/README.md``.
"""

from .aug import (AUG_VERSION, EXPR_SOURCES, KINDS, LICENSES, TEMPLATES, AugConfig, AugLibrary, assemble,
                  build_expression, build_identity, config_from_dict, draw_view, get_library, gt_targets,
                  rebuild_view_spec, sample_view_spec, view_spec_from_seed)

__all__ = ["AUG_VERSION", "AugConfig", "AugLibrary", "EXPR_SOURCES", "KINDS", "LICENSES", "TEMPLATES", "assemble",
           "build_expression", "build_identity", "config_from_dict", "draw_view", "get_library", "gt_targets",
           "rebuild_view_spec", "sample_view_spec", "view_spec_from_seed"]
