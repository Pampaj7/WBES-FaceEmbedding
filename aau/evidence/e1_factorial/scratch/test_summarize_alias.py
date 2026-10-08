#!/usr/bin/env python3
"""Prova dei percorsi di e1_summarize.py (celle, differenze appaiate, controlli) sui soli dati esistenti.

    aau/run.sh aau/evidence/e1_factorial/scratch/test_summarize_alias.py [--with-fv DIR]

C2M a 10.548 passi punta ai file di C3M a 21.096 (alias), quindi "C3M - C2M" a 10.548 = e036 - e072 di C3M.
FLAME: i risultati esistenti del congiunto e del BFM-only (aau/runs/ws_flame) fanno da C3M e036 e da "C2M"
e036 (checkpoint mappati per la prova). --with-fv: fv_expr di una prova di e1_eval_body.sh. Scrive in
scratch/alias_out, non tocca aau/runs/evidence/e1.
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import e1_summarize as s  # noqa: E402

out = HERE / "alias_out"
out.mkdir(exist_ok=True)
for f in ("protocol.md", "protocol.sha256", "design.json"):
    shutil.copy(s.OUT / f, out / f)
flame = s.RUNS / "ws_flame/b6649295c3f8"
for t, arm in (("e036", "joint"), ("e1c2me036", "bfm_only")):
    d = out / "flame" / t / "b6649295c3f8"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "joint").exists():
        (d / "joint").symlink_to(flame / arm)
if "--with-fv" in sys.argv:
    src = Path(sys.argv[sys.argv.index("--with-fv") + 1])
    if not (out / "fv_expr").exists():
        (out / "fv_expr").symlink_to(src)
alias = lambda c, e: ("c3m", "072") if (c, e) == ("c2m", "036") else (c, e)  # noqa: E731
h, n, k, fv, ev = s.hifi_stage, s.now_dir, s.expected_ckpt, s.fv_stage, s.eval_ckpt
s.hifi_stage = lambda c, e, p: h(*alias(c, e), p)
s.now_dir = lambda c, e: n(*alias(c, e))
s.fv_stage = lambda c, e: fv(*alias(c, e))
s.expected_ckpt = lambda c, e: k(*alias(c, e))
fake = {os.path.realpath(s.RUNS / "x3dmm_joint_bfm_ict_s1234_1019532/mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d/checkpoints/best_by_xtopo_mesh_clean.pth"): ("c3m", "036"),
        os.path.realpath(s.RUNS / "remesh_v1recipe_areanorm_s1234_1019310/mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d/checkpoints/best_by_xtopo_mesh_clean.pth"): ("c3m", "072")}
s.eval_ckpt = lambda st: k(*fake[ev(st)]) if ev(st) in fake else ev(st)
s.OUT = out
sys.argv = [sys.argv[0], "--workers", "16"]
s.main()
