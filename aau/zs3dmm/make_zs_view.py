#!/usr/bin/env python3
"""Vista ``id<NNNNNN>_GTready_<topologia>.npz`` sulle mesh di un dominio zero-shot, piu' le GT rinominate.

    aau/run.sh aau/zs3dmm/make_zs_view.py --prefix hifi --topo-dir datasets/HIFI3D/topo \\
        --gt-dir datasets/HIFI3D/gt --out-dir datasets/HIFI3D/eval_view --id-offset 900000

Copia di ``v2_work/genict/make_train_ready.py`` con il prefisso del dominio, senza lo split
train/held-out e SENZA operatori:
  - nessun modello vede questi domini, quindi il pool intero e' il pool di eval (come
    ``datasets/ICT/eval_view_heldout``), e i 100 soggetti li estrae ``rebuild_subject_split``
    con WBES_EVAL_SEED, sia per i modelli sia per le baseline;
  - i symlink puntano alla sola geometria (``topo/``, chiavi ``V``/``F``): e' quello che leggono
    le baseline (``common.load_verts_faces``) e quello da cui ``zs_zeroshot.sbatch`` calcola gli
    operatori su /tmp del nodo, per i soli soggetti valutati. Gli operatori di 3000 mesh nella
    home costerebbero decine di GB (59 MB per identita' su ICT, ict_ops.sbatch).

``intrinsic_utils.SUBJECT_RE_ANY`` e' ``(id\\d+)``: ``hifi0000_GTready_original.npz`` non da'
nessun soggetto, da qui i symlink ``id<offset + NNNN>``. Offset a 6 cifre (900000 HIFI3D,
910000 FaceVerse), fuori da BFM id0000-, FLAME id1000-, ICT id10000-14999 e dalle identita'
ICT nuove di ``aau/data_scale`` (da id20000, dati di training).

Scrive ``npz/`` (symlink), ``gt_matrix.npz`` (GT del protocollo ICT, vertex-mean-L2 maxabs) e
``gt_coef_matrix.npz`` (distanza nei coefficienti standardizzati), con gli stessi nomi.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--topo-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--id-offset", type=int, required=True)
    args = ap.parse_args()

    name_re = re.compile(rf"^{args.prefix}(?P<num>\d+)_GTready_(?P<variant>.+)\.npz$")
    gt_files = {"gt_matrix.npz": f"{args.prefix}_matrix_distances_maxabs.npz",
                "gt_coef_matrix.npz": f"{args.prefix}_coef_distances.npz"}

    data_out = args.out_dir / "npz"
    data_out.mkdir(parents=True, exist_ok=True)
    for stale in data_out.glob("*.npz"):
        stale.unlink()

    def rename(name: str) -> str:
        m = re.fullmatch(rf"{args.prefix}(\d+)", name)
        if not m:
            raise SystemExit(f"nome GT non {args.prefix}NNNN: {name}")
        return f"id{args.id_offset + int(m.group(1)):06d}"

    renamed = None
    for out_name, src_name in gt_files.items():
        with np.load(args.gt_dir / src_name, allow_pickle=True) as z:
            D = z["D_orig"]
            names = [rename(str(n)) for n in z["names"]]
        if renamed is not None and names != renamed:
            raise SystemExit(f"{src_name}: nomi in ordine diverso dalla GT primaria")
        renamed = names
        np.savez(args.out_dir / out_name, D_orig=D, names=np.array(names))

    n_link = 0
    for p in sorted(args.topo_dir.glob(f"{args.prefix}*_GTready_*.npz")):
        m = name_re.match(p.name)
        if not m:
            continue
        link = data_out / f"id{args.id_offset + int(m['num']):06d}_GTready_{m['variant']}.npz"
        link.symlink_to(p.resolve())
        n_link += 1

    meta = {
        "source_topo": str(args.topo_dir),
        "source_gt": {k: str(args.gt_dir / v) for k, v in gt_files.items()},
        "id_offset": args.id_offset,
        "n_symlinks": n_link,
        "n_symlinks_expected": len(renamed) * 6,
        "n_subjects": len(renamed),
        "id_range": [renamed[0], renamed[-1]],
        "note": "vista di eval zero-shot, sola geometria: gli operatori si calcolano su /tmp nel job",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if n_link != len(renamed) * 6:
        raise SystemExit(f"{n_link} symlink su {len(renamed) * 6} attesi in {args.topo_dir}")


if __name__ == "__main__":
    main()
