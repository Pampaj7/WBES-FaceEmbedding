#!/usr/bin/env python3
"""Espone le immagini frontali di Multiface con i nomi che vogliono i runner (WS3b).

    python3 aau/recon/ws3b_stage_images.py                      # gira sul frontend
    python3 aau/recon/ws3b_stage_images.py --pending-out /tmp/x --method prnet \
        --capture m--20180105--0000--002539136--GHS

I tre runner scrivono ``<out>/<stem dell'immagine>.npz``, e lo stem di un png Multiface e'
il solo numero di frame (``000213.png``): tre camere dello stesso frame si sovrascriverebbero
a vicenda, e non resterebbe scritto da nessuna parte a quale soggetto e segmento appartiene.
Invece di rinominare 3084 npz a posteriori, o di toccare i runner, si costruisce un albero
di **symlink gia' battezzati**

    aau/runs/multiface_ws3b/images/<capture>/<sogg>__<segmento>__<frame>__<camera>.png

e si passa quello a ``--images``.  I runner scrivono nel json anche il percorso
dell'immagine: e' il symlink, che resta valido perche' l'albero e' nella dir di run e non
in /tmp.

Vengono tenuti solo i frame che hanno la mesh tracciata corrispondente in
``datasets/Multiface/prep/tracked``: senza GT la ricostruzione non serve a niente.

Modo ``--pending-out``: lo stesso albero ma per un solo metodo e una sola capture, e solo
per le immagini la cui mesh non c'e' ancora.  E' cosi' che ``ws3b_recon.sbatch`` riparte
dopo un job interrotto senza rifare il lavoro gia' fatto (i runner non hanno skip-if-exists
e non glielo si vuole aggiungere).

Solo stdlib: niente numpy, quindi nessun bisogno di un job Slurm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3b_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--full-dir", type=Path, default=common.MF_FULL)
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--pending-out", type=Path, default=None,
                   help="albero dei soli frame ancora da ricostruire, per un metodo")
    p.add_argument("--method", type=str, default="", choices=("",) + common.METHODS)
    p.add_argument("--capture", type=str, default="", help="una sola capture")
    return p.parse_args()


def collect_items(full_dir: Path) -> list[common.Item]:
    """Un elemento per immagine con GT, in ordine stabile."""
    items: list[common.Item] = []
    n_orphan = 0
    for capture_dir in sorted(d for d in full_dir.iterdir() if d.is_dir()):
        images_root = capture_dir / "images"
        if not images_root.is_dir():
            continue
        subject = common.subject_of(capture_dir.name)
        segments = sorted(d.name for d in images_root.iterdir() if d.is_dir())
        version = common.version_of(segments)
        for segment in segments:
            for camera_dir in sorted(d for d in (images_root / segment).iterdir() if d.is_dir()):
                for image in sorted(camera_dir.glob("*.png")):
                    frame = image.stem
                    gt_name = common.gt_name_of(subject, segment, frame)
                    if not common.gt_mesh_path(gt_name).is_file():
                        n_orphan += 1
                        continue
                    items.append(common.Item(
                        name=common.item_name(subject, segment, frame, camera_dir.name),
                        subject=subject, segment=segment, frame=frame, camera=camera_dir.name,
                        version=version, capture=capture_dir.name,
                        image=str(image), gt_name=gt_name,
                    ))
    if n_orphan:
        print(f"[ws3b-stage] {n_orphan} immagini senza mesh tracciata, scartate")
    return items


def link(items: list[common.Item], root: Path) -> int:
    """Symlink ``<root>/<capture>/<name>.png`` -> immagine vera.  Idempotente."""
    n_new = 0
    for it in items:
        target = root / it.capture / f"{it.name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(it.image)
        n_new += 1
    return n_new


def main() -> None:
    args = parse_args()
    items = collect_items(args.full_dir.resolve())
    if not items:
        raise SystemExit(f"nessuna immagine sotto {args.full_dir}")

    if args.pending_out is not None:
        if not args.method or not args.capture:
            raise SystemExit("--pending-out vuole anche --method e --capture")
        out_dir = common.recon_dir(args.method, args.out_root)
        pending = [it for it in items
                   if it.capture == args.capture and not (out_dir / f"{it.name}.npz").is_file()]
        if not pending:
            print(f"[ws3b-stage] {args.method} {args.capture}: gia' completa, niente da fare")
            return
        link(pending, args.pending_out.resolve())
        print(f"[ws3b-stage] {args.method} {args.capture}: {len(pending)} immagini da fare "
              f"-> {args.pending_out}")
        return

    n_new = link(items, common.images_dir(args.out_root))
    common.write_manifest(common.manifest_path(args.out_root), items)
    kept = common.write_auc_manifest(common.auc_manifest_path(args.out_root), items)

    subjects = sorted({it.subject for it in items})
    print(f"[ws3b-stage] {len(items)} immagini, {len(subjects)} soggetti, "
          f"{len({(it.subject, it.segment) for it in items})} celle soggetto x segmento, "
          f"{n_new} symlink nuovi")
    for subject in subjects:
        mine = [it for it in items if it.subject == subject]
        cameras = sorted({it.camera for it in mine})
        print(f"  {subject} ({mine[0].version}): {len(mine):>4} immagini, "
              f"{len({it.segment for it in mine})} segmenti, camere {','.join(cameras)}")
    print(f"[ws3b-stage] manifest -> {common.manifest_path(args.out_root)}")
    print(f"[ws3b-stage] manifest per l'AUC ({len(kept)} elementi, una camera per soggetto) "
          f"-> {common.auc_manifest_path(args.out_root)}")


if __name__ == "__main__":
    main()
