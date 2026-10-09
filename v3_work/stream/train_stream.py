#!/usr/bin/env python
"""Trainer v3 con ``--stream``: i batch di training dal buffer ad anello dei produttori (pipeline P1).

    aau/run.sh v3_work/stream/train_stream.py --stream /tmp/$SLURM_JOB_ID/stream --total-steps 2000 \
        --steps-per-epoch 200 --batch_subjects 16 --max_meshes_per_subject_train 4 ... (flag di train_v3.py)
    torchrun --standalone --nproc-per-node 8 v3_work/stream/train_stream.py --stream ... (DDP)

Perche' un wrapper e non un flag in v3_work/trainer/train_v3.py: la directory del run e' un hash di TUTTI gli
argomenti (train_v3.make_run_dir), quindi un flag nuovo nel parser cambierebbe la directory delle ablazioni in
corso alla loro ripresa (job prelazionabili con --requeue) e le farebbe ripartire da zero. Qui il parser e'
quello di train_v3 piu' i flag ``--stream*``; senza ``--stream`` e' train_v3 tale e quale.

Con ``--stream`` cambiano solo i batch di TRAINING (tutto il resto e' train_v3.run: modello, loss, ottimizzatore,
EMA, DDP, checkpoint e ripresa, eval online sul dataset di ``--data_dir``/``--data-spec``/``--store``):
  * ``train_v3.epoch_plans`` -> ``consumer.StreamPlans``: ``--steps-per-epoch`` piani estratti dall'anello
    (``--batch_subjects`` identita' distinte, fino a ``--max_meshes_per_subject_train`` viste ciascuna, rumore
    del batch come in v2);
  * l'embedder legge le viste dal consumatore invece che dal dataset;
  * la GT del batch e' ``consumer.StreamGT`` (dagli s_i delle viste) invece della matrice di ``--dist_npz``;
  * ``losses_v3.domain_of`` conosce anche i domini dello stream (pesi fra domini delle loss log).
Statistiche dello stream per epoca: ``<run_dir>/stream_stats_rank<r>.jsonl``.

GT di riferimento dopo E12 (anello di ``producer.py --canonical-gt``):
  * ``--stream-gt fr``: GT-FR (forma, mm) al volo, per il controllo con la testa standard (ctrl-FR);
  * ``--stream-gt sr``: GT-SR (Procrustes) per u e log S_i per s, per ``--head factorized|factorized2``; le chiavi
    dello stream prendono log S_i dalle viste (``StreamLogCS``), il bias iniziale di s e' la media di log S sulle
    identita' dell'anello alla partenza (media fra i rank); ``--size-mask-domains`` vale coi domini dello stream;
  * scala: ``--stream-gt-mm`` o, a 0, ``targets.gt_unit`` (media geometrica delle unita' tarate di E12).
    ``--gt-scale`` agisce solo sulla GT di ``--dist_npz`` (eval online), non su quella dello stream;
  * ``--input-norm global``: X in mm veri / L0 dalle aree delle viste (consumer.py), ``--stream-scale 0``;
    ``--scale-table`` e ``--size-table`` restano obbligatori per i soggetti dell'eval online.
Piu' nodi (torchrun con rendezvous, un anello per nodo): gli shard si dividono fra i rank del nodo
(LOCAL_RANK, LOCAL_WORLD_SIZE); i produttori di nodi diversi vanno avviati con semi diversi. ``--stream-extra``:
anelli condivisi (CephFS) di produttori solo CPU su altri nodi, divisi col rank globale.
Rilascio: ``--stream-sources open_core`` addestra sul solo nucleo aperto; ``--stream-log-views`` registra ogni vista
usata con la provenienza (producer.py --provenance), da cui regen.py la rigenera.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
TRAINER = THIS_DIR.parent / "trainer"
for _p in (TRAINER, THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_v3 as tv  # noqa: E402

STREAM: dict = {}


def build_parser():
    p = tv.build_parser()
    g = p.add_argument_group("stream (v3_work/stream)")
    g.add_argument("--stream", default="", help="directory dell'anello dei produttori (producer.py --ring)")
    g.add_argument("--stream-reuse", type=float, default=4.0, help="usi massimi di un'identita' prima del fallback")
    g.add_argument("--stream-rot", default="30,15,10", help="yaw,pitch,roll massimi in gradi a ogni uso")
    g.add_argument("--stream-scale", type=float, default=0.1, help="scala uniforme in 1 +- s a ogni uso")
    g.add_argument("--stream-min-groups", type=int, default=0, help="identita' nell'anello prima di partire")
    g.add_argument("--stream-wait-s", type=float, default=1800.0)
    g.add_argument("--stream-prefetch", type=int, default=4, help="thread che preparano il batch successivo")
    g.add_argument("--stream-gt-mm", type=float, default=0.0,
                   help="unita' fisiche per unita' della GT (mm; d_P con --stream-gt sr). 0 = unificata: il suo massimo; "
                        "fr/sr: targets.gt_unit")
    g.add_argument("--stream-extra", default="",
                   help="anelli in piu', separati da virgola (es. un anello condiviso su CephFS di produttori solo CPU): "
                        "letti da tutti i rank, shard divisi col rank globale")
    g.add_argument("--stream-extra-refresh-s", type=float, default=10.0, help="rilettura delle directory in piu'")
    g.add_argument("--stream-extra-mirror", default="",
                   help="directory locale (su /tmp) dove un thread per rank copia gli shard degli anelli in piu' prima "
                        "dell'uso (consigliato con CephFS: letto direttamente costa ~2 s per passo)")
    g.add_argument("--stream-sources", default="",
                   help="solo i gruppi di queste fonti (preset di sources.SOURCE_PRESETS, es. open_core, o domini)")
    g.add_argument("--stream-log-views", action="store_true",
                   help="registro delle viste usate con la provenienza: <run_dir>/views_used/*.npz (views_log.py)")
    g.add_argument("--stream-gt", default="unified", choices=["unified", "fr", "sr"],
                   help="GT del batch: unificata (s_i) o GT di E12 (producer.py --canonical-gt)")
    return p


def shard_split(rank: int, world: int) -> tuple[int, int]:
    """(rank, numero di rank) fra cui dividere gli shard: tutti su un nodo, quelli del nodo su piu' nodi."""
    import os
    lw = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    if lw != world:
        return int(os.environ.get("LOCAL_RANK", rank)), lw
    return rank, world


def consumer():
    if "consumer" not in STREAM:
        from common import dist_info
        from consumer import StreamConsumer, StreamGT
        import sources
        args = STREAM["args"]
        rank, world, _ = dist_info()
        sp = sources.Unified()
        if args.stream_gt == "unified":
            gt_mm = float(args.stream_gt_mm) or sources.gt_scale_mm()
        else:
            import targets
            gt_mm = float(args.stream_gt_mm) or targets.gt_unit(args.stream_gt)
        STREAM["gt"] = StreamGT(sp.A, gt_mm)
        sr, sw = shard_split(rank, world)
        STREAM["consumer"] = StreamConsumer(
            args.stream, rank=rank, world=world, reuse=args.stream_reuse, seed=args.seed,
            input_norm=args.input_norm, area_weights=args.area_weights,
            rot_deg=[float(x) for x in args.stream_rot.split(",")], scale=args.stream_scale,
            min_groups=args.stream_min_groups, wait_s=args.stream_wait_s, prefetch=args.stream_prefetch,
            gt=STREAM["gt"], gt_kind=args.stream_gt, global_unit_mm=args.global_unit_mm, global_ops=args.global_ops,
            shard_rank=sr, shard_world=sw, extra_refresh_s=args.stream_extra_refresh_s,
            extra_rings=[(r, rank, world) for r in args.stream_extra.split(",") if r],
            domains=sources.parse_sources(args.stream_sources) if args.stream_sources else None,
            log_views=args.stream_log_views, extra_mirror=args.stream_extra_mirror)
        STREAM["log"] = STREAM["run_dir"] / f"stream_stats_rank{rank}.jsonl"
        unit = {"unified": "mm (unificata)", "fr": "mm (GT-FR di E12)", "sr": "d_P (GT-SR di E12)"}[args.stream_gt]
        tv.log0(f"[stream] anello {args.stream}: rank {rank}/{world} (shard {sr} di {sw}), riuso <= "
                f"{args.stream_reuse:g}, rotazione {args.stream_rot} gradi, scala +-{args.stream_scale:g}, GT "
                f"{args.stream_gt} al volo ({gt_mm:.6g} {unit} per unita'), input_norm={args.input_norm}"
                + (f", anelli in piu' {args.stream_extra}" if args.stream_extra else "")
                + (f", fonti {sources.parse_sources(args.stream_sources)}" if args.stream_sources else "")
                + (", registro delle viste usate" if args.stream_log_views else ""))
    return STREAM["consumer"]


def stream_epoch_plans(args, data, epoch, S, steps, B, drawcfg, seed_r):
    from consumer import StreamPlans
    rng = np.random.default_rng(int(seed_r) + 977 + int(epoch))
    return StreamPlans(consumer(), steps, B, int(args.max_meshes_per_subject_train), drawcfg, rng, epoch,
                       STREAM["log"])


class StreamLogCS:
    """log S_i (mm) per la testa fattorizzata: le chiavi dello stream dalle viste (StreamGT.log_s), gli altri
    soggetti (eval online) dalla tabella di ``--size-table``."""

    def __init__(self, table: dict) -> None:
        self.table = table

    def __getitem__(self, sid: str) -> float:
        gt = STREAM.get("gt")
        if gt is not None and sid in gt.log_s:
            return gt.log_s[sid]
        if sid in self.table:
            return self.table[sid]
        raise KeyError(f"{sid}: log S assente (anello senza producer.py --canonical-gt?)")

    def __contains__(self, sid: str) -> bool:
        gt = STREAM.get("gt")
        return (gt is not None and sid in gt.log_s) or sid in self.table


def stream_size_init() -> float:
    """Media di log S sulle identita' dell'anello alla partenza (pool del rank, poi media fra i rank)."""
    c = consumer()
    c.refresh()
    t0 = time.time()
    c._wait_for(max(int(STREAM["args"].batch_subjects), 32))
    print(f"[stream] rank {c.rank}: {len(c.pool)} gruppi nel pool dopo {time.time() - t0:.0f}s di attesa", flush=True)
    logs = []
    for seq, g in c.pool:
        hg = c.readers[seq].groups[g]
        if "S" not in hg:
            raise SystemExit(f"{c.readers[seq].path}: shard senza S (producer.py --canonical-gt)")
        logs.append(float(np.log(hg["S"])))
    v = tv.all_mean(float(np.mean(logs)))
    tv.log0(f"[stream] bias iniziale di s: media di log S su {len(logs)} identita' del rank 0 e media fra i rank = "
            f"{v:.4f} (S = {np.exp(v):.1f} mm)")
    return v


def install() -> None:
    """Gli agganci di ``--stream`` nel modulo train_v3 (solo in questo processo)."""
    import factorized_v3
    import losses_v3
    import model_v3
    from common import domain_of

    class StreamData(tv.Data):
        def __init__(self, args, rank, world) -> None:
            super().__init__(args, rank, world)
            if args.head in factorized_v3.FACTORIZED:
                self.log_cs = StreamLogCS(self.log_cs)
                self.size_init = stream_size_init()

    class StreamEmbedder(model_v3.StepEmbedder):
        def bind(self, dataset, perturbation) -> None:
            super().bind(consumer(), perturbation)

    def stream_step_batch(**kw):
        kw["gt"], kw["name_to_idx"] = STREAM["gt"], STREAM["gt"].name_to_idx
        return losses_v3.StepBatch(**kw)

    def stream_domain_of(sid: str) -> str:
        gt = STREAM.get("gt")
        if gt is not None and sid in gt.domain:
            return gt.domain[sid]
        return domain_of(sid)

    tv.epoch_plans = stream_epoch_plans
    tv.StepEmbedder = StreamEmbedder
    tv.StepBatch = stream_step_batch
    tv.Data = StreamData
    losses_v3.domain_of = stream_domain_of
    factorized_v3.domain_of = stream_domain_of      # --size-mask-domains coi domini dello stream


def main() -> None:
    import faulthandler
    import signal
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)    # kill -USR1 <pid>: stack senza fermare il run
    if float(os.environ.get("WBES_STACK_DUMP_S", "0") or 0) > 0:   # diagnosi di un blocco: stack ogni N secondi
        faulthandler.dump_traceback_later(float(os.environ["WBES_STACK_DUMP_S"]), repeat=True)
    signal.signal(signal.SIGTERM, tv._on_term)
    args = build_parser().parse_args()
    tv.check_args(args)
    if args.stream:
        if int(args.total_steps) <= 0 or int(args.steps_per_epoch) <= 0:
            raise SystemExit("--stream richiede --total-steps e --steps-per-epoch")
        if not Path(args.stream).is_dir():
            raise SystemExit(f"--stream {args.stream}: directory assente (avviare prima producer.py)")
        if args.head in ("factorized", "factorized2") and args.stream_gt != "sr":
            raise SystemExit("--head factorized*: la GT di u e' GT-SR, serve --stream-gt sr")
        if args.input_norm == "global" and args.stream_scale != 0:
            raise SystemExit("--input-norm global: --stream-scale 0 (la scala cambierebbe la taglia senza il "
                             "bersaglio; l'augmentation di scala con bersaglio e' --scale-aug)")
        STREAM["args"] = args
        # la run dir PRIMA che run() riscriva epochs e save_every negli args (l'hash li contiene)
        STREAM["run_dir"] = tv.make_run_dir(args)
        install()
    try:
        tv.run(args)
    finally:
        tv.dv.kill_all()


if __name__ == "__main__":
    main()
