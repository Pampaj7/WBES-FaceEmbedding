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
"""
from __future__ import annotations

import sys
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
                   help="mm per unita' della GT (0 = quella della GT unificata di training, il suo massimo)")
    return p


def consumer():
    if "consumer" not in STREAM:
        from common import dist_info
        from consumer import StreamConsumer, StreamGT
        import sources
        args = STREAM["args"]
        rank, world, _ = dist_info()
        sp = sources.Unified()
        gt_mm = float(args.stream_gt_mm) or sources.gt_scale_mm()
        STREAM["gt"] = StreamGT(sp.A, gt_mm)
        STREAM["consumer"] = StreamConsumer(
            args.stream, rank=rank, world=world, reuse=args.stream_reuse, seed=args.seed,
            input_norm=args.input_norm, area_weights=args.area_weights,
            rot_deg=[float(x) for x in args.stream_rot.split(",")], scale=args.stream_scale,
            min_groups=args.stream_min_groups, wait_s=args.stream_wait_s, prefetch=args.stream_prefetch,
            gt=STREAM["gt"])
        STREAM["log"] = STREAM["run_dir"] / f"stream_stats_rank{rank}.jsonl"
        tv.log0(f"[stream] anello {args.stream}: rank {rank}/{world}, riuso <= {args.stream_reuse:g}, rotazione "
                f"{args.stream_rot} gradi, scala +-{args.stream_scale:g}, GT unificata al volo ({gt_mm:.4f} mm per "
                f"unita'), input_norm={args.input_norm}")
    return STREAM["consumer"]


def stream_epoch_plans(args, data, epoch, S, steps, B, drawcfg, seed_r):
    from consumer import StreamPlans
    rng = np.random.default_rng(int(seed_r) + 977 + int(epoch))
    return StreamPlans(consumer(), steps, B, int(args.max_meshes_per_subject_train), drawcfg, rng, epoch,
                       STREAM["log"])


def install() -> None:
    """Gli agganci di ``--stream`` nel modulo train_v3 (solo in questo processo)."""
    import losses_v3
    import model_v3
    from common import domain_of

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
    losses_v3.domain_of = stream_domain_of


def main() -> None:
    import faulthandler
    import signal
    faulthandler.enable()
    signal.signal(signal.SIGTERM, tv._on_term)
    args = build_parser().parse_args()
    tv.check_args(args)
    if args.stream:
        if int(args.total_steps) <= 0 or int(args.steps_per_epoch) <= 0:
            raise SystemExit("--stream richiede --total-steps e --steps-per-epoch")
        if not Path(args.stream).is_dir():
            raise SystemExit(f"--stream {args.stream}: directory assente (avviare prima producer.py)")
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
