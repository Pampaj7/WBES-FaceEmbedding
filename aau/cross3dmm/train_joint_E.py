#!/usr/bin/env python3
"""Entrypoint del confronto E congiunto / controllo congiunto: patch di train_v2 + cache di train_fast
+ token di taglia per dominio. Nessun file di ricerca modificato.

  - v2_work/train_v2/train_v2.py: batch a dominio singolo, GT con NaN fuori blocco protetta, eval
    online su un dominio (--eval_domain), con il suo context manager _patched_v1;
  - v2_work/fastio/train_fast.py: cache in RAM e --frame, come le ablazioni B/C/E;
  - aau/models/ablation_hooks.py: modello e dataset col token. La tabella e' per DOMINIO: due
    tabelle SizeTokenTable (aau/cross3dmm/joint_E_prep.py build), ciascuna standardizzata sui
    soggetti di training del suo dominio, scelte per file dal dominio del soggetto (train_v2.domain_of).

Flag propri (tutto il resto va a train_fast.py e da li' al trainer v1):
    --eval_domain D              come train_v2
    --size-token-json-bfm J      \\ insieme o nessuno dei due (controllo: nessuno)
    --size-token-json-ict J      /

    aau/run.sh aau/cross3dmm/train_joint_E.py --eval_domain bfm --size-token-json-bfm <bfm.json> \
        --size-token-json-ict <ict.json> --cache-residency ram --frame rms --data_dir <vista> ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work/fastio"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/train_v2"))
sys.path.insert(0, str(REPO_ROOT / "aau/models"))

import train_fast  # noqa: E402  (mette sul path il pacchetto robustness e fast_data)


class DomainSizeTokenTable:
    """Due SizeTokenTable, una per dominio; token(nome) usa quella del dominio del soggetto."""

    def __init__(self, tables: dict):
        from train_v2 import domain_of

        self._domain_of = domain_of
        self.tables = tables
        self.path = " + ".join(str(t.path) for t in tables.values())
        self.collection = "+".join(tables)
        self.log_r = {}
        for dom, t in tables.items():
            bad = [n for n in t.log_r if domain_of(n.split("_", 1)[0]) != dom]
            if bad:
                raise SystemExit(f"tabella {t.path}: {len(bad)} nomi fuori dal dominio {dom} (primo {bad[0]})")
            self.log_r.update(t.log_r)

    def token(self, name: str) -> float:
        return self.tables[self._domain_of(name.split("_", 1)[0])].token(name)


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--eval_domain", type=str, default="")
    p.add_argument("--size-token-json-bfm", type=Path, default=None)
    p.add_argument("--size-token-json-ict", type=Path, default=None)
    known, rest = p.parse_known_args()
    seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else None
    batch = int(rest[rest.index("--batch_subjects") + 1]) if "--batch_subjects" in rest else None
    if batch is None:
        raise SystemExit("serve --batch_subjects esplicito (lo usa il batching a dominio singolo)")

    if (known.size_token_json_bfm is None) != (known.size_token_json_ict is None):
        raise SystemExit("--size-token-json-bfm e --size-token-json-ict vanno insieme")
    if known.size_token_json_bfm is not None:
        import robustness.train_runner  # noqa: F401  (carica data_utils/eval_utils/model_helpers)
        import ablation_hooks
        from ablation_hooks import SizeTokenTable, install

        tables = {"bfm": SizeTokenTable(known.size_token_json_bfm),
                  "ict": SizeTokenTable(known.size_token_json_ict)}
        for dom, t in tables.items():
            if t.collection != dom or t.seed is None or seed != int(t.seed):
                raise SystemExit(f"token {dom}: tabella {t.collection} seed {t.seed}, run seed {seed}: "
                                 f"le statistiche devono venire dal training di QUESTO run")
            print(f"[joint-E] token {dom}: media {t.mean:.5f} std {t.std:.5f} da {t.path}", flush=True)
        install(size_token_json=known.size_token_json_bfm)
        ablation_hooks._STATE["table"] = DomainSizeTokenTable(tables)

    import train_v2

    sys.argv = [sys.argv[0]] + rest
    with train_v2._patched_v1(batch, str(known.eval_domain)):
        train_fast.main()
    print(f"[v2] domain-blocked permutations={train_v2.STATS['blocked_permutations']} "
          f"batches={train_v2.STATS['batches']}", flush=True)


if __name__ == "__main__":
    main()
