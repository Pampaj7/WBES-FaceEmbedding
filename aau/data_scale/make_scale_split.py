#!/usr/bin/env python3
"""Split esplicito del run grande (e quello ridotto dello smoke), dalla politica congelata.

    python3 aau/data_scale/make_scale_split.py            # solo stdlib, gira sul frontend

held-out = ``heldout_frozen.json`` (politica joint_exact: ESATTAMENTE gli held-out del congiunto
x3dmm_joint_bfm_ict_s1234_1019532); training = tutti gli altri soggetti di BFM (id0000-0499, 392 come
nel congiunto), ICT-5000 (id10000-14999) e ICT nuove (id20000-69999). ``online_eval``: i 16 soggetti
dell'eval online del congiunto (dalla sua run dir), passati esplicitamente al trainer, cosi' i
``best_by_*`` sono scelti sugli stessi soggetti.
Uscite: split_scale.json (run grande) e split_smoke.json (40 BFM + 60 ICT-5000 + le 500 nuove
dei primi due shard in training; held-out completo, l'eval online ne usa 16).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
fz = json.loads((HERE / "heldout_frozen.json").read_text())
held = set(fz["bfm"]) | set(fz["ict_view"]) | set(fz.get("gnm", []))
bfm = [f"id{i:04d}" for i in range(500)]
ict = [f"id{i:05d}" for i in range(10000, 15000)]
new = [f"id{i:05d}" for i in range(20000, 70000)]
train = [s for s in bfm + ict + new if s not in held]
assert not set(train) & held
run_dir = Path(fz["compared_model"]["x3dmm_joint_bfm_ict_s1234_1019532"])
online = json.loads((run_dir / "online_eval_summary.json").read_text())["selected_subjects"]
assert sorted(online) == sorted(fz["online_eval_joint_1019532"]) and set(online) <= held, online
joint_train = set(json.loads((HERE.parents[1] / "aau/runs/ws2_cross3dmm/splits.json").read_text())
                  ["models"]["joint"]["train"])
assert set(s for s in train if s in set(bfm) | set(ict)) == joint_train, "il training vecchio non e' quello del congiunto"
heldout = sorted(held, key=lambda s: int(s[2:]))
note = f"politica {fz['policy']}: {fz['reason']}"
(HERE / "split_scale.json").write_text(json.dumps(
    {"source": "aau/data_scale/heldout_frozen.json", "note": note, "train": train, "heldout": heldout,
     "online_eval": online,
     "counts": {"train_bfm": sum(s in set(bfm) for s in train),
                "train_ict5000": sum(s in set(ict) for s in train),
                "train_ict_new": sum(s in set(new) for s in train),
                "heldout_bfm": len(fz["bfm"]), "heldout_ict": len(fz["ict_view"])}}, indent=0) + "\n")
smoke = ([s for s in bfm if s not in held][:40] + [s for s in ict if s not in held][:60] + new[:500])
(HERE / "split_smoke.json").write_text(json.dumps(
    {"source": "split_scale.json ridotto per lo smoke", "train": smoke, "heldout": heldout,
     "online_eval": online}, indent=0) + "\n")
print(json.loads((HERE / "split_scale.json").read_text())["counts"], "smoke train", len(smoke),
      "online_eval", online)


# --- 7 ottobre: run con GNM (datasets/GNM_DISTILL, id100000-110099) --------------------------------
import random  # noqa: E402

gnm = [f"id{i}" for i in range(100000, 110100)]
gnm_val = [s for s in gnm if s in held]
assert len(gnm_val) == 100, len(gnm_val)
train_all = [s for s in bfm + ict + new + gnm if s not in held]
# eval online in piu' su GNM: 16 dei 100 di validazione, seme dichiarato (non scelgono i best_by_*)
gnm_online = sorted(random.Random(20261007).sample(gnm_val, 16))
heldout_all = sorted(held, key=lambda s: int(s[2:]))
(HERE / "split_scale_all.json").write_text(json.dumps(
    {"source": "aau/data_scale/heldout_frozen.json", "note": note, "train": train_all, "heldout": heldout_all,
     "online_eval": online, "online_eval_extra": {"gnm": gnm_online},
     "counts": {"train_bfm": sum(s in set(bfm) for s in train_all),
                "train_ict5000": sum(s in set(ict) for s in train_all),
                "train_ict_new": sum(s in set(new) for s in train_all),
                "train_gnm": sum(s in set(gnm) for s in train_all),
                "heldout_bfm": len(fz["bfm"]), "heldout_ict": len(fz["ict_view"]), "heldout_gnm": len(gnm_val)}},
    indent=0) + "\n")
smoke_all = ([s for s in bfm if s not in held][:40] + [s for s in ict if s not in held][:60] + new[:250]
             + [s for s in gnm if s not in held][:250])
(HERE / "split_smoke_all.json").write_text(json.dumps(
    {"source": "split_scale_all.json ridotto per lo smoke", "train": smoke_all, "heldout": heldout_all,
     "online_eval": online, "online_eval_extra": {"gnm": gnm_online}}, indent=0) + "\n")
print("con GNM:", json.loads((HERE / "split_scale_all.json").read_text())["counts"], "gnm online", gnm_online[:4])
