# E1: varieta' di 3DMM contro quantita' di identita'

Piano: `paper/PLAN_MASSIVE.md`, sezione 13. Regole (scritte prima dei numeri): `aau/runs/evidence/e1/protocol.md` +
`protocol_amendment.md`; stato: `status.md`; risultati: `summary.md`.

Nessuna modifica al trainer (`v2_work/fastio/train_steps.py`) ne' a file esistenti: le celle sono split nuovi.

| file | cosa fa |
| --- | --- |
| `make_e1_splits.py` | `split_{c2m,c2f,c3f,c2fgnm,c2f40,c3f40,g1}.json` dallo split del run su scala (stratificati, seme 1234, C3F annidata in C2F) e `subsets.json` |
| `e1_design.py` | con le funzioni del trainer: blocchi, passi per dominio, identita' viste a 10.548/21.096 passi, cache esatta e picco di RAM previsto -> `aau/runs/evidence/e1/design.{json,md}` |
| `e1_train.sbatch` + `e1_train_body.sh` | training di una cella (corpo letto all'avvio del job); C2M si ferma dopo l'epoca 72; `WBES_E1_SMOKE=1` = prova sul nodo cpu |
| `e1_eval.sbatch` + `e1_eval_body.sh` | HIFI3D, FaceVerse con espressioni, FLAME, NoW sui due checkpoint, con le pipeline esistenti |
| `e1_summarize.py` + `.sbatch` | matrice, differenze appaiate, regola primaria, controlli -> `summary.md`, `matrix.csv`, `paired.csv`, `controls.csv` |
| `e1_zs_summarize.{sh,py}` | la tabella standard di `aau/zs3dmm` sui bracci HIFI3D delle celle (`aau/runs/evidence/e1/hifi/`) |
| `e1_launch.sh` | sottomette cancelli, training (V100, in ordine di priorita'), eval (A100, afterok) e riepilogo; job in `aau/runs/evidence/e1/jobs.md` |
| `e1_gate_smoke.{py,sbatch}` | cancello: loss dell'epoca 1 dello smoke V100 entro il 5% di quella su L40S |
| `e1_gate_ugt.sbatch` | cancello di C3F-UGT: GT unificata verificata (`check.json`) |
| `smoke/` | split ridotti per lo smoke (`make_smoke_splits.py`) |
| `scratch/test_summarize_alias.py` | prova di `e1_summarize.py` su valutazioni esistenti (alias, numeri non di E1) |
| `scratch/prepass_bench/bench.sh` | velocita' del pre-pass per tipo di nodo |
