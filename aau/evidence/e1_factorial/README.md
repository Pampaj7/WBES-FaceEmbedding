# E1: varieta' di 3DMM contro quantita' di identita'

Piano: `paper/PLAN_MASSIVE.md`, sezione 13. Protocollo e regola di lettura (scritti prima dei numeri):
`aau/runs/evidence/e1/protocol.md`; risultati: `aau/runs/evidence/e1/summary.md`.

Nessuna modifica al trainer (`v2_work/fastio/train_steps.py`) ne' a file esistenti: le celle sono split nuovi.

| file | cosa fa |
| --- | --- |
| `make_e1_splits.py` | `split_{c2m,c2f,c3f,g1}.json` dallo split del run su scala (stratificati, seme 1234, C3F annidata in C2F) e `subsets.json` |
| `e1_design.py` | con le funzioni del trainer: blocchi, passi per dominio, identita' viste a 10.548/21.096 passi, cache esatta e picco di RAM previsto -> `aau/runs/evidence/e1/design.{json,md}` |
| `e1_train.sbatch` + `e1_train_body.sh` | training di una cella (corpo letto all'avvio del job); C2M si ferma dopo l'epoca 72; `WBES_E1_SMOKE=1` = prova sul nodo cpu |
| `e1_eval.sbatch` + `e1_eval_body.sh` | HIFI3D, FaceVerse con espressioni, FLAME, NoW sui due checkpoint, con le pipeline esistenti |
| `e1_summarize.py` + `.sbatch` | matrice, differenze appaiate, regola primaria, controlli -> `summary.md`, `matrix.csv`, `paired.csv`, `controls.csv` |
| `e1_zs_summarize.{sh,py}` | la tabella standard di `aau/zs3dmm` sui bracci HIFI3D delle celle (`aau/runs/evidence/e1/hifi/`) |
| `e1_launch.sh` | sottomette training, eval (afterok) e riepilogo; i job sono in `aau/runs/evidence/e1/jobs.md` |
| `smoke/` | split ridotti per lo smoke (`make_smoke_splits.py`) |
| `scratch/test_summarize_alias.py` | prova del percorso delle differenze di `e1_summarize.py` sui soli dati di C3M |
