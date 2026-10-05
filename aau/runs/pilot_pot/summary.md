# Pilota pot: operatori DiffusionNet col pozzo di potenziale (alpha 0.55) per la robustezza al crop

## Criterio di successo (fissato il 2026-10-04, prima di qualunque risultato)

Un braccio passa se, rispetto al controllo appaiato `remesh_v1recipe_current_s1234_1055026`
(rilancio a --mem=180G di 1054511, cancellato per thrashing di memoria a fine run;
stessa ricetta v1, stesso wrapper train_fast.py con cache, `--frame current`, seed 1234):

1. Spearman sulle coppie con `crop` (eval_by_topology, 100 soggetti held-out) **>= controllo + 0.05**;
2. zero-shot ICT held-out (WBES_EVAL_SEED=1234, scenario clean, protocollo mesh-pair, all_cross)
   **>= controllo - 0.02**.

Entrambe le condizioni. Un seed solo: e' un pilota, un passaggio va confermato su altri seed.

## Bracci

- `pot_m55`: operatori col pozzo (alpha 0.55, scala comune BFM 127507, `potential_operators.py
  --alpha-mode global`), pooling mean+max ristretto alla ROI del pozzo (roi_mask > 0.5).
- `dual`: due rami di operatori sullo stesso input (`aau/models/dn_dual_ops.py`): ogni blocco
  diffonde con la base standard e con quella del pozzo e concatena prima della MLP; width 103
  invece di 128 per avere gli stessi parametri (693140 contro 691584, +0.2%). Pooling pieno.
- Su ICT gli operatori col pozzo usano la scala comune di ICT (`calib_ict.json`), stesso alpha.

## Risultati

In attesa: training, eval appaiate e zero-shot ICT in coda (la tabella la riscrive aau/models/pilot_summary.py, job con afterany sulle eval).

Primo tentativo perso (1055016 m55 a epoca 93, 1055017 dual a epoca 16): morti alle 10:24 del
5 ottobre insieme a 1055540-1055545, che hanno saturato la quota da 1 TB; exit 1 senza traceback,
MaxRSS 39 e 62 GB su 180 e 200. Rifatti da zero (train_runner.py non ha resume: il checkpoint
non ha lo stato di ReduceLROnPlateau), su L40S, run dir su /tmp con sync nella home:
training 1056124 (m55) e 1056125 (dual); eval BFM 1056126/1056128, ICT 1056127/1056129, ICT del
controllo riusato (1055264), tabella 1056130.
