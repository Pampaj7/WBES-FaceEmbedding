# Celle di sviluppo mancanti (matrice completa, solo dev, niente Ava-256)

Generato da `aau/full_matrix/fm_celle.py` dai CSV di `fm/` (= `fm_eval.py` rilanciato su FaceVerse neutra e FaMoS con le colonne che mancavano; uscite complete in `fm/`). Insieme `nocrop`, passo `comune`, IC 95% percentile, 1000 repliche per soggetto col seme della matrice (FaceVerse neutra 271049; FaMoS `famos_eval.bootstrap_counts(15, n, 1234)`).

| riga del paper | colonna | dominio | GT | Spearman [IC 95%] | righe | soggetti | fonte |
|---|---|---|---|---|---|---|---|
| 10x identities, d_F cal. (post hoc) | `factorizedc3m_e123|form_cal` | faceverse_neutral | FR | 0.347 [0.268, 0.419] | 99000 | 100 | `aau/runs/evidence/celle_mancanti/fm/spearman.csv:10` |
| 10x identities, d_P | `factorizedc3m_e123|shape` | faceverse_neutral | SR | 0.382 [0.293, 0.457] | 99000 | 100 | `aau/runs/evidence/celle_mancanti/fm/spearman.csv:77` |
| NICP to a template, size-normalized | `cs_nicp_template` | faceverse_neutral | SR | 0.175 [0.092, 0.257] | 99000 | 100 | `aau/runs/evidence/celle_mancanti/fm/spearman.csv:86` |
| NICP to a template, size-normalized (post hoc) | `cs_nicp_template` | famos | SR | 0.379 [0.087, 0.647] | 105 | 15 | `aau/runs/evidence/celle_mancanti/fm/spearman.csv:11772` |

## Calcolo delle colonne nuove

- C3M e123 sulla FaceVerse neutra: `v3_work/trainer/ablations/c3f/eval_body.sh`, passo `fvn`, checkpoint `epoch123_ema.pth` di `trainer_v3/factorized/c3m/runs` (come e205), d_F cal. con la c di `factorized_calibration.csv` (`factorizedc3m_e123`, la stessa delle altre colonne di C3M e123).
- NICP su template cs, FaceVerse neutra: `aau/baselines_mm/blmm_template.py --views faceverse_neutral --modes cs` (definizione di `blmm.build_template` / `blmm.enroll`, come le altre viste).
- NICP su template cs, FaMoS: `aau/full_matrix/fm_compute.py famos-template --mode cs` (lo stesso template delle 80 persone FaMoS TRAIN del modo mm, nelle coordinate cs di `blmm.work_coords`); post hoc come il template mm di FaMoS (costruito qui).

## Controllo: le altre celle non cambiano

Righe di `spearman.csv` con le stesse chiavi nella matrice completa: 11358 su 11360 delle due viste; n_righe diversi 0; scarto massimo punto 0.00e+00, IC 0.00e+00. `deltas.csv` (senza `migliore_vs_migliore`, scelto sul punto): 14280 righe confrontate, scarto massimo 0.00e+00. Dettagli in `controls.json`.

Righe della matrice completa assenti dalla rilettura: famos fr cs_nicp_p2tri, famos sr cs_nicp_p2tri. Causa (non legata alle colonne nuove): dal commit 21bfa5e (15 ottobre, dopo la matrice completa) `blmm.METRICS` ha il passo `nicp_corr`, che contiene `nicp_p2tri`; `fact_paired.famos_frame` scorre i passi e per `nicp_corr` (nessuno shard su FaMoS) sovrascrive `cs_nicp_p2tri` con una matrice tutta NaN, che `fm_eval` dichiara non applicabile. La maschera di FaMoS resta 105 righe (la colonna esce intera, non le righe), quindi le altre celle, comprese le 4 nuove, non cambiano.

