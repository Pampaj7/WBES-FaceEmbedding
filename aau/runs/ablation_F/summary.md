# Variante F: crop casuali in training (ipotesi H6)

**Criterio, fissato prima dei numeri (BOARD_DIARY, "Ipotesi H6"):** almeno +0.05 di AUC sulle
coppie con crop di Multiface **oppure** +0.05 di Spearman sulle coppie con crop di REMESH, senza
perdere piu' di 0.02 altrove. Resi operativi cosi':
- Multiface: media dei Delta di AUC b_vs_c sulle tre coppie con crop (tracked->crop, remesh->crop, crop->crop);
- REMESH: Delta Spearman del gruppo `crop` di eval_by_topology.py (100 soggetti held-out, seed 1234);
- altrove: Delta >= -0.02 su REMESH noisy e resample e su Multiface tracked->tracked, tracked->noisy, down->up.
Delta = F - controllo. Controllo appaiato: `remesh_v1recipe_current_s1234_1055026` (stessa ricetta v1, stesso
wrapper con cache, seed 1234, frame current). F differisce solo per la vista dati `datasets/REMESH/view_F`:
5 crop casuali in piu' per soggetto di training, che il trainer etichetta `crop` (statistiche in
`aau/runs/ablation_F/crops_stats.md`). Multiface ha 13 soggetti: gli IC delle AUC sono larghi.

## REMESH, Spearman vs D_GT per gruppo (eval_by_topology.py, frame current)

Budget: F 120/120 (best 112), controllo 120/120 (best 114).

| gruppo | coppie | F | controllo | Δ |
|---|---|---|---|---|
| crop | 49500 | 0.6514 | 0.7653 | -0.1139 |
| noisy | 39600 | 0.7446 | 0.7829 | -0.0383 |
| resample | 59400 | 0.7806 | 0.8048 | -0.0242 |
| all | 148500 | 0.7214 | 0.7816 | -0.0602 |

## Multiface WS3a duro, AUC auc_b_vs_c [IC 95% bootstrap sui soggetti]

Metriche `latent_abl_F` e `latent_ctrl_s1234` in `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/multiface_ws3a_hard/summary_hard_F.csv`.

| topologie | F | controllo | Δ |
|---|---|---|---|
| tracked->tracked | 0.986 [0.956, 0.999] | 0.999 [0.995, 1.000] | -0.013 |
| tracked->crop | 0.711 [0.473, 0.867] | 0.697 [0.430, 0.874] | +0.014 |
| remesh->crop | 0.719 [0.478, 0.891] | 0.692 [0.415, 0.878] | +0.028 |
| crop->crop | 0.991 [0.948, 1.000] | 0.991 [0.950, 1.000] | -0.000 |
| tracked->noisy | 0.984 [0.947, 0.998] | 0.999 [0.996, 1.000] | -0.015 |
| down->up | 0.979 [0.944, 0.997] | 0.982 [0.936, 0.999] | -0.003 |

## Verdetto

- Guadagno sul crop: REMESH -0.1139, Multiface (media 3 coppie) +0.014: NON raggiunto (soglia +0.05).
- Perdite altrove oltre -0.02: REMESH noisy -0.038, REMESH resample -0.024.
- **NON PASSA** il criterio fissato.
