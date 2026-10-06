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

| braccio | crop (Δ) | noisy (Δ) | resample (Δ) | all (Δ) | ICT all_cross [CI 95%] (Δ) | esito |
| --- | --- | --- | --- | --- | --- | --- |
| controllo | 0.7653 | 0.7829 | 0.8048 | 0.7816 | 0.3693 [0.308, 0.429] | controllo |
| pot_m55 | 0.4280 (-0.3373) | 0.3576 (-0.4253) | 0.4352 (-0.3696) | 0.4014 (-0.3802) | 0.1314 [0.080, 0.178] (-0.2380) | NON PASSA (crop -0.337, ICT -0.238) |
| pot_dual | 0.7568 (-0.0085) | 0.7780 (-0.0050) | 0.7923 (-0.0125) | 0.7710 (-0.0105) | 0.3151 [0.255, 0.374] (-0.0543) | NON PASSA (crop -0.009, ICT -0.054) |

Coppie per gruppo (BFM): crop 49500, noisy 39600, resample 59400, all 148500

## Sorgenti

- `controllo`: checkpoint `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/remesh_v1recipe_current_s1234_1055026`; eval BFM `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_frame/remesh_v1recipe_current_s1234_1055026_maxabs`; ICT `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d_41fb92c8/ict_zeroshot_pilot_clean` (data `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT/eval_view_heldout`)
- `pot_m55`: checkpoint `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/pilot_pot_m55_s1234_1056124`; eval BFM `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_frame/pilot_pot_m55_s1234_1056124_maxabs`; ICT `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d_40210e73/ict_zeroshot_pilot_clean` (data `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT/eval_view_heldout_pot055`)
- `pot_dual`: checkpoint `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/pilot_pot_dual_s1234_1056125`; eval BFM `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_frame/pilot_pot_dual_s1234_1056125_maxabs`; ICT `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w103_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__84e790e5_bcfda4f6/ict_zeroshot_pilot_clean` (data `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT/eval_view_heldout`)
