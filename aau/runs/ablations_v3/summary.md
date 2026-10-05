# Ablazioni v3: frame rms + token di taglia (B), operatori robusti ad area 1 (C), combinazione (E)

## Criterio di successo (BOARD_DIARY, "Progetto della versione nuova e ablazioni", fissato prima dei risultati)

> Criterio per ciascuna, fissato ora: almeno +0.03 sul margine latent − Chamfer medio sulle 30 coppie
> o +0.05 sulle coppie con crop, senza perdere più di 0.02 sullo zero-shot ICT.

Operativamente, rispetto al controllo appaiato `remesh_v1recipe_current_s1234_1055026` (stessa ricetta v1,
stesso wrapper train_fast.py con cache in RAM, `--frame current`, operatori standard, seed 1234):

1. Δ margine medio sulle 30 celle ordinate (eval_cells.py, 100 held-out) **>= +0.03**,
   oppure Δ Spearman del gruppo `crop` (uno Spearman su tutte le coppie con crop, come il pilota pot)
   **>= +0.05**;
2. **e** zero-shot ICT held-out (WBES_EVAL_SEED=1234, scenario clean, all_cross) **>= controllo − 0.02**.

Un seed solo: un passaggio va confermato su altri seed.

## Bracci

- `B`: frame rms dell'input + token di taglia (log del raggio rms della mesh grezza, pesato per area,
  standardizzato su media e std delle mesh dei 400 soggetti di training BFM), concatenato dopo il pooling
  mean+max, prima della proiezione a 256 (`aau/models/ablation_hooks.py`). Su ICT il token e' standardizzato
  sui 4500 soggetti di training ICT (le coordinate grezze ICT non sono in unita' BFM).
- `C`: operatori robusti ad area 1 (`robust_laplacian.mesh_laplacian`, mollify di default, mesh centrata e
  riscalata ad area 1, k_eig 128; frame tangenti e gradienti come `compute_operators`), frame standard.
- `E`: B + C.

## Risultati

| braccio | crop (Δ) | noisy (Δ) | resample (Δ) | all (Δ) | latent medio 30 celle (Δ) | margine − Chamfer 30 celle (Δ) | ICT all_cross [CI 95%] (Δ) | esito |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controllo | 0.7653 | 0.7829 | 0.8048 | 0.7816 | 0.7870 | 0.4436 | 0.3693 [0.308, 0.429] | controllo |
| B | 0.7773 (+0.0120) | 0.8414 (+0.0585) | 0.8320 (+0.0272) | 0.8086 (+0.0270) | 0.8177 (+0.0308) | 0.4744 (+0.0308) | 0.1350 [0.105, 0.171] (-0.2343) | NON PASSA (margine +0.031, crop +0.012, ICT -0.234) |
| C | 0.7489 (-0.0165) | 0.8050 (+0.0221) | 0.7917 (-0.0132) | 0.7742 (-0.0074) | 0.7821 (-0.0049) | 0.4387 (-0.0049) | 0.2822 [0.233, 0.332] (-0.0871) | NON PASSA (margine -0.005, crop -0.016, ICT -0.087) |
| E | 0.8000 (+0.0347) | 0.8644 (+0.0815) | 0.8437 (+0.0389) | 0.8296 (+0.0480) | 0.8357 (+0.0487) | 0.4923 (+0.0487) | 0.1145 [0.085, 0.147] (-0.2548) | NON PASSA (margine +0.049, crop +0.035, ICT -0.255) |

Coppie per gruppo (BFM): crop 49500, noisy 39600, resample 59400, all 148500; 30 celle da 4950 coppie. Chamfer medio sulle 30 celle (v1, stessi soggetti): 0.3433.

## Sorgenti

- `controllo`: training `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/remesh_v1recipe_current_s1234_1055026`; eval BFM `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ablations_v3/eval_bfm/remesh_v1recipe_current_s1234_1055026`; ICT `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d_41fb92c8/ict_zeroshot_abl3_clean` (data `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT/eval_view_heldout`)
- `B`: training `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/abl3_B_s1234_1055113`; eval BFM `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ablations_v3/eval_bfm/abl3_B_s1234_1055113`; ICT `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d_474a13fe/ict_zeroshot_abl3_clean` (data `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT/eval_view_heldout`)
- `C`: training `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/abl3_C_s1234_1055099`; eval BFM `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ablations_v3/eval_bfm/abl3_C_s1234_1055099`; ICT `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d_3f422c1c/ict_zeroshot_abl3_clean` (data `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT/eval_view_heldout_robust_area1`)
- `E`: training `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/abl3_E_s1234_1055114`; eval BFM `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ablations_v3/eval_bfm/abl3_E_s1234_1055114`; ICT `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d_c2a49876/ict_zeroshot_abl3_clean` (data `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT/eval_view_heldout_robust_area1`)
- Chamfer per cella: `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/eval_mixed_xtopo_rank0p5_id0p25_bs5_best_57bad1df/topology/topology_breakdown_summary.csv`
