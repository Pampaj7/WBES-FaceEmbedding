# E congiunto BFM+ICT contro controllo congiunto

## Criterio, scritto il 5 ottobre prima di qualunque numero

**E congiunto passa se batte il controllo congiunto di almeno +0.03 sul margine medio BFM e non
perde piu' di 0.02 su ICT.** Reso operativo cosi', Delta = E - controllo:

1. **BFM**: margine medio = media sulle 30 coppie ordinate di topologie (protocollo mesh-pair
   cross-topologia: `compare_model_vs_chamfer_topology_breakdown.py --ordered_topology_pairs
   --topology_pair_mode cross_only`, scenario clean) di (Spearman latent - Spearman Chamfer) contro la GT,
   sui 104 soggetti BFM held-out dello split. Passa se Delta >= +0.03. Chamfer e' lo stesso nei
   due bracci (stessi soggetti, stesse mesh), quindi il Delta del margine e' il Delta del latent.
2. **ICT**: Spearman mesh-pair `all_cross` (stesso protocollo, tutte le coppie cross-topologia) sui 100
   soggetti ICT held-out estratti con seed 1234 dagli held-out ICT dello split. Passa se
   Delta >= -0.02.
3. Servono entrambe. Un seed solo: un passaggio va confermato su altri seed.

E' il criterio delle ablazioni v3 (`aau/runs/ablations_v3/summary.md`) con lo zero-shot ICT sostituito
dall'ICT in dominio, perche' qui ICT e' nel training. Le altre righe (gruppi crop/noisy/resample/all
di eval_by_topology, IC bootstrap per soggetto, WS3a Multiface) sono descrittive e non entrano nel verdetto.

## Disegno

- Sottoinsieme: BFM 500 soggetti (tutti) + ICT 1500 (`np.random.default_rng(1234).choice` sui 5000);
  split del trainer sull'unione (rebuild_subject_split, eval_fraction 0.2, seed 1234, come train_v2):
  BFM 396 train / 104 held-out, ICT 1204 / 296. GT del congiunto attuale (`datasets/JOINT_BFM_ICT/gt_matrix.npz`).
- Controllo: frame current, operatori del congiunto attuale (BFM cotangente area 1, ICT `train_ready`),
  niente token. E: frame rms, operatori robusti ad area 1 per entrambi i domini, token di taglia
  standardizzato per dominio sui soggetti di training del dominio (`size_token_joint_{bfm,ict}_s1234.json`).
- Ricetta v1, 120 epoche, cache in RAM (train_fast.py), batch a dominio singolo ed eval online su BFM
  (train_v2), seed 1234. Codice: `aau/cross3dmm/{joint_E_prep.py,train_joint_E.py,train_joint_E.sbatch,joint_E_eval.sbatch}`.

## Risultati

**Verdetto: PASSA (BFM Delta margine +0.0315 >= +0.03; ICT Delta all_cross +0.0008 >= -0.02).**

| braccio | dominio | soggetti | margine medio 30 celle (Δ) | latent medio 30 celle | Chamfer medio | mesh-pair all_cross latent [IC 95%] (Δ) | Chamfer all_cross | crop | noisy | resample | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctrl | bfm | 104 | 0.4711 | 0.8505 | 0.3794 | 0.8446 [0.810, 0.873] | 0.2669 | 0.8272 | 0.8612 | 0.8582 | 0.8446 |
| ctrl | ict | 100 | 0.4033 | 0.9729 | 0.5696 | 0.9725 [0.964, 0.978] | 0.3987 | 0.9613 | 0.9800 | 0.9775 | 0.9725 |
| e | bfm | 104 | 0.5026 (+0.0315) | 0.8820 | 0.3794 | 0.8736 [0.845, 0.895] (+0.0290) | 0.2669 | 0.8583 | 0.8944 | 0.8881 | 0.8736 |
| e | ict | 100 | 0.4043 (+0.0010) | 0.9739 | 0.5696 | 0.9733 [0.966, 0.979] (+0.0008) | 0.3987 | 0.9640 | 0.9772 | 0.9793 | 0.9733 |

Gruppi crop/noisy/resample/all: uno Spearman su tutte le coppie del gruppo (eval_cells.py, aggregazione di eval_by_topology), stessi soggetti.

## Controlli

- bfm: Chamfer per cella fra i due bracci, differenza massima 0.00e+00 (30 celle); latent per cella breakdown contro eval_cells, differenza massima ctrl 1.1e-06, e 1.2e-06; leak ctrl {'pair_metrics': 0, 'cells': 0, 'pair_metrics_eq_split': True, 'cells_eq_split': True}, e {'pair_metrics': 0, 'cells': 0, 'pair_metrics_eq_split': True, 'cells_eq_split': True}
- ict: Chamfer per cella fra i due bracci, differenza massima 0.00e+00 (30 celle); latent per cella breakdown contro eval_cells, differenza massima ctrl 3.3e-07, e 1.9e-07; leak ctrl {'pair_metrics': 0, 'cells': 0, 'pair_metrics_eq_split': True, 'cells_eq_split': True}, e {'pair_metrics': 0, 'cells': 0, 'pair_metrics_eq_split': True, 'cells_eq_split': True}

## WS3a Multiface duro, AUC b_vs_c (Δ rispetto al controllo congiunto)

Csv `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/multiface_ws3a_hard/summary_hard_jointE.csv`. Token di E su Multiface: `tokA` = raggio convertito in unita' BFM e standardizzato con le statistiche BFM del training, `tok0` = token neutro (0).

| metrica | tracked->tracked | tracked->crop | remesh->crop | tracked->noisy | down->up | crop->crop |
|---|---|---|---|---|---|---|
| latent_jointE_ctrl | 0.991 [0.972, 0.999] | 0.671 [0.400, 0.890] | 0.673 [0.388, 0.897] | 0.989 [0.969, 0.998] | 0.989 [0.967, 0.999] | 0.988 [0.954, 0.999] |
| latent_jointE_e_tokA | 0.999 [0.997, 1.000] (+0.007) | 0.579 [0.418, 0.750] (-0.092) | 0.564 [0.394, 0.743] (-0.108) | 0.992 [0.985, 0.997] (+0.004) | 0.913 [0.821, 0.980] (-0.076) | 0.977 [0.931, 0.996] (-0.011) |
| latent_jointE_e_tok0 | 0.998 [0.996, 1.000] (+0.007) | 0.571 [0.411, 0.739] (-0.100) | 0.560 [0.392, 0.738] (-0.113) | 0.991 [0.981, 0.996] (+0.002) | 0.891 [0.790, 0.972] (-0.098) | 0.971 [0.917, 0.995] (-0.017) |

## Sorgenti

- `ctrl`: training `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/joint_E/joint_E_ctrl_s1234_1055737`; eval `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/joint_E/eval/joint_E_ctrl_s1234_1055737__bfm`, `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/joint_E/eval/joint_E_ctrl_s1234_1055737__ict`
- `e`: training `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/joint_E/joint_E_e_s1234_1055738`; eval `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/joint_E/eval/joint_E_e_s1234_1055738__bfm`, `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/joint_E/eval/joint_E_e_s1234_1055738__ict`
