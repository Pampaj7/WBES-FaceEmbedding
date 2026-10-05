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

(in attesa delle eval)

## Mancanti

- ctrl bfm: run non indicato
- ctrl ict: run non indicato
- e bfm: run non indicato
- e ict: run non indicato
