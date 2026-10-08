# Protocollo dichiarato prima del calcolo: competitori diretti sulla verita' geometrica di HIFI3D

Dichiarato il 2026-10-08 14:10 CEST, prima di calcolare qualunque descrittore o embedding dei competitori
(nessun numero di ShapeDNA, HKS, WKS, Uni3D o OpenShape esiste a quest'ora). Fonte dei competitori:
`literature/COMPETITORS_GEOM_2026-10-08.md`. Numeri gia' noti in anticipo: quelli di
`aau/runs/data_scale_ood/hifi/summary.md` (Chamfer eval nocrop 0.372, e108 nocrop 0.630) e di
`aau/runs/arcface_render_zs/results_hifi3d.md`.

**Dati.** Le 600 mesh (100 soggetti x 6 topologie) di `datasets/HIFI3D/eval_view/npz`, soggetti di
`select_subjects` seed 1234: gli stessi delle pair_metrics di e108 (controllato nel summary). Coordinate
native, nel frame delle mesh (HIFI3D: alto +y, naso +z), senza ricanonizzare. Vertici non referenziati dai
triangoli tolti prima di tutto (non sono superficie).

**Spettrali (senza pesi).** Operatore del repo (`diffusion_net.geometry.compute_operators`): Laplaciano
cotangente `potpourri3d.cotan_laplacian(denom_eps=1e-10)`, massa concentrata `potpourri3d.vertex_areas`
+ 1e-8 x media, `eigsh` shift-invert con sigma = 1e-8. Mesh scalata ad area totale 1 (equivale a
lambda_i x A). 101 autocoppie (lambda_0 ~ 0 compreso).
- ShapeDNA k=50 e k=100: (lambda_1, ..., lambda_k), lambda_0 escluso, nessun ripesamento. Distanza L2.
- HKS globale: HKS(x, t) = sum_{i=0..100} exp(-lambda_i t) phi_i(x)^2, autovettori M-ortonormali, NON
  scalata (forma "naturale" di pyFM). 100 tempi log-spaziati in [4 ln10 / (4 pi 100), 4 ln10 / (4 pi)]:
  la regola di Sun et al. 2009 (t_min = 4 ln10 / lambda_max, t_max = 4 ln10 / lambda_1) con gli
  autovalori della legge di Weyl per area 1 (lambda_k ~ 4 pi k), quindi la stessa griglia per tutte
  le mesh, fissata senza guardare i dati. Globale = media pesata per area (massa concentrata). Distanza L2.
- WKS globale: WKS(x, e) = sum_{i=1..100} exp(-(e - log lambda_i)^2 / (2 sigma^2)) phi_i(x)^2, NON
  scalata. Regola di pyFM `auto_WKS` con gli stessi autovalori di Weyl: e_min = log(4 pi),
  e_max = log(400 pi), sigma = 7 (e_max - e_min) / 100, 100 energie in [e_min + 2 sigma, e_max - 2 sigma].
  Globale = media pesata per area. Distanza L2.
- Perche' non scalate: la versione scalata di pyFM divide per sum_i exp(...) (HKS: la traccia del calore),
  e allora la media pesata per area vale 1/A per ogni mesh (sum_x m_x phi_i(x)^2 = 1): descrittore costante.
  Nota: anche le non scalate, mediate per area, dipendono solo dagli autovalori (sum_i exp(-lambda_i t) / A);
  si calcolano comunque puntuali e poi mediate, e l'identita' e' un controllo.

**Neurali (pesi pubblici, zero-shot).** Stessi punti per i due modelli: 10.000 punti campionati in modo
uniforme sulla superficie (triangolo estratto con probabilita' proporzionale all'area, coordinate
baricentriche uniformi), seme `stable_seed(1234, soggetto, topologia)`. Normalizzazione del loro codice:
centro = media dei punti, scala = raggio massimo (Uni3D `pc_norm`, OpenShape `normalize_pc`, identiche).
RGB costante 0.4 (il loro valore per dati senza colore). Embedding globale dell'encoder di punti;
distanza = 1 - coseno. fp32, eval, nessun dropout.
- Uni3D: la variante piu' grande, uni3d-g (`eva_giant_patch14_560`, pc_feat_dim 1408, embed_dim 1024,
  num_group 512, group_size 64, pc_encoder_dim 512, patch_dropout 0: `scripts/inference.sh`).
- OpenShape: `openshape-pointbert-vitg14-rgb`, PointBERT `scaling=4`, in_channel 6 (xyz + rgb),
  out_channel 1280 (README del repo).
- FPS: i due repo lo prendono da estensioni CUDA (pointnet2_ops, dgl); qui una FPS in torch puro che parte
  dall'indice 0 (come pointnet2_ops; dgl parte da un indice casuale), quindi deterministica.

**Frame.** Riga primaria: frame nativo. Ablazione dichiarata ora, per entrambi i neurali: rotazione nota
Rx(+90) (x, y, z) -> (x, -z, y), che porta l'alto +y in +z, l'asse verticale dei dati di training dei due
modelli (OpenShape `y_up`: scambia y e z prima della normalizzazione). Solo come ablazione.

**Misure.**
- Spearman con la GT `maxabs`, scenari `nocrop_cross` (PRIMARIO, 20 coppie ordinate di topologie),
  `all_cross` (30) e `subject_pair_mean` (media sulle 30 per coppia di soggetti). Righe = quelle delle
  pair_metrics di e108 (soggetto a < b). IC 95% bootstrap per soggetto, 1000 repliche, con
  `weighted_bootstrap_spearman` e i semi di `aau/zs3dmm/zs_summarize.py`.
- Delta appaiati competitore - e108 e competitore - Chamfer eval, sugli stessi tre scenari, con
  `paired_bootstrap` di zs_summarize.py (stesse repliche per i due lati).
- Controllo: le righe di riferimento (e108 latent, Chamfer eval, delta e108 - Chamfer eval) ricalcolate
  con gli stessi semi devono coincidere con `aau/runs/data_scale_ood/hifi/table_cells.csv` e `paired.csv`.
- Riconoscimento: protocollo di `aau/runs/arcface_render_zs/results_hifi3d.md` (funzioni di
  `zs_expr_summarize.py`, repliche `stable_seed(1234, "expr_recognition")`), rank-1, mAP, AUC, blocco
  nocrop primario e crop a parte; righe di riferimento ArcFace (3 viste), Chamfer faceBench e BFM+ICT
  ricalcolate e confrontate con `aau/runs/arcface_render_zs/hifi3d/recognition.csv`.

Nessun iperparametro si sceglie guardando le metriche: tutti quelli sopra sono fissati qui.

## Aggiunta -- 2026-10-08 14:28 CEST, prima di calcolare qualunque iscrizione su template

Gia' visti a quest'ora: i numeri degli spettrali (ShapeDNA, HKS, WKS) e la riproduzione esatta delle righe
di riferimento. Nessun numero esiste per la riga qui sotto, ne' per Uni3D e OpenShape.

**NICP su template (iscrizione).** Baseline di `aau/runs/indomain_recog` (protocollo, revisione 1, punto B),
riusata identica: funzioni `enroll`, `template_distances`, `_init`, `_enroll_one` di `aau/indomain/ir_template.py`
importate, non riscritte. Iscrizione di una mesh: `load_verts` (maxabs) + `sample_pts` di faceBench (4096
VERTICI estratti, non punti di superficie: e' la ricetta della baseline), seme `mesh_seed(soggetto, topologia)`
= crc32, ICP di similarita' template -> mesh, `nonrigid_icp_align` di faceBench, Procrustes con scala verso il
template. Distanza = L2 media per vertice fra le due iscrizioni (simmetrica).
Template: quello medio del 3DMM di HIFI3D, costruito con la ricetta di `build_template`: media vertice per
vertice delle `original` (maxabs) di 100 soggetti del pool di 500 che NON sono fra i 100 valutati (l'analogo
dei soggetti di training del congiunto nella ricetta originale; HIFI3D non ha training), scelti con
`rng(1234)` fra i 400 non valutati in ordine, poi maxabs; 4096 vertici del template scelti con `rng(0)`.
Il template di BFM/ICT di indomain_recog non si usa: regione e topologia diverse da `mask_face` di HIFI3D.
Stesse misure e stesse repliche degli altri competitori: Spearman (nocrop_cross primario, all_cross,
subject_pair_mean), delta appaiati contro e108 e contro Chamfer eval, riconoscimento.

*Nota di esecuzione, 2026-10-08 15:40 CEST (non cambia il protocollo):* gli embedding di Uni3D e OpenShape
sono stati calcolati su CPU (fp32, 32 core, job 1061608) e non su L40S. Il job L40S restava in coda: con 48G
partenza stimata 2026-10-09 22:21 (job 1061553), con 7G stimata 17:44 (job 1061588; l'unico nodo con L40S
libere aveva 8 GB di RAM non allocati); entrambi cancellati. Stesso codice, stessa precisione, stessi punti.
Il checkpoint di Uni3D contiene un `set` fuori dai pesi: caricato con `weights_only=True` ammettendo solo quel
tipo builtin. Caricamento `strict=True` riuscito per entrambi (Uni3D-g 1017.4 M parametri, OpenShape 32.3 M).

## Aggiunta -- 2026-10-09 00:44 CEST, revisione del critic, prima di calcolare le righe nuove

Gia' visti a quest'ora: tutti i numeri del summary dell'8 ottobre; quelli di
`aau/runs/data_scale_ood/arcface_vs_scale_hifi3d.md` (riconoscimento di e108, nocrop: rank-1 0.782) e di
`aau/runs/evidence/e8/methods_spearman.csv` (GT unificata, nocrop: e108 0.301, Chamfer eval 0.230). Nessun numero
esiste per le righe 2, 3, 4 qui sotto ne' per i competitori con la GT unificata. Le sezioni dell'8 ottobre restano
come sono; le righe nuove stanno in una sezione datata del summary.

1. **Riconoscimento di e108 (correzione).** Il summary dell'8 ottobre dice che e108 non ha la riga di
   riconoscimento perche' mancano gli embedding per mesh: e' FALSO, esistono
   (`aau/runs/ws_hifi3d/data_328f2bfc1a/scale_e108_embed`, gli stessi di arcface_vs_scale_hifi3d.md). Si aggiunge
   la riga, letta con `zs_arcface_vs_scale.distances_b` (stesso checkpoint del breakdown, controllato), sulle
   stesse repliche delle altre righe di riconoscimento, nei blocchi nocrop (primario) e crop, con i delta appaiati
   e108 - NICP su template, e108 - ICP rigido + Chamfer, e108 - ArcFace (3 viste), e108 - Chamfer faceBench.
   Controllo: la riga nocrop deve coincidere con `arcface_vs_scale_hifi3d/recognition.csv`.
2. **GT unificata.** Tutte le righe di Spearman (e108, Chamfer eval, tutti i competitori e le ablazioni) anche con
   la GT unificata di `datasets/UNIFIED_GT/eval/hifi3d_gt_matrix.npz` (v3_work/unified_gt/make_eval_gt.py),
   sostituita per nome di soggetto con `zs_summarize.with_gt`: stesse righe, stessi scenari, STESSI semi delle
   righe maxabs (cambia solo la GT, come in v3_work/unified_gt/eval_methods.py). Delta appaiati competitore - e108
   e competitore - Chamfer eval con la GT unificata. Controllo: e108, Chamfer eval e la differenza appaiata
   e108 - Chamfer eval devono coincidere con `aau/runs/evidence/e8/methods_*.csv` (GT `unified`, che per HIFI3D
   coincide con questa a 3e-8).
3. **Frame esatto del training (ablazione dichiarata).** Rx(+90) non e' la trasformazione del training di
   OpenShape: `src/data.py` (commit abe5aa42b7c9) fa `xyz[:, [1, 2]] = xyz[:, [2, 1]]` (scambio y/z, una
   RIFLESSIONE, det -1), poi `normalize_pc` e, in training, una rotazione casuale attorno a z. Riga nuova
   OpenShape con lo scambio y/z esatto, (x, y, z) -> (x, z, y), poi la stessa normalizzazione; tutto il resto
   (punti, semi, RGB, FPS) come le righe dell'8 ottobre. Uni3D, stessa verifica (commit 64e03c3c42c1): il dataset
   di pre-training (`data/datasets.py`, `Ensembled_embedding`) applica `pc_norm` SENZA scambio e ruota attorno a y
   (`rotate_point_cloud`); lo scambio y/z c'e' solo nei dataset di valutazione con `--openshape_setting`, che
   `scripts/pretrain.sh` e `scripts/inference.sh` non passano. La trasformazione esatta di Uni3D e' quindi
   l'identita', cioe' la riga primaria gia' calcolata: nessuna riga nuova per Uni3D. Le righe primarie restano
   quelle del frame nativo.
4. **ShapeDNA normalizzata.** Accanto alla riga grezza, ShapeDNA k=50 e k=100 con gli autovalori divisi per il
   primo non nullo, (lambda_1, ..., lambda_k) / lambda_1 (normalizzazione di scala di Reuter et al. 2006),
   distanza L2, dagli stessi autovalori. Le righe dell'8 ottobre sono gia' riscalate per area (mesh ad area 1,
   cioe' lambda_i x A): la variante "riscalata per area" coincide con quelle e non si ripete.

Stesse misure e stesse repliche delle altre righe (Spearman con IC e delta appaiati con GT maxabs e unificata,
riconoscimento). Nessun iperparametro nuovo.

---

# Risultati: competitori diretti su HIFI3D (verita' geometrica)

Soggetti: 100 (`select_subjects`, seed 1234; gli stessi delle pair_metrics di e108, controllato), mesh da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/HIFI3D/eval_view/npz`, e108 e Chamfer eval da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/scale_e108_topology`. GT `maxabs`. CI 95% bootstrap per soggetto, 1000 repliche; delta appaiati sulle stesse repliche per i due lati.

## Spearman con la GT maxabs (mesh-pair, clean)

| metodo | nocrop_cross (PRIMARIO) | all_cross | subject_pair_mean |
| --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 (riferimento) | 0.630 [0.57, 0.69] | 0.541 [0.47, 0.61] | 0.795 [0.74, 0.84] |
| Chamfer eval (riferimento) | 0.372 [0.32, 0.42] | 0.336 [0.29, 0.38] | 0.743 [0.68, 0.80] |
| ShapeDNA k=50 | 0.108 [0.08, 0.14] | 0.074 [0.05, 0.10] | 0.298 [0.20, 0.40] |
| ShapeDNA k=100 | 0.031 [0.02, 0.04] | 0.023 [0.01, 0.03] | 0.249 [0.11, 0.38] |
| HKS globale | 0.077 [0.04, 0.11] | 0.043 [0.02, 0.06] | 0.239 [0.12, 0.35] |
| WKS globale | 0.025 [0.01, 0.04] | 0.028 [0.01, 0.05] | 0.201 [0.07, 0.32] |
| NICP su template (iscrizione) | 0.351 [0.27, 0.43] | 0.195 [0.15, 0.25] | 0.335 [0.23, 0.42] |
| Uni3D-g | 0.132 [0.09, 0.18] | 0.140 [0.10, 0.19] | 0.359 [0.25, 0.45] |
| OpenShape PointBERT | 0.138 [0.10, 0.18] | 0.120 [0.08, 0.16] | 0.304 [0.19, 0.41] |

### Delta appaiati: competitore - e108 e competitore - Chamfer eval

| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |
| --- | --- | --- | --- |
| ShapeDNA k=50 - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.522 [-0.576, -0.462] (0.108 / 0.630) | 1.000 |
| ShapeDNA k=50 - BFM+ICT+GNM (10^5), e108 | all_cross | -0.467 [-0.529, -0.402] (0.074 / 0.541) | 1.000 |
| ShapeDNA k=50 - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.497 [-0.600, -0.396] (0.298 / 0.795) | 1.000 |
| ShapeDNA k=100 - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.598 [-0.655, -0.536] (0.031 / 0.630) | 1.000 |
| ShapeDNA k=100 - BFM+ICT+GNM (10^5), e108 | all_cross | -0.518 [-0.577, -0.456] (0.023 / 0.541) | 1.000 |
| ShapeDNA k=100 - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.547 [-0.679, -0.411] (0.249 / 0.795) | 1.000 |
| HKS globale - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.553 [-0.607, -0.491] (0.077 / 0.630) | 1.000 |
| HKS globale - BFM+ICT+GNM (10^5), e108 | all_cross | -0.498 [-0.560, -0.433] (0.043 / 0.541) | 1.000 |
| HKS globale - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.556 [-0.669, -0.452] (0.239 / 0.795) | 1.000 |
| WKS globale - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.605 [-0.663, -0.539] (0.025 / 0.630) | 1.000 |
| WKS globale - BFM+ICT+GNM (10^5), e108 | all_cross | -0.513 [-0.573, -0.451] (0.028 / 0.541) | 1.000 |
| WKS globale - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.595 [-0.733, -0.474] (0.201 / 0.795) | 1.000 |
| NICP su template (iscrizione) - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.279 [-0.381, -0.176] (0.351 / 0.630) | 1.000 |
| NICP su template (iscrizione) - BFM+ICT+GNM (10^5), e108 | all_cross | -0.346 [-0.430, -0.267] (0.195 / 0.541) | 1.000 |
| NICP su template (iscrizione) - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.460 [-0.571, -0.355] (0.335 / 0.795) | 1.000 |
| Uni3D-g - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.498 [-0.565, -0.425] (0.132 / 0.630) | 1.000 |
| Uni3D-g - BFM+ICT+GNM (10^5), e108 | all_cross | -0.401 [-0.475, -0.327] (0.140 / 0.541) | 1.000 |
| Uni3D-g - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.436 [-0.535, -0.328] (0.359 / 0.795) | 1.000 |
| OpenShape PointBERT - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.492 [-0.562, -0.414] (0.138 / 0.630) | 1.000 |
| OpenShape PointBERT - BFM+ICT+GNM (10^5), e108 | all_cross | -0.421 [-0.495, -0.344] (0.120 / 0.541) | 1.000 |
| OpenShape PointBERT - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.491 [-0.601, -0.373] (0.304 / 0.795) | 1.000 |
| ShapeDNA k=50 - Chamfer eval | nocrop_cross | -0.264 [-0.315, -0.213] (0.108 / 0.372) | 1.000 |
| ShapeDNA k=50 - Chamfer eval | all_cross | -0.262 [-0.311, -0.214] (0.074 / 0.336) | 1.000 |
| ShapeDNA k=50 - Chamfer eval | subject_pair_mean | -0.445 [-0.562, -0.326] (0.298 / 0.743) | 1.000 |
| ShapeDNA k=100 - Chamfer eval | nocrop_cross | -0.341 [-0.385, -0.291] (0.031 / 0.372) | 1.000 |
| ShapeDNA k=100 - Chamfer eval | all_cross | -0.313 [-0.356, -0.270] (0.023 / 0.336) | 1.000 |
| ShapeDNA k=100 - Chamfer eval | subject_pair_mean | -0.495 [-0.635, -0.366] (0.249 / 0.743) | 1.000 |
| HKS globale - Chamfer eval | nocrop_cross | -0.295 [-0.350, -0.239] (0.077 / 0.372) | 1.000 |
| HKS globale - Chamfer eval | all_cross | -0.293 [-0.338, -0.244] (0.043 / 0.336) | 1.000 |
| HKS globale - Chamfer eval | subject_pair_mean | -0.504 [-0.636, -0.388] (0.239 / 0.743) | 1.000 |
| WKS globale - Chamfer eval | nocrop_cross | -0.347 [-0.393, -0.300] (0.025 / 0.372) | 1.000 |
| WKS globale - Chamfer eval | all_cross | -0.308 [-0.355, -0.264] (0.028 / 0.336) | 1.000 |
| WKS globale - Chamfer eval | subject_pair_mean | -0.542 [-0.696, -0.408] (0.201 / 0.743) | 1.000 |
| NICP su template (iscrizione) - Chamfer eval | nocrop_cross | -0.021 [-0.109, +0.065] (0.351 / 0.372) | 0.684 |
| NICP su template (iscrizione) - Chamfer eval | all_cross | -0.141 [-0.204, -0.079] (0.195 / 0.336) | 1.000 |
| NICP su template (iscrizione) - Chamfer eval | subject_pair_mean | -0.408 [-0.506, -0.310] (0.335 / 0.743) | 1.000 |
| Uni3D-g - Chamfer eval | nocrop_cross | -0.240 [-0.299, -0.183] (0.132 / 0.372) | 1.000 |
| Uni3D-g - Chamfer eval | all_cross | -0.196 [-0.253, -0.131] (0.140 / 0.336) | 1.000 |
| Uni3D-g - Chamfer eval | subject_pair_mean | -0.384 [-0.510, -0.269] (0.359 / 0.743) | 1.000 |
| OpenShape PointBERT - Chamfer eval | nocrop_cross | -0.235 [-0.295, -0.172] (0.138 / 0.372) | 1.000 |
| OpenShape PointBERT - Chamfer eval | all_cross | -0.216 [-0.275, -0.153] (0.120 / 0.336) | 1.000 |
| OpenShape PointBERT - Chamfer eval | subject_pair_mean | -0.439 [-0.571, -0.321] (0.304 / 0.743) | 1.000 |

Riferimento, stessi semi del summary esistente: BFM+ICT+GNM (10^5), e108 - Chamfer eval

| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |
| --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 - Chamfer eval | nocrop_cross | +0.258 [+0.208, +0.301] (0.630 / 0.372) | 0.000 |
| BFM+ICT+GNM (10^5), e108 - Chamfer eval | all_cross | +0.205 [+0.152, +0.252] (0.541 / 0.336) | 0.000 |
| BFM+ICT+GNM (10^5), e108 - Chamfer eval | subject_pair_mean | +0.052 [-0.004, +0.106] (0.795 / 0.743) | 0.039 |

## Ablazione del frame: rotazione nota Rx(+90) (alto +y -> +z), solo neurali

La riga primaria resta quella del frame nativo.

| metodo | nocrop_cross (PRIMARIO) | all_cross | subject_pair_mean |
| --- | --- | --- | --- |
| Uni3D-g, Rx(+90) (ablazione) | 0.165 [0.13, 0.20] | 0.167 [0.12, 0.20] | 0.431 [0.33, 0.52] |
| OpenShape PointBERT, Rx(+90) (ablazione) | 0.291 [0.21, 0.37] | 0.239 [0.17, 0.31] | 0.435 [0.32, 0.55] |

| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |
| --- | --- | --- | --- |
| Uni3D-g, Rx(+90) (ablazione) - Uni3D-g | nocrop_cross | +0.033 [+0.003, +0.064] (0.165 / 0.132) | 0.013 |
| Uni3D-g, Rx(+90) (ablazione) - Uni3D-g | all_cross | +0.027 [-0.005, +0.059] (0.167 / 0.140) | 0.057 |
| Uni3D-g, Rx(+90) (ablazione) - Uni3D-g | subject_pair_mean | +0.072 [-0.012, +0.167] (0.431 / 0.359) | 0.054 |
| OpenShape PointBERT, Rx(+90) (ablazione) - OpenShape PointBERT | nocrop_cross | +0.153 [+0.077, +0.224] (0.291 / 0.138) | 0.000 |
| OpenShape PointBERT, Rx(+90) (ablazione) - OpenShape PointBERT | all_cross | +0.119 [+0.056, +0.185] (0.239 / 0.120) | 0.000 |
| OpenShape PointBERT, Rx(+90) (ablazione) - OpenShape PointBERT | subject_pair_mean | +0.131 [+0.002, +0.255] (0.435 / 0.304) | 0.025 |
| Uni3D-g, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.465 [-0.528, -0.390] (0.165 / 0.630) | 1.000 |
| Uni3D-g, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | all_cross | -0.374 [-0.443, -0.305] (0.167 / 0.541) | 1.000 |
| Uni3D-g, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.364 [-0.462, -0.269] (0.431 / 0.795) | 1.000 |
| OpenShape PointBERT, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.339 [-0.404, -0.273] (0.291 / 0.630) | 1.000 |
| OpenShape PointBERT, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | all_cross | -0.302 [-0.368, -0.228] (0.239 / 0.541) | 1.000 |
| OpenShape PointBERT, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.361 [-0.466, -0.259] (0.435 / 0.795) | 1.000 |

## Riconoscimento d'identita' (protocollo di `aau/runs/arcface_render_zs/results_hifi3d.md`)

### PRIMARIO: 5 topologie senza crop

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.984 [0.974, 0.992] | 0.991 [0.985, 0.996] | 0.989 [0.985, 0.993] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] | 0 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0 |
| Rigid ICP + NICP + P2Tri | 0.990 [0.982, 0.998] | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] | 1965 |
| BFM+ICT, convenzione BFM | 0.397 [0.363, 0.434] | 0.517 [0.489, 0.549] | 0.755 [0.740, 0.770] | 0 |
| ShapeDNA k=50 | 0.575 [0.564, 0.585] | 0.606 [0.598, 0.614] | 0.682 [0.681, 0.683] | 0 |
| ShapeDNA k=100 | 0.290 [0.271, 0.309] | 0.357 [0.340, 0.374] | 0.577 [0.573, 0.581] | 0 |
| HKS globale | 0.074 [0.056, 0.097] | 0.153 [0.131, 0.177] | 0.612 [0.602, 0.621] | 0 |
| WKS globale | 0.114 [0.098, 0.133] | 0.189 [0.173, 0.207] | 0.535 [0.530, 0.540] | 0 |
| NICP su template (iscrizione) | 0.875 [0.843, 0.905] | 0.930 [0.911, 0.947] | 0.998 [0.997, 0.999] | 0 |
| Uni3D-g | 0.539 [0.511, 0.566] | 0.593 [0.567, 0.619] | 0.695 [0.687, 0.701] | 0 |
| Uni3D-g, Rx(+90) (ablazione) | 0.610 [0.595, 0.624] | 0.654 [0.640, 0.667] | 0.715 [0.711, 0.719] | 0 |
| OpenShape PointBERT | 0.617 [0.596, 0.639] | 0.680 [0.663, 0.700] | 0.728 [0.722, 0.735] | 0 |
| OpenShape PointBERT, Rx(+90) (ablazione) | 0.602 [0.574, 0.630] | 0.680 [0.655, 0.705] | 0.807 [0.789, 0.826] | 0 |

#### Delta appaiati, competitore - riferimento (stesse repliche)

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ShapeDNA k=50 | Chamfer (faceBench, 4096 pt) | +0.097 [+0.063, +0.128] (0.000) | +0.042 [+0.012, +0.068] (0.002) | -0.100 [-0.115, -0.085] (1.000) |
| ShapeDNA k=50 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.409 [-0.422, -0.396] (1.000) | -0.385 [-0.394, -0.376] (1.000) | -0.307 [-0.311, -0.303] (1.000) |
| ShapeDNA k=100 | Chamfer (faceBench, 4096 pt) | -0.188 [-0.222, -0.154] (1.000) | -0.207 [-0.238, -0.177] (1.000) | -0.205 [-0.219, -0.190] (1.000) |
| ShapeDNA k=100 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.694 [-0.713, -0.673] (1.000) | -0.634 [-0.652, -0.617] (1.000) | -0.412 [-0.417, -0.406] (1.000) |
| HKS globale | Chamfer (faceBench, 4096 pt) | -0.403 [-0.441, -0.366] (1.000) | -0.411 [-0.445, -0.378] (1.000) | -0.170 [-0.185, -0.156] (1.000) |
| HKS globale | ArcFace, ombreggiato, 3 viste (riferimento) | -0.909 [-0.929, -0.885] (1.000) | -0.838 [-0.860, -0.814] (1.000) | -0.377 [-0.387, -0.367] (1.000) |
| WKS globale | Chamfer (faceBench, 4096 pt) | -0.363 [-0.401, -0.329] (1.000) | -0.376 [-0.408, -0.346] (1.000) | -0.248 [-0.262, -0.234] (1.000) |
| WKS globale | ArcFace, ombreggiato, 3 viste (riferimento) | -0.870 [-0.889, -0.849] (1.000) | -0.802 [-0.820, -0.784] (1.000) | -0.455 [-0.460, -0.448] (1.000) |
| NICP su template (iscrizione) | Chamfer (faceBench, 4096 pt) | +0.398 [+0.350, +0.438] (0.000) | +0.366 [+0.331, +0.396] (0.000) | +0.215 [+0.201, +0.231] (0.000) |
| NICP su template (iscrizione) | ArcFace, ombreggiato, 3 viste (riferimento) | -0.108 [-0.142, -0.075] (1.000) | -0.061 [-0.080, -0.042] (1.000) | +0.009 [+0.005, +0.013] (0.000) |
| Uni3D-g | Chamfer (faceBench, 4096 pt) | +0.062 [+0.024, +0.098] (0.000) | +0.029 [-0.006, +0.063] (0.052) | -0.088 [-0.102, -0.074] (1.000) |
| Uni3D-g | ArcFace, ombreggiato, 3 viste (riferimento) | -0.445 [-0.473, -0.416] (1.000) | -0.398 [-0.425, -0.371] (1.000) | -0.295 [-0.302, -0.288] (1.000) |
| Uni3D-g, Rx(+90) (ablazione) | Chamfer (faceBench, 4096 pt) | +0.132 [+0.099, +0.163] (0.000) | +0.090 [+0.062, +0.117] (0.000) | -0.068 [-0.082, -0.053] (1.000) |
| Uni3D-g, Rx(+90) (ablazione) | ArcFace, ombreggiato, 3 viste (riferimento) | -0.374 [-0.391, -0.357] (1.000) | -0.337 [-0.352, -0.323] (1.000) | -0.274 [-0.279, -0.269] (1.000) |
| OpenShape PointBERT | Chamfer (faceBench, 4096 pt) | +0.140 [+0.106, +0.173] (0.000) | +0.116 [+0.087, +0.146] (0.000) | -0.055 [-0.070, -0.039] (1.000) |
| OpenShape PointBERT | ArcFace, ombreggiato, 3 viste (riferimento) | -0.367 [-0.390, -0.344] (1.000) | -0.311 [-0.329, -0.290] (1.000) | -0.262 [-0.268, -0.254] (1.000) |
| OpenShape PointBERT, Rx(+90) (ablazione) | Chamfer (faceBench, 4096 pt) | +0.124 [+0.084, +0.164] (0.000) | +0.116 [+0.079, +0.150] (0.000) | +0.025 [+0.005, +0.042] (0.002) |
| OpenShape PointBERT, Rx(+90) (ablazione) | ArcFace, ombreggiato, 3 viste (riferimento) | -0.382 [-0.410, -0.352] (1.000) | -0.311 [-0.335, -0.286] (1.000) | -0.182 [-0.201, -0.165] (1.000) |

### A parte: crop (coppie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.990 [0.984, 0.996] | 0.994 [0.991, 0.998] | 0.994 [0.992, 0.996] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.472 [0.445, 0.499] | 0.536 [0.510, 0.564] | 0.699 [0.684, 0.715] | 0 |
| Rigid ICP + Chamfer | 0.284 [0.237, 0.338] | 0.423 [0.381, 0.470] | 0.888 [0.865, 0.911] | 0 |
| Rigid ICP + NICP + P2Tri | 0.777 [0.740, 0.810] | 0.854 [0.829, 0.877] | 0.997 [0.995, 0.998] | 1965 |
| BFM+ICT, convenzione BFM | 0.085 [0.057, 0.117] | 0.201 [0.171, 0.233] | 0.716 [0.699, 0.734] | 0 |
| ShapeDNA k=50 | 0.014 [0.003, 0.027] | 0.078 [0.063, 0.095] | 0.568 [0.555, 0.582] | 0 |
| ShapeDNA k=100 | 0.013 [0.003, 0.026] | 0.058 [0.046, 0.073] | 0.511 [0.507, 0.515] | 0 |
| HKS globale | 0.012 [0.002, 0.024] | 0.053 [0.039, 0.069] | 0.503 [0.499, 0.506] | 0 |
| WKS globale | 0.042 [0.026, 0.061] | 0.122 [0.104, 0.142] | 0.579 [0.568, 0.589] | 0 |
| NICP su template (iscrizione) | 0.031 [0.011, 0.054] | 0.099 [0.077, 0.122] | 0.640 [0.615, 0.668] | 0 |
| Uni3D-g | 0.535 [0.485, 0.582] | 0.620 [0.578, 0.659] | 0.787 [0.775, 0.799] | 0 |
| Uni3D-g, Rx(+90) (ablazione) | 0.513 [0.465, 0.559] | 0.628 [0.590, 0.664] | 0.808 [0.799, 0.814] | 0 |
| OpenShape PointBERT | 0.190 [0.153, 0.229] | 0.339 [0.303, 0.378] | 0.758 [0.743, 0.774] | 0 |
| OpenShape PointBERT, Rx(+90) (ablazione) | 0.161 [0.117, 0.205] | 0.257 [0.210, 0.306] | 0.708 [0.684, 0.734] | 0 |

#### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ShapeDNA k=50 | Chamfer (faceBench, 4096 pt) | -0.458 [-0.487, -0.429] (1.000) | -0.459 [-0.486, -0.433] (1.000) | -0.131 [-0.150, -0.112] (1.000) |
| ShapeDNA k=50 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.976 [-0.987, -0.963] (1.000) | -0.917 [-0.932, -0.900] (1.000) | -0.426 [-0.440, -0.412] (1.000) |
| ShapeDNA k=100 | Chamfer (faceBench, 4096 pt) | -0.459 [-0.488, -0.432] (1.000) | -0.478 [-0.506, -0.451] (1.000) | -0.188 [-0.204, -0.174] (1.000) |
| ShapeDNA k=100 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.977 [-0.989, -0.963] (1.000) | -0.936 [-0.949, -0.921] (1.000) | -0.484 [-0.488, -0.479] (1.000) |
| HKS globale | Chamfer (faceBench, 4096 pt) | -0.460 [-0.491, -0.431] (1.000) | -0.483 [-0.513, -0.455] (1.000) | -0.196 [-0.213, -0.181] (1.000) |
| HKS globale | ArcFace, ombreggiato, 3 viste (riferimento) | -0.978 [-0.990, -0.965] (1.000) | -0.941 [-0.955, -0.925] (1.000) | -0.492 [-0.495, -0.488] (1.000) |
| WKS globale | Chamfer (faceBench, 4096 pt) | -0.430 [-0.462, -0.399] (1.000) | -0.414 [-0.445, -0.385] (1.000) | -0.120 [-0.138, -0.105] (1.000) |
| WKS globale | ArcFace, ombreggiato, 3 viste (riferimento) | -0.948 [-0.965, -0.930] (1.000) | -0.872 [-0.890, -0.853] (1.000) | -0.416 [-0.426, -0.405] (1.000) |
| NICP su template (iscrizione) | Chamfer (faceBench, 4096 pt) | -0.441 [-0.475, -0.404] (1.000) | -0.438 [-0.469, -0.404] (1.000) | -0.059 [-0.084, -0.031] (1.000) |
| NICP su template (iscrizione) | ArcFace, ombreggiato, 3 viste (riferimento) | -0.959 [-0.980, -0.935] (1.000) | -0.896 [-0.917, -0.872] (1.000) | -0.354 [-0.379, -0.326] (1.000) |
| Uni3D-g | Chamfer (faceBench, 4096 pt) | +0.063 [+0.013, +0.111] (0.009) | +0.084 [+0.039, +0.127] (0.000) | +0.088 [+0.070, +0.106] (0.000) |
| Uni3D-g | ArcFace, ombreggiato, 3 viste (riferimento) | -0.455 [-0.505, -0.408] (1.000) | -0.374 [-0.416, -0.336] (1.000) | -0.207 [-0.219, -0.196] (1.000) |
| Uni3D-g, Rx(+90) (ablazione) | Chamfer (faceBench, 4096 pt) | +0.041 [-0.009, +0.091] (0.050) | +0.092 [+0.051, +0.132] (0.000) | +0.108 [+0.093, +0.122] (0.000) |
| Uni3D-g, Rx(+90) (ablazione) | ArcFace, ombreggiato, 3 viste (riferimento) | -0.477 [-0.524, -0.432] (1.000) | -0.366 [-0.405, -0.331] (1.000) | -0.187 [-0.195, -0.180] (1.000) |
| OpenShape PointBERT | Chamfer (faceBench, 4096 pt) | -0.282 [-0.325, -0.237] (1.000) | -0.197 [-0.239, -0.154] (1.000) | +0.059 [+0.039, +0.079] (0.000) |
| OpenShape PointBERT | ArcFace, ombreggiato, 3 viste (riferimento) | -0.800 [-0.838, -0.760] (1.000) | -0.655 [-0.691, -0.616] (1.000) | -0.236 [-0.252, -0.220] (1.000) |
| OpenShape PointBERT, Rx(+90) (ablazione) | Chamfer (faceBench, 4096 pt) | -0.311 [-0.356, -0.260] (1.000) | -0.279 [-0.329, -0.226] (1.000) | +0.009 [-0.017, +0.038] (0.246) |
| OpenShape PointBERT, Rx(+90) (ablazione) | ArcFace, ombreggiato, 3 viste (riferimento) | -0.829 [-0.870, -0.785] (1.000) | -0.737 [-0.784, -0.689] (1.000) | -0.287 [-0.310, -0.260] (1.000) |

~~E108 non ha una riga di riconoscimento: per i bracci di scala esistono solo le pair_metrics del breakdown (coppie fra soggetti diversi), non gli embedding per mesh che servono per le coppie stessa persona.~~ **Corretto il 2026-10-09: la frase era falsa.** Gli embedding per mesh di e108 esistono (`/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/scale_e108_embed`); la riga di riconoscimento e' nella sezione datata in fondo.


## Controlli

Righe di riferimento ricalcolate con gli stessi semi contro `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/data_scale_ood/hifi` (table_cells.csv, paired.csv; diff massima su punto e CI):

| riga | scenario | esistente | ricalcolato | max_abs_diff |
| --- | --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 | nocrop_cross | 0.6299 [0.5690, 0.6892] | 0.6299 [0.5690, 0.6892] | 0.000000 |
| BFM+ICT+GNM (10^5), e108 | all_cross | 0.5414 [0.4707, 0.6053] | 0.5414 [0.4707, 0.6053] | 0.000000 |
| BFM+ICT+GNM (10^5), e108 | subject_pair_mean | 0.7954 [0.7380, 0.8395] | 0.7954 [0.7380, 0.8395] | 0.000000 |
| Chamfer eval | nocrop_cross | 0.3721 [0.3217, 0.4181] | 0.3721 [0.3217, 0.4181] | 0.000000 |
| Chamfer eval | all_cross | 0.3359 [0.2907, 0.3819] | 0.3359 [0.2907, 0.3819] | 0.000000 |
| Chamfer eval | subject_pair_mean | 0.7431 [0.6765, 0.7962] | 0.7431 [0.6765, 0.7962] | 0.000000 |
| delta BFM+ICT+GNM (10^5), e108 - Chamfer eval | nocrop_cross | +0.2578 [+0.2076, +0.3010] | +0.2578 [+0.2076, +0.3010] | 0.000000 |
| delta BFM+ICT+GNM (10^5), e108 - Chamfer eval | all_cross | +0.2054 [+0.1518, +0.2515] | +0.2054 [+0.1518, +0.2515] | 0.000000 |
| delta BFM+ICT+GNM (10^5), e108 - Chamfer eval | subject_pair_mean | +0.0522 [-0.0039, +0.1061] | +0.0522 [-0.0039, +0.1061] | 0.000000 |

- riconoscimento, riproduzione di `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/hifi3d/recognition.csv` (stesse repliche): max |diff| su punto e CI di rank-1, mAP, AUC = 1.11e-16 su 10 righe (arcface_shaded_3v, chamfer, joint@bfm, nicp_p2tri, rigid_icp_chamfer)
- spettrali: identita' media-per-area = somma spettrale, errore relativo max 9.9e-15; mesh con piu' componenti connesse 0, con vertici non referenziati tolti 0; lambda_0 max 2.4e-04, lambda_1 min 8.02, lambda_100 mediano 1093 (Weyl, area 1: 1257)
- NICP su template (iscrizione): template = media di 100 original non valutate (9518 vertici, 4096 tenuti); iscrizioni fallite 0 su 600 (NaN, contate come +inf nel riconoscimento), mediana 2.79 s per mesh
- Uni3D-g: embedding di dimensione 1024, 10000 punti per mesh
- Uni3D-g, Rx(+90) (ablazione): embedding di dimensione 1024, 10000 punti per mesh
- OpenShape PointBERT: embedding di dimensione 1280, 10000 punti per mesh
- OpenShape PointBERT, Rx(+90) (ablazione): embedding di dimensione 1280, 10000 punti per mesh

## Aggiunta -- 2026-10-09: revisione del critic (protocollo, aggiunta della stessa data)

Le sezioni sopra sono quelle dell'8 ottobre, rigenerate con gli stessi semi e invariate, salvo la frase corretta sul riconoscimento di e108. Qui: (1) riconoscimento di e108; (2) tutte le righe di Spearman con la GT unificata; (3) OpenShape con lo scambio y/z esatto del training; (4) ShapeDNA con autovalori / lambda_1. Stesse repliche delle altre righe.

### 1. Riconoscimento di e108 (embedding di `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/scale_e108_embed`)

Stesso checkpoint del breakdown e stesse distanze latenti delle pair_metrics (controlli in fondo); stesse repliche (`expr_recognition`) di tutte le altre righe di riconoscimento.

#### PRIMARIO: 5 topologie senza crop

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 | 0.782 [0.754, 0.807] | 0.845 [0.822, 0.863] | 0.946 [0.934, 0.957] | 0 |
| NICP su template (iscrizione) | 0.875 [0.843, 0.905] | 0.930 [0.911, 0.947] | 0.998 [0.997, 0.999] | 0 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0 |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.984 [0.974, 0.992] | 0.991 [0.985, 0.996] | 0.989 [0.985, 0.993] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] | 0 |

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 | NICP su template (iscrizione) | -0.093 [-0.127, -0.055] (1.000) | -0.085 [-0.108, -0.061] (1.000) | -0.052 [-0.064, -0.041] (1.000) |
| BFM+ICT+GNM (10^5), e108 | Rigid ICP + Chamfer | -0.213 [-0.242, -0.188] (1.000) | -0.153 [-0.175, -0.134] (1.000) | -0.053 [-0.065, -0.042] (1.000) |
| BFM+ICT+GNM (10^5), e108 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.201 [-0.231, -0.174] (1.000) | -0.146 [-0.168, -0.126] (1.000) | -0.044 [-0.055, -0.032] (1.000) |
| BFM+ICT+GNM (10^5), e108 | Chamfer (faceBench, 4096 pt) | +0.305 [+0.278, +0.336] (0.000) | +0.280 [+0.258, +0.304] (0.000) | +0.163 [+0.153, +0.174] (0.000) |

#### A parte: crop

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 | 0.348 [0.299, 0.395] | 0.475 [0.433, 0.516] | 0.843 [0.822, 0.865] | 0 |
| NICP su template (iscrizione) | 0.031 [0.011, 0.054] | 0.099 [0.077, 0.122] | 0.640 [0.615, 0.668] | 0 |
| Rigid ICP + Chamfer | 0.284 [0.237, 0.338] | 0.423 [0.381, 0.470] | 0.888 [0.865, 0.911] | 0 |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.990 [0.984, 0.996] | 0.994 [0.991, 0.998] | 0.994 [0.992, 0.996] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.472 [0.445, 0.499] | 0.536 [0.510, 0.564] | 0.699 [0.684, 0.715] | 0 |

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 | NICP su template (iscrizione) | +0.317 [+0.263, +0.368] (0.000) | +0.377 [+0.331, +0.424] (0.000) | +0.204 [+0.175, +0.230] (0.000) |
| BFM+ICT+GNM (10^5), e108 | Rigid ICP + Chamfer | +0.064 [+0.001, +0.121] (0.024) | +0.052 [-0.001, +0.102] (0.028) | -0.044 [-0.070, -0.019] (0.999) |
| BFM+ICT+GNM (10^5), e108 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.642 [-0.692, -0.593] (1.000) | -0.519 [-0.562, -0.480] (1.000) | -0.151 [-0.172, -0.129] (1.000) |
| BFM+ICT+GNM (10^5), e108 | Chamfer (faceBench, 4096 pt) | -0.124 [-0.163, -0.086] (1.000) | -0.061 [-0.091, -0.029] (1.000) | +0.144 [+0.131, +0.158] (0.000) |

### 2. Spearman con la GT unificata (tutte le righe)

GT `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/UNIFIED_GT/eval/hifi3d_gt_matrix.npz` (v3_work/unified_gt/make_eval_gt.py: RMS pesato per area in mm sulla regione FLAME comune dopo Procrustes di similarita'), sostituita per nome di soggetto (`zs_summarize.with_gt`) nelle STESSE righe delle pair_metrics di e108; stessi scenari e STESSI semi delle righe maxabs (cambia solo la GT).

| metodo | nocrop_cross (PRIMARIO) | all_cross | subject_pair_mean |
| --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 (riferimento) | 0.301 [0.23, 0.36] | 0.260 [0.20, 0.31] | 0.410 [0.33, 0.50] |
| Chamfer eval (riferimento) | 0.230 [0.19, 0.27] | 0.222 [0.18, 0.27] | 0.455 [0.37, 0.54] |
| ShapeDNA k=50 | 0.128 [0.10, 0.15] | 0.088 [0.07, 0.11] | 0.340 [0.26, 0.42] |
| ShapeDNA k=100 | 0.027 [0.02, 0.04] | 0.019 [0.01, 0.03] | 0.164 [0.05, 0.29] |
| HKS globale | 0.075 [0.04, 0.11] | 0.042 [0.03, 0.06] | 0.238 [0.14, 0.34] |
| WKS globale | 0.029 [0.02, 0.04] | 0.036 [0.02, 0.05] | 0.212 [0.10, 0.32] |
| NICP su template (iscrizione) | 0.450 [0.35, 0.54] | 0.255 [0.19, 0.32] | 0.429 [0.32, 0.54] |
| Uni3D-g | 0.144 [0.10, 0.19] | 0.162 [0.12, 0.21] | 0.414 [0.30, 0.52] |
| OpenShape PointBERT | 0.184 [0.14, 0.23] | 0.167 [0.13, 0.21] | 0.433 [0.33, 0.52] |
| ShapeDNA k=50, lambda_i / lambda_1 | 0.107 [0.07, 0.15] | 0.074 [0.05, 0.10] | 0.300 [0.19, 0.41] |
| ShapeDNA k=100, lambda_i / lambda_1 | 0.097 [0.06, 0.13] | 0.068 [0.04, 0.10] | 0.298 [0.18, 0.41] |
| Uni3D-g, Rx(+90) (ablazione) | 0.227 [0.19, 0.26] | 0.232 [0.19, 0.27] | 0.599 [0.51, 0.68] |
| OpenShape PointBERT, Rx(+90) (ablazione) | 0.294 [0.22, 0.36] | 0.245 [0.18, 0.31] | 0.449 [0.35, 0.54] |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | 0.287 [0.21, 0.36] | 0.251 [0.18, 0.32] | 0.448 [0.35, 0.54] |

#### Delta appaiati con la GT unificata: competitore - e108 e competitore - Chamfer eval

| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |
| --- | --- | --- | --- |
| ShapeDNA k=50 - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.173 [-0.228, -0.110] (0.128 / 0.301) | 1.000 |
| ShapeDNA k=50 - BFM+ICT+GNM (10^5), e108 | all_cross | -0.172 [-0.220, -0.119] (0.088 / 0.260) | 1.000 |
| ShapeDNA k=50 - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.070 [-0.176, +0.027] (0.340 / 0.410) | 0.925 |
| ShapeDNA k=100 - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.274 [-0.334, -0.211] (0.027 / 0.301) | 1.000 |
| ShapeDNA k=100 - BFM+ICT+GNM (10^5), e108 | all_cross | -0.241 [-0.291, -0.187] (0.019 / 0.260) | 1.000 |
| ShapeDNA k=100 - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.246 [-0.375, -0.115] (0.164 / 0.410) | 1.000 |
| HKS globale - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.226 [-0.290, -0.154] (0.075 / 0.301) | 1.000 |
| HKS globale - BFM+ICT+GNM (10^5), e108 | all_cross | -0.218 [-0.271, -0.162] (0.042 / 0.260) | 1.000 |
| HKS globale - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.172 [-0.291, -0.056] (0.238 / 0.410) | 1.000 |
| WKS globale - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.272 [-0.338, -0.206] (0.029 / 0.301) | 1.000 |
| WKS globale - BFM+ICT+GNM (10^5), e108 | all_cross | -0.224 [-0.275, -0.173] (0.036 / 0.260) | 1.000 |
| WKS globale - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.198 [-0.333, -0.075] (0.212 / 0.410) | 1.000 |
| NICP su template (iscrizione) - BFM+ICT+GNM (10^5), e108 | nocrop_cross | +0.149 [+0.048, +0.241] (0.450 / 0.301) | 0.000 |
| NICP su template (iscrizione) - BFM+ICT+GNM (10^5), e108 | all_cross | -0.005 [-0.082, +0.068] (0.255 / 0.260) | 0.577 |
| NICP su template (iscrizione) - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | +0.019 [-0.100, +0.131] (0.429 / 0.410) | 0.361 |
| Uni3D-g - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.157 [-0.226, -0.087] (0.144 / 0.301) | 1.000 |
| Uni3D-g - BFM+ICT+GNM (10^5), e108 | all_cross | -0.098 [-0.163, -0.030] (0.162 / 0.260) | 0.998 |
| Uni3D-g - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | +0.004 [-0.120, +0.138] (0.414 / 0.410) | 0.483 |
| OpenShape PointBERT - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.117 [-0.189, -0.046] (0.184 / 0.301) | 0.999 |
| OpenShape PointBERT - BFM+ICT+GNM (10^5), e108 | all_cross | -0.093 [-0.161, -0.024] (0.167 / 0.260) | 0.998 |
| OpenShape PointBERT - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | +0.023 [-0.095, +0.143] (0.433 / 0.410) | 0.386 |
| ShapeDNA k=50, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.194 [-0.256, -0.134] (0.107 / 0.301) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | all_cross | -0.186 [-0.236, -0.131] (0.074 / 0.260) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.110 [-0.237, +0.005] (0.300 / 0.410) | 0.968 |
| ShapeDNA k=100, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.204 [-0.262, -0.141] (0.097 / 0.301) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | all_cross | -0.192 [-0.242, -0.140] (0.068 / 0.260) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.112 [-0.240, +0.007] (0.298 / 0.410) | 0.969 |
| Uni3D-g, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.074 [-0.134, -0.003] (0.227 / 0.301) | 0.980 |
| Uni3D-g, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | all_cross | -0.028 [-0.087, +0.032] (0.232 / 0.260) | 0.832 |
| Uni3D-g, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | +0.190 [+0.087, +0.298] (0.599 / 0.410) | 0.000 |
| OpenShape PointBERT, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.007 [-0.084, +0.067] (0.294 / 0.301) | 0.573 |
| OpenShape PointBERT, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | all_cross | -0.015 [-0.083, +0.058] (0.245 / 0.260) | 0.677 |
| OpenShape PointBERT, Rx(+90) (ablazione) - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | +0.039 [-0.068, +0.144] (0.449 / 0.410) | 0.240 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.013 [-0.095, +0.057] (0.287 / 0.301) | 0.659 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - BFM+ICT+GNM (10^5), e108 | all_cross | -0.009 [-0.080, +0.067] (0.251 / 0.260) | 0.597 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | +0.038 [-0.059, +0.135] (0.448 / 0.410) | 0.255 |
| ShapeDNA k=50 - Chamfer eval | nocrop_cross | -0.102 [-0.156, -0.051] (0.128 / 0.230) | 1.000 |
| ShapeDNA k=50 - Chamfer eval | all_cross | -0.134 [-0.182, -0.086] (0.088 / 0.222) | 1.000 |
| ShapeDNA k=50 - Chamfer eval | subject_pair_mean | -0.115 [-0.239, +0.007] (0.340 / 0.455) | 0.965 |
| ShapeDNA k=100 - Chamfer eval | nocrop_cross | -0.203 [-0.253, -0.159] (0.027 / 0.230) | 1.000 |
| ShapeDNA k=100 - Chamfer eval | all_cross | -0.203 [-0.248, -0.163] (0.019 / 0.222) | 1.000 |
| ShapeDNA k=100 - Chamfer eval | subject_pair_mean | -0.291 [-0.441, -0.139] (0.164 / 0.455) | 1.000 |
| HKS globale - Chamfer eval | nocrop_cross | -0.154 [-0.217, -0.101] (0.075 / 0.230) | 1.000 |
| HKS globale - Chamfer eval | all_cross | -0.180 [-0.230, -0.126] (0.042 / 0.222) | 1.000 |
| HKS globale - Chamfer eval | subject_pair_mean | -0.217 [-0.369, -0.074] (0.238 / 0.455) | 0.999 |
| WKS globale - Chamfer eval | nocrop_cross | -0.201 [-0.251, -0.150] (0.029 / 0.230) | 1.000 |
| WKS globale - Chamfer eval | all_cross | -0.186 [-0.236, -0.138] (0.036 / 0.222) | 1.000 |
| WKS globale - Chamfer eval | subject_pair_mean | -0.243 [-0.401, -0.103] (0.212 / 0.455) | 0.999 |
| NICP su template (iscrizione) - Chamfer eval | nocrop_cross | +0.220 [+0.135, +0.303] (0.450 / 0.230) | 0.000 |
| NICP su template (iscrizione) - Chamfer eval | all_cross | +0.033 [-0.026, +0.091] (0.255 / 0.222) | 0.115 |
| NICP su template (iscrizione) - Chamfer eval | subject_pair_mean | -0.025 [-0.127, +0.076] (0.429 / 0.455) | 0.665 |
| Uni3D-g - Chamfer eval | nocrop_cross | -0.085 [-0.142, -0.025] (0.144 / 0.230) | 0.999 |
| Uni3D-g - Chamfer eval | all_cross | -0.060 [-0.117, +0.000] (0.162 / 0.222) | 0.972 |
| Uni3D-g - Chamfer eval | subject_pair_mean | -0.041 [-0.182, +0.095] (0.414 / 0.455) | 0.699 |
| OpenShape PointBERT - Chamfer eval | nocrop_cross | -0.045 [-0.104, +0.016] (0.184 / 0.230) | 0.934 |
| OpenShape PointBERT - Chamfer eval | all_cross | -0.056 [-0.111, -0.001] (0.167 / 0.222) | 0.977 |
| OpenShape PointBERT - Chamfer eval | subject_pair_mean | -0.022 [-0.152, +0.104] (0.433 / 0.455) | 0.646 |
| ShapeDNA k=50, lambda_i / lambda_1 - Chamfer eval | nocrop_cross | -0.123 [-0.182, -0.070] (0.107 / 0.230) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - Chamfer eval | all_cross | -0.148 [-0.193, -0.098] (0.074 / 0.222) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - Chamfer eval | subject_pair_mean | -0.154 [-0.288, -0.019] (0.300 / 0.455) | 0.985 |
| ShapeDNA k=100, lambda_i / lambda_1 - Chamfer eval | nocrop_cross | -0.133 [-0.185, -0.080] (0.097 / 0.230) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - Chamfer eval | all_cross | -0.154 [-0.203, -0.106] (0.068 / 0.222) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - Chamfer eval | subject_pair_mean | -0.157 [-0.299, -0.033] (0.298 / 0.455) | 0.991 |
| Uni3D-g, Rx(+90) (ablazione) - Chamfer eval | nocrop_cross | -0.002 [-0.057, +0.052] (0.227 / 0.230) | 0.533 |
| Uni3D-g, Rx(+90) (ablazione) - Chamfer eval | all_cross | +0.009 [-0.044, +0.068] (0.232 / 0.222) | 0.383 |
| Uni3D-g, Rx(+90) (ablazione) - Chamfer eval | subject_pair_mean | +0.145 [+0.035, +0.261] (0.599 / 0.455) | 0.007 |
| OpenShape PointBERT, Rx(+90) (ablazione) - Chamfer eval | nocrop_cross | +0.065 [-0.015, +0.149] (0.294 / 0.230) | 0.061 |
| OpenShape PointBERT, Rx(+90) (ablazione) - Chamfer eval | all_cross | +0.023 [-0.051, +0.092] (0.245 / 0.222) | 0.269 |
| OpenShape PointBERT, Rx(+90) (ablazione) - Chamfer eval | subject_pair_mean | -0.005 [-0.143, +0.125] (0.449 / 0.455) | 0.520 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - Chamfer eval | nocrop_cross | +0.058 [-0.025, +0.141] (0.287 / 0.230) | 0.109 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - Chamfer eval | all_cross | +0.029 [-0.052, +0.102] (0.251 / 0.222) | 0.230 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - Chamfer eval | subject_pair_mean | -0.007 [-0.132, +0.114] (0.448 / 0.455) | 0.534 |

Riferimento con la GT unificata: BFM+ICT+GNM (10^5), e108 - Chamfer eval

| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |
| --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 - Chamfer eval | nocrop_cross | +0.071 [+0.015, +0.123] (0.301 / 0.230) | 0.010 |
| BFM+ICT+GNM (10^5), e108 - Chamfer eval | all_cross | +0.038 [-0.010, +0.086] (0.260 / 0.222) | 0.063 |
| BFM+ICT+GNM (10^5), e108 - Chamfer eval | subject_pair_mean | -0.045 [-0.123, +0.030] (0.410 / 0.455) | 0.867 |

#### Varianti contro la loro riga di base, GT unificata

| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |
| --- | --- | --- | --- |
| ShapeDNA k=50, lambda_i / lambda_1 - ShapeDNA k=50 | nocrop_cross | -0.021 [-0.055, +0.013] (0.107 / 0.128) | 0.874 |
| ShapeDNA k=50, lambda_i / lambda_1 - ShapeDNA k=50 | all_cross | -0.014 [-0.042, +0.015] (0.074 / 0.088) | 0.831 |
| ShapeDNA k=50, lambda_i / lambda_1 - ShapeDNA k=50 | subject_pair_mean | -0.040 [-0.153, +0.068] (0.300 / 0.340) | 0.767 |
| ShapeDNA k=100, lambda_i / lambda_1 - ShapeDNA k=100 | nocrop_cross | +0.070 [+0.032, +0.108] (0.097 / 0.027) | 0.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - ShapeDNA k=100 | all_cross | +0.049 [+0.021, +0.078] (0.068 / 0.019) | 0.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - ShapeDNA k=100 | subject_pair_mean | +0.134 [-0.011, +0.273] (0.298 / 0.164) | 0.039 |
| Uni3D-g, Rx(+90) (ablazione) - Uni3D-g | nocrop_cross | +0.083 [+0.053, +0.116] (0.227 / 0.144) | 0.000 |
| Uni3D-g, Rx(+90) (ablazione) - Uni3D-g | all_cross | +0.070 [+0.040, +0.101] (0.232 / 0.162) | 0.000 |
| Uni3D-g, Rx(+90) (ablazione) - Uni3D-g | subject_pair_mean | +0.185 [+0.106, +0.281] (0.599 / 0.414) | 0.000 |
| OpenShape PointBERT, Rx(+90) (ablazione) - OpenShape PointBERT | nocrop_cross | +0.110 [+0.050, +0.165] (0.294 / 0.184) | 0.001 |
| OpenShape PointBERT, Rx(+90) (ablazione) - OpenShape PointBERT | all_cross | +0.078 [+0.024, +0.129] (0.245 / 0.167) | 0.001 |
| OpenShape PointBERT, Rx(+90) (ablazione) - OpenShape PointBERT | subject_pair_mean | +0.017 [-0.085, +0.128] (0.449 / 0.433) | 0.387 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT | nocrop_cross | +0.103 [+0.038, +0.164] (0.287 / 0.184) | 0.000 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT | all_cross | +0.084 [+0.031, +0.138] (0.251 / 0.167) | 0.001 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT | subject_pair_mean | +0.016 [-0.094, +0.113] (0.448 / 0.433) | 0.397 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT, Rx(+90) (ablazione) | nocrop_cross | -0.007 [-0.025, +0.011] (0.287 / 0.294) | 0.749 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT, Rx(+90) (ablazione) | all_cross | +0.006 [-0.012, +0.023] (0.251 / 0.245) | 0.239 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT, Rx(+90) (ablazione) | subject_pair_mean | -0.001 [-0.030, +0.027] (0.448 / 0.449) | 0.558 |

### 3. Frame esatto del training: OpenShape con lo scambio y/z (ablazione)

OpenShape (`src/data.py`, `y_up`) scambia y e z, (x, y, z) -> (x, z, y): una riflessione (det -1), non la rotazione Rx(+90) delle righe dell'8 ottobre; poi `normalize_pc` e, in training, una rotazione casuale attorno a z. Uni3D (`Ensembled_embedding` del pre-training) non scambia gli assi e ruota attorno a y: la sua trasformazione esatta e' l'identita', cioe' la riga primaria (nessuna riga nuova). La riga primaria di OpenShape resta quella del frame nativo.

#### Spearman, GT maxabs

| metodo | nocrop_cross (PRIMARIO) | all_cross | subject_pair_mean |
| --- | --- | --- | --- |
| OpenShape PointBERT | 0.138 [0.10, 0.18] | 0.120 [0.08, 0.16] | 0.304 [0.19, 0.41] |
| OpenShape PointBERT, Rx(+90) (ablazione) | 0.291 [0.21, 0.37] | 0.239 [0.17, 0.31] | 0.435 [0.32, 0.55] |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | 0.292 [0.21, 0.37] | 0.246 [0.17, 0.32] | 0.435 [0.33, 0.54] |

| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |
| --- | --- | --- | --- |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT | nocrop_cross | +0.154 [+0.081, +0.228] (0.292 / 0.138) | 0.000 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT | all_cross | +0.126 [+0.053, +0.199] (0.246 / 0.120) | 0.000 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT | subject_pair_mean | +0.131 [+0.004, +0.244] (0.435 / 0.304) | 0.021 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT, Rx(+90) (ablazione) | nocrop_cross | +0.001 [-0.015, +0.017] (0.292 / 0.291) | 0.420 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT, Rx(+90) (ablazione) | all_cross | +0.007 [-0.008, +0.022] (0.246 / 0.239) | 0.161 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - OpenShape PointBERT, Rx(+90) (ablazione) | subject_pair_mean | +0.000 [-0.026, +0.024] (0.435 / 0.435) | 0.511 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.338 [-0.409, -0.271] (0.292 / 0.630) | 1.000 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - BFM+ICT+GNM (10^5), e108 | all_cross | -0.296 [-0.365, -0.224] (0.246 / 0.541) | 1.000 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.360 [-0.466, -0.259] (0.435 / 0.795) | 1.000 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - Chamfer eval | nocrop_cross | -0.081 [-0.166, -0.002] (0.292 / 0.372) | 0.977 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - Chamfer eval | all_cross | -0.090 [-0.172, -0.010] (0.246 / 0.336) | 0.987 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) - Chamfer eval | subject_pair_mean | -0.308 [-0.430, -0.179] (0.435 / 0.743) | 1.000 |

#### Riconoscimento

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| OpenShape PointBERT | 0.617 [0.596, 0.639] | 0.680 [0.663, 0.700] | 0.728 [0.722, 0.735] | 0 |
| OpenShape PointBERT, Rx(+90) (ablazione) | 0.602 [0.574, 0.630] | 0.680 [0.655, 0.705] | 0.807 [0.789, 0.826] | 0 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | 0.612 [0.580, 0.641] | 0.692 [0.664, 0.717] | 0.811 [0.792, 0.830] | 0 |

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | Chamfer (faceBench, 4096 pt) | +0.134 [+0.087, +0.179] (0.000) | +0.127 [+0.087, +0.165] (0.000) | +0.029 [+0.011, +0.047] (0.000) |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | ArcFace, ombreggiato, 3 viste (riferimento) | -0.372 [-0.403, -0.342] (1.000) | -0.299 [-0.327, -0.274] (1.000) | -0.178 [-0.197, -0.159] (1.000) |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | BFM+ICT+GNM (10^5), e108 | -0.171 [-0.211, -0.132] (1.000) | -0.153 [-0.184, -0.122] (1.000) | -0.135 [-0.151, -0.118] (1.000) |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | OpenShape PointBERT | -0.005 [-0.041, +0.028] (0.641) | +0.011 [-0.021, +0.040] (0.228) | +0.083 [+0.066, +0.100] (0.000) |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | OpenShape PointBERT, Rx(+90) (ablazione) | +0.010 [-0.017, +0.033] (0.243) | +0.011 [-0.009, +0.029] (0.147) | +0.004 [-0.002, +0.011] (0.114) |

Crop:

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| OpenShape PointBERT | 0.190 [0.153, 0.229] | 0.339 [0.303, 0.378] | 0.758 [0.743, 0.774] | 0 |
| OpenShape PointBERT, Rx(+90) (ablazione) | 0.161 [0.117, 0.205] | 0.257 [0.210, 0.306] | 0.708 [0.684, 0.734] | 0 |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | 0.200 [0.155, 0.250] | 0.305 [0.256, 0.355] | 0.731 [0.707, 0.758] | 0 |

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | Chamfer (faceBench, 4096 pt) | -0.272 [-0.321, -0.219] (1.000) | -0.232 [-0.282, -0.173] (1.000) | +0.032 [+0.007, +0.059] (0.007) |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | ArcFace, ombreggiato, 3 viste (riferimento) | -0.790 [-0.835, -0.741] (1.000) | -0.690 [-0.738, -0.640] (1.000) | -0.263 [-0.287, -0.238] (1.000) |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | BFM+ICT+GNM (10^5), e108 | -0.148 [-0.213, -0.081] (1.000) | -0.170 [-0.234, -0.103] (1.000) | -0.113 [-0.143, -0.081] (1.000) |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | OpenShape PointBERT | +0.010 [-0.052, +0.073] (0.355) | -0.035 [-0.094, +0.028] (0.868) | -0.027 [-0.054, +0.001] (0.970) |
| OpenShape PointBERT, scambio y/z del training (riflessione, ablazione) | OpenShape PointBERT, Rx(+90) (ablazione) | +0.039 [+0.008, +0.070] (0.005) | +0.048 [+0.023, +0.076] (0.000) | +0.023 [+0.012, +0.034] (0.000) |

### 4. ShapeDNA con autovalori / lambda_1

(lambda_1, ..., lambda_k) / lambda_1 dagli stessi autovalori (normalizzazione di scala di Reuter et al. 2006), distanza L2. Le righe grezze sono gia' riscalate per area (mesh ad area 1).

#### Spearman, GT maxabs

| metodo | nocrop_cross (PRIMARIO) | all_cross | subject_pair_mean |
| --- | --- | --- | --- |
| ShapeDNA k=50 | 0.108 [0.08, 0.14] | 0.074 [0.05, 0.10] | 0.298 [0.20, 0.40] |
| ShapeDNA k=50, lambda_i / lambda_1 | 0.058 [0.02, 0.09] | 0.042 [0.02, 0.07] | 0.174 [0.08, 0.27] |
| ShapeDNA k=100 | 0.031 [0.02, 0.04] | 0.023 [0.01, 0.03] | 0.249 [0.11, 0.38] |
| ShapeDNA k=100, lambda_i / lambda_1 | 0.053 [0.02, 0.09] | 0.039 [0.02, 0.06] | 0.178 [0.08, 0.28] |

| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |
| --- | --- | --- | --- |
| ShapeDNA k=50, lambda_i / lambda_1 - ShapeDNA k=50 | nocrop_cross | -0.050 [-0.083, -0.016] (0.058 / 0.108) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - ShapeDNA k=50 | all_cross | -0.033 [-0.059, -0.006] (0.042 / 0.074) | 0.990 |
| ShapeDNA k=50, lambda_i / lambda_1 - ShapeDNA k=50 | subject_pair_mean | -0.124 [-0.243, -0.015] (0.174 / 0.298) | 0.992 |
| ShapeDNA k=100, lambda_i / lambda_1 - ShapeDNA k=100 | nocrop_cross | +0.021 [-0.007, +0.053] (0.053 / 0.031) | 0.082 |
| ShapeDNA k=100, lambda_i / lambda_1 - ShapeDNA k=100 | all_cross | +0.016 [-0.007, +0.044] (0.039 / 0.023) | 0.102 |
| ShapeDNA k=100, lambda_i / lambda_1 - ShapeDNA k=100 | subject_pair_mean | -0.071 [-0.213, +0.083] (0.178 / 0.249) | 0.795 |
| ShapeDNA k=50, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.572 [-0.643, -0.499] (0.058 / 0.630) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | all_cross | -0.500 [-0.567, -0.427] (0.042 / 0.541) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.621 [-0.728, -0.513] (0.174 / 0.795) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | nocrop_cross | -0.577 [-0.652, -0.507] (0.053 / 0.630) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | all_cross | -0.502 [-0.573, -0.434] (0.039 / 0.541) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - BFM+ICT+GNM (10^5), e108 | subject_pair_mean | -0.617 [-0.720, -0.508] (0.178 / 0.795) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - Chamfer eval | nocrop_cross | -0.314 [-0.374, -0.255] (0.058 / 0.372) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - Chamfer eval | all_cross | -0.294 [-0.348, -0.244] (0.042 / 0.336) | 1.000 |
| ShapeDNA k=50, lambda_i / lambda_1 - Chamfer eval | subject_pair_mean | -0.569 [-0.676, -0.466] (0.174 / 0.743) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - Chamfer eval | nocrop_cross | -0.319 [-0.374, -0.261] (0.053 / 0.372) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - Chamfer eval | all_cross | -0.297 [-0.350, -0.246] (0.039 / 0.336) | 1.000 |
| ShapeDNA k=100, lambda_i / lambda_1 - Chamfer eval | subject_pair_mean | -0.565 [-0.675, -0.462] (0.178 / 0.743) | 1.000 |

#### Riconoscimento

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ShapeDNA k=50 | 0.575 [0.564, 0.585] | 0.606 [0.598, 0.614] | 0.682 [0.681, 0.683] | 0 |
| ShapeDNA k=50, lambda_i / lambda_1 | 0.455 [0.433, 0.475] | 0.512 [0.494, 0.530] | 0.676 [0.673, 0.679] | 0 |
| ShapeDNA k=100 | 0.290 [0.271, 0.309] | 0.357 [0.340, 0.374] | 0.577 [0.573, 0.581] | 0 |
| ShapeDNA k=100, lambda_i / lambda_1 | 0.137 [0.124, 0.154] | 0.211 [0.194, 0.230] | 0.627 [0.617, 0.636] | 0 |

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ShapeDNA k=50, lambda_i / lambda_1 | Chamfer (faceBench, 4096 pt) | -0.023 [-0.061, +0.011] (0.888) | -0.052 [-0.085, -0.022] (1.000) | -0.106 [-0.121, -0.091] (1.000) |
| ShapeDNA k=50, lambda_i / lambda_1 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.529 [-0.551, -0.504] (1.000) | -0.479 [-0.498, -0.459] (1.000) | -0.313 [-0.318, -0.309] (1.000) |
| ShapeDNA k=50, lambda_i / lambda_1 | BFM+ICT+GNM (10^5), e108 | -0.328 [-0.360, -0.296] (1.000) | -0.332 [-0.358, -0.307] (1.000) | -0.270 [-0.281, -0.258] (1.000) |
| ShapeDNA k=100, lambda_i / lambda_1 | Chamfer (faceBench, 4096 pt) | -0.340 [-0.376, -0.306] (1.000) | -0.354 [-0.386, -0.322] (1.000) | -0.155 [-0.172, -0.140] (1.000) |
| ShapeDNA k=100, lambda_i / lambda_1 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.847 [-0.863, -0.827] (1.000) | -0.780 [-0.798, -0.760] (1.000) | -0.362 [-0.373, -0.353] (1.000) |
| ShapeDNA k=100, lambda_i / lambda_1 | BFM+ICT+GNM (10^5), e108 | -0.645 [-0.673, -0.614] (1.000) | -0.634 [-0.658, -0.608] (1.000) | -0.319 [-0.331, -0.306] (1.000) |
| ShapeDNA k=50, lambda_i / lambda_1 | ShapeDNA k=50 | -0.120 [-0.137, -0.102] (1.000) | -0.094 [-0.109, -0.079] (1.000) | -0.006 [-0.009, -0.003] (1.000) |
| ShapeDNA k=100, lambda_i / lambda_1 | ShapeDNA k=100 | -0.153 [-0.171, -0.135] (1.000) | -0.146 [-0.162, -0.131] (1.000) | +0.050 [+0.042, +0.057] (0.000) |

Crop:

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ShapeDNA k=50 | 0.014 [0.003, 0.027] | 0.078 [0.063, 0.095] | 0.568 [0.555, 0.582] | 0 |
| ShapeDNA k=50, lambda_i / lambda_1 | 0.010 [0.001, 0.023] | 0.052 [0.039, 0.068] | 0.505 [0.499, 0.511] | 0 |
| ShapeDNA k=100 | 0.013 [0.003, 0.026] | 0.058 [0.046, 0.073] | 0.511 [0.507, 0.515] | 0 |
| ShapeDNA k=100, lambda_i / lambda_1 | 0.010 [0.000, 0.024] | 0.052 [0.039, 0.068] | 0.503 [0.497, 0.509] | 0 |

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ShapeDNA k=50, lambda_i / lambda_1 | Chamfer (faceBench, 4096 pt) | -0.462 [-0.491, -0.434] (1.000) | -0.484 [-0.514, -0.455] (1.000) | -0.194 [-0.211, -0.178] (1.000) |
| ShapeDNA k=50, lambda_i / lambda_1 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.980 [-0.991, -0.966] (1.000) | -0.942 [-0.956, -0.926] (1.000) | -0.489 [-0.495, -0.483] (1.000) |
| ShapeDNA k=50, lambda_i / lambda_1 | BFM+ICT+GNM (10^5), e108 | -0.338 [-0.386, -0.290] (1.000) | -0.423 [-0.464, -0.380] (1.000) | -0.338 [-0.362, -0.315] (1.000) |
| ShapeDNA k=100, lambda_i / lambda_1 | Chamfer (faceBench, 4096 pt) | -0.462 [-0.492, -0.433] (1.000) | -0.484 [-0.515, -0.455] (1.000) | -0.196 [-0.212, -0.180] (1.000) |
| ShapeDNA k=100, lambda_i / lambda_1 | ArcFace, ombreggiato, 3 viste (riferimento) | -0.980 [-0.992, -0.966] (1.000) | -0.942 [-0.957, -0.926] (1.000) | -0.491 [-0.497, -0.486] (1.000) |
| ShapeDNA k=100, lambda_i / lambda_1 | BFM+ICT+GNM (10^5), e108 | -0.338 [-0.386, -0.293] (1.000) | -0.423 [-0.464, -0.382] (1.000) | -0.340 [-0.363, -0.317] (1.000) |
| ShapeDNA k=50, lambda_i / lambda_1 | ShapeDNA k=50 | -0.004 [-0.020, +0.012] (0.720) | -0.025 [-0.045, -0.008] (0.996) | -0.063 [-0.078, -0.048] (1.000) |
| ShapeDNA k=100, lambda_i / lambda_1 | ShapeDNA k=100 | -0.003 [-0.018, +0.011] (0.689) | -0.006 [-0.022, +0.009] (0.782) | -0.008 [-0.015, -0.001] (0.986) |

### Controlli dell'aggiunta

GT unificata contro `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/e8` (eval_methods.py, GT `unified`; righe singole: solo il punto, perche' eval_methods usa per tutte le righe il seme della differenza; differenza appaiata: stesso seme):

| riga | scenario | e8 | ricalcolato | max_abs_diff |
| --- | --- | --- | --- | --- |
| BFM+ICT+GNM (10^5), e108 (solo punto) | nocrop_cross | 0.3010 | 0.3010 | 0.000000 |
| Chamfer eval (solo punto) | nocrop_cross | 0.2297 | 0.2297 | 0.000000 |
| delta BFM+ICT+GNM (10^5), e108 - Chamfer eval (punto, CI, P) | nocrop_cross | +0.0713 [+0.0151, +0.1229] | +0.0713 [+0.0151, +0.1229] | 0.000000 |
| BFM+ICT+GNM (10^5), e108 (solo punto) | subject_pair_mean | 0.4099 | 0.4099 | 0.000000 |
| Chamfer eval (solo punto) | subject_pair_mean | 0.4547 | 0.4547 | 0.000000 |
| delta BFM+ICT+GNM (10^5), e108 - Chamfer eval (punto, CI, P) | subject_pair_mean | -0.0448 [-0.1233, +0.0298] | -0.0448 [-0.1233, +0.0298] | 0.000000 |

- riconoscimento, riproduzione di `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/data_scale_ood/arcface_vs_scale_hifi3d/recognition.csv` (stesse repliche): max |diff| su punto e CI di rank-1, mAP, AUC = 1.11e-16 su 6 righe (arcface_shaded_3v, chamfer, joint@bfm, nicp_p2tri, rigid_icp_chamfer, scale_e108)
- scale_e108: distanze dagli embedding contro latent_distance delle pair_metrics di (a) (148500 righe), max |diff|: 5.98e-07
- OpenShape PointBERT, scambio y/z del training (riflessione, ablazione): embedding di dimensione 1280, 10000 punti per mesh
