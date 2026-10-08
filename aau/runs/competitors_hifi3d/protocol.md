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
