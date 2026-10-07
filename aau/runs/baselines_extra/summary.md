# Protocollo dichiarato prima delle eval: CLIP e DINOv2 su render di sola geometria (punto 3 del rebuttal)

Dichiarato il 2026-10-07 22:05 CEST, prima di calcolare qualunque embedding CLIP ViT-L/14 o DINOv2 ViT-B/14 e
prima di renderizzare BFM held-out con questa pipeline (nessun numero di questi modelli esiste a quest'ora).
Numeri gia' visti e noti in anticipo: le tabelle di `aau/runs/arcface_render_zs/summary.md` (ArcFace su
FaceVerse con espressioni e HIFI3D), la Tabella 2 estesa WS1 (`aau/runs/baselines/ranking/table2_extended_heldout.csv`:
CLIP ViT-B/32 e DINOv2 ViT-S/14 su render ombreggiati interi, held-out, Spearman cross no-crop 0.087 e 0.058),
le tabelle di `aau/runs/ws_hifi3d/summary.md`. Unico controllo fatto prima: il frame di BFM per il renderer
(`--frame-check 3`, job 1060586): `none` da' volto dritto e convesso, come in `aau/runs/indomain_recog`.

**Domanda.** Un encoder visivo generico (CLIP, DINOv2) sugli stessi render di ArcFace riconosce le identita'
e ordina le distanze come ArcFace, NICP e il modello appreso?

**Modelli e licenze (verificate il 7 ottobre).**
- CLIP ViT-L/14, pesi OpenAI (`open_clip` `ViT-L-14`, file `timm/vit_large_patch14_clip_224.openai/open_clip_model.safetensors`,
  sha256 `9ce2e8a8eb...`). Licenza MIT (repo openai/CLIP; la copia timm su HF e' marcata apache-2.0). Embedding = `encode_image` (768-d).
- DINOv2 ViT-B/14 (`dinov2_vitb14`, `dinov2_vitb14_pretrain.pth`, sha256 `0b8b82f85d...`). Licenza Apache-2.0
  (codice e pesi, README di facebookresearch/dinov2). Embedding = token CLS dopo la norm finale (uscita di `forward`, 768-d).
- Pesi scaricati dal frontend (raggiungibile HF e dl.fbaipublicfiles) in `~/.cache`, fuori dal repo. Pacchetti:
  il venv esistente delle baseline (`.venv_baselines`, open_clip 3.3.0, timm), nel container PyTorch 24.10.

**Render (la pipeline di ArcFace, `aau/zs3dmm/zs_arcface_render.py`, non toccata).**
- FaceVerse con espressioni e HIFI3D: i render GIA' fatti in `aau/runs/arcface_render_zs/<dominio>/normals/renders`
  (normal map in spazio camera, camera unica per dominio, yaw 0/-30/+30, 512 px), riusati senza rifarli.
- BFM held-out: i 100 soggetti di WS1 (`common.heldout_subjects`, = held-out di `bfm_only` seme 1234 in
  `aau/runs/ws2_cross3dmm/splits.json`), mesh di `datasets/REMESH/npz_data_topo_500`, 6 topologie. Stesso script
  e stessi argomenti del job ArcFace: prima `shaded` (calibrazione del crop), poi `normals` con il crop copiato.
  Come effetto collaterale lo script calcola anche ArcFace su questi render: entra come riga di confronto.
- **Crop:** la similarita' 2x3 congelata di `arcface_align.json` (calibrata sui render ombreggiati del dominio),
  moltiplicata per 2 -> ritaglio 224x224 dello stesso riquadro del volto che vede ArcFace a 112x112 (bordo nero,
  come nei png di controllo). Nessun resize ulteriore; normalizzazione CLIP (mean/std OpenAI) e ImageNet per DINOv2.
- Embedding per vista L2-normalizzato, media sulle 3 viste, rinormalizzazione, distanza = 1 - coseno
  (`zs_arcface_summarize.arcface_distances`, importata).

**Righe di riferimento: CLIP L/14, normal map, crop, 3 viste; DINOv2 B/14, normal map, crop, 3 viste.**
Ablazione unica (secondaria, non sostituisce le righe di riferimento): render intero 512 -> 224 (bicubico, nessun
crop), come facevano CLIP/DINOv2 in WS1.

**(a) FaceVerse v2 con espressioni e (b) HIFI3D neutro, riconoscimento.** Identico a `aau/runs/arcface_render_zs/protocol.md`:
blocco PRIMARIO 5 topologie senza crop (20 coppie ordinate x 100 query; rank-1, mAP = MRR; AUC di verifica su
1000 coppie stessa persona contro 99.000), blocco crop a parte. Funzioni importate da `zs_expr_summarize.py`,
STESSE repliche bootstrap per soggetto (`stable_seed(1234, "expr_recognition")`, 1000): le righe delle baseline
devono riprodurre `aau/runs/arcface_render_zs/<dominio>/recognition.csv` (controllo riportato).
Delta appaiati di ciascuna riga di riferimento contro: ArcFace normal map 3 viste (stessi render), ArcFace
ombreggiato 3 viste (riferimento di quella tabella), NICP P2Tri, Chamfer faceBench, congiunto BFM+ICT (convenzione BFM).

**(b') HIFI3D, ranking.** Spearman con la GT `maxabs` (protocollo ICT, `datasets/HIFI3D/eval_view/gt_matrix.npz`),
setting `original_to_original` (4950 coppie) e `nocrop_cross_topology` (20 coppie ordinate x 4950); GT `coef`
come colonna secondaria. CI bootstrap per soggetto con `weighted_bootstrap_spearman` del paper; delta appaiati
con `zs_summarize.paired_bootstrap` (stesse repliche per le due colonne) contro Chamfer faceBench, NICP P2Tri,
ArcFace normal map, congiunto (distanza L2 fra gli embedding di `joint_frame-xmymz_flip_ranking`).

**(c) BFM held-out (in dominio), ranking.** Spearman con la D_GT del paper (`common.load_gt_submatrix`),
setting `original_to_original` e `nocrop_cross_topology` di WS1, stesse coppie (i<j, soggetto i in tA, j in tB).
Delta appaiati come in (b') contro: Chamfer faceBench, LPIPS, ArcFace di WS1 (matrici di `aau/runs/baselines`),
ArcFace normal map su questi render, CLIP B/32 e DINOv2 S/14 di WS1, e il modello NeurIPS (`latent_distance` delle
pair table `paper_artifacts/bootstrap_ci/table1_pairlevel_exact`; per `original_to_original` la matrice `latent_v1`).
Nota nota in anticipo: la Spearman di (c) e' in dominio per il modello NeurIPS e fuori dominio per CLIP/DINOv2.

**Lettura fissata ora.** Per ogni riga di riferimento (CLIP, DINOv2) e per ogni dominio:
- riconoscimento: "pari a X" se il CI del delta rank-1 contiene 0, "sopra"/"sotto" se e' tutto da un lato;
  la stessa regola per AUC. X = ArcFace normal map, NICP P2Tri, congiunto;
- ranking: "sopra"/"sotto"/"pari a" Chamfer faceBench e al modello (congiunto su HIFI3D, NeurIPS su BFM) con la
  stessa regola sul delta di Spearman nel setting `nocrop_cross_topology` (primario) e `original_to_original`.
- Messaggio per il rebuttal deciso ora: se CLIP e DINOv2 stanno sotto ArcFace normal map nel primario di (a) e (b),
  la riga ArcFace resta la baseline percettiva di riferimento e CLIP/DINOv2 entrano come "encoder generici, piu' deboli";
  se uno dei due e' pari o sopra ArcFace, entra in tabella principale accanto ad ArcFace.
- Nessuna correzione per confronti multipli: i delta sono riportati tutti, con P(boot <= 0).

**Besnier et al. 2023 e Ma et al. 2021.** Integrati solo se codice e pesi pubblici girano in meno di mezza giornata;
altrimenti si riporta il motivo preciso (sezione in fondo al summary).

**Risorse.** 1 GPU L40S per gli embedding (CLIP L/14 + DINOv2 B/14 su 3 domini x 1800 render x 2 ritagli);
render BFM e summary su CPU. Disco: render BFM ~0.6 GB in `aau/runs/baselines_extra/bfm_heldout` (ignorati da git).

---

# Risultati: FaceVerse v2 con espressioni casuali

Soggetti: 100; embedding da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines_extra/fv_expr` (vfm_embed.py), ArcFace da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/fv_expr`.

## Riconoscimento d'identita'

Baseline da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_faceverse_expr/data_736f96956a/baselines`, congiunto da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_faceverse_expr/data_736f96956a/joint_flip_topology/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse per tutte le righe).

### PRIMARIO: 5 topologie senza crop (2000 query; 1000 coppie stessa persona, 99000 diverse)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.304 [0.274, 0.336] | 0.380 [0.352, 0.411] | 0.651 [0.643, 0.661] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.251 [0.221, 0.281] | 0.332 [0.303, 0.361] | 0.593 [0.584, 0.602] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.304 [0.274, 0.338] | 0.385 [0.355, 0.416] | 0.656 [0.647, 0.666] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.255 [0.226, 0.281] | 0.333 [0.305, 0.360] | 0.596 [0.587, 0.606] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.867 [0.843, 0.889] | 0.911 [0.893, 0.927] | 0.934 [0.921, 0.946] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.750 [0.724, 0.775] | 0.807 [0.785, 0.829] | 0.863 [0.847, 0.879] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.680 [0.634, 0.723] | 0.731 [0.688, 0.770] | 0.875 [0.847, 0.900] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.740 [0.698, 0.783] | 0.775 [0.737, 0.814] | 0.882 [0.853, 0.910] | 0 |
| Rigid ICP + Chamfer | 0.918 [0.895, 0.939] | 0.935 [0.916, 0.953] | 0.986 [0.979, 0.992] | 0 |
| Rigid ICP + NICP + P2P | 0.954 [0.936, 0.971] | 0.964 [0.949, 0.978] | 0.994 [0.991, 0.997] | 0 |
| Rigid ICP + NICP + P2Tri | 0.959 [0.940, 0.975] | 0.968 [0.953, 0.981] | 0.995 [0.992, 0.998] | 0 |

#### Delta appaiati, righe di riferimento - confronti

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.563 [-0.601, -0.523] (1.000) | -0.530 [-0.563, -0.495] (1.000) | -0.283 [-0.295, -0.270] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.446 [-0.484, -0.409] (1.000) | -0.426 [-0.461, -0.391] (1.000) | -0.212 [-0.229, -0.195] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.655 [-0.684, -0.623] (1.000) | -0.588 [-0.614, -0.558] (1.000) | -0.344 [-0.352, -0.335] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.436 [-0.479, -0.393] (1.000) | -0.395 [-0.435, -0.354] (1.000) | -0.231 [-0.258, -0.204] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.376 [-0.421, -0.329] (1.000) | -0.351 [-0.392, -0.308] (1.000) | -0.224 [-0.249, -0.197] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.616 [-0.656, -0.574] (1.000) | -0.578 [-0.613, -0.543] (1.000) | -0.342 [-0.355, -0.328] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.499 [-0.535, -0.463] (1.000) | -0.475 [-0.510, -0.440] (1.000) | -0.271 [-0.288, -0.253] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.708 [-0.737, -0.677] (1.000) | -0.636 [-0.664, -0.606] (1.000) | -0.402 [-0.411, -0.392] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.489 [-0.528, -0.448] (1.000) | -0.443 [-0.476, -0.406] (1.000) | -0.290 [-0.315, -0.265] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.429 [-0.467, -0.390] (1.000) | -0.399 [-0.431, -0.364] (1.000) | -0.282 [-0.306, -0.258] (1.000) |

#### Ablazioni, delta appaiati

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.001 [-0.018, +0.021] (0.481) | +0.004 [-0.010, +0.020] (0.266) | +0.005 [+0.000, +0.009] (0.014) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.004 [-0.011, +0.017] (0.337) | +0.001 [-0.011, +0.014] (0.468) | +0.003 [-0.000, +0.006] (0.026) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.053 [+0.027, +0.079] (0.000) | +0.048 [+0.022, +0.072] (0.000) | +0.059 [+0.050, +0.067] (0.000) |

### A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.398 [0.355, 0.444] | 0.486 [0.451, 0.526] | 0.753 [0.738, 0.769] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.290 [0.245, 0.336] | 0.374 [0.332, 0.418] | 0.674 [0.655, 0.692] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.406 [0.366, 0.452] | 0.493 [0.458, 0.533] | 0.761 [0.745, 0.776] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.295 [0.254, 0.340] | 0.382 [0.342, 0.425] | 0.681 [0.663, 0.699] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.952 [0.937, 0.964] | 0.966 [0.955, 0.976] | 0.963 [0.954, 0.970] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.881 [0.867, 0.896] | 0.909 [0.897, 0.921] | 0.922 [0.910, 0.932] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.243 [0.205, 0.283] | 0.364 [0.325, 0.405] | 0.801 [0.771, 0.829] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.396 [0.336, 0.457] | 0.479 [0.421, 0.537] | 0.809 [0.781, 0.837] | 0 |
| Rigid ICP + Chamfer | 0.513 [0.455, 0.571] | 0.590 [0.540, 0.640] | 0.777 [0.734, 0.814] | 0 |
| Rigid ICP + NICP + P2P | 0.898 [0.864, 0.929] | 0.924 [0.897, 0.947] | 0.981 [0.971, 0.991] | 0 |
| Rigid ICP + NICP + P2Tri | 0.909 [0.876, 0.938] | 0.934 [0.910, 0.956] | 0.983 [0.974, 0.992] | 0 |

#### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.554 [-0.600, -0.506] (1.000) | -0.480 [-0.518, -0.437] (1.000) | -0.210 [-0.227, -0.192] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.483 [-0.527, -0.434] (1.000) | -0.423 [-0.460, -0.381] (1.000) | -0.169 [-0.187, -0.151] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.511 [-0.552, -0.467] (1.000) | -0.447 [-0.484, -0.408] (1.000) | -0.230 [-0.244, -0.216] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.002 [-0.057, +0.067] (0.492) | +0.008 [-0.049, +0.068] (0.399) | -0.056 [-0.082, -0.027] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.155 [+0.101, +0.214] (0.000) | +0.122 [+0.074, +0.177] (0.000) | -0.048 [-0.075, -0.017] (0.999) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.662 [-0.713, -0.613] (1.000) | -0.592 [-0.638, -0.548] (1.000) | -0.289 [-0.310, -0.269] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.591 [-0.639, -0.542] (1.000) | -0.535 [-0.580, -0.491] (1.000) | -0.248 [-0.269, -0.228] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.619 [-0.668, -0.572] (1.000) | -0.559 [-0.598, -0.517] (1.000) | -0.309 [-0.325, -0.293] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.106 [-0.174, -0.034] (0.999) | -0.104 [-0.164, -0.041] (0.999) | -0.135 [-0.165, -0.105] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.047 [-0.009, +0.101] (0.053) | +0.010 [-0.040, +0.061] (0.356) | -0.127 [-0.155, -0.100] (1.000) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.008 [-0.020, +0.039] (0.293) | +0.007 [-0.014, +0.029] (0.245) | +0.008 [+0.002, +0.013] (0.003) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.005 [-0.027, +0.035] (0.379) | +0.007 [-0.019, +0.032] (0.267) | +0.007 [+0.001, +0.013] (0.011) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.108 [+0.062, +0.153] (0.000) | +0.112 [+0.072, +0.155] (0.000) | +0.079 [+0.063, +0.098] (0.000) |

### Controllo

- riproduzione di `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/fv_expr/recognition.csv` (stesse repliche): max |diff| su punto e CI di rank-1, mAP, AUC = 1.11e-16 su 14 righe (arcface_normals_3v, arcface_shaded_3v, chamfer, joint@bfm, nicp_p2p, nicp_p2tri, rigid_icp_chamfer)

---

# Risultati: HIFI3D, neutre

Soggetti: 100; embedding da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines_extra/hifi3d` (vfm_embed.py), ArcFace da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/hifi3d`.

## Riconoscimento d'identita'

Baseline da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/baselines`, congiunto da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_ranking/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse per tutte le righe).

### PRIMARIO: 5 topologie senza crop (2000 query; 1000 coppie stessa persona, 99000 diverse)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.597 [0.582, 0.614] | 0.642 [0.628, 0.658] | 0.708 [0.702, 0.715] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.510 [0.485, 0.534] | 0.571 [0.551, 0.592] | 0.668 [0.661, 0.675] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.573 [0.556, 0.590] | 0.630 [0.615, 0.646] | 0.716 [0.708, 0.726] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.516 [0.496, 0.537] | 0.579 [0.561, 0.598] | 0.677 [0.671, 0.684] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.998 [0.995, 1.000] | 0.999 [0.998, 1.000] | 0.997 [0.996, 0.998] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.984 [0.974, 0.992] | 0.991 [0.985, 0.996] | 0.989 [0.985, 0.993] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.397 [0.363, 0.434] | 0.517 [0.489, 0.549] | 0.755 [0.740, 0.770] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] | 0 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0 |
| Rigid ICP + NICP + P2P | 0.988 [0.979, 0.996] | 0.989 [0.980, 0.997] | 0.999 [0.999, 1.000] | 1965 |
| Rigid ICP + NICP + P2Tri | 0.990 [0.982, 0.998] | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] | 1965 |

#### Delta appaiati, righe di riferimento - confronti

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.401 [-0.417, -0.383] (1.000) | -0.357 [-0.371, -0.341] (1.000) | -0.289 [-0.295, -0.283] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.387 [-0.403, -0.368] (1.000) | -0.349 [-0.364, -0.332] (1.000) | -0.281 [-0.288, -0.274] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.393 [-0.411, -0.373] (1.000) | -0.348 [-0.365, -0.330] (1.000) | -0.292 [-0.298, -0.285] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.119 [+0.084, +0.155] (0.000) | +0.078 [+0.044, +0.110] (0.000) | -0.074 [-0.088, -0.060] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.200 [+0.167, +0.235] (0.000) | +0.125 [+0.095, +0.156] (0.000) | -0.047 [-0.060, -0.034] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.488 [-0.513, -0.464] (1.000) | -0.428 [-0.448, -0.406] (1.000) | -0.329 [-0.336, -0.322] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.474 [-0.500, -0.448] (1.000) | -0.420 [-0.440, -0.398] (1.000) | -0.321 [-0.328, -0.313] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.480 [-0.506, -0.456] (1.000) | -0.419 [-0.441, -0.398] (1.000) | -0.332 [-0.339, -0.325] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.033 [-0.004, +0.068] (0.044) | +0.007 [-0.025, +0.038] (0.342) | -0.114 [-0.128, -0.100] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.113 [+0.079, +0.147] (0.000) | +0.054 [+0.024, +0.085] (0.001) | -0.087 [-0.101, -0.072] (1.000) |

#### Ablazioni, delta appaiati

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.024 [-0.046, -0.004] (0.988) | -0.012 [-0.029, +0.004] (0.937) | +0.008 [+0.004, +0.013] (0.000) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.006 [-0.011, +0.024] (0.242) | +0.008 [-0.006, +0.022] (0.130) | +0.009 [+0.005, +0.013] (0.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.087 [+0.059, +0.114] (0.000) | +0.071 [+0.048, +0.094] (0.000) | +0.040 [+0.033, +0.046] (0.000) |

### A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.773 [0.760, 0.787] | 0.803 [0.794, 0.813] | 0.822 [0.817, 0.827] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.666 [0.645, 0.686] | 0.719 [0.703, 0.736] | 0.772 [0.765, 0.779] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.761 [0.743, 0.778] | 0.799 [0.786, 0.811] | 0.831 [0.825, 0.838] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.676 [0.653, 0.699] | 0.730 [0.714, 0.747] | 0.785 [0.779, 0.793] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.997 [0.994, 1.000] | 0.999 [0.997, 1.000] | 0.999 [0.998, 0.999] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.990 [0.984, 0.996] | 0.994 [0.991, 0.998] | 0.994 [0.992, 0.996] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.085 [0.057, 0.117] | 0.201 [0.171, 0.233] | 0.716 [0.699, 0.734] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.472 [0.445, 0.499] | 0.536 [0.510, 0.564] | 0.699 [0.684, 0.715] | 0 |
| Rigid ICP + Chamfer | 0.284 [0.237, 0.338] | 0.423 [0.381, 0.470] | 0.888 [0.865, 0.911] | 0 |
| Rigid ICP + NICP + P2P | 0.752 [0.714, 0.788] | 0.834 [0.810, 0.858] | 0.995 [0.993, 0.997] | 1965 |
| Rigid ICP + NICP + P2Tri | 0.777 [0.740, 0.810] | 0.854 [0.829, 0.877] | 0.997 [0.995, 0.998] | 1965 |

#### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.224 [-0.237, -0.210] (1.000) | -0.196 [-0.205, -0.185] (1.000) | -0.177 [-0.181, -0.172] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.217 [-0.231, -0.202] (1.000) | -0.192 [-0.202, -0.180] (1.000) | -0.172 [-0.177, -0.167] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.004 [-0.043, +0.036] (0.575) | -0.051 [-0.077, -0.024] (1.000) | -0.175 [-0.180, -0.170] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.301 [+0.276, +0.327] (0.000) | +0.266 [+0.241, +0.292] (0.000) | +0.123 [+0.108, +0.136] (0.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.688 [+0.659, +0.717] (0.000) | +0.601 [+0.572, +0.630] (0.000) | +0.106 [+0.089, +0.123] (0.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.331 [-0.352, -0.310] (1.000) | -0.279 [-0.295, -0.263] (1.000) | -0.227 [-0.234, -0.220] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.324 [-0.346, -0.302] (1.000) | -0.275 [-0.292, -0.258] (1.000) | -0.222 [-0.229, -0.215] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.111 [-0.150, -0.071] (1.000) | -0.134 [-0.163, -0.105] (1.000) | -0.225 [-0.232, -0.218] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.194 [+0.165, +0.223] (0.000) | +0.183 [+0.156, +0.211] (0.000) | +0.073 [+0.058, +0.086] (0.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.581 [+0.544, +0.616] (0.000) | +0.518 [+0.481, +0.552] (0.000) | +0.056 [+0.039, +0.073] (0.000) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.012 [-0.031, +0.007] (0.904) | -0.004 [-0.017, +0.009] (0.706) | +0.009 [+0.005, +0.012] (0.000) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.010 [-0.011, +0.033] (0.209) | +0.011 [-0.004, +0.028] (0.090) | +0.013 [+0.009, +0.018] (0.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.107 [+0.086, +0.130] (0.000) | +0.083 [+0.066, +0.101] (0.000) | +0.050 [+0.044, +0.057] (0.000) |

### Controllo

- riproduzione di `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/hifi3d/recognition.csv` (stesse repliche): max |diff| su punto e CI di rank-1, mAP, AUC = 1.11e-16 su 12 righe (arcface_shaded_3v, chamfer, joint@bfm, nicp_p2p, nicp_p2tri, rigid_icp_chamfer)

## Ranking: Spearman con la GT

Coppie: `nocrop_cross_topology` = 20 coppie ordinate di topologie x 4950 = 99000 righe; `original_to_original` = 4950.

### Spearman con la GT `maxabs`, CI 95% bootstrap per soggetto (1000 repliche)

| metodo | nocrop_cross_topology | original_to_original | distanze NaN (cross) |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.195 [0.143, 0.254] | 0.457 [0.362, 0.544] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.107 [0.071, 0.144] | 0.335 [0.230, 0.435] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.196 [0.136, 0.268] | 0.414 [0.305, 0.508] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.130 [0.090, 0.172] | 0.389 [0.287, 0.494] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.242 [0.156, 0.332] | 0.258 [0.172, 0.355] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.245 [0.172, 0.322] | 0.285 [0.204, 0.372] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.245 [0.197, 0.294] | 0.642 [0.580, 0.703] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.325 [0.278, 0.371] | 0.876 [0.845, 0.902] | 0 |
| Rigid ICP + Chamfer | 0.355 [0.274, 0.433] | 0.401 [0.310, 0.484] | 0 |
| Rigid ICP + NICP + P2P | 0.355 [0.266, 0.438] | 0.409 [0.315, 0.502] | 776 |
| Rigid ICP + NICP + P2Tri | 0.389 [0.305, 0.468] | 0.431 [0.340, 0.522] | 776 |

Delta appaiati A - B dello Spearman (stesse repliche), [CI 95%] (P<=0), righe finite per entrambi:

| A | B | nocrop_cross_topology | original_to_original |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.130 [-0.192, -0.066] (1.000) | -0.419 [-0.513, -0.321] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.194 [-0.285, -0.104] (1.000) | +0.026 [-0.094, +0.142] (0.375) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.046 [-0.115, +0.019] (0.925) | +0.199 [+0.093, +0.292] (0.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.049 [-0.109, +0.006] (0.957) | -0.185 [-0.278, -0.097] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.089 [+0.052, +0.125] (0.000) | +0.122 [+0.034, +0.213] (0.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.219 [-0.265, -0.166] (1.000) | -0.541 [-0.637, -0.437] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.283 [-0.369, -0.199] (1.000) | -0.096 [-0.218, +0.032] (0.935) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.135 [-0.209, -0.064] (1.000) | +0.077 [-0.027, +0.185] (0.073) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.138 [-0.188, -0.090] (1.000) | -0.307 [-0.400, -0.209] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.089 [-0.125, -0.052] (1.000) | -0.122 [-0.213, -0.034] (1.000) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.001 [-0.022, +0.023] (0.469) | -0.043 [-0.093, +0.003] (0.962) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.023 [+0.010, +0.037] (0.000) | +0.054 [+0.010, +0.092] (0.007) |

### Spearman con la GT `coef`, CI 95% bootstrap per soggetto (1000 repliche)

| metodo | nocrop_cross_topology | original_to_original | distanze NaN (cross) |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.054 [0.015, 0.096] | 0.131 [0.029, 0.235] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.028 [-0.006, 0.064] | 0.116 [0.024, 0.205] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.058 [0.008, 0.111] | 0.147 [0.047, 0.252] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.039 [-0.004, 0.084] | 0.152 [0.049, 0.249] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.192 [0.123, 0.254] | 0.200 [0.123, 0.283] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.180 [0.112, 0.241] | 0.212 [0.133, 0.288] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.043 [0.009, 0.075] | 0.105 [0.027, 0.188] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.069 [0.030, 0.108] | 0.162 [0.067, 0.252] | 0 |
| Rigid ICP + Chamfer | 0.155 [0.069, 0.241] | 0.165 [0.069, 0.257] | 0 |
| Rigid ICP + NICP + P2P | 0.128 [0.046, 0.204] | 0.131 [0.045, 0.223] | 776 |
| Rigid ICP + NICP + P2Tri | 0.139 [0.051, 0.213] | 0.137 [0.052, 0.217] | 776 |

Delta appaiati A - B dello Spearman (stesse repliche), [CI 95%] (P<=0), righe finite per entrambi:

| A | B | nocrop_cross_topology | original_to_original |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.015 [-0.061, +0.035] (0.718) | -0.030 [-0.129, +0.076] (0.698) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.085 [-0.156, -0.000] (0.975) | -0.006 [-0.116, +0.104] (0.545) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.138 [-0.202, -0.076] (1.000) | -0.069 [-0.173, +0.032] (0.919) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.011 [-0.034, +0.055] (0.324) | +0.026 [-0.071, +0.128] (0.303) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.025 [-0.010, +0.062] (0.099) | +0.015 [-0.074, +0.100] (0.364) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.040 [-0.085, +0.010] (0.946) | -0.046 [-0.143, +0.071] (0.788) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.111 [-0.183, -0.030] (0.998) | -0.021 [-0.111, +0.076] (0.657) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.163 [-0.229, -0.093] (1.000) | -0.084 [-0.174, +0.006] (0.965) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.014 [-0.057, +0.034] (0.738) | +0.011 [-0.081, +0.119] (0.410) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.025 [-0.062, +0.010] (0.901) | -0.015 [-0.100, +0.074] (0.636) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.005 [-0.015, +0.023] (0.317) | +0.016 [-0.027, +0.057] (0.245) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.011 [-0.008, +0.028] (0.109) | +0.036 [-0.010, +0.079] (0.061) |

---

# Risultati: BFM held-out (in dominio, i 100 soggetti di WS1)

Soggetti: 100; embedding da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines_extra/bfm_heldout` (vfm_embed.py), ArcFace da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines_extra/bfm_heldout`.


## Ranking: Spearman con la GT

Coppie: `nocrop_cross_topology` = 20 coppie ordinate di topologie x 4950 = 99000 righe; `original_to_original` = 4950.

### Spearman con la GT `paper`, CI 95% bootstrap per soggetto (1000 repliche)

| metodo | nocrop_cross_topology | original_to_original | distanze NaN (cross) |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.129 [0.084, 0.177] | 0.332 [0.240, 0.442] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.114 [0.077, 0.156] | 0.370 [0.265, 0.473] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.143 [0.097, 0.187] | 0.344 [0.237, 0.443] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.110 [0.074, 0.153] | 0.397 [0.297, 0.486] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.223 [0.163, 0.283] | 0.280 [0.205, 0.357] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.220 [0.162, 0.277] | 0.301 [0.228, 0.377] | 0 |
| Chamfer faceBench (WS1) | 0.469 [0.400, 0.534] | 0.644 [0.571, 0.712] | 0 |
| LPIPS AlexNet, ombreggiato (WS1) | 0.127 [0.099, 0.157] | 0.590 [0.503, 0.662] | 0 |
| ArcFace, ombreggiato, crop fisso (WS1) | 0.218 [0.160, 0.272] | 0.293 [0.220, 0.364] | 0 |
| CLIP ViT-B/32 laion2b, ombreggiato intero (WS1) | 0.087 [0.057, 0.117] | 0.358 [0.261, 0.456] | 0 |
| DINOv2 ViT-S/14, ombreggiato intero (WS1) | 0.058 [0.030, 0.089] | 0.378 [0.278, 0.478] | 0 |
| Modello NeurIPS (BFM, v1) | 0.789 [0.737, 0.830] | 0.830 [0.779, 0.872] | 0 |

Delta appaiati A - B dello Spearman (stesse repliche), [CI 95%] (P<=0), righe finite per entrambi:

| A | B | nocrop_cross_topology | original_to_original |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer faceBench (WS1) | -0.340 [-0.413, -0.258] (1.000) | -0.312 [-0.429, -0.202] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | LPIPS AlexNet, ombreggiato (WS1) | +0.002 [-0.045, +0.051] (0.486) | -0.258 [-0.368, -0.143] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, crop fisso (WS1) | -0.089 [-0.141, -0.042] (1.000) | +0.039 [-0.054, +0.134] (0.209) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.094 [-0.152, -0.033] (0.999) | +0.052 [-0.056, +0.163] (0.159) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Modello NeurIPS (BFM, v1) | -0.659 [-0.718, -0.588] (1.000) | -0.498 [-0.606, -0.394] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.015 [-0.017, +0.051] (0.207) | -0.038 [-0.129, +0.057] (0.775) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer faceBench (WS1) | -0.355 [-0.422, -0.280] (1.000) | -0.273 [-0.383, -0.174] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | LPIPS AlexNet, ombreggiato (WS1) | -0.013 [-0.053, +0.026] (0.746) | -0.220 [-0.329, -0.124] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, crop fisso (WS1) | -0.104 [-0.157, -0.057] (1.000) | +0.077 [-0.018, +0.164] (0.057) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.109 [-0.161, -0.055] (1.000) | +0.090 [-0.017, +0.197] (0.054) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Modello NeurIPS (BFM, v1) | -0.675 [-0.728, -0.610] (1.000) | -0.460 [-0.567, -0.364] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.015 [-0.051, +0.017] (0.793) | +0.038 [-0.057, +0.129] (0.225) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.014 [-0.002, +0.029] (0.048) | +0.012 [-0.036, +0.060] (0.321) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | -0.003 [-0.016, +0.009] (0.733) | +0.027 [-0.027, +0.078] (0.151) |

---

# Besnier et al. 2023 e Ma et al. 2021 (le baseline geometriche di Lee et al. 2025): non integrate

Verificato il 2026-10-07. Regola del protocollo: integrare solo se codice e pesi pubblici girano in meno di
mezza giornata di lavoro.

**Besnier, Arguillère, Pierson, Daoudi, "Toward mesh-invariant 3D generative deep learning with geometric
measures", Computers & Graphics 115 (2023), arXiv:2306.15762.**
- Codice: nessun link nel paper (testo dell'arXiv controllato: c'e' solo "the Python code is built over the one
  from [3]"). Nessun repository nell'account GitHub dell'autore (`tbesnier`: ScanTalk, ScanMove, STM, PaNDaS,
  bm-shapes, deep_deformer, robustAE). `robustAE` (MIT) contiene solo README e LICENSE, un "Initial commit" del
  19-11-2025; `deep_deformer` non ha README e non e' dichiarato come codice del paper. Ricerca GitHub per titolo:
  0 risultati.
- Pesi: nessuno pubblicato. Il modello e' un autoencoder addestrato su COMA (topologia FLAME): rifarlo vuol dire
  reimplementare il paper e ottenere COMA (licenza MPI, registrazione dell'utente), cioe' ben oltre mezza giornata,
  e il risultato sarebbe la nostra reimplementazione, non la loro baseline.
- Cosa abbiamo gia': la componente di misura del loro metodo e' la distanza kernel fra varifold/currents, che e'
  gia' in tabella come baseline (`varifold`, `currents`, WS1, `aau/runs/baselines/ranking/table2_extended_heldout.csv`).

**Ma, Liang, Liang, Wu, "3D facial similarity measure based on deformation field", IEEE RCAR 2021, pp. 364-369.**
- Codice: nessun repository trovato (ricerca GitHub per titolo e parole chiave: 0 risultati), nessun link noto.
- Paper IEEE dietro paywall; il metodo (campo di deformazione fra mesh in corrispondenza) presuppone una
  registrazione, cioe' la famiglia gia' coperta da ICP rigido + NICP (P2P, P2Tri) nelle nostre tabelle.
- Reimplementarlo dal solo paper richiede piu' di mezza giornata e non darebbe la loro baseline.

**Conclusione:** nessuna delle due entra. Nel rebuttal: "nessuna delle due ha codice o pesi pubblici; le loro
famiglie (misure geometriche varifold/currents, registrazione + deformazione) sono rappresentate da varifold,
currents, ICP e NICP".

---

# Giudizio (scritto il 2026-10-07, dopo i numeri)

Job: render BFM held-out 1060587 (CPU, ombreggiato + normal map + ArcFace), embedding CLIP/DINOv2 1060588
(FaceVerse, HIFI3D) e 1060591 (BFM), 1 L40S ciascuno; summary 1060600 (FaceVerse; fallito poi su HIFI3D per un
bug di formattazione, corretto) e 1060602/1060603 (HIFI3D, BFM). Frame BFM 1060586. Controlli superati: le righe
gia' pubblicate in `aau/runs/arcface_render_zs` si riproducono esattamente (max |diff| 1.1e-16, FaceVerse 14 righe,
HIFI3D 12); le righe WS1 su BFM riproducono `table2_extended_heldout.csv` (Chamfer 0.469 / 0.644, ArcFace 0.218 / 0.293,
CLIP B/32 0.087 / 0.358, DINOv2 S/14 0.058 / 0.378); il congiunto su HIFI3D riproduce `summary_frame-xmymz_flip.md`
(0.245 GT maxabs, 0.043 GT coef). ArcFace sui nuovi render ombreggiati BFM (0.220) coincide con ArcFace WS1 (0.218).

**Lettura fissata nel protocollo, riconoscimento (primario, rank-1, senza crop).**
- FaceVerse con espressioni: CLIP 0.304 [0.274, 0.336], DINOv2 0.251 [0.221, 0.281]. **Sotto** ArcFace normal map
  (-0.563 e -0.616), sotto NICP P2Tri, sotto Chamfer faceBench e **sotto il congiunto** (-0.376 [-0.421, -0.329],
  -0.429). Stessa cosa sull'AUC (0.651 e 0.593).
- HIFI3D: CLIP 0.597 [0.582, 0.614], DINOv2 0.510 [0.485, 0.534]. **Sotto** ArcFace normal map (-0.401, -0.488) e
  NICP. **Sopra il congiunto sul rank-1** (CLIP +0.200 [+0.167, +0.235], DINOv2 +0.113 [+0.079, +0.147]) ma
  **sotto sull'AUC** (-0.047, -0.087). Contro Chamfer: CLIP sopra sul rank-1 (+0.119), DINOv2 pari (+0.033
  [-0.004, +0.068]); entrambi sotto sull'AUC.

**Ranking (Spearman con la GT, primario `nocrop_cross_topology`).**
- HIFI3D (GT maxabs): CLIP 0.195, DINOv2 0.107. Sotto Chamfer faceBench (-0.130, -0.219). Contro il congiunto (0.245):
  CLIP **pari** (-0.049 [-0.109, +0.006]), DINOv2 sotto (-0.138). Contro ArcFace normal map: CLIP pari, DINOv2 sotto.
  In `original_to_original` CLIP sta sopra ArcFace (+0.199) ma sotto congiunto e Chamfer. Con la GT coef tutti i
  metodi stanno fra 0.03 e 0.19 e quasi tutti i delta contengono 0.
- BFM held-out (in dominio, GT del paper): CLIP 0.129, DINOv2 0.114, contro **0.789 del modello NeurIPS** (delta
  -0.659 e -0.675) e 0.469 di Chamfer (-0.340, -0.355); sotto ArcFace normal map (-0.094, -0.109); pari a LPIPS.
  In `original_to_original` (0.332 e 0.370) pari ad ArcFace, sotto LPIPS, Chamfer e NeurIPS.

**Messaggio per il rebuttal (regola del protocollo).** CLIP e DINOv2 stanno sotto ArcFace normal map nel primario di
(a) e (b): **ArcFace resta la baseline percettiva di riferimento**, e CLIP/DINOv2 entrano come "encoder visivi
generici, piu' deboli". Fra i due CLIP L/14 batte DINOv2 B/14 nel riconoscimento (+0.053 su FaceVerse, +0.087 su
HIFI3D) e nel ranking su HIFI3D (+0.089); su BFM sono pari. Rispetto alle versioni piccole di WS1 (B/32 e S/14, render
ombreggiati interi) i modelli grandi su normal map salgono nel cross-topology BFM (0.129 contro 0.087, 0.114 contro
0.058; solo stime puntuali, non appaiate) ma restano lontanissimi dal modello appreso.

**Ablazione (crop calibrato contro render intero).** Effetto piccolo: FaceVerse rank-1 +0.001 (CLIP) e +0.004
(DINOv2), entrambi con CI che contiene 0; HIFI3D: il render intero toglie 0.024 [-0.046, -0.004] a CLIP e
aggiunge 0.006 a DINOv2 (CI con 0); BFM ranking entro 0.03, CI con 0. La scelta del ritaglio non spiega il
livello basso.

**Diagnosi (post hoc, non nel protocollo; job 1060601, FaceVerse).** Negli embedding pesa piu' la topologia
dell'identita': coseno medio CLIP fra soggetti diversi con la stessa topologia 0.946, fra lo stesso soggetto con
topologie diverse 0.934 (DINOv2: 0.883 contro 0.786). Il rank-1 original -> noisy crolla a 0.06 (CLIP) e 0.02
(DINOv2), contro 0.42 / 0.28 su original -> remesh. Gli encoder generici vedono la tassellazione nella normal map
(la "texture" dei triangoli), che ArcFace, addestrato sull'identita', ignora. Il caricamento dei pesi e' stato
verificato a parte: CLIP zero-shot sul render da' p = 1.0 a "a 3D rendering of a human face".

**Besnier 2023 e Ma 2021:** non integrate, motivi in `other_baselines.md` (nessun codice ne' pesi pubblici).

**Cosa resta aperto.** (1) Nessun FaceVerse neutro, quindi il calo su FaceVerse resta non separabile fra
espressioni e dominio, come per ArcFace. (2) Le NaN di NICP su HIFI3D (776 coppie cross nel ranking, 1965 distanze
nel riconoscimento) rendono quelle righe un limite inferiore. (3) Molti confronti senza correzione per la
molteplicita': le conclusioni sopra si reggono su delta con P(boot <= 0) pari a 0.000 o 1.000.
