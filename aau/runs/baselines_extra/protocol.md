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
