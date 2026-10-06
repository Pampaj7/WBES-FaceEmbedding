# Protocollo dichiarato prima delle eval: ArcFace su render di sola geometria, zero-shot FaceVerse con espressioni (e HIFI3D)

Dichiarato il 2026-10-06 19:21:40 CEST, prima di generare qualunque render o embedding ArcFace su FaceVerse o HIFI3D
(nessun numero ArcFace su questi domini esiste a quest'ora). Numeri gia' visti e noti in anticipo:
le baseline e i modelli di `aau/runs/ws_faceverse_expr/summary.md` (NICP P2Tri rank-1 0.959,
congiunto BFM 0.680) e l'AUC 1.000 di ArcFace su Multiface (13 soggetti).

**Domanda.** ArcFace su render di sola geometria, con la pipeline di Multiface, riconosce le
identita' FaceVerse con espressioni quanto NICP, o piu' del modello appreso?

**Pipeline (identica a Multiface, `aau/multiface/ws3a_render.py` + `ws3a_perceptual.py`):**
- mesh: `datasets/FACEVERSE_ZS/expr_view/npz`, i 100 soggetti di `select_subjects` (seed 1234), 6 topologie;
- rotazione fissa dal frame del dominio a quello del renderer (alto -y, naso -z), UNA per dominio,
  scelta guardando i render di controllo prima di calcolare gli embedding: FaceVerse ha gia' alto -y
  e naso -z (tabella di aau/runs/ws_frame) -> attesa identita'; HIFI3D alto +y, naso +z -> attesa Rx(180).
  Le facce non si invertono: il renderer ombreggia a due facce, il verso dei triangoli non entra;
- `normalize_maxabs` per mesh, camera unica calcolata su tutte le 600 mesh del dominio, yaw 0, -30, +30,
  512 px, `render_mesh` (grigio ombreggiato);
- crop fisso ricalibrato su QUESTI render: detector di insightface su tutti i render del dominio, mediana
  per yaw dei 5 landmark, `estimate_norm` -> similarita' 2x3 congelata; poi il detector non si usa piu'.
  Conteggio dei fallimenti del detector per topologia riportato;
- ArcFace `w600k_r50.onnx`: embedding per vista, L2, media sulle viste, rinormalizzazione; distanza = 1 - coseno.

**PRIMARIO (identico a `aau/runs/ws_faceverse_expr/protocol.md`, revisione 1):** riconoscimento
d'identita' sulle 5 topologie senza crop. Retrieval: 20 coppie ordinate (t1, t2) x 100 query, galleria
di 100 mesh in t2; rank-1 e mAP (= MRR). Verifica: AUC di -distanza, 1000 coppie stessa persona contro
99.000 persone diverse, 10 coppie non ordinate di topologie. IC 95% bootstrap per soggetto, 1000
repliche, con le STESSE repliche del summary esistente (seme `stable_seed(1234, "expr_recognition")`):
le righe delle baseline devono riprodurre quei numeri, ed e' il controllo. Funzioni di calcolo importate
da `aau/zs3dmm/zs_expr_summarize.py`, non riscritte.
- Riga di riferimento: **ArcFace, 3 viste, ombreggiato**.
- Delta APPAIATI ArcFace - {NICP P2Tri, ICP rigido + Chamfer, Chamfer faceBench, BFM+ICT in convenzione BFM}.
- Crop: le stesse misure, a parte, mai nel primario.

**Lettura fissata ora:** "ArcFace pari a NICP" se il CI del delta ArcFace - NICP P2Tri sul rank-1 contiene
0 o e' positivo; "ArcFace sopra il congiunto" se il CI del delta ArcFace - congiunto (BFM) sul rank-1 e'
tutto sopra 0. Le stesse due letture valgono per l'AUC.

**Ablazioni (secondarie, non sostituiscono la riga di riferimento):** 1 vista (yaw 0) contro 3 viste;
normal map (normali in spazio camera, RGB = (n+1)/2) al posto dell'ombreggiatura, solo se costa meno
di un'ora, con lo STESSO crop dei render ombreggiati (stessa camera, stessa geometria, quindi stesso
riquadro del volto) e anche lei in 1 e 3 viste. Delta appaiati 3 viste - 1 vista e normal map - ombreggiato.

**HIFI3D senza espressioni (se c'e' tempo):** stesse regole su `datasets/HIFI3D/eval_view/npz`, 100
soggetti, neutre. Le baseline faceBench stesso-soggetto (che su HIFI3D mancano) si calcolano con
`zs_bl_same.py` (stessa pipeline, stesso seme). Il congiunto in convenzione BFM completa
(`joint_frame-xmymz_flip`) entra solo se i suoi embedding per mesh sono gia' disponibili o
calcolabili in giornata; altrimenti la riga e' riportata come mancante.

**GPU:** la pipeline ArcFace di Multiface gira su onnxruntime CPU; i job sono CPU-only (`--gres=NONE`).
