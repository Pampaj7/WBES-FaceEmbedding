# Protocollo dichiarato prima delle eval: FaceVerse v2 con espressioni casuali, GT d'identita'

Dichiarato il 2026-10-06 13:27:42 CEST, prima di generare i dati e prima di lanciare qualunque eval di questo test
(nessun numero di FaceVerse con espressioni esiste a quest'ora).

- **GT d'identita':** vertex-mean-L2 dopo normalizzazione maxabs fra le forme NEUTRE (solo
  coefficienti d'identita', espressione zero), topologia `original`: la stessa matrice dello
  zero-shot FaceVerse esistente (`datasets/FACEVERSE_ZS/eval_view/gt_matrix.npz`).
- **Input:** ogni mesh (soggetto, topologia) ha un'espressione casuale propria, combinata con le
  topologie esistenti. Stessi 100 soggetti e stesso protocollo dello zero-shot FaceVerse.
- **PRIMARIO:** mesh-pair cross-topologia SENZA crop (20 coppie ordinate di topologie), GT
  d'identita' neutra; delta modello - Chamfer eval con IC 95% appaiato (stesse repliche
  bootstrap per soggetto).
- **SECONDARIO:** subject-pair-mean (media per coppia di soggetti), stessa GT, delta appaiato.
- **Crop:** riportato a parte (coppie con crop), mai nel primario.
- **Baseline:** Chamfer eval (dagli stessi script, stesse righe: e' il confronto appaiato del
  primario), Chamfer faceBench, ICP rigido, NICP (pipeline faceBench, stessi soggetti).
- **Frame dei modelli (regola fissata ora, nessuna scelta a posteriori):** si valutano tutti e tre
  i bracci in entrambe le convenzioni -- BFM completa (rotazione nativa di FaceVerse + facce
  invertite, suffisso `_flip`) e ICT (Rx(180), suffisso `_frame-xmymz`). La riga di
  riferimento di ogni braccio e' il frame dei suoi dati di training: BFM-only -> convenzione
  BFM, ICT-only -> convenzione ICT, BFM+ICT -> entrambe riportate. Le altre combinazioni sono
  riportate come secondarie, non sostituiscono la riga di riferimento.

---

# Revisione 1 del protocollo, dichiarata il 2026-10-06 13:34:19 CEST (prima di qualunque eval su FaceVerse con espressioni)

Ordine dei fatti, per trasparenza: alle 13:2x ho visto i numeri ESISTENTI della riga espressioni
ICT di WS2 (ricalcolo appaiato, `aau/runs/ws2_rexpr_paired/`, dati del 11-13 settembre) e lo
spostamento medio delle espressioni FaceVerse su 6 soggetti di prova (proprieta' dei dati, non un
risultato). Nessun numero di eval su FaceVerse con espressioni esiste a quest'ora.

Sostituisce la gerarchia della dichiarazione precedente; tutto il resto resta valido.

**PRIMARIO: riconoscimento d'identita'** (solo etichette d'identita', nessuna GT di distanza),
sulle 5 topologie senza crop, 100 soggetti, ogni mesh con un'espressione propria.
- *Retrieval:* query = mesh (soggetto A, topologia t1); galleria = una mesh per soggetto, tutte
  nella topologia t2 != t1 (quindi espressione diversa e topologia diversa dalla query; una sola
  mesh corretta in galleria). Per ogni coppia ordinata (t1, t2): 20 x 100 = 2000 query. Misure:
  rank-1 e mAP (con un solo elemento rilevante l'AP e' 1/rank, quindi mAP = MRR).
- *Verifica:* AUC sulle coppie di mesh con topologie diverse (sempre espressioni diverse):
  stessa persona (100 x 10 coppie non ordinate di topologie) contro persone diverse (tutte le
  coppie di soggetti diversi, stesse coppie di topologie). Punteggio = - distanza.
- IC 95% bootstrap per soggetto (1000 repliche; retrieval: ricampionamento dei soggetti query a
  galleria fissa; verifica: coppie pesate per il prodotto dei conteggi), delta modello - baseline
  APPAIATO sulle stesse repliche.
- Crop: escluso dal primario; le stesse misure con il crop come query o galleria, a parte.

**SECONDARIO: ranking con GT d'identita' neutra**, mesh-pair cross-topologia senza crop (il
"primario" della dichiarazione precedente), delta appaiato contro Chamfer eval; subject-pair-mean
come terziario.

**Baseline** (su tutte le misure, stessi soggetti, stesse coppie):
- Chamfer eval (solo nel secondario: lo script non calcola le coppie stesso-soggetto);
- pipeline faceBench: Chamfer 4096 punti, ICP rigido + Chamfer, ICP rigido + NICP P2P / P2Tri
  (le coppie stesso-soggetto si aggiungono con la stessa pipeline, seme = 1000000 + indice del
  soggetto; le coppie fra soggetti diversi sono quelle delle matrici i<j esistenti);
- **Chamfer su regione stabile all'espressione**, regola geometrica dichiarata ora e uguale per
  tutti i domini, non tarata sui risultati: mesh normalizzata maxabs, portata nel frame canonico
  del dominio (assi "alto" e "avanti" dalla tabella delle convenzioni, aau/runs/ws_frame:
  BFM e FaceVerse alto -y / avanti -z; ICT e HIFI3D alto +y / avanti +z); punta del naso = vertice
  con la coordinata "avanti" massima; regione = vertici con coordinata "alto" >= quella della punta
  del naso (fronte, sopracciglia, occhi, dorso del naso, zigomi superiori; esclusi bocca,
  mandibola, guance inferiori); triangoli con tutti e tre i vertici nella regione; la regione e'
  ricentrata sul suo baricentro e riscalata maxabs su se stessa; Chamfer simmetrico (media delle
  distanze al quadrato nei due versi) su 4096 punti campionati per area, seme fisso per mesh.
  Come controllo, la stessa implementazione sulla mesh intera ("Chamfer intero, stessa
  implementazione"), cosi' la differenza isola l'effetto della regione;
- fit 3DMM (NICP verso BFM/ICT con espressioni, distanza fra coefficienti d'identita'): valutato
  per fattibilita'; se richiede piu' di un giorno di lavoro e' riportato come non fatto.

**Frame dei modelli:** invariato (BFM-only -> convenzione BFM con facce invertite, ICT-only ->
convenzione ICT, BFM+ICT -> entrambe; tutte le combinazioni calcolate, le altre secondarie).

---

Nota d'ordine (2026-10-06 14:02:15 CEST): per collaudare il summarizer l'ho fatto girare quando c'erano solo le
baseline "Chamfer regione stabile" e "Chamfer intero" (nessun modello valutato, faceBench ancora
in corso), quindi ho visto i loro numeri di riconoscimento prima delle eval dei modelli. Il
protocollo sopra non e' stato toccato dopo. Diagnostica della regione stabile vista nello stesso
momento: frazione di vertici tenuti mediana 0.52, ma 35 mesh su 600 sopra 0.70 (punta del naso
non trovata per la regola, probabilmente mento/labbra piu' avanti del naso con alcune
espressioni); la regola NON e' stata cambiata.

---

# Risultati: FaceVerse v2 con espressioni casuali, GT d'identita' neutra

Soggetti: 100 (`select_subjects`, seed 1234, gli stessi dello zero-shot FaceVerse neutro), 6 topologie, un'espressione casuale per mesh: pool 44 blendshape ARKit (esclusi 8 eyeLook*), [3, 8] attivi, coefficienti U(0.3, 1.0). Spostamento maxabs medio 0.0392 (sd fra soggetti 0.0099) su diametro 2.93; jawOpen a 1.0: 0.1199. Crop riestratti: {'n_redrawn': 22, 'max': 4}. CI 95% bootstrap per soggetto, 1000 repliche.

Sorgenti dei modelli: BFM+ICT, convenzione BFM (nativa + facce invertite): `joint_flip_topology/zs_zeroshot`; BFM+ICT, convenzione ICT (Rx 180): `joint_frame-xmymz_topology/zs_zeroshot`; BFM-only, convenzione BFM (nativa + facce invertite): `bfm_only_flip_topology/zs_zeroshot`; ICT-only, convenzione ICT (Rx 180): `ict_only_frame-xmymz_topology/zs_zeroshot`; BFM-only, convenzione ICT (Rx 180) (secondaria): `bfm_only_frame-xmymz_topology/zs_zeroshot`; ICT-only, convenzione BFM (nativa + facce invertite) (secondaria): `ict_only_flip_topology/zs_zeroshot`.

## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop

Retrieval: 2000 query (20 coppie ordinate di topologie x 100), galleria di 100 mesh in un'altra topologia; mAP = MRR (un solo rilevante). Verifica: 1000 coppie stessa persona, 99.000 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | 0.680 [0.634, 0.723] | 0.731 [0.688, 0.770] | 0.875 [0.847, 0.900] | 0 |
| BFM+ICT, convenzione ICT (Rx 180) | 0.562 [0.516, 0.611] | 0.642 [0.598, 0.688] | 0.859 [0.832, 0.885] | 0 |
| BFM-only, convenzione BFM (nativa + facce invertite) | 0.649 [0.601, 0.694] | 0.708 [0.666, 0.747] | 0.873 [0.845, 0.899] | 0 |
| ICT-only, convenzione ICT (Rx 180) | 0.544 [0.503, 0.586] | 0.617 [0.579, 0.656] | 0.838 [0.813, 0.863] | 0 |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | 0.462 [0.415, 0.512] | 0.545 [0.501, 0.590] | 0.799 [0.770, 0.826] | 0 |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | 0.559 [0.514, 0.605] | 0.629 [0.583, 0.671] | 0.843 [0.815, 0.871] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.740 [0.698, 0.783] | 0.775 [0.737, 0.814] | 0.882 [0.853, 0.910] | 0 |
| Rigid ICP + Chamfer | 0.918 [0.895, 0.939] | 0.935 [0.916, 0.953] | 0.986 [0.979, 0.992] | 0 |
| Rigid ICP + NICP + P2P | 0.954 [0.936, 0.971] | 0.964 [0.949, 0.978] | 0.994 [0.991, 0.997] | 0 |
| Rigid ICP + NICP + P2Tri | 0.959 [0.940, 0.975] | 0.968 [0.953, 0.981] | 0.995 [0.992, 0.998] | 0 |
| Chamfer regione stabile | 0.787 [0.748, 0.830] | 0.818 [0.781, 0.856] | 0.894 [0.868, 0.920] | 0 |
| Chamfer intero (stessa implementazione) | 0.751 [0.708, 0.793] | 0.782 [0.743, 0.820] | 0.884 [0.856, 0.911] | 0 |

### Delta appaiati modello - baseline (stesse repliche)

| modello | baseline | rank-1: delta [CI 95%] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Chamfer (faceBench, 4096 pt) | -0.060 [-0.093, -0.031] (1.000) | -0.044 [-0.072, -0.018] (1.000) | -0.008 [-0.026, +0.011] (0.797) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Rigid ICP + Chamfer | -0.238 [-0.272, -0.203] (1.000) | -0.203 [-0.236, -0.172] (1.000) | -0.111 [-0.136, -0.089] (1.000) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Rigid ICP + NICP + P2P | -0.274 [-0.314, -0.238] (1.000) | -0.233 [-0.269, -0.199] (1.000) | -0.119 [-0.146, -0.095] (1.000) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Rigid ICP + NICP + P2Tri | -0.278 [-0.319, -0.241] (1.000) | -0.237 [-0.273, -0.202] (1.000) | -0.120 [-0.146, -0.095] (1.000) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Chamfer regione stabile | -0.107 [-0.164, -0.056] (1.000) | -0.086 [-0.136, -0.039] (0.999) | -0.019 [-0.053, +0.011] (0.893) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Chamfer intero (stessa implementazione) | -0.071 [-0.103, -0.044] (1.000) | -0.050 [-0.077, -0.027] (1.000) | -0.009 [-0.027, +0.009] (0.848) |
| BFM+ICT, convenzione ICT (Rx 180) | Chamfer (faceBench, 4096 pt) | -0.178 [-0.218, -0.136] (1.000) | -0.133 [-0.167, -0.098] (1.000) | -0.023 [-0.042, -0.005] (0.995) |
| BFM+ICT, convenzione ICT (Rx 180) | Rigid ICP + Chamfer | -0.356 [-0.397, -0.316] (1.000) | -0.292 [-0.329, -0.256] (1.000) | -0.127 [-0.151, -0.105] (1.000) |
| BFM+ICT, convenzione ICT (Rx 180) | Rigid ICP + NICP + P2P | -0.392 [-0.436, -0.347] (1.000) | -0.322 [-0.360, -0.282] (1.000) | -0.135 [-0.161, -0.111] (1.000) |
| BFM+ICT, convenzione ICT (Rx 180) | Rigid ICP + NICP + P2Tri | -0.396 [-0.440, -0.351] (1.000) | -0.326 [-0.365, -0.285] (1.000) | -0.136 [-0.162, -0.111] (1.000) |
| BFM+ICT, convenzione ICT (Rx 180) | Chamfer regione stabile | -0.225 [-0.283, -0.172] (1.000) | -0.175 [-0.226, -0.127] (1.000) | -0.035 [-0.069, -0.005] (0.982) |
| BFM+ICT, convenzione ICT (Rx 180) | Chamfer intero (stessa implementazione) | -0.189 [-0.228, -0.148] (1.000) | -0.139 [-0.170, -0.106] (1.000) | -0.025 [-0.043, -0.007] (0.996) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Chamfer (faceBench, 4096 pt) | -0.091 [-0.130, -0.051] (1.000) | -0.067 [-0.099, -0.034] (1.000) | -0.009 [-0.031, +0.012] (0.810) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Rigid ICP + Chamfer | -0.269 [-0.310, -0.231] (1.000) | -0.226 [-0.262, -0.193] (1.000) | -0.113 [-0.138, -0.091] (1.000) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Rigid ICP + NICP + P2P | -0.305 [-0.348, -0.265] (1.000) | -0.256 [-0.294, -0.221] (1.000) | -0.121 [-0.148, -0.097] (1.000) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Rigid ICP + NICP + P2Tri | -0.309 [-0.353, -0.269] (1.000) | -0.260 [-0.299, -0.223] (1.000) | -0.121 [-0.149, -0.097] (1.000) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Chamfer regione stabile | -0.138 [-0.197, -0.086] (1.000) | -0.109 [-0.161, -0.063] (1.000) | -0.021 [-0.056, +0.010] (0.906) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Chamfer intero (stessa implementazione) | -0.102 [-0.140, -0.065] (1.000) | -0.073 [-0.104, -0.042] (1.000) | -0.010 [-0.032, +0.010] (0.850) |
| ICT-only, convenzione ICT (Rx 180) | Chamfer (faceBench, 4096 pt) | -0.196 [-0.227, -0.163] (1.000) | -0.158 [-0.186, -0.130] (1.000) | -0.044 [-0.064, -0.024] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Rigid ICP + Chamfer | -0.374 [-0.407, -0.338] (1.000) | -0.317 [-0.348, -0.285] (1.000) | -0.148 [-0.172, -0.126] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Rigid ICP + NICP + P2P | -0.409 [-0.447, -0.372] (1.000) | -0.347 [-0.380, -0.313] (1.000) | -0.156 [-0.181, -0.132] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Rigid ICP + NICP + P2Tri | -0.414 [-0.451, -0.375] (1.000) | -0.351 [-0.384, -0.316] (1.000) | -0.157 [-0.182, -0.132] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Chamfer regione stabile | -0.243 [-0.292, -0.192] (1.000) | -0.200 [-0.247, -0.155] (1.000) | -0.056 [-0.088, -0.023] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Chamfer intero (stessa implementazione) | -0.207 [-0.236, -0.173] (1.000) | -0.164 [-0.191, -0.137] (1.000) | -0.046 [-0.065, -0.027] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Chamfer (faceBench, 4096 pt) | -0.278 [-0.329, -0.230] (1.000) | -0.230 [-0.277, -0.185] (1.000) | -0.084 [-0.115, -0.051] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Rigid ICP + Chamfer | -0.457 [-0.499, -0.411] (1.000) | -0.390 [-0.430, -0.348] (1.000) | -0.188 [-0.213, -0.162] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Rigid ICP + NICP + P2P | -0.492 [-0.537, -0.447] (1.000) | -0.419 [-0.461, -0.377] (1.000) | -0.196 [-0.223, -0.169] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Rigid ICP + NICP + P2Tri | -0.497 [-0.541, -0.449] (1.000) | -0.423 [-0.465, -0.380] (1.000) | -0.196 [-0.224, -0.169] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Chamfer regione stabile | -0.326 [-0.383, -0.267] (1.000) | -0.273 [-0.324, -0.221] (1.000) | -0.096 [-0.129, -0.063] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Chamfer intero (stessa implementazione) | -0.289 [-0.339, -0.242] (1.000) | -0.237 [-0.282, -0.193] (1.000) | -0.085 [-0.116, -0.052] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Chamfer (faceBench, 4096 pt) | -0.180 [-0.216, -0.148] (1.000) | -0.146 [-0.176, -0.118] (1.000) | -0.039 [-0.058, -0.022] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Rigid ICP + Chamfer | -0.359 [-0.395, -0.321] (1.000) | -0.306 [-0.340, -0.273] (1.000) | -0.143 [-0.167, -0.119] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Rigid ICP + NICP + P2P | -0.394 [-0.434, -0.354] (1.000) | -0.336 [-0.372, -0.299] (1.000) | -0.151 [-0.178, -0.125] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Rigid ICP + NICP + P2Tri | -0.399 [-0.441, -0.358] (1.000) | -0.339 [-0.376, -0.302] (1.000) | -0.152 [-0.178, -0.126] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Chamfer regione stabile | -0.228 [-0.284, -0.173] (1.000) | -0.189 [-0.238, -0.140] (1.000) | -0.051 [-0.083, -0.016] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Chamfer intero (stessa implementazione) | -0.192 [-0.226, -0.160] (1.000) | -0.153 [-0.182, -0.124] (1.000) | -0.041 [-0.059, -0.025] (1.000) |

## SECONDARIO: Spearman con la GT d'identita' neutra, mesh-pair senza crop

| metodo | Spearman [CI 95%] | delta vs Chamfer eval [CI] (P<=0) | delta vs Chamfer regione stabile [CI] (P<=0) |
| --- | --- | --- | --- |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | 0.282 | -0.035 [-0.097, +0.025] (0.849) | +0.101 [+0.015, +0.192] (0.018) |
| BFM+ICT, convenzione ICT (Rx 180) | 0.243 | -0.074 [-0.129, -0.020] (0.996) | +0.062 [-0.016, +0.142] (0.067) |
| BFM-only, convenzione BFM (nativa + facce invertite) | 0.256 | -0.061 [-0.138, +0.014] (0.939) | +0.075 [-0.011, +0.161] (0.039) |
| ICT-only, convenzione ICT (Rx 180) | 0.205 | -0.112 [-0.162, -0.062] (1.000) | +0.024 [-0.064, +0.110] (0.273) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | 0.222 | -0.095 [-0.160, -0.034] (0.998) | +0.041 [-0.026, +0.105] (0.099) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | 0.268 | -0.049 [-0.106, +0.012] (0.945) | +0.087 [+0.000, +0.172] (0.025) |
| Chamfer eval | 0.317 [0.259, 0.370] | - | - |
| Chamfer regione stabile | 0.181 [0.116, 0.249] | - | - |
| Rigid ICP + NICP + P2P | 0.187 [0.107, 0.261] | - | - |
| Chamfer intero (stessa implementazione) | 0.320 [0.260, 0.377] | - | - |
| Rigid ICP + Chamfer | 0.259 [0.173, 0.334] | - | - |
| Chamfer (faceBench, 4096 pt) | 0.319 [0.262, 0.379] | - | - |
| Rigid ICP + NICP + P2Tri | 0.194 [0.117, 0.271] | - | - |

### Terziario: subject-pair-mean (senza crop)

| metodo | Spearman [CI 95%] | delta vs Chamfer eval [CI] (P<=0) | delta vs Chamfer regione stabile [CI] (P<=0) |
| --- | --- | --- | --- |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | 0.424 | -0.002 [-0.113, +0.112] (0.515) | +0.179 [+0.043, +0.299] (0.005) |
| BFM+ICT, convenzione ICT (Rx 180) | 0.398 | -0.028 [-0.122, +0.065] (0.730) | +0.153 [+0.037, +0.273] (0.002) |
| BFM-only, convenzione BFM (nativa + facce invertite) | 0.381 | -0.046 [-0.170, +0.086] (0.766) | +0.135 [+0.000, +0.265] (0.025) |
| ICT-only, convenzione ICT (Rx 180) | 0.318 | -0.108 [-0.202, -0.022] (0.992) | +0.073 [-0.063, +0.206] (0.124) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | 0.323 | -0.104 [-0.224, +0.020] (0.946) | +0.077 [-0.048, +0.196] (0.103) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | 0.411 | -0.015 [-0.114, +0.083] (0.574) | +0.166 [+0.040, +0.296] (0.006) |
| Chamfer intero (stessa implementazione) | 0.418 [0.316, 0.515] | - | - |
| Rigid ICP + NICP + P2Tri | 0.240 [0.145, 0.341] | - | - |
| Rigid ICP + Chamfer | 0.300 [0.194, 0.394] | - | - |
| Chamfer (faceBench, 4096 pt) | 0.462 [0.373, 0.553] | - | - |
| Rigid ICP + NICP + P2P | 0.231 [0.129, 0.323] | - | - |
| Chamfer regione stabile | 0.245 [0.134, 0.354] | - | - |
| Chamfer eval | 0.426 [0.326, 0.522] | - | - |

## A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | 0.243 [0.205, 0.283] | 0.364 [0.325, 0.405] | 0.801 [0.771, 0.829] | 0 |
| BFM+ICT, convenzione ICT (Rx 180) | 0.119 [0.092, 0.150] | 0.219 [0.190, 0.254] | 0.721 [0.696, 0.750] | 0 |
| BFM-only, convenzione BFM (nativa + facce invertite) | 0.226 [0.181, 0.272] | 0.347 [0.301, 0.395] | 0.789 [0.757, 0.821] | 0 |
| ICT-only, convenzione ICT (Rx 180) | 0.130 [0.101, 0.165] | 0.227 [0.193, 0.264] | 0.722 [0.694, 0.750] | 0 |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | 0.128 [0.096, 0.169] | 0.243 [0.208, 0.283] | 0.729 [0.705, 0.754] | 0 |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | 0.084 [0.057, 0.113] | 0.174 [0.145, 0.206] | 0.680 [0.653, 0.708] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.396 [0.336, 0.457] | 0.479 [0.421, 0.537] | 0.809 [0.781, 0.837] | 0 |
| Rigid ICP + Chamfer | 0.513 [0.455, 0.571] | 0.590 [0.540, 0.640] | 0.777 [0.734, 0.814] | 0 |
| Rigid ICP + NICP + P2P | 0.898 [0.864, 0.929] | 0.924 [0.897, 0.947] | 0.981 [0.971, 0.991] | 0 |
| Rigid ICP + NICP + P2Tri | 0.909 [0.876, 0.938] | 0.934 [0.910, 0.956] | 0.983 [0.974, 0.992] | 0 |
| Chamfer regione stabile | 0.497 [0.442, 0.557] | 0.584 [0.531, 0.637] | 0.808 [0.779, 0.838] | 0 |
| Chamfer intero (stessa implementazione) | 0.442 [0.381, 0.500] | 0.521 [0.462, 0.575] | 0.832 [0.804, 0.860] | 0 |

### Delta appaiati, crop

| modello | baseline | rank-1: delta [CI 95%] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Chamfer (faceBench, 4096 pt) | -0.153 [-0.219, -0.085] (1.000) | -0.115 [-0.176, -0.053] (1.000) | -0.008 [-0.038, +0.022] (0.701) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Rigid ICP + Chamfer | -0.270 [-0.340, -0.200] (1.000) | -0.226 [-0.291, -0.162] (1.000) | +0.024 [-0.021, +0.069] (0.139) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Rigid ICP + NICP + P2P | -0.655 [-0.702, -0.605] (1.000) | -0.560 [-0.601, -0.517] (1.000) | -0.180 [-0.205, -0.156] (1.000) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Rigid ICP + NICP + P2Tri | -0.666 [-0.712, -0.617] (1.000) | -0.570 [-0.610, -0.528] (1.000) | -0.182 [-0.208, -0.158] (1.000) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Chamfer regione stabile | -0.254 [-0.324, -0.189] (1.000) | -0.220 [-0.282, -0.156] (1.000) | -0.007 [-0.048, +0.030] (0.672) |
| BFM+ICT, convenzione BFM (nativa + facce invertite) | Chamfer intero (stessa implementazione) | -0.199 [-0.264, -0.132] (1.000) | -0.157 [-0.218, -0.096] (1.000) | -0.031 [-0.060, -0.004] (0.988) |
| BFM+ICT, convenzione ICT (Rx 180) | Chamfer (faceBench, 4096 pt) | -0.277 [-0.332, -0.220] (1.000) | -0.259 [-0.309, -0.208] (1.000) | -0.087 [-0.112, -0.061] (1.000) |
| BFM+ICT, convenzione ICT (Rx 180) | Rigid ICP + Chamfer | -0.394 [-0.455, -0.333] (1.000) | -0.371 [-0.426, -0.314] (1.000) | -0.055 [-0.094, -0.015] (0.995) |
| BFM+ICT, convenzione ICT (Rx 180) | Rigid ICP + NICP + P2P | -0.779 [-0.819, -0.741] (1.000) | -0.705 [-0.742, -0.671] (1.000) | -0.260 [-0.284, -0.235] (1.000) |
| BFM+ICT, convenzione ICT (Rx 180) | Rigid ICP + NICP + P2Tri | -0.790 [-0.827, -0.753] (1.000) | -0.715 [-0.749, -0.681] (1.000) | -0.262 [-0.286, -0.236] (1.000) |
| BFM+ICT, convenzione ICT (Rx 180) | Chamfer regione stabile | -0.378 [-0.442, -0.313] (1.000) | -0.364 [-0.426, -0.300] (1.000) | -0.087 [-0.125, -0.048] (1.000) |
| BFM+ICT, convenzione ICT (Rx 180) | Chamfer intero (stessa implementazione) | -0.323 [-0.377, -0.269] (1.000) | -0.302 [-0.351, -0.253] (1.000) | -0.111 [-0.133, -0.087] (1.000) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Chamfer (faceBench, 4096 pt) | -0.170 [-0.241, -0.093] (1.000) | -0.131 [-0.199, -0.061] (1.000) | -0.020 [-0.056, +0.013] (0.871) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Rigid ICP + Chamfer | -0.287 [-0.353, -0.226] (1.000) | -0.243 [-0.302, -0.187] (1.000) | +0.012 [-0.030, +0.053] (0.287) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Rigid ICP + NICP + P2P | -0.672 [-0.723, -0.617] (1.000) | -0.577 [-0.623, -0.532] (1.000) | -0.192 [-0.220, -0.165] (1.000) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Rigid ICP + NICP + P2Tri | -0.683 [-0.731, -0.629] (1.000) | -0.586 [-0.631, -0.542] (1.000) | -0.194 [-0.223, -0.167] (1.000) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Chamfer regione stabile | -0.271 [-0.354, -0.190] (1.000) | -0.236 [-0.311, -0.160] (1.000) | -0.019 [-0.065, +0.022] (0.831) |
| BFM-only, convenzione BFM (nativa + facce invertite) | Chamfer intero (stessa implementazione) | -0.216 [-0.284, -0.140] (1.000) | -0.174 [-0.240, -0.106] (1.000) | -0.044 [-0.076, -0.013] (0.997) |
| ICT-only, convenzione ICT (Rx 180) | Chamfer (faceBench, 4096 pt) | -0.266 [-0.318, -0.213] (1.000) | -0.252 [-0.300, -0.206] (1.000) | -0.087 [-0.115, -0.061] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Rigid ICP + Chamfer | -0.383 [-0.452, -0.310] (1.000) | -0.364 [-0.427, -0.295] (1.000) | -0.055 [-0.098, -0.015] (0.995) |
| ICT-only, convenzione ICT (Rx 180) | Rigid ICP + NICP + P2P | -0.768 [-0.812, -0.724] (1.000) | -0.698 [-0.738, -0.658] (1.000) | -0.260 [-0.285, -0.234] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Rigid ICP + NICP + P2Tri | -0.779 [-0.821, -0.738] (1.000) | -0.707 [-0.746, -0.667] (1.000) | -0.262 [-0.287, -0.235] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Chamfer regione stabile | -0.367 [-0.430, -0.304] (1.000) | -0.357 [-0.419, -0.294] (1.000) | -0.087 [-0.129, -0.047] (1.000) |
| ICT-only, convenzione ICT (Rx 180) | Chamfer intero (stessa implementazione) | -0.312 [-0.361, -0.261] (1.000) | -0.295 [-0.340, -0.250] (1.000) | -0.111 [-0.137, -0.085] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Chamfer (faceBench, 4096 pt) | -0.268 [-0.338, -0.192] (1.000) | -0.235 [-0.301, -0.168] (1.000) | -0.080 [-0.109, -0.049] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Rigid ICP + Chamfer | -0.385 [-0.444, -0.321] (1.000) | -0.347 [-0.404, -0.291] (1.000) | -0.048 [-0.094, -0.001] (0.977) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Rigid ICP + NICP + P2P | -0.770 [-0.814, -0.721] (1.000) | -0.681 [-0.720, -0.637] (1.000) | -0.252 [-0.277, -0.227] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Rigid ICP + NICP + P2Tri | -0.781 [-0.823, -0.733] (1.000) | -0.691 [-0.727, -0.648] (1.000) | -0.254 [-0.279, -0.230] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Chamfer regione stabile | -0.369 [-0.436, -0.298] (1.000) | -0.340 [-0.401, -0.275] (1.000) | -0.079 [-0.116, -0.043] (1.000) |
| BFM-only, convenzione ICT (Rx 180) (secondaria) | Chamfer intero (stessa implementazione) | -0.314 [-0.384, -0.238] (1.000) | -0.278 [-0.342, -0.210] (1.000) | -0.103 [-0.132, -0.071] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Chamfer (faceBench, 4096 pt) | -0.312 [-0.368, -0.257] (1.000) | -0.305 [-0.353, -0.257] (1.000) | -0.129 [-0.154, -0.104] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Rigid ICP + Chamfer | -0.429 [-0.493, -0.363] (1.000) | -0.416 [-0.479, -0.354] (1.000) | -0.097 [-0.142, -0.049] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Rigid ICP + NICP + P2P | -0.814 [-0.853, -0.773] (1.000) | -0.750 [-0.787, -0.713] (1.000) | -0.301 [-0.326, -0.274] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Rigid ICP + NICP + P2Tri | -0.825 [-0.864, -0.785] (1.000) | -0.760 [-0.796, -0.723] (1.000) | -0.303 [-0.328, -0.276] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Chamfer regione stabile | -0.413 [-0.477, -0.352] (1.000) | -0.410 [-0.471, -0.351] (1.000) | -0.128 [-0.169, -0.090] (1.000) |
| ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) | Chamfer intero (stessa implementazione) | -0.358 [-0.413, -0.304] (1.000) | -0.347 [-0.393, -0.299] (1.000) | -0.152 [-0.179, -0.129] (1.000) |

## Controlli

- latent dagli embedding contro `latent_distance` delle pair_metrics, max |diff|: BFM+ICT, convenzione BFM (nativa + facce invertite) 6.99e-07, BFM+ICT, convenzione ICT (Rx 180) 7.14e-07, BFM-only, convenzione BFM (nativa + facce invertite) 6.11e-07, ICT-only, convenzione ICT (Rx 180) 8.76e-07, BFM-only, convenzione ICT (Rx 180) (secondaria) 6.88e-07, ICT-only, convenzione BFM (nativa + facce invertite) (secondaria) 7.63e-07
- regione stabile: frazione di vertici tenuti min 0.344, mediana 0.518, max 0.916
- NICP (asimmetrico): l'orientazione della coppia e' quella delle matrici i<j di alignment_matrix.py, non query -> galleria.
