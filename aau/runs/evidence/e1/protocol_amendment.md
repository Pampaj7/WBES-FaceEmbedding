# E1, emendamento 1 al protocollo (scritto PRIMA di ogni numero delle celle nuove)

8 ottobre 2026, sera. Rimanda a `protocol.md` (sha256 493941994cd41864d2d2b815ca257e53bf6cc46913b075c1aec7155a3deb29f6,
registrato in `protocol.sha256` alle 15:30), che resta invariato. Scritto dopo la revisione del critic
(BLOCCANTE sul disegno) e le decisioni del PI, quando nessuna cella nuova di E1 aveva ancora un checkpoint ne' un
numero: gli unici numeri esistenti sono quelli di C3M su L40S (run su scala 1060130, gia' in
`aau/runs/data_scale_ood/curve.md`) e degli smoke di plumbing (che non producono metriche di test). Dove questo
file e la versione originale divergono, vale questo file. Lo sha256 di questo file e' in `protocol_amendment.sha256`.

## 1. Celle e hardware

- **Tutte le celle si addestrano su V100** (container PyTorch 24.10): C2M, C2F, C3F, C2F-GNM, C3F-UGT, i secondi
  semi C2F-s2 e C3F-s2, C2F40, C3F40, G1 e **C3M rifatta su V100** (stesso split del run su scala, K=46, fermata
  all'epoca 72). Le valutazioni girano tutte sulle A100.
- **La C3M della regola e' quella rifatta su V100.** La C3M su L40S (1060130) resta solo come riferimento.
- **C2F-GNM** = BFM 392 + GNM 5.401 (stratificate per 1/2 espressioni), nessuna ICT: lo stesso totale non-BFM
  di C2F e C3F. Le 844 GNM di C3F sono un suo sottoinsieme.
- **C3F-UGT** = C3F (stesso split, passi, seme, hardware) con la GT di training UNIFICATA
  (`datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm.npz`, verificata da `check_train_gt.py`).
- **Secondi semi:** C2F-s2 e C3F-s2 = C2F e C3F con `--seed` e `block_seed` 2345, stesso split.
- **C2F40 e C3F40:** ICT a 1/40 (1.350 non-BFM, un solo blocco), annidate in C2F e C3F. Servono SOLO alla
  lettura della quantita' (circa 10x contro le celle M); non entrano nella regola sulla varieta'.

## 2. Regola sulla varieta' (sostituisce la regola primaria di protocol.md)

Metrica: Spearman con la GT `maxabs` di HIFI3D, mesh-pair `nocrop_cross` (20 coppie ordinate senza crop),
clean, seme 1234 delle celle. Effetto minimo: **+0.05**.

Delta = Sp(C3F) - max(Sp(C2F), Sp(C2F-GNM)). Il massimo si prende in OGNI replica bootstrap: 1000 repliche per
soggetto, le stesse per tutte le celle. IC 95% = percentili.

Gli esiti si dichiarano a 21.096 passi (lettura principale) e a 10.548 (accanto):

- **SOSTENUTA** se valgono TUTTE:
  - (a) Delta >= +0.05 e IC di Delta con estremo inferiore > 0;
  - (b) dev FaceScape: Delta (stessa definizione, GT `maxabs`, `nocrop_cross`) con punto > 0, cioe' stessa
    direzione;
  - (c) non inferiorita' su FaceVerse con espressioni (convenzione BFM): rank-1(C3F) - max(rank-1(C2F),
    rank-1(C2F-GNM)), con il massimo per replica, ha IC con estremo inferiore > -0.05;
  - (d) Delta maggiore del pavimento del rumore (sotto);
  - (e) secondo seme: Sp(C3F-s2) - Sp(C2F-s2) > 0 (punto).
- **SMENTITA** se:
  - Delta <= 0 con IC che esclude +0.05 (estremo superiore < +0.05);
  - e sul dev FaceScape Delta <= 0 (punto).
- **NON CONCLUDENTE** in ogni altro caso, con l'elenco delle condizioni mancate.

**Pavimento del rumore,** su HIFI3D `nocrop_cross` (GT `maxabs`) allo stesso numero di passi: il massimo fra
|Sp(C3M V100) - Sp(C3M L40S)| (stesso seme, hardware diverso), |Sp(C2F) - Sp(C2F-s2)| e |Sp(C3F) - Sp(C3F-s2)|.
Se e' grande quanto Delta o piu', l'esito e' NON CONCLUDENTE, anche se valgono (a)-(c).

C3M - C2M e C3F - C2F (il confronto originale) restano riportati, solo come descrittivi.

## 3. Quantita' e G1 (descrittive, nessuna regola di decisione)

- **Circa 2.5x:** C2M - C2F, C3M - C3F.
- **Circa 10x:** C2M - C2F40, C3M - C3F40.
- **G1** contro C3M e contro C3F.
- Stessa metrica, stessi IC.

## 4. GT unificata

Tutte le celle si valutano con ENTRAMBE le GT, `maxabs` e unificata, sulle stesse righe e repliche:
- HIFI3D: `datasets/UNIFIED_GT/gt/hifi3d_unified.npz`;
- dev FaceScape e FLAME: con la GT unificata del set di valutazione, se il suo agente la fornisce nel formato di
  `aau/zs3dmm`; altrimenti "-".

**Domanda dichiarata:** addestrando sulla GT unificata, il modello raggiunge le baseline con allineamento su
quella GT?
- Confronto: C3F-UGT - ICP+Chamfer (`rigid_icp_chamfer` di faceBench) con la GT unificata di HIFI3D,
  `nocrop_cross`, sulle righe dove la baseline e' finita.
- **RAGGIUNGE** se l'IC 95% della differenza include lo 0 o e' tutto positivo (estremo superiore >= 0);
  **NO** altrimenti.
- Accanto, descrittivi: C3F-UGT - NICP P2Tri, e C3F-UGT - C3F con entrambe le GT.

## 5. Invariato

Domini e misure di `protocol.md`, IC (1000 repliche per soggetto, seme 1234), "un seme per cella" tranne i
secondi semi qui sopra, limiti dichiarati.
