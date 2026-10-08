# E2 (canonicalizzazione rigida al test) ed E3b (rimesh uniforme al test): protocollo

Dichiarato il 2026-10-08 alle 15:43 CEST, PRIMA di qualunque canonicalizzazione o rimesh dei domini di test
(HIFI3D, FaceVerse, NoW) e prima di ogni embedding o metrica su di essi. Il metodo e' stato messo a punto
SOLO sui domini di training (soggetti held-out di ICT, BFM e GNM del run grande); la storia e' in fondo.
Checkpoint: e108 del run 1060130 (`epoch108.pth`, nessun training).

## E2: canonicalizzazione

- **Riferimento.** Faccia media vertice per vertice delle `original` ICT di training (4008 identita'
  id10000-14999 di `split_scale_all.json`, mesh di `datasets/ICT/train_ready/npz_withops`): ICT e' 54.008
  delle 64.400 identita' di training, GNM (10.000) sta nello stesso frame (+y alto, naso +z). Controllo: la
  faccia media GNM (held-out) si allinea a quella ICT con 0.67 gradi.
- **Metodo** (`aau/evidence/e2_canon/canon.py`, parametri `PARAMS`): similarita' (rotazione, scala,
  traslazione) con ICP trimmed simmetrico e consapevole delle regioni: coppie sorgente -> faccia media intera
  (tenuto l'80% piu' vicino) e nucleo centrale della faccia media (raggio 1.1 dalla punta del naso, in unita'
  del raggio RMS) -> sorgente (tenuto il 90%), Umeyama pesato, scala limitata a [0.67, 1.5] attorno a quella dei
  raggi RMS. 8 start (identita', Rx180, Ry180, Rz180, Rx+-90, Ry+-90), grossolano (1000 punti, 30 iterazioni)
  poi fine sui 2 migliori (4000 punti, 60 iterazioni); start scelto col residuo trimmed simmetrico minimo.
  Si applica alla mesh SOLO la similarita' trovata; facce invariate (det R = +1). Nessuna corrispondenza,
  nessuna deformazione. FaceVerse: niente inversione delle facce (dopo la rotazione e' nella convenzione ICT
  completa, quella del riferimento); i bracci di riferimento senza canonicalizzazione restano quelli pubblicati.
- **Fallimento.** Soglia sul residuo dichiarata ora, dalla calibrazione: T = 2 x p99 del residuo delle
  canonicalizzazioni corrette (720 mesh dei domini di training, tutte entro 15 gradi dalla convenzione) =
  2 x 0.0406 = **T = 0.0811**. Una mesh di test con residuo > T e' un fallimento; la trasformazione si applica
  comunque (nessun ripiego), il tasso si riporta con IC di Wilson, per dominio e per topologia. Si riporta
  anche l'angolo dalla convenzione nota del dominio (HIFI3D e NoW: identita'; FaceVerse: Rx180) e quante mesh
  ne distano piu' di 30 gradi. Tempo per mesh: secondi di `canonicalize` su 1 thread.
- **Valutazione** (stessi protocolli e repliche dei riferimenti, le righe senza canonicalizzazione rifatte
  come controllo e attese identiche):
  - HIFI3D, Spearman con la GT maxabs su `nocrop_cross` (primario), `all_cross`, `subject_pair_mean`, sulle
    righe delle pair_metrics di e108 e col seme della riga e108 pubblicata per gruppo; riconoscimento (rank-1,
    mAP, AUC) sulle 5 topologie senza crop (primario) e a parte col crop, `bootstrap_counts` col seme
    `expr_recognition`;
  - FaceVerse con espressioni: riconoscimento, stesse repliche; riferimento = e108 in convenzione BFM (la riga
    di `curve.md`); la convenzione ICT fissa e' di un altro agente e, se c'e', si mostra accanto;
  - NoW: tau per immagine sui 3 metodi pre-registrati (repliche di `now_summarize`, seme 1234).
  - Delta appaiati canonicalizzato - senza, con IC.
  - Baseline a parita' di allineamento (richiesta del critic): Chamfer faceBench 4096 punti (HIFI3D e
    FaceVerse) e Chamfer grezza (NoW) calcolate sulle STESSE mesh canonicalizzate, coi semi delle matrici
    pubblicate; delta canonicalizzato - senza anche per loro, e e108 - Chamfer a parita' di allineamento.

## E3b: rimesh uniforme al test

- **Generatore** (`aau/evidence/e3_breakdown/remesh.py`): clustering di Voronoi per area con triangolazione
  duale (stile ACVD). Mesh nel frame normalizzato (baricentro e raggio RMS per area); K = A / ((sqrt 3 / 2)
  L^2) celle; suddivisione a punto medio finche' il 99-esimo percentile dei lati e' < L/3; Lloyd pesato per
  area (20 iterazioni); celle definitive geodesiche (Dijkstra multi-sorgente); triangolo duale per ogni
  triangolo fine con tre celle diverse; vertici = centroidi proiettati sulla superficie originale. Diverso da
  tutti i generatori delle topologie di test (decimazione quadrica, smoothing umbrella + quadrica,
  suddivisione + quadrica).
- **Lato fisso, uguale per tutte le mesh di tutti i domini: L = 0.035** raggi RMS, cioe' la densita' della
  `original` ICT (area / raggio^2 mediana 9.76 -> circa 9.2k vertici). Applicato a OGNI mesh in ingresso
  (tutte e 6 le topologie, crop e noisy compresi; su noisy l'area e' gonfiata dal rumore e ne escono piu'
  vertici). FaceVerse: rimesh, poi facce invertite (convenzione BFM, quella della riga e108 di riferimento).
- **Valutazione:** HIFI3D Spearman (stessi gruppi e semi di E2) e riconoscimento; FaceVerse con espressioni
  riconoscimento; rank-1 per coppia di topologie (la domanda: original <-> down8k recupera?); delta appaiati
  rimesh - e108.
- **Diagnosi di down8k** (stessa catena, un passo cambiato al test, `embed_variants.py`): centro per area
  invece che media dei vertici; pooling medio pesato per area invece che per vertice; entrambi. Controllo: la
  variante `base` deve riprodurre gli embedding esistenti.

## Storia del metodo (solo domini di training)

1. Prima versione (ICP di similarita' unilaterale sorgente -> riferimento, scala libera): gli start sbagliati
   vincevano rimpicciolendo la sorgente (ICT noisy messa in Rx180 con residuo 0.0046), 46 s per mesh.
2. Seconda (scala limitata, residuo bilaterale solo per scegliere lo start): ICT corretto, BFM e GNM con pitch
   incoerenti fino a 18 gradi fra topologie e qualche flip sbagliato (regioni diverse dal riferimento).
3. Terza, quella dichiarata: coppie simmetriche con il nucleo centrale. Calibrazione a 3 soggetti, poi
   completa a 40 per dominio (`calibration/calibration.md`): 0/720 oltre 15 gradi dalla convenzione, consistenza
   fra topologie <= 4.6 gradi, con rotazioni casuali (flip + fino a 30 gradi) 20/360 deviano 5-13 gradi e
   nessuna oltre 30.
Il lato L di E3b e' stato fissato a 0.035 dopo una prova su mesh ICT e BFM (0.045 dava ~5.6k vertici su ICT).
