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
