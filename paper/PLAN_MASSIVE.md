# Training massivo: design

Scritto l'8 ottobre 2026 (PI). È una proposta: le decisioni da prendere sono nella sezione 12.

Fonti dei fatti: la scheda dello scout dell'8 ottobre (trainer, dati, nodi) e `aau/runs/data_scale_ood/` (risultati). Le cifre marcate "stima" non sono ancora misurate.

## 0. Obiettivo e vincoli

**Obiettivo:** un modello che, fuori dominio (3DMM mai visti, ricostruzioni, scansioni reali), faccia due cose:
- (a) misuri la distanza geometrica graduata meglio di ogni alternativa;
- (b) riconosca l'identità attraverso discretizzazioni ed espressioni almeno quanto Chamfer, avvicinandosi ad ArcFace e ICP.

**Vincoli:**
- **Scadenze CVPR 2027:** registrazione 10 novembre, paper 16 novembre. I numeri finali del modello massivo servono entro il 24 ottobre; dopo quella data si scrive.
- **Rete di sicurezza:** il run su scala in corso (job 1060130, fine prevista 9-10 ottobre) è già un risultato pubblicabile. Se il massivo fallisce o tarda, il paper esce con quello.
- **Risorse:**
  - 12 GPU (QoS normal);
  - nodi L40S da 8 GPU, con 128-192 CPU logiche e 735 GB di RAM;
  - limite della partizione: 6 giorni;
  - disco: circa 1.000 GB liberi su 2 TiB.

## 1. Diagnosi dai risultati attuali

| | HIFI3D, distanza graduata (Spearman, senza crop) | HIFI3D, rank-1 | FaceVerse con espressioni, rank-1 | NoW, tau |
|---|---|---|---|---|
| su scala e036 / e108 | **0.677 / 0.630** | 0.811 / 0.782 | 0.518 / 0.625 | 0.299 / 0.277 (e072 0.348) |
| Chamfer | 0.372 | 0.477 | 0.740 | 0.246 |
| ICP + Chamfer | 0.355 | **0.996** | 0.918 | **0.612** |
| NICP per coppia (P2Tri) | 0.389 | 0.990 | **0.959** | n/d |
| ArcFace (ombreggiato / normal map) | 0.245 / 0.242 | 0.984 / 0.998 | 0.750 / 0.867 | 0.271 |

**Cosa ci dicono:**
1. **Distanza graduata:** il modello su scala è nettamente il migliore (0.63-0.68 contro al massimo 0.39). È il nostro punto di forza e va protetto.
2. **Riconoscimento:** il modello è debole. La dispersione dello stesso volto fra discretizzazioni ed espressioni è paragonabile alla distanza fra identità vicine.
3. **Espressioni fuori dominio:** l'invarianza appresa su ICT e GNM non si trasferisce a FaceVerse.
4. **NoW:** domina l'allineamento (ICP + Chamfer 0.612): conta più la posa della metrica.

**Cause probabili e rimedi:**
- **H1, supporto delle identità.** Ogni 3DMM è un sottospazio lineare:
  - BFM ha 40 componenti e solo 392 identità;
  - ICT ha 100 componenti;
  - GNM ne ha 253.

  Il modello impara quei sottospazi. Lo conferma un dato: aggiungere GNM e ICT-55k ha portato HIFI3D da 0.43 a 0.68. **Rimedio:** tutti i 3DMM, più diversità fuori dai sottospazi (sezione 2).
- **H2, squilibrio.** ICT occupa il 77% dei passi, GNM il 14%, BFM il 9%. Il calo di HIFI3D fra e036 ed e108 è coerente con una specializzazione. **Rimedio:** campionamento bilanciato per dominio.
- **H3, espressioni.** Oggi vengono solo da ICT (53 blendshape) e da GNM. **Rimedio:**
  - espressioni anche da FLAME (100), FaceScape (52), BFM (10) e dai dati reali (FaMoS);
  - trasferimento di espressioni fra modelli.
- **H4, loss.**
  - La SmoothL1 su D/media pesa l'errore in valore assoluto, quindi dominano le coppie lontane. Le identità vicine, che decidono il riconoscimento, contano poco.
  - L'invarianza (L_id, peso 0.25, divisa per 256) è debole.

  **Rimedio:** loss in scala logaritmica, invarianza forte e un termine di vicinato (sezione 6).
- **H5, capacità.** Il modello ha 0.69M parametri (larghezza 128, 4 blocchi). **Rimedio:** taglie S/M/L confrontate a calcolo fisso.
- **H6, GT non confrontabile fra domini.**
  - Oggi è maxabs su supporti diversi (BFM solo volto, GNM testa intera), divisa per costanti diverse per dominio.
  - Non ci sono coppie fra domini, eppure NoW confronta mesh di template diversi.

  **Rimedio:** GT unificata (sezione 3).
- **H7, frame e normalizzazione.**
  - Il run è stato fatto senza frame canonico né augmentation.
  - Le coordinate xyz sono divise per max|V|, che dipende dal supporto.
  - Il pooling è una media per vertice, che dipende dalla densità della mesh.

  **Rimedio:** sezioni 4 e 5.

## 2. Dati

| Dominio | Natura | Ruolo |
|---|---|---|
| BFM 3DDFA (40 identità / 10 espressioni) | sintetico | training |
| BFM 2017/2019 (199 / 100) | se arriva (modulo di Basilea) | training, sostituisce il 3DDFA |
| ICT-FaceKit (100 / 53) | sintetico | training |
| GNM Head (253 / 383) | sintetico | training |
| FLAME 2020/2023 Open (300 / 100) | sintetico, già su disco | training |
| FaceScape bilineare (52 espressioni) | sintetico, derivato da scansioni reali | dev nella fase 1, training nel run finale |
| FaMoS (95 persone, registrazioni FLAME) | reale | 80 persone in training, 15 in test |
| Florence 4D (CC BY 4.0) | da verificare | training, se adatto |
| HIFI3D, FaceVerse | sintetico | solo test |
| NoW validation, Multiface (13 persone) | reale | solo test |

**Diversità, non solo quantità:**
1. **Code più pesanti nel campionamento dei coefficienti:** mistura 80% N(0,1) e 20% N(0,2²), troncata a 3.5σ.
2. **Deformazioni locali casuali:** bump RBF da 1-3 mm con raggio 10-30 mm, applicati all'identità neutra, quindi entrano nella GT. Obbligano il modello a guardare la geometria locale e non solo i modi PCA.
3. **Coppie di perturbazione:** X e X+δ, con δ fra 0.2σ e 0.5σ. Danno coppie vicine con GT piccola e nota, cioè supervisione fine per il riconoscimento.
4. **Trasferimento di espressioni fra modelli,** attraverso le corrispondenze della sezione 3. Fase 2, se c'è tempo.
5. **Discretizzazioni:** le 6 attuali più generatori diversi (ACVD, Instant Meshes, quadric a vari livelli, rumore lungo la normale correlato, buchi, bordi irregolari). **Un generatore resta fuori dal training** e diventa una topologia di test in più, il "remesher mai visto".
6. **Supporto:** ritagli casuali; collo, orecchie e nuca quando il modello li ha.

**Da chiedere all'utente, in ordine di valore rispetto al tempo:**
1. **FaMoS:** solo le registrazioni FLAME, non le scansioni grezze da 10 GB a soggetto, più 3-5 soggetti grezzi per i test di robustezza. Va verificato nella pagina di download se le registrazioni sono un archivio separato.
2. **BFM 2017/2019:** modulo online dell'Università di Basilea, di solito rapido.
3. **FaceScape completo:** firma di un docente (già in lista).
4. **Facoltativi, tempi lunghi:** NPHM, LYHM/Headspace (moduli accademici).

## 3. GT unificata

- **Spazio comune:** la regione del volto nella topologia FLAME, cioè da fronte a mento e da orecchio a orecchio, senza orecchie, collo, nuca e interno di occhi e bocca.
- **Una volta sola per ogni 3DMM:**
  1. landmark sul template medio, con la stessa procedura per tutti: render frontale, detector 2D, retroproiezione;
  2. NICP da FLAME al template, guidato dai landmark;
  3. coordinate baricentriche.

  Controlli: residuo sui landmark e render sovrapposti.
- **Per ogni identità:**
  1. si prende la forma neutra;
  2. la si porta sui vertici della regione FLAME con la mappa baricentrica (esatto);
  3. si applica Procrustes di similarità verso la media globale;
  4. si ottiene il vettore s_i, pesato per area.
- **GT:** g_ij = ||s_i − s_j|| / sqrt(area totale), cioè un RMS pesato per area.
  - È una distanza euclidea vera, invariante alla similarità e confrontabile fra domini.
  - Si calcola al volo per tutto il batch.
- **Espressioni:** GT dell'identità neutra, come oggi.
- **Vantaggio in più:** permette il controllo dei quasi-duplicati FRA domini, training contro test, nello spazio unificato (critica 9).
- **Rischio:** le corrispondenze fra template hanno un errore sistematico, che diventa un offset nelle coppie fra domini. **Rimedio:** peso ridotto a queste coppie, con ablazione on/off.
- **Valutazione:** tutti i metodi con entrambe le GT, l'unificata e la maxabs di oggi. **Proposta:** l'unificata come primaria, dichiarata ora, prima di qualsiasi training.

## 4. Input, frame, normalizzazione

- **Frame canonico:** tutti i dati di training nel frame FLAME (+y in alto, +z fuori dal volto, normali uscenti, millimetri), ottenuto con lo stesso allineamento della sezione 3.
- **Augmentation:** rotazioni di ±30° yaw, ±15° pitch, ±10° roll; scala ±10%; traslazione. Non costa nulla: gli operatori sono intrinseci e non vanno ricalcolati.
- **Normalizzazione:** centro più sqrt(area) invece di max|V|, con ablazione.
- **Al test: canonicalizzazione rigida,** uguale per tutti i dati. È un ICP di similarità robusto verso la faccia media, con più inizializzazioni.
  - Non è una registrazione: nessuna corrispondenza e nessuna deformazione. Costa una sola volta per mesh, all'iscrizione.
  - Chiude il confondente del frame (FaceVerse in convenzione BFM o ICT) e dovrebbe aiutare su NoW, dove domina la posa.
  - Ablazione: niente canonicalizzazione, con augmentation SO(3) completa.

## 5. Modello

- **Base:** DiffusionNet con k_eig 128. Ablazione con k_eig 64, che dimezza il costo degli autovettori.
- **Taglie,** scelte sul dev a calcolo fisso:
  - S = attuale (128×4, 0.7M parametri);
  - M = 256×6 (stima: circa 4M);
  - L = 384×8 (stima: circa 12M).
- **Pooling:** media pesata per area (massa) più attenzione. Oggi è una media per vertice, che dipende dalla densità.
- **Embedding:** 256 dimensioni, euclideo, non normalizzato.
- **EMA dei pesi** (0.999) per la valutazione.

## 6. Loss

Notazione: z_m è l'embedding della mesh m, d_mn = ||z_m − z_n||, i(m) è l'identità, g è la GT unificata.

1. **L_grad, distanza graduata** (coppie di identità diverse): Huber(log(d_mn+ε) − log(g+ε) − b), dove b è la mediana nel batch di (log d − log g), senza gradiente.
   - L'errore è relativo: le coppie vicine pesano quanto quelle lontane.
   - La scala è libera, quindi non servono costanti per dominio.
   - Le coppie fra domini hanno peso w_x.
2. **L_inv, invarianza** (stessa identità, viste diverse): d_mn² diviso la mediana di d² fra identità diverse, senza gradiente sul denominatore. Peso alto (λ = 1): attacca direttamente il riconoscimento.
3. **L_nbr, vicinato:** KL(P_m || Q_m).
   - P_m(n) ∝ exp(−g²/2σ_m²): la massa maggiore va alle altre viste della stessa identità, che hanno g = 0.
   - Q_m(n) ∝ exp(−d²/2τ_m²), con τ_m = e^b σ_m.
   - σ_m è la distanza GT dal terzo vicino.

   Mette insieme riconoscimento e struttura graduata locale senza distruggere quella globale, che è il problema delle contrastive pure.
4. **Solo per dati reali con le sole etichette:** SupCon.

**Totale:** L = L_grad + λ_inv L_inv + λ_nbr L_nbr (+ λ_sup L_sup).
- Valori di partenza: λ_inv = 1, λ_nbr = 0.5, con una griglia piccola sul dev.
- Sostituisce le tre loss di oggi: S (SmoothL1 lineare), R (hinge) ed L_id (debole).

## 7. Batch e campionamento

- **Per GPU:** 16 identità × 4 viste = 64 mesh, con forward a gruppi di mesh di taglia simile. `v2_work/fastio/batched.py` esiste già, ma il run attuale non lo usa: fa un forward per mesh.
- **Globale, su 8 GPU:** 512 mesh e 128 identità. Embedding e s_i si raccolgono da tutte le GPU per le loss, con costo trascurabile.
- **Scelta delle identità:**
  - 50% casuali, bilanciate per dominio (uniforme fra domini, dati reali al 15-20%);
  - 25% vicini in GT (kNN precalcolato);
  - 25% coppie di perturbazione.
- **Viste:** espressione (p = 0.6), discretizzazione dalla banca, supporto (p = 0.3), rumore (p = 0.5), posa e scala.
- **Batch multi-dominio:** oggi i batch sono monodominio; la GT unificata permette di mescolarli.

## 8. Pipeline e GPU

**Il collo di bottiglia vero sono gli operatori, non la GPU:**
- 9-60 MB per mesh, non compressi;
- 211.7 GiB di RAM per 19.669 mesh;
- un forward per mesh: 38.7 mesh/s su una L40S.

Un milione di viste precalcolate non entra né in RAM né su disco.

**P1 (proposta): produttori e consumatori sullo stesso nodo.**
- **Produttori CPU** (circa 100 CPU logiche): generano viste fresche e ne calcolano gli operatori compressi (fp16, indici int32). Scrivono gli shard in un buffer ad anello in RAM.
- **8 rank DDP:** leggono gli shard in mmap, una sola copia per nodo, e riusano ogni vista R volte con augmentation gratuite (posa e scala).
- **Forward a gruppi.**
- **Stime, da misurare nella fase 0:**
  - 30-50 viste fresche/s per nodo, cioè 5-8 milioni di viste uniche in 48 h (oggi sono 801k);
  - consumo di 400-640 mesh/s su 8 GPU, cioè 20-30 volte il calcolo del run attuale.
- **Uso delle GPU:**
  - run finale su 8 GPU in un nodo;
  - le altre 4 per LODO, ablazioni e valutazioni;
  - 12 GPU su due nodi (6+6, ognuno con i suoi produttori) solo se il test NCCL fra nodi va bene. L'interconnessione oggi è sconosciuta.
- **Robustezza:** riavvio automatico da checkpoint, `--time` realistico, catena di job.

**P2 (rimandata):**
- **Cosa sarebbe:** operatori costruiti su GPU (Laplaciano cotangente, massa, gradienti), diffusione implicita con gradiente coniugato, e una banca di tassellazioni fatta di mappe baricentriche sul template. Darebbe dati infiniti senza CPU né disco.
- **Perché si rimanda:** è la via "mastodontica" vera, ma è una riscrittura di DiffusionNet con rischio numerico.
- **Regola:** si apre solo se la fase 0 misura meno di 20 viste fresche/s per nodo, oppure dopo CVPR.

## 9. Protocollo (dichiarato ora, prima dei numeri)

- **Test,** mai in training né in selezione:
  - HIFI3D;
  - FaceVerse con espressioni;
  - NoW validation;
  - Multiface;
  - 15 persone di FaMoS, sulle scansioni grezze se disponibili;
  - la topologia "remesher mai visto".
- **Dev per gli iperparametri:** FaceScape bilineare. Resta fuori dai run della fase 1, poi entra nel run finale con la configurazione già fissata.
- **Punteggio dev:** media fra Spearman graduato senza crop e rank-1 con espressioni.
- **Checkpoint finale:** l'ultimo EMA.
- **Metriche di test:**
  - Spearman con entrambe le GT: senza crop (primario), `all_cross`, `subject_pair_mean`;
  - riconoscimento: rank-1, mAP, AUC, TAR@FAR;
  - NoW tau;
  - crop riportato a parte.
- **Baseline:**
  - Chamfer, ICP + Chamfer;
  - NICP per coppia e su template;
  - ArcFace su render ombreggiati e normal map;
  - spettrali, Uni3D, OpenShape;
  - il run su scala attuale e il congiunto.
- **Analisi, non selezione:**
  - curva "prestazioni fuori dominio contro numero di 3DMM in training" (da 1 a 5 domini, a calcolo fisso). È la figura centrale per le critiche 1 e 5;
  - LODO sui domini di training.
- **Modello di rilascio (critica 10):**
  - dopo il paper, un run su tutto, test inclusi, usato solo per il rilascio e mai per le affermazioni del paper;
  - FLAME e FaceScape hanno licenze non commerciali, quindi i pesi rilasciati saranno non commerciali.

## 10. Calendario

| Fase | Date | Cosa | GPU |
|---|---|---|---|
| 0 | 8-11 ottobre | GT unificata e corrispondenze; generatori FLAME, FaceScape ed espressioni; trainer v3 (DDP, gruppi, produttori, loss, EMA); misure di viste fresche/s, throughput a gruppi per taglia, NCCL fra nodi | 1-2 |
| 1 | 12-16 ottobre | circa 12 run piccoli da 1 GPU × 8 h: loss, bilanciamento, taglia, normalizzazione, k_eig, coppie fra domini; curva dei domini | 8-10 |
| 2 | 17-20 ottobre | run finale, 8 (o 12) GPU per 60-72 h | 8-12 |
| 3 | 21-24 ottobre | valutazione su tutti i test, critic, numeri finali | 2-4 |
| scrittura | 25 ottobre-16 novembre | paper | — |

## 11. Rischi

- **Tempo.** La rete di sicurezza è il run su scala attuale. Nessuna fase può sforare di più di 2 giorni senza riaprire il piano.
- **Qualità delle corrispondenze fra template.** Si controlla con il residuo sui landmark e con il peso w_x in ablazione.
- **Il riconoscimento potrebbe restare sotto ICP e ArcFace.** In quel caso il paper si regge sulla distanza graduata, sul costo e sulla robustezza, che sono già dimostrati.
- **Coda per 8 GPU in un nodo.** Il trainer è elastico (accetta un numero variabile di GPU) e il job va sottomesso presto, con un `--time` stretto.
- **RAM.** Il buffer ad anello sta in `/tmp` e conta contro `--mem`. Va dimensionato con una prova, non a stima: la lezione dell'OOM del 7 ottobre.

## 12. Decisioni per l'utente

**Approvate dall'utente l'8 ottobre 2026: tutte e cinque.** L'utente fornisce FaMoS e richiede BFM 2017/2019. Il piano resta soggetto alle correzioni del critic sul design.

1. **Test fissi fuori dal training** (HIFI3D, FaceVerse, NoW, Multiface, 15 persone FaMoS), con LODO solo come analisi. Raccomandato.
2. **GT unificata come primaria,** dichiarata ora, con la maxabs come secondaria.
3. **Canonicalizzazione rigida al test** come impostazione primaria.
4. **P1 subito, P2 rimandata.**
5. **Dati:** registrazioni FaMoS e BFM 2017/2019.

## Costo in agenti

- **Fase 0:** 2 coder Opus in parallelo (uno per GT e dati, uno per il trainer), più 1 critic su questo design.
- **Fase 1:** un runner Haiku lancia i run e raccoglie i risultati.
- **Fase 3:** 1 critic sul risultato finale.

## 13. Evidenze prima del run (decisione dell'utente dell'8 ottobre)

Il run massivo non parte finché le scelte principali non hanno un'evidenza misurata. Ogni esperimento costa al massimo 1 GPU per poche ore oppure nessun training.

| # | Scelta da sostenere | Evidenza già in mano | Cosa manca | Esperimento | Costo |
|---|---|---|---|---|---|
| E1 | La varietà di 3DMM, non la quantità, migliora il fuori dominio | HIFI3D, distanza graduata senza crop: BFM-only 0.206, ICT-only 0.382, BFM+ICT 0.428, BFM+ICT+GNM su scala 0.63-0.68 | Il salto confonde varietà e quantità (64k identità contro circa 4k) e forse la vicinanza GNM-HIFI3D | **Fattoriale 2×2** (2 o 3 domini × poche o molte identità, stessi passi) più **GNM-only**; più avanti FLAME e FaceScape. Matrice di trasferimento su HIFI3D, FaceVerse, FaceScape (dev) e NoW | 5-6 run piccoli, trainer attuale |
| E2 | La canonicalizzazione rigida al test aiuta | Studio del frame (`aau/runs/ws_frame/`); su NoW ICP + Chamfer 0.612 contro Chamfer 0.246 | Effetto sul nostro modello | e108 rivalutato con canonicalizzazione su HIFI3D, FaceVerse e NoW | nessun training |
| E3 | Il riconoscimento debole è un problema di invarianza e di risoluzione fine (H4) | Graduata forte (0.63) ma rank-1 0.78 | Da dove viene l'errore: topologia (noisy? up60k?), espressione, vicini | Breakdown per coppia di topologie; dispersione intra-identità contro distanza dal vicino più prossimo, sugli embedding e108 | nessun training |
| E4 | La loss nuova migliora il riconoscimento senza peggiorare la graduata | nessuna | tutto | Ablazione a scala piccola: loss attuale contro log, +inv, +nbr | 3-4 run piccoli, modifica del trainer |
| E5 | Il bilanciamento per dominio riduce la specializzazione | e036 → e108 su HIFI3D cala da 0.677 a 0.630, IC sovrapposti | confronto diretto | bilanciato contro attuale, stessi dati e passi | 1 run |
| E6 | Un modello più grande aiuta | nessuna | tutto | S contro M a calcolo fisso | 1 run |
| E7 | Pooling per area e sqrt(area) sono più robusti al supporto | ricetta BFM-only con area-norm | confronto | ablazione | 1-2 run |
| E8 | La GT unificata è valida | Spearman fra GT maxabs e GT dei coefficienti su HIFI3D: 0.089, quindi la scelta della GT pesa | corrispondenze; correlazione con la maxabs; accordo con lo studio umano | costruzione della GT più i controlli | CPU, studio umano |
| E9 | P1 regge il throughput | stime | misure | eigsh/s su un nodo L40S, forward a gruppi per taglia, NCCL fra due nodi | job brevi |

**Regola di avvio del run massivo:**
- **Condizioni necessarie:**
  - E1 mostra che i domini aggiunti aiutano a parità di identità;
  - E9 dà almeno 20 viste fresche/s per nodo;
  - E8 ha corrispondenze con un residuo sui landmark accettabile.
- **Configurazione:**
  - E4-E7 decidono la configurazione;
  - una scelta che non mostra un vantaggio sul dev torna all'impostazione attuale.
- **Se E1 smentisce la tesi** (conta solo la quantità), il piano cambia: più identità dagli stessi domini, che costa meno.

**Ordine:** prima E2, E3 ed E1, perché toccano la tesi centrale e costano poco. Poi E8 ed E9, che servono comunque al trainer. Infine E4-E7, che richiedono modifiche al trainer.

## 14. Revisione dopo il critic (8 ottobre, verdetto BLOCCANTE)

Ha la precedenza sulle sezioni precedenti dove le contraddice.

1. **Le loss della §6 collassano.** Il critic lo ha verificato su un modello giocattolo: con normalizzatori senza gradiente e senza ancora di scala, la distanza mediana scende a 1e-6 e compaiono NaN. **Correzioni:**
   - normalizzatori con gradiente (come lo stress di `latent_loss.py`), oppure un'ancora di scala (la hinge a margine fisso);
   - test anti-collasso obbligatorio per ogni variante;
   - L_nbr spento di default: con σ_m al terzo vicino cade sulle viste sorelle, a g=0.
2. **La diagnosi H4 è sbagliata.** Rank-1 di e108 per coppia: original↔noisy circa 1.00, remesh↔up60k 0.98-1.00, ma original↔down8k 0.33-0.36. Il guasto è fra mesh native e rimeshate, cioè densità, pooling per vertice e normalizzazione (H7), non fra identità vicine. **Nuova priorità:**
   - pooling per area e normalizzazione sqrt(area);
   - test SENZA training: rimesh dell'input a una densità comune (E3b).
3. **La stima di P1 è smentita.**
   - `compute_operators` costa 2.95 s per V=9.4k: la eigsh 1.2 s, il resto è il loop Python di `build_grad`. Con 16 processi si erano misurate 1.67 mesh/s.
   - **Gate:** `build_grad` vettorizzato + k_eig 64 + V ≤ 10k devono dare almeno 20 viste/s per nodo (E9d).
   - Se il gate non passa, si sceglie fra P2 e un pool fisso di viste precalcolate: si decide con le misure.
4. **La GT primaria resta la maxabs,** già dichiarata. Questa decisione rivede la §3 e la decisione 2. Il critic segnala che la maxabs penalizza l'allineamento (ICP+Chamfer 0.401 contro Chamfer 0.876 su original→original) e che con la GT unificata nessun metodo è ancora stato misurato. L'unificata diventa secondaria, misurata ORA su tutti i metodi esistenti prima di qualsiasi training nuovo.
5. **Popolazione e test.**
   - FaceScape bilineare resta SOLO dev, mai in training: così HIFI3D e FaceVerse restano fuori popolazione, come vuole la risposta alla critica 8.
   - HIFI3D, con 37 bracci già valutati, e FaceVerse sono in pratica "visti durante lo sviluppo" e vanno dichiarati così.
   - I test intatti sono: scansioni di test FaMoS (persone escluse dal training), NoW e Multiface.
6. **Simmetria delle baseline.** Ogni canonicalizzazione applicata al nostro modello si applica anche a Chamfer.
7. **Calendario e ablazioni ridotti:**
   - (a) pooling/normalizzazione;
   - (b) loss v2 contro una loss log ancorata;
   - (c) bilanciamento.

   Niente S/M/L a calcolo piccolo: la taglia si decide con il throughput.

   Il run massivo parte dopo le evidenze E1, E3b, E8 ed E9: realisticamente dopo il 16 ottobre. La rete di sicurezza resta il run su scala.
8. **`batched.py`:**
   - mai misurato su GPU;
   - i gruppi vanno fatti a blocchi diagonali o con bucket larghi;
   - niente fp16 sui valori sparsi.

## 15. Evidenze raccolte (8 ottobre, sera)

| # | Esito | Conseguenza per il design |
|---|---|---|
| E2 | La canonicalizzazione rigida al test NON aiuta. HIFI3D peggiora (Spearman −0.074, rank-1 −0.050); FaceVerse e NoW sono neutri. | **Niente canonicalizzazione al test.** La decisione 3 della §12 è revocata. |
| E3 | Il 74% degli errori su HIFI3D cade sulle coppie con down8k. La causa: centro e pooling sono medie per vertice. Pesati per area SOLO al test, original↔down8k arriva a 1.00 ma noisy scende a 0.7. Su FaceVerse, nelle stesse identità neutre, e108 guadagna +0.291 di rank-1: domina l'espressione. | **Centro e pooling per area, con aree robuste,** addestrati con le viste noisy. **Più fonti di espressioni** (FLAME, BFM 2019, GNM, ICT, FaMoS reali). |
| E3b | Il rimesh uniforme al test sistema down8k ma peggiora noisy, crop e la graduata (−0.174); su FaceVerse +0.142 di rank-1. | Non va adottato come pre-elaborazione; conferma che il problema è la densità. |
| E3c | La struttura locale di e108 non è peggiore delle baseline nello stesso decile: batte Chamfer (+0.079) ed è pari a ICP+Chamfer. | **La loss resta v2.** Le varianti log sono archiviate. |
| E8 | Con la GT unificata (Procrustes) e108 scende a 0.301 su HIFI3D, contro NICP 0.577 e ICP+Chamfer 0.522. | **Il target del training è la domanda aperta:** cella E1 C3F-UGT. Arbitro della GT: lo studio umano. |
| E9 | `build_grad` vettorizzato; 42-126 viste/s per nodo. | P1 è fattibile. Pre-pass E1 1.64× più veloce. |
| D1 | Sul dev FaceScape e108 ≈ Chamfer nella graduata (+0.032 n.s.) e peggiore nel rank-1 con espressioni (−0.139). | Il vantaggio di HIFI3D non è generale; vedi H8, il supporto simile a GNM. |

## 16. Revisione dei risultati delle evidenze (critic del 9 ottobre, BLOCCANTE)

Correzioni d'interpretazione, che hanno la precedenza sulla §15:
1. **E8: il ribaltamento di classifica è un effetto del DIVISORE DI SCALA, non della posa.** Su 500 identità HIFI3D (job 1062067):
   - ρ(maxabs, maxabs con allineamento rigido) = 0.975;
   - ρ(maxabs, normalizzazione per centroid size) = 0.638.

   La GT maxabs divide di fatto per l'estensione verticale della patch, cioè un vertice del bordo; quella unificata divide per la centroid size. Nessuna delle due conserva la dimensione del volto. Il modello e108 normalizza l'ingresso con lo stesso divisore della maxabs, quindi parte del suo vantaggio con quella GT nasce dalla costruzione. **La domanda per lo studio umano diventa: quale normalizzazione di scala corrisponde alla somiglianza percepita?**
2. **E3c: la conclusione "la loss non va cambiata" è RITIRATA.**
   - Il calo dal globale al locale di e108 è maggiore di quello delle baseline.
   - Il vantaggio di e108 sta solo nel decile più lontano.
   - Il confronto "pari a ICP+Chamfer" è assenza di prova, non equivalenza.

   Si aggiunge un braccio `log+inv` alle ablazioni v3.
3. **E3: la causa di down8k va riformulata** come "centro, scala e pooling per vertice": la variante cambiava anche la scala.
4. **Competitori:**
   - va aggiunta la riga di riconoscimento di e108 (0.782, sotto NICP su template 0.875 e ICP+Chamfer 0.996);
   - tutti i competitori vanno valutati anche con la GT unificata;
   - OpenShape e Uni3D vanno valutati anche nel frame esatto del loro training.
5. **C3F-UGT e `ugtmix`:** la GT unificata va TARATA (mediane per dominio allineate alla maxabs), altrimenti il confronto mescola contenuto e scala della GT.
6. **Promossi:** ArcFace contro modello (VIA LIBERA; e108 era fissato prima, anche se e036 fa meglio) e FaMoS (VIA LIBERA). E10 passa con riserve sui confronti di velocità.

## 17. Pipeline P1 a streaming (9 ottobre): pronta, `v3_work/stream/`
- **Verifiche eseguite:**
  - operatori fp32 equivalenti alla pipeline attuale: embedding entro 3e-7;
  - GT al volo identica a `UNIFIED_GT`: 604k coppie, entro 1.3e-6 mm;
  - 2000 passi stabili;
  - DDP con shard disgiunti.
- **Produttori** (64 CPU EPYC): k 64 → 90 viste/s, k 128 → 38 viste/s.
- **Decisioni del PI:**
  - **autovettori in fp32** (fp16 scarta 2.5e-4 sull'embedding: piccolo, ma inutile rischiarlo; il costo è +55% di RAM per vista);
  - **8 core riservati al trainer:** senza, la contesa CPU porta il passo da 1.10 s a 1.74 s.
- **Ancora fuori:** deformazioni RBF, coppie di perturbazione, trasferimento d'espressioni. Si aggiungono se le evidenze le giustificano.
- **BFM 2019** ha ora la mappa unificata: 1478/1478 punti, residuo sui landmark tenuti fuori 0.89 mm.

## 18. Esiti delle ablazioni v3 (9 ottobre)

Sottoinsieme C3F, 21.096 passi, un seme; delta contro ctrl.

| Braccio | Dev FaceScape | FaceVerse rank-1 | Esito |
|---|---|---|---|
| arearobust | **+0.122 [+0.086, +0.160]** | **+0.222** | ADOTTATO |
| bal | +0.050 [+0.032, +0.070] | | ADOTTATO |
| area | n.s. | | no |
| loginv | +0.116 | −0.057 | no per regola: HIFI3D maxabs −0.371. Però HIFI3D unificata +0.097 e NoW +0.146 |
| ugtmix (non tarata) | −0.131 | | no; HIFI3D unificata +0.145 |

**Nota:** loginv e ugtmix si spostano in direzioni opposte a seconda della GT. La scelta della GT, §19, decide anche la loss.

## 19. GT e allineamento "equi": principio e piano (9 ottobre, PI con l'utente)

**Il problema.** Entrambe le GT usate finora normalizzano la scala PER IDENTITÀ:
- maxabs divide per il proprio max|coord| (`make_zs_expr_topologies.py:109`);
- l'unificata usa Procrustes con centroid size.

Così cancellano la dimensione e ridistribuiscono altezza e larghezza. È in tensione con la tesi del paper, che sostiene che allineamento e normalizzazione specifici cancellano tratti identitari.

**Cosa toglie davvero identità**, dalle misure del critic su HIFI3D:
1. le trasformazioni PER COPPIA (ICP a coppie): rompono la coerenza globale e comprimono le distanze, come dice la tesi;
2. il fitting non rigido e le corrispondenze per punto più vicino: assorbono le differenze locali;
3. la normalizzazione di SCALA per identità: cancella la dimensione (ρ 0.638 contro maxabs);
4. l'allineamento RIGIDO per identità: ridistribuisce poco ("effetto Pinocchio"; ρ 0.975), non rimuove la geometria relativa ed è necessario quando la posa non è osservabile.

**Principio dell'invarianza minima.** La GT è invariante SOLO a ciò che non si può osservare nei dati di destinazione:
- moto rigido per i dati metrici;
- similarità per i dati senza scala, come le ricostruzioni monoculari di NoW.

La trasformazione si sceglie per identità con una regola fissa, mai per coppia, e con stimatori robusti.

**Famiglia di GT** (E12, in corso):
- GT-F "form" (mm, frame canonico, allineamento rigido robusto solo per i dati reali);
- GT-EDM, senza allineamento (matrici delle distanze interne, Lele & Richtsmeier);
- GT-S "shape" (centroid size);
- maxabs, legacy.

Baseline banali "solo dimensione" e "solo altezza", per vedere quanto di ogni GT spiega un solo scalare.

**Arbitri:**
- **(a) oggettivo:** l'identificabilità su catture reali ripetute (FaMoS neutre di sequenze diverse). La GT che conserva più identità separa meglio le persone. Regola dichiarata prima dei numeri;
- **(b) percettivo:** lo studio umano. ATTENZIONE: i render attuali (`aau/baselines/render_cache.py`) normalizzano ogni mesh con maxabs prima della camera fissa, quindi le persone vedono volti "alti uguali". Lo studio così è sbilanciato verso maxabs e non può arbitrare scala né dimensione. Serve una versione nuova con render a scala assoluta e triplette scelte dove le GT si contraddicono;
- **(c) robustezza:** le conclusioni del paper devono reggere con GT-F e con GT-S.

**Modello:**
- ingresso in mm con normalizzazione GLOBALE (costante) e centro per area robusta;
- embedding FATTORIZZATO z = (log dimensione, forma), con augmentation di scala per l'equivarianza;
- distanza form derivata con la formula size-and-shape.

Un modello unico serve entrambi i casi d'uso (coder in corso).

**Baseline eque:** ogni metodo riceve la rimozione dei disturbi coerente con la GT. Per GT-F: ICP rigido, senza scala, in mm. Per GT-S: ICP di similarità.

## 20. Esito di E12 e decisione sulla GT (9 ottobre)

**Arbitro di identificabilità** (FaMoS, 95 persone, 2.639 catture neutre): la GT **F (form, mm) con allineamento rigido robusto per identità** separa meglio le persone, con AUC 0.9928 [0.9891, 0.9957]. Batte S, EDM ed EDM-s con IC che escludono lo 0: **la taglia è un tratto identitario e normalizzarla toglie identità.**

**F pura** (nessuna trasformazione per identità) NON è equa. Su HIFI3D è dominata dalla posizione del volto nel frame del modello: spostamento mediano 5.5 mm, p95 15.9 mm. Non si osserva dalla geometria, e tutti i metodi restano sotto 0.12.

**Decisione: GT di riferimento = F + rigida robusta ("FR").** È coerente con il principio dell'invarianza minima: si toglie solo la posa, che non si osserva; la taglia resta.

**Con FR su HIFI3D nocrop:**

| Metodo | Spearman |
|---|---|
| NICP P2Tri | **0.367** |
| e108 | 0.194 |
| Chamfer | 0.155 |

- NICP − e108 = +0.174 [+0.092, +0.253].
- Sul dev FaceScape e108 batte Chamfer con ogni GT (con F: +0.119).
- Su FaceVerse nessuna differenza è significativa.

**Lettura:**
- il vantaggio del modello attuale esiste solo con la maxabs, la GT su cui è addestrato e che ha lo stesso divisore di scala del suo ingresso;
- il modello è cieco alla taglia per costruzione, quindi con FR non può competere.

**L'esperimento decisivo ora:** braccio `factorized` (ingresso in mm, ramo taglia più ramo forma, GT-SR) e braccio `ctrl-FR`. La domanda: il modello addestrato sulla GT giusta si avvicina a NICP?

**Conseguenza per la tesi del paper:** "l'allineamento toglie identità" va precisata.
- La normalizzazione di SCALA per identità toglie identità: l'arbitro lo mostra.
- L'allineamento RIGIDO per identità è necessario e fa parte della GT.
- Il vantaggio dei metodi con registrazione (NICP) con FR va riconosciuto. La metrica appresa deve vincere sul costo (circa 0.1 s di iscrizione e ricerca istantanea, contro 1.15 s per mesh e un ICP per coppia) e su robustezza e assenza di corrispondenze, oppure avvicinarsi in accuratezza.

## 21. Gate di lancio del run massivo (decisione dell'utente, 9 ottobre sera)

**Il run massivo NON parte senza una revisione insieme all'utente.** Prima del lancio il PI presenta:
- gli esiti dei bracci decisivi (ctrl-FR, factorized, factorized2) e del run a scala piena con la GT FR, contro NICP, ICP+Chamfer e Chamfer;
- lo stato delle "verità": arbitro E12, studio umano v2, letteratura;
- la configurazione proposta (testa, GT, domini, quota di espressioni, passi, GPU: 12 L40S più le A100 secondarie);
- il critic sulla configurazione.

Nel frattempo si aspettano i risultati, senza lanciare il run.

## 22. Configurazione proposta per il run massivo (bozza per la revisione con l'utente, 10 ottobre)

Bozza del coder per il PI, da rivedere con l'utente e col critic (gate della §21). Run massivo non lanciato; lanciati
solo lo smoke a 2 L40S (job 1067645) e la prova a 6 + 6 L40S (job 1067683, in attesa). Ogni numero ha la sua fonte;
"R" = `aau/runs/evidence/trainer_v3/factorized_results.md`. IC 95% bootstrap per soggetto.
Aggiornata il 10 ottobre notte dopo due giri del critic sulla configurazione (22.0).

### 22.0 Correzioni dopo i due giri del critic

**Primo giro** (commit 26aee75, 2ec7c79, 85de102):
- lr 1e-4 costante ed esplicito (`--lr-constant`). Prima valeva il default v2 (ReduceLROnPlateau), che non è
  quello del C3M; ora, senza uno scheduler esplicito, lo stream non parte.
- Batch a dominio singolo con passi uguali fra i domini (`--stream-batch-domains single`), come `--sampler balanced
  --domain-alpha 0` del C3M. Lo stream mescolava i domini nel batch.
- Ricetta per rank del C3M resa esplicita: 5 identità × ≤ 6 viste, produttori con 6 viste per identità.
- `STREAM_SOURCES` obbligatorio, senza default. Preset `validated` (bfm2019, ict, gnm) e `max` (+ FLAME 2023 Open,
  FaMoS TRAIN). Maschera di taglia esplicita. Checkpoint ogni ~10% del run.
- `fact_calib.py calib-ckpt`: c di un checkpoint qualunque, anche dello stream (verificato sul C3M e205: 0.2246).

**Secondo giro** (verdetto RISERVE: tre bloccanti chiusi, ma `validated` cambiava sei cose insieme rispetto al
C3M). Decisione del PI: `validated` è la ricetta del C3M scalata. Ogni deviazione resta una variabile esplicita e
sta nel preset `max` (22.7). In codice, `STREAM_RECIPE=c3m` è il default con `validated`:
- niente moltiplicatori mm_aug. Il critic ha misurato il 27-33% di gruppi ibridi e il 13-18% con espressioni
  trasferite da un altro dominio;
- quota d'espressioni del C3M per dominio (vedi 22.7);
- nessuna rotazione casuale a ogni uso (`STREAM_ROT=0`). Prima era ±30/15/10° fissa, non sovrascrivibile;
- discretizzazioni senza reinserimento nell'identità (`--label-draw perm`): 6 viste, 6 discretizzazioni diverse,
  come `_sample_subject_mesh_entries` del C3M. Prima si estraevano con reinserimento;
- batch misti contati anche per FONTI (`prov.sources` dei gruppi), non solo per template: contatore e log dei
  primi 200 batch per rank;
- log del riuso corretto: il tetto è morbido (22.7, "Riuso").

### 22.1 Esito delle evidenze

**HIFI3D, GT FR (primaria), `nocrop_cross`, ultimo checkpoint EMA** (R, prima tabella e delta appaiati).
Delta = braccio − riferimento, stesse righe e repliche; seme 1234 ; seme 2345.

| braccio (distanza) | Spearman s1234 ; s2345 | − ICP + Chamfer mm | − taglia oracolo | − stimata + ICP cs cal. |
|---|---|---|---|---|
| factorized (d_F cal.) | 0.749 [0.683, 0.811] ; 0.731 [0.660, 0.793] | +0.106 [+0.062, +0.154] ; +0.088 [+0.040, +0.138] | +0.012 [−0.034, +0.062] ; −0.006 [−0.058, +0.048] | +0.034 [−0.001, +0.071] ; +0.017 [−0.019, +0.057] |
| ctrl-FR (z) | 0.757 [0.686, 0.818] ; 0.746 [0.674, 0.812] | +0.114 ; +0.103 (IC > 0) | +0.021 ; +0.009 (n.s.) | +0.043 [−0.000, +0.089] ; +0.032 [−0.014, +0.079] |
| dual (z_F) | 0.764 [0.696, 0.826] ; 0.771 [0.705, 0.828] | +0.121 ; +0.128 (IC > 0) | +0.028 ; +0.034 (n.s.) | **+0.050 [+0.009, +0.090] ; +0.057 [+0.010, +0.105]** |
| C3M e123 ; e205 (d_F cal., un seme) | 0.748 [0.677, 0.811] ; 0.739 [0.668, 0.803] | +0.105 ; +0.096 (IC > 0) | +0.012 ; +0.003 (n.s.) | +0.034 [−0.007, +0.075] ; +0.025 [−0.012, +0.066] |

Riferimenti sulle stesse righe (R, "Baseline" e analisi della taglia): ICP + Chamfer in mm 0.643 [0.572, 0.703];
NICP su template in mm 0.614 [0.505, 0.697]; taglia stimata 0.600 [0.498, 0.690]; taglia oracolo 0.736
[0.658, 0.802]; taglia stimata + ICP cs cal. 0.714 [0.635, 0.778]; oracolo taglia + ICP cs cal. 0.841 [0.785, 0.882].
Tutti i bracci stanno sotto il tetto oracolo + ICP cs cal. (factorized s1234 −0.093 [−0.128, −0.060]).

**Altri domini** (factorized, d_F cal. per FR e d_P per SR; s1234 ; s2345):

| dominio, GT | factorized | riferimento migliore | delta | fonte |
|---|---|---|---|---|
| HIFI3D, SR | 0.622 [0.546, 0.691] ; 0.613 [0.531, 0.689] | NICP per coppia cs 0.595 [0.546, 0.648]; ICP cs 0.581 [0.530, 0.631]; ICP mm 0.366 [0.281, 0.454] | − NICP cs +0.027 [−0.029, +0.083] ; +0.018 [−0.046, +0.082] (n.s.) | R; `baselines_mm/summary.md` |
| dev FaceScape, FR | 0.659 [0.590, 0.721] ; 0.667 [0.593, 0.730] | NICP tpl mm 0.544 [0.469, 0.609]; ICP mm 0.467 [0.401, 0.528] | − NICP tpl +0.119 [+0.030, +0.210] ; +0.127 [+0.039, +0.213] | R |
| dev FaceScape, SR | 0.747 [0.678, 0.799] ; 0.754 [0.686, 0.811] | ICP mm 0.491 [0.427, 0.547]; NICP cs 0.398 [0.329, 0.464] | − NICP cs +0.349 [+0.291, +0.401] ; +0.355 [+0.290, +0.412] | R |
| FaceVerse espr., FR | 0.303 [0.235, 0.369] ; 0.318 [0.253, 0.377] | ICP mm 0.337 [0.262, 0.410] | −0.035 [−0.105, +0.036] ; −0.019 [−0.091, +0.049] (pari) | R |
| FaceVerse neutra, FR (d_F grezza) | 0.337 [0.257, 0.409] ; 0.373 [0.297, 0.444] | ICP mm 0.409 [0.328, 0.485] | regola: INTERMEDIO (contano dominio ed espressioni) | `faceverse_neutral/results.md` |
| FaMoS TEST, FR (15 soggetti, descrittivo) | 0.678 [0.237, 0.897] ; 0.654 [0.214, 0.896] | stimata + ICP cs cal. 0.843 [0.623, 0.941]; ICP mm 0.739 [0.362, 0.914]; dual z_F 0.849 ; 0.860 | IC troppo larghi per concludere | R |

**Criterio "forma oltre la taglia"** (emendamenti 4 e 5, HIFI3D, entrambi i semi): **non soddisfatto** per ctrl-FR,
factorized, factorized2 e dual; C3M (descrittivo) neanche. Esplorativa post hoc (R, sez. (i)): il parziale della sola
d_P di factorized s1234 vale 0.554 [0.487, 0.622]: +0.132 [+0.070, +0.196] contro ICP mm, +0.029 [−0.024, +0.086]
(n.s.) contro NICP per coppia cs. Anche con c = 0.5, il valore più vicino alla c ideale di HIFI3D, il criterio
resta non soddisfatto (R, sez. (ii)).

**Cosa possiamo dire.**
- Con la GT FR e l'ingresso in mm, un modello feed-forward batte ICP + Chamfer e NICP su template in mm su HIFI3D
  (IC sopra 0 in entrambi i semi), e li batte nettamente sul dev FaceScape, con FR e con SR.
- Sulla forma (SR) è alla pari con le registrazioni per coppia (NICP cs, ICP cs) su HIFI3D, senza registrazione.

**Cosa non possiamo dire.**
- Che su HIFI3D il modello colga forma oltre la taglia: contro la taglia oracolo e contro "taglia stimata + ICP cs
  cal." i delta non sono risolti, e il criterio preregistrato fallisce.
- Niente su FaMoS (IC larghi circa 0.6) e niente di superiore su FaceVerse.

### 22.2 Testa: factorized con d_F calibrata

Lo decide la regola preregistrata (emendamento 4, sez. 5; R, "Regola dual"). dual non soddisfa la non inferiorità
in nessuno dei due semi. Le righe con estremo inferiore ≤ −0.03:
- HIFI3D SR, u − d_P di factorized: −0.014 [−0.047, +0.019] ; −0.007 [−0.053, +0.035];
- dev FaceScape FR, z_F − d_F cal.: +0.013 [−0.035, +0.059] ; +0.003 [−0.038, +0.041];
- dev FaceScape SR, u − d_P: −0.022 [−0.049, +0.007] ; **−0.033 [−0.061, −0.005]** (s2345, IC tutto sotto 0).

**Esito: si sceglie factorized con d_F calibrata.** Per informazione, senza effetto sulla regola: dual z_F è il
migliore su HIFI3D FR (0.764 ; 0.771) ed è l'unico braccio col delta contro "taglia stimata + ICP cs cal." sopra 0
in entrambi i semi (tabella 22.1). Anche dual non soddisfa "forma oltre la taglia".
La regola si valuta su HIFI3D e dev FaceScape: la testa è quindi scelta anche su HIFI3D (vedi 22.6).

### 22.3 GT, loss e ricetta

- **Valutazione:** GT FR primaria (§20); GT SR per il ramo forma (d_P); maxabs solo legacy, riportata e mai decisiva.
- **Training** (flag verificati nella riga di lancio del C3M, `factorized/c3m/launch.txt`, e nel protocollo):
  u su GT-SR di E12 × kappa (1.0443 nel C3M); s su log centroid size; `--lambda-size 1` (default); loss v2
  (default). Ricetta dei bracci decisivi: arearobust (`--area robust --area-robust smooth`) + bal (`--sampler
  balanced --domain-alpha 0`), `--input-norm global` (L0 = 100 mm), `--scale-aug 0.8,1.25`, EMA 0.999.
- Nello stream lo stesso si ottiene con `STREAM_ARM=factorized` (`massive_node.sh`: `--stream-gt sr`, kappa da
  `c3f/fr_params.json`, uniforme fra domini dei produttori).

### 22.4 k_eig = 128

Regola dell'emendamento 1 (`ablations/k_ablation/results.md`, un seme, e072): k64 perde più di 0.03 in tre celle,
HIFI3D maxabs −0.046 [−0.077, −0.018], FaceVerse FR −0.043 [−0.097, +0.014] e SR −0.033 [−0.088, +0.021].
Su HIFI3D con FR e SR perde −0.024 [−0.059, +0.011] e −0.018 [−0.053, +0.016] (n.s.). Il costo: 1.047 contro
0.818 s/passo (A100); nello stream 40.5 contro 89.9 viste/s per nodo (`stream/massive_ready.md` §10). k256 è
abbandonato per costo, per decisione dell'utente (emendamento 1).

Limiti della scelta, da dire all'utente:
- la cella peggiore, l'unica con IC tutto sotto 0, è HIFI3D **maxabs**, la GT legacy che in 22.3 "non è mai
  decisiva". Senza maxabs decidono le celle FaceVerse FR −0.043 [−0.097, +0.014] e SR −0.033 [−0.088, +0.021],
  entrambe n.s.;
- l'ablazione è sui bracci robal, con la testa standard, non sulla testa factorized del run;
- un solo seme;
- k64 è **significativamente migliore** su dev FaceScape SR: +0.029 [+0.003, +0.056] (e072).

k64 darebbe 2.2 volte le viste fresche/s (89.9 contro 40.5 per nodo).

### 22.5 Calibrazione c per checkpoint

- c deriva coi passi: C3M 0.257 a e123, 0.225 a e205 (R, tabella di calibrazione). Un c fisso non va bene.
- Regola proposta: per ogni checkpoint valutato, `tools/fact_calib.py` sugli held-out sintetici (mediana, c_LS come
  sensibilità). c si scrive in un file con hash, nel commit, PRIMA di qualunque eval di test di quel checkpoint.
- **Aperto:** oggi gli held-out sono quelli dello split C3M (bfm 108 REMESH, ict 992, gnm 100). Nel preset
  `massive` dello stream ci sono BFM 2019, FLAME 2023 e FaMoS, senza held-out di calibrazione. Va definito prima del
  lancio: semi riservati per i 3DMM; per FaMoS servono persone TRAIN escluse.
- **Anche con `validated` gli held-out bfm sono BFM REMESH**, cioè non il dominio di training (BFM 2019). Per
  `calib-ckpt` (100 soggetti per dominio) la c del dominio bfm è quindi già fuori dominio.
- **c per dominio, da dichiarare con la c globale.** Smoke 1067645, `epoch010_ema` dopo 1000 passi
  (`smoke_validated_1067645/calib_e010/calib.json`):
  - mediana 12.85, LS 11.45;
  - per dominio: bfm 17.76, ict 9.37, gnm 13.09.

  Il valore è atteso: la scala di z_F è libera e l'EMA a 0.999 dopo 1000 passi è ancora vicina
  all'inizializzazione. Il C3M dà 0.225-0.257 a fine run. Va ricontrollata sul checkpoint al 10% del run: se
  resta lontana da quell'ordine, o se lo scarto fra domini resta circa 2×, si discute prima di proseguire.

### 22.6 Selezione del checkpoint e domini

- Checkpoint primario: l'ultimo EMA (§9), dichiarato ora. Gli intermedi sono descrittivi. Se serve una scelta, si fa
  solo su held-out sintetici o sul dev FaceScape, con la regola scritta prima del run. Mai sui test.
- Il C3M mostra che la scelta conta: da e123 a e205 HIFI3D FR cal. passa da 0.748 a 0.739, dev FaceScape FR cal. da
  0.666 a 0.712 (R).
- **Zero-shot, mai in training.** Ma NON "mai in selezione", come diceva la bozza:
  - la testa (22.2) è stata scelta su HIFI3D e dev FaceScape;
  - k128 (22.4) su HIFI3D, FaceVerse e dev FaceScape.

  HIFI3D e FaceVerse vanno dichiarati "usati nella selezione della configurazione" (oltre che "visti durante lo
  sviluppo", §14.5). Intatti, cioè mai usati né in training né in selezione, restano solo **FaMoS TEST** (15
  persone; intatto solo se FaMoS TRAIN resta fuori dal training, 22.7) e **NoW**. Il dev FaceScape serve solo per
  decisioni prese prima del lancio.
- **Training:** BFM (2019 nello stream; 3DDFA nel C3M), ICT, GNM; FLAME 2023 Open e FaMoS TRAIN dipendono dalle
  domande (b) ed (e) qui sotto.

### 22.7 Dati e mix

**Pipeline.** Il C3M ha letto viste pre-generate (tar e store, 64.400 identità di training: BFM 392, ICT 54.008,
GNM 10.000; `aau/data_scale/split_scale_all.json`). Il run massivo è preparato sullo stream P1
(`stream/massive_ready.md`): identità fresche, GT di E12 al volo, provenienza per vista. Lo stream non è mai stato
provato oltre 2000 passi su 2 L40S (§11 di quel file), né su 6 + 6 GPU (job 1067683 in attesa).

**Ricetta del run principale: `validated` = il C3M scalato** (`STREAM_SOURCES=validated`; `STREAM_RECIPE=c3m` è il
suo default, `massive_node.sh`).
- **Per rank come il C3M:** 5 identità × ≤ 6 viste, produttori a 6 viste per identità, batch a dominio singolo con
  passi uguali fra i domini (alpha 0), lr 1e-4 costante.
- **12 rank invece di 6:** 60 identità per passo, il doppio del C3M, **con lo stesso lr**. Dichiarato: il batch
  globale raddoppia e nessuna regola di scala del lr è applicata.
- **Domini:** BFM 2019, ICT, GNM. BFM 2019 sta al posto di BFM REMESH, che lo stream non ha (392 identità fisse da
  3DDFA nel C3M).
- **Maschera di taglia vuota.** Il C3M toglie bfm dalla MSE di s perché le taglie di BFM REMESH sono normalizzate
  mesh per mesh. Quelle di BFM 2019 sono vere (CV 0.049, `massive_ready.md` §3). `STREAM_SIZE_MASK=bfm2019`
  riproduce la maschera.
- **Quota d'espressioni del C3M per dominio** (`STREAM_EXPR_FRAC=bfm2019=0,ict=0.2315,gnm=0.1968`). Viene dalle
  etichette `rexprA/B` di `factorized/c3m/spec.json` col campionatore v1, che prende 6 etichette su 8, una per
  topologia:
  - ICT: 50.000 identità su 54.008 hanno 2 etichette d'espressione su 8, quindi 0.2315;
  - GNM: 5.038 ne hanno 2 su 8, 4.962 una su 7, quindi 0.1968;
  - BFM: 0.

  Simulato col campionatore su 3000 soggetti per dominio: 0.233 e 0.199. Lo stream fissa la quota per gruppo
  (arrotondamento stocastico di f × 6), con la stessa media.
- **Niente moltiplicatori mm_aug.**
- **Nessuna rotazione a ogni uso.** Il C3M ruota solo nel modo `rotation` del rumore (`--rigid_rot_deg 12`), che è
  uguale nello stream.
- **6 discretizzazioni diverse per identità**, senza reinserimento.
- **Differenze che restano:**
  - identità fresche invece di 64.400 fisse;
  - forward a gruppi (il C3M è sequenziale);
  - nel C3M le espressioni stanno sulla topologia `original`, nello stream su una qualunque delle 6;
  - BFM 2019 e maschera vuota (sopra).

**Deviazioni opzionali: DOMANDE PER L'UTENTE.** Ognuna è una variabile di `massive.sbatch`; il preset `max`
(`STREAM_RECIPE=max`, default per ogni `STREAM_SOURCES` diverso da `validated`) le accende tutte. Nessuna è stata
provata con i bracci decisivi. **Raccomandazione del PI: `validated` per il run principale; le deviazioni solo in
run secondari o ablazioni, una alla volta.**

| | deviazione | variabile | `validated` (C3M) | `max` | pro | contro |
|---|---|---|---|---|---|---|
| (f) | moltiplicatori mm_aug | `STREAM_MM_AUG` | nessuno | ibridi 0.3, trasferimenti 0.15, bump 0.15 | identità fuori dal sottospazio di A (6.5-13% RMS); nessuna più vicina ai test dei puri (`mm_aug/README.md`) | 27-33% di gruppi ibridi, 13-18% con espressioni di un altro dominio (critic) |
| (g) | quota d'espressioni | `STREAM_EXPR_FRAC` | 0 ; 0.2315 ; 0.1968 | 0.5 | E1 premia la cella con più espressioni nel riconoscimento; FaceVerse neutra INTERMEDIO | raddoppia la quota del C3M |
| (h) | rotazione a ogni uso | `STREAM_ROT` | 0 | ±30/15/10° | robustezza alla posa | il C3M non l'ha; l'ingresso globale è già nel frame canonico |
| (i) | FLAME 2023 Open, FaMoS TRAIN | `STREAM_SOURCES` | no | sì | quarto 3DMM aperto; espressioni reali | domande (b) ed (e); licenze (sotto) |
| (j) | batch misti | `STREAM_BATCH_DOMAINS=mixed` | no | no | coppie fra domini nella loss | il C3M è a dominio singolo |
| (k) | discretizzazioni con reinserimento | `STREAM_LABEL_DRAW=replace` | no | sì | up60k più raro (pesi), meno CPU | discretizzazioni ripetute nell'identità |

**Riuso.**
- **C3M:** 60.000 passi × 6 rank × 5 identità × 6 mesh = 10.8 M usi su circa 801k mesh distinte, cioè **13.5 usi
  per mesh**. Coi passi uguali fra domini il riuso è molto diverso fra domini: bfm circa 1531 (392 identità × 6
  mesh), gnm circa 48, ict circa 5. Per identità: 1.8 M usi su 64.400, cioè 28.
- **Smoke 1067645** (2 L40S, 24 produttori, 8.8 viste fresche/s): riuso a regime circa 17.6 (16.7-17.9 nelle
  epoche 7-10), cumulato 13.5 a 1000 passi. Circa il 78% delle estrazioni era oltre il tetto.
- **Il tetto `STREAM_REUSE` 4 è morbido.** Se gli eleggibili non bastano si prendono i gruppi meno usati. 4 è un
  numero di progetto, non un requisito. Il log del trainer lo scrive come tale; `summarize_run.py` riporta il
  riuso a regime (ultime 3 epoche), il cumulato e i gruppi oltre il tetto.
- **Mini-smoke 1067709 con la ricetta c3m** (2 L40S, 500 passi): riuso a regime 20.8, cumulato 14.9. Le viste
  fresche scendono da 8.8 a 6.1/s, perché up60k passa dal 4.8% al 16.7% delle viste (una per identità, come nel
  C3M).
- **Stima a 12 GPU (6 + 6), non misurata:** senza flotta circa 11-18; con la flotta di produttori su CPU
  (`STREAM_EXTRA`) circa 3.5. La stima è fatta con la ricetta di prima: con c3m le viste fresche per processo
  calano di circa il 31%, quindi il riuso sale in proporzione. Da misurare con il job 1067683.

**Quantità.** Nello stream la quantità è limitata dalle viste fresche al secondo, non dal disco. Disco misurato con
getfattr il 10 ottobre sera: 783.5 GB liberi su 2199.0 GB, cioè 483 GB sopra la soglia dei 300. L'anello condiviso
su CephFS ha un tetto imposto di 150 GiB. Sulla via tar del C3M, invece, gli shard ICT (50.000 identità) occupano
132.8 GB, quelli GNM 16.1 GB, e la GT densa cresce come N² (17.2 GB a 65.600 identità, `datasets/SCALE_ALL`).
Prima di usare nv-ai-04 va controllata la memoria libera: era occupata per 965 GB su 980 dai job A100
(`massive_ready.md` §14).

**Rilasciabilità** (`literature/LICENZE_RILASCIO_2026-10-09.md`; non è un parere legale).
- **Nucleo aperto** (`open_core`): GNM Apache-2.0, ICT Light MIT, FLAME 2023 Open CC BY 4.0 "con restrizioni d'uso"
  (Readme del pacchetto; da verificare). Mesh ridistribuibili al 100% (`massive_ready.md` §7).
- **BFM 2019 e FLAME 2020:** solo ricetta e semi. `regen.py` rigenera le mesh entro 1.3e-7, non gli operatori.
- **FaMoS: non rilasciabile, nemmeno come ricetta e semi** (riga MPI del file: no per mesh, coefficienti/semi e
  pesi). La bozza diceva "solo ricetta e semi", ed era sbagliato. Anche `sources.LICENSES` scrive ancora "solo
  ricetta" per FaMoS (rango 2): va corretto prima di usarlo.
- **Pesi:** con BFM 2019 (o FLAME 2020, FaMoS) in training i pesi non sono rilasciabili senza un permesso scritto
  (Basilea/Unitectra; MPI). Il modello `validated` quindi non è rilasciabile.

**Domande per l'utente.**
- **(a) FaceScape in training?** Raccomandazione del PI: **no**. Resta dev e zero-shot (§14.5; critic del 10 ottobre).
- **(b) FaMoS TRAIN in training?** Sta nel preset `max`, non in `validated`.
  - Pro: le uniche identità ed espressioni reali.
  - Contro: non rilasciabile nemmeno come ricetta e semi; FaMoS TEST smette di essere zero-shot per dominio (stesso
    sistema di cattura e stessa registrazione FLAME) e resterebbe intatto solo NoW, che viene dallo stesso sistema
    MPI; 80 persone al 20% dei gruppi.
  - Proposta: fuori dal run principale, al massimo in un secondario.
- **(c) Modello a nucleo aperto** (ICT + GNM + FLAME 2023 Open, `STREAM_SOURCES=open_core`) **sulle A100
  secondarie**, con `--requeue` e ripresa provata? È l'unico modello con pesi rilasciabili. Proposta: sì.
- **(d) Run secondari:** secondo seme (il C3M ne ha uno solo) prima del LODO? Proposta: sì, in quest'ordine.
- **(e) FLAME 2023 Open in training?** Aggiunge un quarto 3DMM ed è nel nucleo aperto. Però il concorrente FLAME 2023
  di `baselines_param` diventa "stesso prior", e FaMoS è in topologia FLAME.
- **(f)-(k)** le deviazioni della tabella sopra.
- **(l) BFM 2019 nel run principale?**
  - Pro: i bracci decisivi hanno BFM, quindi `validated` ne è la ricetta scalata; taglie vere.
  - Contro: i pesi non sono rilasciabili.
  - Alternative: (c) come modello rilasciabile accanto al principale; oppure un principale senza BFM, che però non
    è più la ricetta validata.

### 22.8 Calcolo

- **Run principale:** 12 L40S, QoS normal, **6 + 6 su due nodi**. Con un numero di GPU diverso fra i nodi NCCL si
  blocca (misurato con 2 + 1, `massive_ready.md` §6), quindi niente 8 + 4. Così non restano L40S per le valutazioni
  durante il run.
- **A100 di nv-ai-04 (unprivileged):** solo secondari ripartibili e valutazioni brevi.
- **Passo:** 12 rank × 5 identità × ≤ 6 viste = 60 identità e 360 viste per passo (C3M: 30 e 180).
- **Ore: DA MISURARE CON IL JOB 1067683** (prova 6 + 6, 2400 passi, in attesa). s/passo a 12 rank: **[segnaposto]**;
  avvio fino al primo passo: **[segnaposto]**.
- **Stima provvisoria dagli smoke a 2 L40S su un nodo:**
  - smoke 1067645, ricetta di prima: 0.37-0.41 s/passo nelle 10 epoche, GPU al 75%;
  - mini-smoke 1067709, ricetta c3m: 0.435-0.447 s/passo nelle 5 epoche, GPU al 76%. È più lento perché ogni
    identità ha anche la vista up60k (10.342 vertici medi per vista contro 8.357).

  Nessuna delle due è una misura a 12 rank:
  - il passo fra due nodi può essere più lento (con 1 + 1 GPU era uguale: 1.09 contro 1.08 s, §13);
  - i produttori di un nodo servono 6 GPU invece di 2.

  | T (passi) | identità viste (T × 60) | rispetto al C3M | ore, stima provvisoria (0.435-0.447 s/passo, senza avvio) |
  |---|---|---|---|
  | 30.000 | 1.8 M | = C3M | 3.6-3.7 |
  | 60.000 | 3.6 M | 2× | 7.3-7.5 |
  | 120.000 | 7.2 M | 4× | 14.5-14.9 |
  | 240.000 | 14.4 M | 8× | 29.0-29.8 |

  "Identità viste" conta gli usi, riuso compreso. C3M: 60.000 passi × 30 = 1.8 M usi di identità.
- **Domanda (m) per l'utente: T.** Parità col C3M in identità viste = 30k passi; parità in passi = 60k (il doppio
  delle identità). Il C3M non dice che più passi aiutino HIFI3D (22.6).
- **Per riferimento, il C3M** (job 1062944, `factorized/c3m/train.log`, 6 L40S, via tar/store, forward
  sequenziale): 60.000 passi in 205 epoche; 1.036 s/passo in media (somma dei tempi d'epoca 62.163 s), mediana per
  epoca 0.709 s/passo, fino a 3.19 nelle epoche con caricamento di blocco; wall 18 h 39 min, di cui circa 1.4 h di
  avvio e staging.

### 22.9 Concorrenti per il paper

| concorrente | stato | fonte |
|---|---|---|
| ICP + Chamfer (mm, cs), NICP per coppia (mm, cs), NICP su template, taglia stimata e oracolo, composizioni calibrate, Chamfer eval, e108 | fatti, FR e SR, tutti i domini | `baselines_mm/`, R |
| Uni3D-g, OpenShape, ShapeDNA, HKS/WKS | fatti, solo maxabs su HIFI3D: da rivalutare con FR e SR (§16.4) | `competitors_hifi3d/` |
| ArcFace su render | fatto, solo riconoscimento | `arcface_render_zs/` |
| GNM Head e FLAME 2023 Open (fit NICP + proiezione), varifold | in corso (job wbes-bp-paired); GNM è un prior visto | `baselines_param/PROTOCOL.md` |
| MICA su render | da fare; fattibilità MEDIA, asset già su disco | `literature/COMPETITORS_LEARNED_2026-10-10.md` |
| NPHM / MonoNPHM | da fare su un sottoinsieme; licenza dei pesi ignota | idem |
| 3DFacePointCloudNet | da fare; pesi MIT nel repo, rischio di compilazione | idem |
| Point-MAE | da fare se c'è tempo (circa un giorno) | idem |

### 22.10 Rischi aperti

- **Errore di dominio della calibrazione** (R, sez. (iii)): c ideale su HIFI3D 0.503 contro 0.405 degli held-out
  (×1.24), su FaceScape 0.343 (×0.85). k_ICP sbaglia nello stesso modo (×1.25 e ×0.87): lo scarto viene dai domini,
  non dal modello. d_F resta sensibile a c.
- **"Forma oltre la taglia" non soddisfatto:** il vantaggio FR su HIFI3D va presentato come taglia + forma, non come
  sola forma.
- **FaMoS:** 15 soggetti, IC larghi circa 0.6; solo descrittivo.
- **Studio umano v2:** senza conteggio delle risposte. Manca l'arbitro percettivo; resta quello d'identificabilità (E12).
- **Cambio di pipeline:** i bracci decisivi vengono dalla via tar/store con forward sequenziale. Lo stream cambia
  insieme dati, batch e forward a gruppi (gradienti relativi fino a 3.1e-4, `massive_ready.md` §14).
- **Batch globale doppio con lo stesso lr** (60 identità per passo contro 30 del C3M, 22.7): non provato.
- **Un solo seme** nel run principale; **calendario:** numeri finali entro il 24 ottobre (§0).

### 22.11 Cosa serve dall'utente per partire

1. Risposte a (a)-(l) della 22.7. Raccomandazione: `validated` per il run principale, deviazioni (f)-(k) solo in
   run secondari o ablazioni.
2. Il numero di passi T (domanda (m), 22.8), e quindi le ore, con 12 L40S 6 + 6 occupate per tutta la durata.
3. Il s/passo misurato dalla prova 6 + 6 (job 1067683) al posto dei segnaposto della 22.8.
4. Il via alla definizione degli held-out di calibrazione per i domini nuovi (22.5), prima del lancio.
5. Il via al critic su questa configurazione (gate della §21).

### 22.12 Ripresa del run principale e A100 secondarie (11 ottobre)

**Ripresa del run principale: corretta e provata.** Commit `ce0ae82`, evidenze in
`aau/runs/evidence/stream/resume_main/`. Il critic su efffd74 era BLOCCANTE: un requeue faceva ripartire da zero
(l'anello `r<riavvio>` entra nell'hash della run dir) e nessuno salvava su SIGTERM. Ora:
- `STREAM_PREEMPTIBLE=1` di default: su SIGTERM il trainer salva al passo in corso;
- anello `r0` a ogni partenza;
- codice congelato in `STREAM_OUT/code` alla prima partenza;
- ripresa obbligatoria: senza `last.pth`, errore esplicito;
- continuazione dopo un TIMEOUT con un job nuovo: basta lo stesso `STREAM_OUT`;
- `STREAM_CKPT_MIN` resta 20, perché entra nell'hash.

La riga di lancio validata dà ancora `78085a52`; a parzialità spenta ricetta, shard e sha256 sono identici.

Prova a due nodi (2 × 1 L40S, job 1068078 e 1068090):
- `scontrol requeue` a metà dell'epoca 3, `scancel` come un TIMEOUT a metà dell'epoca 5, poi continuazione con un
  job nuovo;
- una sola run dir, ripresa dai passi salvati (255, 457), passi da 1 a 600 senza buchi né doppioni, registro delle
  viste coerente, semi nuovi a ogni segmento;
- con il job sbagliato nell'anello la guardia ferma il job (1068100).

Non provata a 6 + 6. Il job 1067683, la prova 6 + 6, è cancellato perché superato: il s/passo a 12 rank della 22.8
resta da misurare.

**A100 di nv-ai-04 (dato del critic, `critic_scratch_657e6a75`).**
- I produttori sono circa 5 volte più lenti per core (EPYC 7742 a 2.25 GHz; 1.8 contro 8.2 viste fresche/s con 16
  processi, nodo carico).
- Per il secondario a 8 A100 il critic stima circa 6-12 viste fresche/s, con riuso 35-40.
- Il trainer fa 0.70 s per passo, contro 0.41 sulle L40S.

Il secondario su A100 è quindi limitato dai dati. Per confrontarlo serve un controllo appaiato sullo stesso hardware:
stessa configurazione 8 × 5 con `STREAM_PARTIAL=0`.
