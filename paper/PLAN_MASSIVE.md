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
