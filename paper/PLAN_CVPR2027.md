# Piano CVPR 2027 — da "When Alignment Hurts" al paper risottomesso

Scritto il 10 settembre 2026. Scadenze CVPR 2027 (verificate sul sito): registrazione 10 novembre, paper 16 novembre, supplementary 23 novembre, review 25 gennaio, rebuttal 25 gennaio–1 febbraio, decisione 25 febbraio 2027. Conferenza a Seattle, 20–25 giugno 2027. Nessuna proroga.

Oggi mancano **9 settimane e mezzo**. Il piano ha quindi due tracce: la traccia A è ciò che entra in CVPR; la traccia B è ciò che, se non è pronto entro il 24 ottobre, slitta alla scadenza successiva (ICCV 2027, primavera 2027, data non ancora pubblicata).

---

## 1. Diagnosi: perché è stato rifiutato e cosa cambia

Il meta-review dice tre cose, e nessuna riguarda il modello:

| Obiezione | Reviewer | Cosa serve per chiuderla |
|---|---|---|
| Nessuna validazione su dati reali; FaceScape zero-shot crolla (0.115 vs 0.598) | tutti | Un esperimento su scansioni reali e su metodi di ricostruzione veri |
| Circolarità: la metrica impara D_GT e viene valutata su D_GT | Z1mX | Riferimenti esterni indipendenti da D_GT: un secondo 3DMM, identità reali, giudizio umano |
| Poche baseline: solo Chamfer, ICP, NICP | bhBZ, YJz1 | Varifold, currents, DPDist, ArcFace, LPIPS, CLIP, DINOv2 |
| Solo 3000 volti, un solo 3DMM | YJz1 | Training multi-3DMM (BFM + FLAME) su più identità |
| Solo volti neutri | Z1mX | Almeno una valutazione con espressioni |
| Claim cross-topologia troppo largo, BFM mai nominato, invarianza rigida non dichiarata, bias demografico | Z1mX | Correzioni di testo |

**Il cambio di tesi.** Il paper NeurIPS diceva: "la nostra metrica riproduce D_GT meglio di Chamfer". Questo è circolare per costruzione. Il paper CVPR deve dire: "la valutazione basata su allineamento distorce il ranking delle identità, e lo mostriamo contro **quattro riferimenti indipendenti**: la corrispondenza densa di BFM, quella di FLAME, l'identità reale nelle scansioni, e il giudizio umano. Una metrica appresa senza allineamento è coerente con tutti e quattro; le pipeline con ICP no." La metrica resta il contributo tecnico, ma la prova non dipende più da un solo D_GT.

## 2. Cosa esiste già (verificato nel repo)

| Voce | Stato | Dove |
|---|---|---|
| Ambiente e training/eval su AAU | fatto oggi | `aau/` |
| Dati REMESH e checkpoint v1 | scaricati da Hugging Face, operatori in ricalcolo | `hf_downloads/`, `datasets/REMESH/` |
| FLAME: 5000 identità × 6 topologie, 30.000 mesh | fatto | `v2_work/genflame/`, STATUS.md:812 |
| Training congiunto BFM+FLAME, "support bank" | fatto, risultato parziale | `v2_work/train_v2/` |
| D_GT cross-3DMM via corrispondenza BFM→FLAME | fatto, con gate di validazione | `v2_work/xdomain/` |
| Zero-shot BFM→FLAME | Spearman 0.478, contro 0.115 su FaceScape | STATUS.md:398 |
| Varifold e currents | implementati e testati; currents scartato | `v2_work/phase0/measure_distances.py` |
| Renderer mesh→immagine ed estrattori ArcFace, CLIP, DINOv2 | implementati, non integrati in tabella | `v2_work/phase0/render_mesh.py`, `perceptual_embed.py` |
| M3DFB (FG 2025) come baseline di registrazione | importato, 4/16 stimatori usabili | `v2_work/m3dfb/` |
| DPDist | solo citato | — |
| Dati reali (FRGC, ND-2006, BU-3DFE, 3D-TEC) | richieste scritte, mai firmate né spedite | `literature/data_requests/` |
| Studio umano | nulla | — |
| Espressioni | nulla | — |
| Anomalia aperta: in-domain FLAME 0.478 contro 0.751 su BFM | non spiegata | STATUS.md:1671 |
| Esperimento sospeso: scala degli operatori (maxabs vs Weyl), `pot_area` | risultato mai letto | STATUS.md, ultima voce |

Conclusione: le baseline e il multi-3DMM sono a tre quarti. Dati reali, studio umano ed espressioni partono da zero. Sono anche i tre punti che hanno deciso il reject.

### Aggiornamento del 10 settembre, sera (ricognizione WS0)

- **`pot_area` è concluso e positivo.** Risultati in `v2_work/potential/results/pot_area*.json`, tre seed, commit `a99b1ba`. Operatori ricalcolati su mesh ad area unitaria (convenzione Weyl, verificata: autovalori scalati esattamente per l'area) contro `pot_plain`, tre seed: crop +2.2 / +1.4 / +5.1 punti (media +2.9), all +0.9 / +1.2 / +1.8. Il numero non era mai stato trascritto nel diario. Nota: `pot_plain` era la ricetta v1 a 60 epoche, non 120. **Decisione:** la convenzione ad area unitaria diventa lo standard per ogni nuovo training (WS2, WS5). La riproduzione della Tabella 1 (WS0) usa invece la convenzione originale, per confrontarsi col paper.
- **L'"anomalia FLAME in-domain" non esiste.** Lo 0.478 è lo zero-shot BFM→FLAME (STATUS.md:398); la frase alla riga 1687 che lo chiama in-domain è un refuso del diario. La causa "supporto della mesh" è già stata esclusa con un controllo causale (STATUS.md:623, 685-698): il fattore è la coppia cross-topologia di tipo Poisson. Resta un solo esperimento mai eseguito: ricalcolare il GT sulla sola regione croppata (STATUS.md:720-722). Va in WS2, punto 1, al posto dell'anomalia.
- **Le mesh FLAME 5000 non esistono più da nessuna parte.** Non nel checkout, non nello snapshot Hugging Face del 3 maggio, non nel dataset `Synthetic-Faces`. Vanno rigenerate con la pipeline `v2_work/genflame/build_flame_5000.sh` (5 passi, circa 4 ore di CPU in array), che richiede il file ufficiale FLAME 2020 `generic_model.pkl`. Il file è sotto licenza, va scaricato di persona da flame.is.tue.mpg.de. **Blocco per WS2 e WS5 finché l'autore non lo mette in `v2_work/genflame/official/FLAME2020/`.**

- **FaceScape (cartella `datasets/FaceVerse`) recuperato dallo snapshot con operatori**: 110 soggetti, decimati a 10k vertici, più la variante remesh e la matrice GT. Ma c'è **una sola scansione per soggetto** (suffisso `_01`): le altre pose ed espressioni stavano nei PLY grezzi `FaceVerse/extracted/detail`, che non sono né nel repo né nello snapshot. Senza quelli WS3a e WS5 su FaceScape non partono. La pipeline per ridurli e calcolare gli operatori c'è tutta e costa minuti per posa.
- **WS2 in corso (11 set)**: i tre modelli BFM ad area unitaria sono addestrati; le prime eval dei seed 2345/3456 erano contaminate (split di eval fisso al seed 1234, 79-86 soggetti su 100 nel training) e sono state rifatte; il seed 1234 contro v1 dà un effetto piccolo (rotation +1.0, mixed −0.5). Zero-shot BFM→ICT fatto: latent 0.30 contro Chamfer 0.44 (mesh-pair), negativo; il modello mono-3DMM non generalizza. Espressioni ICT v1 non informative (blendshape globali), rifatte con coefficienti per soggetto; training ICT-only e congiunto in corso (fine 13 set). Testo originale: training della ricetta v1 su REMESH con operatori ad area unitaria (WS2), tre seed (1234, 2345, 3456; job 1019310-1019312). È il candidato modello del paper CVPR. **Misurato su L40S: circa 250 s per epoca, 9 ore per 120 epoche**, contro le 23 ore del cluster precedente. La tabella dei costi in §5 va rivista al ribasso di circa 2.5 volte. Attenzione: il seed cambia anche lo split train/held-out, quindi i CI sui tre seed includono la varianza di split.

### Aggiornamento del 10 settembre, pomeriggio (sostituzioni senza asset dell'autore)

L'autore non può fornire FLAME, i PLY FaceScape né la registrazione NoW. Sostituzioni, tutte verificate e scaricate:
- **FLAME → ICT-FaceKit** (USC, licenza MIT, modello nel repo): 100 modi di identità, 53 blendshape di espressione, regione volto 9409 vertici. Dataset ICT-5000 in costruzione con la pipeline gemella di FLAME, più espressioni per i 500 held-out (WS5). In `external/ICT-FaceKit`, `datasets/ICT/`.
- **FaceScape e NoW → Multiface** (Meta, CC BY-NC 4.0, S3 pubblico senza registrazione): 13 soggetti reali, topologia tracciata comune (7306 vertici), 11 segmenti per soggetto (neutro + 10 espressioni), 20 frame per segmento = 2848 mesh scaricate, più immagini di 3 camere frontali calibrate per neutro + 3 espressioni (in download, ~270 GB). In `datasets/Multiface/`.
- **Metodi di ricostruzione per WS3b**: solo quelli con pesi pubblici e senza asset a registrazione (3DDFA_v2, SynergyNet, PRNet), in predisposizione in `aau/recon/`. DECA e MICA esclusi perché richiedono FLAME.
- Conseguenze per il paper: il riferimento "identità reale" ha 13 soggetti, non 100+; compensato da 220 mesh per soggetto tra frame ed espressioni, e da un protocollo a quattro classi di coppie (stesso soggetto stessa espressione; stesso soggetto espressione diversa; soggetti diversi stessa espressione; tutto diverso). Multiface è in corrispondenza densa per costruzione: per il caso senza corrispondenza si usano le varianti rimagliate e decimate, come per REMESH. Nel paper va detto entrambe le cose.
- Letteratura 2025-2026 in `literature/scouting_2026-09-10.md`: nessuno ha anticipato la tesi; da differenziare da Shilova et al. 2026, TGE e AlignFace; da citare M3DFB e *Beyond Fixed Topologies*.

### Aggiornamento del 10 settembre, sera (primi risultati)

- **WS1: BLOCCANTE anche alla seconda revisione (11 set, mattina), terzo giro in corso.** Varifold/currents normalizzati per area crollano solo su noisy (senza noisy varifold cross 0.568 > Chamfer 0.454); proxy bbox 0.284/0.117 batte CLIP e DINOv2 cross; colonna cross da spezzare in tassellazione/perturbazione. Nota precedente: Varifold/currents sottocampionati, ArcFace con fallback su noisy, render senza normalizzazione per mesh, tabella primaria da spostare su held-out. Numeri sotto da considerare provvisori. Tabella 2 estesa in `aau/runs/baselines_fb100/ranking/table2_extended_facebench_first100.csv` (stesso set di 100 soggetti della Tabella 2 del paper; Chamfer riprodotto: 0.7295 e 0.5518 contro 0.729 e 0.552) e in `aau/runs/baselines/ranking/` (held-out dello split v1). Spearman su fb100, original→original / cross-topology no-crop: Chamfer 0.730 / 0.552; varifold 0.636 / 0.235; currents 0.638 / 0.135; ArcFace 0.380 / 0.217; CLIP 0.451 / 0.126; DINOv2 0.429 / 0.063; **LPIPS 0.816 / 0.258**. Lettura: LPIPS su render è la miglior metrica a parità di topologia, meglio di Chamfer, ma nessuna baseline aggiunta regge il cambio di topologia. Va aggiunta la riga latent (v1 e areanorm) dagli eval. DPDist resta da implementare o da dichiarare escluso.
- **Eval topology (Tabella breakdown) completata** in 2h22 su L40S. Ranking e sigma sweep in corso.
- **ICT-5000**: operatori completi (30.000 npz, 284 GB); manca solo la vista train-ready e le espressioni con operatori.
- **Multiface**: 2848 mesh preparate in tre topologie (tracked, remesh, down); operatori in calcolo, standard e ad area unitaria; protocollo a quattro classi di coppie in `aau/multiface/pairs_protocol.json`.
- Letteratura consolidata in `literature/REVIEW_2026-09-10.md`. Jozwik et al. 2022 (PNAS): la distanza euclidea nello spazio parametrico BFM predice i giudizi umani, argomento esterno a favore di D_GT contro l'accusa di circolarità.

- **WS4 pronto per i partecipanti**: `aau/human_study/` con 300 triplette in disaccordo tra GT, Chamfer, LPIPS e latent v1 (25 tipi di disaccordo, matrice latent in `aau/runs/baselines/matrices/latent_v1/`), 30 controlli, interfaccia a file singolo, analisi con bootstrap sui partecipanti. Resta da fare a mano: aprire `index.html` in un browser una volta, poi distribuire lo zip (11 MB) a 25–30 persone. Opzione da valutare: ospitarla come pagina web con raccolta risposte centralizzata.
- **WS6, prime correzioni** in `paper/main_cvpr_draft.tex` e `paper/refs_cvpr_add.bib`: BFM nominato (2009, da confermare con l'autore), invarianza rigida dichiarata con i parametri di augmentation reali, claim cross-topologia ridimensionato, Limitations e appendice "Identity distinctness" con segnaposto, related work differenziato da Shilova, TGE e AlignFace.
- **WS3a duro: BLOCCANTE alla seconda revisione (11 set): ArcFace ancora con detector, riga proxy assente; unico asse informativo il crop. Terzo giro in corso.** Nota precedente: Il 0.72 di varifold è artefatto di sottocampionamento; un proxy bbox di 4 numeri dà AUC 0.996: test risolto da informazione banale. Rifatto nel protocollo duro con render normalizzati e varifold a piena risoluzione. Testo originale: (`aau/runs/multiface_ws3a/summary.md`): su scansioni pulite in topologie gentili, AUC same/different ≥ 0.98 per Chamfer, ICP, LPIPS, ArcFace e latent v1, anche nel caso difficile (stesso soggetto con espressione diversa contro soggetti diversi a parità di espressione). Solo varifold cala (0.72 su tracked). Il test non discrimina: 13 soggetti e nessun disturbo di supporto. **In corso la versione dura**: varianti crop, noisy e up delle mesh Multiface, coppie a topologie miste, tutte le metriche. Se anche quella satura, il risultato da scrivere è che il problema dell'allineamento emerge solo con cambio di supporto e topologia, non con identità pulite.
- **WS3b: RISERVE alla seconda revisione (11 set): conclusione confermata, limiti da scrivere (area +14.8% per 3DDFA, criterio allineato unidirezionale, ex-ex da landmark, 1°/2° indistinguibili).** Nota precedente: Ritaglio recon con raggio diverso per metodo (90.7–91.6 contro 95 mm), criterio 'NoW' da rinominare. Conclusione qualitativa supportata. Numeri sotto provvisori (`aau/runs/multiface_ws3b/summary.md`): 3075 immagini per metodo, 13 soggetti. Classifica globale: errore stile NoW 3DDFA_V2 1.24 mm ≈ PRNet 1.26 < SynergyNet 1.52 (primo posto indeciso, p=0.62/0.38); Chamfer grezzo stesso ordine; **latent v1 scambia PRNet e SynergyNet**. Kendall tau per soggetto: NoW/Chamfer 0.33, NoW/latent 0.03, Chamfer/latent 0.28; ordine identico per soggetto solo nell'8–31% dei casi. AUC identità sulle ricostruzioni 0.90–0.94 per tutti. Lettura: la classifica dei metodi è stabile solo mediata sui soggetti; per soggetto dipende dal criterio. Caveat da dichiarare: GT ritagliata a 95 mm dal naso, recon decimate a 5215 triangoli, latent su T4, 13 soggetti. Da rifare con i modelli areanorm e congiunto quando pronti.

- **Punto metodologico da dichiarare nel paper (WS6)**: negli scenari perturbati dell'eval (jitter, translation, rotation, mixed) e nelle augmentation del training, gli operatori spettrali NON vengono ricalcolati sulla mesh perturbata: restano quelli della mesh pulita, e solo il ramo xyz vede la perturbazione (`compare_model_vs_chamfer_rankings.py:463-471`, `train_runner.py:831-849`). Quindi gli scenari perturbati misurano la robustezza del solo canale xyz; il test onesto sotto rumore è la topologia `noisy` di REMESH, che ha operatori propri. Un reviewer attento lo chiederebbe: va scritto esplicitamente, e la sigma sweep va presentata come "perturbazione del solo input geometrico". In più `GTReadyDatasetNPZ` rinormalizza sempre i vertici per max|coord| a runtime, per cui la convenzione ad area unitaria agisce solo sugli operatori: è coerente con l'esperimento `pot_area` e va descritta così.
- **Eval dei modelli ad area unitaria** lanciate per i seed 2345 e 3456 (job 1019597-1019600, due per seed); il seed 1234 finisce più tardi.

### Aggiornamento del 4 ottobre (ripresa dopo tre settimane di fermo)

- Tre settimane perse: la sessione si era fermata l'11 settembre. Restano 6 settimane; il gate del 24 ottobre è confermato.
- WS2: il guadagno del modello ad area unitaria su crop non è robusto sui tre seed (+0.032 ± 0.031 sul margine latent−Chamfer). Resta come scelta di implementazione, non come contributo.
- In corso: tabella cross-3DMM con i modelli ICT-only e congiunto (addestrati l'11-12 settembre, mai valutati); chiusura del terzo giro di correzioni su WS1 e WS3a.
- Studio umano: zero risposte; serve la distribuzione da parte dell'autore.
- WS3b ribaltato (4 ott): a parità di supporto e scala, Chamfer e criterio ICP concordano (tau 0.80); la metrica appresa discorda da entrambi (tau 0.13–0.18). Sui metodi di ricostruzione reali l'allineamento non cambia la classifica. Da scrivere come risultato negativo.
- WS1 (4 ott): Chamfer resta la miglior baseline cross-topologia; varifold a massa unitaria in calcolo per renderlo equo.

## 3. Sei cantieri

Ogni cantiere ha un prodotto finale che è una tabella o una figura del paper, un criterio di successo misurabile, e un responsabile tra gli agenti. Io coordino; `coder` implementa; `verifier` esegue; `critic` rivede ogni tabella prima che entri nel paper.

### WS0 — Base riproducibile (settimana 1, 10–17 settembre)

- Riprodurre la Tabella 1 del paper con il checkpoint v1 su AAU. Criterio: Spearman entro l'intervallo bootstrap del paper su tutte le 36 celle.
- Chiudere l'esperimento sospeso `pot_area` e l'anomalia FLAME in-domain. Se la causa è la scala degli operatori, si decide una sola convenzione (√area = 1, come in letteratura) e si ricalcolano gli operatori per BFM e FLAME una volta sola. Tutto il resto del piano usa quella convenzione.
- Registro degli esperimenti: una riga per ogni run in `checking_assumptions/experiment_registry.csv`, come già impostato.

### WS1 — Baseline estese (settimane 1–3)

Prodotto: **Tabella 2 estesa**, sulla stessa protocollo REMESH held-out con CI bootstrap.

| Baseline | Stato | Lavoro |
|---|---|---|
| Chamfer, Rigid ICP, NICP P2P, NICP P2Tri | nel paper | rieseguire |
| Varifold | implementato | integrare nell'eval ranking |
| Currents | implementato, non ranka | riportare comunque, una riga |
| DPDist | assente | implementare dal repo ufficiale, oppure sostituire con una distanza point-cloud appresa disponibile e dichiararlo |
| ArcFace su render frontale | implementato | integrare; 3 viste, media |
| LPIPS, CLIP, DINOv2 su render | implementati | integrare |
| M3DFB, 4 stimatori | importato | riportare come pipeline di registrazione alternativa |

Criterio: ogni baseline ha la stessa protocollo e lo stesso CI. Costo: CPU per varifold, una L40S per le percettive, meno di due giorni di calcolo.

### WS2 — Multi-3DMM e D_GT non circolare (settimane 1–4)

Prodotto: **Tabella cross-3DMM**: modello addestrato su BFM, su FLAME, su entrambi; valutato su BFM, FLAME, e cross con D_GT costruito su ciascun 3DMM.

1. Risolvere l'anomalia FLAME in-domain prima di addestrare altro (WS0).
2. Training congiunto BFM 400 + FLAME 4500 identità, ricetta v1, una L40S, 72 ore.
3. Valutazione con D_GT di FLAME e con D_GT BFM→FLAME via corrispondenza (già in `xdomain`). Questo è il primo riferimento esterno: una metrica addestrata su BFM che conserva il ranking definito da FLAME non sta più "ricopiando" il proprio D_GT.
4. Controllo richiesto da Z1mX: distanza minima tra i parametri delle identità di training e held-out, riportata in appendice.

Criterio: cross-3DMM zero-shot sopra 0.6 di Spearman, e congiunto sopra Chamfer su tutte le coppie. Se il congiunto non supera Chamfer nel cross, si riporta lo stesso: è un risultato onesto, e la tesi regge sulle pipeline di allineamento.

### WS3 — Validazione su dati reali (settimane 1–6, percorso critico)

Prodotto: **Tabella "real scans"** e **Figura "ranking dei metodi di ricostruzione"**.

Due esperimenti, in ordine di fattibilità:

**3a. Identità reale come riferimento esterno, senza corrispondenza.** Serve un dataset con più scansioni dello stesso soggetto. Candidati: FaceScape (già usato nel paper, quindi l'accesso c'è: 20 espressioni per soggetto, e la neutra ha più acquisizioni), NoW validation (20 soggetti con scansione ground truth, registrazione gratuita con licenza firmata dal supervisore). Il test è: per ogni metrica, la scansione dello stesso soggetto deve risultare più vicina di quella di un altro soggetto. Si misura con AUC same/different e con il ranking dei soggetti. Non serve alcun D_GT: la circolarità è rotta per costruzione.

**3b. Confronto tra metodi di ricostruzione reali.** Su NoW validation: eseguire 4 metodi pubblici a codice aperto (DECA, MICA, Deep3DFace, 3DDFA_v2, tutti con pesi rilasciati) sulle immagini, ottenere le mesh in topologie diverse, e confrontare il ranking dei metodi ottenuto con: il protocollo ufficiale NoW (con allineamento), Chamfer, ICP, la nostra metrica. La domanda del paper diventa: "la classifica dei metodi cambia a seconda che si allinei o no?" Se cambia, è la prova sul campo del claim. Lo scaffold `faceBench/run_pipeline_batch` esiste già.

Azioni immediate da parte dell'autore: registrarsi a NoW e far firmare la licenza; spedire le richieste già scritte per ND-2006 e BU-3DFE, che servono per la traccia B.

Criterio go/no-go al 24 ottobre: 3a completato su almeno un dataset, 3b su almeno tre metodi. Altrimenti il paper CVPR esce senza 3b e 3b va in traccia B.

### WS4 — Studio umano (settimane 3–7)

Prodotto: **Tabella "accordo con il giudizio umano"**.

Disegno: triplette "quale tra B e C somiglia di più ad A", su render frontali di volti sintetici REMESH e FLAME (nessuna persona reale, quindi nessun problema di consenso). 300 triplette scelte dove le metriche sono in disaccordo, per massimizzare l'informazione. 25–30 partecipanti, 10 minuti a testa, interfaccia web semplice. Si misura l'accordo di ogni metrica con la maggioranza umana. Il riferimento è il giudizio percettivo, non D_GT: è la risposta diretta a Z1mX.

Da fare subito: verificare con il dipartimento se serve un parere etico per uno studio percettivo su volti sintetici, e reclutare partecipanti dal laboratorio.

Criterio: la metrica appresa e D_GT concordano con gli umani almeno quanto Chamfer, e le pipeline ICP di meno. Se D_GT non concorda con gli umani, va scritto: è il risultato più interessante del paper.

### WS5 — Espressioni (settimane 2–5)

Prodotto: **Tabella "robustezza alle espressioni"**, in appendice se manca spazio.

FLAME ha parametri di espressione: generare per 500 identità held-out 5 espressioni ciascuna, come settima "perturbazione". Domanda: il ranking delle identità sopravvive al cambio di espressione? Prima solo valutazione dei modelli esistenti; poi, se il tempo c'è, training con augmentation di espressione.

Criterio: una tabella con Spearman per intensità di espressione, per ogni metrica.

### WS6 — Scrittura e posizionamento (settimane 5–9)

- Riscrittura di abstract e introduzione sulla nuova tesi a quattro riferimenti.
- Correzioni puntuali: il 3DMM si chiama BFM nel testo principale; l'invarianza rigida è ottenuta per augmentation e va detto; il claim cross-topologia è ora dimostrato cross-3DMM e viene formulato esattamente come misurato; paragrafo su bias demografico ereditato dal 3DMM; controllo di distinzione delle identità in appendice.
- Formato CVPR: 8 pagine più riferimenti. Il materiale NeurIPS era più lungo: tagliare la Tabella 3 dei tempi a una riga nel testo, spostare la compressione delle distanze in appendice.
- Rilascio: dataset REMESH no-ops e checkpoint sono già su Hugging Face; aggiungere FLAME 5000, operatori e le matrici GT per entrambi i 3DMM, e lo script dello studio umano.
- Il paper cita e discute i lavori usciti dopo il 15 settembre 2026 anche se CVPR non li considera per il confronto.

## 4. Calendario

| Settimana | Date | Milestone | Go/no-go |
|---|---|---|---|
| 1 | 10–17 set | Tabella 1 riprodotta; convenzione operatori decisa; registrazione NoW e firma licenza; richieste dati spedite; etica studio umano avviata | Tabella 1 combacia |
| 2 | 17–24 set | Baseline varifold e percettive integrate; training congiunto BFM+FLAME lanciato; espressioni FLAME generate | — |
| 3 | 24 set–1 ott | Tabella 2 estesa completa con CI; metodi di ricostruzione eseguiti su NoW validation; interfaccia studio umano pronta | — |
| 4 | 1–8 ott | Tabella cross-3DMM; esperimento 3a su FaceScape; studio umano in raccolta | — |
| 5 | 8–15 ott | Esperimento 3b; tabella espressioni; prima stesura sezioni 3–5 | — |
| 6 | 15–22 ott | Studio umano chiuso e analizzato; tutte le tabelle passate da `critic` | **24 ott: decisione CVPR o traccia B** |
| 7 | 22–29 ott | Stesura completa; figure; supplementary | — |
| 8 | 29 ott–5 nov | Lettura avversariale interna sulla base delle tre review; revisione | — |
| 9 | 5–12 nov | Correzioni finali; registrazione entro il 10 | — |
| 10 | 12–16 nov | Sottomissione | 16 nov |

## 5. Calcolo su AAU

Tutto su L40S, entro il tetto di 12 GPU per utente, senza QoS unprivileged. Stime dai log dei run precedenti, non ancora misurate su L40S.

| Lavoro | GPU·ore stimate | Note |
|---|---|---|
| Riproduzione eval v1 | 30 | 3 stage, CPU-bound in parte |
| Operatori spettrali BFM + FLAME con nuova convenzione | CPU, 1 giorno | 33.000 mesh |
| Training congiunto BFM+FLAME, 2 seed | 150 | 72 h ciascuno |
| Training con espressioni, 1 seed | 75 | opzionale |
| Baseline percettive, 4 estrattori | 20 | |
| Eval cross-3DMM e espressioni | 60 | |
| Metodi di ricostruzione su NoW, 4 metodi | 10 | |

Totale attorno a 350 GPU·ore: quattro L40S per due settimane. Fattibile senza mai toccare il tetto.

## 6. Rischi

| Rischio | Probabilità | Mitigazione |
|---|---|---|
| Licenza NoW o dati reali non arrivano in tempo | media | 3a su FaceScape, che c'è già; 3b in traccia B |
| Anomalia FLAME in-domain non si risolve | media | Riportare FLAME solo come riferimento esterno per la valutazione, non come dominio di training |
| Studio umano senza parere etico in tempo | bassa, volti sintetici | Studio interno al laboratorio con consenso scritto |
| D_GT non concorda con gli umani | possibile | È un risultato: il paper lo riporta e sposta il peso sui riferimenti reali |
| Il congiunto BFM+FLAME non batte Chamfer cross-3DMM | possibile | Tesi principale resta sulle pipeline di allineamento; la metrica diventa "coerente", non "migliore ovunque" |
| Un'esperienza di 9 settimane con 3 cantieri da zero | alta | Il go/no-go del 24 ottobre è vincolante: si sottomette con ciò che è verificato, o si punta a ICCV |

## 7. Cosa serve dall'autore, questa settimana

0. **Scaricare FLAME 2020 (`generic_model.pkl`) da flame.is.tue.mpg.de** e copiarlo in `v2_work/genflame/official/FLAME2020/`. Senza questo file WS2 e WS5 non partono.
1. Registrazione a NoW e firma della licenza da parte del supervisore.
2. Spedire le tre richieste dati già scritte in `literature/data_requests/`.
3. **Copiare sul cluster i PLY grezzi FaceScape (`FaceVerse/extracted/detail`, tutte le pose/espressioni, non solo `_01`)** in `datasets/FaceVerse/extracted/detail/`. Servono per WS3a e WS5.
4. Chiedere al dipartimento se lo studio percettivo su volti sintetici richiede un parere etico.
5. Decidere se DPDist va implementato o sostituito.

## 8. Come lavoreremo

Per ogni tabella: `coder` implementa e verifica; `critic` prova a romperla senza aver scritto il codice; `verifier` la riesegue da zero; io la accetto o la rimando. Nessun numero entra nel paper senza essere passato da tutti e tre. Ogni run ha una riga nel registro esperimenti con script, output e seme. Ogni settimana un aggiornamento di questo file con lo stato reale, non quello sperato.
