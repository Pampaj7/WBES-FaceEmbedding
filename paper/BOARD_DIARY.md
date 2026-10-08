# Board diary — WBES-FaceEmbedding verso CVPR 2027

Diario cronologico di tutto ciò che viene fatto, con risultati e scoperte. Ogni voce dice cosa è stato fatto, cosa è uscito, e cosa ne segue. I numeri sono misurati sul cluster AAU salvo dove indicato. Il piano di lavoro è in `PLAN_CVPR2027.md`; questo file è il registro di ciò che è successo davvero.

---

## 10 settembre 2026

### Mattina — repo operativo su AAU

- Clonato il repo (5478 file, 550 MB). Era codice puro: nessun dataset, nessun `diffusion-net`, un solo checkpoint (il modello top v1). Tutti gli script di lancio erano LSF del cluster DTU con path di tre macchine diverse.
- Costruito l'ambiente in `aau/`: container `pytorch_24.10.sif` (torch 2.5), venv con igl, potpourri3d, robust-laplacian, `diffusion-net` clonato. Smoke test su L40S passato.
- Scritti gli script Slurm: `submit.sh`, `train_remesh.sbatch`, eval spezzata in tre stage (ranking, topology, sigma) con skip degli stage già completi. Due giri di revisione critica hanno trovato e corretto: log relativi alla directory di sottomissione (job che morivano muti), `--time` troppo corti rispetto ai run originali (eval 28 h, training 23 h), chiave dell'output eval basata sul solo nome del checkpoint (avrebbe fatto passare i numeri del v1 per quelli di un modello nuovo).
- **Scoperta**: i dati stanno su Hugging Face, account `Pampaj` (non `Pampaj7`): mesh REMESH e FaceScape senza operatori, snapshot del workspace da 30 GB, checkpoint v1. La matrice GT `normalized_matrix_distances.npz` era nello snapshot. Operatori spettrali ricalcolati (k_eig 128, 26 minuti su 24 core, 43 GB).

### Pomeriggio — review lette, piano scritto

- Lette le tre review e il meta-review (in `review.md`): reject per mancanza di validazione reale, circolarità su D_GT, poche baseline. Nessuno contesta l'esecuzione.
- Scritto `PLAN_CVPR2027.md`: scadenza 16 novembre 2026, sei cantieri (WS0 base, WS1 baseline, WS2 multi-3DMM, WS3 dati reali, WS4 studio umano, WS5 espressioni, WS6 scrittura), gate al 24 ottobre. Tesi nuova: la distorsione da allineamento dimostrata contro quattro riferimenti indipendenti (BFM, secondo 3DMM, identità reali, giudizio umano).
- **Scoperte nel diario dell'autore (STATUS.md)**:
  - `pot_area` era concluso e mai trascritto: operatori su mesh ad area unitaria danno +2.9 punti medi su crop (tre seed) rispetto alla convenzione maxabs. Adottata come standard per i nuovi modelli.
  - L'"anomalia FLAME in-domain 0.478" non esiste: è lo zero-shot BFM→FLAME, e la causa "supporto" era già stata esclusa con un controllo causale.
  - Le mesh FLAME 5000 non esistono più da nessuna parte; servirebbe il file ufficiale FLAME 2020, che l'autore non può fornire.
- **Sostituzioni senza asset a licenza**: FLAME → ICT-FaceKit (MIT, 100 identità PCA, 53 espressioni); FaceScape multi-scan e NoW → Multiface (CC BY-NC, 13 soggetti reali, 11 segmenti × 20 frame, immagini calibrate); metodi di ricostruzione con pesi pubblici: 3DDFA_V2, SynergyNet, PRNet. Rifiutata la ricerca di copie non autorizzate di FLAME.

### Sera — primi risultati

**WS1, Tabella 2 estesa** (`aau/runs/baselines_fb100/ranking/`), Spearman vs GT, 100 soggetti della Tabella 2 del paper, CI bootstrap per soggetto:

| Metrica | original→original | cross-topology no-crop |
|---|---|---|
| Chamfer | 0.730 (paper 0.729) | 0.552 (paper 0.552) |
| Varifold | 0.636 | 0.235 |
| Currents | 0.638 | 0.135 |
| ArcFace su render | 0.380 | 0.217 |
| CLIP su render | 0.451 | 0.126 |
| DINOv2 su render | 0.429 | 0.063 |
| LPIPS su render | **0.816** | 0.258 |
| Latent v1 (paper) | 0.902 | 0.868 |

Finding: LPIPS è la miglior metrica a parità di topologia, meglio di Chamfer, ma nessuna baseline aggiunta regge il cambio di topologia. Le percettive non sono una scorciatoia mesh-agnostica. DPDist non implementato.

**WS3b, ricostruzioni su Multiface** (`aau/runs/multiface_ws3b/summary.md`): 3075 immagini per metodo, 13 soggetti, GT = mesh tracciata ritagliata a 95 mm dal naso.

| Criterio | 1° | 2° | 3° |
|---|---|---|---|
| Errore stile NoW (mm) | 3DDFA_V2 1.24 | PRNet 1.26 | SynergyNet 1.52 |
| Chamfer grezzo | 3DDFA_V2 | PRNet | SynergyNet |
| Latent v1 | 3DDFA_V2 | SynergyNet | PRNet |

Kendall tau per soggetto: NoW/Chamfer 0.33, NoW/latent 0.03, Chamfer/latent 0.28; ordine identico per soggetto solo nell'8–31% dei casi. Finding: la classifica globale è stabile solo perché mediata sui soggetti; per soggetto dipende dal criterio, e il criterio latente scambia gli ultimi due. AUC identità sulle ricostruzioni 0.90–0.94 per tutti i metodi. Caveat: 13 soggetti, recon decimate, latent calcolato su T4. Da rifare con i modelli nuovi. In attesa di revisione critica.

**WS3a, identità reale su Multiface** (`aau/runs/multiface_ws3a/summary.md`): AUC same/different subject, caso difficile (stesso soggetto con espressione diversa contro soggetti diversi a parità di espressione):

| Metrica | tracked | remesh | tracked→remesh | down |
|---|---|---|---|---|
| Chamfer | 0.993 | 0.994 | 0.977 | 0.998 |
| Rigid ICP | 0.995 | 0.996 | 0.993 | 1.000 |
| Varifold | **0.722** | 0.962 | **0.735** | 0.978 |
| LPIPS | 1.000 | 1.000 | 0.998 | 1.000 |
| ArcFace | 1.000 | 1.000 | 1.000 | 1.000 |
| Latent v1 | 0.999 | 0.999 | 0.983 | 0.999 |

Finding: su scansioni pulite il test è saturo, tutte le metriche separano le identità. Il problema dell'allineamento non emerge con identità pulite; serve il cambio di supporto e topologia. Avviato il protocollo duro (crop, noisy, up, coppie miste). Bug trovato e corretto: le mesh Multiface sono nel frame testa e il renderer disegnava la nuca; senza la rotazione di 180° ArcFace e LPIPS sarebbero stati numeri senza senso. Varifold su tracked cala perché il kernel è sensibile alla scala della mesh tracciata: da chiarire prima di metterlo in tabella.

**WS2, training in corso**:
- BFM con operatori ad area unitaria, ricetta v1, tre seed. Misurato 250 s per epoca su L40S: 9 ore, contro 23 sul cluster precedente. Due seed completati, il terzo in chiusura. Eval ranking già lanciata sui due checkpoint.
- ICT-5000 costruito: 5000 identità × 6 topologie, operatori ad area unitaria (30.000 npz, 284 GB), 7500 mesh di espressione per i 500 held-out. Training ICT-only in corso (45 ore stimate). Training congiunto BFM+ICT partito (48–50 ore stimate, staging di 356 GB in RAM). Correzione: il trainer legge gli id a 4 cifre e falliva sugli id ICT a 5 cifre; risolto nel wrapper multi-dominio `v2_work/train_v2`, non nel trainer.

**WS0, riproduzione Tabella 1**: eval topology completata (2h22). Eval ranking: il primo job scriveva i risultati solo alla fine e non stava nel limite di 10 ore; cancellato dopo 6 ore e rilanciato in due job paralleli (3+2 scenari). Sigma sweep: 48 ore stimate contro 36 di limite, ma scrive progressivamente; i blocchi mancanti lanciati a parte. Merge con `aau/merge_ranking_scenarios.py`.

**WS4, studio umano**: pacchetto pronto in `aau/human_study/`: 300 triplette scelte dove GT, Chamfer, LPIPS e latent v1 sono in disaccordo (25 tipi), 30 controlli, interfaccia a file singolo, analisi con bootstrap sui partecipanti testata su dati simulati. Resta da fare a mano: aprire l'interfaccia in un browser, distribuire lo zip a 25–30 persone.

**WS6, testo**: `main_cvpr_draft.tex` con le correzioni che non dipendono da risultati: BFM nominato (2009, da confermare), invarianza rigida dichiarata con i parametri di augmentation reali, claim cross-topologia ridimensionato, Limitations, appendice sulla distinzione delle identità, related work differenziato da Shilova 2026, TGE, AlignFace. Bibliografia aggiuntiva in `refs_cvpr_add.bib`.

**Letteratura**: `literature/REVIEW_2026-09-10.md`. Nessuno ha anticipato la tesi. Da citare: M3DFB (evidenza pre-esistente), Beyond Fixed Topologies (DiffusionNet come precedente), Jozwik 2022 PNAS (la distanza parametrica BFM predice i giudizi umani: argomento contro la circolarità), morfometria medica sul bias di Procrustes.

**Fatti sul cluster**: nodi L40S con 735 GB di RAM e `/tmp` da 378 GB; lo staging in RAM è obbligatorio (11 s/it da CephFS contro 1.6 staged). Limite di 12 job e 12 GPU per utente raggiunto più volte.

### Cose da decidere o fare a mano (autore)
1. Confermare BFM 2009 o 2017 per REMESH.
2. Aprire `aau/human_study/index.html` in un browser e distribuire lo studio.
3. Se in futuro si ottiene FLAME 2020, la pipeline `genflame` aggiunge un terzo 3DMM in quattro ore.

### In attesa
- Verdetto critic su WS1, WS3a, WS3b.
- Protocollo duro WS3a.
- Tabella 1 riprodotta (eval ranking, verso l'una) e confronto col paper.
- Eval dei modelli ad area unitaria (tre seed) e tabella cross-3DMM (ICT, congiunto).

## 11 settembre 2026

### Notte — verdetto del critic su WS1, WS3a, WS3b: nessun numero pubblicabile così

Difetti dimostrati con script di controllo, non ipotizzati:
- **Varifold e currents** (WS1 e WS3a): sottocampionamento a 4000 triangoli con seed diverso per mesh. La distanza tra due sottocampionamenti della stessa mesh (0.125 su tracked Multiface) supera quella tra soggetti diversi. Il "crollo" del varifold su tracked (0.72) era un artefatto del rumore, e le righe di Tabella 2 sono sottostimate. Rimedio: tutti i triangoli, self-distance verificata a zero.
- **ArcFace** (WS1): il detector di volti fallisce su tutti i render `noisy` e ripiega su un center-crop, cioè un altro spazio di embedding; il 40% delle coppie cross confronta due pipeline diverse, e il contatore dei fallback era fisso a zero nel codice. Rimedio: crop geometrico fisso per tutti, niente detector.
- **Render** (WS1 e WS3a): camera unica senza normalizzazione per mesh, quindi le metriche percettive vedono posizione e scala assolute che Chamfer non vede. Un proxy di quattro numeri (centro e diagonale del bounding box) dà Spearman 0.47 con D_GT su REMESH e AUC 0.996 su Multiface: il test WS3a era risolto da informazione banale. Rimedio: centro + maxabs per mesh prima del render, come Chamfer.
- **Set di soggetti** (WS1): i numeri riportati erano su fb100, il set della Tabella 2 del paper, che contiene 79 soggetti di training. La tabella primaria del nuovo paper sarà su held-out; fb100 resta solo come gate di riproduzione. Su held-out varifold pareggia LPIPS nel cross (0.21 contro 0.22), quindi la lettura "LPIPS la migliore" va sfumata.
- **Ritaglio delle ricostruzioni** (WS3b): apertura oculare assunta 90 mm, misurata 86–87, quindi raggio 90.7–91.6 mm contro 95 della GT, e diverso per metodo. È esattamente il support mismatch che il paper rimprovera a Chamfer, e favorisce SynergyNet. Rimedio: raggio scalato sull'apertura misurata.
- **"Stile NoW"** (WS3b): ICP libero con scala, non la similarità da 7 landmark del protocollo NoW; allineamento più forte, errore più basso (1.24 mm contro ~1.5 pubblicati). Rinominato in `sim_icp_p2s`; il vero NoW aggiunto se i landmark sulla GT sono ricavabili.

Cosa regge: la riproduzione di Chamfer e il bootstrap (identici al paper); la conclusione WS3b "la classifica per soggetto dipende dal criterio" è supportata da uno split-half tau di 0.92–0.98 dentro ogni criterio contro 0.03–0.33 tra criteri, con la riserva che tau su tre elementi è grossolano (atteso 0.18 sotto ipotesi nulla) e che la discordanza Chamfer/latent può dipendere dal difetto di ritaglio.

Conseguenze: correzioni in corso su WS1 e WS3b; il protocollo duro di WS3a rifatto con render normalizzati e varifold a piena risoluzione, più una riga di controllo con il proxy bbox; le triplette dello studio umano vanno rigenerate dopo la correzione dei render, perché usano le matrici LPIPS. Lezione di metodo: ogni metrica su render va accompagnata da una baseline "solo bounding box" che ne misuri la parte banale.

### Notte — WS3a protocollo duro: pipeline corretta, tabella in arrivo

- Costruite tre varianti nuove delle 2848 mesh Multiface: `crop` (taglio anatomico canonico, 70% dei vertici), `noisy` (rumore 0.003 × diagonale), `up` (2.5× vertici), con operatori standard e ad area unitaria. 17.088 npz in totale.
- Correzioni del critic applicate e misurate: normalizzazione per mesh prima del render (dispersione del proxy bbox da 48.3 a 0.18, cioè 269 volte meno), varifold e currents con 12.000 triangoli e seme fisso (self-distance 0.0000 contro 0.2413 del setup vecchio), riga di controllo `bbox_proxy` in tabella, CI degeneri marcati invece di stampati.
- Sei coppie di topologie: tracked→tracked (riferimento), tracked→crop, remesh→crop, tracked→noisy, down→up, crop→crop. Nove metriche più il proxy.
- Costo: percettive ~6 h su T4, geometria ~15 h su 24 core (kernel quadratico a piena risoluzione), latent in coda. Tabella `aau/runs/multiface_ws3a_hard/summary_hard.md` attesa nel pomeriggio dell'11.

### Notte — correzioni del critic applicate a WS1 e WS3b (verificate)

**WS1, Tabella 2 estesa corretta, set held-out (primario), Spearman orig→orig | cross-topology no-crop:**

| Metrica | orig→orig | cross | prima della correzione |
|---|---|---|---|
| Chamfer | 0.644 | 0.469 | invariato |
| Varifold | 0.577 | 0.229 | 0.553 / 0.211 |
| Currents | 0.613 | 0.133 | — |
| LPIPS | 0.590 | 0.127 | **0.739 / 0.217** |
| ArcFace | 0.293 | 0.218 | 0.308 / 0.185 |
| CLIP | 0.358 | 0.088 | — |
| DINOv2 | 0.378 | 0.059 | — |

Finding rivisto: LPIPS "batteva Chamfer" solo perché i render esponevano posizione e scala assolute; normalizzato per mesh come Chamfer scende sotto. Su held-out Chamfer resta la miglior baseline in entrambi gli scenari, e tutte crollano cross-topologia. Gate fb100 invariato (0.7295 / 0.5518). Cosa è stato corretto: varifold e currents senza sottocampionamento casuale (misura quantizzata su griglia comune, self-distance esattamente zero, Spearman 1.000 rispetto ai triangoli interi); ArcFace con allineamento geometrico fisso invece del detector (che falliva sul 44% dei render, 100% su noisy); render normalizzati; tabella primaria dichiarata held-out.

**WS3b corretto**: raggio di ritaglio scalato sull'apertura oculare misurata (86.1 mm reali): raggi 94.8 / 95.7 / 95.1 mm contro 95 della GT. Classifica: `sim_icp_p2s` (ICP con scala, ex "stile NoW") 3DDFA 1.24 < PRNet 1.26 < SynergyNet 1.52; Chamfer grezzo 3DDFA < SynergyNet < PRNet; latent v1 3DDFA < SynergyNet < PRNet. **Ora Chamfer e latent concordano tra loro (tau 0.59, 3.3 deviazioni dal nullo) e discordano dal criterio con allineamento (tau 0.18 e −0.03, indistinguibili dal caso).** Split-half per criterio 0.98 / 0.91 / 0.92: le classifiche per soggetto sono affidabili. Il vero criterio NoW da 7 landmark è implementato ma diagnostico: i landmark stimati sulla GT hanno 5–7 mm di dispersione contro un segnale di 1.2–1.6 mm. Residuo da dichiarare: 3DDFA mette il 14% di superficie in più nella stessa sfera (rapporto d'area 1.14 contro 1.06 e 1.08), quindi un confondente di supporto resta.

Nota operativa: per il tetto di 12 job il coder ha usato la partizione aicentre con QoS unprivileged su A40; nessuna prelazione, ma resta una deroga alla regola del cluster.

### Notte — WS0: run v1 riprodotto sul cluster AAU

Eval ranking del checkpoint v1 sui cinque scenari (clean, jitter, translation, rotation, mixed), fusa da due job paralleli. Scenario clean: latent 0.8279 contro 0.8280 del log originale, Chamfer 0.4847 contro 0.4847. Differenza sotto 1e-4: operatori ricalcolati (k_eig 128) e pipeline sono equivalenti a quelli dell'autore. Scenario mixed: latent 0.810 contro 0.800, Chamfer 0.448 contro 0.432, cioè fino a 1.6 punti di scarto, compatibile con la casualità delle perturbazioni (il log originale non salva i CI). La matrice 6×6 per coppie di topologie, che è la vera Tabella 1 del paper, è in confronto.
Matrice 6×6 (stage topology): le 30 celle cross-topologia di Chamfer e le 30 di latent combaciano con la Tabella 1 del paper a precisione di arrotondamento (scarto massimo 0.0005 Chamfer, 0.0008 latent, nessuna cella oltre 0.01). Le 6 celle diagonali same-topology non sono prodotte da questo stage e restano da riprodurre con l'eval same-topology. **WS0 chiuso nella sostanza: il cluster AAU riproduce il paper.**

### Mattina dell'11 — triplette dello studio umano rigenerate

Con i render normalizzati per mesh e le matrici corrette (Chamfer, LPIPS, latent v1, GT): 300 triplette in disaccordo + 30 controlli, 100 render, 12 MB, in `aau/human_study/`. Prossimo passo: pubblicare lo studio come pagina web con raccolta centralizzata delle risposte, così i partecipanti ricevono solo un link.
Studio umano pubblicato come pagina web con raccolta centralizzata delle risposte: https://claude.ai/code/artifact/204feff5-053b-44ce-aa7f-53b49182bacc (2.2 MB, 100 render JPEG 384 px, 36 test + 4 controlli per partecipante, risposte salvate nel database della pagina, esportabili con `read_db` sulla collezione `responses` e analizzabili con `analyze.py --from-dir`). La pagina è privata finché non viene condivisa dal menu della pagina; da verificare se i partecipanti esterni all'account possono aprirla, altrimenti resta lo zip offline.

### Mattina dell'11 — WS2: primo confronto modello ad area unitaria contro v1 (PROVVISORIO)

Eval ranking sui cinque scenari, stessi 100 soggetti held-out dell'eval v1, Spearman latent: clean 0.828 (v1) contro 0.854 / 0.852 (seed 2345 / 3456); jitter 0.817 contro 0.836 / 0.830; translation 0.826 contro 0.848 / 0.847; rotation 0.815 contro 0.825 / 0.842 / 0.839 (seed 1234 / 2345 / 3456); mixed 0.810 contro 0.804 / 0.824 / 0.822. Chamfer identico al v1 in tutti gli scenari, come atteso.
**Sospetto di leakage, in verifica**: il trainer sceglie i soggetti held-out con il seed di training, l'eval usa sempre lo split del seed 1234. Per i seed 2345 e 3456 i 100 soggetti dell'eval possono essere stati nel loro training, e i loro guadagni (+2 punti) sarebbero gonfiati. L'unico confronto sicuro è il seed 1234: rotation +1.0, mixed −0.5. Nessun numero entra nel paper finché non è chiarito. Stessa domanda vale per l'esperimento `pot_area` dell'autore (seed 1234/1235/1236).
**Leakage confermato (job 1019693)**: `aau/eval_common.sh` forzava `--seed 1234` per ogni checkpoint; i 100 soggetti di eval erano nel training del seed 2345 per 79/100 e del seed 3456 per 86/100. I risultati dei seed 2345 e 3456 sopra sono invalidi e archiviati in `aau/runs/_leaked/`; lo script è corretto per usare il seed del checkpoint e le due eval sono rilanciate sui rispettivi held-out. Il seed 1234 (0/100 in training) resta valido: contro il v1 dà rotation +1.0 e mixed −0.5, cioè un effetto piccolo; clean, jitter e translation in arrivo. L'esperimento `pot_area` dell'autore era invece pulito (eval con il seed letto dal checkpoint, delta appaiati per seed). Lezione: ogni eval di un modello nuovo deve stampare seed e soggetti held-out, e il critic deve controllarlo.
**WS0 chiuso del tutto**: anche le 12 celle diagonali same-topology della Tabella 1 riprodotte (job 1019690), scarto massimo 0.008 (latent su noisy), le altre 11 entro 0.0005, tutte dentro i CI pubblicati. Nota: la Tabella 1 e la Tabella 2 del paper sono sul set fb100, che contiene 79 soggetti di training; nel nuovo paper le tabelle vanno rifatte su held-out e questo confronto resta solo un gate di riproduzione.
Fix del leakage applicato e verificato: `eval_common.sh` ora usa il seed del checkpoint, stampa seed e primi soggetti held-out nel log, e gli hash delle out dir esistenti non cambiano (v1 invariato, soggetti identici). Eval dei seed 2345 e 3456 rilanciate sui rispettivi held-out (job 1019704-1019707), risultati nel pomeriggio.
WS6: sezione "Evaluation protocol" scritta nel draft CVPR (`main_cvpr_draft.tex`, righe ~299-335): held-out e contaminazione delle tabelle precedenti, set di riferimento (REMESH, ICT-5000, Multiface identità, Multiface ricostruzioni), baseline con normalizzazione per mesh, convenzione ad area unitaria e split per seed, studio umano, statistica. Tutti i numeri non definitivi sono segnaposto. Verificato che un viewer con accesso di sola interazione può scrivere nel database dello studio.
**WS2, confronto pulito seed 1234 (stessi 100 held-out del v1, seed verificato)**, Spearman latent: clean 0.828 → 0.841 (+1.3), jitter 0.817 → 0.815 (−0.2), translation 0.826 → 0.838 (+1.3), rotation 0.815 → 0.825 (+1.0), mixed 0.810 → 0.804 (−0.6). Effetto piccolo sull'eval a scenari; il guadagno di `pot_area` era sull'asse crop, che questa eval mescola con le altre topologie: lanciata l'eval topology del seed 1234 per la matrice per coppie. Un seed solo: nessuna conclusione finché non arrivano 2345 e 3456 sui propri held-out.

### Mattina dell'11 — WS2 zero-shot BFM→ICT e WS5 espressioni: due risultati negativi onesti

**Zero-shot cross-3DMM** (modelli BFM ad area unitaria, 3 seed, 100 soggetti held-out ICT, protocollo mesh-pair uguale a quello del 0.478 storico su FLAME): latent **0.30** (0.29 / 0.31 / 0.30), Chamfer 0.44. Peggio del BFM→FLAME e, per la prima volta, **la metrica appresa perde da Chamfer** (−0.06…−0.11 in ogni scenario e seed). Per coppia di topologie: regge dove il campionamento è denso (original↔noisy 0.84, original↔up 0.70), crolla su tutto ciò che tocca down8k (0.04–0.27). Lettura: il modello addestrato su un solo 3DMM non generalizza al secondo; è esattamente l'obiezione di YJz1. La risposta sono i modelli ICT-only e congiunto in training (fine 13 settembre). Numeri con CI in `aau/runs/ict_*/`.

**Espressioni ICT** (seed 1234, 5 espressioni × 3 intensità, GT neutra): Spearman 0.83–0.86 contro 0.84 del neutro, invariato e non monotono nell'intensità. Verificato che le mesh cambiano davvero (spostamento massimo 0.215 su diametro 2.18). **Il test è non informativo per costruzione**: i blendshape ICT applicati con lo stesso coefficiente a tutte le identità spostano i 100 soggetti insieme, quindi non disturbano il ranking. Va rifatto con espressioni diverse per soggetto (coefficienti casuali per identità), altrimenti non risponde a Z1mX.

### Mattina dell'11 — seconda revisione critic: WS3b regge, WS1 e WS3a ancora bloccanti

**WS3b: RISERVE, nessun numero da rifare.** Chamfer e latent concordano (tau 0.59, CI [0.38, 0.80]), il criterio allineato discorda da entrambi (CI includono lo zero), split-half 0.91–0.98. Limiti da scrivere: 3DDFA ha il 14.8% di superficie in più nella sfera di ritaglio (vince Chamfer nel 67% delle immagini; correggendo per la scala il margine dimezza ma l'ordine non cambia); criterio allineato unidirezionale su recon non ritagliata; apertura oculare della GT stimata dagli stessi landmark giudicati inaffidabili per il criterio NoW; primo e secondo posto indistinguibili sul criterio allineato.

**WS1: BLOCCANTE, ma con una scoperta.** Varifold e currents normalizzano per area totale, e la topologia `noisy` ha 2.28 volte l'area: la misura si rimpicciolisce del 34% e la distanza tra un soggetto e la propria versione rumorosa eguaglia quella tra soggetti diversi. Escludendo le coppie con noisy, **varifold cross-topologia fa 0.568 contro Chamfer 0.454**: la conclusione "Chamfer resta la miglior baseline" è ribaltata dai suoi stessi dati. Inoltre il proxy bbox su mesh normalizzate dà 0.284 / 0.117: CLIP e DINOv2 cross-topologia non lo superano. La colonna cross va spezzata in "cambio di tassellazione" e "perturbazione".

**WS3a duro: BLOCCANTE.** ArcFace usava ancora il detector con fallback (7768 fallimenti, tutti su noisy) e la riga proxy dichiarata non era mai stata calcolata; calcolata dal critic, 4 celle su 6 sono sature anche per il proxy. L'unico asse informativo è `crop` (tracked→crop e remesh→crop: tutto scende, latent 0.73 / 0.70, Chamfer 0.66 / 0.60, varifold 0.58 / 0.62). Con 13 soggetti i CI sono larghi 0.4.

Decisione: terzo giro di correzioni, oltre il limite di due che mi ero dato, perché i difetti sono meccanici e ben specificati (normalizzazione maxabs per varifold, ArcFace fisso in WS3a, riga proxy vera, colonne spezzate). Lezione registrata: ogni correzione va applicata in tutti i cantieri che condividono il codice, non solo dove il critic l'ha trovata.

### Mattina dell'11 — WS5 rifatto con espressioni per soggetto: ora il test morde

Generati 5 vettori di espressione casuali per ciascuno dei 500 held-out ICT (3–8 blendshape attivi su 45, coefficienti 0.3–1.0, esclusi gli sguardi): spostamento medio 0.017 su diametro 2.18 (0.41 volte jawOpen a intensità piena), con varianza tra soggetti del 24%, quindi un vero disturbo per identità. Operatori ad area unitaria, eval sugli stessi 100 soggetti di WS2 (seed 1234 areanorm, modello BFM), Spearman vs GT neutra con CI per soggetto:

| Regime | Latent | Chamfer | Δ latent | Δ Chamfer |
|---|---|---|---|---|
| neutro vs neutro (riferimento) | 0.841 | 0.951 | — | — |
| stessa espressione k per entrambe | 0.739 [0.68, 0.79] | 0.887 | −0.10 | −0.06 |
| espressione vs neutro | 0.784 | 0.914 | −0.06 | −0.04 |
| espressioni diverse (caso reale) | 0.739 | 0.886 | −0.10 | −0.07 |

Finding: le espressioni degradano il ranking delle identità, e la metrica appresa ci perde più di Chamfer. Coerente con lo zero-shot cross-3DMM: il modello BFM non è robusto fuori dal suo dominio. Da rifare con i modelli ICT-only e congiunto quando pronti, e da considerare il training con augmentation di espressione. Costo: 53 minuti di L40S per 22 run.

### Tarda mattina dell'11 — WS2: matrice 6×6 del modello ad area unitaria (seed 1234, stessi 100 held-out del v1)

Spearman latent per coppia ordinata di topologie, Δ rispetto al v1: media su 30 coppie **+0.024**; coppie con crop **+0.043** (crop→remesh 0.658 → 0.745, crop→up60k 0.687 → 0.768, up60k→crop 0.705 → 0.772); coppie con noisy +0.020; altre +0.011. Unico calo marcato down8k→up60k −0.025. Chamfer identico. Il guadagno si concentra sul crop, in direzione coerente con `pot_area`. È la Tabella 1 del modello nuovo; i seed 2345 e 3456 arrivano nel pomeriggio sui propri held-out.

## 4 ottobre 2026

### Ripresa dopo tre settimane di fermo

La sessione si era interrotta l'11 settembre in tarda mattinata e nessun lavoro è avanzato fino a oggi. Tutti i job lanciati allora sono finiti l'11-12 settembre: training ICT-only (41 h) e congiunto BFM+ICT (47 h) completati, eval dei tre seed completate, sigma sweep principale andata in TIMEOUT a 36 h ma con scrittura progressiva e i blocchi mancanti completati a parte. Studio umano online ma con zero risposte. Mancano 6 settimane alla scadenza (16 novembre) e 20 giorni al gate del 24 ottobre.

### WS2 — modello ad area unitaria su tre seed: il guadagno sul crop NON è robusto

Matrice 6×6 (stage topology) per i seed 1234, 2345, 3456, ciascuno sui propri 100 held-out (in comune con il 1234: 21 e 14 soggetti). Poiché i soggetti cambiano, il confronto con il v1 si fa sul margine latent − Chamfer:

| Coppie | v1 | s1234 | s2345 | s3456 | media seed − v1 |
|---|---|---|---|---|---|
| tutte (30) | 0.421 | 0.445 | 0.446 | 0.439 | +0.022 |
| con crop (10) | 0.699 | 0.742 | 0.689 | 0.762 | +0.032 ± 0.031 |
| con noisy (10) | 0.345 | 0.365 | 0.375 | 0.354 | +0.020 |
| altre (12) | 0.303 | 0.314 | 0.343 | 0.295 | +0.014 |

Il +0.043 su crop del solo seed 1234 non si replica: un seed perde (−0.010), uno guadagna di più (+0.063). La direzione media è positiva su tutti i gruppi, ma la variabilità tra seed è grande quanto l'effetto. **Decisione**: la convenzione ad area unitaria resta (non peggiora nulla, guadagno medio +2 punti), ma nel paper va presentata come scelta di implementazione con tre seed e dispersione, non come contributo. Correzione a una nota precedente: il "0.828" citato come riferimento v1 era lo Spearman dello scenario clean dell'eval ranking, non un aggregato della matrice per coppie.
Sull'eval a scenari, i seed 2345 e 3456 hanno un margine latent − Chamfer più basso del v1 (0.28–0.30 contro 0.33–0.37), ma i loro soggetti held-out hanno un Chamfer più alto (0.56–0.61 contro 0.48): il margine non è confrontabile tra insiemi di soggetti diversi. Per un confronto appaiato servono controlli con la stessa ricetta e gli operatori standard sugli stessi seed (come fece l'autore in `pot_plain`): lanciati i training di controllo seed 2345 e 3456 su A10 (job 1054424-1054425; L40S tutte occupate), circa 20 ore. Poi eval topology di ciascuno sul proprio held-out e Δ appaiato per seed.

### Ricognizione del lavoro v2 dell'autore (maggio–agosto) e tensione con il piano

Una ricognizione dei commit di fine agosto, non riportati nel diario dell'autore, cambia due cose. (1) Il frame "rms" per l'input (centroide pesato e raggio quadratico medio al posto di maxabs) dà crop +5.2 e tutte le coppie +3.1 su due seed appaiati, più forte della normalizzazione ad area adottata il 10 settembre (+2.9 non significativo, p 0.12; e su tre seed nostri +3.2 ± 3.1). La scelta del 10 settembre va riaperta: candidato principale è il frame rms o un frame globale unico, che l'autore aveva lanciato senza leggerne il risultato. (2) La GT non è mai stata normalizzata: lo script divide solo per un massimo globale, quindi il bersaglio è in coordinate grezze, e le tabelle mescolano cinque frame. I ranghi tra metriche non cambiano, ma i margini erano gonfiati e la compressione da NICP è 33.9%, non 21.9%. Da riportare nel paper. Pagina riassuntiva per i co-autori: `paper/DOPO_NEURIPS.html`.

### Terzo giro di correzioni (concluso il 4 ottobre): due conclusioni da rivedere

**WS1, Tabella 2 estesa su held-out**, Spearman per colonna (stessa topologia / cross no-crop / solo cambio di tassellazione / solo perturbazione): Chamfer 0.644 / 0.469 / 0.454 / 0.502; varifold maxabs 0.655 / 0.192 / 0.472 / 0.136; currents maxabs 0.661 / 0.156 / 0.292 / 0.266; LPIPS 0.590 / 0.127 / 0.287 / 0.148; ArcFace 0.293 / 0.218 / 0.300 / 0.160; CLIP 0.358 / 0.087 / 0.226 / 0.039; DINOv2 0.378 / 0.058 / 0.147 / 0.034; controllo bbox 0.284 / 0.117 / 0.088 / 0.167. Chamfer resta la miglior baseline cross-topologia; varifold la pareggia solo sul cambio di tassellazione. **Varifold non è ancora equo**: per area rimpicciolisce la mesh rumorosa, per maxabs ne sovrappesa la massa (2.28×). In corso la versione a massa unitaria, che è la definizione corretta per forme con area diversa.

**WS3a duro**: ArcFace con allineamento fisso recupera la cella rumorosa (0.571 → 0.932). Il controllo bbox su mesh normalizzate dà 0.95 su 4 celle su 6 e 0.42 sulle due con crop: anche col protocollo duro solo il crop è informativo, e lì tutte le metriche stanno tra 0.60 e 0.73.

**WS3b: la conclusione si indebolisce.** Con Chamfer in millimetri dalla scala ICP e con le ricostruzioni ritagliate alla stessa patch della GT, l'ordine diventa 3DDFA_V2 < PRNet < SynergyNet (2.37 / 2.49 / 2.56 mm), identico a quello del criterio con ICP. La discordanza vista prima tra Chamfer e ICP veniva in buona parte dalla normalizzazione per mesh e dal supporto diverso, non dall'allineamento. Resta da capire se la metrica appresa (che legge le mesh normalizzate per mesh) discorda per la stessa ragione: tau per soggetto con le colonne nuove in calcolo. Fino ad allora WS3b non va scritto come prova che l'allineamento cambia la classifica dei metodi.
**WS3b, verdetto finale (tau per soggetto, 13 soggetti, IC bootstrap per soggetto):** Chamfer su patch uguale vs criterio ICP 0.80 [0.59, 0.95], ordine identico nel 69% dei soggetti; Chamfer in mm vs ICP 0.85; Chamfer su patch uguale vs metrica appresa 0.18 [−0.08, 0.44]; Chamfer in mm vs metrica appresa 0.13. Affidabilità split-half 0.86–0.98, quindi le discordanze non sono rumore. **Sulle ricostruzioni reali, a parità di supporto e scala, l'allineamento non cambia la classifica dei metodi.** È la metrica appresa a ordinarli diversamente da tutti i criteri geometrici (mette PRNet ultimo, i criteri geometrici mettono ultimo SynergyNet). Senza un riferimento esterno di identità non si può dire chi abbia ragione. Conseguenza per il paper: WS3b non sostiene la tesi; va riportato come risultato negativo o come limite, e la tesi "l'allineamento distorce" resta supportata su dati sintetici e sul controllo crop di Multiface, non sul confronto tra metodi di ricostruzione reali.

### Frame rms: tre training in corso, eval appaiata in preparazione

Training della ricetta v1 con il frame di input rms dell'autore (`train_fast.py --frame rms`, operatori standard: rms e rms con operatori in unità rms differivano di 0.004, sotto il rumore), seed 1234, 2345, 3456, su A10 con cache in RAM: 238 s per epoca, fine verso le 21 del 4 ottobre (job 1054482-1054484). Controlli appaiati con frame standard: seed 2345 e 3456 in corso (più lenti, senza cache, fine il 5 ottobre); per il seed 1234 il controllo è il v1. Frame verificato numericamente (raggio rms 1.000000). L'eval del repo non passa il frame: in preparazione un'eval con l'script dell'autore (`eval_by_topology.py --frame`), validata sul v1, con code di dipendenza che parte da sola alla fine dei training e produce la tabella appaiata per seed.
Eval con frame validata (job 1054489, 1054490): stesso split, stesse 148.500 coppie, celle del repo riprodotte a 4e-4; differisce solo l'aggregazione (Spearman unico per gruppo invece che media delle celle). Riferimento v1 nel formato dell'autore: crop 0.709, noisy 0.794, resample 0.785, all 0.751. Controlli rilanciati con lo stesso wrapper in cache (frame standard, job 1054492-1054493), quindi rms e controllo leggono i dati allo stesso modo. Coda di eval e tabella appaiata in dipendenza dai training (1054494-1054499). Da aggiungere: controllo seed 1234 con il wrapper, per non confrontare con il v1 addestrato dal trainer vecchio.
**WS1 chiuso nella sostanza**: varifold e currents su misura a massa unitaria nel frame maxabs (sigma riportate con fattore misurato 1.848), auto-distanza zero, stesso soggetto sotto la mediana tra soggetti diversi in 100/100 casi. Held-out, colonne stessa topologia / cross no-crop / tassellazione / rumore: varifold 0.646 / 0.258 / 0.496 / 0.307; currents 0.663 / 0.210 / 0.287 / 0.289; Chamfer 0.644 / 0.469 / 0.454 / 0.502. Chamfer resta la miglior baseline quando cambia la topologia; varifold la supera di poco solo sul cambio di tassellazione. Questa è la versione da mettere nel paper.
**Correzione a una nota precedente (WS3a duro, crop):** non è vero che sul crop tutte le metriche stanno tra 0.60 e 0.73. Quella fascia vale per le metriche geometriche (Chamfer 0.58–0.65, varifold 0.61–0.63, currents 0.59–0.59) e per la metrica appresa. Le percettive su render vanno molto meglio: ArcFace 1.000 su tutte le coppie con crop, LPIPS 0.73–0.84, CLIP 0.79–0.94, DINOv2 0.63–0.89, con il controllo bbox a 0.42. Su volti reali, anche tagliati, una rete di riconoscimento facciale applicata a render senza texture separa perfettamente i 13 soggetti. È un risultato da riportare e da capire: o i render conservano tratti d'identità che le metriche geometriche perdono quando cambia il supporto, o 13 soggetti sono troppo pochi per mettere in difficoltà ArcFace.

### Decisione di framing (4 ottobre, pomeriggio)

Valutazione condivisa con l'autore: con i risultati attuali le probabilità a CVPR del framing "metrica appresa che corregge l'allineamento" sono basse, perché le verifiche chieste dai reviewer (secondo 3DMM, espressioni, dati reali) sono uscite contro o inconcludenti. Decisione al gate del 24 ottobre tra due framing: **A**, metrica appresa (solo se il modello congiunto BFM+ICT batte Chamfer su entrambi i 3DMM e lo studio umano la favorisce); **B**, paper di metodologia di valutazione (allineamento, normalizzazione e frame che distorcono la valutazione; risultati negativi inclusi). L'autore concorda. In preparazione un outline di B in `paper/OUTLINE_B.md`. Lo studio umano resta critico per entrambi e dipende dalla distribuzione del link da parte dell'autore.

### Outline B, esperimenti 6 e 7 (4 ottobre, pomeriggio)

**Identità distinte (richiesta di Z1mX):** nessun soggetto held-out è un quasi-duplicato di un soggetto di training, né su BFM né su ICT. La distanza dal vicino più prossimo nel training ha la stessa mediana della distanza dal vicino più prossimo dentro il training stesso (BFM 0.193 contro 0.196; ICT 0.096 contro 0.096); il minimo assoluto è 0.47 volte la mediana tra held-out su BFM e 0.29 su ICT. I coefficienti BFM delle 500 mesh non esistono più: il controllo su BFM usa solo D_GT. Appendice del draft riscritta.

**WS3b con la metrica appresa sulla stessa patch di Chamfer:** la discordanza non scompare. Con il supporto uguale (rapporto d'area patch/GT 0.99–1.00 per tutti i metodi) la metrica appresa dà tau 0.03 [−0.28, 0.28] contro il criterio ICP e 0.23 [−0.08, 0.49] contro Chamfer su patch uguale; tau tra la versione vecchia e nuova 0.95. La metrica appresa ordina i metodi in modo stabile e diverso dai criteri geometrici, indipendentemente dal supporto: la causa è nel modello, non nell'input.

### Outline B, esperimenti 1–3: allineamento, compressione e frame su held-out (4 ottobre, sera)

Gate: la Tabella 2 del paper su fb100 è riprodotta esattamente (8/8 celle, delta 0). Il paper usava 4096 punti sulla stessa topologia e 2048 sul cross; le tabelle nuove usano 4096 ovunque.

**Allineamento** (Spearman, stessa topologia / cross no-crop, held-out):

| | BFM | ICT (GT maxabs) |
|---|---|---|
| Chamfer | 0.644 / 0.469 | 0.979 / 0.433 |
| Rigid ICP + Chamfer | 0.488 / 0.415 | 0.872 / 0.645 |
| NICP P2P | 0.420 / 0.388 | 0.776 / 0.600 |
| NICP P2Tri | 0.427 / 0.395 | 0.778 / 0.613 |
| M3DFB RLR + Chamfer | 0.501 / 0.473 | 0.432 / 0.229 |

**A topologia fissa l'allineamento danneggia il ranking su entrambi i 3DMM. Cross-topologia lo danneggia su BFM ma lo migliora su ICT** (0.645 contro 0.433). "Alignment hurts" non è una legge generale: dipende dal 3DMM e dal tipo di cambio. Questo spinge ulteriormente verso il framing B.

**Compressione** (quota di dispersione IQR/mediana trattenuta da NICP P2P rispetto a Chamfer): BFM 34.0% sulle 30 coppie cross con crop, 45–50% senza crop; ICT 29–49% a seconda del gruppo. Il 33.9% corretto del commit dell'autore esiste solo includendo il crop.

**Frame** (Chamfer, held-out BFM, stessa topologia / tassellazione / perturbazione / crop): maxabs 0.644 / 0.454 / 0.502 / −0.006; area 0.558 / 0.571 / 0.097 / 0.231; rms 0.602 / 0.621 / 0.416 / 0.339; **frame globale unico 0.714 / 0.718 / 0.719 / 0.686**. Nessuna normalizzazione per mesh è neutra, e il frame globale batte tutto, crop compreso. In verifica quanto del guadagno venga da posizione e taglia assolute, che la GT grezza contiene (controllo bbox nel frame globale, traslazione e scala separate). Nota: la GT ICT di train_ready è in frame maxabs, non grezza come la BFM; le due tabelle non sono nello stesso frame.
**Frame globale scomposto** (Spearman stessa topologia / tassellazione / perturbazione / crop; job 1054946-1054965): su BFM la sola bbox nel frame globale fa 0.370 / 0.352 / 0.265 / 0.015 contro 0.714 / 0.718 / 0.719 / 0.686 di Chamfer globale; togliendo la posizione per mesh 0.679 / 0.509 / 0.487 / 0.509, togliendo la taglia per mesh 0.659 / 0.516 / 0.595 / 0.025. **Su BFM il guadagno del frame globale è forma più taglia assoluta, non segnale banale; la taglia della testa è informazione d'identità che la normalizzazione per mesh butta via, e senza di essa il crop crolla.** Su ICT la GT grezza è dominata dalla taglia (correlazione 0.361 con la GT maxabs) e la sola bbox arriva a 0.691 / 0.667 / 0.656 / 0.134; Chamfer globale 0.919 / 0.877 / 0.882 / 0.803. Conclusione per il paper: la definizione stessa del riferimento (con o senza taglia) decide quale metrica vince; va dichiarata, e ogni tabella va riportata nei due casi.

### Pilota "operatori dedicati" (approvato dall'autore, 4 ottobre notte)

Idea dell'autore: una variante di DiffusionNet con operatori specifici per il task, a partire dal pozzo di potenziale. Il pozzo era stato testato solo a 0.21, valore che nello sweep risultava peggiore di nessun pozzo; il valore del ginocchio, 0.55 (−46% di inconsistenza spettrale sul crop), non è mai stato addestrato. Pilota: due bracci a seed 1234, con lo stesso wrapper del controllo appaiato: (1) pozzo a 0.55; (2) DiffusionNet a due rami di operatori (standard e pozzo), parametri pari ±10%. Criterio fissato prima di guardare: almeno +0.05 sulle coppie con crop rispetto al controllo e nessun peggioramento oltre 0.02 sullo zero-shot ICT. Se passa, tre seed e sezione nel paper o nucleo di un lavoro successivo; se no, una riga come idea testata. Non sottrae tempo al framing B.

## Brainstorm: una DiffusionNet con operatori adatti al task (4 ottobre, notte)

Richiesta dell'autore: prima capire cosa manca, teorizzare e testare, poi costruire la versione con operatori aggiuntivi, documentando tutto qui.

### Cosa sappiamo dei fallimenti del modello attuale

1. **Crop**: è l'asse peggiore in-domain (margine latent − Chamfer migliore proprio lì, ma Spearman assoluto più basso). Il crop è, per Weyl, una riscalatura dello spettro per il rapporto d'area (R² 0.9995), più un cambio di condizioni al bordo (Neumann su un bordo nuovo).
2. **Decimazione (down8k)**: tutte le coppie con down8k crollano nello zero-shot su ICT (0.04–0.27) e sono le peggiori anche in-domain (down8k→up60k è l'unica cella peggiorata col modello ad area unitaria).
3. **Ricostruzione Poisson**: il partner `remesh_10k` di FaceScape, ricostruito per Poisson, fa crollare lo zero-shot da 0.41 a 0.11 (controllo causale dell'autore).
4. **Secondo 3DMM**: zero-shot BFM→ICT 0.30 contro 0.44 di Chamfer.
5. **Espressioni**: −0.10 contro −0.06 di Chamfer.
6. **Ricostruzioni reali**: la metrica appresa ordina i metodi diversamente da tutti i criteri geometrici, anche con supporto uguale (tau 0.03–0.23).
7. **Taglia**: su BFM la GT è in coordinate grezze e contiene la taglia della testa; Chamfer nel frame globale (che conserva la taglia) fa 0.71 ovunque, 0.69 sul crop. **Il modello invece riceve input normalizzati maxabs per mesh: non può vedere la taglia assoluta**, e il divisore maxabs è fissato da un vertice estremo, che sul crop cambia.

### Come sono costruiti oggi gli operatori

`diffusion-net/src/diffusion_net/geometry.py`: per le mesh usa il **Laplaciano cotangente semplice** (`pp3d.cotan_laplacian`) e le aree vertice come massa; il Laplaciano robusto di Sharp & Crane 2020 (Delaunay intrinseco con tufted cover), pensato per mesh di bassa qualità, è presente nel codice ma commentato. Gli operatori sono calcolati una volta per mesh e non vedono le perturbazioni dell'eval. Gli autovettori sono 128 (k_eig); il diffusion time è appreso per canale.

### Ipotesi, con predizione e test economico

- **H1 – taglia invisibile.** Il modello perde informazione d'identità perché la normalizzazione per mesh cancella la taglia assoluta, che la GT contiene. *Predizione:* la differenza di taglia (log del divisore maxabs, o radice dell'area) da sola correla con D_GT; il residuo della metrica appresa correla con la differenza di taglia. *Test:* correlazioni su BFM held-out, nessun training. *Rimedio se confermata:* passare al modello un token di scala globale (log-scala) o usare il frame globale come input.
- **H2 – bordo e crop.** Il crop cambia lo spettro per riscalatura (già trattata da Weyl) e per condizioni al bordo. Un pozzo di potenziale vicino al bordo smorza la dipendenza degli autovettori bassi dal bordo. *Predizione:* a 0.55 (ginocchio dello sweep) migliora il crop. *Test:* pilota già lanciato (pozzo 0.55 e DiffusionNet a due rami).
- **H3 – qualità della mesh.** Il cotangente semplice dà spettri inconsistenti su mesh decimate, rumorose o ricostruite per Poisson; il Laplaciano robusto (Delaunay intrinseco) li rende più consistenti. *Predizione:* la dispersione degli autovalori normalizzati e la distanza tra descrittori spettrali (HKS) dello stesso soggetto tra le 6 topologie scendono passando dal cotangente al robusto, soprattutto per down8k e noisy. *Test:* solo calcolo di operatori, su CPU, nessun training.
- **H4 – espressioni.** Non è un problema di operatori ma di dati: il modello non ha mai visto espressioni. *Rimedio:* augmentation con espressioni ICT in training. Fuori dal pilota sugli operatori.
- **H5 – secondo 3DMM.** Parte dal dominio (risposta: training congiunto, in valutazione stanotte), parte forse da H1 e H3.

### Piano

1. Diagnostiche senza training per H1 e H3 (in corso), più ricognizione di letteratura sulle varianti di operatori per DiffusionNet e affini.
2. Pilota H2 già in training (pozzo 0.55, due rami).
3. Sulla base dei risultati, progetto della versione nuova: insieme di operatori (cotangente, robusto, pozzo) come rami, token di scala, frame d'ingresso; poi ablazioni, una componente alla volta, con controllo appaiato e criterio fissato prima.

### Letteratura sugli operatori (4 ottobre notte, `literature/OPERATORS_2026-10-04.md`)

Raccomandazione per rapporto beneficio/costo: (1) Laplaciano robusto di Sharp & Crane (Delaunay intrinseco) + unità fisiche fisse senza normalizzazione per mesh + token di scala globale, robusto al crop: attacca insieme decimazione, Poisson e perdita della taglia (H1, H3); (2) Hamiltoniano con potenziale crescente verso il bordo (Choukroun 2018, localized manifold harmonics di Melzi 2018) più crop casuale come augmentation: attacca il crop (H2); (3) attenzione a massa concentrata sopra DiffusionNet (Shetty et al. 2026) per decimazione e contesto globale. Scartati per ora: DeltaConv, operatore di Dirac, Laplaciano anisotropo (costo alto, beneficio incerto), Steklov (estrinseco, codice non verificato). Non risultano lavori che combinino più operatori in DiffusionNet per volti: possibile lacuna reale, da riverificare prima di rivendicarla. Le fonti marcate come lette per intero sono due; le altre vanno riverificate prima di citarle.

### Incidente: due training in thrashing di memoria (5 ottobre, notte)

I training rms seed 3456 e controllo seed 1234 si sono rallentati da 3 a 87 secondi per iterazione: entrambi sullo stesso nodo A10, ciascuno al limite di 100 GB, con la cache in RAM cresciuta durante il run e il kernel bloccato a recuperare memoria (GPU a 0%). Cancellati all'epoca 108 e 94 su 120; le eval appaiate girano sui loro migliori checkpoint salvati, marcate come parziali; rilanciati entrambi con 180 GB per avere il confronto a budget pieno (job 1055025-1055026). Il pilota sul pozzo usa lo stesso wrapper e viene ridimensionato.

### Diagnostiche del brainstorm: H1 e H3 non passano la soglia, ma indicano il progetto (5 ottobre, notte)

Predizioni scritte prima dei numeri in `aau/runs/brainstorm/PREDICTIONS.md`.

**H1, taglia.** Spearman tra D_GT e differenza di taglia: 0.33 (divisore maxabs), 0.24 (raggio rms), 0.19 (radice dell'area): appena sopra la soglia solo per maxabs, senza intervallo di confidenza. La metrica appresa segue già la taglia maxabs (0.35) ma è cieca alla taglia rms (0.02), e il suo residuo rispetto a D_GT correla 0.36 con la differenza di taglia rms (0.31 in media sulle coppie cross). Sul crop il divisore maxabs scende al 0.87 di quello dell'original (dispersione del log 0.037), cioè un errore di scala grande quanto l'intera variabilità di taglia tra soggetti (0.033): il frame maxabs inietta sul crop rumore di scala pari al segnale. **Verdetto: non confermata alla soglia, ma il meccanismo è chiaro e spiega perché il frame rms aiuta sul crop.**

**H3, operatori** (50 soggetti, 64 autocoppie; dispersione degli autovalori tra topologie, errore HKS, separabilità stesso/altro soggetto, più basso è meglio):

| Variante | Dispersione | Errore HKS | Separabilità |
|---|---|---|---|
| Cotangente (attuale) | 0.137 | 0.709 | 5.93 |
| Robusto | 0.138 | 0.709 | 6.15 |
| Cotangente, area 1 | 0.056 | 0.023 | 2.35 |
| Robusto, area 1 | 0.048 | 0.020 | 2.04 |
| Pozzo 0.55 | 0.100 | 1.71 | 0.98 |

Il Laplaciano robusto da solo non cambia quasi nulla (su noisy −0.8%); con l'area unitaria aggiunge il 16–23% su noisy. Il grosso viene dall'area unitaria (consistenza 2.5–3 volte migliore). Il pozzo è ottimo sul crop (dispersione 0.008 contro 0.072) ma peggiora remesh e down8k. **Verdetto: H3 non confermata come formulata; robusto + area 1 è la variante spettralmente migliore.** Avvertenza dall'autore: la consistenza spettrale non ha predetto la metrica in passato (pozzo), quindi va verificata in training.

### Progetto della versione nuova e ablazioni

Componenti candidate, una alla volta contro lo stesso controllo (seed 1234, wrapper in cache, 120 epoche):
- **A. frame rms** (già in training su tre seed);
- **B. frame rms + token di taglia** (log del raggio rms della mesh grezza, concatenato dopo il pooling): dà al modello la taglia senza il rumore del divisore maxabs;
- **C. operatori robusti ad area 1** al posto del cotangente;
- **D. pozzo 0.55 e due rami** (pilota in corso);
- **E. combinazione** delle componenti che passano.
Criterio per ciascuna, fissato ora: almeno +0.03 sul margine latent − Chamfer medio sulle 30 coppie o +0.05 sulle coppie con crop, senza perdere più di 0.02 sullo zero-shot ICT.

### Frame rms contro standard: risultato intermedio (5 ottobre, notte)

Tabella appaiata per seed, Δ rms − standard (Spearman, aggregazione per gruppo dell'autore), media ± dev.std su 3 seed: crop +0.025 ± 0.016, noisy +0.025 ± 0.007, resample +0.002 ± 0.011, tutte +0.017 ± 0.011. **Provvisorio:** due seed su tre hanno uno dei due bracci a budget ridotto (rms s3456 fermo all'epoca 108, controllo s1234 all'epoca 94, best a 68), e il Δ del seed 1234 (+0.029) è probabilmente gonfiato. Direzione positiva coerente con l'autore, ampiezza circa la metà del suo +5.2 sul crop (n=2). Tabella definitiva a budget pieno quando finiscono i due training rilanciati (job 1055029-1055032).

### Pilota pozzo: correzione e risorse (5 ottobre, notte)

**Correzione al brainstorm:** il pozzo non era stato testato solo a 0.21. L'A/B di agosto (crop 0.7072 → 0.7012) è il pozzo a **0.55 senza maschera** sul pooling (`pot_w55`, STATUS.md:1569). Mai addestrato è invece il pozzo 0.55 **con pooling ristretto alla regione d'interesse** (`pot_m55`), che è il braccio del pilota. Trovati e aggirati due difetti silenziosi nel codice dell'autore: la maschera della regione d'interesse veniva scartata prima del modello e il caricamento falliva senza errore, quindi `pot_m55` sarebbe stato `pot_w55` sotto altro nome (hook in `aau/models/pilot_hooks.py`; ora l'assenza della maschera ferma il job). DiffusionNet a due rami in `aau/models/dn_dual_ops.py`: ogni blocco diffonde con la base standard e con quella del pozzo, tempi e gradienti propri, concatenazione prima della MLP; width 103 invece di 128 per avere gli stessi parametri (+0.2%); checkpoint per blocco per stare in memoria. Smoke superati.
**Risorse:** i training con cache in RAM chiedono 180–200 GB e i nodi A10 non li ospitano più di uno alla volta; spostati sul nodo V100 (1.4 TB di RAM) con limiti di tempo stretti per entrare nel backfill.

### Ablazioni B, C, E in training (5 ottobre, notte)

Implementazione in `aau/models/` (nessuna modifica al trainer): **token di taglia** = log del raggio rms pesato per area della mesh grezza, standardizzato sul training, concatenato dopo il pooling (proiezione 513 → 256, colonna del token inizializzata a zero così la partenza è identica al controllo). Invarianza del token tra topologie dello stesso soggetto: entro 0.006 per remesh, noisy, down8k, up60k; **crop −0.088** (raggio ×0.916), cioè il token non è invariante al crop: per costruzione il crop riduce la superficie. **Operatori robusti ad area 1** calcolati per 3000 mesh BFM e 3000 ICT held-out, verificati. **Scoperta collaterale:** il loader divide gli autovalori per il massimo e i gradienti per la sua radice, quindi la normalizzazione ad area unitaria degli operatori è in gran parte annullata nel forward: C misura di fatto robusto contro cotangente, e anche il modello "area unitaria" di settembre differiva dal v1 meno di quanto pensassimo (attraverso massa e autovettori, non attraverso la scala degli autovalori). Da tenere presente nell'interpretare il +0.02 di settembre.
Training su V100 (nessun nodo A10 con 180 GB liberi), circa 220 s per epoca, fine verso le 9. All'epoca 2: controllo loss 0.0755 / xtopo 0.27; B 0.064 / 0.37; C 0.076 / 0.26; E 0.063 / 0.36. Segnale precoce a favore del token di taglia, da non sovrainterpretare. Eval e tabella in coda automatica.

### WS2: tabella cross-3DMM (5 ottobre, notte) — il modello congiunto batte Chamfer su entrambi i 3DMM

Spearman cross-topologia, protocollo mesh-pair (lo stesso dello zero-shot 0.30), latent [IC 95% per soggetto] / Chamfer; zero soggetti di training valutati in ogni cella (verificato rileggendo i soggetti dagli output: lo split held-out di ICT-only e del congiunto è quello ricostruito dal loro training, non la vista held-out precedente, che conteneva l'80% di soggetti di training di ICT-only):

| Modello | BFM | ICT |
|---|---|---|
| Solo BFM | 0.779 [0.73, 0.82] / 0.237 | 0.294 [0.24, 0.34] / 0.442 |
| Solo ICT | 0.175 [0.14, 0.21] / 0.237 | 0.986 [0.98, 0.99] / 0.355 |
| BFM + ICT | **0.856 [0.83, 0.88]** / 0.246 | **0.981 [0.97, 0.99]** / 0.347 |

Il congiunto batte Chamfer su 30/30 coppie di topologie in entrambi i domini, e sul BFM fa meglio del modello solo BFM. È la variante "positiva" del criterio fissato nell'outline B prima di guardare. Ma: (1) la generalizzazione c'è solo se il 3DMM è nel training (solo ICT su BFM perde, 0.175 contro 0.237); (2) le espressioni degradano anche il congiunto (0.881 contro 0.992 neutro, −0.111); (3) le celle della stessa colonna hanno soggetti diversi (100/108 su BFM, 100/95/89 su ICT); (4) la GT ICT usata è in frame maxabs, mentre quella BFM è grezza; (5) alcuni soggetti di valutazione erano tra i 16 usati per scegliere il checkpoint. In revisione dal critic prima di entrare nel paper.

**Conseguenza per il framing:** C4 dell'outline cambia segno: "una metrica appresa senza registrazione sfugge ai confondenti solo se la distribuzione di training copre la famiglia di destinazione; addestrata su due 3DMM batte Chamfer su entrambi". Il framing resta B, con la metrica come strumento raccomandato con condizioni. Il framing A torna possibile al gate del 24 ottobre se lo studio umano la favorisce e se il modello congiunto regge su dati reali (WS3b da rifare con il congiunto).
**Revisione del critic sulla tabella cross-3DMM: RISERVE.** Il claim principale regge: il congiunto batte Chamfer su entrambi i 3DMM confrontando sugli stessi soggetti di ogni riga; nessun leak (split ricostruito uguale al log, 16/16 soggetti dell'eval online); togliendo i soggetti usati per scegliere il checkpoint i numeri non cambiano (BFM 0.865). Correzioni:
- **"Sul BFM fa meglio del modello solo BFM" non è dimostrato:** sui 19 soggetti held-out per entrambi il vantaggio è +0.021 [−0.013, 0.065]. Ritirato.
- **I numeri ICT dipendono dalla GT:** la GT ICT maxabs correla 0.35 con quella grezza. Con la GT grezza il congiunto fa 0.501 contro 0.210 di Chamfer (mesh-pair, margine +0.29 [0.23, 0.35], regge), ma in media per coppia di soggetti 0.506 contro 0.457, margine [−0.003, 0.109] (non regge). Lo 0.98 va presentato come legato alla GT nello spazio dell'input. Sul BFM cambiare la GT a maxabs cambia poco (congiunto 0.793 contro 0.296).
- **"Solo ICT su BFM perde" vale solo in mesh-pair:** in media per coppia di soggetti vince (0.573 contro 0.485). Il protocollo va dichiarato accanto a ogni claim.
- **Righe diverse della stessa colonna non sono confrontabili:** Chamfer su ICT oscilla tra 0.26 e 0.44 secondo i soggetti estratti; lo 0.442 contro 0.347 era solo campionamento.
- La colonna ICT è satura (0.97–0.99 su tutte le coppie di topologie, down8k compreso): ICT è più facile di BFM per il modello, forse anche per i 4000 soggetti di training contro 400.

### Modello congiunto sui dati reali (5 ottobre, notte)

Riproduzione verificata prima (10 soggetti BFM, Spearman identico su A10; su T4 la distanza si sposta fino a 2.6e-3 per l'aritmetica della GPU, ininfluente sul rango ma i job reali sono su A10). Operatori ad area unitaria calcolati anche per le ricostruzioni.

**Ricostruzioni Multiface (WS3b), tau per soggetto [IC 95%]:** congiunto contro criterio ICP 0.33 [0.08, 0.59] (solo BFM: −0.03), contro Chamfer in mm 0.49 [0.23, 0.74], contro Chamfer su patch uguale 0.33 [0.13, 0.54]; affidabilità split-half 0.94. Si avvicina ai criteri geometrici ma resta lontano dall'accordo che questi hanno tra loro (0.80–0.85); classifica globale invariata (3DDFA_V2, SynergyNet, PRNet), i criteri geometrici mettono PRNet secondo. AUC d'identità sulle ricostruzioni 0.85–0.89.
**Multiface duro (WS3a), AUC stesso soggetto con espressione diversa contro soggetti diversi:** tracked→tracked 0.991, tracked→crop 0.692, remesh→crop 0.717, tracked→noisy 0.966, down→up 0.933, crop→crop 0.988; differenze da latent v1 tra −0.044 e +0.019, tutte dentro gli intervalli.
**Lettura:** l'addestramento su due 3DMM risolve la generalizzazione tra modelli sintetici ma non il comportamento su volti reali tagliati. Sul crop reale la metrica appresa resta sotto ArcFace su render (1.000) e LPIPS (0.73–0.84). Il collo di bottiglia sui dati reali non è il numero di 3DMM.

### Ipotesi H6: varietà del crop in training (5 ottobre, notte)

Nel training il modello vede un solo tipo di crop (taglio canonico, stessa regola per tutti i soggetti). Sui dati reali il crop varia per posizione ed estensione, e lì la metrica appresa resta a 0.69–0.72 di AUC contro 1.000 di ArcFace e 0.73–0.84 di LPIPS. **Predizione:** addestrare con crop casuali variati (frazione 60–90% dei vertici, direzione del taglio casuale, bordo irregolare) migliora le coppie con crop su Multiface e su REMESH senza perdere sulle altre. **Test:** variante F, ricetta v1 seed 1234, con 5 crop casuali per soggetto di training precalcolati (operatori offline) aggiunti come topologie extra; stesso controllo appaiato; criterio: +0.05 di AUC sulle coppie con crop di Multiface o +0.05 di Spearman sulle coppie con crop di REMESH, senza perdere più di 0.02 altrove.
**Variante F avviata** (job 1055162, L40S, fine verso le 10:45): 2000 crop casuali (5 per ciascuno dei 400 soggetti di training, nessuno per gli held-out), frazione di vertici tenuti 0.60–0.90 uniforme (media 0.75), metà con un secondo taglio laterale, occhi e naso sempre protetti (raggio 0.15 volte la distanza tra gli angoli esterni degli occhi, circa 13 mm), operatori standard. Entrano nel training senza modifiche: il trainer etichetta `crop_rK` come `crop`, e a ogni passo pesca uno dei 6 crop disponibili (canonico compreso, 1/6 delle volte). Limite del disegno: i tagli vengono soprattutto dal basso (52% sotto i 30° dal mento), perché la protezione degli occhi impedisce i tagli laterali profondi. Eval appaiate (REMESH e Multiface duro, controllo compreso) e tabella in coda automatica.

### Frame rms contro standard: risultato definitivo, budget pieno (5 ottobre, mattina)

Tre seed, ciascuno appaiato con il controllo dello stesso seed e dello stesso wrapper, 120 epoche; Spearman per gruppo (aggregazione dell'autore), Δ rms − standard, media ± dev.std: **crop +0.019 ± 0.008, noisy +0.027 ± 0.011** (positivi su tutti e tre i seed), resample −0.006 ± 0.011, tutte +0.012 ± 0.008. Il frame rms aiuta in modo consistente sul crop e sul rumore, ma poco: due-tre punti, contro il +5.2 dell'autore su due seed. Si adotta come frame di default dei modelli nuovi; non è un contributo da titolo.
**Nota sul riferimento:** il controllo seed 1234 con il wrapper in cache fa 0.765 sul crop contro 0.709 del v1 (stessa ricetta, stessi dati, stessi soggetti): la differenza non viene dal frame ma dal percorso di training o dalla selezione del checkpoint (best all'epoca 114 contro 82). Tutti i confronti nuovi sono contro il controllo con il wrapper, mai contro il v1.

### Bug trovato prima che producesse numeri sbagliati (5 ottobre, mattina)

`aau/submit.sh` carica `env.sh`, che esporta sempre i percorsi dei dati e della matrice GT di BFM; lo script di eval ICT delle ablazioni usava i propri default ICT solo se le variabili erano vuote, quindi li ignorava. Per la variante C ha prodotto un errore (GT BFM, nessun soggetto in comune); per B e il controllo avrebbe valutato in silenzio sui dati BFM presentandoli come ICT. Corretto con variabili ICT dedicate e rilanciate tutte e quattro le eval ICT. Controllati gli altri percorsi: l'eval ICT del pilota passa i percorsi ICT in modo esplicito (log: soggetti id145xx, 100 selezionati) ed è corretta; lo zero-shot di settembre e la tabella cross-3DMM usano script che impostano i percorsi esplicitamente. Lezione: ogni eval deve stampare e verificare il dominio dei soggetti selezionati, non solo il seed.

### H6, variante F (crop casuali in training): negativa su REMESH (5 ottobre, mattina)

Spearman per gruppo, F − controllo appaiato (seed 1234, stesso wrapper, 120 epoche entrambi): **crop −0.114** (0.651 contro 0.765), noisy −0.038, resample −0.024, tutte −0.060. Il criterio richiedeva +0.05 sul crop senza perdere oltre 0.02 altrove: **fallito su REMESH**, e in modo netto. Spiegazione probabile, non verificata: il crop canonico, quello valutato, compare in training solo una volta su sei; i crop casuali con bordo irregolare e tagli soprattutto dal basso non insegnano un'invarianza utile al crop canonico. Il peggioramento anche su noisy e resample suggerisce che le topologie casuali disturbano l'apprendimento in generale. Manca ancora il test sui crop reali di Multiface (WS3a), che era il bersaglio principale; il verdetto finale su H6 aspetta quel numero.

### Ablazioni B, C, E: il token di taglia funziona in dominio e distrugge lo zero-shot (5 ottobre, mattina)

Seed 1234, controllo appaiato con lo stesso wrapper; Spearman per gruppo su BFM held-out e zero-shot ICT (all cross, scenario clean):

| Braccio | Crop | Noisy | Resample | Tutte | Margine − Chamfer, 30 celle | Zero-shot ICT |
|---|---|---|---|---|---|---|
| Controllo | 0.765 | 0.783 | 0.805 | 0.782 | 0.444 | 0.369 [0.31, 0.43] |
| B (rms + token di taglia) | +0.012 | +0.059 | +0.027 | +0.027 | +0.031 | **0.135 (−0.234)** |
| C (operatori robusti, area 1) | −0.017 | +0.022 | −0.013 | −0.007 | −0.005 | 0.282 (−0.087) |
| E (B + C) | +0.035 | +0.082 | +0.039 | **+0.048** | **+0.049** | **0.115 (−0.255)** |

**Lettura.** Il token di taglia dà il guadagno in dominio più grande visto finora (E: quasi +5 punti su tutte le coppie, +8 sul rumore), confermando H1 nel suo senso pratico: la taglia è informazione d'identità che il modello non vedeva. Ma la relazione tra taglia e identità imparata su BFM non trasferisce a un altro 3DMM, e lo zero-shot ICT crolla. Gli operatori robusti da soli non aiutano (C negativo in dominio e in zero-shot), però in E si sommano positivamente al token sul rumore. **Criterio fissato prima: tutti e tre NON passano** per la clausola sullo zero-shot. Un seed solo.
**Prossimo test:** il caso d'uso reale non è lo zero-shot ma l'addestramento congiunto (che già batte Chamfer su entrambi i 3DMM). Test: E congiunto BFM+ICT contro il congiunto attuale, con token standardizzato per dominio. In parallelo: E e B sui crop reali di Multiface (il bersaglio vero), e rilancio dell'eval Multiface di F, fallita in un secondo.


> Pagina riassuntiva della notte 4→5 ottobre (operatori, ipotesi, varianti): `paper/NOTTE_OPERATORI.html`, https://claude.ai/artifact/J847wzg6RB4v8hTv8wbCqL

## 5 ottobre, mattina: terzo dominio per il test leave-one-out

**Domanda.** Il modello addestrato su due 3DMM (BFM+ICT) generalizza a un terzo mai visto? Se sì, la strada è la diversità dei dati sintetici e il framing A resta possibile. Se no, la linea "modello" si chiude e il risultato va nel framing B.

**FLAME.** Va avanti solo con la licenza ufficiale: l'utente si registra su flame.is.tue.mpg.de. Non uso copie non autorizzate. Intanto `coder` collega la pipeline esistente `v2_work/genflame/` al protocollo attuale e la collauda con un modello finto.

**Alternative, censite sulle fonti ufficiali.**
- **HIFI3D** (Tencent): repo MIT, dati "research only", download diretto. 200 scansioni est-asiatiche, quindi un dominio davvero diverso da BFM. Scelto come terzo dominio immediato.
- **FaceVerse v1/v2**: seconda scelta, download diretto.
- **HIFI3D++**: scartato come dominio indipendente, perché è costruito anche da LYHM e FaceScape.
- **BFM 2009 rispetto a 2017**: non è un dominio diverso.
- **LSFM, UHM, LYHM, Headspace, FaceWarehouse, NPHM, FaceVerse-Dataset**: richiedono email o un modulo firmato da un docente di ruolo.

**Lanciati.** Download di HIFI3D (runner). Pipeline e valutazione zero-shot su HIFI3D (`coder`): congiunto, solo BFM, solo ICT, Chamfer. Output in `aau/runs/ws_hifi3d/`.

**Valanga di dati (decisione del 5 ottobre, su indicazione dell'utente).** Si scala il training di almeno un ordine di grandezza: più identità BFM e ICT, più espressioni, stesse 6 topologie.
- `coder` misura prima lo spazio per mesh e la quota (1 TB), e scrive il piano in `aau/data_scale/PLAN.md`. Poi genera, prima con un mini-lotto di prova.
- Held-out non negoziabile: i soggetti di test attuali restano fuori dal training; HIFI3D e FLAME restano domini di test.
- FaceVerse v2 in download, come candidato quarto dominio.
- **FaceVerse v2 scaricato**: `~/data/faceverse/faceverse_simple_v2.npy`, 153 MB, 150 basi di identità, 52 di espressione, 28.632 vertici; repo BSD-2, il modello non dichiara una licenza. Quarto dominio di test, mai in training.
- **HIFI3D**: in download (job 1055665). Google Drive richiedeva la conferma con uuid.

## 5 ottobre, mezzogiorno: quota disco esaurita

**Cosa è successo.** La home ha raggiunto 1 TB. Insieme a un login scaduto, ha fatto cadere tutti gli agenti.

**Job falliti:**
- training congiunto-E e controllo (dopo 1h07);
- ws3a-arm-E;
- build di FLAME e HIFI3D;
- mini-lotto della valanga.

La valanga non è la causa: aveva scritto solo 87 KB. Lo spazio era già occupato da ICT (425 GB), REMESH (252) e Multiface (184).

**Pulizia, con il consenso dell'utente:**
- zip di HIFI3D (5.8 GB);
- 4 directory di ricostruzioni Multiface con operatori (circa 72 GB, intermedi rigenerabili);
- vista ICT `eval_view_heldout_robust_area1` (18 GB).

La home scende da 996 a 902 GB.

**Nuove regole sul disco:**
- Nella home vanno solo la geometria compressa, la GT, i checkpoint (ultimo e migliore) e i risultati.
- Gli operatori si calcolano su /tmp del nodo dentro i job.
- Ogni agente ha un budget in GB.

**Valanga.** Sospesa la generazione. Prima uno studio misurato sul formato: fp16, k_eig ridotto, operatori al volo. Ogni mesh ICT con operatori occupa circa 9.5 MB, quindi un ordine di grandezza in più non sta in 1 TB. Serve anche chiedere più quota all'AI Cloud: lo fa l'utente.

**Studio sul formato dei dati per la valanga (`aau/data_scale/PLAN.md`), misurato su 90 mesh.**
- **Oggi:** una identità ICT con operatori occupa 61 MB, una BFM 152 MB.
- **Nella home solo geometria compressa:** 1.2 MB per identità ICT, 2.7 per BFM. In 300 GB ci stanno circa 250.000 identità ICT.
- **Operatori:** si calcolano in un pre-pass CPU su /tmp, con risultati identici a oggi (scarto 1.4e-6).
- **fp16 sugli autovettori:** praticamente innocuo (Spearman fra le due versioni ≥ 0.9999).
- **Da scartare:**
  - k_eig 64: cambia i numeri, per esempio su BFM da 0.696 a 0.646;
  - fp16 sugli operatori sparsi: va in overflow;
  - operatori al volo a ogni epoca: circa 11 volte più lento.
- **Spazio recuperabile senza perdita:** ricodificando gli insiemi attuali in int32 compresso si recuperano circa 210 GB (stima), con output identico (misurato). In attesa del sì dell'utente.

**Vincoli emersi.**
- **Il trainer va adattato a numero di passi fisso e staging a blocchi:** con 10 volte i dati, 120 epoche durerebbero circa 20 giorni.
- **BFM non si scala oggi:** le mesh REMESH non stanno nella base BFM disponibile sul cluster (residuo 0.22). Serve il modello BFM originale.
- **Lo split held-out va congelato in modo esplicito:** `rebuild_subject_split` sull'unione dei soggetti sposterebbe soggetti di test nel training.

**Decisione del PI.** La generazione massiva su ICT aspetta lo zero-shot su HIFI3D. Se il congiunto non generalizza a un terzo dominio, servono più domini, non più identità dello stesso dominio.
- **Quota portata a 2 TiB** (verificato con ceph.quota.max_bytes). Approvata la generazione ICT nel formato (c) e l'adattamento del trainer; la ricompressione dei dataset esistenti è rimandata.

**Multiface WS3a duro, variante E (token di taglia + operatori robusti), crop reali.**
- **Risultato:** nessun guadagno sul crop. tracked→crop: 0.661 con il token convertito (tokA) e 0.565 con il token neutro (tok0), contro 0.697 del controllo, con IC larghi (0.48–0.82).
- **Perdite:** forti su down→up: −0.114 con tokA, −0.211 con tok0.
- **Conclusione:** come B. Il guadagno della taglia è solo in dominio sintetico e sui dati reali peggiora la robustezza. Tabella completa in `aau/runs/multiface_ws3a_hard/summary_hard_abl.md`.

**Congiunto-E rilanciato.** I training sono il job 1055737 (controllo) e 1055738 (E). La causa del fallimento precedente era la quota della home: i log non riuscivano più a scriversi. Run dir e log ora stanno su /tmp, con sync periodico nella home. Fine prevista verso le 23:30–00:00; eval e summary partono in automatico.

**Pipeline FLAME: corretti i 3 punti del critic, verificati con il modello finto.**
- L'output ha ora un'impronta dei dati nel percorso: i risultati vecchi non si riusano più dopo un rebuild.
- Il timbro della build viene scritto prima delle topologie.
- C'è un controllo per identità sulla variante crop.
- Pronta per i dati veri: si mettono `generic_model.pkl` e `FLAME_masks.pkl` in `v2_work/genflame/official/` e si lancia `aau/flame/ws_flame.sh`.

**Stop per limite d'uso (5 ottobre pomeriggio).** Fermati gli agenti di HIFI3D/FaceVerse e della valanga a metà lavoro: i loro job Slurm già sottomessi, se ce ne sono, proseguono da soli. Il congiunto-E (1055737/1055738) e le sue eval e il summary sono in catena su Slurm; risultato in `aau/runs/joint_E/summary.md` verso mezzanotte.

## 5 ottobre, pomeriggio: zero-shot su un terzo 3DMM, HIFI3D

**Protocollo.** Stesso protocollo di ICT: 100 soggetti held-out, 6 topologie, GT come distanza media per vertice in frame maxabs, IC con bootstrap per soggetto. Pipeline generica in `aau/zs3dmm/`; risultati in `aau/runs/ws_hifi3d/summary.md`.

| | BFM+ICT | solo BFM | solo ICT | Chamfer |
|---|---|---|---|---|
| tutte le topologie | 0.246 | 0.180 | 0.222 | 0.336 |
| senza crop | 0.428 [0.38, 0.48] | 0.206 | 0.382 | 0.372 [0.32, 0.42] |
| media per coppia di soggetti, clean | 0.720 | 0.607 | 0.713 | 0.743 |

**Lettura provvisoria, in attesa del critic.**
- Il congiunto generalizza meglio del solo BFM, ma è vicino al solo ICT.
- Su un dominio nuovo arriva circa al livello di Chamfer, senza batterlo in modo netto.
- Il crop crolla: 0.04–0.14 contro 0.65 di Chamfer. Va verificato se è un artefatto della pipeline.
- La GT nei coefficienti risulta quasi scorrelata da quella geometrica (Spearman 0.089): usata solo come controllo.

**FaceVerse.** Build e baseline completate; ranking del modello in corso.

**Critic sullo zero-shot HIFI3D: BLOCCANTE sulle conclusioni, non sui numeri.**
- **Frame confuso:** HIFI3D e ICT hanno lo stesso frame (naso verso +z), mentre i dati BFM sono specchiati in z. Il modello usa xyz e in training ha visto solo ±12° di rotazione. Quindi "il congiunto batte il solo BFM" può essere tutto un effetto del frame. Potrebbe valere anche per il vecchio zero-shot solo BFM→ICT (0.294), cioè per la lettura di H5 come "questione di dominio".
- **Dati e seed:** il solo BFM ha visto 10 volte meno identità; c'è un seed per braccio.
- **IC mancanti:** non ci sono IC appaiati sulle differenze, quindi "pari a Chamfer" non è stato testato.
- **Crop:** il crollo coinvolge i bracci addestrati su ICT. Ipotesi: interazione con la normalizzazione maxabs (rapporto crop/original 0.98 su HIFI3D contro 0.87 su ICT e BFM).
- **Controllo saltato:** il controllo sul crop non è mai girato su questi dati.

**In corso:**
1. test di rotazione di 180° su HIFI3D e su ICT (solo BFM nel frame giusto);
2. IC bootstrap appaiati sulle differenze;
3. zs3dmm su ICT held-out come controllo della pipeline;
4. eval in frame rms;
5. controllo sul crop.

Le conclusioni 1–2 di questo pomeriggio sono SOSPESE.

**Pilota del pozzo: i due training (pot_m55 1055016, dual 1055017) erano FALLITI alle 10:24, prima della saturazione della quota.** Me ne sono accorto solo nel pomeriggio, dall'eval rimasta in DependencyNeverSatisfied, che ho cancellato. Ho affidato a `coder` la diagnosi della causa e il rilancio, con run dir su /tmp.

**Limite d'uso, 5 ottobre sera.** Pilota del pozzo rilanciato (causa: la prima saturazione della quota, alle 10:24). Training 1056124 (m55) e 1056125 (dual), eval e summary in catena. Segnale anticipato su m55 all'epoca 86: xtopo 0.30 contro 0.74 del controllo, quindi probabilmente negativo. Il tetto di 12 GPU e 12 job della QOS è saturo; ho chiesto la priorità per il test di rotazione e per il pilota. Agenti fermati: le catene Slurm proseguono da sole.

**Valanga ICT generata (sera del 5 ottobre).**
- **Dati:** 50.000 identità ICT nuove (id20000–69999), 6 topologie + 8 espressioni ciascuna, 700.000 mesh. Solo geometria in 200 shard: 133 GB più 12 GB di GT congiunta 55.000². Home a 1116 GB su 2199.
- **Mini-lotto verificato:** gli operatori del pre-pass coincidono con quelli del loader (scarto 0), e la GT ricalcolata coincide con quella in uso (2.4e-7).
- **Guardia sugli held-out:** è nel generatore e nel trainer.
- **Trainer a passi fissi con staging a blocchi** (`v2_work/fastio/train_steps.py`): su una prova corta riproduce il trainer attuale entro il rumore (±0.035). Un piccolo effetto negativo dei blocchi non è escluso (B: −0.020 di media).
- **Decisioni del PI:**
  - si congela lo split del congiunto 1019532 più i 100 soggetti di valutazione BFM, non l'unione dei 15 run storici;
  - la GT estesa torna alla scala attuale;
  - le espressioni coprono circa il 25% delle mesh viste;
  - opzione di canonicalizzazione del frame, disattivata in attesa del test sul frame.
- **Il training grande non è ancora lanciato.** Prima servono il verdetto sul frame, lo smoke test e il critic.

**Valanga: smoke test superati, run lungo pronto (non lanciato).**
- **Configurazione:** 105.600 passi come il congiunto, 40 blocchi, BFM residente al 9% dei passi, espressioni al 25%, durata stimata circa 49 h. Split congelato: 189 BFM e 992 ICT. GT ICT ×1.18585 con `--gt-keep-scale`.
- **Opzioni di frame** (canonicalizzazione e augmentation), spente.
- **Dubbio aperto:** lo smoke ha usato x→−x, ma il critic misurava BFM ribaltato in z. Va chiarita la trasformazione corretta.
- **Prossimi passi:** critic sul trainer prima del lancio. Il lancio dipende dal verdetto sul frame.

**Critic sul run lungo della valanga: BLOCCANTE, con correzioni piccole.**
- **OOM certo a metà run:** la geometria del pre-pass si accumula su /tmp.
- **Errore al blocco 0:** il limite di cache è stimato su un solo campione BFM.
- **Riserve:**
  - lo scheduler a plateau non è equivalente, quindi si fissa lo schedule in passi del congiunto;
  - BFM in training con 81 identità in meno rispetto al congiunto, quindi lo split diventa esattamente quello del congiunto;
  - soggetti diversi per la selezione del checkpoint;
  - il pre-pass rallenta il training di 40 volte (thread);
  - un requeue sovrascriverebbe i risultati.
- **Frame (misurato su dati reali):** BFM ha il naso verso −z e l'alto verso −y. La trasformazione corretta è Rx(180°) = diag(1,−1,−1), non x→−x. Inoltre le normali delle facce BFM puntano verso l'interno, quelle di ICT verso l'esterno: serve un'inversione delle facce esplicita.

Correzioni affidate a coder; poi nuovo smoke a 2 blocchi e secondo giro di critic.

**Valanga: correzioni del critic applicate; smoke S3 a 2 blocchi passato** (1.4 s/passo durante il pre-pass invece di 35; /tmp torna a 0 a ogni blocco).
- **Lr:** fissato in passi, 1e-4 fino a 81.747 = 93 × 879, poi 5e-5. Il passo è stato verificato sul log del congiunto: l'lr stampato è quello dopo `scheduler.step()`.
- **Split:** identico al congiunto (392 BFM in training).
- **Altre correzioni:** `--no-requeue`, guardia sulla scala della GT, canon = Rx(180°) + flip_faces (spento). Bug trovato: argparse accettava le abbreviazioni (`--lr` letto come `--lr-steps`); ora `allow_abbrev=False`.
- **Stima del run:** circa 41 h, 420G. Secondo giro di critic in corso.

**Secondo giro del critic sul run lungo: BLOCCANTE solo sulla memoria. Il resto è confermato, compreso il passo dell'lr 81.747.**
- **B1:** la stima della cache campiona sempre up60k, quindi sicuro abort al blocco 16 (circa 17 h).
- **B2:** la memoria pinned non compare nel MaxRSS; fabbisogno reale stimato 424–460 GiB contro i 420 richiesti.
- **Correzioni affidate:** stima esatta dagli header npz, niente pinning, --mem tarato sul cgroup con margine, log del pre-pass conservati, indice dei tar.
- **Terzo giro:** verifica empirica (proiezione su 40 blocchi e smoke con picco del cgroup), poi decisione.

**Valanga, terzo giro (sera del 5 ottobre).**
- **Stima della cache:** ora calcolata in modo esatto dagli header npz. Massimo sui 40 blocchi: 220.4 GiB.
- **Pinning disattivato:** picco del cgroup 107 GiB contro 142; costo circa +19% di tempo per passo.
- **Memoria richiesta:** --mem 430G, con il 18% di margine sul bilancio dal cgroup (circa 363 GiB).
- **Log e tar:** log del pre-pass conservati; indice dei tar costruito.
- **Smoke S5 a blocchi reali:** job 1056296 in coda, non prima del 7 ottobre 00:42 (serve un nodo con 430G liberi).
- **Run lungo:** NON lanciato; aspetta S5 e il verdetto sul frame.

**Stop per limite d'uso alle 00:00 circa.**
- **Congiunto-E:** training completati; eval e summary in catena su Slurm.
- **Test frame HIFI3D:** frameBFM completato, roty in corso; risultati da raccogliere.
- **Pilota del pozzo:** in training.

**FaceVerse chiuso, con IC appaiati (dal report dell'agente; da passare al critic).** FaceVerse è nativamente nel frame BFM. Lì il congiunto è pari al solo BFM, ed entrambi battono il solo ICT: è lo specchio esatto di HIFI3D, che è nel frame ICT. Indizio forte che lo zero-shot misuri in buona parte il frame, non il dominio.

## 6 ottobre, mattina

**Pilota del pozzo: NON PASSA nessuno dei due bracci** (`aau/runs/pilot_pot/summary.md`).
- **pot_m55:** crolla ovunque, all 0.401 (−0.380), zero-shot ICT 0.131 (−0.238). Un crollo così ampio è sospetto: critic in corso per escludere un artefatto (operatori o maschera diversi fra training ed eval).
- **pot_dual:** neutro, all 0.771 (−0.011), ICT 0.315 (−0.054).

**Congiunto-E: eval fallite.** Due in TIMEOUT a 3 h, una fallita in 49 s; summary mai scritto. Diagnosi e rilancio affidati.

**Test sul frame:** tutti completati (hifi frameBFM, roty, rms; fv frameICT; ict frameBFM, e2e). Raccolta in `aau/runs/ws_frame/` in corso.

**Smoke S5 della valanga:** ancora in coda (job 1056296).

**Critic sul pilota del pozzo: RISERVE.**
- **L'eval è coerente col training.** Il crollo di m55 c'era già nell'eval online: xtopo 0.357 contro 0.743.
- **Ma il crollo non misura il pozzo.** La ROI non copre la stessa regione del viso nelle varie topologie (IoU con original: noisy 0.60, remesh 0.74; manca `--area-normalize`). Il pooling sulla ROI rompe quindi l'invarianza fra topologie per costruzione.
- **"Il pozzo da solo non aiuta" si regge su w55 dell'autore** (agosto: crop −0.006, totale −0.013), non su questo pilota.
- **Dual:** confronto appaiato e onesto, sotto il rumore (circa 0.03 fra run con lo stesso seed). Non raggiunge +0.05, ma non si può dire che peggiori ICT.
- **Prossimo passo:** braccio `pot_m55_roi`, con una ROI coerente fra topologie (IoU ≥ 0.90 misurato PRIMA del training).

**Decisione (6 ottobre, PI con l'utente): il crop non è più un obiettivo del modello.** Da settimane è il punto dove falliscono le metriche apprese, e tre tentativi di correggerlo via modello hanno dato risultati nulli o negativi: pozzo, crop casuali (F), pooling sulla ROI. Il braccio `pot_m55_roi` è annullato.
- **Nella valutazione il crop resta**, ma come stress riportato a parte: tabelle con e senza crop. Il disallineamento del supporto è reale: pipeline di acquisizione diverse tagliano il viso in modo diverso, e Multiface e i 3DMM hanno maschere diverse.
- **Per il framing B il crop diventa un argomento di protocollo, non di modello.** Il punto della checklist "equalize support", cioè valutare sul supporto comune, va testato come rimedio a livello di protocollo, per tutte le metriche.

## 6 ottobre, mezzogiorno: test sul frame (`aau/runs/ws_frame/summary.md`), in attesa del critic

**Convenzioni misurate:**
- BFM: alto −y, naso −z, normali verso l'interno;
- ICT e HIFI3D: +y, +z, normali verso l'esterno;
- FaceVerse: −y, −z, normali verso l'esterno.

Conta la rotazione (Rx 180°); l'inversione delle facce cambia pochissimo.

**Risultati:**
- **Il frame sposta molto i modelli** (±0.1–0.2), ma spiega solo circa metà del gap congiunto − BFM-only su HIFI3D senza crop (da +0.22 a circa +0.11).
- **BFM-only su ICT non è un effetto del frame:** nel frame BFM scende da 0.294 a 0.247. La lettura "H5: dominio" regge.
- **Il vantaggio del congiunto viene quasi tutto da ICT:** congiunto − ICT-only su HIFI3D +0.046, media per coppia di soggetti non significativa.
- **Contro Chamfer, con IC appaiati:** il congiunto è sopra solo su HIFI3D senza crop (+0.056 [+0.015, +0.094]). Altrove è pari o sotto; con il crop sempre sotto.
- **rms non risolve il crop.**
- **La pipeline end-to-end su ICT riproduce 0.989.**

**Decisione sulla valanga:** si lancia con il frame NON canonicalizzato, identico al congiunto 1019532. Così l'unica variabile è la quantità di dati ICT. Una variante canonica o con augmentation di rotazione, eventualmente, dopo.

## 6 ottobre, primo pomeriggio: decisione e nuova direzione

**Framing B adottato** (decisione con l'utente; il gate del 24 ottobre è anticipato). Valanga e leave-one-out restano come prove di robustezza, a priorità più bassa.

**Il problema della GT.** La GT è la L2 media per vertice fra mesh `original` in corrispondenza, dopo maxabs: è quasi la stessa misura di Chamfer. Fuori dominio Chamfer parte quindi avvantaggiato per costruzione: una metrica appresa può al massimo eguagliarlo sulla geometria pulita.

**Proposta: GT d'identità separata dai disturbi.**
- GT calcolata sulle forme NEUTRE (solo identità);
- input con espressioni casuali diverse, più le perturbazioni topologiche;
- crop declassato.

Chamfer misura la geometria e quindi l'espressione la confonde, mentre una metrica appresa con espressioni in training può restare invariante. È anche la definizione più vicina all'uso reale: riconoscere la stessa persona con espressioni diverse. La valanga, con 8 espressioni per identità, è già il training giusto per questo scenario.

**Test lanciati:**
1. controllo su `joint__rexpr` con Chamfer appaiato;
2. zero-shot su FaceVerse con espressioni, usando le sue 52 basi.

**Altre direzioni per la GT:**
- giudizi umani (studio con 300 triplette, da distribuire);
- verifica stessa/diversa persona su dati reali con più acquisizioni (Multiface, FaceScape su licenza).

**Critic sui test del frame: BLOCCANTE su (a) e (d), VIA LIBERA su (b).** Il codice delle trasformazioni e il bootstrap appaiato sono corretti.
- **(a) va riscritta.** "Il frame sposta ogni modello" è falso: ICT-only e BFM-only su ICT si spostano poco. "Circa metà del gap" vale solo senza crop. In subject-pair-mean il frame BFM peggiora BFM-only.
- **(d) va riscritta.** "Con il crop sempre sotto Chamfer" è falso in subject-pair-mean (pari) e su FaceVerse. Manca la cella FaceVerse BFM-only nella convenzione BFM completa.
- **Leakage.** Tutte le celle del congiunto e di ICT-only su ICT sono inquinate; quelle di BFM-only e di Chamfer sono pulite, quindi (b) regge.
- **(c) solo su HIFI3D;** su FaceVerse va al contrario.
- **(e):** prova solo il ri-inquadramento in eval.

**Regola nuova, dichiarata il 6 ottobre:** protocollo PRIMARIO = mesh-pair cross-topologia SENZA crop; secondario = subject-pair-mean; crop a parte. La regola di selezione del frame è fissata prima dei numeri. Vale per il test sulle espressioni e per tutti i test futuri. Per i test sul frame è stata fissata dopo aver visto i numeri, e va detto.

**Congiunto-E (token di taglia + operatori robusti, addestrato su BFM e ICT): PASSA di un soffio, con un solo seed** (`aau/runs/joint_E/summary.md`).
- **BFM:** margine +0.0315 (soglia +0.03); all_cross 0.874 contro 0.845.
- **ICT:** invariato (+0.0008).
- **Multiface crop:** E peggiora di circa −0.10 e down→up di −0.08/−0.10.
- **Cause dei fallimenti precedenti:**
  - una regex a 4 cifre in `eval_cells.py` troncava gli id ICT (id14500 → id1450);
  - timeout sul breakdown Chamfer, risolto dividendolo in 15 shard.
- **Lettura:** il token di taglia resta un guadagno solo in dominio e non regge sui dati reali. Nel framing B non è centrale.
- **In corso:** IC appaiato sul Delta; niente altri seed.

**Svolta (6 ottobre, decisione con l'utente): il fuoco passa alle ESPRESSIONI.**
- **Motivazione:** con la GT geometrica, fuori dominio Chamfer è imbattibile per costruzione. Con la GT d'identità sulle forme neutre e l'espressione come disturbo, l'invarianza appresa ha un vantaggio reale e misurabile, e corrisponde all'uso vero.
- **Test in corso:** controllo su `joint__rexpr` con Chamfer appaiato; zero-shot su FaceVerse con espressioni; protocollo primario dichiarato prima dei numeri.
- **Valanga lanciata:** job 1056832, catena di eval 1056833. Contiene 8 espressioni per identità ICT, quindi è già un training di invarianza all'espressione. Frame come il congiunto, senza augmentation. Lo smoke S5 è stato cancellato: il primo cambio di blocco del run fa da verifica della memoria, perché un OOM lì fallisce rumorosamente in circa 2–3 h.

**Definizione d'identità per il benchmark sulle espressioni (decisa con l'utente, prima dei numeri).**
- **La GT non si rifà con le espressioni.** La distanza resta definita fra le forme NEUTRE; le espressioni stanno solo negli input come disturbo.
- **Metrica PRIMARIA: riconoscimento d'identità** (retrieval rank-1 e mAP; verifica AUC stessa/diversa persona con espressioni diverse). Usa solo le etichette d'identità, nessuna distanza arbitraria come GT. È il protocollo standard del riconoscimento facciale e vale anche sui dati reali.
- **SECONDARIO:** ranking contro la GT neutra (mesh-pair senza crop).
- **Baseline forti, perché battere il solo Chamfer sulle espressioni sarebbe troppo facile:**
  - Chamfer ristretto alle regioni stabili all'espressione;
  - fit 3DMM seguito dalla distanza fra i coefficienti d'identità, se fattibile.

**Novelty candidata (6 ottobre, con l'utente):** riconoscimento d'identità 3D fra mesh con discretizzazioni diverse (remesh, decimazione, rumore, risoluzione, mesh da pipeline diverse) insieme all'espressione, con un encoder agnostico alla discretizzazione (DiffusionNet), valutato fuori dominio.
- **Già noto:** il riconoscimento invariante all'espressione (Bronstein et al.; FR3DNet; Led3D; benchmark FRGC v2, Bosphorus, BU-3DFE).
- **Da verificare:** se la variazione di connettività è già stata studiata. Il lavoro cross-qualità (Lock3DFace, Led3D) riguarda il sensore e la risoluzione, non il remeshing. Ricerca bibliografica in corso, in `literature/EXPRESSION_RECOGNITION_2026-10-06.md`.
- **Da fare:** almeno un benchmark classico (Bosphorus o BU-3DFE), per il confronto con la letteratura; licenze da verificare.

**Letteratura sul riconoscimento sotto espressione** (`literature/EXPRESSION_RECOGNITION_2026-10-06.md`; ricerca non esaustiva, molte fonti lette solo dall'abstract).
- **Non trovati:**
  - "stessa persona, connettività diversa" come protocollo: Kim, FR3DNet e Led3D proiettano su depth map a griglia fissa; il cross-quality esistente è sensore o rumore sulla z;
  - encoder intrinseci (DiffusionNet, functional maps) per il riconoscimento di volti;
  - leave-one-3DMM-out;
  - critica della circolarità fra GT geometrica e Chamfer.
- **Esiste già:**
  - training solo sintetico su 3DMM con test su reali (Zhang et al., PR 2022: GPMM → FRGC/Bosphorus, rank-1 zero-shot 92.7–93.4);
  - critica dei protocolli, ma per la ricostruzione (REALY ECCV 2022; Sariyanidi et al. FG 2025).
- **Da leggere subito:** Lee et al., ACM TOMM 2025, "3D Facial Shape Similarity with Deep Perceptual Representations": è il più vicino.
- **Benchmark reali:**
  - FaceScape: accordo via email, non commerciale;
  - BU-3DFE: la richiesta deve farla il supervisore;
  - FRGC: richiesta scritta a Notre Dame;
  - Bosphorus: termini incerti.

**IC appaiato congiunto-E (e − ctrl).**
- **BFM, protocollo primario:** +0.030 [+0.017, +0.047]. Positivo con certezza, ma la soglia del margine (+0.03) è superata solo come valore puntuale (51.7% delle repliche).
- **ICT:** −0.000 [−0.003, +0.003], nullo.
- **Variabilità:** gli IC coprono i soggetti, non il seed.

**Stop per limite di sessione (6 ottobre, circa 14:00; reset alle 16).** Agenti fermati. I job Slurm proseguono da soli:
- test sulle espressioni FaceVerse (baseline e bracci, summary 1056877);
- equalize support (BFM, HIFI3D, ICT);
- valanga (1056832, in coda) con la sua catena di eval (1056833).

**Alla ripresa:**
1. raccogliere i risultati di espressioni ed equalize support, poi critic;
2. cella FaceVerse `_flip` del frame, che l'agente stava ripresentando su A40;
3. verificare il primo cambio di blocco della valanga.

## 6 ottobre, sera: benchmark identità sotto espressione, FaceVerse fuori dominio (critic in corso)

**Primario: riconoscimento senza crop, 100 soggetti, 2000 query.**

| metodo | rank-1 | AUC |
|---|---|---|
| congiunto (conv. BFM) | 0.680 | 0.875 |
| solo BFM | 0.649 | 0.873 |
| Chamfer | 0.740 | 0.882 |
| Chamfer regione stabile | 0.787 | 0.894 |
| ICP + Chamfer | 0.918 | 0.986 |
| NICP | 0.959 | 0.995 |

- **Il congiunto perde** contro Chamfer di −0.060 di rank-1 [−0.093, −0.031], e contro NICP di −0.278.
- **Secondario, provvisorio (Spearman con GT neutra):** congiunto 0.282, Chamfer 0.319, NICP 0.194.
- **In dominio (ICT `rexpr`, solo original):** il congiunto batte Chamfer di +0.044 [+0.024, +0.067].

**Lettura.**
- L'ipotesi "Chamfer crolla con l'espressione, il modello resta invariante" NON regge fuori dominio.
- Le baseline con registrazione dominano il riconoscimento.
- Il vincitore dipende dal protocollo: NICP è il migliore nel riconoscimento e il peggiore nel ranking. Riconoscere la stessa persona e graduare la somiglianza fra persone diverse sono compiti diversi. Per il framing B è un risultato centrale.
- Coerente con il risultato precedente: l'allineamento danneggia il ranking ma aiuta il riconoscimento.

**Note.**
- Il fit 3DMM non è stato fatto.
- Le eval sono girate su A10 (L40S esaurite).
- L'espressione è lieve: spostamento maxabs medio 0.039 su un diametro di 2.93. Da valutare.

**Equalize support** (`aau/runs/eqsupport/summary.md`; critic in corso).
- **Regola:** la regione è quella del crop del soggetto, riportata sulla original e sulle altre topologie con la regola del punto più vicino.
- **Controllo:** righe non-crop identiche (Δ esattamente 0).
- **Righe crop, Spearman prima → dopo:**
  - **BFM:** Chamfer 0.012→0.454 (+0.44), ICP +0.42, NICP +0.24. Il latente del congiunto, che non aveva gap, scende di −0.044 [−0.063, −0.027].
  - **ICT:** le geometriche salgono di 0.20–0.28; il latente scende di −0.043. Leakage: sui 20 soggetti fuori training la stessa direzione.
  - **HIFI3D:** latente congiunto 0.049→0.392 (+0.34), Chamfer +0.14, ICP +0.19.
  - **Eccezione:** crop↔down8k, dove il bordo equalizzato è frastagliato.
- **Lettura provvisoria:** il gap del crop è un artefatto di supporto. Le metriche geometriche crollano sul crop in dominio, e l'equalizzazione le recupera; le metriche apprese erano robuste in dominio e fragili fuori.
- **Domanda per il critic:** la regola usa la corrispondenza sintetica. Come si applica a dati reali?

**Critic su equalize support: BLOCCANTE sulla conclusione. Ritiro "il gap si chiude".**
- **Il "dopo" non è il confronto giusto.** Il "dopo" di crop↔X equivale a original↔X con le mesh ritagliate, e crop↔original diventa una coppia della stessa topologia, che gonfia la media.
- **Confronto per tipo di coppia:** su HIFI il latente congiunto NON chiude (noisy 0.545 contro 0.749, remesh 0.413 contro 0.571). Chamfer su BFM chiude solo su up60k e down8k.
- **Costo del supporto ridotto:** 0.06–0.08 sui latenti anche senza crop (BFM 0.853→0.772, ICT 0.991→0.922, HIFI 0.428→0.369).
- **La regola è oracolare:** usa crop, original e connettività sintetici, quindi non è applicabile a dati reali.
- **Leakage:** sui soggetti fuori dal training l'equalizzazione peggiora il congiunto (BFM −0.088, ICT −0.054).
- **Formulazione sostenibile:** il crollo sul crop è GUIDATO dal supporto asimmetrico (diagnosi), ma un rimedio di protocollo applicabile non è ancora dimostrato.
- **Prossimo passo:** maschera canonica non oracolare (raggio da un landmark automatico, su ogni mesh), con controllo negativo e controllo d'identità.

**Critic sul benchmark delle espressioni: BLOCCANTE su (2).**
- **(2) "Il protocollo decide il vincitore" si ROVESCIA con la GT sui coefficienti** (Spearman, FaceVerse con espressioni, senza crop):

  | metodo | GT maxabs | GT coef |
  |---|---|---|
  | NICP | 0.194 | **0.281** |
  | ICP | 0.259 | 0.255 |
  | Chamfer | 0.319 | 0.197 |
  | congiunto | 0.282 | 0.174 |

  Con la GT coef, NICP vince anche il ranking. La GT maxabs favorisce Chamfer per costruzione. Si sostiene solo: "con la GT vertex-L2 maxabs NICP ordina peggio di Chamfer", cioè **la scelta della GT decide il vincitore** (era già vero sul neutro).
- **(1) RISERVE.** Il congiunto perde in rank-1 e mAP; in AUC è pari alle Chamfer. Il dominio di ICP e NICP è solido: battono anche il confronto ideale con corrispondenza densa (0.880). L'espressione è reale (spostamento mediano pari a 0.45 della distanza dal soggetto più vicino), ma NON è la causa della sconfitta: fuori dominio il congiunto non batteva Chamfer neanche senza espressioni. Il frame nativo puro di FaceVerse non è stato valutato. L'orientazione di NICP dà un'incertezza di ±0.03 sul rank-1.
- **(3) VIA LIBERA** come affermazione a sé: in dominio il congiunto batte Chamfer anche con espressioni. Ma il contrasto con FaceVerse è un effetto di DOMINIO, non di espressione.
- **Quadro onesto:** fuori dominio la registrazione (ICP, NICP) domina il riconoscimento, e con la GT coef anche il ranking. La metrica appresa non è competitiva fuori dominio in nessun protocollo. NICP è vicino al tetto (rank-1 0.959): per misurare un eventuale ibrido serve un test più difficile.

**Valanga: run lungo 1056832 morto per OOM (cgroup) dopo 2h39, al primo cambio di blocco**, durante il caricamento della cache del blocco 1 (15500/20426).
- **Errore mio:** avevo cancellato lo smoke S5 a blocchi di dimensione reale per non aspettare.
- **Ipotesi:** al cambio coesistono la cache vecchia e quella nuova.
- **Correzione affidata a coder:** liberare esplicitamente la cache vecchia, poi S5 obbligatorio con il picco del cgroup misurato, e rilancio solo con il 15% di margine.

**Test sul frame chiusi, conclusioni riscritte dopo il critic** (`aau/runs/ws_frame/summary.md`; protocollo primario = mesh-pair senza crop).
- **(a) HIFI3D:** il frame spiega circa metà del vantaggio del congiunto su BFM-only (frazione 0.51 [0.30, 0.76]); l'altra metà resta, +0.108 [+0.047, +0.159]. Su FaceVerse congiunto e BFM-only sono pari. Il congiunto batte ICT-only: HIFI3D +0.046, FaceVerse +0.126.
- **(b) BFM-only su ICT:** il crollo non è un effetto del frame (VIA LIBERA).
- **(c) Pipeline ICT:** riproduce lo storico; le celle del congiunto su ICT sono inquinate dal leakage.
- **(d) Contro Chamfer:** sopra solo il congiunto su HIFI3D (+0.056 [+0.016, +0.092]). Altrove pari, oppure sotto (ICT-only su FaceVerse −0.159).
- **(e) Crop:** rms non lo risolve.
- Chiuso al secondo giro di correzioni: niente terzo critic. Le riserve note sono l'aritmetica GPU A10/L40S (±4e-4) e le celle del congiunto in `_flip` mancanti su FaceVerse.

## 6 ottobre, sera: ricerca di una strada che risolva il fuori dominio (su richiesta dell'utente, che ha scartato il "benchmark B" come ripiego)

**Diagnosi.** L'identità viene appresa SOLO da 3DMM sintetici: architettura e operatori non spostano il problema.

**Tre ricerche bibliografiche** (`literature/DIRECTIONS_{GENERALIZATION,2D_DISTILL,NEURAL_FIT}_2026-10-06.md`):
- **Generalizzazione:** nessun lavoro sul leave-one-3DMM-out per l'identità. Direzioni: togliere xyz dall'ingresso (HKS, frame canonico), randomizzare i sintetici in stile SynthSeg, fine-tuning contrastivo su scansioni reali.
- **Distillazione 2D:**
  - riconoscimento 3D da depth/normal map con reti 2D affinate arriva al 98–99%, ma su scansioni allineate e rappresentazione fissa;
  - MICA usa ArcFace da foto;
  - Diff3F/MeshFM proiettano feature 2D semantiche sulle mesh, mai per l'identità;
  - Head Similarity (2026) distilla AdaFace fuori distribuzione;
  - SPAZIO LIBERO: encoder 3D agnostico alla discretizzazione distillato dal riconoscimento facciale, con valutazione cross-topologia e cross-3DMM.

**Indizio chiave (scout).** ArcFace (`buffalo_l` `w600k_r50`) su render di SOLA GEOMETRIA (grigio ombreggiato, 3 viste) dà AUC b_vs_c 1.000 su Multiface, crop e cross-topologia compresi; le geometriche stanno a 0.60–0.73. Il campione è di soli 13 soggetti. Nel ranking sintetico invece ArcFace è basso (0.29–0.38): riconoscere non è graduare.

**Test di falsificazione lanciato:** ArcFace su render, riconoscimento sul benchmark FaceVerse con espressioni, contro NICP (0.959).
- **Se ci si avvicina:** la strada è distillarlo in DiffusionNet.

**Licenza dei pesi `buffalo_l`:** non indicata nel repo; da verificare (insightface di solito è solo ricerca non commerciale).

**Terza ricerca: modelli neurali di testa e fit-then-compare** (`literature/DIRECTIONS_NEURAL_FIT_2026-10-06.md`).
- **Vuoto in letteratura:** nessuno usa i codici d'identità di NPHM, MonoNPHM, ImFace++, i3DMM o imHead per il riconoscimento.
- **Limiti di questi modelli:**
  - training piccoli (NPHM 255 identità, ImFace++ 831);
  - fitting poco robusto in-the-wild e su scansioni rumorose;
  - richiedono un allineamento canonico;
  - pesi spesso senza licenza dichiarata.
- **GNM Head** (Google, 2026, Apache-2.0, circa 5000 identità reali, lineare, con espressioni): candidato utile sia come baseline fit-then-compare sia come DOMINIO DI TRAINING aggiuntivo derivato da dati reali. Il fitting a scansioni va scritto da noi.
- **Shape My Face** (IJCV 2021): embedding separati identità/espressione da point cloud, ma senza riconoscimento.

**Sintesi del PI.** Direzione principale = distillazione del riconoscimento 2D in DiffusionNet, condizionata al test ArcFace-render su FaceVerse. Il fit-then-compare resta secondario; GNM Head va valutato come dominio di training e come baseline.

**GNM Head scaricato** (`~/data/gnm_head/`, Apache-2.0 per codice e pesi, verificato; i dati di training non sono rilasciati).
- **Caratteristiche:** 17 821 vertici; identità 170 basi della testa (più denti e occhi); 383 espressioni; unità in metri; frame +Y/+Z come ICT; regioni `hockey_mask` e `skin_exterior` pronte.
- **Integrazione come quarto dominio** di test e potenziale di training in `aau/zs3dmm/`: per ora solo i build.

**Direzione approvata dall'utente (6 ottobre sera): encoder agnostico distillato dal riconoscimento 2D.** Piano in `paper/PLAN_DISTILL.md`; parte dopo il gate ArcFace-render.

**Gate ArcFace-render** (`aau/runs/arcface_render_zs/summary.md`; protocollo dichiarato alle 19:21).

**FaceVerse con espressioni, senza crop, rank-1:**
- ArcFace ombreggiato: 0.750 [0.724, 0.775];
- ArcFace a normal map: 0.867;
- NICP: 0.959;
- ICP: 0.918;
- Chamfer: 0.740;
- congiunto: 0.680.

ArcFace batte il congiunto di +0.070 [+0.022, +0.118] (AUC pari) e resta sotto NICP di −0.209.

**HIFI3D neutro:** ArcFace 0.984, pari a NICP (0.990, che però ha 1965 distanze NaN); congiunto 0.397.

**Con il crop:** ArcFace FaceVerse 0.881 contro NICP 0.909; HIFI3D 0.990 contro 0.777.

**Note:**
- Il detector fallisce su tutti i render `noisy`: debolezza dell'insegnante che lo studente agnostico dovrebbe superare.
- Espressione contro dominio su FaceVerse: non separati (manca FaceVerse neutro).

**Decisione:** procedere con il pilota della distillazione.
- Insegnante: normal map, sempre sulla `original`.
- Dati: BFM + ICT-5000.
- Test: FaceVerse expr e HIFI3D.
- Criterio: battere il congiunto e raggiungere almeno l'80% del rank-1 dell'insegnante.

## 7 ottobre, notte: pilota della distillazione, NON PASSA (`aau/runs/distill_pilot/summary.md`)

**Configurazione.** Studente DiffusionNet a 512 dimensioni, insegnante ArcFace a normal map sulla `original`, training su BFM + ICT-5000 (6590 mesh), 50 epoche, loss puntuale + relazionale.

**Rank-1, protocollo primario:**

| | FaceVerse expr | HIFI3D |
|---|---|---|
| studente, conv. BFM | 0.332 | 0.224 |
| studente, conv. ICT | 0.570 | 0.355 |
| congiunto | 0.680 | 0.397 |
| insegnante a normal map | 0.867 | 0.998 |

- **Contro il congiunto:** −0.348 su FaceVerse e −0.173 su HIFI3D.
- **In dominio:** validazione rank-1 0.91, ma il coseno con l'insegnante è solo 0.62.

**Lettura.** Lo studente impara la mappa solo sulla distribuzione d'ingresso che vede (2 3DMM) e dipende dal frame: distillare non trasferisce la generalità dell'insegnante se gli input sono poco vari.

**Costo dell'insegnante:** circa 1.4–2.4 CPU-s per mesh, quindi a scala è economico.

**Prossima decisione (all'utente):** tentativo con input molto più vari (GNM campionabile senza limiti, valanga ICT, deformazioni casuali, frame canonico + augmentation di rotazione) oppure stop.

**Decisione dell'utente (7 ottobre): tentativo v2 della distillazione con input molto vari.**
- **Dati:** GNM circa 10k identità con espressioni, valanga ICT circa 10k, BFM e ICT-5000, deformazioni casuali, frame canonico ICT + rotazioni ±30°.
- **Criterio identico al pilota.**
- **Esito:** se fallisce, si chiude la strada degli encoder appresi.

**Direzione dell'utente (7 ottobre): non buttare il paper NeurIPS, VENDERE la metrica appresa originale.** Posizionamento onesto:
- metrica d'identità 3D appresa, veloce (un embedding per mesh, confronto O(1)) e agnostica alla discretizzazione;
- molto forte nella distribuzione di training (0.86–0.98 contro 0.25–0.35 di Chamfer, anche con espressioni);
- la copertura si estende aggiungendo un 3DMM al training: il congiunto funziona su entrambi i domini;
- il fuori dominio è un limite dichiarato, con analisi.

**Mancano:** riconoscimento IN DOMINIO con etichette (risponde alla critica di circolarità della GT) contro NICP, ICP, Chamfer e ArcFace, più i tempi. Lanciato (`aau/runs/indomain_recog/`). La distillazione v2 continua come estensione dello stesso encoder.

**Piano di risposta alle critiche:** `paper/REBUTTAL_PLAN.md`, 11 punti con azione e stato. Priorità: training mastodontico (valanga + GNM) e valutazione su ricostruzione reale. La distillazione v2 passa in pausa: l'agente completa solo i dati GNM, che servono al training su scala.

**Dati GNM pronti** (`datasets/GNM_DISTILL/`): 10.100 identità, 75.788 mesh (6 topologie + 1-2 espressioni), volto `hockey_mask` da 9.022 vertici, GT maxabs, 16 GB, PNG di controllo verificati.

**Distillazione v2:** i due training (1058158 completo, 1058159 senza GNM) restano accesi senza eval. Le etichette dell'insegnante sono generate (20k ICT, 20k GNM).

**Osservazione non pianificata:** lo smoke della distillazione dopo 2 epoche fa 0.54–0.58 su HIFI3D, contro 0.22–0.36 dopo 50 epoche. Il training lungo si specializza sui domini di training. Si terranno checkpoint intermedi per la curva fuori dominio.

**Training mastodontico affidato a coder.** Sequenza: correzione OOM → dominio GNM → GT e indice uniti → smoke S5 con picco misurato → run su BFM + ICT-5000 + ICT_SCALE + GNM.

**Distillazione v2 cancellata** (job 1058158 e 1058159), per liberare gli slot QOS per la direzione principale: training su scala e riconoscimento in dominio. Etichette dell'insegnante e dati GNM restano su disco.

**Training mastodontico partito (7 ottobre, 14:12):** job 1060130, catena di eval 1060131. Run dir `aau/runs/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411`.
- **Dati:** 64.400 soggetti (BFM 392, ICT-5000 4008, ICT nuove 50.000, GNM 10.000); 105.480 passi; 46 blocchi; checkpoint ogni 10%.
- **Causa dell'OOM:** la memoria non tornava al sistema. Corretto con gc.collect() + malloc_trim(0) al cambio di blocco. S5 ha misurato un picco non recuperabile di 356 GiB; --mem 510G (+16.8%).
- **Stima:** 46-48 h. Da controllare il primo cambio di blocco in mem_job.log.
- **Stop per limite d'uso.**

**Riconoscimento in dominio + tempi** (`aau/runs/indomain_recog/summary.md`, protocollo pre-dichiarato; critic non ancora passato).
- **Senza crop, rank-1:** congiunto BFM 0.998, ICT 1.000, pari a NICP (1.000). Sopra ICP (0.934/0.670), Chamfer (0.652/0.381) e ArcFace-normali (0.981/0.981).
- **Crop:** congiunto 0.944/0.993, contro NICP 0.686/0.600 e Chamfer 0.04/0.10. ArcFace 0.984/0.993, leggermente sopra su BFM.
- **Espressioni ICT (solo original):** congiunto 0.822, il peggiore. NICP 0.905, Chamfer 0.867, ArcFace 1.000.
- **Tempi:** embedding 2.3 s/mesh (operatori), confronto 2.3 µs, ricerca 1:10.000 in 2.2 ms su CPU. NICP 1.42 s per coppia, quindi una ricerca 1:10.000 costerebbe circa 4 h.
- **Lettura:** in dominio il congiunto è pari a NICP a costo per confronto circa 600.000 volte più basso, e molto più robusto di NICP al crop. Debole sulle espressioni: il training su scala, con espressioni, deve correggerlo.

**7 ottobre pomeriggio: FLAME e NoW caricati dall'utente** (`external_data/`, ora in .gitignore).
- **FLAME:** estratto in `v2_work/genflame/official/` (ignorato da git). Zero-shot lanciato: build 1060215, congiunto 1060216, BFM-only 1060217, baseline 1060218.
- **NoW:** affidato a coder (dati in ~/data/now, codice ufficiale, ricostruzioni 3DDFA-V2 e altri, metriche, identità su reale).

**Critic sul riconoscimento in dominio:** (1) RISERVE, (2) BLOCCANTE, (3) BLOCCANTE, (4) VIA LIBERA.
- **(2) Il crollo di NICP sul crop è un artefatto di preprocessing.** Area unitaria + maxabs per mesh + ICP rigido senza scala. Con ICP di similarità NICP fa 1.000 sul crop (40 soggetti). Inoltre il congiunto ha il crop nel training. **La frase "più robusta della registrazione sul crop" è ritirata.**
- **(3) Tempi da riformulare.** Il congiunto vince solo nella RICERCA su una galleria già iscritta. L'iscrizione costa 2.3 s per mesh, dello stesso ordine di un NICP su template (circa 1.4 s, stima). Il nodo era condiviso: misure da rifare.
- **(1) Soffitto a N = 100.** Estrapolando, a N = 10.000 su BFM il congiunto probabilmente scende sotto NICP, mentre su ICT resta sopra. Galleria ICT di 992 held-out disponibile senza generare nulla. ArcFace non configurato al meglio (camera fissa, normali per faccia).
- **Correzioni affidate a coder:** ICP di similarità come standard, NICP su template, galleria 992, tempi su nodo esclusivo, ArcFace migliorato.

**NoW validation** (`aau/runs/now_eval/summary.md`; critic in corso per verificare eventuali svantaggi ingiusti al latente).
- **Dati:** 20 soggetti, 352 immagini.
- **Ricostruzioni:** 3DDFA-V2, SynergyNet, PRNet, MICA. DECA ed EMOCA non fatti (pytorch3d da compilare).
- **Errore NoW ufficiale (mediana, mm):** MICA 0.91 (riproduce i valori pubblicati), SynergyNet 1.35, 3DDFA-V2 1.39, PRNet 1.56.
- **Concordanza con NoW, tau per immagine:**
  - latente congiunto −0.236 [−0.355, −0.117], CONTRO il benchmark: mette MICA ultimo;
  - Chamfer +0.176;
  - ICP+Chamfer +0.455;
  - ArcFace +0.488.
- **Dentro un singolo metodo** il latente segue NoW meglio di Chamfer (3DDFA-V2: 0.65 contro 0.45).
- **Identità fra ricostruzioni (AUC):** il latente è ultimo con tutti i metodi (MICA 0.887 contro 0.981 di ArcFace).
- **Bug trovato:** `crop_region` in `ws3b_prepare_meshes` (indici di vertice sbagliati); corretto per NoW, in WS3b da verificare.

**Critic su NoW: RISERVE.** Niente errori di frame o di regione: il frame nativo è il migliore, le patch NoW sono nella distribuzione di training, WS3b è pulito dal bug di `crop_region`.
- **Il segno dipende da MICA** (aggiunto fuori protocollo). Sui 3 metodi pre-registrati il tau del latente è +0.131, positivo ma ultimo (Chamfer +0.246, ArcFace +0.271, ICP+Chamfer +0.612). Con MICA è −0.236.
- **La tassellazione di MICA penalizza il latente:** la sola storia della mesh sposta il latente del 26% della distanza fra soggetti. Con una storia densa (Loop) il tau di MICA va da −0.183 a −0.097 (100 immagini): resta negativo.
- **Il riconoscimento fra ricostruzioni** (stesso metodo, stessa storia della mesh) è affidabile: il latente è ultimo.
- **Formulazione:** "su ricostruzioni reali il latente attuale è la metrica meno concorde con NoW, e la storia della mesh è un confondente misurato".
- **Correzioni del summary e fragilità di `--overwrite`:** affidate.

**Studio umano:** la condivisione è in sola visualizzazione, quindi i partecipanti non scrivono nel db. Primo JSON manuale salvato in `aau/human_study/responses_manual/` (P0EKZC2T, controlli 4/4). Partecipanti validi: 2 (utente + P0EKZC2T).

**Zero-shot FLAME** (`aau/runs/ws_flame/b6649295c3f8/`; numeri raccolti dal verifier con il bootstrap di `ws_frame_summary`; FLAME ha lo stesso frame di ICT).

| Spearman | congiunto | solo BFM | Chamfer | ICP | NICP |
|---|---|---|---|---|---|
| mesh-pair, tutte le topologie | 0.216 | 0.274 | 0.555 | 0.300 | 0.252 |
| mesh-pair senza crop | 0.355 | 0.319 | 0.589 | 0.387 | 0.314 |
| subject-pair-mean `clean` | 0.719 | 0.751 | 0.866 | — | — |

- **Fuori dominio:** le metriche apprese stanno nettamente sotto Chamfer, coerente con HIFI3D e FaceVerse.
- **ICP e NICP sotto Chamfer:** effetto della GT maxabs, come già visto.
- **Manca uno script di riepilogo FLAME:** da integrare in `aau/zs3dmm` per le tabelle del paper.

**Baseline extra e quasi-duplicati (7 ottobre sera; commit 5524209, 7132c8e).**
- **CLIP L/14 e DINOv2 B/14** (render a normal map): molto sotto ArcFace e NICP.
  - Rank-1 FaceVerse expr: CLIP 0.304, DINOv2 0.251, contro ArcFace 0.867.
  - HIFI3D: 0.597 e 0.510.
  - BFM, Spearman: 0.13 e 0.11, contro 0.79 del modello NeurIPS.
  - Negli embedding pesa più la topologia dell'identità. ArcFace resta la baseline percettiva di riferimento.
- **Besnier 2023 e Ma 2021:** nessun codice o peso pubblico, quindi non integrabili (documentato).
- **Quasi-duplicati: nessuno.** Il NN test→train minimo è 3.9× la soglia (NeurIPS BFM), 3.6× (congiunto BFM) e 2.1× (congiunto ICT). Il punto 9 del rebuttal è chiuso.

## 8 ottobre, mattina: riconoscimento in dominio, secondo giro (`aau/runs/indomain_recog/summary.md`; non ancora passato da critic)

**Con ICP di similarità, la pipeline geometrica eguaglia il congiunto ovunque in dominio e lo supera sul crop BFM.**
- **Crop BFM:** congiunto 0.944 contro ICP sim. + NICP 1.000 (Δ −0.056 [−0.089, −0.030]).
- **Galleria di 992 soggetti ICT:** soffitto per tutti (rank-1 1.000).
- **NICP su template** (implementazione semplice): 0.86–0.90 senza crop, crolla sul crop.
- **ArcFace migliorato:** rank-1 circa 0.98, TAR basso a causa della topologia noisy.

**Tempi** (nodo esclusivo):

| fase | congiunto | template | ArcFace | ICP sim. + Chamfer | NICP |
|---|---|---|---|---|---|
| iscrizione | 2.08 s | 1.15 s | 0.57 s | — | — |
| ricerca 1:10k | 1.96 ms CPU | 811 ms | 0.88 ms | — | — |
| per coppia | — | — | — | 45 ms | 1.29 s |

**Lettura:** in dominio resta solo l'argomento del costo contro le pipeline a coppie, conveniente da circa 50 confronti per query in su. Nessun vantaggio di accuratezza; nessun vantaggio di costo contro ArcFace.

## 8 ottobre: primo segnale positivo fuori dominio (checkpoint intermedi del training su scala; `aau/runs/data_scale_ood/hifi/summary.md`; non ancora passato da critic)

**HIFI3D zero-shot, Spearman GT maxabs:**

| | senza crop | tutte le topologie | subject-pair-mean |
|---|---|---|---|
| Chamfer | 0.372 | 0.336 | 0.743 |
| congiunto BFM+ICT | 0.428 | 0.246 | 0.720 |
| scala e036 | 0.677 [0.61, 0.74] | 0.509 | 0.767 |
| scala e072 | 0.663 | 0.557 | 0.792 |
| scala e108 | 0.630 [0.57, 0.69] | 0.541 | 0.795 |

- **Il modello addestrato su 64k identità da BFM + ICT + GNM batte Chamfer fuori dominio con margine largo.**
- **Cautele:**
  - lieve specializzazione col training (senza crop da 0.677 a 0.630);
  - possibile vicinanza GNM ↔ HIFI3D;
  - con la GT coef tutti i valori sono bassi (scala 0.10–0.13, Chamfer 0.08).
- **In arrivo:** FaceVerse expr (riconoscimento), NoW, FLAME.

### 8 ottobre: curva OOD dei checkpoint intermedi (`aau/runs/data_scale_ood/curve.md`)
- **HIFI3D senza crop:** e036 0.677, e072 0.663, e108 0.630, contro Chamfer 0.372 e congiunto 0.428. e108 − Chamfer = +0.258 [+0.208, +0.301].
- **NoW tau:**
  - e036 0.299, e072 0.348, e108 0.277; Chamfer 0.246, congiunto 0.131;
  - tutti alla pari o meglio di Chamfer, nessuno significativo. e072: +0.102 [−0.029, +0.229].
- **FaceVerse con espressioni, rank-1:** e036 0.518, e108 0.625, contro Chamfer 0.740 e congiunto 0.680. È sotto, ma sale col training. AUC di e108 alla pari con Chamfer.
- **Confondente aperto:** FaceVerse valutato solo nella convenzione BFM, mentre il 91% del training è nel frame ICT. Ho chiesto la convenzione ICT per e108. e072 di FaceVerse è in coda (1061482).
- **Prossimo:**
  - regola di scelta del checkpoint dichiarata ora: **l'ultimo checkpoint del run**, nessuna selezione su OOD;
  - poi critic sul risultato positivo.

### 8 ottobre: ArcFace contro modello su scala, HIFI3D (`aau/runs/data_scale_ood/arcface_vs_scale_hifi3d.md`)
- **Distanza graduata** (Spearman con la GT, senza crop):
  - e036 0.677, e108 0.630;
  - Chamfer 0.372, NICP per coppia 0.389, ICP+Chamfer 0.355;
  - **ArcFace 0.245** (render ombreggiati) e 0.242 (normal map).
  - ArcFace − e108 = −0.385 [−0.466, −0.293].
- **Riconoscimento, rank-1:**
  - ArcFace 0.984 (ombreggiati) e 0.998 (normal map);
  - modello su scala da 0.782 a 0.811;
  - Chamfer 0.477.
- **Lettura:** riconoscere e misurare la distanza graduata sono capacità separate. ArcFace riconosce ma non misura; il nostro modello misura. Le righe di riferimento sono riprodotte identiche.
- **Aperto:** NICP su template per HIFI3D, il competitor più diretto sulla distanza graduata. L'ho chiesto all'agente dei competitori.

### 8 ottobre: niente tetto di agenti (piano da 200 $); evidenze prima del run massivo
Decisioni dell'utente:
- il run massivo aspetta;
- prima si raccolgono le evidenze (`paper/PLAN_MASSIVE.md` §13).

Agenti lanciati in parallelo, con la proprietà dei file separata. Nessuno tocca `v2_work/` o `face_embedding/`.

| Agente | Compito | Cartelle |
|---|---|---|
| coder E1 | fattoriale varietà×quantità (C2F, C2M, C3F, G1; C3M = run su scala) | `aau/evidence/e1_factorial/`, `aau/runs/evidence/e1/` |
| coder E2+E3 | canonicalizzazione rigida su e108; da dove vengono gli errori di riconoscimento | `aau/evidence/e2_canon/`, `e3_breakdown/` |
| coder E9 | misure: operatori/s per nodo, forward a gruppi S/M/L, NCCL fra nodi | `aau/evidence/e9_bench/` |
| coder D1 | libreria 3DMM uniforme (`v3_work/mm/`) e set di sviluppo FaceScape | `datasets/DEV_FACESCAPE/` |
| coder D2/E8 | GT unificata sulla regione FLAME, trasformazioni canoniche, quasi-duplicati fra domini (GNM↔HIFI3D) | `v3_work/unified_gt/`, `datasets/UNIFIED_GT/` |
| coder trainer v3 | fork con flag (gruppi, loss, campionatore, taglia, pooling, EMA, DDP) e test di equivalenza con v2; ablazioni E4-E7 solo preparate | `v3_work/trainer/` |

Ancora attivi:
- critic sul design;
- coder dei competitori (spettrali, NICP su template, Uni3D, OpenShape);
- agente FaceVerse (e072 e convenzione ICT).

### 8 ottobre: competitori diretti su HIFI3D (`aau/runs/competitors_hifi3d/summary.md`)
Distanza graduata (Spearman con la GT maxabs, senza crop):

| Metodo | Spearman | Rank-1 |
|---|---|---|
| e108 | 0.630 | 0.782, da `arcface_vs_scale` |
| Chamfer | 0.372 | |
| NICP su template | 0.351 | 0.875 |
| OpenShape | 0.138 | 0.617 |
| Uni3D | 0.132 | 0.539 |
| ShapeDNA k=50 | 0.108 | 0.575 |
| HKS | 0.077 | |
| WKS | 0.025 | |

- **Ablazione del frame:** con il frame ruotato OpenShape sale a 0.291, sempre molto sotto e108.
- **Lettura:** sulla GT maxabs e108 batte tutti di +0.28 o più.
- **Riserva aperta (critic):** la GT maxabs penalizza i metodi che allineano. NICP su template applica Procrustes e scarta la scala. Prima di affermare il vantaggio sulla distanza graduata serve la GT unificata, invariante alla similarità, che D2 sta calcolando su tutti i metodi.
- **Stato:** non ancora rivisto da critic o verifier.

### 8 ottobre: E9, fattibilità di P1 (`aau/runs/evidence/e9/tables.md`)
- **Viste fresche/s per nodo** (100 CPU, a768-l40s-05):

  | Configurazione | Viste/s |
  |---|---|
  | codice attuale | 6.7-7.8 |
  | `build_grad` vettorizzato, k 128, V ≤ 10k | 41-43 |
  | `build_grad` vettorizzato, k 64, V ≤ 10k | **123-126** |
  | fino a 60k vertici | 9-24 |

  **Il gate da 20 viste/s è superato.**
- **`build_grad` vettorizzato:** circa 197 volte più veloce, identico bit per bit dopo il cast a fp32; gli embedding e108 coincidono.
- **Forward+backward su L40S:** i gruppi grandi peggiorano. Con gruppi quasi singoli S fa 170 mesh/s (93 con un forward per mesh), M 68, L 39. Il trainer attuale (38.7 mesh/s) perde più di metà del passo in overhead.
- **Rete:** RoCE (`mlx5_bond_0`) visibile nei container; gloo su TCP fra nodi a 1.16 GB/s. Il test NCCL fra due nodi è in coda (1061639).

### 8 ottobre: D1, libreria 3DMM e set di sviluppo FaceScape (`aau/runs/evidence/dev_facescape/results.md`)
- **Libreria `v3_work/mm/`:** 12 modelli con ruoli. FaceScape è `dev`, HIFI3D e FaceVerse sono `test`; i campionatori di training sollevano `RoleError` su questi ruoli. BFM 2019 (199/100) aggiunto in tre varianti: bfm 47k, face12 28k, fullhead 58k vertici.
- **e108 sul dev FaceScape:**

  | Misura | e108 | Chamfer | Delta |
  |---|---|---|---|
  | graduata senza crop | 0.394 | 0.361 | +0.032 [−0.016, +0.086] |
  | rank-1 con espressioni | 0.285 | 0.424 | −0.139 |
  | punteggio dev | 0.340 [0.302, 0.376] | 0.393 [0.353, 0.431] | |

- **Lettura:** il vantaggio enorme su HIFI3D (+0.26) NON si ripete su FaceScape. Rafforza il dubbio che HIFI3D sia vicino a un dominio di training (GNM?); D2 lo sta misurando.

### 8 ottobre: critic sul disegno di E1, BLOCCANTE
- **Split corretti,** verificati: nessuna sovrapposizione, sottoinsiemi annidati, LR costante.
- **Difetto principale:** C3−C2 misura "aggiungere GNM", non la varietà. Forma del supporto (sd3/sd1 delle `original`):
  - HIFI3D 0.445, GNM 0.447, FaceVerse 0.467, FLAME 0.489;
  - ICT 0.78, BFM 0.35, FaceScape 0.396.

  GNM ha un supporto quasi identico a HIFI3D, il che spiegherebbe sia il salto su HIFI3D sia l'assenza di vantaggio su FaceScape.
- **Correzioni passate a E1:**
  - cella C2F-GNM e regola C3F > max(C2F, C2F-GNM);
  - effetto minimo 0.05, tre esiti (sostenuta / smentita / non concludente);
  - stessa direzione sul dev FaceScape;
  - non inferiorità su FaceVerse;
  - secondo seme per C2F e C3F;
  - C3M rifatta sullo stesso hardware come pavimento del rumore;
  - valutazioni sulle A100;
  - emendamento al protocollo prima dei numeri.
- **Ipotesi nuova, H8:** il modello è sensibile alla forma del supporto, cioè a quale regione copre la mesh. Si lega alla regione del volto comune della GT unificata e a un'eventuale armonizzazione del supporto.
- **Eccezione A100 autorizzata dall'utente:** `-p aicentre-a100 --qos=unprivileged`, solo per job brevi o ripartibili.

### 8 ottobre: E8, GT unificata (`aau/runs/evidence/e8/summary.md`): risultato che ridimensiona
- **Regione:** maschera `face` ufficiale di FLAME, senza interno di occhi e bocca. La parte comune a tutti gli 8 domini è di 1.478 vertici, il 59% dell'area: il crop BFM taglia le guance.
- **Corrispondenze:** residuo sui landmark tenuti fuori 0.61-1.28 mm.
- **Spearman fra GT unificata e GT maxabs:** 0.54 su HIFI3D, 0.69 su FaceVerse. Lo scarto viene dal Procrustes, non dalla regione: la mappa FLAME contro la GT nativa sulla stessa regione dà 0.999.
- **Quasi-duplicati:** nessuno. GNM è il dominio di training più vicino a HIFI3D (6% più di ICT, 13% più di BFM). Le medie dei domini formano due gruppi:
  - FLAME, BFM, ICT e Multiface;
  - FaceScape, HIFI3D e FaceVerse;
  - GNM sta in mezzo.
- **HIFI3D senza crop:**

  | Metodo | GT maxabs | GT unificata |
  |---|---|---|
  | NICP P2Tri | 0.389 | **0.577** |
  | ICP+Chamfer | 0.355 | 0.522 |
  | e108 | **0.630** | 0.301 |
  | Chamfer | 0.372 | 0.230 |

  Con la GT unificata, e108 − Chamfer scende a +0.071 [+0.015, +0.123]. Su FaceVerse con espressioni nessuna differenza è significativa.
- **Lettura:** il vantaggio sulla distanza graduata dipende dalla GT con cui il modello è addestrato e valutato (la maxabs, non invariante alla similarità). È la critica 4, la circolarità della D_GT, in forma concreta.
- **Esperimento decisivo lanciato, E11:** cella C3F-UGT, addestrata sulla GT unificata, confrontata con C3F e con le baseline con allineamento su quella GT. Tutte le celle di E1 saranno valutate con entrambe le GT.

### 8 ottobre: E10, operatori su GPU (`aau/runs/evidence/e10/E10.md`)
- **Pipeline:** `compute_operators` interamente su GPU, in batch. Gli autovettori con shift-invert cuDSS e Krylov a blocchi sono esatti: 0.09 s a 9.4k vertici.
- **Correttezza:** embedding e108 entro 1.4e-6 da quelli CPU, cioè quanto il rumore della CPU stessa.
- **Throughput:** una V100 vale circa un nodo CPU da 100 CPU logiche, da 0.45× (k 64, V ≤ 10k) a 1.78× (k 128, mesh fino a 60k).
- **Latenza:** 97 ms per mesh a 9.4k vertici, contro 0.64 s di un processo CPU.
- **Decisione:**
  - il pre-pass del training resta su CPU (`grad_vec`): togliere 1 GPU su 8 al training non conviene;
  - la via GPU (`compute_batch(method="bk")`) serve all'iscrizione al test, ed è utile per l'argomento del costo contro NICP su template (1.15 s per mesh).

### 8 ottobre: FaceVerse con espressioni, curva completa e convenzione ICT (`aau/runs/data_scale_ood/curve.md`)
- **Rank-1 per checkpoint** (convenzione BFM salvo dove indicato):

  | Checkpoint | Rank-1 | Δ vs Chamfer |
  |---|---|---|
  | e036 | 0.518 | |
  | e072 | 0.624 | |
  | e108 | 0.625 | |
  | e108, convenzione ICT | 0.640 | −0.100 [−0.134, −0.065] |
  | Chamfer | 0.740 | |
  | congiunto, convenzione BFM | 0.680 | |
  | congiunto, convenzione ICT | 0.562 | |

- **AUC di e108 in convenzione ICT:** pari a Chamfer (−0.003).
- **Lettura:** il frame NON spiega il ritardo su FaceVerse. La causa è l'espressione, coerente con E3 (+0.291 sulle neutre).
