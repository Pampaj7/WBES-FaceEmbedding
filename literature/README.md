# Dati e letteratura per sbloccare l'analisi cross-topologia

Ricerca del 19 agosto 2026. Obiettivo: trovare **scan reali di volti** che permettano di
misurare similarità e ranking fra identità diverse rappresentate in topologie diverse.

---

## 1. Perché serve, in una riga

Il ground truth attuale è `d(i,j) = media sui vertici di ||V_i[v] − V_j[v]||` su mesh in
corrispondenza densa. Ha tre difetti che nessun esperimento sul modello può rimuovere:

1. **Non è identità, è geometria.** Nessuna etichetta biometrica entra mai nel pipeline.
2. **Dipende dalla topologia in cui lo misuri.** Misurato sulle stesse 500 identità in
   topologia BFM contro topologia FLAME, l'accordo è ρ = 0,8653 su coordinate grezze e
   **ρ = 0,5734 con la normalizzazione `maxabs` effettivamente usata**
   (`v2_work/xdomain/gt_matrices/agreement.json`).
3. **È circolare.** Il paper afferma che la metrica appresa preserva il ranking d'identità
   meglio di Chamfer; ma quel ranking *è definito* come L2 per vertice con corrispondenza
   perfetta, e Chamfer ne è l'approssimazione senza corrispondenza.

Ciò che manca è un'**etichetta d'identità che non derivi dalla geometria**. Due strade la
forniscono, entrambe con dati reali.

---

## 2. Le due strade che sbloccano il problema

### Strada A — dataset con identità ed espressione, ma **una topologia sola**

> **CORREZIONE del 19/08.** Una prima stesura di questa sezione affermava che FaceScape
> distribuisce lo scan grezzo accanto alla registrazione, e quindi darebbe coppie
> cross-topologia gratis. **È falso.** Il paper dice che «the raw scans are processed into TU
> models representing coarse geometry and displacement maps»: gli scan grezzi sono l'input
> della loro pipeline, non l'output rilasciato. Il repo conferma che si scaricano 16.940 modelli
> topologicamente uniformi (847 identità × 20 espressioni), tutti a 26.317 vertici con
> connettività identica, più displacement e texture map. L'affermazione veniva da una ricerca
> web e non dalla fonte.

| dataset | soggetti | per soggetto | topologia | dà |
|---|---|---|---|---|
| **FaceScape** | 847 | 20 espressioni | **uniforme**, 26.317 vert | identità + espressione |
| **NoW** | 100 | 1 (neutra) | scan testa | scan reali, protocollo pubblicato |
| **FaMoS** | 95 | 28 sequenze | registrazioni FLAME | identità sotto movimento |
| **NPHM** | — | — | mesh texturate | qualità, recente |

Restano preziosi per l'**asse espressivo** e per l'etichetta d'identità. Non risolvono la
topologia.

**Terza via dentro FaceScape**: distribuiscono anche oltre 400k immagini multi-view con i
parametri di camera. Ricostruendo le mesh con una pipeline diversa dalla loro si ottengono
topologie genuinamente diverse dello stesso volto — è l'unico modo noto per avere identità,
espressione e topologia nello stesso dataset. Costa lavoro di ricostruzione.

### Strada B — benchmark di *riconoscimento* 3D: **è qui che sta la topologia**

Molti scan dello stesso soggetto in sessioni diverse. Ogni scan è un'acquisizione a sé, quindi
conteggio di vertici e connettività cambiano davvero; l'etichetta è l'ID del soggetto,
**completamente indipendente dalla geometria**. Sono nati esattamente per fornire ground truth
di ranking d'identità.

Nota sul *tipo* di variazione topologica: nei dataset a range image (FRGC, ND-2006, CASIA,
Texas) la mesh nasce triangolando una griglia di profondità, quindi la connettività è
regolare ma la maschera dei pixel validi, la risoluzione e l'allineamento cambiano a ogni
acquisizione. È variazione vera, ma di un tipo specifico — derivata dal sensore, non da un
rimesher come le nostre. Bosphorus distribuisce point cloud, senza connettività: andrebbe
costruita.

| dataset | anno | soggetti | scan | scan/soggetto | tipo |
|---|---|---|---|---|---|
| **ND-2006** | 2007 | 888 | 13.450 | ~15 | range image, laser |
| **FRGC v2** | 2005 | 466 | 4.007 | ~8,6 | range image, laser |
| **UoY** | 2008 | 350 | 5.000 | ~14 | mesh, stereo |
| **Bosphorus** | 2008 | 105 | 4.666 | ~44 | point cloud, stereo |
| **BJUT-3D** | 2009 | 500 | 1.200 | ~2,4 | mesh |
| **Texas-3D** | 2010 | 118 | 1.149 | ~10 | range image, stereo |
| **BU-3DFE** | 2006 | 100 | 2.500 | 25 | mesh, 6 espressioni × 4 intensità |
| **LS3DFace** | — | 1.853 | 31.860 | ~17 | unione di dataset pubblici |

Fonte: tabella dei dataset in *3D Face Recognition: A Survey* (arXiv:2108.11082),
`papers/2108_11082.pdf`.

**FRGC v2 e ND-2006 sono lo standard del campo per il ranking d'identità, e — dopo la
correzione qui sopra — sono anche l'unica fonte pronta all'uso di variazione topologica reale.** Se il nostro claim è
«questa metrica preserva l'ordinamento fra identità», il posto dove dimostrarlo è lì, con il
protocollo che il campo già usa, non su un ground truth che ci siamo definiti da soli.

### Altri, per completezza

| dataset | soggetti | note |
|---|---|---|
| Headspace | 1.519 | teste intere, cuffie in lattice per neutralizzare i capelli |
| LYHM | 1.212 | 3DMM costruito da Headspace |
| Lock3DFace | — | Kinect, bassa qualità ma condizioni realistiche |
| BU-4DFE | 101 | video 3D, 60.600 frame |
| RenderMe-360, Multiface, NeRSemble | — | avatar/multi-view, più orientati a rendering |

---

## 3. Accesso: cosa serve fare

Tutti richiedono un accordo accademico, e **quasi tutti richiedono la firma di un docente
strutturato** — uno studente non può richiederli da solo. È un'azione per Leonardo più il
supervisore, non qualcosa che si risolve scaricando.

| dataset | dove | vincolo noto |
|---|---|---|
| FaceScape | https://nju-3dv.github.io/projects/FaceScape/ | non commerciale; studenti devono far richiedere al supervisore. Sito di download spostato al 27/01/2026 |
| NoW | https://now.is.tue.mpg.de/ | registrazione, non commerciale |
| FaMoS | via FLAME-Universe (MPI) | non commerciale |
| Headspace / LYHM | https://www-users.york.ac.uk/~np7/research/Headspace/ | modulo firmato da staff accademico, a Nick Pears |
| FRGC v2 / ND-2006 | University of Notre Dame, CVRL | accordo di licenza |
| Bosphorus | Boğaziçi University | accordo accademico |
| BU-3DFE / BU-4DFE | Binghamton University | accordo accademico |

**Non ho verificato lo stato corrente di ciascun accordo** — le pagine cambiano e alcune di
queste distribuzioni sono ferme da anni. Prima di pianificare su un dataset, va confermato che
sia ancora distribuito.

---

## 4. Paper scaricati e cosa danno

In `papers/`, con testo estratto a fianco.

| file | paper | perché ci serve |
|---|---|---|
| `2108_11082` | *3D Face Recognition: A Survey* (Deakin, 2021) | la tabella completa dei dataset e i protocolli di valutazione standard del riconoscimento 3D |
| `2111_01082` | *FaceScape: 3D Facial Dataset and Benchmark* (PAMI 2023) | la fonte che ha smentito la mia affermazione sugli scan grezzi: dice che sono l'input della pipeline, non l'output rilasciato |
| `2511_19958` | *GFT-GCN: Privacy-Preserving 3D Face Mesh Recognition with Spectral Diffusion* (NII, 2025) | riconoscimento facciale 3D **spettrale** — il vicino più diretto del nostro approccio, da leggere per primo |
| `2203_09729` | *REALY: Rethinking the Evaluation of 3D Face Reconstruction* (ECCV 2022) | il contraltare metodologico: sostiene che serve **più** registrazione, noi sosteniamo il contrario |
| `2212_02761` | *Learning Neural Parametric Head Models* (NPHM) | dataset recente ad alta qualità e rappresentazione implicita |
| `2607_07486` | *Discovering Geometric Biases in 3D Face Reconstruction* (2026) | bias geometrici dei 3DMM; rilevante per la critica demografica e per il perimetro sintetico |
| `2109_11204` | *Towards Fine-grained 3D Face Dense Registration* | come si ottiene la corrispondenza densa che il nostro GT assume già data |

Non scaricato: *Reconstructing A Large Scale 3D Face Dataset for Deep 3D Face Identification*
(arXiv:2010.08391) — il PDF non si scarica da questo nodo, da recuperare a mano.

---

## 5. Cosa cambierebbe, concretamente

Con FRGC/ND-2006 (topologia) e FaceScape (espressione) diventano possibili tre cose che oggi
non lo sono --- ma le prime due richiedono dataset diversi, non uno solo:

1. **Un ground truth d'identità non circolare.** «Stesso soggetto» è un'etichetta, non una
   distanza calcolata da noi. Il ranking si valuta come recognition: stesso soggetto vicino,
   soggetti diversi lontani.
2. **Cross-topologia reale invece che sintetica.** Oggi le nostre sei topologie le generiamo noi
   da BFM con `crop.py` e `remesh.py`. Due acquisizioni indipendenti dello stesso soggetto sono
   una differenza di topologia che esiste nel mondo, non una che abbiamo fabbricato. Viene dai
   benchmark di riconoscimento, **non** da FaceScape.
3. **L'asse espressivo.** 20 espressioni per soggetto separano «identità» da «configurazione»,
   che è la distinzione che una metrica d'identità deve fare e che oggi non testiamo affatto.

Il costo è di accesso e preprocessing, non di calcolo — e l'accesso richiede una firma che non
può dare un dottorando.
