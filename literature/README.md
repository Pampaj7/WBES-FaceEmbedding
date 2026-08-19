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

## 3. Accesso — verificato il 19/08/2026

Controllato direttamente sulle pagine di distribuzione, non dedotto. **Tutti vivi tranne
Bosphorus**, il cui dominio non risolve più.

### Notre Dame CVRL — https://cvrl.nd.edu/projects/data/

La fonte più ricca, e tutto in un unico accordo.

| dataset | contenuto | dimensione |
|---|---|---|
| **FRGC v2.0** | 466 soggetti, 4.007 scan 3D + immagini | ~72 GB |
| **ND-2006** | 888 soggetti, 13.450 scan, 6 espressioni | ~29 GB |
| **3D-TEC** | **107 coppie di gemelli**, neutro e sorriso | — |
| ND-Collection D | 277 soggetti, 953 scan frontali | — |

Procedura: scaricare l'accordo di licenza, farlo firmare da **chi è autorizzato ad assumere
impegni legali per l'ente** (non basta il supervisore: serve chi firma per DTU), rispedirlo da
email istituzionale a `cvrl@nd.edu`. Consegna via Globus.

### BU-3DFE — Binghamton, `lijun@cs.binghamton.edu`

100 soggetti × 25 modelli (1 neutro + 6 emozioni × 4 intensità) = 2.500, **~35.000 vertici per
modello**, più texture a 1040×1329. Serve l'accordo scritto del destinatario *e* del direttore
dell'ufficio ricerca dell'ente. Testuale: «Students are not eligible to be a recipient».

### FaceScape — `nju3dv@nju.edu.cn`, oggetto `[FaceScape Dataset Request]`

- **TU models**: 847 soggetti × 20 espressioni = 16.940, topologia uniforme, **120 GB**
- **Multi-view**: 359 soggetti × 20, oltre 400k immagini con parametri di camera e forme 3D
  ricostruite
- Modelli bilineari, più strumenti Python per landmark e regioni

Due vincoli che contano per un paper: la pubblicazione è limitata ai soggetti in una
«publishable list» approvata, e le texture dei soggetti 360–847 sono mascherate per privacy.
Il 10% della collezione è trattenuto per benchmark futuri.

### Bosphorus — non raggiungibile

`bosphorus.ee.boun.edu.tr` non risolve. Da considerare non disponibile finché non emerge un
mirror.

---

## 3-bis. 3D-TEC: il dataset che separa identità da geometria

Centosette **coppie di gemelli**, neutro e sorriso. È il test più affilato possibile della
critica di circolarità che pesa sul nostro ground truth.

Due gemelli identici hanno geometria quasi identica e identità diverse. Il nostro `D_GT`
attuale — L2 per vertice in corrispondenza densa — li dichiarerebbe **la stessa persona**. Una
metrica d'identità deve separarli; una metrica geometrica travestita da metrica d'identità no.

È un esperimento piccolo (428 scan) che produce un'affermazione netta, e nessuno degli altri
dataset la consente. Se serve una singola figura per rispondere alla causa di rifiuto C2, è
questa.

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
