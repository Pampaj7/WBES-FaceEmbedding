# Fit-then-compare in uno spazio d'identità appreso: ricerca bibliografica

Data: 2026-10-06. Metodo: alphaXiv (lettura dei PDF), WebSearch/WebFetch, API GitHub per licenze e attività dei repo. Nessun download di dati o pesi.

Legenda dell'affidabilità, usata in tutto il documento:
- **[V]** verificato in questa sessione leggendo il paper, la pagina del progetto o il repo.
- **[S]** letto solo in uno snippet di ricerca o in una fonte secondaria. Da ricontrollare prima di citarlo.
- **[M]** dalla memoria del modello, non verificato qui. Non citare senza controllo.
- **[N]** cercato e non trovato. Non significa che non esista: la ricerca è stata ampia ma non sistematica.

---

## 0. Sintesi (leggere questa se si ha poco tempo)

1. **Nessuno, per quanto trovato, usa i codici d'identità dei modelli neurali di testa (NPHM, ImFace++, imHead) per riconoscimento o verifica 3D.** Nei loro paper il codice d'identità si valuta solo per errore di ricostruzione (Chamfer, F-score, normal consistency) [V]. Nessuna misura di stabilità fra espressioni, nessun EER, nessun rank-1. Questo è il vuoto principale e il nostro spazio di novità.
2. **Il fitting a scansioni arbitrarie funziona ma non è robusto fuori dominio.** NPHM è addestrato su 255 identità, ImFace++ su 831, i3DMM su 64. imHead, che quantifica il fenomeno, riporta che NPHM, MonoNPHM e NPM "face a performance drop when tested on in-the-wild data" [V]. I metodi con deformazione in avanti (NPHM) falliscono con scansioni rumorose perché il root-finding stabilisce male le corrispondenze inverse (imHead, Fig. 14) [V]. Costo: circa 138 s per fit per NPHM, 40 s per imHead, circa 90 s per scansione per ImFace++ (8 ore per 319 scansioni su una GPU) [V, numeri dei paper, hardware diverso].
3. **Il candidato più interessante e più praticabile oggi è GNM Head (Google, arXiv 2607.23687, luglio 2026, repo Apache-2.0 aggiornato oggi).** Modello lineare addestrato su circa 5000 identità reali e circa 150 000 campioni con espressioni, identità e espressione separate [V]. Il fitting a scansioni 3D non è documentato nel repo; va scritto da noi (è un modello lineare, quindi ICP più minimi quadrati, come Amberg 2008). I dati di training non risultano rilasciati [S, dal paper: "model" pubblico, non il dataset].
4. **Evidenza a favore della nostra misura NICP più distanza geometrica:** in Shape-My-Face (IJCV 2021) la rete appresa per registrare scansioni di volti è *meno* accurata di NICP sui landmark (BU-3DFE, media sul volto interno: NICP 4.12 mm contro SMF 5.52-5.70 mm; 3DMD: 4.43 contro 5.57 mm) [V]. Usare la rete solo per inizializzare NICP dimezza il tempo (45-60 s contro circa 20 s) mantenendo la qualità [V].
5. **Storico del fit-then-compare 3D** (Amberg 2008, Gilani 2018, Kakadiaris 2007): il modello deformabile fittato più confronto dei coefficienti ha prodotto risultati molto alti su FRGC v2 e Bosphorus, ma quasi sempre con modelli lineari costruiti da registrazioni dense e valutazione in-dominio. Non dice nulla sulla robustezza fuori dominio, che è proprio quello che vogliamo misurare.
6. **Proposta:** un banco di prova unico che confronta, sulle stesse coppie, (a) NICP più distanza geometrica (la nostra baseline), (b) fit nei tre spazi GNM, ImFace++, NPHM più confronto dei codici, (c) fit, neutralizzazione dell'espressione, NICP sulla malla neutralizzata e distanza geometrica. L'ipotesi (c) unisce ciò che abbiamo misurato (NICP più geometria domina) con l'invarianza all'espressione del modello. Dettagli in sezione 5.

---

## 1. Modelli parametrici neurali di testa e volto

### Tabella riassuntiva

| Modello | Anno, sede | Dati di training | Codice e pesi, licenza | Stato |
|---|---|---|---|---|
| NPHM | CVPR 2023, arXiv 2212.02761 | 255 identità, circa 5200 scansioni, circa 23 espressioni, Artec Eva, 1.5M vertici [V] | Codice MIT; **dataset con licenza separata "molto più restrittiva"** (modulo ToS); pesi NPM e NPHM su Google Drive, licenza dei pesi non dichiarata [V] | Rilasciato |
| MonoNPHM | CVPR 2024 (Highlight), arXiv 2312.06740 | Estensione di NPHM a 488 soggetti (ToS) [V] | Codice MIT, "non copre dati né modelli preaddestrati"; checkpoint su Drive [V] | Rilasciato |
| Pix2NPHM | arXiv 2512.17773, dic. 2025 (sede non verificata) | 102K registrazioni NPHM da dataset 3D pubblici (NPHM, FaceScape, MimicMe, LYHM e altri) più video 2D [V] | Repo esiste (aggiornato 2025-12-22), ma README "Code coming soon" quando l'ho letto; licenza non dichiarata; dataset delle 102K registrazioni promesso, non trovato link [V] | Non confermato |
| ImFace | CVPR 2022, arXiv 2203.14510 | FaceScape | Codice MIT [V] | Rilasciato |
| ImFace++ | arXiv 2312.04028 (v3 ott. 2024; sede IEEE, non verificata quale) | FaceScape: 16 434 scansioni, 831 identità, 20 espressioni; test cross-dataset su 100 scansioni LYHM [V] | Stesso repo MIT; pesi ImFace++ su FaceScape rilasciati dic. 2023 [S]. Licenza dei pesi legata a FaceScape (non commerciale, su richiesta) [M] | Rilasciato |
| i3DMM | CVPR 2021, arXiv 2011.14143 | 64 persone, 58 in training, 10 espressioni, circa 610 scansioni, Treedys [V] | "Will make code available"; non verificato il rilascio [N] | Debole: dati troppo pochi |
| ImHead | arXiv 2510.10793, ott. 2025 (sede non verificata) | 4000 identità, circa 50 000 scansioni, da MimicMe (3dMD, circa 60K vertici) con completamento a testa intera via FLAME e NPHM e rifinitura NICP [V] | Pagina progetto con link ai dati su Drive, nessun link al modello; repo GitHub non trovato [V, parziale]. Licenza dati MimicMe non nota | Non confermato |
| GNM Head | Google, arXiv 2607.23687, lug. 2026 (v3 set. 2026) | Circa 5000 individui, circa 150 000 campioni; multi-vista 22 camere; più asset sintetici per denti e lingua [V] | **Apache-2.0** per l'ecosistema, backend NumPy, JAX, PyTorch, TF [V]. Dati di training non rilasciati [S] | Rilasciato oggi (repo pushed 2026-10-06) |
| DPHM | CVPR 2024, arXiv 2312.01068 | Prior diffusivo sugli spazi latenti di tipo NPHM, per sequenze depth monoculari | Repo pubblico, licenza non verificata [S] | Rilasciato |
| SCULPTOR | arXiv 2209.06423 (2022) | 72 soggetti, 144 TAC (dataset LUCY) [V] | Nessun codice trovato [V] | Non rilevante: modello cranio più volto da TAC chirurgiche |
| GNPM | arXiv 2209.10621 (2022) | Umani vestiti, 4D | Non verificato | **Non è un modello di testa.** Modello neurale parametrico per dinamiche 4D di corpi, forma e posa separate [S]. Escluderlo dalla lista |

Non esaminati a fondo, solo titoli e abstract: GPHM (ECCV 2024, Gaussian, 2407.15070), Gaussian Eigen Models (2407.04545), GRMM (2509.02141). Sono modelli orientati al rendering, non alla geometria da scansione, quindi di bassa priorità. "HeadNeRF-like" non ha dato candidati geometrici utili.

### Domande puntuali

**Su quali dati reali sono addestrati?**
Tutti piccoli e con diversità limitata, tranne GNM e imHead. NPHM: 255 identità, 29% donne, ma gli esperimenti principali del paper usano 87 identità in training e 18 in test (23 nel repo) [V]. Notare l'incoerenza fra "255" e "87": il numero di identità effettivamente usato nel modello rilasciato va controllato nel repo. imHead cita esplicitamente limiti demografici e di capelli ereditati da NPHM, e dichiara che i modelli precedenti hanno "less than 300 subjects" [V]. MimicMe è 73% bianchi, 13% asiatici [V].

**Il fitting a scansioni arbitrarie è robusto?**
- NPHM si fitta a nuvole di punti (5000 punti da una singola depth map nel paper, script `scripts/fitting/fitting_pointclouds.py` nel repo) [V]. Robusto a rumore gaussiano fino a 1.5 mm e a 250-10 000 punti nelle prove del paper, ma tali test usano scansioni dello stesso scanner e della stessa distribuzione [V].
- Richiede **allineamento rigido in uno spazio canonico di tipo FLAME**, ottenuto da landmark 2D (MediaPipe sui rendering, retroproiettati) [V]. Questo sposta il punto di fallimento sull'allineamento: se il landmarking fallisce su una malla fuori dominio, il fit fallisce prima di cominciare.
- Lo spazio canonico di NPHM ha la bocca aperta: le scansioni a bocca chiusa vengono rappresentate tramite deformazione, e il paper nota che il fit non è banale (richiede root-finding SNARF) [V].
- ImFace++ lavora solo sulla regione frontale ritagliata (sfera di 10 cm, origine 4 cm dietro la punta del naso) e richiede la normalizzazione e la costruzione di una malla pseudo-impermeabile [V]. Cross-dataset su LYHM: Chamfer 0.49 mm per ImFace++ contro 0.58 per NPHM su un esempio mostrato; solo qualitativo più un caso [V, un solo esempio in figura].
- imHead riporta Chamfer sul test MimicMe (identità neutra): NPHM 0.618, MonoNPHM 0.614, imHead-Full 0.533 (unità non esplicitate nel testo che ho letto) [V]. Il peggioramento di NPHM da NPHM-test a MimicMe-test (0.558 a 0.618) è il dato più vicino a una misura di degrado fuori dominio.

**Il codice d'identità resta stabile al variare dell'espressione?**
Nessun paper lo misura direttamente [N]. Evidenza indiretta:
- NPHM fitta l'identità su una scansione neutra, poi la tiene *fissa* per fittare le espressioni [V]. Nel test "single-expression" (identità ed espressione stimate insieme da una sola depth map) il Chamfer è 0.207e-2 contro 0.182e-2 del caso neutro [V]: il modello ricostruisce bene, ma ciò non dice se *lo stesso z_id* esca da espressioni diverse.
- ImFace++ mostra disaccoppiamento solo con visualizzazioni PCA e t-SNE delle espressioni (alcune espressioni restano intrecciate, tipo tristezza e rabbia) [V].
- Egger et al. (FG 2021, arXiv 2109.14203) mostrano che identità ed espressione nei 3DMM *non* sono ortogonali e "possono spiegarsi a vicenda sorprendentemente bene"; l'ambiguità non si risolve con un prior statistico [V, abstract e sintesi]. Vale per modelli lineari; per quelli neurali è ipotesi non testata, ma da attendersi.
- Pix2NPHM nota che senza il prior feed-forward "the optimization cannot properly disentangle identity and expression" [V].

**Qualcuno li usa per riconoscimento o verifica?**
[N] per NPHM, MonoNPHM, ImFace(++), i3DMM, imHead, GNM. Gli unici usi "downstream" dei codici sono: classificazione di emozioni (Pix2NPHM su AffectNet, accuratezza 8 classi 71.1% con NPHM contro 66.0% con FLAME [V]); generazione audio-driven (FaceTalk); avatar. Un lavoro vicino per spirito ma non per modello: arXiv 2510.11223 mostra che i soli coefficienti di *espressione* FLAME identificano 1429 parlanti con 61.14% di accuratezza [V]. Conseguenza utile per noi: **i codici d'espressione portano informazione d'identità**, quindi "separato dall'espressione" non è garantito nemmeno in senso inverso, e va testato.

---

## 2. Registrazione e corrispondenza apprese fra topologie diverse

| Lavoro | Anno, sede | Cosa fa | Dati | Codice, licenza | Rilevanza per noi |
|---|---|---|---|---|---|
| Neural Jacobian Fields | SIGGRAPH 2022 (TOG 41(4)), arXiv 2205.02904 | Rete che predice jacobiani per triangolo; mappe fra malle arbitrarie senza topologia condivisa | Addestrata su umani STAR, generalizza a malle mai viste secondo gli autori [S] | Repo ThibaultGROUEIX/NeuralJacobianFields, licenza NOASSERTION nell'API GitHub (da leggere a mano) [V] | Mezzo, non un registratore completo: serve un solver a valle |
| Neural Face Rigging (NFR) | SIGGRAPH 2023 [S] | Rigging e retargeting di espressione su malle facciali di topologia arbitraria, basato su NJF | 3DMM lineare più catture 4D reali [V] | github.com/dafei-qin/NFR_pytorch, **MIT**, attivo [V] | **Idea:** usarlo per "neutralizzare" l'espressione di una malla di topologia arbitraria prima del confronto. Non verificato che preservi l'identità |
| NICP (Neural ICP) | ECCV 2024, arXiv 2312.14024 | Registrazione umana a SMPL con campo neurale localizzato più ICP neurale auto-supervisionato | MoCap, corpi SMPL+H | Solo inferenza, checkpoint inclusi, licenza non dichiarata nel repo [V] | **Nessun test sui volti** [V]. Per i volti servirebbe riaddestrare |
| Shape My Face (SMF) | IJCV 2021, arXiv 2012.09235 | Auto-encoder nuvola di punti a malla: encoder PointNet con attenzione, due decoder (identità, espressione) con **embedding ipersferici z_id e z_exp** e modello PCA per la bocca | 9 database di volti; 4DFAB e MeIn3D non pubblici [V] | github.com/mbahri/smf [S]; licenza non verificata | **Molto rilevante.** È già un fit-then-compare con identità ed espressione separate, su nuvole di punti di qualunque origine. Ha test su scansioni "in the wild" (iPhone, light stage) e stabilità al ricampionamento [V]. Non riporta riconoscimento [V] |
| TEMPEH / ToFu | CVPR 2023 / ICCV 2021 | Volti in corrispondenza densa da immagini multi-vista, non da scansioni | Dataset del gruppo | TEMPEH: licenza NOASSERTION [V] | Poco: parte da immagini calibrate |
| MeshLoom | arXiv 2606.17027, giu. 2026 | Registrazione feed-forward di sequenze di malle di topologia arbitraria, multi-categoria | Texverse (circa 30K sequenze animate), non volti [V] | Non verificato | Non testato su volti; mostra che le reti di registrazione generaliste sono ancora in fase di prova. Cita NJF, NICP, NDP come predecessori e ne elenca i limiti [V] |
| Diff3F | CVPR 2024, arXiv 2311.17024 | Feature semantiche da diffusione, aggregate sulla superficie; corrispondenza fra forma, specie e topologia diverse senza training | Nessuno (foundation models) | github.com/niladridutt/Diffusion-3D-Features [S] | Funziona su malle e nuvole di punti "e raw scan" [S]. Sui volti non verificato; accuratezza probabilmente troppo grossolana per l'identità |
| Quasi-conformal registration per volti parziali | arXiv 2405.09880, 2024 | Registrazione dei volti parziali con reti più mappe quasi-conformi, applicata al riconoscimento | 20 000 volti generati da 3DMM su CelebA; test su LFW con malle ricostruite da immagini [V] | Non verificato | **Evidenza debole e contraddittoria con la nostra.** Riporta NICP al 51% di accuratezza contro 91.73% del proprio metodo, ma su volti parziali sintetici ricostruiti da 2D, non su scansioni reali [V]. Non va preso come confronto affidabile |
| Deep functional maps (Litany 2017 e successivi, DiffusionNet, ULRSSM, Smooth Shells, ecc.) | 2017-2026 | Corrispondenza da funzioni di base spettrali | Benchmark FAUST, SCAPE, SMAL, remeshed | Vari | **Nessun test sui volti trovato** [N]. Il campo valuta su corpi e animali. Aspettarsi errori di decine di mm sui volti perché la geometria intrinseca di due volti è quasi isometrica fra identità: le mappe spettrali confondono le parti simili (guance, fronte) |

Altri lavori 2026 emersi dalla ricerca, **non letti**, solo titoli: TopoRig (2609.15746, rigging facciale topology-agnostic), FreeTalk (2603.15512, talking head su scansioni a topologia arbitraria), MATCH (2603.15811, registrazione Gaussiana feed-forward di teste), RegHead (2607.12206), Hyper-Network Neural Functional Maps (2606.30131), Registration-Free Learnable Multi-View Capture (2605.01450), Ordered Diffusion for 3D Human Registration (2608.05804, corpi).

**Robustezza fuori dominio:** per le reti di registrazione a template, l'unico dato quantitativo verificato su volti è SMF: sulle scansioni di test fuori campione l'errore sui landmark è comparabile a NICP ma peggiore (vedi sintesi punto 4), e *senza* il meccanismo di attenzione le reti sono "molto sensibili al rumore": 100 punti casuali aggiunti bastano per deformazioni significative [V]. Le reti generaliste (NJF, MeshLoom, Diff3F) non sono validate sui volti.

---

## 3. Il 3DMM fitting come baseline di riconoscimento 3D sotto espressione

| Lavoro | Anno, sede | Metodo | Risultato | Affidabilità |
|---|---|---|---|---|
| Blanz e Vetter, "Face recognition based on fitting a 3D morphable model" | TPAMI 25(9), 2003 | Fit del 3DMM a **immagini** 2D; confronto dei coefficienti di forma e tessitura | Valutato su CMU-PIE e FERET, non su scansioni 3D, [M] | Sede [S], dataset [M]. Non è un risultato su FRGC o Bosphorus |
| Kakadiaris et al., Annotated Face Model | TPAMI 29(4), 2007 | Modello deformabile annotato, fit con ICP, simulated annealing e deformazione elastica | 87.0% di verification rate a 0.1% FAR su FRGC v2, 4007 volti | [S] da fonte secondaria, controllare |
| Amberg, Knothe, Vetter, "Expression invariant 3D face recognition with a morphable model" | FG 2008 | Modello lineare appreso da 175 soggetti (una scansione neutra ciascuno) più 50 espressioni per un sottoinsieme; fit NICP robusto con reponderazione e test di compatibilità; inizializzazione con rilevatore del naso; **distanza = angolo di Mahalanobis fra i coefficienti d'identità** | La rimozione dell'espressione aumenta molto il riconoscimento anche su dati difficili senza perdita su dati senza espressione [S] | Metodo [S]; **cifre non recuperate**, il PDF dell'articolo non era accessibile. Da leggere direttamente |
| Gilani, Mian, Shafait, Reid, "Dense 3D face correspondence" | TPAMI 40(7), 2018, arXiv 1410.5058 | Correspondenza densa per propagazione di keypoint e curve geodetiche, modello deformabile K3DM, fit iterativo | **98.5% su FRGC v2, 98.6% su Bosphorus** "face recognition accuracy" | [V] dall'abstract. Protocollo (rank-1? quali galleria e probe?) non specificato nell'abstract |
| Gilani e Mian, "Learning from millions of 3D scans" (FR3DNet) | CVPR 2018, arXiv 1711.05942 | Rete addestrata su scansioni **sintetiche** | Rank-1: BU-3DFE 99.2, FRGC v2 98.4, Bosphorus 99.3, BU-4DFE 96.5; più 10.9 punti su FRGC v2 rispetto al baseline solo 2D | [S] |

Note per noi:
- Le cifre storiche sono altissime ma **in-dominio**: stesso scanner, volti per lo più frontali, gallery e probe dello stesso dataset. Non si traducono in robustezza fuori dominio. Sono il tetto, non la prova.
- **Tensione da chiarire con le nostre misure:** FR3DNet è addestrato su dati sintetici e va molto bene su FRGC e Bosphorus, mentre noi abbiamo misurato che le metriche apprese da sintetico non generalizzano. La differenza plausibile è che i loro test sono su scansioni di scanner reali ben comportati, mentre i nostri sono su 3DMM e scansioni con topologie ed espressioni diverse. Va verificato con i nostri dati, non assunto.
- Amberg 2008 è la ricetta storica più vicina alla nostra ipotesi: **fit con NICP robusto, modello a identità ed espressione separate, distanza di Mahalanobis angolare sui coefficienti d'identità** [S]. Riprodurla con modelli moderni (GNM al posto del modello a 175 soggetti) è un esperimento diretto e a basso rischio.

---

## 4. Stabilità del codice d'identità fra espressioni e dataset

Trovato:
- **Egger et al., FG 2021 (arXiv 2109.14203):** ambiguità identità-espressione nei 3DMM, misurata su tre modelli (due da scansioni di alta qualità, uno da immagini e video) [V]. Il lavoro più vicino a una misura diretta, ma non su verifica e solo su modelli lineari.
- **arXiv 2510.11223:** la stabilità della stima dei parametri FLAME (indicatore "drift-to-noise ratio") correla fortemente in modo negativo con la capacità di identificare [V]. Misura sulla stima da video 2D, ma la tesi è trasferibile: **la qualità del riconoscimento segue la stabilità della stima**.
- **SMF:** analisi statistica di stabilità al ricampionamento (Sez. 5.3): ricampionare la scansione produce solo piccole variazioni nelle registrazioni [V]. Nessuna misura fra espressioni diverse della stessa persona.

Non trovato [N]:
- Qualunque misura intra-classe e inter-classe dei codici z_id di NPHM, ImFace(++), i3DMM, imHead, GNM su scansioni della stessa persona in espressioni diverse.
- Qualunque confronto fra più modelli neurali sulla stessa serie di scansioni per valutare la *comparabilità* dei codici d'identità.
- Qualunque studio dell'effetto del prior (penalità sulla norma dei latenti) sul collasso verso l'identità media.

---

## 5. Opzioni praticabili in 4-5 settimane su cluster con GPU

Ipotesi di calcolo (stima nostra dai numeri dei paper, non misurata): 5000 scansioni a circa 100 s per fit sono circa 140 ore-GPU, cioè mezza giornata su 12 GPU L40S. Il collo di bottiglia non è il calcolo ma l'allineamento rigido, il preprocessing e le licenze.

### Opzione A (settimane 1-2). Banco di prova del fit-then-compare, solo modelli già pronti
Stesse coppie, stesse metriche (EER, rank-1, d-prime, confusione fra espressioni), per:
1. NICP più distanza geometrica (la nostra baseline).
2. **GNM (Apache-2.0)**: fit lineare con ICP più minimi quadrati e prior sul coefficiente; confronto con angolo di Mahalanobis alla Amberg.
3. **ImFace++ (MIT, pesi FaceScape)**: fit al latente da scansione ritagliata.
4. **NPHM/MonoNPHM (codice MIT; pesi Drive)**: fit via `fitting_pointclouds.py`.
5. FLAME (licenza non commerciale [M]) come riferimento classico.

Per ogni modello tre varianti di confronto: distanza latente (coseno e Mahalanobis), distanza geometrica fra le due malle neutre decodificate, distanza geometrica dopo NICP fra le malle neutre. Rischio principale: l'allineamento canonico. Mitigazione: inizializzare con landmark da rendering (stesso metodo di NPHM) e rifinire con ICP rigido; registrare i casi di fallimento come dato.

### Opzione B (settimane 2-4). Fit, neutralizza, registra, confronta
Il fit del modello serve solo a **rimuovere l'espressione**: si decodifica la malla con espressione zero e identità fittata, poi si applica NICP più distanza geometrica sulla neutralizzata. Unisce ciò che sappiamo funzionare (NICP più geometria) con la parte che gli manca (invarianza all'espressione). Variante: NFR (MIT) per neutralizzare malle di topologia arbitraria senza fit del modello. Rischio: il fit può "assorbire" parte dell'identità nell'espressione (ambiguità di Egger) o regredire verso la media; va misurato confrontando neutralizzata e scansione neutra reale della stessa persona.

### Opzione C (settimane 3-5, se A e B danno segnale). Apprendere il codice con una perdita d'identità
Distillare il fit lento in un encoder da nuvola di punti (come Pix2NPHM ma da punti) e aggiungere una perdita metrica fra espressioni della stessa persona. Dati: NPHM (ToS), FaceScape (richiesta), le 102K registrazioni di Pix2NPHM se verranno rilasciate. Rischio alto di licenze e di tempo; da affrontare solo se i primi due esperimenti mostrano che il codice porta informazione d'identità.

### Cose da non fare
- Contare su imHead: modello e codice non verificati come rilasciati.
- Contare su SCULPTOR, GNPM, i3DMM (dati troppo pochi, codice non verificato, non-modello di testa rispettivamente).
- Usare reti di registrazione generaliste (MeshLoom, Diff3F, NJF da solo) come registratore principale: nessuna validazione sui volti.

---

## 6. Spazio di novità

1. **Primo benchmark di verifica d'identità basato sui codici di modelli neurali di testa.** Nessuno l'ha pubblicato per quanto trovato [N]. Anche un risultato negativo ("il codice d'identità dei modelli neurali non supera NICP più geometria fuori dominio") è un contributo, perché oggi si assume il contrario.
2. **Confronto di comparabilità fra spazi latenti di modelli diversi** sulle stesse scansioni, con stabilità intra-persona e separazione inter-persona.
3. **Neutralizzazione dell'espressione tramite il modello seguita da registrazione non rigida e distanza geometrica** (Opzione B): ibrido nuovo rispetto ad Amberg 2008, che confrontava i coefficienti, e rispetto ai modelli neurali, che non fanno verifica.
4. **Misura della fuga d'identità nel codice d'espressione e viceversa** (spunto da 2510.11223 e Egger 2021) per modelli neurali.
5. **Collasso verso la media sotto il prior**: relazione fra intensità della regolarizzazione e potere discriminante, in particolare per identità lontane dal training (bambini, etnie poco rappresentate, citate da imHead).

---

## 7. Cose da ricontrollare prima di citare

- Cifre di Amberg 2008 (non recuperate), Kakadiaris 2007 e Gilani CVPR 2018 (solo da snippet).
- Sede esatta di Pix2NPHM, imHead e ImFace++.
- Licenza dei pesi di NPHM, MonoNPHM e ImFace++, e licenza esatta del repo NJF e di TEMPEH.
- Se i dati di training di GNM e imHead siano o saranno disponibili.
- Se GNM contiene un fit a scansioni 3D non documentato nel README.
- Numero effettivo di identità nel modello NPHM rilasciato (87 contro 255).
- Cosa sia il dataset "DAViD" (65K identità, un'espressione) nella tabella di Pix2NPHM: non verificato, probabilmente sintetico.

## 8. Fonti principali

- NPHM: https://arxiv.org/abs/2212.02761, https://github.com/SimonGiebenhain/NPHM
- MonoNPHM: https://arxiv.org/abs/2312.06740, https://github.com/SimonGiebenhain/MonoNPHM
- Pix2NPHM: https://arxiv.org/abs/2512.17773, https://github.com/SimonGiebenhain/Pix2NPHM
- ImFace e ImFace++: https://arxiv.org/abs/2203.14510, https://arxiv.org/abs/2312.04028, https://github.com/MingwuZheng/ImFace
- i3DMM: https://arxiv.org/abs/2011.14143
- imHead: https://arxiv.org/abs/2510.10793, https://rolpotamias.github.io/imHead/
- GNM Head: https://arxiv.org/abs/2607.23687, https://github.com/google/GNM
- DPHM: https://arxiv.org/abs/2312.01068, https://github.com/tangjiapeng/DPHM
- Shape My Face: https://arxiv.org/abs/2012.09235, https://github.com/mbahri/smf
- NICP: https://arxiv.org/abs/2312.14024, https://github.com/riccardomarin/NICP
- Neural Jacobian Fields: https://arxiv.org/abs/2205.02904, https://github.com/ThibaultGROUEIX/NeuralJacobianFields
- Neural Face Rigging: https://dafei-qin.github.io/NFR/, https://github.com/dafei-qin/NFR_pytorch
- MeshLoom: https://arxiv.org/abs/2606.17027
- Diff3F: https://arxiv.org/abs/2311.17024
- Gilani et al. 2018: https://arxiv.org/abs/1410.5058; Gilani e Mian 2018: https://arxiv.org/abs/1711.05942
- Egger et al. 2021: https://arxiv.org/abs/2109.14203
- Identità nei coefficienti d'espressione: https://arxiv.org/abs/2510.11223
- Quasi-conformal partial faces: https://arxiv.org/abs/2405.09880
