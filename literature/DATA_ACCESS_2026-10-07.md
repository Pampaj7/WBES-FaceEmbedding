# Accesso ai dati 3D di volti: verifica licenze (2026-10-07)

Metodo: lettura di pagine ufficiali, README e LICENSE nei repo (WebFetch/curl su testo, nessun download di dati). Le pagine di is.tue.mpg.de e famos.is.tue.mpg.de non sono state leggibili per intero (login / host non risolto): dove manca, e' segnato "non verificato".

## Tabella

| Fonte | Licenza (letta) | Accesso | Contenuto | Link |
|---|---|---|---|---|
| FaceScape, dataset | FaceScape Dataset License Agreement (CITE Lab, NJU). Solo ricerca non commerciale, no distribuzione, no sublicenza. Ritratti/rendering pubblicabili solo per i soggetti "portrait-authorized" | Modulo firmato inviato per email. **Il PDF dice: "prohibited for students or business entities to apply"; solo ricercatori/docenti.** Mail sul PDF: facescape@outlook.com; sul sito nju-3dv: nju3dv@nju.edu.cn, oggetto "[FaceScape Dataset Request]". Download via Google Drive/Baidu | 847 soggetti x 20 espressioni = 16.940 TU-model (obj base + displacement png 4K + texture jpg 4K, ~120 GB); multi-vista 359 soggetti, 7.120 tuple, >400k immagini. Soggetti 360-847: occhi pixelati | https://nju-3dv.github.io/projects/FaceScape/ , https://github.com/zhuhao-nju/facescape |
| FaceScape, bilinear model v1.6 | Stesso FaceScape License Agreement (il doc dice "no-commercial-use and no-distribution" e richiede di accettarlo) | **Nessuna license key**: link Google Drive / NJU Drive pubblici (v1.6). Versioni precedenti: serve la key | 4 varianti .npz/.npy: front/standard (50 id x 52 expr), full (300 id x 52), PCA both. Mesh topologia simmetrica. Pacchetto 4,67 GB (cifra del sito nju-3dv, riferita ai modelli bilineari) | https://raw.githubusercontent.com/zhuhao-nju/facescape/master/doc/external_link_fsbm.md , https://github.com/zhuhao-nju/facescape/blob/master/doc/doc_bilinear_model.md |
| NFR (Qin et al., SIGGRAPH 2023) | Repo codice `dafei-qin/NFR_pytorch`: MIT (API GitHub). Dati: nessuna licenza dichiarata sul sample pack HF; dati derivati da ICT-FaceKit (MIT) e Multiface (CC-BY-NC 4.0) | Codice e pesi: Google Drive pubblico. Sample pack: Hugging Face `HKU-CGVU/NFR-sample-data`, 3,0 GB | **Non rilasciano scansioni reali.** Sample pack = 40 identita' ICT (sintetiche, PCA di ICT) con matrici pre-processate + neutral.obj. Set completo del paper ~800 GB: non risulta ospitato (il readme dice solo che e' "about 800GB" e che si rigenera) | https://github.com/dafei-qin/NFR_pytorch , https://huggingface.co/datasets/HKU-CGVU/NFR-sample-data |
| NoW benchmark | NoW license (MPI): "sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects"; no redistribuzione; vieta anche training per uso commerciale; citare RingNet (CVPR 2019) | Registrazione sul sito (account MPI) e accettazione licenza; test set: si inviano mesh+landmark a ringnet@tue.mpg.de o dalla pagina Downloads, max 5 submission per metodo. Restrizioni su studente/docente: **non specificate** nelle pagine lette | 2.054 immagini, 100 soggetti (55 F, 45 M): neutral 620, expression 675, occlusion 528, selfie 231; iPhone X. Scansioni 3dMD ~120k vertici, espressione neutra. **Validation: 20 soggetti con scansioni GT pubbliche; test: 80 soggetti, scansioni nascoste.** Landmark 2D su tutte le immagini; 7 landmark 3D per l'allineamento | https://now.is.tue.mpg.de/ , https://now.is.tue.mpg.de/license.html , https://github.com/soubhiksanyal/now_evaluation |
| FaMoS (TEMPEH) | Licenza MPI non commerciale (stesso testo di NoW/TEMPEH: ricerca, educazione, arte non commerciali; no redistribuzione; no militare/sorveglianza) | Registrazione + accettazione licenza su tempeh.is.tue.mpg.de | 95 soggetti x 28 sequenze, ~600.000 mesh di testa a 60 fps (registrazioni in topologia FLAME). Dimensioni in GB: non verificate | https://tempeh.is.tue.mpg.de/ , https://github.com/TimoBolkart/TEMPEH |
| NeRSemble | "Terms of Use" del dataset, accettati col form; **testo integrale non letto** (il repo non ha file di licenza). Presumibile uso solo ricerca, ma non verificato | Google Form di richiesta, approvazione per email (il sito del benchmark dice di solito entro un giorno), poi script di download. Chi firma: non specificato | >220 partecipanti (la pagina dice "over 220"), >4.700 sequenze, 16 camere, 3208x2200, 73 fps; >1,5 TB. Video e calibrazioni, **non mesh** nel dataset principale. Il NeRSemble Benchmark aggiunge 391 immagini espressive per ricostruzione 3D con GT point cloud posata e neutra e tracking FLAME | https://tobias-kirschstein.github.io/nersemble/ , https://github.com/tobias-kirschstein/nersemble-data , https://kaldir.vc.cit.tum.de/nersemble_benchmark/ |
| Florence 4D | Vedi sezione dedicata: pagina ufficiale MICC = CC BY 4.0; "MIT" compare solo nel catalogo AI4Europe | Google Form "Download Dataset" (approvazione non descritta) | 95 identita' (12 CoMA reali, 63 sintetiche, 20 scansioni reali), 6.650 sequenze singole + 198.550 multi-espressione, 5.023 vertici per mesh, 60/90 frame; 70 espressioni. Dimensione: non verificata (il record Zenodo da 284 MB e' il PDF del paper) | https://www.micc.unifi.it/resources/datasets/florence-4d-facial-expression/ |

## Florence 4D: contraddizione risolta in parte

- Pagina ufficiale MICC (verificata): "Creative Commons Attribution 4.0 International".
- Zenodo record 13939013 (verificato): CC BY 4.0, ma e' il **paper PDF** (2,7 MB, tipo "Conference paper"), non il dataset. Quindi il CC BY di Zenodo riguarda l'articolo.
- "MIT": compare nel catalogo AI4Europe, che non si e' riuscito a leggere (redirect a catalogue.aiodp.eu senza contenuto utile). Origine dell'MIT: metadato di catalogo, non verificato.
- Conclusione: il riferimento primario e' la pagina MICC, CC BY 4.0. Da usare con cautela: 12 identita' derivano da CoMA e 20 sono scansioni di persone reali (consenso scritto dichiarato), quindi la licenza dei dati derivati potrebbe essere piu' restrittiva di CC BY. Verificare scrivendo a MICC prima di redistribuire.

## NoW: metodi con pesi pubblici valutati su NoW

| Metodo | Licenza (letta) | Note |
|---|---|---|
| DECA | LICENSE MPI non commerciale. Vieta anche "to train methods/algorithms/neural networks/etc. for commercial use". Richiede FLAME (registrazione) | Il README riporta risultati NoW (9% meglio dello stato dell'arte di allora) |
| EMOCA (v1/v2) | LICENSE MPI non commerciale, stessa famiglia. Repo marcato "deprecated" in favore di inferno | Basato su DECA |
| MICA | "Software Copyright License for Non-Commercial Scientific Research Purposes" (MPI); copre Model & Software; no terze parti senza permesso. Richiede FLAME 2020 e pesi InsightFace | Addestrato su ~2.300 soggetti; valutato su NoW e Stirling. I dataset di training hanno licenze proprie, non verificate |
| 3DDFA-V2 | Codice MIT. Addestrato su 300W-LP (pesi: nessuna restrizione dichiarata nel repo; licenza di 300W-LP non verificata) | Valutazione su NoW citata dalla comunita': nel repo letto non c'e' menzione NoW |

Nota: RingNet, FLAME e DECA/EMOCA/MICA sono tutti MPI non commerciali. Altri metodi comunemente riportati su NoW (SPECTRE, Deep3DFaceRecon, ecc.) non sono stati verificati.

## Verificato / incerto

Verificato leggendo il testo:
- Testo completo del FaceScape License Agreement (PDF del repo): solo ricercatori/docenti, studenti esclusi.
- Bilinear v1.6 senza license key; l'accordo si applica comunque (no commerciale, no distribuzione).
- Composizione NoW (validation 20 soggetti con scansioni, test 80 nascosti), procedura di valutazione, licenza NoW.
- Licenze MPI di DECA, EMOCA, MICA, MIT di 3DDFA-V2 e di NFR_pytorch.
- NFR: rilascia solo 40 identita' sintetiche ICT (3 GB); niente scansioni reali.
- Florence: CC BY 4.0 sulla pagina ufficiale.
- NeRSemble: dimensioni e procedura di accesso via form; benchmark con 391 immagini e GT 3D.

Incerto o non verificato:
- Se NoW, FaMoS e la registrazione MPI limitino a docenti o accettino studenti (pagine di registrazione non lette). Il testo licenza letto non pone limiti.
- Termini d'uso integrali di NeRSemble e chi deve firmare.
- Dimensioni in GB di FaMoS e di Florence 4D.
- Licenza del sample pack NFR su Hugging Face e origine del "MIT" di Florence.
- Se i dati FaceScape "bilinear v1.6" scaricati senza key siano comunque vincolati a docente/non-studente: il documento li lega al License Agreement, ma nessuna firma e' richiesta tecnicamente.
- Gli strumenti di fetch usano un modello di riassunto: le citazioni tra virgolette provengono dal testo dei LICENSE o del PDF salvo dove indicato; per l'invio dei moduli rileggere le fonti.

## Raccomandazione

(a) Training su larga scala
1. **FaceScape (subito)**: l'unico grande set di scansioni 3D reali con identita' variate (847 soggetti x 20 espressioni). **Deve firmare un docente/ricercatore, non lo studente** (PDF: studenti vietati). Chiedere al supervisore di firmare e mandare il modulo.
2. **Bilinear model v1.6 (subito, nessuna richiesta)**: serve per generare identita' sintetiche in quantita' (300 id x 52 expr), sotto stesso accordo non commerciale. Utile per pre-training.
3. **FaMoS**: registrazione immediata, 95 soggetti, grande volume in topologia FLAME (espressioni e pose). Utile per varianza intra-identita'.
4. Florence 4D e NFR sample: poco utili per metrica d'identita' (poche identita' reali; NFR e' sintetico). Florence solo come extra, dopo chiarita la licenza.
5. NeRSemble: richiedere comunque, ma e' video multivista e non mesh: utile solo se si estraggono geometrie con tracking proprio (pesante).

(b) Valutazione reale di metodi di ricostruzione
1. **NoW**: registrarsi subito. Validation (20 soggetti, scansioni GT pubbliche) per sviluppo; test (80 soggetti) con invio a ringnet@tue.mpg.de. Pesi pubblici da valutare: DECA, EMOCA, MICA (tutti MPI non commerciali, con FLAME registrato) e 3DDFA-V2 (MIT).
2. **NeRSemble Benchmark**: richiedere il form: 391 immagini espressive con GT 3D (point cloud neutra e posata) e tracking FLAME, un secondo set reale indipendente da NoW.
3. FaceScape ha anche un benchmark single-view (paper TPAMI), ma non letto in questa sessione.
4. Per la metrica d'identita': NoW ha 100 soggetti con piu' immagini per soggetto (circa 20), quindi permette anche test di consistenza d'identita' tra immagini, ma la scansione GT e' solo neutra: verificare sul vostro setup.

Cose da fare in ordine: (1) modulo FaceScape firmato dal docente; (2) registrazione MPI unica (NoW + FaMoS + FLAME); (3) form NeRSemble; (4) email a MICC per la licenza di Florence solo se serve.

Non coperto: restrizione su pubblicazione di ritratti (FaceScape: solo soggetti "portrait-authorized"), da tenere presente per le figure del paper.
