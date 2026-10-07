# Critiche NeurIPS: risposta punto per punto

Scritto il 7 ottobre 2026. Obiettivo: risolvere le critiche dei reviewer e risottomettere la metrica appresa originale. Fonte delle critiche: `paper/review.md`.

| # | Critica (reviewer) | Risposta | Stato |
|---|---|---|---|
| 1 | Solo 3000 volti; servono circa 10^5 e più 3DMM (FLAME, FaceScape, NFR) (YJz1) | Training "mastodontico" su BFM + ICT-55k (valanga, con espressioni) + GNM Head (circa 10k identità reali, Apache-2.0), più FaceScape bilineare se la licenza lo consente. Test su HIFI3D e FaceVerse tenuti fuori. | Valanga generata (700k mesh). Trainer a blocchi pronto, ma OOM al cambio di blocco da correggere. GNM in integrazione. Licenze FaceScape/NFR in verifica. FLAME: licenza da chiedere (utente). |
| 2 | Mai testata su ricostruzione reale (NoW) (YJz1, Z1mX, AC) | Valutazione di più metodi di ricostruzione su NoW validation e/o Multiface: ranking dei metodi con ogni metrica, confrontato con l'errore ufficiale e con lo studio umano. | Multiface WS3b parziale (3DDFA-v2, SynergyNet, PRNet). NoW: registrazione da fare (utente). |
| 3 | Solo baseline geometriche (bhBZ, YJz1, AC) | ArcFace su render, LPIPS, varifold, currents, ICP, NICP. Da aggiungere: CLIP/DINOv2 su render, DPDist se fattibile. | ArcFace, LPIPS, varifold e currents fatti. CLIP/DINOv2 da fare. |
| 4 | Nessuno studio umano; circolarità della D_GT (bhBZ, Z1mX) | (a) Protocollo di **riconoscimento con etichette d'identità** (nessuna D_GT). (b) Studio umano con 300 triplette. | (a) In dominio in corso (`aau/runs/indomain_recog/`), fuori dominio fatto. (b) Pronto, da distribuire (utente). |
| 5 | Cross-topologia solo fra topologie dello stesso 3DMM (Z1mX) | Claim riformulato. Test cross-3DMM (HIFI3D, FaceVerse, GNM) e su Multiface reale, con il limite dichiarato: un dominio nuovo va aggiunto al training. | Dati pronti. Il messaggio dipende dal training su scala. |
| 6 | Solo volti neutri (Z1mX) | Benchmark con espressioni: in dominio il congiunto batte Chamfer, +0.044 [+0.024, +0.067]. La valanga ha 8 espressioni per identità. | Fatto in dominio; con il training su scala da rifare. |
| 7 | Dipendenza da un backbone invariante ai moti rigidi (Z1mX) | Studio sul frame: l'orientamento sposta i modelli fino a metà del gap. Dichiarato, con frame canonico e augmentation. | Analisi fatta (`aau/runs/ws_frame/`). |
| 8 | 3DMM non nominato; bias demografico (Z1mX) | Nominare BFM nel testo; discutere il bias (BFM europeo; HIFI3D e FaceVerse est-asiatici come test). | Da scrivere. |
| 9 | Identità di training e test davvero distinte? (Z1mX) | Split congelato e controllo di quasi-duplicati (distanza minima ≫ soglia), già applicato alla valanga. | Da riportare anche per lo split originale. |
| 10 | Dataset non disponibile (YJz1) | Rilascio su Hugging Face di dati e checkpoint, con le licenze dei 3DMM rispettate. | Da fare a fine lavori. |
| 11 | Novità tecnica limitata (bhBZ) | Scala, multi-3DMM, protocollo di riconoscimento, analisi dei confondenti (frame, supporto, GT), tempi contro la registrazione. | In corso. |

## Cosa serve dall'utente
- Distribuire lo studio umano: https://claude.ai/code/artifact/204feff5-053b-44ce-aa7f-53b49182bacc (provarlo, condividerlo, 25-30 partecipanti).
- **Una sola registrazione MPI** (is.tue.mpg.de) per NoW, FLAME e FaMoS.
- **FaceScape completo:** modulo firmato da un DOCENTE (gli studenti sono esclusi). Il bilineare v1.6 è scaricabile senza chiave e lo scarichiamo noi.
- **Facoltativo:** il form del NeRSemble Benchmark (secondo set reale con riferimento 3D).
- Licenze verificate in `literature/DATA_ACCESS_2026-10-07.md`.
