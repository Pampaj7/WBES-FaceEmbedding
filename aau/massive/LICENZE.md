# Licenze dei dati del modello da rilasciare

Scritto il 13 ottobre 2026 (coder, su incarico del PI). NON è un parere legale. Va rivisto dall'ufficio legale o di trasferimento tecnologico di AAU prima di qualunque pubblicazione.

**La decisione di rilasciare i pesi spetta all'utente.** Questo file non autorizza nessuna pubblicazione e nessun upload; finora non è stato pubblicato nulla.

Fonti già raccolte: `literature/LICENZE_RILASCIO_2026-10-09.md`, `literature/DATA_ACCESS_2026-10-07.md`, i README e LICENSE su disco (`~/data/*/`, `v2_work/genflame/official/`). Le righe marcate "da verificare" non sono state rilette sulla pagina ufficiale in questa sessione.

## Tabella

| Dataset (ruolo nel run) | Licenza | Training per ricerca | Pesi derivati: redistribuzione | Fonte |
|---|---|---|---|---|
| **BFM 2019** (generatore) | Basel Face Model 2019, licenza dell'Università di Basilea: solo ricerca non commerciale, §3.3 niente derivati né distribuzione | sì | **no** senza permesso scritto (Unitectra, mail@unitectra.ch) | faces.dmi.unibas.ch; `literature/LICENZE_RILASCIO_2026-10-09.md` |
| **ICT-FaceKit Light** (generatore) | MIT | sì | sì (avviso MIT) | github.com/ICT-VGL/ICT-FaceKit, commit da5f95a |
| **GNM Head** (generatore) | Apache-2.0 | sì | sì (Apache-2.0, NOTICE) | github.com/google/GNM; `~/data/gnm_head/LICENSE_APACHE-2.0` |
| **FLAME 2023 Open** (generatore) | CC BY 4.0 con restrizioni d'uso (niente pornografia né deepfake), Readme del pacchetto | sì | sì, con attribuzione e le stesse restrizioni | flame.is.tue.mpg.de; `v2_work/genflame/official/FLAME2023Open/FLAME2023_Open Readme.pdf` |
| FLAME_masks.pkl (maschera "face" di FLAME 2023 e di FaMoS) | file del sito FLAME, licenza non letta | sì | da verificare | `v2_work/genflame/official/readme` |
| **HIFI3D++** (`AI-NEXT-Shape.mat`, ex test, ora generatore) | codice MIT (Tencent AI Lab); il 3DMM è "for research purpose only", per uso commerciale contattare gli autori | sì | **dubbio**: al più non commerciale, e solo dopo un parere | `~/data/hifi3d/readme.md:10`, `~/data/hifi3d/README.md:50-56` |
| **FaceVerse v2** (`faceverse_simple_v2.npy`, ex test, ora generatore con espressioni) | uso di ricerca non commerciale (repo Tsinghua), da verificare | sì | **dubbio**: al più non commerciale | github.com/LizhenWangT/FaceVerse; nessun file di licenza su disco |
| **FaceScape bilineare v1.6** (ex dev, ora generatore con espressioni) | FaceScape License Agreement (NJU): solo ricerca non commerciale, niente distribuzione né sublicenza. Il PDF esclude gli studenti come richiedenti: l'accordo deve averlo firmato un ricercatore o docente | sì, **se l'accordo è firmato da chi ne ha titolo** (da verificare) | **no** senza permesso (nju3dv@nju.edu.cn) | `literature/DATA_ACCESS_2026-10-07.md:9-10`; `~/data/facescape_bilinear/README.md` |
| **FaMoS TRAIN** (74 persone in training, 6 held-out) | licenza MPI non commerciale (ricerca, educazione, arte); niente redistribuzione; niente usi militari o di sorveglianza. Persone reali: dati biometrici (GDPR) | sì | **no** senza permesso (ps-license@tue.mpg.de) | tempeh.is.tue.mpg.de; `literature/DATA_ACCESS_2026-10-07.md:13` |
| FaMoS TEST (15 persone) | MPI | **escluso** (validazione esterna) | n/a | `aau/famos/split.json` |
| Ava-256 | — | **escluso** (test confermativo vergine) | n/a | — |
| FLAME 2020 e versioni non ufficiali | licenza MPI di FLAME | **non usato** (FaMoS gira sul template di FLAME 2023 Open) | n/a | — |

## Conseguenze per i pesi del run `release`

- Il modello è addestrato insieme su BFM 2019, FaceScape e FaMoS, che vietano i derivati o li sottopongono a permesso, e su HIFI3D++ e FaceVerse, che sono solo non commerciali. Per questo **i pesi così come sono NON sono redistribuibili.**
- Per renderli redistribuibili servono, almeno a uso non commerciale, permessi scritti da Basilea (Unitectra), MPI e NJU, più un parere su HIFI3D++ e FaceVerse. Il permesso va chiesto per tutti e cinque.
- Un modello con pesi rilasciabili esiste solo sul nucleo aperto: GNM + ICT Light + FLAME 2023 Open (`STREAM_SOURCES=open_core`). Non è questo run.
- Il dataset di training resta tracciabile anche quando non è ridistribuibile. Ogni vista usata ha seme, fonte e licenza ereditata nel registro `views_used/` e nello shard (`prov.license`), e `regen.py` rigenera le mesh dai semi per chi ha i modelli. Per BFM 2019, HIFI3D, FaceVerse e FaceScape si rilasciano al più ricetta e semi; per FaMoS nemmeno quelli (`literature/LICENZE_RILASCIO_2026-10-09.md`).
- Nota sul codice: `sources.LICENSES` scrive ancora per FaMoS "solo ricetta". Il testo non è stato cambiato per non toccare i byte degli shard dei run esistenti. Vale questa tabella: per FaMoS niente ricetta né semi.
