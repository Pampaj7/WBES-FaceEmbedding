# FaMoS: dati, split, GT unificata, set di test reale (8-9 ottobre 2026)

Codice in `aau/famos/`, dati in `datasets/FAMOS/` e `external_data/famos/` (entrambi gitignored, licenza MPI
non commerciale). Qui solo numeri aggregati.

## Catena

| passo | script | job | esito |
| --- | --- | --- | --- |
| split per persona (dichiarato prima di ogni uso) | `famos_split.py` -> `aau/famos/split.json` | frontend | TEST = 079..093 (15, tutti i soggetti delle scansioni TEMPEH), TRAIN = 80 |
| verifica estrazione + inventario | `famos_inventory.py` | 1062049 | 605.802/605.802 ply, byte identici all'elenco dell'archivio, log 7z OK; scansioni 8350 (+readme), tutte con la registrazione dello stesso fotogramma |
| sottocampionamento + neutra | `famos_subsample.py` | 1062034 | 1 ogni 10 per sequenza: 61.770 fotogrammi, 3,74 GB (float32, mm) |
| GT unificata | `famos_unified.py` | 1062049 | `unified_gt.md` |
| set di test reale | `famos_test_view.py` | 1062049 | 1284 patch (851 scansioni, 433 registrazioni) |
| prova e108 + Chamfer | `famos_eval.py` / `.sbatch` | 1062050 | `eval_e108/results.md` |

## Scelte

- **Neutra di riferimento**: FaMoS non ha una sequenza neutra. Si parte dai primi fotogrammi di tutte le sequenze
  e si prende il medoide nello spazio unificato. Si scartano i primi fotogrammi oltre 2 volte la mediana della
  distanza dal medoide e si fa la media rigida dei restanti. Rumore della stima (meta' pari contro dispari):
  mediana 0,23 mm, massimo 0,59 mm.
- **Dimensione**: le registrazioni FLAME pesano 60 KB a fotogramma, quindi 1 ogni 10 da' 3,7 GB, non 10-20 GB.
- **Pre-elaborazione del test**: quella delle scansioni NoW (frame T7, ritaglio `compute_mask`, 5215 triangoli).
  I 7 landmark vengono dalla registrazione dello stesso fotogramma, che dista 0,50 mm (mediana) dalla scansione.
  **Deviazione**: `bfs_orient` prima della decimazione. Senza, 3 scansioni con un triangolo girato mandavano
  `igl.qslim` oltre 48 GB e il Pool si bloccava (job 1062034). Sulle altre mesh non cambia nulla.
- **Ruoli**: galleria = fotogramma piu' vicino alla neutra; peak / nearneutral = il piu' lontano / il piu' vicino
  per sequenza (`expr_mm`, GT unificata dalla registrazione).

## Avvertenze

- `nearneutral -> scan` e' quasi saturo (rank-1 0,995): sono catture della stessa sessione. Il test
  informativo e' `peak`.
- 15 persone: CI larghi, e' una prova della pipeline.
- Rischio: le scansioni NoW vengono dallo stesso sistema di acquisizione (`FaMoS_1804..._TA`). Una
  sovrapposizione di persone con FaMoS non e' verificabile dagli id e non e' stata controllata.

## Pulizia (dopo tutte le verifiche)

- Cancellati gli archivi `test_scans.zip.00*` (dopo la verifica: 8351 file, byte identici) e
  `registrations.zip.00*`.
- Cancellati i 612.450 grezzi non usati. Restano i 1702 file del set di test (`datasets/FAMOS/raw_keep.txt`),
  sufficienti a rigenerarlo identico.
- Non piu' rieseguibili senza riscaricare: `famos_inventory.py` e `famos_subsample.py`.
- A regime: 6,4 GB in `external_data/famos`, 3,8 GB in `datasets/FAMOS`. Picco: circa 236 GB (archivi delle
  registrazioni + estrazione completa), sopra i 200 GB per via dell'estrazione gia' avviata.
