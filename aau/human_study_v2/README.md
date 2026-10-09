# Studio umano v2: GT a confronto con render a scala assoluta

Arbitro percettivo di `paper/PLAN_MASSIVE.md` §19 fra le GT di E12: GT-F, GT-S, GT-EDM e maxabs. Il protocollo,
scritto prima dei dati, sta in `PROTOCOL.md` (hash in `PROTOCOL.sha256`). Questo file spiega come si usa il
codice e riporta il calcolo di potenza. Lo studio v1 (`aau/human_study/`, `docs/human_study/`) non e' toccato.

## Cosa cambia dalla v1

| | v1 | v2 |
|---|---|---|
| geometria | maxabs per mesh (volti "alti uguali") | mm nel frame di GT-F, una trasformazione per dominio |
| camera | fissa, ma su mesh gia' normalizzate | ortografica, 1.959 px/mm, finestra unica per tutti i volti e le viste |
| viste | frontale | frontale, 3/4, profilo nella stessa immagine |
| visibilita' | painter's algorithm | ray casting (open3d/Embree) |
| triplette | metriche (GT, Chamfer, LPIPS, latent) in disaccordo | GT di E12 in disaccordo, 4 tipi bilanciati |
| analisi | accordo con la maggioranza per tripletta | accordo per risposta, test per strato, Holm |

**Disposizione della prova.** Ogni volto e' una striscia verticale (frontale, 3/4, profilo). La pagina mostra tre
colonne: candidato 1 | riferimento | candidato 2. La scelta segue tre criteri:
- le righe allineano la stessa vista dei tre volti, cosi' si confronta frontale con frontale e profilo con profilo;
- tutto sta in uno schermo senza animazioni, mentre una rotazione automatica mostrerebbe una vista alla volta e
  il confronto dipenderebbe dalla memoria;
- il riferimento al centro sta alla stessa distanza dai due candidati, e il lato dei candidati resta sorteggiato
  come nella v1.

## Esecuzione

Tutto su CPU, nel venv della GT unificata (`v3_work/unified_gt/run.sh`: open3d per il ray casting):

```bash
sbatch aau/human_study_v2/run.sbatch                          # gt, render, select, selftest (~5 min)
HS2_STEPS="gt select" sbatch aau/human_study_v2/run.sbatch      # se E12 cambia cgt.py / gt.py
HS2_STEPS=power sbatch aau/human_study_v2/run.sbatch            # potenza (~3 min)
```

| file | ruolo |
|---|---|
| `hs2.py` | percorsi, soggetti, frame di GT-F (da `aau/runs/evidence/e12/frames.json`, altrimenti `gt.frames`) |
| `gt_v2.py` | tutte le GT di E12 (`cgt.all_gts`) sui 100 soggetti BFM, piu' maxabs legacy, in `datasets/HUMAN_STUDY_V2/gt/` |
| `render_v2.py` | render in `datasets/HUMAN_STUDY_V2/renders/`, `camera.json`, `render_check.json`, `checks/size_extremes.png` |
| `select_triplets_v2.py` | `triplets.json`, `triplets_stats.md`, `docs/human_study_v2/{triplets.js,img/}` |
| `analyze_v2.py` | analisi (`--responses-dir` o `--form-csv`), `--self-test` |
| `power_v2.py` | `power_v2.md`, `power_v2.json` |

Le matrici e i PNG stanno fuori da git (`datasets/`). In git vanno solo i JPEG dei volti sintetici, come nella v1.

**Stato delle GT.** E12 non produce matrici BFM: `gt_v2.py` chiama lo stesso codice di E12 sui soggetti BFM. I
manifest registrano:
- l'impronta di `cgt.py` e `gt.py`;
- se E12 ha concluso (`gt.json`).

Oggi lo stato e' "definitivo" e la selezione e' identica a quella fatta prima della consegna di E12. Un cambio
del frame di BFM blocca `select` finche' non si rifanno i render.

## Raccolta e analisi

La pagina invia al Google Form condiviso con la v1. Il payload porta `study_version: "v2"`, id `v2_*` e
`triplets_hash`. Dal foglio delle risposte, esportato in CSV:

```bash
aau/human_study_v2/analyze_v2.py --form-csv risposte.csv     # o --responses-dir per i JSON scaricati a mano
```

Vengono ignorati, e contati:
- i payload della v1;
- quelli fatti su triplette con un'altra impronta;
- i duplicati.

L'esclusione e' quella della v1: piu' di 1 errore sui 4 controlli.

L'analisi produce:
- **primarie:** per ogni tipo `X_vs_Y` la quota delle risposte con X (IC bootstrap per partecipante, p a segni
  ribaltati, Holm);
- **secondarie:** l'accordo complessivo per GT e le differenze appaiate.

## Calcolo di potenza

`power_v2.py`, 4.000 studi simulati per cella. Ipotesi del modello:
- 9 prove per strato e partecipante, su 60 triplette;
- effetto casuale per partecipante di sd 0.5 logit (nella v1: 0.21-0.38 oltre la binomiale);
- effetto per tripletta di sd 0 o 0.8 logit;
- test a segni ribaltati per partecipante.

Le soglie sono due. Con alfa 0.05 si ha un solo confronto. Con alfa 0.0125 = 0.05 / 4 si ha il caso peggiore di
Holm sui 4 tipi.

| q (quota con X nello strato) | accordo X - Y | N per 80%, alfa 0.05 | N per 80%, alfa 0.0125 | N per 90%, alfa 0.0125 |
|---:|---:|---:|---:|---:|
| 0.55 | 0.10 | 140-200 | 200-300 | 250->400 |
| 0.575 | 0.15 | 60-70 | 90-100 | 120-160 |
| 0.60 | 0.20 | 35 | 50 | 70 |
| 0.65 | 0.30 | 15-20 | 25 | 30 |

**Differenziale realistico.** Nella v1 (7 partecipanti, 36 triplette "in disaccordo" a testa) l'accordo per
risposta era 0.619 per LPIPS e 0.488 per maxabs. E' un differenziale di 0.13 fra due misure che litigano sulle
stesse triplette, cioe' q circa 0.57 in uno strato.

**Conclusione: servono 100 partecipanti tenuti** (90-100 per l'80% a 0.15 con Holm), con un minimo utile di 60.
Il controllo di attenzione della v1 ha scartato 0 persone su 7, quindi 100 inviti completati bastano con un
margine piccolo. Per distinguere un differenziale di 0.10 ne servono 200-300.

## Limiti

Dettagli in `PROTOCOL.md` §6:
- **Taglia.** Le REMESH sono gia' allineate per similarita' una per una. La CV della centroid size e' 1.9%; le
  altezze vanno da 158 a 182 mm, cioe' da 309 a 357 px nei render.
- **Strato `EDM_vs_F`.** Confronta soprattutto posa residua e taglia: la baseline "solo taglia" sta con EDM nel
  97% delle triplette, le varianti rigide di F nel 65-67%.
