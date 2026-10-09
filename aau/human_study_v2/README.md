# Studio umano v2: form contro shape, con render a scala assoluta

Arbitro percettivo di `paper/PLAN_MASSIVE.md` §19. La domanda principale e' se, nel giudizio umano di
somiglianza, conti la taglia assoluta (GT-F) o no (GT-S). La seconda e' se maxabs sia peggiore della shape. Il
protocollo, scritto prima dei dati, sta in `PROTOCOL.md` (revisione 2, hash in `PROTOCOL.sha256`). Lo studio v1
(`aau/human_study/`, `docs/human_study/`) non e' toccato.

## Cosa cambia dalla v1

| | v1 | v2 |
|---|---|---|
| volti | BFM REMESH (taglia gia' normalizzata, CV 1.9%) | GNM Head, 100 identita' campionate, CV della taglia 5.3% |
| geometria | maxabs per mesh (volti "alti uguali") | mm nel frame di GT-F + rigida robusta, mai una scala per mesh |
| camera | fissa, ma su mesh gia' normalizzate | ortografica, 1.745 px/mm, una finestra per tutti i volti e le viste |
| viste | frontale | frontale, 3/4, profilo nella stessa immagine |
| visibilita' | painter's algorithm | ray casting (open3d/Embree) |
| triplette | metriche in disaccordo | F contro S (principale), S contro maxabs, margini grandi |
| sessione | 36 + 4 controlli | 3 di prova + 60 test (36 + 24) + 4 controlli, circa 10 minuti |
| analisi | accordo con la maggioranza per tripletta | quota per strato, segni ribaltati per partecipante, Holm |

**Disposizione della prova.** Ogni volto e' una striscia verticale (frontale, 3/4, profilo). La pagina mostra tre
colonne: candidato 1 | riferimento | candidato 2. La scelta segue tre criteri:
- le righe allineano la stessa vista dei tre volti, cosi' si confronta frontale con frontale e profilo con profilo;
- tutto sta in uno schermo senza animazioni, mentre una rotazione automatica mostrerebbe una vista alla volta e
  il confronto dipenderebbe dalla memoria;
- il riferimento al centro sta alla stessa distanza dai due candidati, e il lato dei candidati resta sorteggiato.

**Dominio.** GNM Head e' sotto Apache-2.0 (codice e pesi, `~/data/gnm_head/PROVENANCE.md`): i render sono
pubblicabili, citando ploumpis2026gnmhead. ICT (MIT) aveva un CV di 4.8%. BFM REMESH (1.9%) e FaceScape (1.6%)
sono troppo stretti, e FaceScape non e' distribuibile. Nei render la testa ha la stessa rigida robusta della GT F,
e si vede la maschera del volto (`hockey_mask`, che contiene tutta la regione delle GT) con gli occhi.

## GT dello studio

Tutte con le funzioni di `v3_work/canonical_gt/cgt.py`; dettagli in `gt_v2.py`.

| GT | definizione |
|---|---|
| F | regione in mm, rigida robusta per identita' verso mu (la GT form di riferimento di E12) |
| S | la stessa centrata e scalata alla centroid size di mu (Procrustes pieno); NON la S di E12 a punto fisso |
| maxabs | legacy zero-shot: patch `hockey_mask`, maxabs per mesh, media per vertice della L2 |
| secondarie | EDM, EDM_s, unified, F_rig_ls, F_pure, "solo taglia", "solo altezza" |

Spearman F-S sulle 4.950 coppie: 0.59, contro 0.96 su BFM REMESH. F segue la taglia: F con "solo taglia" 0.80.

## Esecuzione

Tutto su CPU, nel venv della GT unificata (`v3_work/unified_gt/run.sh`: open3d per il ray casting):

```bash
sbatch aau/human_study_v2/run.sbatch                            # gt, render, select, selftest (~5 min)
HS2_STEPS="gt select" sbatch aau/human_study_v2/run.sbatch      # se E12 cambia cgt.py / gt.py, PRIMA della raccolta
HS2_STEPS=power sbatch aau/human_study_v2/run.sbatch            # potenza (~5 min)
```

| file | ruolo |
|---|---|
| `hs2.py` | dominio GNM, campionamento delle identita', frame di GT-F, rigida robusta, triangoli da renderizzare |
| `gt_v2.py` | GT in `datasets/HUMAN_STUDY_V2/gt/` (+ `size.csv`, `rigid.csv`, `gt_corr.csv`) |
| `render_v2.py` | render in `datasets/HUMAN_STUDY_V2/renders/`, `camera.json`, `render_check.json`, `checks/size_extremes.png` |
| `select_triplets_v2.py` | `triplets.json`, `triplets_stats.md`, `docs/human_study_v2/{triplets.js,img/}` |
| `analyze_v2.py` | analisi (`--form-csv` o `--responses-dir`), `--self-test` |
| `power_v2.py` | `power_v2.md`, `power_v2.json` |

Matrici e PNG stanno fuori da git (`datasets/`); nella pagina vanno solo i JPEG.

## Raccolta e analisi

La pagina invia al Google Form condiviso con la v1. Il payload porta `study_version: "v2"`, id `v2_*` e
`triplets_hash` (`30f02a3b4150927c`). Dal foglio esportato in CSV:

```bash
aau/human_study_v2/analyze_v2.py --form-csv risposte.csv
```

Vengono ignorati, e contati:
- i payload della v1;
- quelli fatti su triplette con un'altra impronta;
- i duplicati.

L'esclusione e' quella della v1: piu' di 1 errore sui 4 controlli.

L'analisi produce:
- **primarie:** per `F_vs_S` e `S_vs_maxabs`, la quota delle risposte con la prima GT, con IC bootstrap per
  partecipante, p a segni ribaltati e Holm su 2;
- **secondarie:** l'accordo complessivo per GT e le differenze appaiate.

## Calcolo di potenza

`power_v2.py` (tabelle complete in `power_v2.md`). Il modello simula:
- 36 prove per partecipante su 120 triplette (F_vs_S) e 24 su 80 (S_vs_maxabs);
- un effetto casuale logit per partecipante con sd 0.5 (nella v1: 0.21-0.38 oltre la binomiale);
- un effetto per tripletta con sd 0 o 0.8;
- il test a segni ribaltati.

La soglia alfa = 0.025 corrisponde al caso peggiore di Holm su 2 strati. N tenuti:

| q nello strato | accordo X - Y | F_vs_S 80% | F_vs_S 90% | S_vs_maxabs 80% | S_vs_maxabs 90% |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.10 | 100 | 120-150 | 100-120 | 150->200 |
| 0.575 | 0.15 | 40 | 50 | 50 | 60-70 |
| 0.60 | 0.20 | 25 | 30 | 25-30 | 35 |
| 0.65 | 0.30 | 12 | 15 | 12-15 | 15-18 |

**Differenziale realistico.** Nella v1 l'accordo per risposta era 0.619 (LPIPS) contro 0.488 (maxabs), un
differenziale di 0.13. Quei margini erano circa 3 volte piu' piccoli di quelli di adesso (mediana 0.42-0.48), e
nello strato principale le teste differiscono fino al 30% in taglia. Si pianifica quindi su q = 0.60
(differenziale 0.20), con q = 0.575 come caso prudente.

**Servono 40 partecipanti tenuti** (minimo utile 25):
- per F_vs_S, l'80% di potenza anche a q = 0.575;
- il 90% a q = 0.60 su entrambi gli strati.

Nella v1 il controllo di attenzione non ha escluso nessuno su 7.

## Limiti

Dettagli in `PROTOCOL.md` §7:
- identita' sintetiche di un solo modello;
- la taglia si giudica solo per confronto fra i tre volti, a parita' di camera;
- i render hanno la rigida robusta per identita', come la GT F.
