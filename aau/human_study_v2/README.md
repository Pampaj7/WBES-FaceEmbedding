# Studio umano v2: taglia, forma e normalizzazione nel giudizio umano di somiglianza

Arbitro percettivo di `paper/PLAN_MASSIVE.md` §19 fra le GT di E12. Il protocollo, scritto prima dei dati, sta in
`PROTOCOL.md` (revisione 3, hash in `PROTOCOL.sha256`), con le letture dichiarate per ogni strato. Lo studio v1
(`aau/human_study/`, `docs/human_study/`) non e' toccato.

## Disegno in breve

| | |
|---|---|
| volti | GNM Head (Apache-2.0), 100 identita' campionate, CV della taglia 5.3% |
| geometria | mm nel frame di GT-F di E12 + rigida robusta per identita' (come la GT F); mai una scala per mesh |
| camera | ortografica, 1.745 px/mm, una finestra per tutti i volti e le viste; ray casting |
| immagine | striscia verticale frontale / 3/4 / profilo; prova = candidato \| riferimento \| candidato |
| strati | `F_vs_S` (principale: la taglia conta?), `F_vs_size` (forma oltre la taglia?), `S_vs_maxabs` a taglia neutra |
| sessione | 3 di prova + 60 test (24/18/18) + 4 controlli, circa 10 minuti |
| analisi | quota con X per strato, SE incrociato partecipanti x triplette, t di Satterthwaite, Holm |
| obiettivo | 60 partecipanti tenuti |

**Disposizione della prova.** Le righe allineano la stessa vista dei tre volti. Tutto sta in uno schermo,
senza animazioni che obblighino a ricordare. Il riferimento al centro sta alla stessa distanza dai due candidati,
e il lato dei candidati e' sorteggiato.

**Istruzioni.** Sono neutre: "Which face is more similar to the reference face? Ignore light and shadows". La
parola "shape" non compare. La resa comune (grigio, stessa luce, stessa camera) e' detta una volta sola.

## GT e strati

Le GT sono calcolate con le funzioni di `v3_work/canonical_gt/cgt.py` (`gt_v2.py`):
- **F:** rigida robusta per identita', in mm;
- **S:** la stessa centrata e scalata per centroid size;
- **maxabs:** legacy zero-shot;
- **"solo taglia":** |Δ log CS|;
- **secondarie:** EDM, EDM_s, unified, F_rig_ls, F_pure, "solo altezza".

| strato | chi sta con X (quota delle triplette) | lettura di q > 0.5 |
|---|---|---|
| `F_vs_S` | F, EDM, "solo taglia" (100%) contro S, unified, maxabs | la taglia conta |
| `F_vs_size` | F, S, maxabs, unified, EDM (99-100%) contro "solo taglia" | oltre la taglia conta la forma |
| `S_vs_maxabs` | S, unified; F e "solo taglia" al 50% (bilanciati), "solo altezza" al 36% | maxabs e' peggiore della shape |

`S_vs_maxabs` e' a taglia neutra (contrasto di taglia ≤ 1%) e bilanciato. Il self-test lo verifica: partecipanti
simulati che seguono F o la sola taglia NON producono un effetto in questo strato (p 0.81 e 0.92).

## Esecuzione

Tutto su CPU, nel venv della GT unificata (`v3_work/unified_gt/run.sh`: open3d per il ray casting). Il nodo `cpu`
ha avuto errori di prolog: in quel caso conviene `-p prioritized --gres=NONE`.

```bash
sbatch aau/human_study_v2/run.sbatch                          # gt, render, select, selftest (~5 min)
HS2_STEPS=power sbatch aau/human_study_v2/run.sbatch          # alfa empirico e potenza (~20 min con 16 CPU)
```

| file | ruolo |
|---|---|
| `hs2.py` | dominio GNM, campionamento, frame di GT-F, rigida robusta, triangoli da renderizzare |
| `gt_v2.py` | GT in `datasets/HUMAN_STUDY_V2/gt/` |
| `render_v2.py` | render, `camera.json`, `render_check.json`, `checks/size_extremes.png` |
| `select_triplets_v2.py` | strati (`STRATA`: vincoli di taglia e bilanciamento), `triplets.json`, `triplets_stats.md`, `docs/human_study_v2/` |
| `analyze_v2.py` | analisi (`--form-csv` o `--responses-dir`); `--self-test` con i comportamenti di `SCENARIOS` |
| `power_v2.py` | alfa empirico dei tre test e potenza del primario: `power_v2.md`, `power_v2.json` |

## Raccolta e analisi

La pagina invia al Google Form condiviso con la v1. Il payload porta:
- `study_version: "v2"`;
- id `v2_*`;
- `triplets_hash` (`9cea595d967d69ab`);
- dimensioni della finestra e DPR, dichiarati nel consenso.

Dopo un invio riuscito lo stesso browser mostra "already taken part".

```bash
aau/human_study_v2/analyze_v2.py --form-csv risposte.csv
```

**Inferenza primaria.** Il test a segni ribaltati per partecipante ignora che anche le triplette sono un
campione: in simulazione il suo alfa arriva a 0.19. Il primario usa quindi V = V_P + V_T - V_0, cioe' i bootstrap
sui soli partecipanti e sulle sole triplette meno la varianza binomiale (Owen 2007). Il p si legge su una t di
Satterthwaite. L'alfa empirico va da 0.043 a 0.075 (mediana circa 0.056) su 48 condizioni.

## Calcolo di potenza

Tabelle complete in `power_v2.md`. Il caso prudente ha sd 1.0 fra partecipanti e 0.8 fra triplette, con alfa
0.05/3 (Holm, caso peggiore). N tenuti per l'80%:

| differenziale (2q - 1) | F_vs_S | F_vs_size | S_vs_maxabs |
|---:|---:|---:|---:|
| 0.15 | 200 | 200 | 250 |
| 0.20 | **60** | 80 | 80 |
| 0.30 | 25 | 30 | 30 |

Con sd 0.5 fra partecipanti (nella v1 era 0.21-0.38), a 0.20, ne bastano 40-50.

**Pianificazione: differenziale 0.20, 60 partecipanti tenuti.** La v1 dava 0.13 con margini circa 3 volte piu'
piccoli, e qui il contrasto di taglia di `F_vs_S` e' ben visibile.

**Limite:** con 60 partecipanti:
- gli strati secondari arrivano al 76-77% nel caso prudente;
- un differenziale di 0.15 richiederebbe 200-250 persone.

## Limiti

Dettagli in `PROTOCOL.md` §7:
- identita' sintetiche di un solo modello;
- "solo altezza" in `S_vs_maxabs` favorisce maxabs (non neutralizzabile insieme agli altri vincoli);
- la taglia si giudica solo per confronto fra i tre volti;
- il flag "gia' partecipato" vale per un solo browser.
