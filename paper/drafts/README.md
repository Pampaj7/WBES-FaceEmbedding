# Bozze CVPR 2027: GT, protocollo, costo (10 ottobre 2026)

Bozze da rivedere. Non toccano `paper/main_cvpr_draft.tex` né `paper/main_short.tex`. Ogni numero ha un commento
`% fonte: ...` con il file del repository da cui viene; dove il numero non c'è c'è un `\todo{...}`.

| File | Sezione | Etichette definite |
|---|---|---|
| `sec_gt.tex` | What Should a 3D Face Distance Measure? (tesi precisata, invarianza minima, FR/SR e formula size-and-shape, arbitro FaMoS, perché non la maxabs, rivalutazione dei 22 metodi) | `sec:gt`, `sec:gt_principle`, `sec:gt_definition`, `sec:gt_arbiter`, `sec:gt_maxabs`, `eq:robust_rigid`, `eq:fr_sr`, `eq:size_shape`, `tab:gt_arbiter`, `tab:gt_reeval` |
| `sec_protocol.tex` | Evaluation Protocol (domini e ruoli, topologie, metriche, bootstrap e regole, baseline con rimozione dei disturbi coerente, baseline banali, studio umano v2) | `sec:protocol`, `sec:protocol_domains`, `sec:protocol_metrics`, `sec:protocol_baselines`, `sec:trivial_baselines`, `sec:protocol_human`, `tab:domain_roles`, `tab:trivial` |
| `sec_cost.tex` | Computational Cost (iscrizione, ricerca 1:N, tabella) | `sec:cost`, `tab:cost` |

Usano dal draft principale: `\todo`, `sec:distance_compression`, `sec:remesh_benchmark`, `sec:method`,
`eq:ground_truth_distance`, `eq:spearman_correlation`. Nessuna compilazione eseguita: sul frontend non c'è LaTeX.
Controllati solo graffe, ambienti e `$` bilanciati, con uno script.

## `\todo` aperti

### In attesa di risultati in arrivo

| Dipendenza | Dove | Cosa serve |
|---|---|---|
| **Bracci `factorized` e `ctrl-FR`** (C3F, 2 semi; protocollo `aau/runs/evidence/trainer_v3/factorized_protocol.md` + emendamenti 1-3) | `sec_protocol.tex` (regole di decisione), `sec_gt.tex` (rimando al modello) | Verdetto della regola g ("raggiunge / si avvicina / non si avvicina a NICP") e factorized − ctrl-FR, entrambi i semi |
| **Run massivo** (non lanciato; gate in PLAN_MASSIVE §21: revisione con l'utente) | `sec_gt.tex` Tab. `tab:gt_reeval`, ultima riga; `sec_protocol.tex` Tab. `tab:domain_roles` (domini di training definitivi), punteggio dev; `sec_cost.tex` (accuratezza finale) | Spearman maxabs / Procrustes / FR del modello finale su HIFI3D; rank-1 e FR su HIFI3D, FaMoS test, NoW; conferma dei domini di training |
| **Studio umano v2** (`aau/human_study_v2/PROTOCOL.md`, revisione 4, obiettivo 60 partecipanti tenuti; non ancora distribuito) | `sec_gt.tex` (arbitro percettivo), `sec_protocol.tex` (sottosezione Human Study) | N tenuti ed esclusi; q con IC per `F_vs_S`, `F_vs_size`, `S_vs_maxabs` (test a cascata, soglia 0.04) |
| **Tempi del modello finale** | `sec_cost.tex` (2 punti + cella della tabella) | Misura end-to-end sullo stesso nodo L40S: operatori su GPU (via `bk`) + forward della testa fattorizzata; ricerca 1:N. Oggi 0.15 s è la somma di 97 ms (V100, E10) e 55 ms (L40S, indomain_recog): non è end-to-end |

### Lavori da fare (non dipendono dai run in corso)

1. **Baseline eque in mm** (`sec_gt.tex`, `sec_protocol.tex`): Chamfer, ICP + Chamfer, NICP P2Tri e NICP su template
   oggi normalizzano ogni mesh con maxabs (e il template applica Procrustes con scala). Sono quindi coerenti con SR,
   non con FR. Vanno rifatti in mm con ICP rigido senza scala (riga FR) e con ICP di similarità (riga SR), su HIFI3D,
   FaceScape dev, FaceVerse e FaMoS test. **Può cambiare la tabella `tab:gt_reeval` a favore delle baseline.**
2. **Colonna SR** per tutti i 22 metodi (`sec_protocol.tex`): E12 riporta maxabs, unificata, F, S, EDM, ma non SR
   (la S di E12 non è SR: scala attorno a un punto fisso, senza rigida).
3. **Baseline banali sulle mesh osservate** (`sec_protocol.tex`): quelle di E12 sono oracolo (taglia e altezza
   dalla GT). Su HIFI3D la "solo taglia" oracolo fa 0.736 con FR, sopra ogni metodo: serve la versione stimata
   dalle mesh (centroid size e altezza di ogni realizzazione, fra topologie), con Spearman FR e riconoscimento.
4. **IC appaiato EDM − EDM-s** sull'arbitro FaMoS (`sec_gt.tex`): oggi solo i punti (0.991 contro 0.985).
5. **Numeri del critic 0.975 / 0.638** (job 1062067, PLAN_MASSIVE §16.1: maxabs con e senza rigida, maxabs contro
   normalizzazione per centroid size): non stanno in un file di risultati, quindi li ho TOLTI da `sec_gt.tex`.
   Se servono (sono l'argomento "è il divisore, non la rotazione"), rifarli con il codice di E12.
6. **Definizione della maxabs** (`sec_gt.tex`): il codice (`build_zs_gt.py`) fa una media delle norme per vertice;
   l'Eq. `eq:ground_truth_distance` del draft principale scrive un RMS. Va allineato.
7. **NoW** (`sec_protocol.tex`): statistica (Kendall τ contro l'errore ufficiale?) e uso di SR come riferimento.
8. **FaMoS test, riconoscimento** (`sec_protocol.tex`): protocollo definitivo (scansioni grezze contro registrazioni).
9. **"Remesher mai visto"** (`sec_protocol.tex`): generatore e numeri, se esiste (PLAN_MASSIVE §2.5).
10. **Punteggio dev** (`sec_protocol.tex`): dichiarare con quale GT si calcola (FR?) prima del run finale.
11. **NICP su template 2.79 s su HIFI3D** (`sec_cost.tex`): verificare hardware e thread di quel run.
12. **Rilascio dei protocolli con hash** nel supplementare (`sec_protocol.tex`): decisione.

## Riserve di contenuto da tenere visibili nel paper

- **Lunghezza:** `sec_gt.tex` è circa 1.5-2 pagine con le due tabelle; i dettagli tolti sono marcati `SUPP` nei
  commenti (tabella completa dei 22 metodi, IC del rank-1, riga "solo altezza").
- **Arbitro FaMoS:** differenze d'AUC piccole, vicino al tetto. Il Procrustes completo (che toglie la scala) è
  **pari** a FR sull'AUC (+0.0002 [−0.0012, +0.0016]) e perde solo sul rapporto intra/inter. La rigida LS batte la
  robusta e la maxabs batte FR (+0.0020), ma su una regione più grande. Tutte e 95 le persone, test incluse, sono
  entrate nella scelta della GT. Scritto così nella bozza.
- **Con FR, NICP per coppia batte il modello attuale** (+0.174 [+0.092, +0.253]), e e108 non batte Chamfer in modo
  significativo. La bozza lo dice e sposta l'argomento su costo e modello fattorizzato.
- **Costo:** su CPU a un thread la nostra iscrizione (2.08 s) è più lenta di NICP su template (1.15 s); il vantaggio
  c'è solo con gli operatori su GPU, mentre il template non è stato portato su GPU. Detto nel testo.

## Critiche dei recensori coperte (`paper/REBUTTAL_PLAN.md`)

| # | Critica | Dove |
|---|---|---|
| 2 | Mai testata su ricostruzioni reali | `sec_protocol.tex`: NoW e FaMoS test come test intatti (numeri `\todo`) |
| 3 | Solo baseline geometriche | `sec_protocol.tex`: ArcFace, Uni3D, OpenShape, spettrali (22 metodi) |
| 4 | Circolarità della D_GT | `sec_gt.tex`: arbitro su catture reali ripetute; `sec_protocol.tex`: riconoscimento senza GT, studio umano v2 |
| 5 | Cross-topologia solo dentro un 3DMM | `sec_protocol.tex`: test su 3DMM mai visti e su dati reali |
| 8 | Bias demografico | `sec_protocol.tex`: due gruppi di medie, gruppo est-asiatico come ipotesi non verificata |
| 9 | Identità di training e test distinte | `sec_protocol.tex`: quasi-duplicati E8, FaMoS distanza minima 2.17 mm |
| 11 | Novità limitata | `sec_gt.tex`: principio dell'invarianza minima e GT form/shape (nessun benchmark le separa) |

Non coperte qui: 1 (scala dei dati), 6 (espressioni, numeri), 7 (backbone e frame), 10 (rilascio).

## Chiavi bib da aggiungere

Non sono in `references.bib` né in `refs_cvpr_add.bib`. DOI verificati da `literature/GT_FORM_SHAPE_2026-10-09.md`:

| Chiave | Riferimento |
|---|---|
| `kendall1984shape` | Kendall 1984, doi:10.1112/blms/16.2.81 |
| `kendall1989survey` | Kendall 1989, doi:10.1214/ss/1177012582 |
| `dryden2016statistical` | Dryden & Mardia 2016, doi:10.1002/9781119072492 |
| `klingenberg2016size` | Klingenberg 2016, doi:10.1007/s00427-016-0539-2 |
| `klingenberg2020pinocchio` | Klingenberg 2020, doi:10.1007/s11692-020-09520-y |
| `walker2000pinocchio` | Walker 2000, doi:10.1080/106351500750049770 |
| `vonCramon2007pinocchio` | von Cramon-Taubadel et al. 2007, doi:10.1002/ajpa.20616 |
| `siegel1982resistant` | Siegel & Benson 1982, Biometrics 38:341, PMID 6810969 |
| `rohlf1990extensions` | Rohlf & Slice 1990, doi:10.2307/2992207 |
| `lele1991edma` | Lele & Richtsmeier 1991, doi:10.1002/ajpa.1330860307 |
| `russ2006normalization` | Russ et al. 2006, doi:10.1109/CVPR.2006.13 |
| `cole2016heritability` | Cole et al. 2016, doi:10.1371/journal.pgen.1006174 |

**Da verificare** (non nella nota di letteratura): `owen2007pigeonhole` (Owen 2007, citato dal protocollo dello
studio umano; presumo "The pigeonhole bootstrap", Ann. Appl. Stat.), `bfm2019`, `gnm`, `flame`, `famos`, `hifi3d`,
`faceverse`, `uni3d`, `openshape`, `reuter2006shapedna`, `sun2009hks`, `aubry2011wks`. Già presenti: `now`, `REALY`,
`mica`, `facescape`, `ict-facekit`, `wuu2022multiface`, `deng2019arcface`, `diffusionet`.
