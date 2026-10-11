# Diagnostica D3, emendamento 1 (prima dei numeri di D3): disegno leave-one-generator-out, teste (i)-(iii)

Scritto l'11 ottobre 2026 dal coder, su richiesta del PI, dopo il protocollo (`PROTOCOL_D3.md`, sha256 `c75619e6...`,
commit 21afea8) e PRIMA di qualunque numero di D3: nessuna testa e' stata valutata, nessuno Spearman o delta di D3
esiste. L'impronta sha256 di questo file sta nel messaggio del commit che lo introduce e in
`PROTOCOL_D3_emendamento_1.sha256`. Motivo: i fatti trovati dal critic su D1/D2 (sez. 1) rendono il disegno del
protocollo in parte prevedibile e in parte sbagliato. Il nuovo disegno (sez. 2-7) SOSTITUISCE quello del protocollo
dove lo dice la sez. 2.

**Cosa e' stato calcolato dopo il protocollo** (nessun numero di D3):
- FaMoS TRAIN (sez. 3 (b) del protocollo, invariata): mesh, tabella, GT e operatori (job 1068058; K3 passa, scarto
  3.1e-7 mm), embedding dei tre bracci (job 1068061, caduto dopo il primo braccio con codice 127 all'avvio di
  singularity su nv-ai-04, e job 1068067 con i tentativi ripetuti);
- embedding di C3M e123 su FaceVerse neutra (job 1068067, passo `fvn` di `eval_body.sh`); quel passo scrive anche, di
  serie, `d3/fvn/form/v3factorizedc3me123/form_spearman.csv` e `form_eval.json` (Spearman di d_P e d_F del braccio,
  valori di RIFERIMENTO, non di una testa): il coder non li ha letti;
- sonde tecniche: job 1068062 (caricamento dei domini e tempi dei fit, nessuno Spearman stampato), job 1068087
  (copertura delle tabelle di scala e soggetti non valutati dei pool, sez. 3);
- il job `stats` del vecchio disegno (1068068) e' stato cancellato prima di partire: nessun log, nessuna uscita.

## 1. Fatti e numeri gia' visti (anteprime del critic, dichiarate)

Anteprime del critic su D1/D2, in `/home/create.aau.dk/ga41wf/critic_scratch_657e6a75/diagD/` (fuori dal repo:
`d1_probe.json`, `d2_exp*.json`, `d2_variety*.json`, `critic-*.out`). Sono sonde ridge sui vettori GT (la sonda di
D2), non le teste di questo emendamento; s1234 / s2345 dove ci sono entrambi:
- **P1 domini.** FaceScape dev, HIFI3D e FaceVerse NON sono dati reali: sono campioni di 3DMM (FaceScape bilineare
  v1.6, AI-NEXT, FaceVerse simple v2). L'unico dominio reale e' FaMoS. Le parole "reale" del protocollo (sez. 1, 3, 8:
  "FaceScape reale", "reale -> reale") sono sbagliate e si correggono qui: FaceScape, HIFI3D e FaceVerse sono "3DMM non
  visti dai bracci"; "reale" e' solo FaMoS.
- **P2 generatori visti.** Sonda in dominio meno d_P: ict +0.001 / -0.006, gnm -0.005 / 0.000; FLAME 2023 suddiviso:
  sonda 0.913 / 0.907 contro d_P 0.806. Il gap e' della testa e dipende dal generatore.
- **P3 trasferimento.** FaceScape -> HIFI3D SR 0.598 / 0.523 contro d_P 0.622 / 0.613; HIFI3D -> FaceScape 0.653 /
  0.675 contro 0.747 / 0.753; sonda congiunta 50 + 50: FaceScape 0.842 / 0.833, HIFI3D 0.725 / 0.691. FLAME ->
  FaceScape 0.797 (s1234, d_P 0.747), FLAME -> HIFI3D 0.567 (d_P 0.622); ict + gnm + FLAME -> FaceScape 0.834 / 0.825,
  -> HIFI3D 0.612 / 0.593; ict + gnm + FLAME + HIFI3D -> FaceScape 0.816 / 0.800; ict + gnm + FLAME + FaceScape ->
  HIFI3D 0.606 / 0.585; FaceScape + HIFI3D -> FLAME 0.823 / 0.822 (d_P 0.806); ict + gnm + FaceScape + HIFI3D ->
  FLAME 0.837 / 0.816. Su HIFI3D nessuna sorgente aiuta.
- **P4 allineamenti non supervisionati** (covarianza del dominio di test): CORAL 0.39-0.59 su FaceScape (d_P 0.745 /
  0.751) e 0.37-0.50 su HIFI3D (0.619 / 0.611); sbiancamenti 0.38-0.62. Mahalanobis con l'inversa della covarianza
  INTRA-soggetto stimata sul dominio di test con le etichette di soggetto: FaceScape 0.792-0.813 (+0.05 / +0.07),
  HIFI3D 0.567-0.632 (da -0.044 a +0.012).
- **P5 invarianza.** FaceScape: sonda addestrata sulle sole original 0.851 / 0.853 contro 0.889 / 0.886 (circa 0.04
  del guadagno e' invarianza alla discretizzazione); d_P sulle righe original-original 0.777 / 0.777 e sugli
  embedding mediati sulle etichette 0.801 / 0.791 (righe incrociate 0.745 / 0.751); sonda 0.891 / 0.888 e 0.898 /
  0.894. HIFI3D: d_P 0.619 / 0.611 (incrociate), 0.639 / 0.622 (original), 0.636 / 0.621 (mediati); sonda 0.717 /
  0.696, 0.723 / 0.701, 0.725 / 0.703.
- **P6 regola.** Per supervisione e statistiche di FaceScape, HIFI3D e FaceVerse si usano solo soggetti NON valutati
  dei pool, mai i 100 di test.

**Cosa e' quindi prevedibile** (scritto ora): su HIFI3D la testa LOGO probabilmente non passa (P3); su FaceScape una
testa con FLAME fra le sorgenti probabilmente guadagna (P3); su FLAME le sorgenti 3DMM danno +0.01-0.03 alle sonde
(P3); CORAL peggiora (P4). La regola della sez. 6 e' fissata sapendolo.

## 2. Cosa cambia e cosa resta del protocollo

SOSTITUITE: sez. 1 (domande: sez. 4 e 6 qui), sez. 3 (domini di training: le sorgenti della sez. 3 qui; resta la
pipeline di FaMoS TRAIN, gia' eseguita), sez. 5 (resta solo CORAL, esplorativa, sez. 4 qui; lo sbiancamento (e') esce),
sez. 8 (letture R_A, R_S, R_E: sez. 6 qui). RESTANO: sez. 2 (bracci, embedding, d_P, `form_cal`, B), sez. 4 (famiglia
della testa, perdita, inizio, ottimizzatore, griglia, scelta, rifit, c_h; cambiano solo i fold, sez. 4 qui), sez. 6
(viste FaceScape, HIFI3D, FaceVerse: righe, maschere, semi, bootstrap), sez. 7 (controlli, estesi in sez. 7 qui), sez.
9-10.

## 3. Sorgenti per la testa

| sorgente | generatore | soggetti x etichette | embedding | GT d_P |
|---|---|---|---|---|
| `bfm`, `ict`, `gnm` | held-out sintetici di D1 (visti dai bracci) | 100 x 5 ciascuno | calibrazione (`calib_heldout`) | `c3f/gt_sr.npz` x `dP_per_unit` |
| `flame2023_s1` | FLAME 2023 Open, D1 (suddiviso) | 200 x 5 | `d1/emb` | `d1/gt_flame2023_s1.npz` |
| `facescape` | FaceScape bilineare v1.6, pool NON valutato | 200 x 5 | NUOVI | `CANONICAL_GT/eval/facescape_sr.npz` x `dP_per_unit` |
| `hifi3d` | AI-NEXT (HIFI3D), pool NON valutato | 200 x 5 | NUOVI | `hifi3d_sr.npz` |
| `faceverse` | FaceVerse simple v2, pool NON valutato (mesh neutre) | 200 x 5 | NUOVI | `faceverse_sr.npz` |
| `famos` | FaMoS TRAIN, reale | 80 x 5 | `d3/famos/emb` | `d3/famos/gt_famos.npz` |

**Pool non valutati.** Ogni pool ha 500 identita' (`id9x0000`-`id9x0499`); le 100 valutate sono quelle di
`zs_stage.select_subjects(seme 1234)`, cioe' i soggetti degli store ufficiali; le sorgenti sono **i primi 200 per id
fra i 400 non valutati** (da `id9x0000` a `id9x0248`, job 1068087). Le mesh sono gli STESSI file di `eval_view/npz`
(le 6 topologie di `make_zs_topologies.py` esistono per tutto il pool; qui senza crop), messi in scena come
`zs_stage.py` (FaceVerse con `--flip-faces`: F[:, ::-1], come gli store neutri ufficiali); operatori `areanorm` k_eig
128; embedding `eval_v3.py -- zs_embed.py` con gli argomenti di D1 e le tabelle di scala ufficiali delle viste
(`factorized/scale_tables/devfs_eval.npz`, `hifi3d_eval.npz`, `faceverse_neutral/scale_tables/fv_eval.npz`: coprono
tutte le 3.000 mesh di ogni pool). **Perche' 200 e non 400** (stima misurata): gli operatori pesano 70, 50 e 170 MB
per soggetto con 6 topologie (FaceScape, HIFI3D, FaceVerse; cache esistenti), quindi 400 per pool sono circa 98 GB
temporanei; l'embedding di FaceVerse costa circa 1.1 s per mesh su A100 (job 1068067), quindi circa 3 ore-GPU per
400 per pool e tre bracci; la CV costa come n^2 per blocco (circa 3 volte). 200 dimezza tutto e da' gia' a ogni pool
il doppio dei soggetti di bfm, ict, gnm. **Soggetti di controllo**: i primi 5 soggetti VALUTATI di ogni pool
(ordinati) passano per la stessa pipeline SOLO per il controllo K7 (embedding contro lo store ufficiale); mai in
teste o statistiche.

## 4. Bersagli, schema LOGO, teste

**Bersagli primari** T: `facescape` (i 100 soggetti valutati, righe `nocrop_cross` di `fact_paired`, 88.725),
`hifi3d` (98.224), `faceverse` (vista NEUTRA `faceverse_neutral`, 99.000), `flame2023_s1` (le 200 identita' di D1,
righe di D1: 398.000; conteggi del bootstrap di D1, gruppo 3, seme 20261113). **Secondari**: `faceverse` con
espressioni (stessa testa della neutra) e `famos` (FaMoS TRAIN come bersaglio, 80 persone, righe persone diverse ed
etichette diverse, 63.200; bootstrap per persona `diag.boot_counts(80, 0, seed=20261123)`: "validazione per
persona" = la testa non ha mai visto FaMoS e l'IC ricampiona le persone).

**LOGO.** Per ogni bersaglio T la testa si addestra su TUTTE le sorgenti tranne il generatore di T (T = FaceScape,
HIFI3D, FaceVerse: il loro pool; T = FLAME: `flame2023_s1`; T = FaMoS: `famos`) e si valuta sulle righe di test di T.
Accanto, per i 4 bersagli primari, la variante **"senza FaMoS"** (tutte tranne T e `famos`), per isolare il contributo
del dato reale. I soggetti di test di FaceScape, HIFI3D e FaceVerse non sono mai fra le sorgenti (verificato).

**Teste.**
- **(i) appresa**: la famiglia del protocollo (d_h^2 = alpha^2 ||dx||^2 + ||W dx||^2), perdita stress, griglia di 31
  configurazioni, sui blocchi delle sorgenti (uno per sorgente, pesi uguali). CV per soggetto DENTRO le sorgenti, 5 fold
  x 3 ripetizioni; **unico cambiamento**: i fold di ogni sorgente vengono da `default_rng(SeedSequence([20261121, q,
  j]))`, j = indice della sorgente nella lista (`bfm`, `ict`, `gnm`, `flame2023_s1`, `facescape`, `hifi3d`,
  `faceverse`, `famos`), cosi' una sorgente ha gli stessi fold qualunque siano le altre. Punteggio, scelta, rifit su
  tutte le sorgenti e c_h come la sez. 4 del protocollo (mediane pesate, pesi uguali per sorgente).
- **(ii) Mahalanobis intra-soggetto**, senza GT: per ogni sorgente i residui di u dalla media del soggetto (5
  etichette), covarianza di Ledoit-Wolf dei residui; Sigma_w = media sulle sorgenti; d = ||Sigma_w^{-1/2} du|| x
  dp_per_unit; c dalle coppie delle sorgenti (regola di `fact_calib`). Solo sorgenti: niente dal bersaglio.
- **(iii) d_P** (e `form_cal`) del braccio: il riferimento.
- **CORAL, solo esplorativa** (anteprima negativa, P4): come la sez. 5 del protocollo (riferimento = covarianza media di
  bfm, ict, gnm), ma la covarianza del bersaglio viene dai 200 soggetti NON valutati del suo pool (P6: mai i soggetti di
  test), senza etichette; solo per FaceScape, HIFI3D e FaceVerse (FLAME e FaMoS non hanno un pool separato).

## 5. Righe, SR e FR, statistiche

- **Righe incrociate** (primarie): quelle della sez. 6 del protocollo (e di D1 per FLAME, `diag.rows` per FaMoS).
- **Invarianza contro pesatura** (descrittivo): righe **original-original** (una per coppia di soggetti di test, le
  due mesh `original`) e righe **mediate sulle etichette** (z = [s, u] mediato sulle 5 etichette del soggetto, una
  riga per coppia), con gli stessi conteggi del bootstrap del bersaglio. Se il guadagno sulle righe incrociate e' molto
  piu' grande di quello sulle righe mediate, e' invarianza; se e' simile, e' pesatura.
- **SR e FR separate.** SR: d_h con la GT SR. FR: d_F,h = sqrt((S_i - S_j)^2 + S_i S_j (c_h d_h)^2), S del modello, c_h
  dalle sole sorgenti, con la GT FR; accanto **"FR senza taglia"**: d_h letta con la GT FR.
- **Bootstrap**: 1000 repliche per bersaglio coi suoi conteggi; media sui 4 bersagli primari replica per replica
  (soggetti disgiunti: indipendenti). IC 95% percentile, condizionato alle teste addestrate.
- **Delta** appaiati: testa - d_P (SR), d_F testa - `form_cal` (FR), "senza taglia" testa - d_P (entrambe con FR);
  (i) - (i) senza FaMoS; testa - B (viste di `fact_paired`, descrittivo).

## 6. Letture (dichiarate qui, prima dei numeri)

**R_LOGO (primaria)**: testa (i), sorgenti "tutte tranne T" (con FaMoS), righe incrociate, i 4 bersagli primari. Per
braccio: n_SR = bersagli con delta SR >= +0.05 e IC > 0; FR_ok = delta FR >= -0.03 (stima puntuale) su tutti e 4.
- **PASSA**: n_SR >= 3 e FR_ok; **SR SI, FR NO**: n_SR >= 3, non FR_ok; **NO**: n_SR < 3.
- Verdetto: quello comune ai due semi (s1234, s2345); semi diversi: **NON RISOLTO**. C3M e123 descrittivo.
- Accanto: delta SR medio sui 4 bersagli con IC, e delta e IC per bersaglio.
Perche': +0.05 con IC che esclude lo 0 come nel protocollo (soglia di R3 di D2); "3 su 4": una testa addestrata una
volta deve servire su un generatore nuovo in generale, non su uno solo, e un insuccesso si tollera (P3 rende probabile
quello su HIFI3D: dichiarato); FR su tutti e 4: nessun bersaglio peggiorato oltre il rumore di seme (sez. 8 del
protocollo). **Lettura**: PASSA = una testa lineare appresa su molti generatori e' la leva economica per il paper; NO
= il gap della testa dipende dal generatore e non si chiude con le sorgenti disponibili.

**Secondarie** (stessa regola, verdetti descrittivi): (ii) intra-soggetto; (i) senza FaMoS. **Contributo del reale**:
(i) - (i) senza FaMoS per bersaglio, con IC, nessuna soglia.

**Curva sul numero di sorgenti** (secondaria): per bersaglio primario e braccio, la testa (i) con gli iperparametri
scelti per "tutte tranne T" (nessuna CV nuova: dichiarato) rifittata su OGNI sottoinsieme di k sorgenti delle 7
disponibili, k in {1, 2, 4} (7, 21 e 35 sottoinsiemi), e k = 7 (la testa di R_LOGO); lo stesso per (ii). Delta SR
puntuale sulle righe incrociate di T; per k media, minimo e massimo sui sottoinsiemi. "Il guadagno cresce con le
sorgenti" se la media non decresce da k = 1 a 2, 4, 7 su almeno 3 bersagli su 4 in entrambi i semi (descrittivo).

Descrittivi: invarianza contro pesatura, FaceVerse con espressioni, FaMoS come bersaglio, CORAL, B.

## 7. Controlli aggiunti

- **K1** esteso: riferimenti dei bracci su FLAME (d_P con SR, `form_cal` con FR, conteggi di D1) = `d1_spearman.csv`
  (scarto <= 1e-9).
- **K4** esteso ai pool: 205 soggetti x 5 etichette x 3 bracci per pool, Z finiti, checkpoint della calibrazione.
- **K6** esteso: sorgenti dei pool disgiunte dai soggetti valutati; la sorgente di T esclusa dalla sua testa.
- **K7 riproduzione**: max |z nostro - z ufficiale| per etichetta sui 5 soggetti di controllo di ogni pool <= 1e-2
  (le mesh sono gli stessi file; in D1 lo scarto era <= 3.1e-3 per le mesh identiche); se non passa ci si ferma.

Uscite in piu': `d3_curve.csv`; pool in `datasets/DIAG_D3/pools` (link o copie con le facce invertite) e operatori in
`datasets/V3_OPS_CACHE/diag_d3/pools` (circa 50 GB, fuori da git, cancellati a fine lavoro), embedding in
`d3/pools/<pool>/emb` (npz fuori da git). Restano esclusi FaMoS TEST e Ava-256.
