# ULRSSM: distanze per coppia e via template (concorrente "corrispondenza densa imparata")

Metodo: Cao, Roetzer, Bernard, "Unsupervised Learning of Robust Spectral Shape Matching", ACM TOG (SIGGRAPH) 2023.
Codice ufficiale (MIT) clonato FUORI dal repo e importato (`ULRSSM_DIR`, default `~/ext/ulrssm`, commit
`0625cbcbb7cd007893dd665b21ef55b275563472`); la colla e' in `aau/competitors/ulrssm/` (docstring dei moduli per le
definizioni esatte). Pesi, mesh e npz NON sono in git.

## Cosa c'e' qui

| file | contenuto | in git |
| --- | --- | --- |
| `selection.json` | regola e scelta fra ingresso `xyz` e `wks` (errore di validazione sui generatori, prima della valutazione) | si' |
| `<tag>/<vista>.json` | tempi per mesh e per coppia, mesh fallite, diagnostica della rigida robusta, NaN | si' |
| `<tag>/spearman.csv`, `<tag>/paired.csv`, `<tag>/results.md`, `<tag>/paired_info.json` | Spearman FR / SR con IC, delta appaiati, tabelle | si' |
| `<tag>/<vista>.npz` | matrici simmetriche (mesh x mesh) e configurazioni trasferite | no (`*.npz`) |
| `<tag>/pairs_<vista>.csv` | le distanze per coppia in formato lungo (sotto) | no (`pairs_*.csv`) |
| `<tag>/parts/` | uscite grezze dei job GPU | no |
| `smoke/` | controlli della mappa e tempi di prova | si' |

`<tag>`: `faces_xyz`, `faces_wks` (ULRSSM riaddestrato sui generatori di training, ingresso xyz o WKS),
`zeroshot_faust` (pesi pubblici FAUST, controllo).

## Formato delle distanze per coppia (`pairs_<vista>.csv`)

Una riga per coppia NON ordinata di mesh, chiavi uguali alle righe di E12 / delle baseline parametriche:

    subject_a,topology_a,subject_b,topology_b,d_pair_fr,d_pair_sr,d_tpl_fr,d_tpl_sr

- coppie: `ulr_common.pairs` (= `gw_percettivi/gp_common.pairs`): tutte le i < j, nell'ordine delle mesh della vista
  (soggetti di `blmm.subjects` ordinati, topologie `crop, down8k, noisy, original, remesh, up60k`), con soggetti
  diversi E topologie diverse: 148.500 per vista, crop compreso (= all_cross). FaMoS TEST: le 105 coppie delle 15
  scansioni di galleria (`topology = scan_gallery`, soggetto = `view_id`).
- le distanze sono simmetriche: la riga (a, b) di all_cross e la (b, a) leggono la stessa cella (cercare entrambi
  gli ordini, oppure usare le matrici dell'npz con `subjects`/`topologies` e l'`Index` di `zs_expr_summarize`).
- `d_pair_fr` (mm): variante (a), mappa punto-a-punto di ULRSSM nei due versi, rigida pesata SENZA scala sui punti
  corrispondenti, RMS pesato per area (aree baricentriche della mesh d'arrivo), media dei due versi.
- `d_pair_sr` (mm alla taglia di riferimento): lo stesso con ogni mesh moltiplicata per CS_ref / CS_i (centroid size
  robusta di `blmm.mesh_scalars`, CS_ref del dominio da `blmm.params`).
- `d_tpl_fr` (mm), `d_tpl_sr` (d_P, adimensionale, come la GT SR): variante (b), ogni mesh mappata sul template
  fisso (media di GNM Head, generatore di training), i suoi vertici trasferiti sui 1478 punti della regione unificata
  e passati alla funzione della GT di valutazione importata senza modifiche (`train_fr_sr.factor`:
  `cgt.Canon.rigid_robust` verso mu, poi a_i per FR e pre-forme z_i = (a_i - c_i) / S_i per SR; distanze con la
  formula di `train_fr_sr.save_eval`). Controllo `gt_check` in `<vista>.json`: la stessa funzione sui punti nativi di
  FaMoS TEST ridà `famos_test_{fr,sr}`.
- `NaN`: mesh fallita in preparazione (elencata in `<vista>.json`).

Rieseguire su un dominio nuovo (es. Ava-256, NON eseguito): `ULR_SPEC=<json>` nei passi `eval` e `merge` di
`aau/competitors/ulrssm/ulr.sbatch` (formato della spec nel docstring di `ulr_common.py`, lo stesso di
`gw_percettivi`); lo Spearman di `ulr_paired.py` usa le righe e le GT delle viste di questo repo.
