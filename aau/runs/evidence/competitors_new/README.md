# Concorrenti nuovi: Gromov-Wasserstein (POT) e feature percettive su render (DINOv2, CLIP, LPIPS, DreamSim)

(ULRSSM, nella stessa cartella, ha il suo README in `ulrssm/`.)

Codice: `aau/competitors/gw_percettivi/` (`gp_common.py`, `gp_gw.py`, `gp_percettivi.py`, `gp_eval.py`, `gp.sbatch`;
venv e pesi fuori dal repo: `gp_env.sh`, `gp_setup.sbatch`). Definizioni complete nelle docstring dei moduli; qui il
formato delle uscite, le scelte e i limiti. Numeri: `results.md` (generato) e `spearman.csv`.

## Formato delle uscite

Per metodo `m` in `gw`, `dinov2`, `clip`, `lpips`, `dreamsim` e vista `v` in `hifi3d`, `facescape`, `faceverse`
(con espressioni), `faceverse_neutral`, `famos` (TEST, le 15 scansioni di galleria):

- `m/v.npz` (fuori da git): `subjects`, `topologies`, `paths` (n mesh, nell'ordine di `blmm.subjects(v)` x
  `blmm.TOPOLOGIES`, crop compreso; FaMoS: `bp.meshes("famos")`, topologia `scan_gallery`), `D_fr`, `D_sr` (n x n,
  simmetriche, diagonale 0, NaN dove la distanza non e' calcolata), `cs` (centroid size robusta, mm). Percettivi anche
  `E_fr`, `E_sr` (n, 3 viste, d: embedding L2 per vista). Le matrici GW hanno valori solo sulle coppie di
  `gp_common.pairs` (soggetti diversi e topologie diverse; FaMoS: soggetti diversi), le percettive su tutte.
- `m/pairs_v.csv` (fuori da git per la regola `pairs_*.csv`): una riga per coppia non ordinata di `gp_common.pairs`
  con `subject_a, topology_a, subject_b, topology_b, d_fr, d_sr`. Sono le 148.500 coppie di all_cross per vista (crop
  compreso) e le 105 di FaMoS. Le distanze sono simmetriche: la riga (a, b) e la riga (b, a) di all_cross leggono la
  stessa cella, quindi per unirle alla matrice: `M[pos[(subject_a, topology_a)], pos[(subject_b, topology_b)]]` sulle
  chiavi di `zes.Index(blmm.subjects(v))` (come `blmm_eval.add_columns`).
- `m/v.json`: configurazione, tempi, controlli (mesh fallite, mesh fuori quadro, controllo LPIPS Gram contro forward).
- Colonne nella valutazione: `<m>_fr` (render o matrici in mm: confronto con GT FR) e `<m>_sr` (mesh a CS_ref: con GT
  SR). `spearman.csv`: `domain, group, gt, kind, column, method, mode, point, ci_low, ci_high, n_boot, n_rows, seed`
  (`kind`: `rho` con GT FR e SR, `partial` dato l'oracolo, `q20` quintile basso: `fact_paired._rep`).

## Scelte

- **Mesh in mm**: `blmm.to_mm` (una trasformazione per dominio nel frame canonico, nessuna rigida per mesh e nessuna
  GT), centrata sul baricentro pesato per area. SR: x CS_ref / CS_i, CS_i = centroid size robusta di
  `blmm.mesh_scalars` (cache della vista se c'e').
- **Posa**: gia' canonica, verificata senza GT (`percettivi/posecheck_<v>.json`): angolo dell'ICP rigido verso la
  original di un soggetto NON valutato, e punta del naso (z massima) sul piano mediano.
- **GW**: misura = FPS sui vertici + pesi di Voronoi d'area (densita' dei vertici indifferente), matrici euclidee o
  geodetiche (metodo del calore), solutore POT; configurazione scelta sul pilota (`gw/pilot_choice.json`,
  `gw/pilot_scores.csv`), con 12 soggetti NON valutati per vista e la loro GT, mai quella dei valutati.
- **Percettivi**: renderer e modelli delle baseline percettive gia' nel repo (`perceptual_matrix.py`), camera unica per
  dominio in mm (semilato 1,3 L_d), 3 viste (yaw 0, +-30), nessun iperparametro scelto sui dati.

## Limiti dichiarati

- GW e' invariante alle isometrie, compresa la RIFLESSIONE: un volto e il suo specchio hanno distanza 0.
- Sul crop GW confronta insiemi diversi della superficie: misura la differenza di supporto, non solo d'identita'.
- I percettivi sono ciechi alla taglia per costruzione salvo il render in scala metrica: la colonna `_fr` (camera in
  mm) la vede, la `_sr` no.

## Pilota di GW (`gw/pilot_choice.json`, `gw/pilot_scores.csv`)

12 soggetti NON valutati per vista (`blmm.template_subjects`), 6 coppie ordinate di topologie per coppia di soggetti
(`gp_gw.pilot_pairs`, crop compreso), 1.584 coppie in tutto, GT FR / SR di quei soggetti. Punteggio = media sulle 4
viste di (rho(mm, GT FR) + rho(CS_ref, GT SR)) / 2. Budget fissato prima dei punteggi: mediana <= 1 s per soluzione
(un thread), perche' la valutazione sono 1,19 milioni di soluzioni. Griglia fissata dai tempi misurati prima (job 1070585,
1070589): n = 512 fuori budget (CG 3-6 s per soluzione), entropico fuori budget a ogni n (2-18 s), nel pilota solo come
confronto descrittivo.

| configurazione | punteggio | s per soluzione (mediana) |
|---|---|---|
| euclid, n = 128, CG | 0.154 | 0.056 |
| **euclid, n = 256, CG (scelta)** | **0.183** | 0.460 |
| geo, n = 128, CG | 0.060 | 0.054 |
| geo, n = 256, CG | 0.061 | 0.443 |
| euclid, n = 128, entropico eps 0.005 (fuori budget) | 0.231 | 6.8 |
| euclid, n = 256, entropico eps 0.005 (fuori budget) | 0.234 | 13.6 |
| geo, n = 128, entropico eps 0.005 (fuori budget) | 0.046 | 5.4 |
| geo, n = 256, entropico eps 0.005 (fuori budget) | 0.029 | 11.0 |

L'entropico euclideo batte il CG sul pilota di +0,05 (soprattutto su FaceVerse e sulle colonne SR): il CG parte
dall'accoppiamento prodotto e si ferma in minimi locali peggiori. Non e' stato eseguito sulle viste valutate (circa
4.500 ore-core su CPU con POT); e' il primo candidato se si vuole un GW piu' forte (serve un solutore entropico a
lotti su GPU, fuori da POT). Le geodetiche (metodo del calore) sono peggiori delle euclidee in ogni configurazione.

## Controlli

- Posa (`percettivi/posecheck_*.json`): angolo ICP mediano 2,1 (HIFI3D), 2,5 (FaceScape), 4,5 (FaceVerse con
  espressioni), 4,0 (neutra), 3,6 gradi (FaMoS); massimo 13,0 gradi (FaceVerse con espressioni); naso a 1-3 mm dal piano
  mediano, a meta' altezza. Nessuna mesh fuori quadro in nessuna vista e modo.
- LPIPS in forma chiusa contro `lpips.LPIPS.forward` su 20 coppie per vista, modo e yaw: scarto massimo 5,3e-8.
- Percorso `--spec` (dominio nuovo) provato su 3 soggetti valutati di HIFI3D con lo spec scritto a mano (frame di
  `frames.json`, L e CS_ref di `params.json`, taglia ricalcolata, non dalla cache): GW FR identico (scarto 0), GW SR
  scarto relativo 1,4e-7 (taglia ricalcolata contro cache 2,9e-8); percettivi scarto relativo <= 3e-3 (torch su CPU,
  nodi e numero di thread diversi). Uscite della prova cancellate.
- Valutazione: 148.500 / 148.500 righe per vista (49.500 col crop), 105 / 105 su FaMoS, nessuna distanza NaN.

## Job (12 ottobre 2026, solo CPU, partizione prioritized)

venv 1070542, 1070561, 1070564 (gli ultimi due: correzioni dello sbatch e delle versioni di peft / transformers);
posa 1070567-1070571; percettivi 1070576 (FaMoS), 1070582-1070584, 1070600 (HIFI3D, rilanciato: il 1070581 e' morto
con un segmentation fault di torch su a256-t4-04 durante DINOv2, nessuna uscita scritta); GW: preparazione del pilota
1070594-1070597, pilota 1070598 (38 min, 64 CPU), scelta 1070599, preparazione 1070613-1070617, distanze
1070635-1070645 e 1070647 (circa 25 min per parte, 64 CPU), unione 1070637, 1070640, 1070643, 1070646, 1070648;
valutazione 1070649.
