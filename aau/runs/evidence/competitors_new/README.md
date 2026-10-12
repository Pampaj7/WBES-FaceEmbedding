# Concorrenti nuovi: Gromov-Wasserstein (POT) e feature percettive su render (DINOv2, CLIP, LPIPS, DreamSim)

(ULRSSM, nella stessa cartella, ha il suo README in `ulrssm/`.)

Codice: `aau/competitors/gw_percettivi/` (`gp_common.py`, `gp_gw.py`, `gp_percettivi.py`, `gp_eval.py`, `gp.sbatch`;
venv e pesi fuori dal repo: `gp_env.sh`, `gp_setup.sbatch`). Definizioni complete nelle docstring dei moduli; qui il
formato delle uscite, le scelte e i limiti. Numeri: `results.md` (generato) e `spearman.csv`.

## Giro 2 (dopo il critic, 12 ottobre sera): regole scritte PRIMA di calcolare i piloti nuovi e i valutati

Il critic (BLOCCANTE, nessun errore di calcolo) ha chiesto: GW con partenza W2 nel frame canonico; percettivi anche
su render con normali per vertice e su render di profondita'; triangoli degeneri delle up60k; plug-in solo dichiarati.
Le scelte si fanno sui 12 soggetti NON valutati per vista del pilota, con la loro GT, mai con quella dei valutati:

- **GW**: griglia del pilota estesa con `cgw2` (CG da G0 = accoppiamento W2 nel frame canonico) e `cgmin` (minimo per
  coppia fra la partenza p qᵀ e quella W2; costo = somma), per euclid / geo x n = 128 / 256. Regola invariata: fra le
  configurazioni con mediana <= 1 s per soluzione, il punteggio massimo (media sulle 4 viste di
  (rho(mm, GT FR) + rho(CS_ref, GT SR)) / 2 sulle coppie del pilota); entro 0,01 dal migliore la piu' economica.
- **Percettivi**: tre varianti di render (`flat` = per faccia come il repo, con i degeneri corretti; `smooth` = normali
  per vertice; `depth` = profondita' metrica, stessa camera). Per ogni metodo la variante dichiarata e' quella col
  punteggio massimo sul pilota (stesso punteggio, tutte le coppie di soggetti e topologie diversi, crop compreso);
  entro 0,01 dalla migliore vince la prima nell'ordine flat, smooth, depth.
- Le varianti non dichiarate restano nelle uscite (`declared: false` in `fm_plugin.json`, etichetta "non dichiarata").

## Formato delle uscite

Per metodo `m` in `gw`, `dinov2`, `clip`, `lpips`, `dreamsim`, variante `q` (GW: `euclid_128_cgw2` dichiarata,
`euclid_256_cgw2`, `euclid_256_cg` = giro 1; percettivi: `flat`, `smooth`, `depth`) e vista `v` in `hifi3d`,
`facescape`, `faceverse` (con espressioni), `faceverse_neutral`, `famos` (TEST, le 15 scansioni di galleria):

- `m/q/v.npz` (fuori da git): `subjects`, `topologies`, `paths` (n mesh, nell'ordine di `blmm.subjects(v)` x
  `blmm.TOPOLOGIES`, crop compreso; FaMoS: `bp.meshes("famos")`, topologia `scan_gallery`), `D_fr`, `D_sr` (n x n,
  simmetriche, diagonale 0, NaN dove la distanza non e' calcolata), `cs` (centroid size robusta, mm). Percettivi anche
  `E_fr`, `E_sr` (n, 3 viste, d: embedding L2 per vista). Le matrici GW hanno valori solo sulle coppie di
  `gp_common.pairs` (soggetti diversi e topologie diverse; FaMoS: soggetti diversi), le percettive su tutte.
- `m/q/pairs_v.csv` (fuori da git per la regola `pairs_*.csv`): una riga per coppia non ordinata di `gp_common.pairs`
  con `subject_a, topology_a, subject_b, topology_b, d_fr, d_sr`: le 148.500 coppie di all_cross per vista (crop
  compreso) e le 105 di FaMoS. Distanze simmetriche: la riga (a, b) e la (b, a) di all_cross leggono la stessa cella.
- `m/q/v.json`: configurazione, tempi, controlli (mesh fallite, fuori quadro, triangoli degeneri, LPIPS Gram/forward).
- `m/q/fm_plugin.json`: marca la cartella come plug-in di `aau/full_matrix/fm_eval.py`, con etichetta, famiglia,
  letture (`fr` -> GT FR, `sr` -> GT SR) e `declared` (true solo per la variante scelta dalla regola del pilota; le altre
  hanno `false` e "(non dichiarata)" nell'etichetta). Provato con `fm_eval.plugin_columns`: 15 plug-in, 148.500 /
  148.500 righe di all_cross finite per vista. GW e' NaN sulle righe "stessa topologia" della matrice unificata (non
  richieste; ~29.700 coppie per vista, a n = 128 con partenza W2 meno di un'ora-core).
- Pilota e intermedi (misure GW, distanze dei piloti, parti, render di controllo) in `aau/runs/gw_percettivi/`, fuori da
  git e fuori da `competitors_new`. In `competitors_new` dei piloti restano solo `gw/pilot_choice.json`,
  `gw/pilot_scores.csv`, `percettivi/pilot_choice.json`, `percettivi/pilot_scores.csv` (nessun npz).
- `spearman.csv`: `domain, group, gt, kind, column, method, variant, declared, mode, point, ci_low, ci_high, n_boot,
  n_rows, seed` (`kind`: `rho` con GT FR e SR, `partial` dato l'oracolo, `q20` quintile basso: `fact_paired._rep`;
  `group`: senza crop, all_cross, righe col crop, senza crop senza noisy, `con <t>`).

## Scelte

- **Mesh in mm**: `blmm.to_mm` (una trasformazione per dominio nel frame canonico, nessuna rigida per mesh e nessuna
  GT), centrata sul baricentro pesato per area. SR: x CS_ref / CS_i, CS_i = centroid size robusta di
  `blmm.mesh_scalars` (cache della vista se c'e').
- **Posa**: gia' canonica, verificata senza GT (`percettivi/posecheck_<v>.json`): angolo dell'ICP rigido verso la
  original di un soggetto NON valutato, e punta del naso (z massima) sul piano mediano.
- **GW**: misura = FPS sui vertici + pesi di Voronoi d'area (densita' dei vertici indifferente), matrici euclidee o
  geodetiche (metodo del calore), solutore POT; configurazione scelta sul pilota (`gw/pilot_choice.json`,
  `gw/pilot_scores.csv`), con 12 soggetti NON valutati per vista e la loro GT, mai quella dei valutati. **Partenza del
  CG dichiarata (`cgw2`): G0 = accoppiamento W2 fra i campioni nel frame canonico**, cioe' la stessa posa canonica che
  usano tutti gli altri metodi; il valore resta un GW (stesso funzionale, minimo locale da una partenza migliore), ma
  non e' piu' "senza allineamento" in senso stretto: la partenza usa la posa dei dati.
- **Percettivi**: modelli delle baseline percettive gia' nel repo (`perceptual_matrix.py`), renderer del repo copiato
  in `gp_percettivi.render` con tre colorazioni (per faccia, normali per vertice, profondita'), camera unica per
  dominio in mm (semilato 1,3 L_d), 3 viste (yaw 0, +-30); variante dichiarata scelta sul pilota dei NON valutati.

## Limiti dichiarati

- GW e' invariante alle isometrie, compresa la RIFLESSIONE: un volto e il suo specchio hanno distanza 0 (la partenza W2
  sceglie di fatto il minimo locale nell'orientamento dei dati, ma il funzionale resta cieco allo specchio).
- Sul crop GW confronta insiemi diversi della superficie: misura la differenza di supporto, non solo d'identita'.
- I percettivi sono ciechi alla taglia per costruzione salvo il render in scala metrica: la colonna `_fr` (camera in
  mm) la vede, la `_sr` no.

## Pilota di GW (`gw/pilot_choice.json`, `gw/pilot_scores.csv`)

12 soggetti NON valutati per vista (`blmm.template_subjects`), 6 coppie ordinate di topologie per coppia di soggetti
(`gp_gw.pilot_pairs`, crop compreso), 1.584 coppie, GT FR / SR di quei soggetti. Punteggio = media sulle 4 viste di
(rho(mm, GT FR) + rho(CS_ref, GT SR)) / 2; budget fissato prima: mediana <= 1 s per soluzione (un thread). Giro 1: la
griglia senza `cgw2` / `cgmin` aveva scelto `euclid|256|cg`; giro 2: griglia estesa, stessa regola.

| configurazione | punteggio | s per soluzione |
|---|---|---|
| euclid, 128, CG da p qᵀ | 0.154 | 0.056 |
| **euclid, 128, CG da W2 (dichiarata: entro 0,01 dalla migliore e la piu' economica)** | **0.235** | 0.030 |
| euclid, 128, minimo delle due partenze | 0.236 | 0.090 |
| euclid, 256, CG da p qᵀ (giro 1) | 0.183 | 0.460 |
| euclid, 256, CG da W2 | 0.236 | 0.166 |
| euclid, 256, minimo delle due partenze | 0.232 | 0.652 |
| geo, 128 / 256, CG da W2 | 0.114 / 0.102 | 0.026 / 0.151 |
| geo, 128 / 256, CG da p qᵀ | 0.060 / 0.061 | ~0.05 / 0.44 |
| euclid, 128 / 256, entropico eps 0.005 (fuori budget) | 0.231 / 0.234 | 6.8 / 13.6 |

La partenza W2 chiude il distacco dall'entropico (+0,05 sul CG da p qᵀ) a un costo 15-50 volte piu' basso. Il
minimo delle due partenze non aggiunge nulla. Sulle viste valutate c'e' anche `euclid_256_cgw2` (non dichiarata) e il
`euclid_256_cg` del giro 1, per isolare l'effetto della partenza.

## Pilota dei percettivi (`percettivi/pilot_choice.json`, `percettivi/pilot_scores.csv`)

Stessi 12 soggetti NON valutati x 6 topologie per vista, tutte le coppie di soggetti e topologie diversi (crop
compreso), stesso punteggio; regola per metodo: il massimo, pari entro 0,01 -> ordine flat, smooth, depth.

| metodo | flat | smooth | depth | dichiarata |
|---|---|---|---|---|
| LPIPS | 0.151 | 0.175 | **0.225** | depth |
| DINOv2 | 0.062 | 0.080 | **0.143** | depth |
| CLIP | 0.115 | **0.125** | 0.104 | smooth (flat a 0,010 03: appena fuori dal pari) |
| DreamSim | 0.114 | **0.131** | 0.109 | smooth |

Su HIFI3D da solo la profondita' e' la migliore per tutti e quattro i metodi; sulla media delle 4 viste e dei due modi
CLIP e DreamSim preferiscono `smooth`. La regola vale cosi'.

## Controlli

- Posa (`percettivi/posecheck_*.json`): angolo ICP mediano 2,1 (HIFI3D), 2,5 (FaceScape), 4,5 (FaceVerse con
  espressioni), 4,0 (neutra), 3,6 gradi (FaMoS); massimo 13,0 gradi; naso a 1-3 mm dal piano mediano. Nessuna mesh
  fuori quadro in nessuna vista, modo e variante.
- Renderer copiato (`gp_percettivi.render`): sulle original identico a `render_mesh` (scarto 0 su 4 mesh del pilota).
  Triangoli degeneri: le up60k hanno schegge quasi collineari (HIFI3D pilota: 1.103 triangoli sotto 1e-4 mm², nessuno
  fra 1e-4 e 1e-2 mm², area mediana 0,52 mm²); soglia area < 1e-3 x mediana, normale = quella per vertice. Pixel
  "puntino scuro" sulle up60k da 76-126 a 24-86, pari alle original degli stessi soggetti (25-84). Frazione di
  triangoli corretti sulle up60k valutate di HIFI3D: mediana 2,4%. La prima soglia (1e-6) ne prendeva solo un quarto:
  job percettivi 1070850-1070859 annullati e rifatti.
- LPIPS in forma chiusa contro `lpips.LPIPS.forward`: scarto massimo 5,3e-8.
- Percorso `--spec` (giro 1): GW FR identico alla vista nota, SR 1,4e-7, percettivi <= 3e-3.
- Valutazione: 148.500 / 148.500 righe per vista (49.500 col crop), 105 / 105 su FaMoS, nessuna distanza NaN.

## Job (12 ottobre 2026, solo CPU, partizione prioritized)

Giro 1: venv 1070542, 1070561, 1070564; posa 1070567-1070571; GW pilota 1070594-1070599, distanze `euclid_256_cg`
1070613-1070648; valutazione 1070649. Giro 2: GW pilota W2 1070860 (90 s, 64 CPU), scelta 1070861; GW `euclid_128_cgw2`
1070891, 1070896, 1070901, 1070906, 1070911 (circa 3 min per vista, 64 CPU), `euclid_256_cgw2` 1070893-1070913 (circa
9 min per parte), unioni 1070862-1070866 (giro 1 nella cartella della variante) e 1070892-1070914; percettivi pilota
1070929-1070932, scelta 1070933, valutati 1070934-1070938 (circa 50 min per vista, 32 CPU; tre varianti); valutazione
1070939 (7 min, 64 CPU).
