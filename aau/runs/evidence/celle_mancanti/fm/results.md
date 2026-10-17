# Matrice completa e confrontabile: Spearman con FR e SR, delta appaiati

Codice `aau/full_matrix/` (`fm_eval.py`, definizioni nel docstring), numeri in questa cartella. Ogni metodo di un dominio e' valutato sulle STESSE righe (maschera comune: righe con tutte le colonne del dominio finite), con le STESSE 1000 repliche per soggetto (seme della T1 del dominio, lo stesso per tutti gli insiemi). IC 95% percentile. Delta appaiati = differenza replica per replica. Plug-in di `competitors_new/`: nel passo `comune` solo le varianti dichiarate (GW, percettivi, ULRSSM `faces_xyz`, NoW originale, NoW metrical, REALY e Stirling coi landmark esatti); le non dichiarate e i landmark rilevati (`lmk_det`, colonna aggiuntiva marcata) in passi e sezioni separati. **Il crop e' fuori dalla tesi principale**: le sezioni principali usano solo righe senza crop (nocrop, per perturbazione senza crop, stessa topologia senza crop); crop, all_cross e stessa topologia con la crop sono tutti nella sezione "Crop (descrittivo, supplementary)". FaMoS TEST sta in una sezione descrittiva a parte in fondo. Ava-256 non e' toccato.

## File e colonne (nomi stabili)

- `spearman.csv`: `metodo` (etichetta), `colonna` (id stabile; plug-in `plugin:<metodo>[/<variante>]|<lettura>`), `famiglia`, `nostro`, `dichiarata` (lettura dichiarata per quella GT di una variante dichiarata), `post_hoc` (d_F calibrata, varianti A e B dei fit e composizione, template FaMoS costruito qui), `declared` (variante dichiarata: `declared` di fm_plugin.json; True per i metodi interni), `dominio`, `gt` (fr, sr), `insieme_righe` (nocrop, crop, all_cross, con_perturbazione, coppia_topologie, stessa_topologia, coppia_ordinata, media_20_ordinate_senza_crop, media_5_stessa_topologia), `perturbazione` (`-`, una topologia, `a~b` coppia non ordinata, `a>b` ordinata, `tutte`, `senza_crop`; `con_perturbazione` P = P contro le altre topologie SENZA crop, la scomposizione di nocrop; per P = crop coincide con `crop`; `stessa_topologia` = soggetti diversi con la stessa topologia, `tutte` con crop - crop, `senza_crop` senza), `rho`, `ic_lo`, `ic_hi`, `n_righe`, `n_soggetti`, `seme`, `passo` (`comune`; la sensibilita' del NICP; `varianti non dichiarate`; `righe proprie (n righe a topologie diverse)` dei plug-in con buchi).
- `deltas.csv`: le stesse chiavi + `tipo` (vs_nostro = braccio - concorrente; scala_allineamento = a - b; migliore_vs_migliore), `confronto`, `ruolo` (scala: principale o secondario), `piu_allineato` (scala: a o b), `metodo_a`, `colonna_a`, `post_hoc_a`, `metodo_b`, `colonna_b`, `post_hoc_b`, `rho_a`, `rho_b`, `delta`, `ic_lo`, `ic_hi`, `p_le0` (frazione delle repliche con delta <= 0), `declared_a`, `declared_b`.
- `non_applicabili.csv`: `dominio`, `metodo`, `colonna`, `insieme_righe`, `motivo` (anche le CAUTELE su celle calcolate).
- `controls.json`: controlli di riproduzione, maschera e righe tolte per colonna, fallimenti del NICP per topologia, Spearman di ogni colonna con la taglia oracolo e stimata (`spearman_con_taglia`), plug-in (tipo, passo, righe perse: `views.<dominio>.plugins`).

Letture dichiarate: nostri FR = d_F cal. (post hoc, emendamento 4), d_F non cal. (preregistrata), ctrlfr; SR = d_P (ctrlfr non e' un braccio per SR: e' in `spearman.csv` ma non e' un riferimento dei delta SR). Fit 3DMM: coefficienti (FR e SR), mesh FR con FR, mesh SR con SR, composizione di B con FR (k calibrato solo per B: per preregistrato e vA la composizione non esiste). Geometrici, percettivi, spettrali, reti e riferimenti: entrambe le GT. Plug-in: la GT di `readings` in fm_plugin.json (GW e percettivi: `fr` in mm con FR, `sr` a CS_ref con SR; ULRSSM: per coppia e via template, FR e SR; protocolli: `mm` con entrambe). Tutte le letture sono comunque in `spearman.csv` (`dichiarata` = False).

## Come aggiungere un concorrente (plug-in)

Una cartella `aau/runs/evidence/competitors_new/<metodo>/` (o `<metodo>/<tag>/`) con, per dominio (`hifi3d`, `facescape`, `faceverse`, `faceverse_neutral`, `famos`), `<dominio>.npz` (`subjects`, `topologies`, matrici `D` o `D_<lettura>` (n, n) sulle mesh elencate; FaMoS: `subjects` = view_id, `topologies` = `scan_gallery`) oppure `<dominio>.csv` / `pairs_<dominio>.csv` (`subject_a`, `topology_a`, `subject_b`, `topology_b`, una colonna per lettura), e il file di marcatura OBBLIGATORIO `fm_plugin.json` (anche `{}`; le cartelle senza, es. `pilot/` o `prep/`, si ignorano; facoltativi `label`, `family`, `readings`: {lettura: {`label`, `gt`}}, `declared` (assente o true = dichiarata; false o null = non dichiarata, passo a parte); senza `gt`, una lettura `fr` / `*_fr` e' dichiarata per FR, `sr` / `*_sr` per SR, le altre per entrambe). Poi solo lo stadio finale: `FM_STEP=eval sbatch --cpus-per-task=64 --mem=192G --time=02:00:00 aau/full_matrix/fm.sbatch`. Un plug-in finito su tutta la maschera comune entra nello stesso passo e non cambia le righe degli altri; uno con buchi va in un passo a parte (`passo` = righe proprie), con i bracci di riferimento sulle sue righe.

## Maschera comune e righe perse

Una sola maschera per dominio: le righe con GT dove tutte le colonne interne sono finite (sulla stessa topologia, quelle definite li'). I plug-in non la cambiano: uno finito su tutte le sue righe a topologie diverse (stessa topologia: tutte o nessuna) entra nel passo `comune` (dichiarato) o nel passo `varianti non dichiarate` (stesse righe e repliche); uno con buchi va in un passo `righe proprie (...)` sulle sue righe, e qui si dicono le righe che perde rispetto alla maschera. Righe = coppie ordinate di mesh di soggetti diversi.

| dominio | righe con GT | maschera: topologie diverse / stessa | tolte dalle colonne interne (topologie diverse) | plug-in nel passo comune / non dichiarati | righe perse dai plug-in di questi due passi | plug-in senza stessa topologia | righe proprie: colonne, righe, perse rispetto alla maschera |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 178200 | 148500 / 29700 | 0 | 18 / 28 | 0 | gw/euclid_128_cgw2, gw/euclid_256_cg, gw/euclid_256_cgw2, now_metrical, now_original, realy, stirling | - |
| famos | 105 | 105 / 0 | 0 | 18 / 28 | 0 | - | - |

## Non applicabili e cautele

| dominio | metodo | insieme | motivo |
| --- | --- | --- | --- |
| faceverse_neutral | plug-in now_metrical/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_metrical/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_metrical/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_metrical/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_metrical/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_original/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_original/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_original/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_original/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in now_original/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in realy/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in realy/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in realy/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in realy/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in realy/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in stirling/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in stirling/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in stirling/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in stirling/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | plug-in stirling/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse_neutral | ArcFace-render, ombreggiato 3 viste (E12) | tutti | render ombreggiato calcolato solo da E12 (HIFI3D, FaceVerse con espressioni) |
| faceverse_neutral | OpenShape PointBERT | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| faceverse_neutral | Uni3D-g | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| faceverse_neutral | NICP per coppia in mm | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| faceverse_neutral | NICP per coppia cs | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| faceverse_neutral | Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| famos | plug-in now_metrical/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_metrical/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_metrical/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_metrical/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_metrical/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_original/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_original/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_original/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_original/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in now_original/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in realy/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in realy/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in realy/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in realy/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in realy/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in stirling/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in stirling/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in stirling/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in stirling/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | plug-in stirling/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| famos | NICP per coppia cs | tutti | non calcolato per questo dominio |
| famos | ArcFace-render, ombreggiato 3 viste (E12) | tutti | render ombreggiato calcolato solo da E12 (HIFI3D, FaceVerse con espressioni) |
| famos | OpenShape PointBERT | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| famos | Uni3D-g | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| famos | tutti | crop | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | tutti | all_cross | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | tutti | con_perturbazione | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | tutti | coppia_topologie | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | tutti | stessa_topologia | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | ArcFace-render, normal map 3 viste (G0) | nocrop | CAUTELA (calcolato, non escluso): sulle scansioni parziali il detector trova il volto in 27/45 render ({'-030.00': 11, '+000.00': 11, '+030.00': 5}); il ritaglio a +30 gradi e' tarato su 5 render su 15, fuori dal regime validato in G0 |

## Miglior nostro contro miglior concorrente (famiglie sintetiche, senza crop; letture dichiarate di varianti dichiarate, scelti sul punto: post hoc)

| dominio | GT | insieme | righe | miglior nostro | rho | miglior concorrente | rho | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | FR | nocrop | 99000 | factorized s2345, d_F non cal. | 0.373 [0.293, 0.451] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.479 [0.393, 0.557] | -0.106 [-0.209, -0.005] |
| faceverse_neutral | FR | con_perturbazione original | 39600 | factorized s2345, d_F non cal. | 0.375 [0.294, 0.453] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.480 [0.394, 0.559] | -0.105 [-0.208, -0.003] |
| faceverse_neutral | FR | con_perturbazione remesh | 39600 | factorized s2345, d_F non cal. | 0.378 [0.297, 0.454] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.480 [0.392, 0.558] | -0.102 [-0.206, +0.002] |
| faceverse_neutral | FR | con_perturbazione down8k | 39600 | factorized s2345, d_F non cal. | 0.373 [0.292, 0.450] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.477 [0.392, 0.555] | -0.105 [-0.206, -0.004] |
| faceverse_neutral | FR | con_perturbazione up60k | 39600 | factorized s2345, d_F non cal. | 0.374 [0.293, 0.452] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.481 [0.394, 0.559] | -0.107 [-0.209, -0.005] |
| faceverse_neutral | FR | con_perturbazione noisy | 39600 | factorized s2345, d_F non cal. | 0.367 [0.289, 0.444] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.488 [0.402, 0.565] | -0.121 [-0.225, -0.018] |
| faceverse_neutral | FR | stessa_topologia senza_crop | 24750 | C3M e205, d_F cal. | 0.399 [0.317, 0.474] | ULRSSM [faces_xyz] via template, GT FR (FR) | 0.417 [0.334, 0.495] | -0.018 [-0.100, +0.066] |
| faceverse_neutral | SR | nocrop | 99000 | C3M e205, d_P | 0.382 [0.304, 0.452] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.521 [0.447, 0.590] | -0.139 [-0.225, -0.053] |
| faceverse_neutral | SR | con_perturbazione original | 39600 | C3M e205, d_P | 0.398 [0.321, 0.469] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.523 [0.448, 0.590] | -0.125 [-0.213, -0.039] |
| faceverse_neutral | SR | con_perturbazione remesh | 39600 | C3M e205, d_P | 0.393 [0.313, 0.464] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.523 [0.449, 0.591] | -0.130 [-0.218, -0.041] |
| faceverse_neutral | SR | con_perturbazione down8k | 39600 | C3M e205, d_P | 0.387 [0.307, 0.459] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.520 [0.446, 0.588] | -0.133 [-0.220, -0.047] |
| faceverse_neutral | SR | con_perturbazione up60k | 39600 | C3M e123, d_P | 0.390 [0.303, 0.468] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.523 [0.449, 0.591] | -0.133 [-0.225, -0.044] |
| faceverse_neutral | SR | con_perturbazione noisy | 39600 | factorized s2345, d_P | 0.365 [0.286, 0.437] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.530 [0.454, 0.599] | -0.165 [-0.256, -0.074] |
| faceverse_neutral | SR | stessa_topologia senza_crop | 24750 | C3M e205, d_P | 0.417 [0.336, 0.491] | ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.411 [0.333, 0.483] | +0.005 [-0.070, +0.079] |

Miglior concorrente: * = lettura post hoc (vB, composizione); "(landmark esatti)" = protocollo coi landmark oracolo (NoW originale, NoW metrical, REALY, Stirling). Le letture col rilevatore (`lmk_det`) sono nel passo `righe proprie` (sottocampione HIFI3D e FaceScape), nella sezione "Protocolli: landmark esatti e landmark rilevati".

Ogni lettura nostra di riferimento contro il miglior concorrente della riga sopra (nocrop; delta `vs_nostro`, stesse righe e repliche; gli altri insiemi in `deltas.csv`):

| dominio | GT | insieme | miglior concorrente | lettura nostra | rho nostro | rho concorrente | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_F cal. | 0.341 | 0.479 | -0.138 [-0.242, -0.033] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_F cal. | 0.370 | 0.479 | -0.109 [-0.204, -0.006] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_F non cal. | 0.337 | 0.479 | -0.142 [-0.236, -0.037] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_F non cal. | 0.373 | 0.479 | -0.106 [-0.209, -0.005] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | ctrlfr s1234 | 0.327 | 0.479 | -0.152 [-0.248, -0.052] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | ctrlfr s2345 | 0.299 | 0.479 | -0.180 [-0.286, -0.076] |
| faceverse_neutral | SR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_P | 0.333 | 0.521 | -0.188 [-0.274, -0.096] |
| faceverse_neutral | SR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_P | 0.372 | 0.521 | -0.149 [-0.240, -0.058] |

## Scala dell'allineamento (senza crop)

Delta appaiati a - b (stesse righe, stesse repliche) lungo i gradi di liberta' dell'allineamento. PRINCIPALI: 1 (togliere la taglia: la modalita' cs scala ogni mesh alla cs_ref del dominio, `blmm.work_coords`; non e' una scala per coppia), 1b (togliere la taglia PER COPPIA: NoW originale, Procrustes + ICP con scala stimata per coppia, - NoW metrical, la stessa catena senza scala; landmark esatti; lato piu' allineato a), 2 (ICP cs - NICP cs), 3b (NICP su template cs - NICP per coppia cs), 4 e 5 con FR contro ICP mm, con SR contro ICP cs e NICP cs (con SR il riferimento corretto e' senza taglia). SECONDARI: 2b e 3 in mm, CONFUSI DALLA TAGLIA (in mm il NICP su template segue la taglia, il NICP per coppia la assorbe: vedi la tabella sotto), e 4 e 5 con SR contro ICP mm. 4: fit B con la mesh della GT (FR con FR, SR con SR); 5: nostro preregistrato (d_F non cal. con FR, d_P con SR), entrambi i semi.

Spearman (punto, righe nocrop della maschera) con la taglia oracolo / la taglia stimata (`est_cs`) delle colonne interne della scala (passi 4 e 5: un modello e un seme); tutte le colonne interne e gli insiemi crop e all_cross in `controls.json`, `spearman_con_taglia`. Le colonne NoW del passo 1b sono plug-in: `spearman_con_taglia` copre solo le colonne interne, la loro correlazione con la taglia non e' misurata qui:

| dominio | ICP + Chamfer in mm (per coppia) | ICP + Chamfer cs (per coppia, scala tolta) | NICP per coppia in mm | NICP per coppia cs | NICP su template in mm | NICP su template cs | GNM (visto) vB, mesh d'identita' FR | GNM (visto) vB, mesh d'identita' SR | factorized s1234, d_F non cal. | factorized s1234, d_P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | - | - | - | - | - | - | - | - | - | - |
| facescape | - | - | - | - | - | - | - | - | - | - |
| faceverse | - | - | - | - | - | - | - | - | - | - |
| faceverse_neutral | +0.11 / +0.29 | -0.02 / +0.11 | -0.08 / -0.01 | -0.07 / +0.06 | +0.22 / +0.47 | +0.04 / +0.08 | +0.16 / +0.21 | -0.01 / +0.06 | +0.02 / +0.14 | -0.01 / +0.12 |
| famos | +0.58 / +0.69 | +0.30 / +0.31 | +0.37 / +0.54 | - | +0.65 / +0.88 | +0.15 / +0.06 | +0.60 / +0.70 | +0.05 / +0.13 | +0.24 / +0.38 | +0.03 / +0.14 |

### Confronti principali

GT FR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.035 [-0.018, +0.090] | +0.037 [-0.015, +0.092] | +0.034 [-0.020, +0.089] | +0.035 [-0.019, +0.091] | +0.036 [-0.018, +0.091] | +0.031 [-0.021, +0.086] | +0.039 [-0.015, +0.095] |
| faceverse_neutral | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.021 [-0.060, +0.019] | -0.020 [-0.059, +0.019] | -0.020 [-0.060, +0.020] | -0.020 [-0.061, +0.021] | -0.020 [-0.060, +0.022] | -0.018 [-0.059, +0.022] | - |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.088 [+0.022, +0.148] | +0.092 [+0.026, +0.153] | +0.093 [+0.028, +0.155] | +0.084 [+0.018, +0.148] | +0.090 [+0.023, +0.150] | +0.081 [+0.019, +0.137] | - |
| faceverse_neutral | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.112 [-0.230, +0.002] | -0.106 [-0.226, +0.009] | -0.114 [-0.234, +0.002] | -0.104 [-0.225, +0.008] | -0.109 [-0.228, +0.008] | -0.127 [-0.244, -0.016] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | -0.088 [-0.160, -0.015] | -0.093 [-0.164, -0.021] | -0.096 [-0.169, -0.025] | -0.078 [-0.154, -0.003] | -0.092 [-0.164, -0.021] | -0.083 [-0.155, -0.014] | -0.095 [-0.167, -0.022] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.094 [-0.175, -0.012] | -0.101 [-0.182, -0.019] | -0.096 [-0.176, -0.016] | -0.086 [-0.170, +0.002] | -0.098 [-0.180, -0.017] | -0.091 [-0.173, -0.008] | -0.100 [-0.183, -0.019] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.072 [-0.151, +0.013] | -0.074 [-0.153, +0.010] | -0.074 [-0.152, +0.009] | -0.067 [-0.147, +0.020] | -0.070 [-0.150, +0.014] | -0.076 [-0.156, +0.006] | -0.065 [-0.145, +0.021] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.035 [-0.123, +0.042] | -0.040 [-0.125, +0.038] | -0.037 [-0.125, +0.043] | -0.029 [-0.116, +0.052] | -0.038 [-0.122, +0.040] | -0.037 [-0.122, +0.042] | -0.035 [-0.123, +0.044] |

GT SR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.014 [-0.058, +0.030] | -0.012 [-0.056, +0.033] | -0.015 [-0.058, +0.030] | -0.014 [-0.058, +0.031] | -0.014 [-0.057, +0.031] | -0.018 [-0.060, +0.026] | -0.011 [-0.055, +0.034] |
| faceverse_neutral | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.018 [-0.017, +0.056] | +0.019 [-0.017, +0.057] | +0.019 [-0.018, +0.058] | +0.020 [-0.017, +0.060] | +0.021 [-0.016, +0.061] | +0.023 [-0.013, +0.063] | - |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.074 [+0.014, +0.133] | +0.079 [+0.018, +0.139] | +0.078 [+0.018, +0.138] | +0.070 [+0.008, +0.131] | +0.076 [+0.015, +0.134] | +0.069 [+0.012, +0.124] | - |
| faceverse_neutral | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.148 [-0.258, -0.036] | -0.141 [-0.253, -0.029] | -0.151 [-0.265, -0.039] | -0.142 [-0.253, -0.029] | -0.147 [-0.259, -0.034] | -0.162 [-0.269, -0.054] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.002 [-0.057, +0.062] | -0.000 [-0.060, +0.060] | -0.006 [-0.064, +0.053] | +0.010 [-0.053, +0.073] | -0.001 [-0.059, +0.059] | +0.002 [-0.056, +0.060] | -0.002 [-0.061, +0.059] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | -0.015 [-0.080, +0.054] | -0.020 [-0.085, +0.050] | -0.022 [-0.088, +0.045] | -0.005 [-0.070, +0.069] | -0.018 [-0.082, +0.050] | -0.013 [-0.079, +0.057] | -0.017 [-0.082, +0.052] |
| faceverse_neutral | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | -0.063 [-0.140, +0.017] | -0.063 [-0.140, +0.020] | -0.067 [-0.143, +0.014] | -0.058 [-0.133, +0.025] | -0.061 [-0.138, +0.022] | -0.073 [-0.149, +0.007] | -0.053 [-0.133, +0.031] |
| faceverse_neutral | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | -0.025 [-0.096, +0.051] | -0.027 [-0.098, +0.048] | -0.027 [-0.099, +0.049] | -0.018 [-0.092, +0.059] | -0.027 [-0.100, +0.047] | -0.030 [-0.103, +0.043] | -0.021 [-0.094, +0.056] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.076 [+0.005, +0.152] | +0.079 [+0.006, +0.154] | +0.072 [-0.001, +0.148] | +0.080 [+0.006, +0.158] | +0.075 [+0.005, +0.150] | +0.071 [+0.000, +0.141] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.059 [-0.025, +0.143] | +0.059 [-0.028, +0.145] | +0.057 [-0.030, +0.140] | +0.064 [-0.018, +0.152] | +0.057 [-0.027, +0.141] | +0.055 [-0.027, +0.137] | - |
| faceverse_neutral | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.010 [-0.068, +0.095] | +0.016 [-0.063, +0.098] | +0.012 [-0.067, +0.095] | +0.012 [-0.070, +0.096] | +0.015 [-0.067, +0.099] | -0.004 [-0.078, +0.074] | - |
| faceverse_neutral | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.049 [-0.041, +0.133] | +0.052 [-0.037, +0.137] | +0.051 [-0.039, +0.138] | +0.051 [-0.041, +0.137] | +0.049 [-0.042, +0.133] | +0.038 [-0.048, +0.120] | - |

### Confronti secondari (2b e 3 confusi dalla taglia; 4 e 5 con SR contro un riferimento con la taglia)

GT FR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.131 [+0.049, +0.212] | +0.137 [+0.055, +0.218] | +0.135 [+0.051, +0.216] | +0.129 [+0.045, +0.215] | +0.132 [+0.051, +0.214] | +0.123 [+0.044, +0.200] |
| faceverse_neutral | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.054 [-0.185, +0.072] | -0.054 [-0.183, +0.074] | -0.055 [-0.188, +0.072] | -0.045 [-0.175, +0.084] | -0.056 [-0.188, +0.071] | -0.063 [-0.191, +0.062] |

GT SR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.069 [-0.002, +0.138] | +0.075 [+0.004, +0.144] | +0.072 [-0.001, +0.141] | +0.066 [-0.008, +0.140] | +0.070 [-0.003, +0.141] | +0.063 [-0.004, +0.129] | - |
| faceverse_neutral | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.158 [-0.273, -0.039] | -0.157 [-0.272, -0.038] | -0.160 [-0.277, -0.039] | -0.149 [-0.266, -0.027] | -0.162 [-0.278, -0.041] | -0.165 [-0.281, -0.048] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.016 [-0.050, +0.078] | +0.011 [-0.052, +0.073] | +0.009 [-0.057, +0.073] | +0.024 [-0.045, +0.090] | +0.013 [-0.051, +0.075] | +0.020 [-0.045, +0.082] | +0.009 [-0.057, +0.073] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | -0.001 [-0.077, +0.078] | -0.008 [-0.085, +0.071] | -0.007 [-0.082, +0.070] | +0.008 [-0.070, +0.091] | -0.005 [-0.081, +0.073] | +0.004 [-0.074, +0.084] | -0.006 [-0.083, +0.072] |
| faceverse_neutral | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | -0.049 [-0.127, +0.038] | -0.051 [-0.128, +0.038] | -0.052 [-0.128, +0.035] | -0.044 [-0.121, +0.043] | -0.047 [-0.125, +0.042] | -0.055 [-0.132, +0.031] | -0.043 [-0.120, +0.046] |
| faceverse_neutral | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | -0.011 [-0.094, +0.065] | -0.015 [-0.097, +0.061] | -0.012 [-0.097, +0.063] | -0.005 [-0.089, +0.074] | -0.013 [-0.097, +0.062] | -0.013 [-0.094, +0.065] | -0.010 [-0.094, +0.067] |

## Conteggio

Per ogni passo e GT, in nocrop: numero di famiglie sintetiche (HIFI3D, FaceScape, FaceVerse con espressioni, FaceVerse neutra) con IC 95% appaiato che esclude lo 0 nel verso "piu' allineamento = peggio" / nel verso opposto / non risolte. Lato piu' allineato: b nei passi 1-3 (cs: taglia tolta per mesh; NICP: deformazione; NICP per coppia: deformazione specifica della coppia contro l'iscrizione su un template comune), a nell'1b (scala stimata per coppia), a nel 4 (fit del modello per mesh contro la rigida), b nel 5 (ICP: allineamento esplicito; il braccio non allinea). Le righe 4 e 5 contano ogni modello / seme a parte.

| GT | passo | ruolo | nocrop: peggio / meglio / n.r. |
| --- | --- | --- | --- |
| FR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 0 / 0 / 1 (su 1) |
| FR | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | principale | 0 / 0 / 1 (su 1) |
| FR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 1 / 0 / 0 (su 1) |
| FR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 1 / 0 / 0 (su 1) |
| FR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 0 / 0 / 1 (su 1) |
| FR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 0 / 1 (su 1) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | principale | 1 / 0 / 0 (su 1) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | principale | 1 / 0 / 0 (su 1) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | principale | 0 / 0 / 1 (su 1) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | principale | 0 / 0 / 1 (su 1) |
| SR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 0 / 0 / 1 (su 1) |
| SR | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | principale | 0 / 0 / 1 (su 1) |
| SR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 1 / 0 / 0 (su 1) |
| SR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 0 / 0 / 1 (su 1) |
| SR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 0 / 1 / 0 (su 1) |
| SR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 1 / 0 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 0 / 1 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | principale | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | principale | 0 / 0 / 1 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 1 / 0 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | principale | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | principale | 0 / 0 / 1 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | secondario | 0 / 0 / 1 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | secondario | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | secondario | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | secondario | 0 / 0 / 1 (su 1) |

## Fallimenti del NICP per coppia: selezione delle righe

La maschera comune toglie le righe dove il NICP per coppia (mm o cs) fallisce (matrice singolare). La selezione dipende dalla topologia, non dalla GT: le righe perse sono soprattutto quelle con up60k, che quindi pesa meno negli insiemi nocrop e all_cross; e il NICP non e' penalizzato per i propri fallimenti. Righe perse per topologia (righe con quella topologia su un lato, senza la maschera del NICP):

| dominio | righe perse | original | remesh | down8k | up60k | noisy | crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 0 | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) |

Sensibilita' (`passo` = "sensibilita': NICP fallito = distanza massima" in entrambi i CSV): stesse colonne, maschera senza il NICP (le righe perse rientrano per tutti), NICP fallito = distanza massima della colonna sul dominio (il fallimento conta come la coppia piu' lontana). Il NICP con e senza la sostituzione e i passi della scala che lo usano:

| dominio | GT | insieme | metodo / passo | maschera comune | fallimento = distanza massima |
| --- | --- | --- | --- | --- | --- |

## Protocolli: landmark esatti e landmark rilevati (lmk_det, colonna aggiuntiva marcata)

NoW originale, NoW metrical, REALY e Stirling: lettura dichiarata coi landmark esatti (passo `comune`). Il rilevatore (`lmk_det`, insightface 1k3d68 sui render: la variante che si calcola su Ava-256) e il rumore sui landmark (`lmk_s1`-`lmk_s5`, sigma in mm, non dichiarati) esistono solo su un sottocampione di coppie (HIFI3D e FaceScape): passo `righe proprie`, con i landmark esatti e i bracci di riferimento sulle stesse righe e repliche (le righe perse sono nella tabella della maschera). Spearman (senza crop):

## Varianti non dichiarate (passo `varianti non dichiarate`, descrittivo)

Stesse righe della maschera comune e stesse repliche; mai nel miglior concorrente ne' nella scala. GW `euclid_256_cgw2` e `euclid_256_cg`, percettivi nelle varianti di render non scelte dal pilota, ULRSSM `faces_wks` e `zeroshot_faust`. I delta `vs_nostro` sono in `deltas.csv` con lo stesso `passo`.

GT FR:

| metodo | faceverse_neutral nocrop | faceverse_neutral stessa_topologia senza_crop |
| --- | --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), mm (FR) | 0.035 [0.009, 0.061] | 0.070 [0.016, 0.126] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.043 [0.018, 0.071] | 0.106 [0.054, 0.161] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.036 [0.019, 0.054] | 0.155 [0.082, 0.222] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.067 [0.040, 0.094] | 0.196 [0.112, 0.276] |
| DreamSim ensemble su render [profondita'] (non dichiarata), mm (FR) | 0.076 [0.026, 0.126] | 0.088 [0.031, 0.152] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.082 [0.050, 0.110] | 0.187 [0.116, 0.249] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | 0.141 [0.089, 0.189] | - |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | 0.249 [0.171, 0.320] | - |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.136 [0.103, 0.166] | 0.401 [0.314, 0.481] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.166 [0.126, 0.201] | 0.408 [0.320, 0.487] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, mm (FR) | 0.070 [0.027, 0.117] | 0.107 [0.055, 0.161] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT FR (FR) | 0.046 [-0.005, 0.101] | 0.092 [0.007, 0.179] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, mm (FR) | 0.092 [0.049, 0.136] | 0.159 [0.096, 0.220] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT FR (FR) | 0.089 [0.021, 0.150] | 0.135 [0.044, 0.213] |

GT SR:

| metodo | faceverse_neutral nocrop | faceverse_neutral stessa_topologia senza_crop |
| --- | --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.040 [0.015, 0.066] | 0.083 [0.027, 0.140] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.055 [0.028, 0.086] | 0.141 [0.094, 0.194] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.035 [0.019, 0.050] | 0.173 [0.107, 0.238] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.072 [0.046, 0.099] | 0.204 [0.128, 0.276] |
| DreamSim ensemble su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.087 [0.037, 0.139] | 0.100 [0.040, 0.162] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.091 [0.059, 0.121] | 0.223 [0.156, 0.284] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | 0.121 [0.079, 0.158] | - |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | 0.243 [0.180, 0.297] | - |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.130 [0.099, 0.160] | 0.375 [0.290, 0.452] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.157 [0.120, 0.191] | 0.381 [0.294, 0.459] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, a CS_ref (SR) | 0.082 [0.042, 0.124] | 0.119 [0.064, 0.175] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT SR (d_P) (SR) | 0.057 [0.005, 0.111] | 0.100 [0.017, 0.182] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, a CS_ref (SR) | 0.101 [0.059, 0.144] | 0.175 [0.113, 0.235] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT SR (d_P) (SR) | 0.116 [0.057, 0.172] | 0.178 [0.091, 0.256] |

## Tabelle per dominio (famiglie sintetiche, senza crop)

### FaceVerse neutra: Spearman per insieme di righe (seme 271049)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (99000) | con_perturbazione original (39600) | con_perturbazione remesh (39600) | con_perturbazione down8k (39600) | con_perturbazione up60k (39600) | con_perturbazione noisy (39600) | stessa_topologia senza_crop (24750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.341 [0.263, 0.416] | 0.343 [0.264, 0.420] | 0.343 [0.264, 0.419] | 0.337 [0.258, 0.412] | 0.346 [0.264, 0.424] | 0.336 [0.256, 0.411] | 0.352 [0.271, 0.430] |
| **factorized s1234, d_F non cal.** | 0.337 [0.256, 0.409] | 0.341 [0.259, 0.414] | 0.340 [0.260, 0.413] | 0.335 [0.255, 0.408] | 0.342 [0.259, 0.416] | 0.328 [0.246, 0.397] | 0.351 [0.269, 0.427] |
| factorized s1234, d_P | 0.321 [0.237, 0.391] | 0.325 [0.241, 0.398] | 0.324 [0.241, 0.397] | 0.319 [0.236, 0.393] | 0.326 [0.240, 0.402] | 0.309 [0.224, 0.382] | 0.335 [0.251, 0.410] |
| **factorized s2345, d_F cal.*** | 0.370 [0.298, 0.440] | 0.372 [0.298, 0.442] | 0.376 [0.303, 0.445] | 0.368 [0.294, 0.437] | 0.371 [0.296, 0.441] | 0.366 [0.295, 0.433] | 0.380 [0.306, 0.451] |
| **factorized s2345, d_F non cal.** | 0.373 [0.293, 0.451] | 0.375 [0.294, 0.453] | 0.378 [0.297, 0.454] | 0.373 [0.292, 0.450] | 0.374 [0.293, 0.452] | 0.367 [0.289, 0.444] | 0.381 [0.300, 0.459] |
| factorized s2345, d_P | 0.361 [0.276, 0.439] | 0.363 [0.279, 0.441] | 0.365 [0.279, 0.444] | 0.361 [0.278, 0.439] | 0.362 [0.277, 0.440] | 0.354 [0.269, 0.433] | 0.369 [0.282, 0.448] |
| **ctrlfr s1234** | 0.327 [0.248, 0.399] | 0.337 [0.257, 0.411] | 0.334 [0.256, 0.407] | 0.334 [0.258, 0.407] | 0.339 [0.261, 0.415] | 0.317 [0.238, 0.391] | 0.360 [0.277, 0.436] |
| **ctrlfr s2345** | 0.299 [0.214, 0.374] | 0.308 [0.222, 0.383] | 0.305 [0.221, 0.379] | 0.306 [0.221, 0.379] | 0.304 [0.217, 0.380] | 0.286 [0.200, 0.359] | 0.323 [0.234, 0.401] |
| **C3M e123, d_F cal.*** | 0.347 [0.268, 0.419] | 0.360 [0.276, 0.433] | 0.356 [0.273, 0.428] | 0.356 [0.271, 0.431] | 0.362 [0.279, 0.436] | 0.326 [0.256, 0.394] | 0.388 [0.298, 0.466] |
| **C3M e123, d_F non cal.** | 0.362 [0.272, 0.438] | 0.367 [0.277, 0.444] | 0.368 [0.278, 0.447] | 0.363 [0.273, 0.442] | 0.372 [0.283, 0.451] | 0.351 [0.267, 0.423] | 0.386 [0.292, 0.466] |
| C3M e123, d_P | 0.363 [0.276, 0.441] | 0.368 [0.280, 0.446] | 0.368 [0.280, 0.447] | 0.364 [0.274, 0.441] | 0.372 [0.283, 0.453] | 0.347 [0.263, 0.421] | 0.382 [0.292, 0.461] |
| **C3M e205, d_F cal.*** | 0.336 [0.264, 0.400] | 0.358 [0.283, 0.424] | 0.350 [0.274, 0.418] | 0.352 [0.275, 0.420] | 0.352 [0.276, 0.422] | 0.317 [0.250, 0.380] | 0.399 [0.317, 0.474] |
| **C3M e205, d_F non cal.** | 0.357 [0.273, 0.433] | 0.371 [0.287, 0.449] | 0.366 [0.279, 0.444] | 0.362 [0.275, 0.439] | 0.368 [0.278, 0.451] | 0.343 [0.259, 0.418] | 0.396 [0.310, 0.477] |
| C3M e205, d_P | 0.363 [0.275, 0.442] | 0.377 [0.289, 0.455] | 0.372 [0.282, 0.451] | 0.368 [0.281, 0.447] | 0.371 [0.281, 0.452] | 0.340 [0.259, 0.415] | 0.397 [0.308, 0.478] |
| **ICP + Chamfer in mm (per coppia)** | 0.409 [0.326, 0.481] | 0.415 [0.332, 0.486] | 0.414 [0.331, 0.486] | 0.402 [0.314, 0.475] | 0.412 [0.328, 0.484] | 0.404 [0.325, 0.476] | 0.416 [0.333, 0.489] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.374 [0.293, 0.449] | 0.378 [0.297, 0.452] | 0.380 [0.298, 0.455] | 0.367 [0.281, 0.443] | 0.376 [0.296, 0.451] | 0.373 [0.292, 0.447] | 0.377 [0.297, 0.452] |
| **NICP per coppia in mm** | 0.278 [0.194, 0.360] | 0.278 [0.196, 0.358] | 0.280 [0.195, 0.363] | 0.272 [0.184, 0.359] | 0.279 [0.195, 0.361] | 0.281 [0.201, 0.359] | - |
| **NICP per coppia cs** | 0.286 [0.198, 0.372] | 0.286 [0.199, 0.370] | 0.287 [0.199, 0.373] | 0.282 [0.191, 0.369] | 0.287 [0.198, 0.373] | 0.292 [0.204, 0.373] | - |
| **NICP su template in mm** | 0.223 [0.125, 0.319] | 0.224 [0.125, 0.320] | 0.225 [0.124, 0.322] | 0.228 [0.130, 0.323] | 0.223 [0.125, 0.320] | 0.218 [0.121, 0.318] | 0.224 [0.126, 0.320] |
| **NICP su template cs** | 0.174 [0.089, 0.257] | 0.180 [0.092, 0.263] | 0.173 [0.088, 0.253] | 0.178 [0.092, 0.261] | 0.178 [0.091, 0.262] | 0.164 [0.083, 0.242] | 0.180 [0.090, 0.264] |
| **varifold in mm (massa unitaria)** | 0.285 [0.223, 0.346] | 0.323 [0.255, 0.389] | 0.307 [0.243, 0.368] | 0.323 [0.255, 0.390] | 0.322 [0.254, 0.389] | 0.338 [0.262, 0.413] | 0.333 [0.269, 0.394] |
| **GNM (visto) preregistrato, coefficienti** | 0.240 [0.164, 0.309] | 0.244 [0.168, 0.314] | 0.236 [0.159, 0.303] | 0.244 [0.168, 0.310] | 0.244 [0.166, 0.312] | 0.236 [0.161, 0.305] | 0.240 [0.164, 0.308] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.232 [0.140, 0.318] | 0.235 [0.144, 0.320] | 0.230 [0.135, 0.317] | 0.237 [0.144, 0.323] | 0.232 [0.138, 0.318] | 0.225 [0.135, 0.312] | 0.232 [0.140, 0.319] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.240 [0.157, 0.320] | 0.246 [0.162, 0.325] | 0.238 [0.155, 0.319] | 0.243 [0.161, 0.322] | 0.244 [0.160, 0.324] | 0.229 [0.147, 0.308] | 0.242 [0.158, 0.322] |
| **GNM (visto) vA, coefficienti*** | 0.230 [0.152, 0.304] | 0.235 [0.155, 0.308] | 0.225 [0.146, 0.299] | 0.233 [0.154, 0.305] | 0.234 [0.154, 0.308] | 0.226 [0.151, 0.301] | 0.229 [0.152, 0.303] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.236 [0.142, 0.321] | 0.239 [0.145, 0.324] | 0.235 [0.140, 0.321] | 0.242 [0.147, 0.326] | 0.237 [0.141, 0.321] | 0.230 [0.138, 0.318] | 0.237 [0.141, 0.322] |
| GNM (visto) vA, mesh d'identita' SR* | 0.222 [0.135, 0.301] | 0.228 [0.140, 0.307] | 0.222 [0.134, 0.303] | 0.225 [0.138, 0.304] | 0.226 [0.138, 0.307] | 0.211 [0.125, 0.290] | 0.224 [0.137, 0.303] |
| **GNM (visto) vB, coefficienti*** | 0.299 [0.197, 0.384] | 0.298 [0.196, 0.383] | 0.299 [0.195, 0.385] | 0.301 [0.199, 0.384] | 0.299 [0.198, 0.383] | 0.303 [0.201, 0.387] | 0.297 [0.196, 0.381] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.321 [0.225, 0.400] | 0.322 [0.224, 0.401] | 0.318 [0.222, 0.398] | 0.324 [0.229, 0.404] | 0.320 [0.224, 0.398] | 0.321 [0.225, 0.399] | 0.321 [0.225, 0.401] |
| GNM (visto) vB, mesh d'identita' SR* | 0.373 [0.288, 0.449] | 0.375 [0.292, 0.453] | 0.372 [0.286, 0.448] | 0.373 [0.289, 0.450] | 0.373 [0.288, 0.450] | 0.372 [0.287, 0.446] | 0.373 [0.288, 0.450] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.382 [0.293, 0.458] | 0.383 [0.293, 0.460] | 0.380 [0.290, 0.458] | 0.384 [0.296, 0.461] | 0.381 [0.292, 0.456] | 0.383 [0.294, 0.459] | 0.382 [0.293, 0.458] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.171 [0.102, 0.241] | 0.169 [0.099, 0.241] | 0.175 [0.104, 0.245] | 0.173 [0.103, 0.242] | 0.174 [0.102, 0.245] | 0.165 [0.095, 0.235] | 0.173 [0.103, 0.245] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.214 [0.134, 0.291] | 0.212 [0.131, 0.289] | 0.220 [0.140, 0.300] | 0.216 [0.135, 0.293] | 0.214 [0.131, 0.291] | 0.208 [0.122, 0.286] | 0.215 [0.134, 0.292] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.169 [0.089, 0.244] | 0.166 [0.085, 0.241] | 0.170 [0.089, 0.245] | 0.172 [0.092, 0.246] | 0.173 [0.091, 0.249] | 0.163 [0.082, 0.235] | 0.172 [0.091, 0.246] |
| **FLAME 2023 Open vA, coefficienti*** | 0.179 [0.108, 0.249] | 0.177 [0.106, 0.248] | 0.183 [0.110, 0.252] | 0.180 [0.110, 0.250] | 0.181 [0.109, 0.251] | 0.173 [0.103, 0.245] | 0.181 [0.109, 0.251] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.220 [0.134, 0.299] | 0.218 [0.133, 0.298] | 0.227 [0.141, 0.306] | 0.223 [0.138, 0.301] | 0.220 [0.133, 0.299] | 0.214 [0.124, 0.296] | 0.221 [0.135, 0.301] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.175 [0.094, 0.248] | 0.172 [0.090, 0.246] | 0.176 [0.094, 0.250] | 0.178 [0.099, 0.251] | 0.179 [0.096, 0.256] | 0.169 [0.089, 0.242] | 0.178 [0.097, 0.252] |
| **FLAME 2023 Open vB, coefficienti*** | 0.249 [0.143, 0.341] | 0.247 [0.141, 0.342] | 0.250 [0.141, 0.342] | 0.255 [0.146, 0.346] | 0.248 [0.143, 0.340] | 0.246 [0.142, 0.339] | 0.246 [0.141, 0.337] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.315 [0.223, 0.393] | 0.314 [0.223, 0.392] | 0.318 [0.229, 0.396] | 0.316 [0.224, 0.395] | 0.314 [0.222, 0.394] | 0.313 [0.220, 0.395] | 0.316 [0.224, 0.395] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.358 [0.275, 0.438] | 0.356 [0.272, 0.437] | 0.359 [0.275, 0.439] | 0.359 [0.276, 0.438] | 0.361 [0.278, 0.441] | 0.357 [0.274, 0.438] | 0.360 [0.277, 0.439] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.367 [0.280, 0.444] | 0.365 [0.278, 0.443] | 0.369 [0.284, 0.446] | 0.368 [0.282, 0.445] | 0.367 [0.280, 0.445] | 0.365 [0.276, 0.443] | 0.368 [0.281, 0.446] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.153 [0.110, 0.198] | 0.173 [0.125, 0.221] | 0.174 [0.124, 0.222] | 0.180 [0.129, 0.230] | 0.172 [0.124, 0.221] | 0.124 [0.064, 0.192] | 0.171 [0.122, 0.221] |
| **ShapeDNA k=50** | 0.073 [0.036, 0.109] | 0.095 [0.037, 0.146] | 0.099 [0.044, 0.150] | 0.096 [0.040, 0.147] | 0.098 [0.042, 0.151] | 0.094 [0.034, 0.159] | 0.154 [0.071, 0.232] |
| **HKS globale** | 0.036 [0.009, 0.063] | 0.045 [0.006, 0.085] | 0.044 [0.006, 0.084] | 0.045 [0.006, 0.086] | 0.046 [0.007, 0.086] | 0.057 [-0.004, 0.111] | 0.074 [0.013, 0.134] |
| **WKS globale** | 0.051 [0.023, 0.076] | 0.063 [0.019, 0.105] | 0.063 [0.023, 0.099] | 0.057 [0.017, 0.093] | 0.057 [0.021, 0.090] | 0.111 [0.050, 0.168] | 0.077 [0.006, 0.142] |
| **taglia stimata dalla mesh (non oracolo)** | 0.143 [0.045, 0.237] | 0.144 [0.044, 0.242] | 0.140 [0.044, 0.230] | 0.144 [0.044, 0.241] | 0.145 [0.044, 0.242] | 0.144 [0.052, 0.234] | 0.149 [0.047, 0.248] |
| **taglia oracolo (S di FR)** | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.067 [0.039, 0.096] | 0.088 [0.048, 0.132] | 0.069 [0.033, 0.106] | 0.087 [0.047, 0.128] | 0.082 [0.043, 0.123] | 0.126 [0.047, 0.205] | 0.121 [0.065, 0.180] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.069 [0.042, 0.099] | 0.087 [0.045, 0.131] | 0.072 [0.037, 0.108] | 0.087 [0.048, 0.129] | 0.084 [0.046, 0.126] | 0.140 [0.064, 0.219] | 0.115 [0.061, 0.176] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.082 [0.042, 0.122] | 0.099 [0.050, 0.145] | 0.099 [0.049, 0.148] | 0.115 [0.059, 0.171] | 0.094 [0.048, 0.138] | 0.100 [0.040, 0.157] | 0.164 [0.089, 0.233] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.077 [0.038, 0.119] | 0.092 [0.046, 0.140] | 0.090 [0.041, 0.143] | 0.108 [0.054, 0.164] | 0.088 [0.041, 0.132] | 0.095 [0.034, 0.153] | 0.156 [0.083, 0.229] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.097 [0.068, 0.126] | 0.141 [0.095, 0.184] | 0.124 [0.082, 0.163] | 0.140 [0.093, 0.184] | 0.132 [0.089, 0.174] | 0.103 [0.035, 0.174] | 0.206 [0.133, 0.275] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.098 [0.068, 0.126] | 0.142 [0.095, 0.186] | 0.125 [0.083, 0.163] | 0.142 [0.094, 0.185] | 0.133 [0.089, 0.174] | 0.106 [0.044, 0.170] | 0.209 [0.135, 0.278] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.234 [0.157, 0.307] | 0.258 [0.171, 0.335] | 0.244 [0.164, 0.317] | 0.256 [0.168, 0.332] | 0.257 [0.169, 0.334] | 0.266 [0.172, 0.349] | - |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.218 [0.154, 0.272] | 0.255 [0.181, 0.315] | 0.245 [0.177, 0.304] | 0.253 [0.181, 0.312] | 0.255 [0.181, 0.314] | 0.227 [0.144, 0.300] | - |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.160 [0.107, 0.209] | 0.185 [0.124, 0.241] | 0.180 [0.118, 0.237] | 0.190 [0.127, 0.248] | 0.185 [0.124, 0.240] | 0.230 [0.153, 0.298] | 0.275 [0.187, 0.351] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.139 [0.093, 0.182] | 0.160 [0.106, 0.209] | 0.158 [0.106, 0.206] | 0.164 [0.109, 0.214] | 0.160 [0.106, 0.208] | 0.213 [0.145, 0.273] | 0.240 [0.165, 0.307] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.407 [0.326, 0.481] | 0.413 [0.331, 0.489] | 0.414 [0.332, 0.488] | 0.410 [0.324, 0.487] | 0.413 [0.331, 0.489] | 0.434 [0.349, 0.512] | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.386 [0.309, 0.460] | 0.393 [0.315, 0.467] | 0.394 [0.315, 0.467] | 0.389 [0.309, 0.464] | 0.394 [0.317, 0.468] | 0.416 [0.336, 0.493] | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.479 [0.393, 0.557] | 0.480 [0.394, 0.559] | 0.480 [0.392, 0.558] | 0.477 [0.392, 0.555] | 0.481 [0.394, 0.559] | 0.488 [0.402, 0.565] | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.307 [0.213, 0.402] | 0.307 [0.213, 0.404] | 0.310 [0.218, 0.407] | 0.310 [0.217, 0.404] | 0.306 [0.213, 0.402] | 0.311 [0.217, 0.410] | - |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.195 [0.153, 0.233] | 0.256 [0.200, 0.306] | 0.255 [0.198, 0.303] | 0.251 [0.194, 0.299] | 0.255 [0.198, 0.304] | 0.209 [0.126, 0.290] | 0.381 [0.301, 0.453] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.182 [0.140, 0.220] | 0.244 [0.188, 0.292] | 0.243 [0.188, 0.291] | 0.239 [0.184, 0.287] | 0.243 [0.188, 0.289] | 0.191 [0.104, 0.274] | 0.364 [0.286, 0.430] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.249 [0.192, 0.301] | 0.304 [0.238, 0.363] | 0.304 [0.237, 0.362] | 0.294 [0.229, 0.352] | 0.308 [0.241, 0.368] | 0.270 [0.188, 0.343] | 0.417 [0.334, 0.495] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.232 [0.176, 0.284] | 0.285 [0.219, 0.347] | 0.281 [0.215, 0.342] | 0.279 [0.212, 0.340] | 0.286 [0.220, 0.347] | 0.283 [0.199, 0.359] | 0.377 [0.293, 0.456] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (99000) | con_perturbazione original (39600) | con_perturbazione remesh (39600) | con_perturbazione down8k (39600) | con_perturbazione up60k (39600) | con_perturbazione noisy (39600) | stessa_topologia senza_crop (24750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.308 [0.225, 0.387] | 0.310 [0.226, 0.388] | 0.310 [0.226, 0.389] | 0.304 [0.221, 0.383] | 0.312 [0.227, 0.392] | 0.303 [0.221, 0.381] | 0.318 [0.231, 0.398] |
| factorized s1234, d_F non cal. | 0.335 [0.254, 0.407] | 0.338 [0.258, 0.415] | 0.338 [0.257, 0.412] | 0.333 [0.254, 0.404] | 0.340 [0.257, 0.417] | 0.326 [0.247, 0.395] | 0.348 [0.266, 0.425] |
| **factorized s1234, d_P** | 0.333 [0.251, 0.405] | 0.338 [0.256, 0.409] | 0.337 [0.256, 0.407] | 0.332 [0.250, 0.402] | 0.339 [0.256, 0.413] | 0.323 [0.244, 0.392] | 0.347 [0.263, 0.421] |
| factorized s2345, d_F cal.* | 0.336 [0.252, 0.409] | 0.337 [0.252, 0.411] | 0.341 [0.257, 0.414] | 0.333 [0.248, 0.407] | 0.336 [0.252, 0.410] | 0.333 [0.249, 0.404] | 0.344 [0.258, 0.419] |
| factorized s2345, d_F non cal. | 0.369 [0.285, 0.444] | 0.370 [0.285, 0.445] | 0.373 [0.289, 0.448] | 0.367 [0.283, 0.443] | 0.369 [0.285, 0.445] | 0.363 [0.282, 0.437] | 0.377 [0.290, 0.453] |
| **factorized s2345, d_P** | 0.372 [0.291, 0.444] | 0.374 [0.292, 0.447] | 0.376 [0.294, 0.450] | 0.371 [0.290, 0.443] | 0.373 [0.291, 0.446] | 0.365 [0.286, 0.437] | 0.380 [0.298, 0.454] |
| ctrlfr s1234 | 0.326 [0.243, 0.400] | 0.336 [0.252, 0.412] | 0.330 [0.245, 0.405] | 0.334 [0.250, 0.409] | 0.337 [0.254, 0.415] | 0.317 [0.236, 0.389] | 0.356 [0.269, 0.434] |
| ctrlfr s2345 | 0.305 [0.224, 0.377] | 0.311 [0.229, 0.383] | 0.312 [0.231, 0.383] | 0.313 [0.231, 0.382] | 0.311 [0.228, 0.383] | 0.293 [0.214, 0.368] | 0.327 [0.241, 0.400] |
| C3M e123, d_F cal.* | 0.333 [0.245, 0.413] | 0.346 [0.253, 0.428] | 0.342 [0.249, 0.419] | 0.344 [0.251, 0.424] | 0.346 [0.253, 0.424] | 0.309 [0.228, 0.381] | 0.372 [0.272, 0.456] |
| C3M e123, d_F non cal. | 0.374 [0.282, 0.452] | 0.381 [0.289, 0.461] | 0.380 [0.287, 0.460] | 0.377 [0.283, 0.457] | 0.383 [0.289, 0.467] | 0.359 [0.273, 0.434] | 0.399 [0.303, 0.483] |
| **C3M e123, d_P** | 0.382 [0.293, 0.457] | 0.389 [0.300, 0.465] | 0.387 [0.297, 0.463] | 0.385 [0.295, 0.460] | 0.390 [0.303, 0.468] | 0.363 [0.283, 0.434] | 0.402 [0.311, 0.479] |
| C3M e205, d_F cal.* | 0.329 [0.255, 0.398] | 0.351 [0.275, 0.422] | 0.345 [0.266, 0.415] | 0.345 [0.265, 0.417] | 0.342 [0.261, 0.417] | 0.308 [0.236, 0.376] | 0.389 [0.303, 0.467] |
| C3M e205, d_F non cal. | 0.371 [0.287, 0.444] | 0.387 [0.304, 0.460] | 0.382 [0.297, 0.456] | 0.375 [0.290, 0.450] | 0.381 [0.295, 0.459] | 0.354 [0.272, 0.428] | 0.411 [0.323, 0.489] |
| **C3M e205, d_P** | 0.382 [0.304, 0.452] | 0.398 [0.321, 0.469] | 0.393 [0.313, 0.464] | 0.387 [0.307, 0.459] | 0.389 [0.310, 0.463] | 0.357 [0.281, 0.426] | 0.417 [0.336, 0.491] |
| **ICP + Chamfer in mm (per coppia)** | 0.383 [0.298, 0.456] | 0.389 [0.305, 0.461] | 0.389 [0.305, 0.463] | 0.376 [0.290, 0.452] | 0.387 [0.301, 0.460] | 0.378 [0.296, 0.450] | 0.390 [0.304, 0.463] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.397 [0.320, 0.467] | 0.400 [0.323, 0.471] | 0.403 [0.325, 0.475] | 0.389 [0.308, 0.460] | 0.400 [0.324, 0.471] | 0.396 [0.318, 0.466] | 0.400 [0.324, 0.472] |
| **NICP per coppia in mm** | 0.314 [0.232, 0.391] | 0.314 [0.232, 0.391] | 0.317 [0.234, 0.396] | 0.309 [0.224, 0.390] | 0.317 [0.235, 0.394] | 0.315 [0.235, 0.390] | - |
| **NICP per coppia cs** | 0.323 [0.242, 0.401] | 0.321 [0.241, 0.400] | 0.325 [0.242, 0.404] | 0.320 [0.236, 0.399] | 0.325 [0.243, 0.403] | 0.327 [0.247, 0.404] | - |
| **NICP su template in mm** | 0.156 [0.061, 0.253] | 0.156 [0.061, 0.254] | 0.157 [0.060, 0.254] | 0.160 [0.066, 0.258] | 0.155 [0.060, 0.251] | 0.150 [0.055, 0.248] | 0.156 [0.061, 0.254] |
| **NICP su template cs** | 0.175 [0.092, 0.257] | 0.181 [0.097, 0.265] | 0.174 [0.093, 0.257] | 0.178 [0.096, 0.261] | 0.178 [0.093, 0.262] | 0.165 [0.087, 0.241] | 0.181 [0.095, 0.264] |
| **varifold in mm (massa unitaria)** | 0.276 [0.211, 0.337] | 0.313 [0.243, 0.377] | 0.297 [0.232, 0.358] | 0.313 [0.243, 0.377] | 0.312 [0.242, 0.376] | 0.333 [0.251, 0.407] | 0.321 [0.254, 0.383] |
| **GNM (visto) preregistrato, coefficienti** | 0.207 [0.127, 0.277] | 0.210 [0.127, 0.282] | 0.203 [0.121, 0.273] | 0.211 [0.130, 0.281] | 0.211 [0.130, 0.279] | 0.202 [0.122, 0.273] | 0.207 [0.127, 0.278] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.174 [0.078, 0.265] | 0.178 [0.079, 0.270] | 0.172 [0.073, 0.265] | 0.180 [0.085, 0.269] | 0.175 [0.075, 0.266] | 0.166 [0.071, 0.257] | 0.174 [0.077, 0.265] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.228 [0.145, 0.308] | 0.233 [0.150, 0.313] | 0.226 [0.144, 0.304] | 0.231 [0.148, 0.312] | 0.234 [0.151, 0.315] | 0.217 [0.134, 0.296] | 0.231 [0.147, 0.310] |
| **GNM (visto) vA, coefficienti*** | 0.211 [0.130, 0.284] | 0.215 [0.133, 0.288] | 0.206 [0.125, 0.279] | 0.213 [0.134, 0.285] | 0.215 [0.133, 0.288] | 0.207 [0.126, 0.283] | 0.210 [0.130, 0.283] |
| GNM (visto) vA, mesh d'identita' FR* | 0.179 [0.080, 0.273] | 0.182 [0.083, 0.277] | 0.177 [0.078, 0.271] | 0.185 [0.088, 0.278] | 0.180 [0.079, 0.275] | 0.171 [0.076, 0.263] | 0.179 [0.080, 0.273] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.222 [0.138, 0.300] | 0.226 [0.142, 0.305] | 0.221 [0.138, 0.300] | 0.224 [0.142, 0.303] | 0.228 [0.145, 0.306] | 0.212 [0.128, 0.290] | 0.225 [0.142, 0.304] |
| **GNM (visto) vB, coefficienti*** | 0.304 [0.202, 0.390] | 0.302 [0.201, 0.388] | 0.304 [0.202, 0.392] | 0.305 [0.204, 0.393] | 0.303 [0.201, 0.390] | 0.307 [0.205, 0.393] | 0.301 [0.201, 0.386] |
| GNM (visto) vB, mesh d'identita' FR* | 0.275 [0.171, 0.366] | 0.276 [0.171, 0.366] | 0.272 [0.170, 0.363] | 0.279 [0.175, 0.370] | 0.274 [0.170, 0.365] | 0.273 [0.172, 0.363] | 0.275 [0.171, 0.366] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.399 [0.315, 0.473] | 0.400 [0.316, 0.475] | 0.398 [0.313, 0.472] | 0.400 [0.314, 0.475] | 0.399 [0.315, 0.474] | 0.398 [0.313, 0.472] | 0.399 [0.315, 0.474] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.358 [0.259, 0.439] | 0.359 [0.259, 0.441] | 0.355 [0.256, 0.438] | 0.361 [0.263, 0.441] | 0.357 [0.259, 0.438] | 0.357 [0.258, 0.439] | 0.358 [0.258, 0.439] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.150 [0.073, 0.223] | 0.151 [0.072, 0.225] | 0.152 [0.073, 0.223] | 0.154 [0.078, 0.225] | 0.151 [0.073, 0.224] | 0.145 [0.067, 0.217] | 0.153 [0.075, 0.227] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.172 [0.086, 0.255] | 0.172 [0.084, 0.255] | 0.176 [0.090, 0.258] | 0.175 [0.089, 0.256] | 0.170 [0.084, 0.253] | 0.167 [0.078, 0.249] | 0.173 [0.087, 0.255] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.162 [0.078, 0.240] | 0.161 [0.079, 0.241] | 0.162 [0.078, 0.240] | 0.167 [0.084, 0.244] | 0.163 [0.081, 0.243] | 0.157 [0.073, 0.236] | 0.166 [0.080, 0.246] |
| **FLAME 2023 Open vA, coefficienti*** | 0.156 [0.077, 0.229] | 0.156 [0.076, 0.231] | 0.157 [0.078, 0.230] | 0.159 [0.084, 0.231] | 0.156 [0.076, 0.231] | 0.150 [0.070, 0.223] | 0.158 [0.079, 0.233] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.174 [0.081, 0.260] | 0.173 [0.082, 0.260] | 0.178 [0.087, 0.264] | 0.177 [0.086, 0.262] | 0.172 [0.079, 0.259] | 0.169 [0.076, 0.257] | 0.174 [0.081, 0.260] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.171 [0.087, 0.249] | 0.170 [0.089, 0.249] | 0.171 [0.087, 0.246] | 0.176 [0.094, 0.254] | 0.173 [0.091, 0.251] | 0.166 [0.085, 0.244] | 0.175 [0.090, 0.254] |
| **FLAME 2023 Open vB, coefficienti*** | 0.268 [0.159, 0.360] | 0.265 [0.154, 0.360] | 0.269 [0.158, 0.364] | 0.275 [0.165, 0.366] | 0.266 [0.157, 0.358] | 0.265 [0.155, 0.360] | 0.265 [0.156, 0.357] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.274 [0.173, 0.366] | 0.274 [0.175, 0.365] | 0.277 [0.176, 0.366] | 0.276 [0.173, 0.365] | 0.272 [0.170, 0.363] | 0.273 [0.170, 0.367] | 0.275 [0.173, 0.366] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.382 [0.297, 0.459] | 0.380 [0.295, 0.459] | 0.382 [0.297, 0.459] | 0.384 [0.298, 0.462] | 0.382 [0.297, 0.461] | 0.382 [0.297, 0.461] | 0.383 [0.298, 0.461] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.345 [0.250, 0.430] | 0.344 [0.246, 0.432] | 0.346 [0.251, 0.429] | 0.347 [0.251, 0.428] | 0.343 [0.247, 0.430] | 0.344 [0.246, 0.433] | 0.346 [0.250, 0.431] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.166 [0.122, 0.213] | 0.187 [0.138, 0.237] | 0.189 [0.141, 0.239] | 0.196 [0.148, 0.250] | 0.186 [0.137, 0.236] | 0.131 [0.065, 0.204] | 0.185 [0.138, 0.237] |
| **ShapeDNA k=50** | 0.070 [0.030, 0.105] | 0.089 [0.031, 0.142] | 0.093 [0.035, 0.148] | 0.091 [0.034, 0.143] | 0.091 [0.033, 0.147] | 0.093 [0.028, 0.154] | 0.147 [0.063, 0.227] |
| **HKS globale** | 0.037 [0.009, 0.065] | 0.046 [0.005, 0.087] | 0.046 [0.005, 0.088] | 0.047 [0.005, 0.087] | 0.046 [0.005, 0.088] | 0.065 [-0.001, 0.128] | 0.078 [0.017, 0.140] |
| **WKS globale** | 0.049 [0.022, 0.076] | 0.062 [0.019, 0.103] | 0.058 [0.019, 0.095] | 0.057 [0.018, 0.094] | 0.051 [0.016, 0.086] | 0.111 [0.047, 0.175] | 0.068 [-0.003, 0.134] |
| **taglia stimata dalla mesh (non oracolo)** | 0.090 [-0.001, 0.181] | 0.090 [-0.004, 0.184] | 0.088 [-0.000, 0.176] | 0.090 [-0.005, 0.184] | 0.090 [-0.004, 0.184] | 0.091 [0.003, 0.179] | 0.093 [-0.005, 0.189] |
| **taglia oracolo (S di FR)** | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.075 [0.047, 0.104] | 0.103 [0.063, 0.145] | 0.079 [0.043, 0.117] | 0.101 [0.062, 0.141] | 0.096 [0.058, 0.136] | 0.124 [0.041, 0.205] | 0.146 [0.093, 0.203] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.079 [0.051, 0.110] | 0.104 [0.065, 0.148] | 0.084 [0.048, 0.124] | 0.103 [0.065, 0.146] | 0.100 [0.061, 0.143] | 0.140 [0.058, 0.220] | 0.142 [0.089, 0.204] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.094 [0.053, 0.133] | 0.113 [0.062, 0.155] | 0.112 [0.060, 0.159] | 0.133 [0.074, 0.186] | 0.105 [0.058, 0.147] | 0.122 [0.062, 0.177] | 0.190 [0.111, 0.258] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.092 [0.051, 0.132] | 0.106 [0.058, 0.152] | 0.105 [0.055, 0.155] | 0.130 [0.075, 0.183] | 0.102 [0.056, 0.145] | 0.124 [0.060, 0.180] | 0.183 [0.104, 0.254] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.106 [0.076, 0.133] | 0.159 [0.115, 0.200] | 0.139 [0.098, 0.177] | 0.158 [0.113, 0.200] | 0.150 [0.108, 0.189] | 0.084 [0.018, 0.156] | 0.237 [0.163, 0.303] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.107 [0.077, 0.134] | 0.161 [0.116, 0.201] | 0.140 [0.098, 0.177] | 0.160 [0.114, 0.202] | 0.150 [0.108, 0.188] | 0.087 [0.026, 0.153] | 0.241 [0.167, 0.308] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.200 [0.120, 0.274] | 0.219 [0.131, 0.301] | 0.209 [0.126, 0.286] | 0.220 [0.130, 0.301] | 0.219 [0.130, 0.302] | 0.228 [0.130, 0.314] | - |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.225 [0.160, 0.278] | 0.262 [0.191, 0.320] | 0.252 [0.182, 0.309] | 0.263 [0.192, 0.321] | 0.260 [0.188, 0.320] | 0.234 [0.152, 0.302] | - |
| LPIPS alex su render [profondita'], mm (FR) | 0.143 [0.090, 0.194] | 0.167 [0.104, 0.226] | 0.161 [0.096, 0.218] | 0.171 [0.107, 0.232] | 0.166 [0.104, 0.225] | 0.201 [0.123, 0.273] | 0.248 [0.161, 0.328] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.140 [0.093, 0.184] | 0.161 [0.108, 0.212] | 0.157 [0.104, 0.207] | 0.166 [0.110, 0.218] | 0.161 [0.108, 0.212] | 0.214 [0.142, 0.277] | 0.243 [0.167, 0.311] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.383 [0.297, 0.463] | 0.389 [0.303, 0.468] | 0.391 [0.303, 0.471] | 0.385 [0.293, 0.466] | 0.390 [0.303, 0.470] | 0.408 [0.319, 0.490] | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.401 [0.326, 0.474] | 0.407 [0.330, 0.481] | 0.410 [0.333, 0.483] | 0.404 [0.327, 0.479] | 0.410 [0.335, 0.484] | 0.431 [0.350, 0.508] | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.521 [0.447, 0.590] | 0.523 [0.448, 0.590] | 0.523 [0.449, 0.591] | 0.520 [0.446, 0.588] | 0.523 [0.449, 0.591] | 0.530 [0.454, 0.599] | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.330 [0.246, 0.417] | 0.329 [0.244, 0.418] | 0.334 [0.249, 0.421] | 0.332 [0.246, 0.419] | 0.330 [0.245, 0.417] | 0.333 [0.247, 0.422] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.193 [0.151, 0.229] | 0.251 [0.194, 0.300] | 0.248 [0.192, 0.296] | 0.245 [0.189, 0.293] | 0.249 [0.192, 0.298] | 0.227 [0.147, 0.301] | 0.370 [0.291, 0.441] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.198 [0.158, 0.231] | 0.262 [0.210, 0.306] | 0.260 [0.208, 0.303] | 0.257 [0.204, 0.302] | 0.261 [0.208, 0.305] | 0.223 [0.141, 0.296] | 0.387 [0.317, 0.451] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.249 [0.193, 0.303] | 0.304 [0.238, 0.366] | 0.302 [0.235, 0.365] | 0.294 [0.230, 0.355] | 0.308 [0.241, 0.373] | 0.271 [0.190, 0.344] | 0.417 [0.329, 0.499] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.253 [0.201, 0.303] | 0.311 [0.249, 0.369] | 0.306 [0.246, 0.364] | 0.305 [0.244, 0.362] | 0.312 [0.249, 0.371] | 0.308 [0.228, 0.377] | 0.411 [0.333, 0.483] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.068 [-0.139, +0.011] | -0.072 [-0.143, +0.007] | -0.071 [-0.142, +0.006] | -0.065 [-0.137, +0.013] | -0.066 [-0.140, +0.014] | -0.068 [-0.139, +0.009] | -0.064 [-0.137, +0.013] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.033 [-0.114, +0.053] | -0.035 [-0.120, +0.055] | -0.037 [-0.118, +0.050] | -0.030 [-0.113, +0.060] | -0.031 [-0.113, +0.058] | -0.037 [-0.116, +0.046] | -0.026 [-0.108, +0.062] |
| NICP per coppia in mm | +0.063 [-0.027, +0.152] | +0.065 [-0.025, +0.155] | +0.063 [-0.028, +0.153] | +0.065 [-0.031, +0.157] | +0.066 [-0.026, +0.157] | +0.055 [-0.032, +0.140] | - |
| NICP per coppia cs | +0.054 [-0.043, +0.147] | +0.057 [-0.041, +0.151] | +0.056 [-0.043, +0.152] | +0.055 [-0.045, +0.150] | +0.059 [-0.039, +0.154] | +0.044 [-0.050, +0.132] | - |
| NICP su template in mm | +0.117 [+0.013, +0.219] | +0.119 [+0.015, +0.223] | +0.118 [+0.015, +0.222] | +0.109 [+0.006, +0.210] | +0.122 [+0.019, +0.225] | +0.117 [+0.015, +0.219] | +0.128 [+0.023, +0.232] |
| NICP su template cs | +0.166 [+0.067, +0.272] | +0.163 [+0.060, +0.271] | +0.170 [+0.070, +0.275] | +0.159 [+0.057, +0.267] | +0.168 [+0.064, +0.276] | +0.171 [+0.074, +0.272] | +0.172 [+0.068, +0.280] |
| varifold in mm (massa unitaria) | +0.056 [-0.009, +0.128] | +0.020 [-0.046, +0.094] | +0.037 [-0.026, +0.108] | +0.014 [-0.050, +0.087] | +0.023 [-0.042, +0.098] | -0.002 [-0.078, +0.077] | +0.019 [-0.047, +0.091] |
| GNM (visto) preregistrato, coefficienti | +0.100 [-0.009, +0.201] | +0.099 [-0.010, +0.199] | +0.108 [+0.000, +0.208] | +0.093 [-0.017, +0.195] | +0.102 [-0.010, +0.206] | +0.100 [-0.001, +0.198] | +0.112 [+0.002, +0.213] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.109 [+0.007, +0.214] | +0.108 [+0.003, +0.210] | +0.114 [+0.013, +0.218] | +0.100 [+0.001, +0.204] | +0.114 [+0.009, +0.220] | +0.111 [+0.008, +0.213] | +0.120 [+0.017, +0.225] |
| GNM (visto) vA, coefficienti | +0.110 [+0.005, +0.208] | +0.108 [+0.003, +0.207] | +0.118 [+0.012, +0.219] | +0.104 [-0.002, +0.203] | +0.112 [+0.004, +0.213] | +0.109 [+0.009, +0.212] | +0.122 [+0.015, +0.221] |
| GNM (visto) vA, mesh d'identita' FR | +0.104 [+0.002, +0.210] | +0.103 [-0.001, +0.208] | +0.109 [+0.009, +0.214] | +0.095 [-0.007, +0.203] | +0.109 [+0.003, +0.215] | +0.105 [+0.003, +0.209] | +0.115 [+0.011, +0.222] |
| GNM (visto) vB, coefficienti | +0.041 [-0.060, +0.152] | +0.044 [-0.058, +0.155] | +0.044 [-0.057, +0.156] | +0.036 [-0.065, +0.146] | +0.046 [-0.059, +0.156] | +0.033 [-0.067, +0.146] | +0.055 [-0.048, +0.165] |
| GNM (visto) vB, mesh d'identita' FR | +0.020 [-0.070, +0.118] | +0.021 [-0.068, +0.122] | +0.025 [-0.066, +0.123] | +0.013 [-0.077, +0.111] | +0.026 [-0.065, +0.125] | +0.015 [-0.070, +0.111] | +0.030 [-0.061, +0.128] |
| GNM (visto) vB, composizione S_B + k d_P | -0.041 [-0.126, +0.052] | -0.040 [-0.127, +0.053] | -0.037 [-0.122, +0.056] | -0.047 [-0.132, +0.046] | -0.035 [-0.121, +0.059] | -0.047 [-0.128, +0.047] | -0.030 [-0.116, +0.065] |
| FLAME 2023 Open preregistrato, coefficienti | +0.170 [+0.067, +0.274] | +0.173 [+0.067, +0.279] | +0.169 [+0.065, +0.276] | +0.164 [+0.064, +0.268] | +0.172 [+0.067, +0.280] | +0.171 [+0.073, +0.271] | +0.178 [+0.072, +0.285] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.127 [+0.017, +0.232] | +0.130 [+0.018, +0.237] | +0.123 [+0.011, +0.228] | +0.121 [+0.013, +0.223] | +0.131 [+0.017, +0.239] | +0.128 [+0.022, +0.230] | +0.136 [+0.023, +0.244] |
| FLAME 2023 Open vA, coefficienti | +0.162 [+0.057, +0.267] | +0.166 [+0.060, +0.272] | +0.161 [+0.056, +0.269] | +0.157 [+0.057, +0.261] | +0.164 [+0.058, +0.273] | +0.163 [+0.063, +0.264] | +0.171 [+0.065, +0.278] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.120 [+0.012, +0.221] | +0.124 [+0.016, +0.228] | +0.116 [+0.010, +0.218] | +0.114 [+0.009, +0.215] | +0.126 [+0.012, +0.234] | +0.122 [+0.017, +0.227] | +0.131 [+0.020, +0.235] |
| FLAME 2023 Open vB, coefficienti | +0.092 [-0.014, +0.204] | +0.096 [-0.013, +0.209] | +0.093 [-0.015, +0.206] | +0.082 [-0.026, +0.192] | +0.097 [-0.011, +0.212] | +0.090 [-0.017, +0.204] | +0.106 [-0.000, +0.219] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.026 [-0.070, +0.133] | +0.029 [-0.070, +0.135] | +0.025 [-0.072, +0.130] | +0.021 [-0.072, +0.125] | +0.032 [-0.067, +0.138] | +0.023 [-0.074, +0.131] | +0.036 [-0.061, +0.143] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.026 [-0.123, +0.079] | -0.022 [-0.122, +0.084] | -0.026 [-0.123, +0.079] | -0.031 [-0.125, +0.074] | -0.022 [-0.118, +0.085] | -0.029 [-0.122, +0.079] | -0.016 [-0.114, +0.090] |
| ArcFace-render, normal map 3 viste (G0) | +0.187 [+0.095, +0.272] | +0.170 [+0.078, +0.255] | +0.169 [+0.077, +0.253] | +0.157 [+0.065, +0.243] | +0.173 [+0.079, +0.259] | +0.212 [+0.097, +0.320] | +0.180 [+0.090, +0.268] |
| ShapeDNA k=50 | +0.267 [+0.182, +0.348] | +0.248 [+0.157, +0.339] | +0.244 [+0.155, +0.330] | +0.241 [+0.151, +0.331] | +0.247 [+0.157, +0.338] | +0.242 [+0.142, +0.343] | +0.198 [+0.088, +0.303] |
| HKS globale | +0.305 [+0.222, +0.387] | +0.298 [+0.206, +0.388] | +0.299 [+0.208, +0.390] | +0.292 [+0.202, +0.380] | +0.300 [+0.207, +0.392] | +0.279 [+0.185, +0.372] | +0.278 [+0.172, +0.382] |
| WKS globale | +0.290 [+0.210, +0.370] | +0.279 [+0.198, +0.368] | +0.280 [+0.200, +0.358] | +0.280 [+0.199, +0.361] | +0.288 [+0.208, +0.368] | +0.225 [+0.122, +0.329] | +0.274 [+0.178, +0.370] |
| taglia stimata dalla mesh (non oracolo) | +0.197 [+0.097, +0.286] | +0.198 [+0.096, +0.289] | +0.203 [+0.104, +0.290] | +0.193 [+0.094, +0.282] | +0.201 [+0.097, +0.293] | +0.191 [+0.096, +0.280] | +0.203 [+0.099, +0.297] |
| taglia oracolo (S di FR) | +0.132 [+0.005, +0.251] | +0.134 [+0.007, +0.254] | +0.134 [+0.009, +0.254] | +0.128 [+0.002, +0.249] | +0.136 [+0.007, +0.259] | +0.127 [-0.003, +0.246] | +0.143 [+0.016, +0.264] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.274 [+0.188, +0.354] | +0.254 [+0.167, +0.337] | +0.274 [+0.189, +0.355] | +0.250 [+0.161, +0.332] | +0.263 [+0.175, +0.347] | +0.210 [+0.097, +0.327] | +0.231 [+0.129, +0.325] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.258 [+0.175, +0.341] | +0.244 [+0.155, +0.333] | +0.244 [+0.156, +0.329] | +0.222 [+0.132, +0.316] | +0.252 [+0.165, +0.340] | +0.235 [+0.147, +0.323] | +0.188 [+0.088, +0.292] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.243 [+0.152, +0.325] | +0.202 [+0.114, +0.289] | +0.219 [+0.129, +0.301] | +0.197 [+0.106, +0.282] | +0.213 [+0.123, +0.299] | +0.232 [+0.116, +0.345] | +0.146 [+0.046, +0.247] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.107 [+0.026, +0.182] | +0.085 [+0.002, +0.166] | +0.099 [+0.017, +0.177] | +0.081 [-0.002, +0.162] | +0.089 [+0.003, +0.171] | +0.070 [-0.009, +0.150] | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.181 [+0.104, +0.258] | +0.157 [+0.077, +0.241] | +0.163 [+0.080, +0.247] | +0.147 [+0.066, +0.230] | +0.161 [+0.080, +0.245] | +0.106 [+0.018, +0.187] | +0.077 [-0.013, +0.178] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.067 [-0.143, +0.019] | -0.070 [-0.147, +0.016] | -0.071 [-0.146, +0.015] | -0.072 [-0.148, +0.014] | -0.068 [-0.145, +0.019] | -0.098 [-0.175, -0.010] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.046 [-0.123, +0.039] | -0.050 [-0.130, +0.036] | -0.050 [-0.131, +0.035] | -0.052 [-0.132, +0.033] | -0.048 [-0.129, +0.040] | -0.081 [-0.160, +0.010] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.138 [-0.242, -0.033] | -0.138 [-0.241, -0.031] | -0.137 [-0.238, -0.028] | -0.140 [-0.243, -0.035] | -0.135 [-0.239, -0.029] | -0.153 [-0.254, -0.051] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.034 [-0.088, +0.148] | +0.036 [-0.087, +0.150] | +0.033 [-0.087, +0.146] | +0.027 [-0.094, +0.139] | +0.040 [-0.084, +0.156] | +0.025 [-0.099, +0.142] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.146 [+0.078, +0.214] | +0.086 [+0.019, +0.155] | +0.088 [+0.022, +0.157] | +0.086 [+0.019, +0.156] | +0.091 [+0.022, +0.163] | +0.127 [+0.018, +0.239] | -0.030 [-0.105, +0.047] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.091 [+0.024, +0.160] | +0.039 [-0.034, +0.111] | +0.040 [-0.033, +0.110] | +0.043 [-0.029, +0.115] | +0.037 [-0.035, +0.111] | +0.066 [-0.020, +0.156] | -0.065 [-0.149, +0.022] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.072 [-0.151, +0.013] | -0.074 [-0.153, +0.010] | -0.074 [-0.152, +0.009] | -0.067 [-0.147, +0.020] | -0.070 [-0.150, +0.014] | -0.076 [-0.156, +0.006] | -0.065 [-0.145, +0.021] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.037 [-0.118, +0.051] | -0.037 [-0.122, +0.054] | -0.040 [-0.120, +0.048] | -0.031 [-0.117, +0.058] | -0.034 [-0.117, +0.056] | -0.045 [-0.124, +0.039] | -0.027 [-0.110, +0.065] |
| NICP per coppia in mm | +0.059 [-0.025, +0.140] | +0.063 [-0.022, +0.146] | +0.060 [-0.022, +0.144] | +0.063 [-0.027, +0.147] | +0.063 [-0.021, +0.146] | +0.046 [-0.031, +0.121] | - |
| NICP per coppia cs | +0.051 [-0.034, +0.141] | +0.055 [-0.031, +0.144] | +0.053 [-0.033, +0.143] | +0.053 [-0.036, +0.144] | +0.056 [-0.029, +0.147] | +0.036 [-0.047, +0.121] | - |
| NICP su template in mm | +0.113 [-0.004, +0.225] | +0.117 [-0.003, +0.228] | +0.115 [-0.001, +0.229] | +0.107 [-0.014, +0.220] | +0.119 [-0.000, +0.231] | +0.109 [-0.008, +0.221] | +0.127 [+0.008, +0.239] |
| NICP su template cs | +0.162 [+0.052, +0.275] | +0.161 [+0.050, +0.278] | +0.167 [+0.057, +0.280] | +0.157 [+0.047, +0.269] | +0.164 [+0.052, +0.281] | +0.163 [+0.058, +0.271] | +0.171 [+0.059, +0.289] |
| varifold in mm (massa unitaria) | +0.052 [-0.015, +0.121] | +0.018 [-0.053, +0.088] | +0.033 [-0.035, +0.103] | +0.012 [-0.057, +0.083] | +0.020 [-0.051, +0.092] | -0.011 [-0.085, +0.064] | +0.018 [-0.054, +0.089] |
| GNM (visto) preregistrato, coefficienti | +0.097 [-0.010, +0.198] | +0.097 [-0.012, +0.201] | +0.104 [-0.002, +0.207] | +0.091 [-0.018, +0.193] | +0.099 [-0.010, +0.204] | +0.092 [-0.012, +0.191] | +0.111 [+0.005, +0.212] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.105 [-0.004, +0.214] | +0.106 [-0.004, +0.213] | +0.110 [+0.002, +0.219] | +0.098 [-0.011, +0.208] | +0.111 [+0.000, +0.222] | +0.102 [-0.009, +0.210] | +0.119 [+0.010, +0.227] |
| GNM (visto) vA, coefficienti | +0.107 [-0.006, +0.206] | +0.106 [-0.006, +0.209] | +0.115 [+0.005, +0.216] | +0.103 [-0.011, +0.201] | +0.108 [-0.006, +0.212] | +0.101 [-0.002, +0.199] | +0.121 [+0.008, +0.222] |
| GNM (visto) vA, mesh d'identita' FR | +0.100 [-0.007, +0.211] | +0.101 [-0.008, +0.210] | +0.105 [-0.006, +0.216] | +0.093 [-0.014, +0.205] | +0.106 [-0.005, +0.218] | +0.097 [-0.012, +0.207] | +0.114 [+0.004, +0.226] |
| GNM (visto) vB, coefficienti | +0.037 [-0.054, +0.134] | +0.042 [-0.049, +0.139] | +0.041 [-0.050, +0.138] | +0.035 [-0.057, +0.129] | +0.043 [-0.050, +0.139] | +0.025 [-0.064, +0.117] | +0.054 [-0.039, +0.148] |
| GNM (visto) vB, mesh d'identita' FR | +0.016 [-0.079, +0.116] | +0.019 [-0.076, +0.117] | +0.022 [-0.073, +0.120] | +0.011 [-0.085, +0.109] | +0.023 [-0.074, +0.124] | +0.007 [-0.085, +0.109] | +0.029 [-0.066, +0.130] |
| GNM (visto) vB, composizione S_B + k d_P | -0.045 [-0.130, +0.045] | -0.042 [-0.129, +0.048] | -0.040 [-0.127, +0.050] | -0.049 [-0.136, +0.041] | -0.039 [-0.126, +0.051] | -0.055 [-0.140, +0.036] | -0.031 [-0.118, +0.057] |
| FLAME 2023 Open preregistrato, coefficienti | +0.166 [+0.060, +0.266] | +0.171 [+0.064, +0.273] | +0.165 [+0.057, +0.266] | +0.162 [+0.058, +0.261] | +0.169 [+0.061, +0.271] | +0.163 [+0.059, +0.262] | +0.177 [+0.071, +0.279] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.123 [+0.012, +0.228] | +0.128 [+0.016, +0.234] | +0.120 [+0.008, +0.227] | +0.119 [+0.012, +0.223] | +0.128 [+0.015, +0.239] | +0.120 [+0.009, +0.221] | +0.136 [+0.024, +0.243] |
| FLAME 2023 Open vA, coefficienti | +0.158 [+0.053, +0.257] | +0.164 [+0.057, +0.264] | +0.157 [+0.049, +0.258] | +0.155 [+0.049, +0.252] | +0.161 [+0.052, +0.263] | +0.155 [+0.052, +0.252] | +0.170 [+0.062, +0.269] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.117 [+0.007, +0.227] | +0.122 [+0.012, +0.231] | +0.113 [+0.005, +0.222] | +0.113 [+0.003, +0.218] | +0.122 [+0.009, +0.234] | +0.114 [+0.003, +0.222] | +0.130 [+0.019, +0.240] |
| FLAME 2023 Open vB, coefficienti | +0.088 [-0.012, +0.185] | +0.094 [-0.007, +0.193] | +0.090 [-0.009, +0.188] | +0.080 [-0.016, +0.173] | +0.094 [-0.007, +0.193] | +0.082 [-0.017, +0.181] | +0.105 [+0.006, +0.200] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.022 [-0.076, +0.121] | +0.027 [-0.070, +0.128] | +0.022 [-0.076, +0.121] | +0.019 [-0.075, +0.117] | +0.029 [-0.070, +0.129] | +0.015 [-0.083, +0.117] | +0.035 [-0.067, +0.136] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.030 [-0.121, +0.067] | -0.024 [-0.118, +0.073] | -0.029 [-0.121, +0.065] | -0.033 [-0.122, +0.062] | -0.025 [-0.117, +0.072] | -0.037 [-0.129, +0.061] | -0.017 [-0.108, +0.079] |
| ArcFace-render, normal map 3 viste (G0) | +0.184 [+0.093, +0.269] | +0.168 [+0.071, +0.256] | +0.166 [+0.071, +0.252] | +0.155 [+0.061, +0.242] | +0.170 [+0.076, +0.258] | +0.204 [+0.089, +0.306] | +0.180 [+0.087, +0.269] |
| ShapeDNA k=50 | +0.263 [+0.182, +0.341] | +0.246 [+0.156, +0.336] | +0.241 [+0.153, +0.328] | +0.239 [+0.149, +0.326] | +0.244 [+0.153, +0.334] | +0.234 [+0.131, +0.333] | +0.197 [+0.089, +0.299] |
| HKS globale | +0.301 [+0.217, +0.378] | +0.296 [+0.207, +0.380] | +0.296 [+0.208, +0.379] | +0.290 [+0.203, +0.373] | +0.297 [+0.207, +0.380] | +0.271 [+0.178, +0.358] | +0.277 [+0.177, +0.376] |
| WKS globale | +0.286 [+0.205, +0.366] | +0.277 [+0.186, +0.363] | +0.277 [+0.189, +0.353] | +0.278 [+0.192, +0.361] | +0.285 [+0.196, +0.366] | +0.217 [+0.124, +0.314] | +0.273 [+0.170, +0.371] |
| taglia stimata dalla mesh (non oracolo) | +0.193 [+0.075, +0.297] | +0.196 [+0.076, +0.301] | +0.200 [+0.087, +0.300] | +0.191 [+0.072, +0.296] | +0.198 [+0.076, +0.302] | +0.183 [+0.070, +0.285] | +0.202 [+0.080, +0.311] |
| taglia oracolo (S di FR) | +0.128 [-0.015, +0.260] | +0.132 [-0.013, +0.266] | +0.131 [-0.012, +0.262] | +0.126 [-0.018, +0.257] | +0.133 [-0.010, +0.270] | +0.119 [-0.027, +0.249] | +0.142 [-0.002, +0.278] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.270 [+0.182, +0.351] | +0.252 [+0.160, +0.339] | +0.271 [+0.183, +0.351] | +0.248 [+0.158, +0.334] | +0.260 [+0.167, +0.346] | +0.202 [+0.085, +0.317] | +0.230 [+0.127, +0.330] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.254 [+0.174, +0.333] | +0.242 [+0.159, +0.326] | +0.241 [+0.154, +0.323] | +0.220 [+0.136, +0.308] | +0.248 [+0.163, +0.332] | +0.227 [+0.139, +0.312] | +0.187 [+0.090, +0.289] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.239 [+0.151, +0.319] | +0.200 [+0.111, +0.284] | +0.216 [+0.126, +0.301] | +0.195 [+0.105, +0.279] | +0.210 [+0.120, +0.294] | +0.224 [+0.108, +0.329] | +0.145 [+0.047, +0.245] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.103 [+0.017, +0.189] | +0.083 [-0.006, +0.176] | +0.096 [+0.010, +0.184] | +0.079 [-0.008, +0.171] | +0.085 [-0.005, +0.178] | +0.062 [-0.026, +0.151] | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.177 [+0.092, +0.260] | +0.155 [+0.067, +0.245] | +0.160 [+0.069, +0.249] | +0.145 [+0.057, +0.234] | +0.158 [+0.069, +0.249] | +0.098 [+0.009, +0.186] | +0.076 [-0.018, +0.182] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.070 [-0.152, +0.018] | -0.072 [-0.154, +0.016] | -0.074 [-0.156, +0.015] | -0.074 [-0.155, +0.016] | -0.071 [-0.154, +0.018] | -0.106 [-0.188, -0.014] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.050 [-0.123, +0.025] | -0.052 [-0.126, +0.024] | -0.054 [-0.128, +0.021] | -0.054 [-0.128, +0.024] | -0.051 [-0.126, +0.025] | -0.089 [-0.166, -0.010] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.142 [-0.236, -0.037] | -0.140 [-0.234, -0.034] | -0.140 [-0.235, -0.038] | -0.142 [-0.236, -0.040] | -0.139 [-0.237, -0.033] | -0.161 [-0.254, -0.060] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.030 [-0.087, +0.141] | +0.034 [-0.084, +0.144] | +0.030 [-0.085, +0.140] | +0.025 [-0.090, +0.134] | +0.036 [-0.083, +0.149] | +0.017 [-0.100, +0.126] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.142 [+0.075, +0.210] | +0.084 [+0.015, +0.153] | +0.085 [+0.016, +0.153] | +0.084 [+0.016, +0.152] | +0.088 [+0.017, +0.157] | +0.119 [+0.013, +0.227] | -0.031 [-0.106, +0.053] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.088 [+0.024, +0.153] | +0.037 [-0.029, +0.107] | +0.036 [-0.028, +0.105] | +0.041 [-0.023, +0.107] | +0.034 [-0.031, +0.105] | +0.058 [-0.017, +0.138] | -0.066 [-0.140, +0.013] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.049 [-0.127, +0.038] | -0.051 [-0.128, +0.038] | -0.052 [-0.128, +0.035] | -0.044 [-0.121, +0.043] | -0.047 [-0.125, +0.042] | -0.055 [-0.132, +0.031] | -0.043 [-0.120, +0.046] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.063 [-0.140, +0.017] | -0.063 [-0.140, +0.020] | -0.067 [-0.143, +0.014] | -0.058 [-0.133, +0.025] | -0.061 [-0.138, +0.022] | -0.073 [-0.149, +0.007] | -0.053 [-0.133, +0.031] |
| NICP per coppia in mm | +0.019 [-0.063, +0.103] | +0.024 [-0.061, +0.108] | +0.020 [-0.063, +0.104] | +0.022 [-0.063, +0.113] | +0.023 [-0.061, +0.106] | +0.008 [-0.070, +0.087] | - |
| NICP per coppia cs | +0.010 [-0.068, +0.095] | +0.016 [-0.063, +0.098] | +0.012 [-0.067, +0.095] | +0.012 [-0.070, +0.096] | +0.015 [-0.067, +0.099] | -0.004 [-0.078, +0.074] | - |
| NICP su template in mm | +0.178 [+0.061, +0.281] | +0.181 [+0.064, +0.284] | +0.180 [+0.063, +0.283] | +0.172 [+0.054, +0.276] | +0.185 [+0.067, +0.288] | +0.173 [+0.060, +0.277] | +0.191 [+0.074, +0.296] |
| NICP su template cs | +0.159 [+0.045, +0.262] | +0.157 [+0.043, +0.263] | +0.163 [+0.051, +0.265] | +0.154 [+0.042, +0.258] | +0.162 [+0.047, +0.266] | +0.157 [+0.049, +0.258] | +0.166 [+0.053, +0.274] |
| varifold in mm (massa unitaria) | +0.057 [-0.006, +0.125] | +0.025 [-0.039, +0.094] | +0.040 [-0.023, +0.108] | +0.019 [-0.044, +0.086] | +0.027 [-0.037, +0.096] | -0.010 [-0.083, +0.067] | +0.026 [-0.042, +0.091] |
| GNM (visto) preregistrato, coefficienti | +0.126 [+0.028, +0.222] | +0.128 [+0.027, +0.225] | +0.134 [+0.036, +0.230] | +0.121 [+0.023, +0.219] | +0.128 [+0.028, +0.227] | +0.120 [+0.025, +0.216] | +0.140 [+0.042, +0.237] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.106 [+0.006, +0.206] | +0.105 [+0.006, +0.206] | +0.111 [+0.011, +0.211] | +0.101 [+0.002, +0.204] | +0.106 [+0.002, +0.209] | +0.106 [+0.008, +0.203] | +0.117 [+0.016, +0.217] |
| GNM (visto) vA, coefficienti | +0.123 [+0.018, +0.219] | +0.123 [+0.017, +0.220] | +0.131 [+0.026, +0.226] | +0.119 [+0.014, +0.217] | +0.124 [+0.020, +0.223] | +0.115 [+0.012, +0.212] | +0.137 [+0.032, +0.235] |
| GNM (visto) vA, mesh d'identita' SR | +0.112 [+0.014, +0.217] | +0.112 [+0.013, +0.219] | +0.116 [+0.018, +0.220] | +0.108 [+0.009, +0.214] | +0.112 [+0.013, +0.218] | +0.110 [+0.016, +0.211] | +0.122 [+0.025, +0.229] |
| GNM (visto) vB, coefficienti | +0.030 [-0.059, +0.130] | +0.036 [-0.054, +0.138] | +0.033 [-0.056, +0.133] | +0.026 [-0.061, +0.126] | +0.036 [-0.054, +0.137] | +0.016 [-0.074, +0.116] | +0.046 [-0.040, +0.148] |
| GNM (visto) vB, mesh d'identita' SR | -0.065 [-0.142, +0.012] | -0.062 [-0.141, +0.015] | -0.061 [-0.137, +0.016] | -0.068 [-0.146, +0.011] | -0.060 [-0.139, +0.018] | -0.075 [-0.149, +0.002] | -0.052 [-0.130, +0.026] |
| FLAME 2023 Open preregistrato, coefficienti | +0.183 [+0.077, +0.284] | +0.187 [+0.080, +0.291] | +0.185 [+0.076, +0.287] | +0.178 [+0.073, +0.277] | +0.189 [+0.082, +0.292] | +0.178 [+0.072, +0.278] | +0.194 [+0.087, +0.297] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.171 [+0.071, +0.279] | +0.176 [+0.075, +0.285] | +0.175 [+0.071, +0.285] | +0.165 [+0.065, +0.273] | +0.176 [+0.071, +0.286] | +0.166 [+0.066, +0.272] | +0.181 [+0.077, +0.292] |
| FLAME 2023 Open vA, coefficienti | +0.178 [+0.071, +0.276] | +0.182 [+0.073, +0.281] | +0.180 [+0.069, +0.279] | +0.173 [+0.068, +0.270] | +0.183 [+0.075, +0.285] | +0.173 [+0.067, +0.272] | +0.189 [+0.080, +0.290] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.162 [+0.063, +0.267] | +0.168 [+0.067, +0.270] | +0.166 [+0.066, +0.273] | +0.156 [+0.058, +0.259] | +0.167 [+0.066, +0.271] | +0.156 [+0.059, +0.259] | +0.172 [+0.070, +0.279] |
| FLAME 2023 Open vB, coefficienti | +0.066 [-0.032, +0.167] | +0.072 [-0.027, +0.175] | +0.068 [-0.028, +0.170] | +0.056 [-0.039, +0.155] | +0.073 [-0.028, +0.177] | +0.058 [-0.042, +0.163] | +0.083 [-0.014, +0.183] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.049 [-0.129, +0.037] | -0.043 [-0.124, +0.045] | -0.045 [-0.125, +0.042] | -0.052 [-0.134, +0.037] | -0.043 [-0.125, +0.043] | -0.059 [-0.142, +0.027] | -0.036 [-0.117, +0.051] |
| ArcFace-render, normal map 3 viste (G0) | +0.167 [+0.074, +0.252] | +0.151 [+0.057, +0.241] | +0.148 [+0.054, +0.235] | +0.135 [+0.042, +0.221] | +0.153 [+0.060, +0.243] | +0.191 [+0.074, +0.299] | +0.162 [+0.066, +0.251] |
| ShapeDNA k=50 | +0.264 [+0.183, +0.341] | +0.248 [+0.156, +0.337] | +0.243 [+0.154, +0.331] | +0.241 [+0.149, +0.325] | +0.248 [+0.153, +0.338] | +0.229 [+0.131, +0.324] | +0.200 [+0.092, +0.305] |
| HKS globale | +0.296 [+0.212, +0.370] | +0.291 [+0.201, +0.375] | +0.291 [+0.200, +0.374] | +0.285 [+0.196, +0.368] | +0.293 [+0.202, +0.377] | +0.257 [+0.167, +0.353] | +0.269 [+0.167, +0.369] |
| WKS globale | +0.285 [+0.204, +0.357] | +0.276 [+0.192, +0.355] | +0.279 [+0.194, +0.353] | +0.275 [+0.191, +0.349] | +0.288 [+0.204, +0.363] | +0.211 [+0.119, +0.304] | +0.279 [+0.182, +0.380] |
| taglia stimata dalla mesh (non oracolo) | +0.244 [+0.134, +0.340] | +0.247 [+0.137, +0.346] | +0.249 [+0.140, +0.343] | +0.242 [+0.132, +0.338] | +0.249 [+0.140, +0.348] | +0.232 [+0.123, +0.323] | +0.254 [+0.144, +0.356] |
| taglia oracolo (S di FR) | +0.315 [+0.186, +0.427] | +0.319 [+0.189, +0.433] | +0.318 [+0.189, +0.430] | +0.313 [+0.183, +0.427] | +0.321 [+0.194, +0.434] | +0.304 [+0.176, +0.415] | +0.328 [+0.200, +0.441] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.255 [+0.170, +0.330] | +0.233 [+0.143, +0.314] | +0.253 [+0.165, +0.330] | +0.228 [+0.140, +0.308] | +0.240 [+0.148, +0.320] | +0.182 [+0.067, +0.285] | +0.205 [+0.111, +0.297] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.242 [+0.156, +0.320] | +0.231 [+0.142, +0.314] | +0.232 [+0.144, +0.316] | +0.202 [+0.114, +0.290] | +0.237 [+0.150, +0.319] | +0.199 [+0.113, +0.292] | +0.164 [+0.067, +0.261] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.226 [+0.139, +0.304] | +0.177 [+0.089, +0.261] | +0.197 [+0.111, +0.277] | +0.172 [+0.083, +0.255] | +0.189 [+0.099, +0.274] | +0.236 [+0.116, +0.332] | +0.107 [+0.003, +0.205] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.109 [+0.039, +0.179] | +0.076 [+0.002, +0.148] | +0.085 [+0.014, +0.157] | +0.069 [-0.004, +0.142] | +0.079 [+0.007, +0.152] | +0.089 [+0.014, +0.171] | - |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.194 [+0.116, +0.265] | +0.177 [+0.097, +0.253] | +0.180 [+0.098, +0.254] | +0.166 [+0.087, +0.242] | +0.178 [+0.099, +0.255] | +0.109 [+0.024, +0.189] | +0.104 [+0.015, +0.195] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.050 [-0.133, +0.035] | -0.051 [-0.135, +0.037] | -0.054 [-0.135, +0.034] | -0.053 [-0.137, +0.033] | -0.050 [-0.132, +0.036] | -0.085 [-0.172, +0.007] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.068 [-0.140, +0.006] | -0.070 [-0.143, +0.004] | -0.073 [-0.145, +0.000] | -0.073 [-0.146, +0.004] | -0.071 [-0.144, +0.005] | -0.108 [-0.183, -0.030] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.188 [-0.274, -0.096] | -0.185 [-0.272, -0.093] | -0.186 [-0.274, -0.092] | -0.188 [-0.275, -0.097] | -0.184 [-0.271, -0.091] | -0.208 [-0.293, -0.114] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.004 [-0.098, +0.106] | +0.009 [-0.095, +0.111] | +0.003 [-0.101, +0.104] | -0.001 [-0.104, +0.100] | +0.010 [-0.095, +0.115] | -0.010 [-0.117, +0.092] | - |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.136 [+0.067, +0.201] | +0.076 [+0.007, +0.144] | +0.077 [+0.009, +0.144] | +0.075 [+0.009, +0.142] | +0.079 [+0.010, +0.147] | +0.100 [-0.008, +0.207] | -0.040 [-0.111, +0.038] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | +0.081 [+0.016, +0.143] | +0.027 [-0.041, +0.093] | +0.030 [-0.036, +0.095] | +0.027 [-0.041, +0.093] | +0.028 [-0.041, +0.095] | +0.015 [-0.061, +0.093] | -0.064 [-0.140, +0.013] |

## Crop (descrittivo, supplementary)

Fuori dalla tesi principale (decisione del PI): qui, completi, tutti gli insiemi con la crop: `crop` (una crop contro una topologia diversa), `all_cross` (topologie diverse, crop compresa), `stessa_topologia crop` (crop - crop) e `stessa_topologia tutte`. Stesse righe della maschera, stesse repliche, stesse colonne delle sezioni principali.

### Miglior nostro contro miglior concorrente (crop)

| dominio | GT | insieme | righe | miglior nostro | rho | miglior concorrente | rho | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | FR | crop | 49500 | C3M e205, d_F non cal. | 0.345 [0.262, 0.420] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.481 [0.394, 0.559] | -0.135 [-0.225, -0.039] |
| faceverse_neutral | FR | all_cross | 148500 | C3M e123, d_F non cal. | 0.354 [0.270, 0.431] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.479 [0.393, 0.557] | -0.125 [-0.218, -0.032] |
| faceverse_neutral | FR | stessa_topologia crop | 4950 | ctrlfr s1234 | 0.386 [0.304, 0.464] | ICP + Chamfer in mm (per coppia) | 0.444 [0.363, 0.515] | -0.058 [-0.143, +0.022] |
| faceverse_neutral | FR | stessa_topologia tutte | 29700 | C3M e205, d_F cal. | 0.395 [0.312, 0.470] | ICP + Chamfer in mm (per coppia) | 0.419 [0.337, 0.492] | -0.024 [-0.095, +0.050] |
| faceverse_neutral | SR | crop | 49500 | C3M e205, d_P | 0.372 [0.294, 0.442] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.523 [0.448, 0.591] | -0.151 [-0.235, -0.063] |
| faceverse_neutral | SR | all_cross | 148500 | C3M e205, d_P | 0.378 [0.304, 0.448] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | 0.521 [0.447, 0.589] | -0.143 [-0.229, -0.057] |
| faceverse_neutral | SR | stessa_topologia crop | 4950 | C3M e205, d_P | 0.419 [0.338, 0.498] | ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.469 [0.384, 0.553] | -0.050 [-0.140, +0.032] |
| faceverse_neutral | SR | stessa_topologia tutte | 29700 | C3M e205, d_P | 0.417 [0.339, 0.490] | ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.419 [0.340, 0.491] | -0.002 [-0.073, +0.069] |

Miglior concorrente: * = lettura post hoc (vB, composizione); "(landmark esatti)" = protocollo coi landmark oracolo (NoW originale, NoW metrical, REALY, Stirling). Le letture col rilevatore (`lmk_det`) sono nel passo `righe proprie` (sottocampione HIFI3D e FaceScape), nella sezione "Protocolli: landmark esatti e landmark rilevati" (crop: "Protocolli, landmark esatti e rilevati (crop)").

Ogni lettura nostra di riferimento contro il miglior concorrente (crop, all_cross):

| dominio | GT | insieme | miglior concorrente | lettura nostra | rho nostro | rho concorrente | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_F cal. | 0.282 | 0.481 | -0.198 [-0.304, -0.092] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_F cal. | 0.272 | 0.481 | -0.208 [-0.309, -0.101] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_F non cal. | 0.306 | 0.481 | -0.175 [-0.273, -0.074] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_F non cal. | 0.310 | 0.481 | -0.170 [-0.275, -0.066] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | ctrlfr s1234 | 0.296 | 0.481 | -0.185 [-0.279, -0.094] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | ctrlfr s2345 | 0.269 | 0.481 | -0.211 [-0.309, -0.113] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_F cal. | 0.319 | 0.479 | -0.160 [-0.259, -0.055] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_F cal. | 0.335 | 0.479 | -0.144 [-0.241, -0.044] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_F non cal. | 0.326 | 0.479 | -0.153 [-0.245, -0.055] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_F non cal. | 0.352 | 0.479 | -0.128 [-0.229, -0.027] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | ctrlfr s1234 | 0.316 | 0.479 | -0.163 [-0.259, -0.069] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | ctrlfr s2345 | 0.289 | 0.479 | -0.190 [-0.293, -0.091] |
| faceverse_neutral | SR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_P | 0.304 | 0.523 | -0.219 [-0.306, -0.132] |
| faceverse_neutral | SR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_P | 0.316 | 0.523 | -0.207 [-0.301, -0.109] |
| faceverse_neutral | SR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s1234, d_P | 0.323 | 0.521 | -0.198 [-0.282, -0.109] |
| faceverse_neutral | SR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) (landmark esatti) | factorized s2345, d_P | 0.352 | 0.521 | -0.169 [-0.262, -0.078] |

### Scala dell'allineamento (crop): confronti principali

GT FR:

| dominio | passo | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.017 [-0.066, +0.034] | +0.025 [-0.024, +0.074] | +0.032 [-0.021, +0.089] | +0.037 [-0.016, +0.092] |
| faceverse_neutral | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.024 [-0.063, +0.015] | -0.022 [-0.061, +0.017] | - | - |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.043 [-0.014, +0.099] | +0.061 [+0.002, +0.119] | - | - |
| faceverse_neutral | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.180 [-0.281, -0.080] | -0.140 [-0.248, -0.035] | - | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.006 [-0.058, +0.069] | -0.052 [-0.119, +0.017] | -0.120 [-0.190, -0.051] | -0.097 [-0.167, -0.026] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.011 [-0.082, +0.061] | -0.061 [-0.138, +0.014] | -0.144 [-0.223, -0.068] | -0.106 [-0.187, -0.026] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.011 [-0.086, +0.066] | -0.047 [-0.122, +0.033] | -0.099 [-0.183, -0.008] | -0.070 [-0.148, +0.014] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.006 [-0.080, +0.067] | -0.022 [-0.101, +0.053] | -0.103 [-0.191, -0.019] | -0.045 [-0.129, +0.030] |

GT SR:

| dominio | passo | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.057 [-0.100, -0.017] | -0.019 [-0.059, +0.021] | -0.013 [-0.057, +0.030] | -0.012 [-0.055, +0.032] |
| faceverse_neutral | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.018 [-0.018, +0.055] | +0.018 [-0.018, +0.055] | - | - |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.034 [-0.024, +0.092] | +0.048 [-0.008, +0.103] | - | - |
| faceverse_neutral | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.211 [-0.303, -0.115] | -0.174 [-0.273, -0.071] | - | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.046 [-0.005, +0.099] | +0.030 [-0.025, +0.084] | -0.026 [-0.078, +0.025] | -0.005 [-0.061, +0.054] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.015 [-0.043, +0.077] | +0.008 [-0.053, +0.074] | -0.071 [-0.135, -0.001] | -0.025 [-0.089, +0.042] |
| faceverse_neutral | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | -0.052 [-0.123, +0.025] | -0.047 [-0.117, +0.029] | -0.087 [-0.175, +0.003] | -0.058 [-0.135, +0.023] |
| faceverse_neutral | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | -0.040 [-0.111, +0.027] | -0.018 [-0.089, +0.051] | -0.089 [-0.170, -0.007] | -0.031 [-0.103, +0.041] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.080 [+0.015, +0.149] | +0.078 [+0.011, +0.149] | - | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.049 [-0.031, +0.128] | +0.056 [-0.025, +0.139] | - | - |
| faceverse_neutral | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | -0.017 [-0.093, +0.062] | +0.001 [-0.075, +0.084] | - | - |
| faceverse_neutral | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | -0.006 [-0.092, +0.078] | +0.030 [-0.058, +0.112] | - | - |

### Scala dell'allineamento (crop): confronti secondari

GT FR:

| dominio | passo | crop | all_cross |
| --- | --- | --- | --- |
| faceverse_neutral | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.022 [-0.045, +0.084] | +0.090 [+0.014, +0.162] |
| faceverse_neutral | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.133 [-0.239, -0.029] | -0.088 [-0.205, +0.029] |

GT SR:

| dominio | passo | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- |
| faceverse_neutral | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | -0.024 [-0.084, +0.038] | +0.034 [-0.032, +0.101] | - | - |
| faceverse_neutral | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.201 [-0.299, -0.100] | -0.179 [-0.287, -0.071] | - | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.103 [+0.049, +0.156] | +0.049 [-0.012, +0.108] | -0.013 [-0.071, +0.045] | +0.007 [-0.054, +0.068] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.072 [-0.001, +0.146] | +0.027 [-0.047, +0.103] | -0.058 [-0.136, +0.024] | -0.014 [-0.090, +0.064] |
| faceverse_neutral | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.005 [-0.058, +0.079] | -0.028 [-0.100, +0.053] | -0.074 [-0.158, +0.016] | -0.046 [-0.123, +0.040] |
| faceverse_neutral | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.017 [-0.054, +0.085] | +0.001 [-0.075, +0.073] | -0.076 [-0.163, +0.008] | -0.019 [-0.098, +0.054] |

### Conteggio (crop)

Per ogni passo e GT, in crop e all_cross: numero di famiglie sintetiche (HIFI3D, FaceScape, FaceVerse con espressioni, FaceVerse neutra) con IC 95% appaiato che esclude lo 0 nel verso "piu' allineamento = peggio" / nel verso opposto / non risolte. Lato piu' allineato: b nei passi 1-3 (cs: taglia tolta per mesh; NICP: deformazione; NICP per coppia: deformazione specifica della coppia contro l'iscrizione su un template comune), a nell'1b (scala stimata per coppia), a nel 4 (fit del modello per mesh contro la rigida), b nel 5 (ICP: allineamento esplicito; il braccio non allinea). Le righe 4 e 5 contano ogni modello / seme a parte.

| GT | passo | ruolo | crop: peggio / meglio / n.r. | all_cross: peggio / meglio / n.r. |
| --- | --- | --- | --- | --- |
| FR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| FR | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| FR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 0 / 0 / 1 (su 1) | 1 / 0 / 0 (su 1) |
| FR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 0 / 0 / 1 (su 1) | 1 / 0 / 0 (su 1) |
| FR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 0 / 1 / 0 (su 1) | 0 / 0 / 1 (su 1) |
| FR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 1 / 0 (su 1) | 0 / 1 / 0 (su 1) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 0 / 1 / 0 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 0 / 1 / 0 (su 1) | 0 / 1 / 0 (su 1) |
| SR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 1 / 0 (su 1) | 0 / 1 / 0 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 1 / 0 (su 1) | 0 / 1 / 0 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | principale | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | secondario | 0 / 1 / 0 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | secondario | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | secondario | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | secondario | 0 / 0 / 1 (su 1) | 0 / 0 / 1 (su 1) |

### Fallimenti del NICP per coppia (crop)

| dominio | GT | insieme | metodo / passo | maschera comune | fallimento = distanza massima |
| --- | --- | --- | --- | --- | --- |

### Protocolli, landmark esatti e rilevati (crop)

### Varianti non dichiarate (crop)

GT FR:

| metodo | faceverse_neutral crop | faceverse_neutral all_cross |
| --- | --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), mm (FR) | 0.051 [0.011, 0.092] | 0.039 [0.010, 0.067] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.090 [0.047, 0.134] | 0.056 [0.027, 0.087] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.082 [0.046, 0.118] | 0.047 [0.028, 0.066] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.118 [0.070, 0.165] | 0.080 [0.048, 0.111] |
| DreamSim ensemble su render [profondita'] (non dichiarata), mm (FR) | 0.081 [0.025, 0.139] | 0.077 [0.026, 0.129] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.145 [0.100, 0.192] | 0.100 [0.066, 0.131] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | 0.085 [0.058, 0.111] | 0.113 [0.074, 0.150] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | 0.093 [0.066, 0.118] | 0.178 [0.127, 0.226] |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.180 [0.135, 0.222] | 0.144 [0.109, 0.175] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.193 [0.143, 0.238] | 0.161 [0.122, 0.197] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, mm (FR) | 0.097 [0.029, 0.159] | 0.078 [0.031, 0.125] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT FR (FR) | 0.080 [0.016, 0.143] | 0.053 [0.002, 0.107] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, mm (FR) | 0.036 [-0.014, 0.086] | 0.068 [0.032, 0.108] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT FR (FR) | 0.066 [-0.012, 0.145] | 0.081 [0.013, 0.144] |

GT SR:

| metodo | faceverse_neutral crop | faceverse_neutral all_cross |
| --- | --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.073 [0.032, 0.114] | 0.048 [0.020, 0.078] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.111 [0.070, 0.156] | 0.071 [0.041, 0.105] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.097 [0.063, 0.130] | 0.049 [0.032, 0.066] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.139 [0.093, 0.185] | 0.089 [0.058, 0.118] |
| DreamSim ensemble su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.088 [0.030, 0.147] | 0.087 [0.035, 0.139] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.168 [0.121, 0.212] | 0.113 [0.080, 0.145] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | 0.121 [0.085, 0.153] | 0.112 [0.077, 0.146] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | 0.212 [0.155, 0.267] | 0.219 [0.166, 0.269] |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.186 [0.142, 0.228] | 0.135 [0.104, 0.165] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.202 [0.155, 0.247] | 0.147 [0.112, 0.181] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, a CS_ref (SR) | 0.113 [0.044, 0.172] | 0.091 [0.045, 0.135] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT SR (d_P) (SR) | 0.094 [0.032, 0.155] | 0.065 [0.014, 0.117] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, a CS_ref (SR) | 0.048 [0.002, 0.097] | 0.075 [0.041, 0.111] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT SR (d_P) (SR) | 0.088 [0.013, 0.161] | 0.107 [0.044, 0.165] |

### FaceVerse neutra (crop): Spearman per insieme di righe (seme 271049)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | crop (49500) | all_cross (148500) | stessa_topologia crop (4950) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.282 [0.206, 0.360] | 0.319 [0.242, 0.395] | 0.335 [0.237, 0.431] | 0.349 [0.266, 0.427] |
| **factorized s1234, d_F non cal.** | 0.306 [0.221, 0.390] | 0.326 [0.249, 0.400] | 0.345 [0.247, 0.438] | 0.349 [0.269, 0.428] |
| factorized s1234, d_P | 0.293 [0.210, 0.374] | 0.310 [0.229, 0.384] | 0.336 [0.242, 0.428] | 0.335 [0.251, 0.412] |
| **factorized s2345, d_F cal.*** | 0.272 [0.207, 0.335] | 0.335 [0.267, 0.400] | 0.326 [0.232, 0.412] | 0.370 [0.298, 0.441] |
| **factorized s2345, d_F non cal.** | 0.310 [0.231, 0.388] | 0.352 [0.273, 0.427] | 0.340 [0.249, 0.429] | 0.374 [0.295, 0.450] |
| factorized s2345, d_P | 0.306 [0.226, 0.382] | 0.341 [0.259, 0.417] | 0.336 [0.244, 0.424] | 0.364 [0.277, 0.441] |
| **ctrlfr s1234** | 0.296 [0.225, 0.363] | 0.316 [0.242, 0.388] | 0.386 [0.304, 0.464] | 0.364 [0.285, 0.439] |
| **ctrlfr s2345** | 0.269 [0.192, 0.340] | 0.289 [0.207, 0.360] | 0.367 [0.276, 0.451] | 0.330 [0.242, 0.405] |
| **C3M e123, d_F cal.*** | 0.306 [0.232, 0.373] | 0.333 [0.256, 0.402] | 0.358 [0.269, 0.441] | 0.382 [0.298, 0.458] |
| **C3M e123, d_F non cal.** | 0.339 [0.256, 0.416] | 0.354 [0.270, 0.431] | 0.359 [0.269, 0.443] | 0.381 [0.294, 0.460] |
| C3M e123, d_P | 0.335 [0.253, 0.411] | 0.353 [0.270, 0.428] | 0.353 [0.261, 0.441] | 0.378 [0.292, 0.456] |
| **C3M e205, d_F cal.*** | 0.299 [0.229, 0.364] | 0.324 [0.255, 0.387] | 0.378 [0.288, 0.462] | 0.395 [0.312, 0.470] |
| **C3M e205, d_F non cal.** | 0.345 [0.262, 0.420] | 0.353 [0.269, 0.429] | 0.382 [0.286, 0.468] | 0.393 [0.309, 0.472] |
| C3M e205, d_P | 0.345 [0.258, 0.419] | 0.357 [0.272, 0.434] | 0.380 [0.280, 0.467] | 0.394 [0.304, 0.474] |
| **ICP + Chamfer in mm (per coppia)** | 0.317 [0.248, 0.379] | 0.373 [0.297, 0.442] | 0.444 [0.363, 0.515] | 0.419 [0.337, 0.492] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.333 [0.262, 0.399] | 0.348 [0.274, 0.417] | 0.411 [0.335, 0.484] | 0.382 [0.302, 0.455] |
| **NICP per coppia in mm** | 0.295 [0.214, 0.371] | 0.283 [0.205, 0.362] | - | - |
| **NICP per coppia cs** | 0.290 [0.209, 0.368] | 0.287 [0.202, 0.368] | - | - |
| **NICP su template in mm** | 0.162 [0.098, 0.225] | 0.195 [0.112, 0.278] | 0.253 [0.158, 0.349] | 0.228 [0.129, 0.325] |
| **NICP su template cs** | 0.110 [0.058, 0.158] | 0.147 [0.079, 0.214] | 0.223 [0.133, 0.307] | 0.186 [0.101, 0.267] |
| **varifold in mm (massa unitaria)** | 0.255 [0.203, 0.304] | 0.270 [0.214, 0.324] | 0.434 [0.357, 0.509] | 0.346 [0.281, 0.408] |
| **GNM (visto) preregistrato, coefficienti** | 0.239 [0.164, 0.306] | 0.240 [0.164, 0.306] | 0.245 [0.164, 0.321] | 0.240 [0.165, 0.308] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.234 [0.147, 0.316] | 0.232 [0.142, 0.317] | 0.234 [0.140, 0.324] | 0.232 [0.140, 0.318] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.244 [0.166, 0.320] | 0.241 [0.160, 0.319] | 0.259 [0.172, 0.337] | 0.244 [0.162, 0.323] |
| **GNM (visto) vA, coefficienti*** | 0.233 [0.157, 0.305] | 0.231 [0.153, 0.304] | 0.240 [0.160, 0.318] | 0.231 [0.154, 0.303] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.238 [0.148, 0.323] | 0.237 [0.145, 0.322] | 0.237 [0.139, 0.331] | 0.237 [0.143, 0.324] |
| GNM (visto) vA, mesh d'identita' SR* | 0.228 [0.145, 0.305] | 0.224 [0.137, 0.301] | 0.243 [0.157, 0.322] | 0.227 [0.141, 0.304] |
| **GNM (visto) vB, coefficienti*** | 0.292 [0.194, 0.376] | 0.297 [0.196, 0.380] | 0.289 [0.195, 0.378] | 0.295 [0.195, 0.378] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.323 [0.232, 0.400] | 0.321 [0.226, 0.400] | 0.324 [0.231, 0.404] | 0.322 [0.226, 0.402] |
| GNM (visto) vB, mesh d'identita' SR* | 0.379 [0.295, 0.453] | 0.375 [0.291, 0.450] | 0.386 [0.306, 0.462] | 0.375 [0.291, 0.451] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.385 [0.299, 0.462] | 0.383 [0.296, 0.459] | 0.388 [0.300, 0.469] | 0.383 [0.295, 0.461] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.173 [0.105, 0.242] | 0.172 [0.104, 0.242] | 0.175 [0.103, 0.250] | 0.174 [0.104, 0.246] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.214 [0.135, 0.289] | 0.214 [0.133, 0.289] | 0.215 [0.131, 0.298] | 0.215 [0.133, 0.293] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.170 [0.097, 0.244] | 0.169 [0.090, 0.243] | 0.174 [0.098, 0.251] | 0.172 [0.092, 0.246] |
| **FLAME 2023 Open vA, coefficienti*** | 0.181 [0.112, 0.248] | 0.179 [0.110, 0.247] | 0.183 [0.111, 0.257] | 0.181 [0.109, 0.251] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.219 [0.135, 0.297] | 0.220 [0.136, 0.298] | 0.219 [0.131, 0.305] | 0.220 [0.135, 0.300] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.177 [0.101, 0.250] | 0.175 [0.095, 0.249] | 0.180 [0.105, 0.260] | 0.178 [0.096, 0.252] |
| **FLAME 2023 Open vB, coefficienti*** | 0.223 [0.122, 0.317] | 0.240 [0.137, 0.331] | 0.205 [0.101, 0.306] | 0.239 [0.135, 0.330] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.306 [0.220, 0.384] | 0.312 [0.222, 0.391] | 0.299 [0.204, 0.385] | 0.313 [0.221, 0.393] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.348 [0.265, 0.429] | 0.355 [0.272, 0.434] | 0.341 [0.249, 0.426] | 0.356 [0.273, 0.435] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.357 [0.268, 0.433] | 0.363 [0.277, 0.440] | 0.349 [0.251, 0.434] | 0.364 [0.278, 0.442] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.181 [0.130, 0.233] | 0.162 [0.116, 0.208] | 0.220 [0.153, 0.284] | 0.178 [0.127, 0.230] |
| **ShapeDNA k=50** | 0.059 [0.012, 0.104] | 0.067 [0.030, 0.101] | 0.084 [-0.007, 0.177] | 0.142 [0.064, 0.217] |
| **HKS globale** | 0.038 [-0.004, 0.081] | 0.035 [0.006, 0.066] | 0.072 [-0.008, 0.159] | 0.073 [0.014, 0.133] |
| **WKS globale** | 0.056 [0.019, 0.089] | 0.049 [0.021, 0.074] | 0.087 [0.003, 0.169] | 0.076 [0.006, 0.139] |
| **taglia stimata dalla mesh (non oracolo)** | 0.022 [0.002, 0.049] | 0.089 [0.030, 0.149] | 0.116 [0.019, 0.218] | 0.143 [0.044, 0.241] |
| **taglia oracolo (S di FR)** | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.101 [0.056, 0.149] | 0.075 [0.043, 0.108] | 0.181 [0.100, 0.263] | 0.130 [0.070, 0.189] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.103 [0.058, 0.151] | 0.077 [0.045, 0.110] | 0.195 [0.109, 0.276] | 0.127 [0.071, 0.190] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.104 [0.055, 0.155] | 0.088 [0.047, 0.132] | 0.193 [0.106, 0.273] | 0.168 [0.093, 0.237] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.103 [0.057, 0.150] | 0.084 [0.044, 0.127] | 0.199 [0.118, 0.277] | 0.163 [0.090, 0.233] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.150 [0.102, 0.195] | 0.111 [0.077, 0.142] | 0.277 [0.196, 0.356] | 0.217 [0.142, 0.285] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.153 [0.104, 0.199] | 0.112 [0.079, 0.145] | 0.278 [0.193, 0.359] | 0.219 [0.145, 0.287] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.089 [0.062, 0.115] | 0.169 [0.117, 0.219] | - | - |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.196 [0.142, 0.248] | 0.200 [0.143, 0.248] | - | - |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.149 [0.091, 0.205] | 0.150 [0.100, 0.197] | 0.277 [0.200, 0.350] | 0.275 [0.191, 0.346] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.151 [0.096, 0.203] | 0.134 [0.091, 0.177] | 0.242 [0.168, 0.310] | 0.238 [0.165, 0.302] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.423 [0.343, 0.497] | 0.411 [0.331, 0.485] | - | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.400 [0.324, 0.474] | 0.389 [0.313, 0.463] | - | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.481 [0.394, 0.559] | 0.479 [0.393, 0.557] | - | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.307 [0.213, 0.405] | 0.307 [0.213, 0.403] | - | - |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.172 [0.133, 0.210] | 0.178 [0.139, 0.212] | 0.388 [0.296, 0.470] | 0.378 [0.298, 0.448] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.192 [0.147, 0.235] | 0.172 [0.132, 0.210] | 0.388 [0.294, 0.472] | 0.367 [0.288, 0.433] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.279 [0.221, 0.337] | 0.253 [0.197, 0.305] | 0.441 [0.350, 0.526] | 0.418 [0.336, 0.495] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.286 [0.226, 0.347] | 0.242 [0.185, 0.295] | 0.442 [0.351, 0.529] | 0.386 [0.299, 0.465] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | crop (49500) | all_cross (148500) | stessa_topologia crop (4950) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.255 [0.180, 0.330] | 0.288 [0.211, 0.364] | 0.301 [0.199, 0.400] | 0.315 [0.230, 0.395] |
| factorized s1234, d_F non cal. | 0.303 [0.223, 0.383] | 0.324 [0.244, 0.398] | 0.340 [0.243, 0.430] | 0.346 [0.265, 0.422] |
| **factorized s1234, d_P** | 0.304 [0.224, 0.382] | 0.323 [0.246, 0.393] | 0.347 [0.253, 0.435] | 0.347 [0.266, 0.420] |
| factorized s2345, d_F cal.* | 0.247 [0.175, 0.311] | 0.303 [0.226, 0.371] | 0.298 [0.200, 0.385] | 0.336 [0.251, 0.411] |
| factorized s2345, d_F non cal. | 0.307 [0.222, 0.385] | 0.347 [0.265, 0.421] | 0.337 [0.245, 0.423] | 0.369 [0.286, 0.444] |
| **factorized s2345, d_P** | 0.316 [0.236, 0.389] | 0.352 [0.273, 0.424] | 0.345 [0.257, 0.429] | 0.374 [0.293, 0.446] |
| ctrlfr s1234 | 0.291 [0.213, 0.359] | 0.314 [0.234, 0.388] | 0.376 [0.282, 0.458] | 0.359 [0.276, 0.436] |
| ctrlfr s2345 | 0.274 [0.191, 0.343] | 0.294 [0.215, 0.364] | 0.365 [0.265, 0.448] | 0.333 [0.249, 0.407] |
| C3M e123, d_F cal.* | 0.293 [0.214, 0.368] | 0.320 [0.237, 0.395] | 0.345 [0.257, 0.434] | 0.367 [0.275, 0.450] |
| C3M e123, d_F non cal. | 0.352 [0.267, 0.426] | 0.367 [0.279, 0.443] | 0.381 [0.292, 0.463] | 0.396 [0.303, 0.476] |
| **C3M e123, d_P** | 0.355 [0.273, 0.429] | 0.373 [0.288, 0.445] | 0.383 [0.295, 0.465] | 0.399 [0.312, 0.473] |
| C3M e205, d_F cal.* | 0.297 [0.225, 0.364] | 0.318 [0.246, 0.385] | 0.380 [0.289, 0.467] | 0.387 [0.303, 0.463] |
| C3M e205, d_F non cal. | 0.367 [0.286, 0.440] | 0.369 [0.289, 0.444] | 0.416 [0.331, 0.490] | 0.411 [0.328, 0.487] |
| **C3M e205, d_P** | 0.372 [0.294, 0.442] | 0.378 [0.304, 0.448] | 0.419 [0.338, 0.498] | 0.417 [0.339, 0.490] |
| **ICP + Chamfer in mm (per coppia)** | 0.299 [0.233, 0.363] | 0.351 [0.274, 0.420] | 0.421 [0.337, 0.493] | 0.393 [0.310, 0.467] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.356 [0.286, 0.420] | 0.370 [0.301, 0.435] | 0.434 [0.359, 0.502] | 0.405 [0.329, 0.472] |
| **NICP per coppia in mm** | 0.323 [0.245, 0.400] | 0.316 [0.237, 0.393] | - | - |
| **NICP per coppia cs** | 0.322 [0.243, 0.397] | 0.322 [0.242, 0.398] | - | - |
| **NICP su template in mm** | 0.122 [0.062, 0.181] | 0.137 [0.059, 0.212] | 0.190 [0.098, 0.288] | 0.161 [0.068, 0.254] |
| **NICP su template cs** | 0.110 [0.061, 0.161] | 0.148 [0.081, 0.215] | 0.213 [0.125, 0.298] | 0.185 [0.104, 0.267] |
| **varifold in mm (massa unitaria)** | 0.250 [0.197, 0.302] | 0.263 [0.204, 0.319] | 0.415 [0.333, 0.490] | 0.333 [0.267, 0.394] |
| **GNM (visto) preregistrato, coefficienti** | 0.204 [0.127, 0.276] | 0.206 [0.127, 0.275] | 0.209 [0.127, 0.287] | 0.207 [0.127, 0.277] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.176 [0.085, 0.263] | 0.175 [0.079, 0.264] | 0.173 [0.074, 0.271] | 0.174 [0.077, 0.265] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.230 [0.151, 0.305] | 0.228 [0.147, 0.305] | 0.246 [0.163, 0.326] | 0.233 [0.151, 0.311] |
| **GNM (visto) vA, coefficienti*** | 0.211 [0.132, 0.287] | 0.211 [0.131, 0.284] | 0.217 [0.135, 0.293] | 0.211 [0.131, 0.284] |
| GNM (visto) vA, mesh d'identita' FR* | 0.180 [0.087, 0.270] | 0.179 [0.080, 0.272] | 0.177 [0.078, 0.278] | 0.179 [0.079, 0.273] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.225 [0.147, 0.302] | 0.223 [0.141, 0.299] | 0.240 [0.161, 0.319] | 0.227 [0.144, 0.301] |
| **GNM (visto) vB, coefficienti*** | 0.296 [0.197, 0.383] | 0.301 [0.201, 0.386] | 0.294 [0.198, 0.386] | 0.299 [0.200, 0.384] |
| GNM (visto) vB, mesh d'identita' FR* | 0.276 [0.179, 0.363] | 0.275 [0.175, 0.364] | 0.274 [0.176, 0.365] | 0.275 [0.173, 0.366] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.402 [0.321, 0.473] | 0.400 [0.317, 0.473] | 0.407 [0.324, 0.480] | 0.400 [0.317, 0.474] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.360 [0.264, 0.439] | 0.358 [0.261, 0.439] | 0.361 [0.266, 0.444] | 0.358 [0.260, 0.440] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.152 [0.078, 0.222] | 0.151 [0.076, 0.222] | 0.153 [0.076, 0.232] | 0.153 [0.076, 0.227] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.172 [0.090, 0.251] | 0.172 [0.087, 0.253] | 0.171 [0.083, 0.259] | 0.172 [0.086, 0.255] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.164 [0.082, 0.239] | 0.163 [0.081, 0.239] | 0.168 [0.087, 0.248] | 0.166 [0.082, 0.245] |
| **FLAME 2023 Open vA, coefficienti*** | 0.157 [0.085, 0.228] | 0.156 [0.079, 0.229] | 0.159 [0.080, 0.236] | 0.158 [0.079, 0.233] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.173 [0.088, 0.256] | 0.173 [0.083, 0.258] | 0.170 [0.079, 0.261] | 0.174 [0.081, 0.259] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.173 [0.091, 0.250] | 0.172 [0.090, 0.248] | 0.178 [0.097, 0.259] | 0.175 [0.091, 0.253] |
| **FLAME 2023 Open vB, coefficienti*** | 0.238 [0.132, 0.335] | 0.258 [0.150, 0.350] | 0.219 [0.112, 0.323] | 0.256 [0.150, 0.347] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.265 [0.169, 0.354] | 0.271 [0.172, 0.360] | 0.255 [0.153, 0.351] | 0.271 [0.170, 0.363] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.371 [0.284, 0.452] | 0.378 [0.293, 0.457] | 0.363 [0.272, 0.449] | 0.380 [0.295, 0.460] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.333 [0.238, 0.420] | 0.341 [0.246, 0.426] | 0.323 [0.219, 0.416] | 0.342 [0.246, 0.427] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.196 [0.145, 0.249] | 0.175 [0.128, 0.225] | 0.237 [0.176, 0.304] | 0.193 [0.144, 0.245] |
| **ShapeDNA k=50** | 0.062 [0.020, 0.102] | 0.066 [0.027, 0.100] | 0.085 [-0.004, 0.174] | 0.137 [0.057, 0.211] |
| **HKS globale** | 0.037 [-0.007, 0.083] | 0.036 [0.005, 0.068] | 0.054 [-0.034, 0.150] | 0.074 [0.012, 0.137] |
| **WKS globale** | 0.059 [0.022, 0.092] | 0.048 [0.022, 0.074] | 0.090 [0.002, 0.168] | 0.068 [-0.001, 0.133] |
| **taglia stimata dalla mesh (non oracolo)** | 0.015 [0.001, 0.030] | 0.054 [0.003, 0.106] | 0.074 [-0.022, 0.171] | 0.090 [-0.007, 0.186] |
| **taglia oracolo (S di FR)** | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.119 [0.073, 0.166] | 0.086 [0.056, 0.119] | 0.212 [0.131, 0.295] | 0.156 [0.100, 0.214] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.118 [0.074, 0.165] | 0.088 [0.057, 0.123] | 0.219 [0.135, 0.296] | 0.153 [0.098, 0.215] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.119 [0.067, 0.168] | 0.101 [0.058, 0.143] | 0.220 [0.130, 0.299] | 0.195 [0.117, 0.262] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.120 [0.074, 0.165] | 0.099 [0.059, 0.140] | 0.227 [0.143, 0.302] | 0.190 [0.113, 0.260] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.168 [0.122, 0.213] | 0.123 [0.089, 0.154] | 0.311 [0.231, 0.385] | 0.248 [0.173, 0.313] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.172 [0.123, 0.217] | 0.124 [0.091, 0.156] | 0.315 [0.235, 0.390] | 0.252 [0.178, 0.318] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.083 [0.055, 0.110] | 0.147 [0.093, 0.197] | - | - |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.203 [0.149, 0.253] | 0.207 [0.151, 0.257] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | 0.133 [0.077, 0.187] | 0.134 [0.085, 0.183] | 0.244 [0.167, 0.317] | 0.247 [0.163, 0.324] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.149 [0.095, 0.201] | 0.134 [0.089, 0.180] | 0.238 [0.163, 0.309] | 0.239 [0.167, 0.304] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.397 [0.314, 0.474] | 0.386 [0.302, 0.465] | - | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.414 [0.338, 0.486] | 0.404 [0.329, 0.476] | - | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.523 [0.448, 0.591] | 0.521 [0.447, 0.589] | - | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.329 [0.244, 0.418] | 0.329 [0.245, 0.417] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.167 [0.126, 0.206] | 0.175 [0.135, 0.211] | 0.368 [0.274, 0.452] | 0.366 [0.287, 0.437] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.200 [0.155, 0.244] | 0.185 [0.149, 0.219] | 0.402 [0.309, 0.479] | 0.389 [0.317, 0.453] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.279 [0.219, 0.341] | 0.252 [0.198, 0.305] | 0.436 [0.348, 0.523] | 0.417 [0.332, 0.496] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.306 [0.245, 0.366] | 0.263 [0.208, 0.315] | 0.469 [0.384, 0.553] | 0.419 [0.340, 0.491] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.034 [-0.098, +0.035] | -0.054 [-0.119, +0.017] | -0.109 [-0.187, -0.023] | -0.071 [-0.142, +0.007] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.051 [-0.126, +0.031] | -0.029 [-0.103, +0.054] | -0.077 [-0.175, +0.031] | -0.033 [-0.115, +0.055] |
| NICP per coppia in mm | -0.012 [-0.092, +0.065] | +0.036 [-0.045, +0.119] | - | - |
| NICP per coppia cs | -0.008 [-0.089, +0.077] | +0.032 [-0.055, +0.120] | - | - |
| NICP su template in mm | +0.121 [+0.040, +0.200] | +0.124 [+0.036, +0.211] | +0.081 [-0.019, +0.185] | +0.120 [+0.019, +0.220] |
| NICP su template cs | +0.173 [+0.082, +0.263] | +0.172 [+0.080, +0.262] | +0.112 [-0.009, +0.244] | +0.163 [+0.059, +0.269] |
| varifold in mm (massa unitaria) | +0.028 [-0.033, +0.094] | +0.049 [-0.013, +0.117] | -0.099 [-0.177, -0.016] | +0.003 [-0.060, +0.074] |
| GNM (visto) preregistrato, coefficienti | +0.043 [-0.057, +0.150] | +0.079 [-0.024, +0.179] | +0.090 [-0.035, +0.225] | +0.108 [+0.003, +0.213] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.049 [-0.045, +0.144] | +0.087 [-0.011, +0.185] | +0.101 [-0.008, +0.215] | +0.116 [+0.014, +0.219] |
| GNM (visto) vA, coefficienti | +0.049 [-0.058, +0.148] | +0.088 [-0.014, +0.187] | +0.095 [-0.030, +0.214] | +0.117 [+0.013, +0.220] |
| GNM (visto) vA, mesh d'identita' FR | +0.045 [-0.049, +0.143] | +0.082 [-0.016, +0.184] | +0.097 [-0.013, +0.213] | +0.112 [+0.009, +0.215] |
| GNM (visto) vB, coefficienti | -0.009 [-0.101, +0.089] | +0.022 [-0.075, +0.128] | +0.046 [-0.062, +0.160] | +0.053 [-0.048, +0.162] |
| GNM (visto) vB, mesh d'identita' FR | -0.040 [-0.117, +0.046] | -0.002 [-0.089, +0.092] | +0.011 [-0.082, +0.112] | +0.027 [-0.062, +0.124] |
| GNM (visto) vB, composizione S_B + k d_P | -0.103 [-0.179, -0.022] | -0.064 [-0.145, +0.023] | -0.053 [-0.144, +0.046] | -0.035 [-0.115, +0.057] |
| FLAME 2023 Open preregistrato, coefficienti | +0.110 [+0.011, +0.212] | +0.148 [+0.051, +0.251] | +0.160 [+0.037, +0.283] | +0.175 [+0.073, +0.283] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.069 [-0.033, +0.167] | +0.105 [-0.002, +0.207] | +0.119 [+0.000, +0.240] | +0.133 [+0.020, +0.242] |
| FLAME 2023 Open vA, coefficienti | +0.102 [+0.005, +0.202] | +0.140 [+0.044, +0.240] | +0.152 [+0.032, +0.273] | +0.167 [+0.064, +0.276] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.064 [-0.032, +0.163] | +0.099 [-0.006, +0.199] | +0.116 [+0.001, +0.232] | +0.128 [+0.019, +0.233] |
| FLAME 2023 Open vB, coefficienti | +0.060 [-0.042, +0.164] | +0.079 [-0.022, +0.185] | +0.130 [+0.006, +0.259] | +0.110 [+0.006, +0.222] |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.024 [-0.111, +0.075] | +0.007 [-0.086, +0.110] | +0.035 [-0.079, +0.150] | +0.036 [-0.066, +0.144] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.074 [-0.165, +0.022] | -0.044 [-0.136, +0.054] | -0.014 [-0.126, +0.102] | -0.016 [-0.111, +0.088] |
| ArcFace-render, normal map 3 viste (G0) | +0.101 [+0.009, +0.188] | +0.157 [+0.068, +0.238] | +0.115 [-0.009, +0.224] | +0.171 [+0.075, +0.257] |
| ShapeDNA k=50 | +0.224 [+0.138, +0.318] | +0.252 [+0.169, +0.334] | +0.251 [+0.118, +0.395] | +0.206 [+0.102, +0.313] |
| HKS globale | +0.245 [+0.155, +0.335] | +0.284 [+0.201, +0.368] | +0.263 [+0.118, +0.396] | +0.276 [+0.165, +0.380] |
| WKS globale | +0.227 [+0.148, +0.311] | +0.271 [+0.194, +0.346] | +0.248 [+0.129, +0.371] | +0.273 [+0.180, +0.371] |
| taglia stimata dalla mesh (non oracolo) | +0.260 [+0.183, +0.342] | +0.230 [+0.150, +0.304] | +0.218 [+0.110, +0.321] | +0.205 [+0.103, +0.297] |
| taglia oracolo (S di FR) | +0.073 [-0.046, +0.197] | +0.110 [-0.015, +0.230] | +0.126 [-0.002, +0.254] | +0.139 [+0.015, +0.260] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.181 [+0.092, +0.269] | +0.244 [+0.159, +0.320] | +0.153 [+0.018, +0.289] | +0.219 [+0.121, +0.318] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.179 [+0.089, +0.264] | +0.231 [+0.151, +0.312] | +0.142 [+0.013, +0.266] | +0.180 [+0.078, +0.285] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.133 [+0.049, +0.215] | +0.208 [+0.123, +0.288] | +0.058 [-0.069, +0.185] | +0.132 [+0.029, +0.234] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.194 [+0.121, +0.266] | +0.150 [+0.082, +0.213] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.134 [+0.056, +0.206] | +0.169 [+0.098, +0.242] | +0.058 [-0.035, +0.156] | +0.074 [-0.015, +0.170] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.141 [-0.210, -0.062] | -0.092 [-0.163, -0.012] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.117 [-0.191, -0.041] | -0.070 [-0.142, +0.009] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.198 [-0.304, -0.092] | -0.160 [-0.259, -0.055] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.024 [-0.143, +0.091] | +0.012 [-0.109, +0.123] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.110 [+0.038, +0.182] | +0.141 [+0.073, +0.208] | -0.053 [-0.146, +0.037] | -0.030 [-0.103, +0.044] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.003 [-0.064, +0.070] | +0.066 [+0.002, +0.132] | -0.106 [-0.205, -0.013] | -0.070 [-0.150, +0.017] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.011 [-0.086, +0.066] | -0.047 [-0.122, +0.033] | -0.099 [-0.183, -0.008] | -0.070 [-0.148, +0.014] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.028 [-0.104, +0.050] | -0.022 [-0.097, +0.060] | -0.066 [-0.157, +0.028] | -0.033 [-0.116, +0.055] |
| NICP per coppia in mm | +0.011 [-0.063, +0.090] | +0.043 [-0.035, +0.120] | - | - |
| NICP per coppia cs | +0.016 [-0.064, +0.101] | +0.039 [-0.040, +0.122] | - | - |
| NICP su template in mm | +0.144 [+0.044, +0.241] | +0.131 [+0.024, +0.230] | +0.092 [-0.027, +0.211] | +0.121 [+0.006, +0.231] |
| NICP su template cs | +0.196 [+0.100, +0.293] | +0.179 [+0.081, +0.278] | +0.122 [+0.005, +0.251] | +0.164 [+0.055, +0.278] |
| varifold in mm (massa unitaria) | +0.051 [-0.014, +0.121] | +0.056 [-0.009, +0.123] | -0.089 [-0.165, -0.011] | +0.003 [-0.065, +0.073] |
| GNM (visto) preregistrato, coefficienti | +0.067 [-0.043, +0.176] | +0.086 [-0.019, +0.190] | +0.100 [-0.025, +0.229] | +0.109 [+0.004, +0.212] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.072 [-0.034, +0.176] | +0.094 [-0.009, +0.199] | +0.111 [-0.017, +0.227] | +0.117 [+0.012, +0.225] |
| GNM (visto) vA, coefficienti | +0.072 [-0.036, +0.176] | +0.095 [-0.014, +0.195] | +0.105 [-0.015, +0.225] | +0.118 [+0.008, +0.221] |
| GNM (visto) vA, mesh d'identita' FR | +0.068 [-0.039, +0.174] | +0.089 [-0.014, +0.196] | +0.108 [-0.018, +0.226] | +0.113 [+0.006, +0.222] |
| GNM (visto) vB, coefficienti | +0.014 [-0.079, +0.106] | +0.029 [-0.060, +0.123] | +0.056 [-0.046, +0.159] | +0.054 [-0.035, +0.149] |
| GNM (visto) vB, mesh d'identita' FR | -0.017 [-0.101, +0.073] | +0.005 [-0.083, +0.096] | +0.021 [-0.075, +0.118] | +0.028 [-0.063, +0.124] |
| GNM (visto) vB, composizione S_B + k d_P | -0.079 [-0.159, +0.002] | -0.057 [-0.138, +0.028] | -0.043 [-0.135, +0.046] | -0.034 [-0.118, +0.052] |
| FLAME 2023 Open preregistrato, coefficienti | +0.133 [+0.027, +0.233] | +0.154 [+0.052, +0.252] | +0.170 [+0.047, +0.288] | +0.176 [+0.071, +0.277] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.092 [-0.011, +0.196] | +0.112 [+0.009, +0.217] | +0.130 [+0.009, +0.248] | +0.134 [+0.026, +0.240] |
| FLAME 2023 Open vA, coefficienti | +0.125 [+0.022, +0.224] | +0.147 [+0.044, +0.244] | +0.162 [+0.045, +0.276] | +0.168 [+0.063, +0.270] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.087 [-0.014, +0.189] | +0.106 [+0.002, +0.209] | +0.126 [+0.006, +0.244] | +0.129 [+0.020, +0.237] |
| FLAME 2023 Open vB, coefficienti | +0.083 [-0.017, +0.181] | +0.086 [-0.011, +0.182] | +0.140 [+0.027, +0.253] | +0.111 [+0.013, +0.208] |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.000 [-0.092, +0.095] | +0.014 [-0.078, +0.110] | +0.046 [-0.058, +0.156] | +0.036 [-0.063, +0.136] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.051 [-0.139, +0.038] | -0.037 [-0.128, +0.053] | -0.004 [-0.106, +0.103] | -0.015 [-0.110, +0.078] |
| ArcFace-render, normal map 3 viste (G0) | +0.125 [+0.025, +0.218] | +0.164 [+0.074, +0.248] | +0.125 [+0.004, +0.237] | +0.171 [+0.079, +0.259] |
| ShapeDNA k=50 | +0.247 [+0.161, +0.338] | +0.259 [+0.179, +0.338] | +0.261 [+0.141, +0.395] | +0.207 [+0.106, +0.309] |
| HKS globale | +0.268 [+0.177, +0.363] | +0.291 [+0.209, +0.371] | +0.273 [+0.138, +0.411] | +0.277 [+0.174, +0.376] |
| WKS globale | +0.250 [+0.164, +0.336] | +0.277 [+0.197, +0.357] | +0.258 [+0.139, +0.380] | +0.273 [+0.175, +0.372] |
| taglia stimata dalla mesh (non oracolo) | +0.283 [+0.196, +0.375] | +0.237 [+0.140, +0.323] | +0.229 [+0.101, +0.350] | +0.206 [+0.088, +0.315] |
| taglia oracolo (S di FR) | +0.097 [-0.040, +0.230] | +0.117 [-0.024, +0.252] | +0.136 [-0.002, +0.270] | +0.140 [+0.000, +0.272] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.205 [+0.106, +0.297] | +0.251 [+0.163, +0.334] | +0.164 [+0.027, +0.299] | +0.220 [+0.115, +0.322] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.202 [+0.116, +0.293] | +0.238 [+0.159, +0.317] | +0.152 [+0.035, +0.274] | +0.181 [+0.090, +0.279] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.156 [+0.066, +0.242] | +0.215 [+0.130, +0.294] | +0.068 [-0.052, +0.186] | +0.133 [+0.032, +0.232] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.217 [+0.142, +0.296] | +0.157 [+0.084, +0.226] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.157 [+0.068, +0.238] | +0.176 [+0.095, +0.256] | +0.068 [-0.028, +0.164] | +0.075 [-0.017, +0.174] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.117 [-0.197, -0.037] | -0.085 [-0.163, -0.000] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.094 [-0.166, -0.021] | -0.063 [-0.135, +0.008] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.175 [-0.273, -0.074] | -0.153 [-0.245, -0.055] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.001 [-0.109, +0.106] | +0.019 [-0.094, +0.125] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.133 [+0.057, +0.211] | +0.148 [+0.080, +0.218] | -0.043 [-0.133, +0.058] | -0.029 [-0.102, +0.051] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.026 [-0.039, +0.094] | +0.073 [+0.012, +0.135] | -0.096 [-0.181, -0.008] | -0.069 [-0.140, +0.009] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.005 [-0.058, +0.079] | -0.028 [-0.100, +0.053] | -0.074 [-0.158, +0.016] | -0.046 [-0.123, +0.040] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.052 [-0.123, +0.025] | -0.047 [-0.117, +0.029] | -0.087 [-0.175, +0.003] | -0.058 [-0.135, +0.023] |
| NICP per coppia in mm | -0.018 [-0.094, +0.059] | +0.006 [-0.070, +0.087] | - | - |
| NICP per coppia cs | -0.017 [-0.093, +0.062] | +0.001 [-0.075, +0.084] | - | - |
| NICP su template in mm | +0.183 [+0.088, +0.267] | +0.186 [+0.085, +0.273] | +0.157 [+0.039, +0.273] | +0.186 [+0.071, +0.286] |
| NICP su template cs | +0.194 [+0.103, +0.281] | +0.175 [+0.074, +0.264] | +0.134 [+0.012, +0.261] | +0.162 [+0.050, +0.269] |
| varifold in mm (massa unitaria) | +0.054 [-0.007, +0.121] | +0.060 [+0.002, +0.123] | -0.069 [-0.145, +0.010] | +0.014 [-0.050, +0.080] |
| GNM (visto) preregistrato, coefficienti | +0.100 [-0.000, +0.201] | +0.117 [+0.022, +0.211] | +0.138 [+0.014, +0.263] | +0.140 [+0.044, +0.237] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.074 [-0.020, +0.173] | +0.095 [-0.001, +0.192] | +0.100 [-0.015, +0.216] | +0.114 [+0.016, +0.215] |
| GNM (visto) vA, coefficienti | +0.093 [-0.006, +0.194] | +0.112 [+0.012, +0.207] | +0.130 [+0.009, +0.246] | +0.136 [+0.036, +0.233] |
| GNM (visto) vA, mesh d'identita' SR | +0.079 [-0.015, +0.182] | +0.100 [+0.008, +0.201] | +0.107 [-0.007, +0.224] | +0.120 [+0.024, +0.226] |
| GNM (visto) vB, coefficienti | +0.008 [-0.083, +0.103] | +0.022 [-0.065, +0.118] | +0.052 [-0.055, +0.155] | +0.048 [-0.041, +0.146] |
| GNM (visto) vB, mesh d'identita' SR | -0.098 [-0.172, -0.020] | -0.077 [-0.152, +0.000] | -0.061 [-0.153, +0.023] | -0.053 [-0.129, +0.024] |
| FLAME 2023 Open preregistrato, coefficienti | +0.153 [+0.052, +0.254] | +0.172 [+0.068, +0.270] | +0.193 [+0.071, +0.306] | +0.194 [+0.088, +0.296] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.141 [+0.040, +0.253] | +0.160 [+0.060, +0.267] | +0.178 [+0.062, +0.303] | +0.181 [+0.079, +0.291] |
| FLAME 2023 Open vA, coefficienti | +0.147 [+0.048, +0.246] | +0.167 [+0.063, +0.261] | +0.188 [+0.067, +0.299] | +0.189 [+0.082, +0.288] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.131 [+0.032, +0.235] | +0.151 [+0.055, +0.255] | +0.168 [+0.053, +0.290] | +0.172 [+0.071, +0.278] |
| FLAME 2023 Open vB, coefficienti | +0.066 [-0.032, +0.168] | +0.065 [-0.030, +0.164] | +0.128 [+0.011, +0.240] | +0.091 [-0.005, +0.191] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.066 [-0.150, +0.025] | -0.055 [-0.137, +0.028] | -0.016 [-0.120, +0.086] | -0.033 [-0.115, +0.052] |
| ArcFace-render, normal map 3 viste (G0) | +0.108 [+0.009, +0.205] | +0.148 [+0.056, +0.236] | +0.110 [-0.010, +0.227] | +0.155 [+0.059, +0.245] |
| ShapeDNA k=50 | +0.242 [+0.153, +0.333] | +0.257 [+0.175, +0.334] | +0.261 [+0.134, +0.395] | +0.210 [+0.108, +0.311] |
| HKS globale | +0.268 [+0.175, +0.356] | +0.287 [+0.204, +0.364] | +0.292 [+0.149, +0.430] | +0.273 [+0.172, +0.375] |
| WKS globale | +0.245 [+0.161, +0.325] | +0.275 [+0.197, +0.348] | +0.257 [+0.139, +0.382] | +0.279 [+0.181, +0.377] |
| taglia stimata dalla mesh (non oracolo) | +0.289 [+0.205, +0.368] | +0.269 [+0.185, +0.343] | +0.273 [+0.144, +0.394] | +0.257 [+0.145, +0.358] |
| taglia oracolo (S di FR) | +0.286 [+0.167, +0.394] | +0.304 [+0.181, +0.414] | +0.328 [+0.209, +0.443] | +0.328 [+0.204, +0.440] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.186 [+0.092, +0.276] | +0.234 [+0.149, +0.312] | +0.127 [-0.002, +0.256] | +0.194 [+0.099, +0.289] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.184 [+0.098, +0.268] | +0.224 [+0.144, +0.303] | +0.119 [+0.005, +0.235] | +0.157 [+0.062, +0.252] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.133 [+0.044, +0.219] | +0.198 [+0.116, +0.277] | +0.031 [-0.095, +0.158] | +0.095 [-0.008, +0.195] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.101 [+0.027, +0.169] | +0.116 [+0.048, +0.184] | - | - |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.156 [+0.069, +0.239] | +0.189 [+0.111, +0.258] | +0.108 [+0.004, +0.205] | +0.108 [+0.020, +0.196] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.092 [-0.167, -0.009] | -0.064 [-0.144, +0.020] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.110 [-0.179, -0.038] | -0.081 [-0.150, -0.010] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.219 [-0.306, -0.132] | -0.198 [-0.282, -0.109] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.025 [-0.122, +0.073] | -0.006 [-0.108, +0.094] | - | - |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.104 [+0.034, +0.174] | +0.138 [+0.069, +0.203] | -0.055 [-0.144, +0.036] | -0.042 [-0.111, +0.034] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | -0.002 [-0.065, +0.062] | +0.060 [-0.002, +0.121] | -0.123 [-0.208, -0.038] | -0.072 [-0.146, +0.003] |

## FaMoS TEST (descrittivo)

**Non usare per scegliere il setting di Ava-256** ne' per nessuna selezione: 15 persone, 105 righe (blocco scan gallery -> scan), IC larghissimi; nessuna crop, nessuna perturbazione. Cautele: ArcFace-render (ritaglio tarato su pochi render, vedi le cautele sopra), NICP su template con un template costruito qui dalle persone TRAIN (post hoc).

| dominio | GT | insieme | righe | miglior nostro | rho | miglior concorrente | rho | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | FR | nocrop | 105 | ctrlfr s1234 | 0.831 [0.575, 0.949] | Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.787 [0.505, 0.920] | +0.044 [-0.084, +0.239] |
| famos | SR | nocrop | 105 | C3M e205, d_P | 0.756 [0.512, 0.889] | FLAME 2023 Open vB, mesh d'identita' SR* | 0.791 [0.626, 0.889] | -0.035 [-0.235, +0.116] |

Scala dell'allineamento, FaMoS (tutti i passi, principali e secondari):

GT FR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.138 [-0.125, +0.418] | - | - | - | - | - | - |
| famos | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.176 [-0.433, +0.055] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | -0.054 [-0.232, +0.095] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.071 [-0.235, +0.093] | - | - | - | - | - | - |
| famos | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.123 [-0.312, +0.101] | - | - | - | - | - | - |
| famos | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.127 [-0.309, +0.049] | - | - | - | - | - | - |

GT SR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.063 [-0.283, +0.161] | - | - | - | - | - | - |
| famos | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.122 [-0.086, +0.347] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.155 [-0.048, +0.452] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.179 [-0.026, +0.477] | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.129 [-0.091, +0.417] | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.117 [-0.084, +0.361] | - | - | - | - | - | - |

GT FR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.020 [-0.223, +0.265] | - | - | - | - | - |
| famos | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.032 [-0.319, +0.286] | - | - | - | - | - |

GT SR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | -0.231 [-0.464, -0.041] | - | - | - | - | - | - |
| famos | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.406 [-0.806, -0.087] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.218 [-0.105, +0.613] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.242 [-0.092, +0.631] | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.192 [-0.170, +0.587] | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.180 [-0.142, +0.547] | - | - | - | - | - | - |

### FaMoS TEST: Spearman per insieme di righe (seme 1234)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (105) |
| --- | --- |
| **factorized s1234, d_F cal.*** | 0.678 [0.237, 0.897] |
| **factorized s1234, d_F non cal.** | 0.616 [0.222, 0.824] |
| factorized s1234, d_P | 0.412 [0.088, 0.680] |
| **factorized s2345, d_F cal.*** | 0.654 [0.214, 0.896] |
| **factorized s2345, d_F non cal.** | 0.612 [0.191, 0.831] |
| factorized s2345, d_P | 0.437 [0.107, 0.698] |
| **ctrlfr s1234** | 0.831 [0.575, 0.949] |
| **ctrlfr s2345** | 0.818 [0.569, 0.944] |
| **C3M e123, d_F cal.*** | 0.688 [0.292, 0.902] |
| **C3M e123, d_F non cal.** | 0.568 [0.203, 0.788] |
| C3M e123, d_P | 0.427 [0.118, 0.679] |
| **C3M e205, d_F cal.*** | 0.739 [0.382, 0.911] |
| **C3M e205, d_F non cal.** | 0.577 [0.214, 0.804] |
| C3M e205, d_P | 0.486 [0.178, 0.734] |
| **ICP + Chamfer in mm (per coppia)** | 0.739 [0.362, 0.914] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.601 [0.254, 0.815] |
| **NICP per coppia in mm** | 0.719 [0.418, 0.874] |
| **NICP su template in mm*** | 0.688 [0.338, 0.891] |
| **NICP su template cs*** | 0.365 [0.029, 0.664] |
| **varifold in mm (massa unitaria)** | 0.688 [0.368, 0.902] |
| **GNM (visto) preregistrato, coefficienti** | 0.624 [0.195, 0.883] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.586 [0.145, 0.859] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.460 [0.139, 0.720] |
| **GNM (visto) vA, coefficienti*** | 0.619 [0.193, 0.878] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.593 [0.161, 0.870] |
| GNM (visto) vA, mesh d'identita' SR* | 0.234 [-0.089, 0.614] |
| **GNM (visto) vB, coefficienti*** | 0.606 [0.167, 0.836] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.686 [0.307, 0.910] |
| GNM (visto) vB, mesh d'identita' SR* | 0.513 [0.194, 0.749] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.753 [0.393, 0.938] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.457 [0.064, 0.760] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.565 [0.134, 0.856] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.217 [-0.108, 0.556] |
| **FLAME 2023 Open vA, coefficienti*** | 0.480 [0.065, 0.785] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.579 [0.153, 0.856] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.161 [-0.175, 0.522] |
| **FLAME 2023 Open vB, coefficienti*** | 0.666 [0.269, 0.878] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.668 [0.292, 0.908] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.481 [0.160, 0.714] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.708 [0.372, 0.923] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.450 [0.138, 0.667] |
| **ShapeDNA k=50** | -0.065 [-0.308, 0.247] |
| **HKS globale** | -0.061 [-0.331, 0.311] |
| **WKS globale** | -0.124 [-0.400, 0.239] |
| **taglia stimata dalla mesh (non oracolo)** | 0.715 [0.392, 0.885] |
| **taglia oracolo (S di FR)** | 0.787 [0.546, 0.900] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.447 [0.077, 0.704] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.427 [0.072, 0.690] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.247 [-0.096, 0.580] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.292 [-0.017, 0.634] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.499 [0.120, 0.769] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.477 [0.084, 0.753] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.787 [0.505, 0.920] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.305 [-0.034, 0.626] |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.599 [0.147, 0.874] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.159 [-0.141, 0.474] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.785 [0.488, 0.925] |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.609 [0.265, 0.811] |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.517 [0.161, 0.707] |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.502 [0.134, 0.757] |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.774 [0.532, 0.914] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.482 [0.132, 0.714] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.743 [0.440, 0.911] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.389 [0.086, 0.644] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (105) |
| --- | --- |
| factorized s1234, d_F cal.* | 0.662 [0.323, 0.858] |
| factorized s1234, d_F non cal. | 0.816 [0.593, 0.916] |
| **factorized s1234, d_P** | 0.740 [0.548, 0.883] |
| factorized s2345, d_F cal.* | 0.616 [0.225, 0.839] |
| factorized s2345, d_F non cal. | 0.774 [0.493, 0.898] |
| **factorized s2345, d_P** | 0.728 [0.516, 0.869] |
| ctrlfr s1234 | 0.709 [0.394, 0.871] |
| ctrlfr s2345 | 0.691 [0.378, 0.869] |
| C3M e123, d_F cal.* | 0.614 [0.236, 0.818] |
| C3M e123, d_F non cal. | 0.785 [0.520, 0.904] |
| **C3M e123, d_P** | 0.714 [0.484, 0.867] |
| C3M e205, d_F cal.* | 0.642 [0.284, 0.833] |
| C3M e205, d_F non cal. | 0.794 [0.505, 0.913] |
| **C3M e205, d_P** | 0.756 [0.512, 0.889] |
| **ICP + Chamfer in mm (per coppia)** | 0.548 [0.185, 0.799] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.612 [0.306, 0.787] |
| **NICP per coppia in mm** | 0.779 [0.525, 0.897] |
| **NICP su template in mm*** | 0.373 [-0.133, 0.731] |
| **NICP su template cs*** | 0.379 [0.087, 0.647] |
| **varifold in mm (massa unitaria)** | 0.582 [0.170, 0.842] |
| **GNM (visto) preregistrato, coefficienti** | 0.418 [-0.012, 0.772] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.274 [-0.208, 0.690] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.547 [0.188, 0.825] |
| **GNM (visto) vA, coefficienti*** | 0.438 [0.024, 0.780] |
| GNM (visto) vA, mesh d'identita' FR* | 0.279 [-0.201, 0.701] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.439 [0.075, 0.803] |
| **GNM (visto) vB, coefficienti*** | 0.745 [0.418, 0.882] |
| GNM (visto) vB, mesh d'identita' FR* | 0.439 [-0.001, 0.794] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.766 [0.592, 0.880] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.623 [0.285, 0.875] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.390 [0.027, 0.713] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.291 [-0.186, 0.691] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.412 [0.083, 0.748] |
| **FLAME 2023 Open vA, coefficienti*** | 0.383 [-0.006, 0.712] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.279 [-0.198, 0.682] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.373 [0.039, 0.741] |
| **FLAME 2023 Open vB, coefficienti*** | 0.714 [0.431, 0.877] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.483 [0.093, 0.819] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.791 [0.626, 0.889] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.661 [0.333, 0.887] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.435 [0.081, 0.661] |
| **ShapeDNA k=50** | -0.002 [-0.265, 0.304] |
| **HKS globale** | -0.015 [-0.333, 0.349] |
| **WKS globale** | -0.115 [-0.368, 0.189] |
| **taglia stimata dalla mesh (non oracolo)** | 0.333 [-0.178, 0.687] |
| **taglia oracolo (S di FR)** | 0.213 [-0.232, 0.573] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.657 [0.315, 0.835] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.608 [0.241, 0.809] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.333 [0.012, 0.626] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.407 [0.077, 0.667] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.633 [0.307, 0.810] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.639 [0.323, 0.812] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.504 [0.051, 0.780] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.469 [0.099, 0.729] |
| LPIPS alex su render [profondita'], mm (FR) | 0.381 [-0.118, 0.751] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.246 [-0.148, 0.573] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.581 [0.239, 0.820] |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.703 [0.429, 0.817] |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.776 [0.587, 0.887] |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.756 [0.541, 0.876] |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.605 [0.190, 0.827] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.560 [0.278, 0.760] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.663 [0.275, 0.835] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.609 [0.337, 0.786] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop |
| --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.062 [-0.223, +0.068] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.077 [-0.197, +0.382] |
| NICP per coppia in mm | -0.041 [-0.339, +0.211] |
| NICP su template in mm | -0.010 [-0.332, +0.216] |
| NICP su template cs | +0.313 [-0.195, +0.647] |
| varifold in mm (massa unitaria) | -0.010 [-0.263, +0.185] |
| GNM (visto) preregistrato, coefficienti | +0.054 [-0.237, +0.302] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.092 [-0.170, +0.382] |
| GNM (visto) vA, coefficienti | +0.059 [-0.201, +0.281] |
| GNM (visto) vA, mesh d'identita' FR | +0.085 [-0.184, +0.378] |
| GNM (visto) vB, coefficienti | +0.072 [-0.183, +0.403] |
| GNM (visto) vB, mesh d'identita' FR | -0.008 [-0.231, +0.199] |
| GNM (visto) vB, composizione S_B + k d_P | -0.075 [-0.252, +0.050] |
| FLAME 2023 Open preregistrato, coefficienti | +0.221 [-0.049, +0.461] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.113 [-0.183, +0.385] |
| FLAME 2023 Open vA, coefficienti | +0.198 [-0.045, +0.438] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.099 [-0.182, +0.378] |
| FLAME 2023 Open vB, coefficienti | +0.012 [-0.227, +0.268] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.010 [-0.213, +0.189] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.030 [-0.218, +0.099] |
| ArcFace-render, normal map 3 viste (G0) | +0.228 [-0.226, +0.612] |
| ShapeDNA k=50 | +0.743 [+0.188, +1.112] |
| HKS globale | +0.739 [+0.219, +1.089] |
| WKS globale | +0.802 [+0.203, +1.195] |
| taglia stimata dalla mesh (non oracolo) | -0.038 [-0.420, +0.266] |
| taglia oracolo (S di FR) | -0.109 [-0.565, +0.185] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.231 [-0.006, +0.495] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.431 [-0.024, +0.851] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.178 [-0.075, +0.446] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | -0.109 [-0.498, +0.129] |
| LPIPS alex su render [profondita'], mm (FR) | +0.079 [-0.230, +0.371] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.107 [-0.332, +0.008] |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.069 [-0.185, +0.340] |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.161 [-0.136, +0.433] |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.176 [-0.089, +0.446] |
| ULRSSM [faces_xyz] per coppia, mm (FR) | -0.096 [-0.389, +0.068] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | -0.065 [-0.423, +0.187] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop |
| --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.123 [-0.312, +0.101] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.015 [-0.233, +0.271] |
| NICP per coppia in mm | -0.103 [-0.329, +0.118] |
| NICP su template in mm | -0.071 [-0.358, +0.247] |
| NICP su template cs | +0.251 [-0.202, +0.549] |
| varifold in mm (massa unitaria) | -0.072 [-0.307, +0.138] |
| GNM (visto) preregistrato, coefficienti | -0.007 [-0.278, +0.306] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.030 [-0.251, +0.416] |
| GNM (visto) vA, coefficienti | -0.003 [-0.262, +0.290] |
| GNM (visto) vA, mesh d'identita' FR | +0.024 [-0.268, +0.402] |
| GNM (visto) vB, coefficienti | +0.010 [-0.224, +0.366] |
| GNM (visto) vB, mesh d'identita' FR | -0.069 [-0.311, +0.223] |
| GNM (visto) vB, composizione S_B + k d_P | -0.136 [-0.338, +0.087] |
| FLAME 2023 Open preregistrato, coefficienti | +0.159 [-0.099, +0.433] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.051 [-0.245, +0.409] |
| FLAME 2023 Open vA, coefficienti | +0.137 [-0.118, +0.421] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.037 [-0.250, +0.390] |
| FLAME 2023 Open vB, coefficienti | -0.049 [-0.268, +0.227] |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.052 [-0.304, +0.236] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.092 [-0.299, +0.109] |
| ArcFace-render, normal map 3 viste (G0) | +0.166 [-0.225, +0.521] |
| ShapeDNA k=50 | +0.681 [+0.175, +1.020] |
| HKS globale | +0.677 [+0.210, +1.017] |
| WKS globale | +0.740 [+0.150, +1.121] |
| taglia stimata dalla mesh (non oracolo) | -0.099 [-0.432, +0.267] |
| taglia oracolo (S di FR) | -0.171 [-0.591, +0.135] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.169 [-0.040, +0.401] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.369 [-0.048, +0.704] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.117 [-0.085, +0.343] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | -0.171 [-0.505, +0.075] |
| LPIPS alex su render [profondita'], mm (FR) | +0.018 [-0.300, +0.395] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.168 [-0.389, +0.030] |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.007 [-0.192, +0.239] |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.099 [-0.128, +0.316] |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.115 [-0.084, +0.334] |
| ULRSSM [faces_xyz] per coppia, mm (FR) | -0.158 [-0.416, +0.020] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | -0.127 [-0.405, +0.120] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop |
| --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.192 [-0.170, +0.587] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.129 [-0.091, +0.417] |
| NICP per coppia in mm | -0.039 [-0.284, +0.221] |
| NICP su template in mm | +0.368 [-0.125, +0.939] |
| NICP su template cs | +0.362 [+0.082, +0.664] |
| varifold in mm (massa unitaria) | +0.158 [-0.208, +0.574] |
| GNM (visto) preregistrato, coefficienti | +0.323 [-0.136, +0.797] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.193 [-0.141, +0.579] |
| GNM (visto) vA, coefficienti | +0.302 [-0.140, +0.769] |
| GNM (visto) vA, mesh d'identita' SR | +0.301 [-0.020, +0.681] |
| GNM (visto) vB, coefficienti | -0.005 [-0.257, +0.316] |
| GNM (visto) vB, mesh d'identita' SR | -0.026 [-0.188, +0.181] |
| FLAME 2023 Open preregistrato, coefficienti | +0.350 [-0.079, +0.749] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.328 [+0.006, +0.650] |
| FLAME 2023 Open vA, coefficienti | +0.357 [-0.079, +0.779] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.368 [+0.025, +0.710] |
| FLAME 2023 Open vB, coefficienti | +0.026 [-0.248, +0.336] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.050 [-0.182, +0.096] |
| ArcFace-render, normal map 3 viste (G0) | +0.305 [+0.008, +0.704] |
| ShapeDNA k=50 | +0.742 [+0.433, +1.034] |
| HKS globale | +0.756 [+0.378, +1.122] |
| WKS globale | +0.855 [+0.542, +1.148] |
| taglia stimata dalla mesh (non oracolo) | +0.407 [-0.091, +0.978] |
| taglia oracolo (S di FR) | +0.527 [+0.047, +1.025] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.132 [-0.091, +0.460] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.333 [+0.051, +0.709] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.102 [-0.103, +0.392] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.271 [+0.027, +0.623] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.494 [+0.157, +0.919] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | +0.159 [-0.197, +0.548] |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.037 [-0.153, +0.295] |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.035 [-0.243, +0.194] |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.016 [-0.169, +0.205] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.180 [-0.022, +0.491] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | +0.131 [-0.003, +0.390] |

Varianti non dichiarate, FaMoS (passo `varianti non dichiarate`):

GT FR:

| metodo | famos nocrop |
| --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), mm (FR) | 0.005 [-0.255, 0.344] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.474 [0.064, 0.739] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.329 [-0.102, 0.677] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.363 [-0.076, 0.681] |
| DreamSim ensemble su render [profondita'] (non dichiarata), mm (FR) | 0.233 [-0.096, 0.589] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.546 [0.171, 0.804] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | 0.693 [0.344, 0.879] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | 0.795 [0.529, 0.913] |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.676 [0.282, 0.890] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.658 [0.257, 0.892] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, mm (FR) | 0.106 [-0.238, 0.440] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT FR (FR) | 0.174 [-0.216, 0.552] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, mm (FR) | 0.277 [-0.065, 0.583] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT FR (FR) | 0.203 [-0.152, 0.531] |

GT SR:

| metodo | famos nocrop |
| --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), a CS_ref (SR) | -0.060 [-0.335, 0.283] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.653 [0.306, 0.831] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.469 [0.054, 0.726] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.428 [-0.038, 0.701] |
| DreamSim ensemble su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.222 [-0.129, 0.583] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.673 [0.378, 0.826] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | 0.337 [-0.018, 0.576] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | 0.521 [0.220, 0.775] |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.482 [0.072, 0.759] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.442 [0.030, 0.729] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, a CS_ref (SR) | 0.012 [-0.258, 0.312] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT SR (d_P) (SR) | -0.071 [-0.411, 0.338] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, a CS_ref (SR) | 0.284 [-0.043, 0.573] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT SR (d_P) (SR) | 0.023 [-0.292, 0.389] |

## Controlli

```
{
 "views": {
  "faceverse_neutral": {
   "load": {
    "varifold_check_rel_max_vs_shards": 1.5308867107322044e-07,
    "fits_nocrop_vs_saved_max_abs": {
     "gnm_orig_coef": 3.552713678800501e-15,
     "gnm_orig_fr": 1.4058012254292152e-09,
     "gnm_orig_sr": 1.8303679613640433e-09,
     "gnm_va_coef": 4.440892098500626e-15,
     "gnm_va_fr": 9.778534301729813e-10,
     "gnm_va_sr": 1.3419119287982895e-09,
     "gnm_vb_coef": 3.552713678800501e-15,
     "gnm_vb_fr": 4.745290549390546e-10,
     "gnm_vb_sr": 6.625389681325089e-10,
     "flame2023_orig_coef": 9.444112158973894e-13,
     "flame2023_orig_fr": 2.9046029470514156e-11,
     "flame2023_orig_sr": 2.3517687797180997e-11,
     "flame2023_va_coef": 1.283861905676531e-12,
     "flame2023_va_fr": 2.6711896583542227e-11,
     "flame2023_va_sr": 2.472604065939521e-11,
     "flame2023_vb_coef": 1.5698553568199713e-13,
     "flame2023_vb_fr": 1.8562179571190995e-11,
     "flame2023_vb_sr": 2.1976337416518277e-11
    }
   },
   "t1_control": {
    "n_compared": 66,
    "max_abs_diff_point_ci": 8.326672684688674e-17,
    "rows_t1_mask": 99000,
    "seed": 271049
   },
   "rows": 178200,
   "gt_spread_within_subject_pair": 0.0,
   "mask": {
    "rows_mask": 178200,
    "rows_mask_cross": 148500,
    "rows_mask_same": 29700,
    "rows_valid_gt": 178200,
    "columns_missing_on_same": [
     "mm_nicp_p2tri",
     "cs_nicp_p2tri"
    ],
    "rows_dropped_by_column_cross": {}
   },
   "plugins": {
    "kind": {
     "plugin:clip/depth|fr": "non dichiarata",
     "plugin:clip/depth|sr": "non dichiarata",
     "plugin:clip/flat|fr": "non dichiarata",
     "plugin:clip/flat|sr": "non dichiarata",
     "plugin:clip/smooth|fr": "dichiarata",
     "plugin:clip/smooth|sr": "dichiarata",
     "plugin:dinov2/depth|fr": "dichiarata",
     "plugin:dinov2/depth|sr": "dichiarata",
     "plugin:dinov2/flat|fr": "non dichiarata",
     "plugin:dinov2/flat|sr": "non dichiarata",
     "plugin:dinov2/smooth|fr": "non dichiarata",
     "plugin:dinov2/smooth|sr": "non dichiarata",
     "plugin:dreamsim/depth|fr": "non dichiarata",
     "plugin:dreamsim/depth|sr": "non dichiarata",
     "plugin:dreamsim/flat|fr": "non dichiarata",
     "plugin:dreamsim/flat|sr": "non dichiarata",
     "plugin:dreamsim/smooth|fr": "dichiarata",
     "plugin:dreamsim/smooth|sr": "dichiarata",
     "plugin:gw/euclid_128_cgw2|fr": "dichiarata",
     "plugin:gw/euclid_128_cgw2|sr": "dichiarata",
     "plugin:gw/euclid_256_cg|fr": "non dichiarata",
     "plugin:gw/euclid_256_cg|sr": "non dichiarata",
     "plugin:gw/euclid_256_cgw2|fr": "non dichiarata",
     "plugin:gw/euclid_256_cgw2|sr": "non dichiarata",
     "plugin:lpips/depth|fr": "dichiarata",
     "plugin:lpips/depth|sr": "dichiarata",
     "plugin:lpips/flat|fr": "non dichiarata",
     "plugin:lpips/flat|sr": "non dichiarata",
     "plugin:lpips/smooth|fr": "non dichiarata",
     "plugin:lpips/smooth|sr": "non dichiarata",
     "plugin:now_metrical|mm": "dichiarata",
     "plugin:now_original|mm": "dichiarata",
     "plugin:realy|mm": "dichiarata",
     "plugin:stirling|mm": "dichiarata",
     "plugin:ulrssm/faces_wks|pair_fr": "non dichiarata",
     "plugin:ulrssm/faces_wks|pair_sr": "non dichiarata",
     "plugin:ulrssm/faces_wks|tpl_fr": "non dichiarata",
     "plugin:ulrssm/faces_wks|tpl_sr": "non dichiarata",
     "plugin:ulrssm/faces_xyz|pair_fr": "dichiarata",
     "plugin:ulrssm/faces_xyz|pair_sr": "dichiarata",
     "plugin:ulrssm/faces_xyz|tpl_fr": "dichiarata",
     "plugin:ulrssm/faces_xyz|tpl_sr": "dichiarata",
     "plugin:ulrssm/zeroshot_faust|pair_fr": "non dichiarata",
     "plugin:ulrssm/zeroshot_faust|pair_sr": "non dichiarata",
     "plugin:ulrssm/zeroshot_faust|tpl_fr": "non dichiarata",
     "plugin:ulrssm/zeroshot_faust|tpl_sr": "non dichiarata"
    },
    "comune": [
     "plugin:clip/smooth|fr",
     "plugin:clip/smooth|sr",
     "plugin:dinov2/depth|fr",
     "plugin:dinov2/depth|sr",
     "plugin:dreamsim/smooth|fr",
     "plugin:dreamsim/smooth|sr",
     "plugin:gw/euclid_128_cgw2|fr",
     "plugin:gw/euclid_128_cgw2|sr",
     "plugin:lpips/depth|fr",
     "plugin:lpips/depth|sr",
     "plugin:now_metrical|mm",
     "plugin:now_original|mm",
     "plugin:realy|mm",
     "plugin:stirling|mm",
     "plugin:ulrssm/faces_xyz|pair_fr",
     "plugin:ulrssm/faces_xyz|pair_sr",
     "plugin:ulrssm/faces_xyz|tpl_fr",
     "plugin:ulrssm/faces_xyz|tpl_sr"
    ],
    "varianti non dichiarate": [
     "plugin:clip/depth|fr",
     "plugin:clip/depth|sr",
     "plugin:clip/flat|fr",
     "plugin:clip/flat|sr",
     "plugin:dinov2/flat|fr",
     "plugin:dinov2/flat|sr",
     "plugin:dinov2/smooth|fr",
     "plugin:dinov2/smooth|sr",
     "plugin:dreamsim/depth|fr",
     "plugin:dreamsim/depth|sr",
     "plugin:dreamsim/flat|fr",
     "plugin:dreamsim/flat|sr",
     "plugin:gw/euclid_256_cg|fr",
     "plugin:gw/euclid_256_cg|sr",
     "plugin:gw/euclid_256_cgw2|fr",
     "plugin:gw/euclid_256_cgw2|sr",
     "plugin:lpips/flat|fr",
     "plugin:lpips/flat|sr",
     "plugin:lpips/smooth|fr",
     "plugin:lpips/smooth|sr",
     "plugin:ulrssm/faces_wks|pair_fr",
     "plugin:ulrssm/faces_wks|pair_sr",
     "plugin:ulrssm/faces_wks|tpl_fr",
     "plugin:ulrssm/faces_wks|tpl_sr",
     "plugin:ulrssm/zeroshot_faust|pair_fr",
     "plugin:ulrssm/zeroshot_faust|pair_sr",
     "plugin:ulrssm/zeroshot_faust|tpl_fr",
     "plugin:ulrssm/zeroshot_faust|tpl_sr"
    ],
    "rows_lost_cross_in_comune": {
     "plugin:clip/smooth|fr": 0,
     "plugin:clip/smooth|sr": 0,
     "plugin:dinov2/depth|fr": 0,
     "plugin:dinov2/depth|sr": 0,
     "plugin:dreamsim/smooth|fr": 0,
     "plugin:dreamsim/smooth|sr": 0,
     "plugin:gw/euclid_128_cgw2|fr": 0,
     "plugin:gw/euclid_128_cgw2|sr": 0,
     "plugin:lpips/depth|fr": 0,
     "plugin:lpips/depth|sr": 0,
     "plugin:now_metrical|mm": 0,
     "plugin:now_original|mm": 0,
     "plugin:realy|mm": 0,
     "plugin:stirling|mm": 0,
     "plugin:ulrssm/faces_xyz|pair_fr": 0,
     "plugin:ulrssm/faces_xyz|pair_sr": 0,
     "plugin:ulrssm/faces_xyz|tpl_fr": 0,
     "plugin:ulrssm/faces_xyz|tpl_sr": 0,
     "plugin:clip/depth|fr": 0,
     "plugin:clip/depth|sr": 0,
     "plugin:clip/flat|fr": 0,
     "plugin:clip/flat|sr": 0,
     "plugin:dinov2/flat|fr": 0,
     "plugin:dinov2/flat|sr": 0,
     "plugin:dinov2/smooth|fr": 0,
     "plugin:dinov2/smooth|sr": 0,
     "plugin:dreamsim/depth|fr": 0,
     "plugin:dreamsim/depth|sr": 0,
     "plugin:dreamsim/flat|fr": 0,
     "plugin:dreamsim/flat|sr": 0,
     "plugin:gw/euclid_256_cg|fr": 0,
     "plugin:gw/euclid_256_cg|sr": 0,
     "plugin:gw/euclid_256_cgw2|fr": 0,
     "plugin:gw/euclid_256_cgw2|sr": 0,
     "plugin:lpips/flat|fr": 0,
     "plugin:lpips/flat|sr": 0,
     "plugin:lpips/smooth|fr": 0,
     "plugin:lpips/smooth|sr": 0,
     "plugin:ulrssm/faces_wks|pair_fr": 0,
     "plugin:ulrssm/faces_wks|pair_sr": 0,
     "plugin:ulrssm/faces_wks|tpl_fr": 0,
     "plugin:ulrssm/faces_wks|tpl_sr": 0,
     "plugin:ulrssm/zeroshot_faust|pair_fr": 0,
     "plugin:ulrssm/zeroshot_faust|pair_sr": 0,
     "plugin:ulrssm/zeroshot_faust|tpl_fr": 0,
     "plugin:ulrssm/zeroshot_faust|tpl_sr": 0
    },
    "missing_on_same": [
     "plugin:gw/euclid_128_cgw2|fr",
     "plugin:gw/euclid_128_cgw2|sr",
     "plugin:now_metrical|mm",
     "plugin:now_original|mm",
     "plugin:realy|mm",
     "plugin:stirling|mm",
     "plugin:gw/euclid_256_cg|fr",
     "plugin:gw/euclid_256_cg|sr",
     "plugin:gw/euclid_256_cgw2|fr",
     "plugin:gw/euclid_256_cgw2|sr"
    ],
    "righe_proprie": {}
   },
   "spearman_con_taglia": {
    "nocrop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.1227,
      "est_cs": 0.243
     },
     "factorized_s1234|form": {
      "oracle_size": 0.0187,
      "est_cs": 0.1411
     },
     "factorized_s1234|shape": {
      "oracle_size": -0.0089,
      "est_cs": 0.1166
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.1521,
      "est_cs": 0.2645
     },
     "factorized_s2345|form": {
      "oracle_size": 0.0566,
      "est_cs": 0.1551
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0279,
      "est_cs": 0.1268
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.0058,
      "est_cs": 0.1714
     },
     "ctrlfr_s2345|z": {
      "oracle_size": -0.0102,
      "est_cs": 0.2089
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.0649,
      "est_cs": 0.1953
     },
     "factorizedc3m_e123|form": {
      "oracle_size": -0.0043,
      "est_cs": 0.1158
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": -0.0078,
      "est_cs": 0.1117
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.0477,
      "est_cs": 0.2009
     },
     "factorizedc3m_e205|form": {
      "oracle_size": -0.0083,
      "est_cs": 0.1166
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": -0.0057,
      "est_cs": 0.1131
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.1058,
      "est_cs": 0.2938
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": -0.0176,
      "est_cs": 0.1097
     },
     "mm_nicp_p2tri": {
      "oracle_size": -0.0792,
      "est_cs": -0.0096
     },
     "cs_nicp_p2tri": {
      "oracle_size": -0.0658,
      "est_cs": 0.0644
     },
     "mm_nicp_template": {
      "oracle_size": 0.2187,
      "est_cs": 0.4718
     },
     "cs_nicp_template": {
      "oracle_size": 0.0352,
      "est_cs": 0.0848
     },
     "varifold": {
      "oracle_size": 0.0651,
      "est_cs": 0.2642
     },
     "gnm_orig_coef": {
      "oracle_size": 0.1388,
      "est_cs": 0.0897
     },
     "gnm_orig_fr": {
      "oracle_size": 0.1854,
      "est_cs": 0.2203
     },
     "gnm_orig_sr": {
      "oracle_size": 0.0826,
      "est_cs": 0.0627
     },
     "gnm_va_coef": {
      "oracle_size": 0.1015,
      "est_cs": 0.0726
     },
     "gnm_va_fr": {
      "oracle_size": 0.1881,
      "est_cs": 0.2212
     },
     "gnm_va_sr": {
      "oracle_size": 0.0537,
      "est_cs": 0.032
     },
     "gnm_vb_coef": {
      "oracle_size": 0.0129,
      "est_cs": 0.075
     },
     "gnm_vb_fr": {
      "oracle_size": 0.1566,
      "est_cs": 0.2139
     },
     "gnm_vb_sr": {
      "oracle_size": -0.0069,
      "est_cs": 0.0634
     },
     "gnm_vb_comp": {
      "oracle_size": 0.1074,
      "est_cs": 0.1755
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.0985,
      "est_cs": 0.0455
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.146,
      "est_cs": 0.1406
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0689,
      "est_cs": 0.017
     },
     "flame2023_va_coef": {
      "oracle_size": 0.1052,
      "est_cs": 0.0579
     },
     "flame2023_va_fr": {
      "oracle_size": 0.1538,
      "est_cs": 0.1668
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0644,
      "est_cs": 0.0233
     },
     "flame2023_vb_coef": {
      "oracle_size": -0.0177,
      "est_cs": 0.0636
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.1401,
      "est_cs": 0.1742
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0033,
      "est_cs": 0.0617
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1016,
      "est_cs": 0.1468
     },
     "arcface_normals": {
      "oracle_size": -0.0185,
      "est_cs": 0.0163
     },
     "shapedna_k50": {
      "oracle_size": 0.016,
      "est_cs": 0.0259
     },
     "hks": {
      "oracle_size": 0.0131,
      "est_cs": 0.0014
     },
     "wks": {
      "oracle_size": 0.0194,
      "est_cs": 0.0092
     },
     "est_cs": {
      "oracle_size": 0.1618,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.1618
     }
    },
    "crop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.1053,
      "est_cs": 0.28
     },
     "factorized_s1234|form": {
      "oracle_size": 0.0246,
      "est_cs": 0.1002
     },
     "factorized_s1234|shape": {
      "oracle_size": -0.0023,
      "est_cs": 0.0429
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.1192,
      "est_cs": 0.3635
     },
     "factorized_s2345|form": {
      "oracle_size": 0.0579,
      "est_cs": 0.1627
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0334,
      "est_cs": 0.0948
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.0317,
      "est_cs": 0.2704
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.0079,
      "est_cs": 0.2697
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.0626,
      "est_cs": 0.2448
     },
     "factorizedc3m_e123|form": {
      "oracle_size": -0.0008,
      "est_cs": 0.0643
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": -0.0038,
      "est_cs": 0.048
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.0361,
      "est_cs": 0.2802
     },
     "factorizedc3m_e205|form": {
      "oracle_size": -0.0142,
      "est_cs": 0.0831
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": -0.011,
      "est_cs": 0.067
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.0869,
      "est_cs": 0.2376
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": -0.0098,
      "est_cs": 0.1043
     },
     "mm_nicp_p2tri": {
      "oracle_size": -0.0542,
      "est_cs": 0.0425
     },
     "cs_nicp_p2tri": {
      "oracle_size": -0.0464,
      "est_cs": 0.1231
     },
     "mm_nicp_template": {
      "oracle_size": 0.1284,
      "est_cs": 0.6131
     },
     "cs_nicp_template": {
      "oracle_size": 0.0157,
      "est_cs": 0.0261
     },
     "varifold": {
      "oracle_size": 0.0591,
      "est_cs": 0.5614
     },
     "gnm_orig_coef": {
      "oracle_size": 0.1458,
      "est_cs": 0.0162
     },
     "gnm_orig_fr": {
      "oracle_size": 0.1866,
      "est_cs": 0.077
     },
     "gnm_orig_sr": {
      "oracle_size": 0.0919,
      "est_cs": -0.0338
     },
     "gnm_va_coef": {
      "oracle_size": 0.109,
      "est_cs": 0.0045
     },
     "gnm_va_fr": {
      "oracle_size": 0.1907,
      "est_cs": 0.0733
     },
     "gnm_va_sr": {
      "oracle_size": 0.0668,
      "est_cs": -0.0603
     },
     "gnm_vb_coef": {
      "oracle_size": 0.013,
      "est_cs": -0.007
     },
     "gnm_vb_fr": {
      "oracle_size": 0.1626,
      "est_cs": 0.0772
     },
     "gnm_vb_sr": {
      "oracle_size": 0.0047,
      "est_cs": -0.0135
     },
     "gnm_vb_comp": {
      "oracle_size": 0.115,
      "est_cs": 0.0491
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.1035,
      "est_cs": 0.0086
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.1495,
      "est_cs": 0.0437
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0745,
      "est_cs": -0.038
     },
     "flame2023_va_coef": {
      "oracle_size": 0.1098,
      "est_cs": 0.0148
     },
     "flame2023_va_fr": {
      "oracle_size": 0.1564,
      "est_cs": 0.0543
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0688,
      "est_cs": -0.0394
     },
     "flame2023_vb_coef": {
      "oracle_size": -0.0082,
      "est_cs": -0.0087
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.1453,
      "est_cs": 0.0547
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0104,
      "est_cs": -0.0316
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1083,
      "est_cs": 0.0277
     },
     "arcface_normals": {
      "oracle_size": -0.0201,
      "est_cs": 0.0493
     },
     "shapedna_k50": {
      "oracle_size": 0.01,
      "est_cs": 0.1548
     },
     "hks": {
      "oracle_size": 0.0243,
      "est_cs": 0.0944
     },
     "wks": {
      "oracle_size": 0.0007,
      "est_cs": 0.1297
     },
     "est_cs": {
      "oracle_size": 0.0155,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.0155
     }
    },
    "all_cross": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.1157,
      "est_cs": 0.2601
     },
     "factorized_s1234|form": {
      "oracle_size": 0.0205,
      "est_cs": 0.1308
     },
     "factorized_s1234|shape": {
      "oracle_size": -0.0068,
      "est_cs": 0.1089
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.1395,
      "est_cs": 0.3063
     },
     "factorized_s2345|form": {
      "oracle_size": 0.0567,
      "est_cs": 0.1635
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0294,
      "est_cs": 0.1363
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.0142,
      "est_cs": 0.2017
     },
     "ctrlfr_s2345|z": {
      "oracle_size": -0.0043,
      "est_cs": 0.2228
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.064,
      "est_cs": 0.1955
     },
     "factorizedc3m_e123|form": {
      "oracle_size": -0.0031,
      "est_cs": 0.0859
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": -0.0063,
      "est_cs": 0.0941
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.0438,
      "est_cs": 0.2041
     },
     "factorizedc3m_e205|form": {
      "oracle_size": -0.0103,
      "est_cs": 0.0822
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": -0.0074,
      "est_cs": 0.0925
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.098,
      "est_cs": 0.2669
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": -0.0142,
      "est_cs": 0.2009
     },
     "mm_nicp_p2tri": {
      "oracle_size": -0.0706,
      "est_cs": -0.0225
     },
     "cs_nicp_p2tri": {
      "oracle_size": -0.059,
      "est_cs": 0.0995
     },
     "mm_nicp_template": {
      "oracle_size": 0.1842,
      "est_cs": 0.5565
     },
     "cs_nicp_template": {
      "oracle_size": 0.0268,
      "est_cs": 0.1552
     },
     "varifold": {
      "oracle_size": 0.0622,
      "est_cs": 0.3827
     },
     "gnm_orig_coef": {
      "oracle_size": 0.141,
      "est_cs": 0.0713
     },
     "gnm_orig_fr": {
      "oracle_size": 0.1857,
      "est_cs": 0.1579
     },
     "gnm_orig_sr": {
      "oracle_size": 0.0855,
      "est_cs": 0.0437
     },
     "gnm_va_coef": {
      "oracle_size": 0.1039,
      "est_cs": 0.0534
     },
     "gnm_va_fr": {
      "oracle_size": 0.1889,
      "est_cs": 0.158
     },
     "gnm_va_sr": {
      "oracle_size": 0.058,
      "est_cs": 0.0168
     },
     "gnm_vb_coef": {
      "oracle_size": 0.0129,
      "est_cs": 0.0541
     },
     "gnm_vb_fr": {
      "oracle_size": 0.1585,
      "est_cs": 0.1513
     },
     "gnm_vb_sr": {
      "oracle_size": -0.003,
      "est_cs": 0.0427
     },
     "gnm_vb_comp": {
      "oracle_size": 0.1099,
      "est_cs": 0.1218
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.1002,
      "est_cs": 0.0382
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.1471,
      "est_cs": 0.1033
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0708,
      "est_cs": 0.009
     },
     "flame2023_va_coef": {
      "oracle_size": 0.1067,
      "est_cs": 0.047
     },
     "flame2023_va_fr": {
      "oracle_size": 0.1546,
      "est_cs": 0.121
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0659,
      "est_cs": 0.012
     },
     "flame2023_vb_coef": {
      "oracle_size": -0.0145,
      "est_cs": 0.051
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.1417,
      "est_cs": 0.1249
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0056,
      "est_cs": 0.0369
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1038,
      "est_cs": 0.101
     },
     "arcface_normals": {
      "oracle_size": -0.0189,
      "est_cs": -0.0143
     },
     "shapedna_k50": {
      "oracle_size": 0.0145,
      "est_cs": 0.0325
     },
     "hks": {
      "oracle_size": 0.0168,
      "est_cs": -0.0073
     },
     "wks": {
      "oracle_size": 0.0139,
      "est_cs": 0.0308
     },
     "est_cs": {
      "oracle_size": 0.0972,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.0972
     }
    }
   },
   "nicp_failures": {
    "rows_lost": 0,
    "by_topology": {
     "original": {
      "lost": 0,
      "rows_without_nicp_mask": 49500
     },
     "remesh": {
      "lost": 0,
      "rows_without_nicp_mask": 49500
     },
     "down8k": {
      "lost": 0,
      "rows_without_nicp_mask": 49500
     },
     "up60k": {
      "lost": 0,
      "rows_without_nicp_mask": 49500
     },
     "noisy": {
      "lost": 0,
      "rows_without_nicp_mask": 49500
     },
     "crop": {
      "lost": 0,
      "rows_without_nicp_mask": 49500
     }
    }
   },
   "seed": 271049,
   "n_methods": 65,
   "plugins_full": [
    "plugin:clip/smooth|fr",
    "plugin:clip/smooth|sr",
    "plugin:dinov2/depth|fr",
    "plugin:dinov2/depth|sr",
    "plugin:dreamsim/smooth|fr",
    "plugin:dreamsim/smooth|sr",
    "plugin:gw/euclid_128_cgw2|fr",
    "plugin:gw/euclid_128_cgw2|sr",
    "plugin:lpips/depth|fr",
    "plugin:lpips/depth|sr",
    "plugin:now_metrical|mm",
    "plugin:now_original|mm",
    "plugin:realy|mm",
    "plugin:stirling|mm",
    "plugin:ulrssm/faces_xyz|pair_fr",
    "plugin:ulrssm/faces_xyz|pair_sr",
    "plugin:ulrssm/faces_xyz|tpl_fr",
    "plugin:ulrssm/faces_xyz|tpl_sr"
   ],
   "plugins_partial": [],
   "seconds": 1067.8202962875366
  },
  "famos": {
   "famos_controls": [
    "factorized_s1234|form: 0.0e+00",
    "factorized_s2345|form: 0.0e+00",
    "factorized2_s1234|form: 0.0e+00",
    "factorized2_s2345|form: 0.0e+00",
    "ctrlfr_s1234|z: 0.0e+00",
    "ctrlfr_s2345|z: 0.0e+00",
    "dual_s1234|zf: 0.0e+00",
    "dual_s2345|zf: 0.0e+00",
    "factorizedc3m_e123|form: 0.0e+00",
    "factorizedc3m_e205|form: 0.0e+00"
   ],
   "famos_arcface_detector": {
    "-030.00": 11,
    "+000.00": 11,
    "+030.00": 5
   },
   "famos_cs_nicp_vs_famos_frame": NaN,
   "mask": {
    "rows_mask": 105,
    "rows_mask_cross": 105,
    "rows_mask_same": 0,
    "rows_valid_gt": 105,
    "columns_missing_on_same": [],
    "rows_dropped_by_column_cross": {}
   },
   "plugins": {
    "kind": {
     "plugin:clip/depth|fr": "non dichiarata",
     "plugin:clip/depth|sr": "non dichiarata",
     "plugin:clip/flat|fr": "non dichiarata",
     "plugin:clip/flat|sr": "non dichiarata",
     "plugin:clip/smooth|fr": "dichiarata",
     "plugin:clip/smooth|sr": "dichiarata",
     "plugin:dinov2/depth|fr": "dichiarata",
     "plugin:dinov2/depth|sr": "dichiarata",
     "plugin:dinov2/flat|fr": "non dichiarata",
     "plugin:dinov2/flat|sr": "non dichiarata",
     "plugin:dinov2/smooth|fr": "non dichiarata",
     "plugin:dinov2/smooth|sr": "non dichiarata",
     "plugin:dreamsim/depth|fr": "non dichiarata",
     "plugin:dreamsim/depth|sr": "non dichiarata",
     "plugin:dreamsim/flat|fr": "non dichiarata",
     "plugin:dreamsim/flat|sr": "non dichiarata",
     "plugin:dreamsim/smooth|fr": "dichiarata",
     "plugin:dreamsim/smooth|sr": "dichiarata",
     "plugin:gw/euclid_128_cgw2|fr": "dichiarata",
     "plugin:gw/euclid_128_cgw2|sr": "dichiarata",
     "plugin:gw/euclid_256_cg|fr": "non dichiarata",
     "plugin:gw/euclid_256_cg|sr": "non dichiarata",
     "plugin:gw/euclid_256_cgw2|fr": "non dichiarata",
     "plugin:gw/euclid_256_cgw2|sr": "non dichiarata",
     "plugin:lpips/depth|fr": "dichiarata",
     "plugin:lpips/depth|sr": "dichiarata",
     "plugin:lpips/flat|fr": "non dichiarata",
     "plugin:lpips/flat|sr": "non dichiarata",
     "plugin:lpips/smooth|fr": "non dichiarata",
     "plugin:lpips/smooth|sr": "non dichiarata",
     "plugin:now_metrical|mm": "dichiarata",
     "plugin:now_original|mm": "dichiarata",
     "plugin:realy|mm": "dichiarata",
     "plugin:stirling|mm": "dichiarata",
     "plugin:ulrssm/faces_wks|pair_fr": "non dichiarata",
     "plugin:ulrssm/faces_wks|pair_sr": "non dichiarata",
     "plugin:ulrssm/faces_wks|tpl_fr": "non dichiarata",
     "plugin:ulrssm/faces_wks|tpl_sr": "non dichiarata",
     "plugin:ulrssm/faces_xyz|pair_fr": "dichiarata",
     "plugin:ulrssm/faces_xyz|pair_sr": "dichiarata",
     "plugin:ulrssm/faces_xyz|tpl_fr": "dichiarata",
     "plugin:ulrssm/faces_xyz|tpl_sr": "dichiarata",
     "plugin:ulrssm/zeroshot_faust|pair_fr": "non dichiarata",
     "plugin:ulrssm/zeroshot_faust|pair_sr": "non dichiarata",
     "plugin:ulrssm/zeroshot_faust|tpl_fr": "non dichiarata",
     "plugin:ulrssm/zeroshot_faust|tpl_sr": "non dichiarata"
    },
    "comune": [
     "plugin:clip/smooth|fr",
     "plugin:clip/smooth|sr",
     "plugin:dinov2/depth|fr",
     "plugin:dinov2/depth|sr",
     "plugin:dreamsim/smooth|fr",
     "plugin:dreamsim/smooth|sr",
     "plugin:gw/euclid_128_cgw2|fr",
     "plugin:gw/euclid_128_cgw2|sr",
     "plugin:lpips/depth|fr",
     "plugin:lpips/depth|sr",
     "plugin:now_metrical|mm",
     "plugin:now_original|mm",
     "plugin:realy|mm",
     "plugin:stirling|mm",
     "plugin:ulrssm/faces_xyz|pair_fr",
     "plugin:ulrssm/faces_xyz|pair_sr",
     "plugin:ulrssm/faces_xyz|tpl_fr",
     "plugin:ulrssm/faces_xyz|tpl_sr"
    ],
    "varianti non dichiarate": [
     "plugin:clip/depth|fr",
     "plugin:clip/depth|sr",
     "plugin:clip/flat|fr",
     "plugin:clip/flat|sr",
     "plugin:dinov2/flat|fr",
     "plugin:dinov2/flat|sr",
     "plugin:dinov2/smooth|fr",
     "plugin:dinov2/smooth|sr",
     "plugin:dreamsim/depth|fr",
     "plugin:dreamsim/depth|sr",
     "plugin:dreamsim/flat|fr",
     "plugin:dreamsim/flat|sr",
     "plugin:gw/euclid_256_cg|fr",
     "plugin:gw/euclid_256_cg|sr",
     "plugin:gw/euclid_256_cgw2|fr",
     "plugin:gw/euclid_256_cgw2|sr",
     "plugin:lpips/flat|fr",
     "plugin:lpips/flat|sr",
     "plugin:lpips/smooth|fr",
     "plugin:lpips/smooth|sr",
     "plugin:ulrssm/faces_wks|pair_fr",
     "plugin:ulrssm/faces_wks|pair_sr",
     "plugin:ulrssm/faces_wks|tpl_fr",
     "plugin:ulrssm/faces_wks|tpl_sr",
     "plugin:ulrssm/zeroshot_faust|pair_fr",
     "plugin:ulrssm/zeroshot_faust|pair_sr",
     "plugin:ulrssm/zeroshot_faust|tpl_fr",
     "plugin:ulrssm/zeroshot_faust|tpl_sr"
    ],
    "rows_lost_cross_in_comune": {
     "plugin:clip/smooth|fr": 0,
     "plugin:clip/smooth|sr": 0,
     "plugin:dinov2/depth|fr": 0,
     "plugin:dinov2/depth|sr": 0,
     "plugin:dreamsim/smooth|fr": 0,
     "plugin:dreamsim/smooth|sr": 0,
     "plugin:gw/euclid_128_cgw2|fr": 0,
     "plugin:gw/euclid_128_cgw2|sr": 0,
     "plugin:lpips/depth|fr": 0,
     "plugin:lpips/depth|sr": 0,
     "plugin:now_metrical|mm": 0,
     "plugin:now_original|mm": 0,
     "plugin:realy|mm": 0,
     "plugin:stirling|mm": 0,
     "plugin:ulrssm/faces_xyz|pair_fr": 0,
     "plugin:ulrssm/faces_xyz|pair_sr": 0,
     "plugin:ulrssm/faces_xyz|tpl_fr": 0,
     "plugin:ulrssm/faces_xyz|tpl_sr": 0,
     "plugin:clip/depth|fr": 0,
     "plugin:clip/depth|sr": 0,
     "plugin:clip/flat|fr": 0,
     "plugin:clip/flat|sr": 0,
     "plugin:dinov2/flat|fr": 0,
     "plugin:dinov2/flat|sr": 0,
     "plugin:dinov2/smooth|fr": 0,
     "plugin:dinov2/smooth|sr": 0,
     "plugin:dreamsim/depth|fr": 0,
     "plugin:dreamsim/depth|sr": 0,
     "plugin:dreamsim/flat|fr": 0,
     "plugin:dreamsim/flat|sr": 0,
     "plugin:gw/euclid_256_cg|fr": 0,
     "plugin:gw/euclid_256_cg|sr": 0,
     "plugin:gw/euclid_256_cgw2|fr": 0,
     "plugin:gw/euclid_256_cgw2|sr": 0,
     "plugin:lpips/flat|fr": 0,
     "plugin:lpips/flat|sr": 0,
     "plugin:lpips/smooth|fr": 0,
     "plugin:lpips/smooth|sr": 0,
     "plugin:ulrssm/faces_wks|pair_fr": 0,
     "plugin:ulrssm/faces_wks|pair_sr": 0,
     "plugin:ulrssm/faces_wks|tpl_fr": 0,
     "plugin:ulrssm/faces_wks|tpl_sr": 0,
     "plugin:ulrssm/zeroshot_faust|pair_fr": 0,
     "plugin:ulrssm/zeroshot_faust|pair_sr": 0,
     "plugin:ulrssm/zeroshot_faust|tpl_fr": 0,
     "plugin:ulrssm/zeroshot_faust|tpl_sr": 0
    },
    "missing_on_same": [],
    "righe_proprie": {}
   },
   "spearman_con_taglia": {
    "nocrop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.4293,
      "est_cs": 0.5609
     },
     "factorized_s1234|form": {
      "oracle_size": 0.2397,
      "est_cs": 0.3813
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.0284,
      "est_cs": 0.1444
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.4383,
      "est_cs": 0.5451
     },
     "factorized_s2345|form": {
      "oracle_size": 0.2611,
      "est_cs": 0.3853
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0581,
      "est_cs": 0.1599
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.5703,
      "est_cs": 0.7143
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.5811,
      "est_cs": 0.7113
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.4664,
      "est_cs": 0.654
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.1735,
      "est_cs": 0.3417
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0365,
      "est_cs": 0.1791
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.5071,
      "est_cs": 0.6674
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.1709,
      "est_cs": 0.3281
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.0785,
      "est_cs": 0.225
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.577,
      "est_cs": 0.6851
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": 0.2983,
      "est_cs": 0.3076
     },
     "mm_nicp_p2tri": {
      "oracle_size": 0.3652,
      "est_cs": 0.5361
     },
     "mm_nicp_template": {
      "oracle_size": 0.6496,
      "est_cs": 0.8822
     },
     "cs_nicp_template": {
      "oracle_size": 0.1488,
      "est_cs": 0.0622
     },
     "varifold": {
      "oracle_size": 0.473,
      "est_cs": 0.6046
     },
     "gnm_orig_coef": {
      "oracle_size": 0.5144,
      "est_cs": 0.6966
     },
     "gnm_orig_fr": {
      "oracle_size": 0.601,
      "est_cs": 0.7142
     },
     "gnm_orig_sr": {
      "oracle_size": 0.1623,
      "est_cs": 0.3749
     },
     "gnm_va_coef": {
      "oracle_size": 0.4906,
      "est_cs": 0.6685
     },
     "gnm_va_fr": {
      "oracle_size": 0.6066,
      "est_cs": 0.6916
     },
     "gnm_va_sr": {
      "oracle_size": -0.1057,
      "est_cs": -0.0123
     },
     "gnm_vb_coef": {
      "oracle_size": 0.2713,
      "est_cs": 0.3458
     },
     "gnm_vb_fr": {
      "oracle_size": 0.6025,
      "est_cs": 0.7006
     },
     "gnm_vb_sr": {
      "oracle_size": 0.0451,
      "est_cs": 0.127
     },
     "gnm_vb_comp": {
      "oracle_size": 0.5262,
      "est_cs": 0.6251
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.3046,
      "est_cs": 0.4895
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.545,
      "est_cs": 0.6518
     },
     "flame2023_orig_sr": {
      "oracle_size": -0.0517,
      "est_cs": 0.0607
     },
     "flame2023_va_coef": {
      "oracle_size": 0.3392,
      "est_cs": 0.5157
     },
     "flame2023_va_fr": {
      "oracle_size": 0.5764,
      "est_cs": 0.6606
     },
     "flame2023_va_sr": {
      "oracle_size": -0.1122,
      "est_cs": -0.0023
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.3487,
      "est_cs": 0.4911
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.5389,
      "est_cs": 0.6587
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0135,
      "est_cs": 0.1134
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.4488,
      "est_cs": 0.5774
     },
     "arcface_normals": {
      "oracle_size": 0.3002,
      "est_cs": 0.3613
     },
     "shapedna_k50": {
      "oracle_size": -0.1207,
      "est_cs": -0.0651
     },
     "hks": {
      "oracle_size": -0.0925,
      "est_cs": -0.0334
     },
     "wks": {
      "oracle_size": -0.0561,
      "est_cs": -0.0601
     },
     "est_cs": {
      "oracle_size": 0.7377,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.7377
     }
    }
   },
   "seed": 1234,
   "n_methods": 64,
   "plugins_full": [
    "plugin:clip/smooth|fr",
    "plugin:clip/smooth|sr",
    "plugin:dinov2/depth|fr",
    "plugin:dinov2/depth|sr",
    "plugin:dreamsim/smooth|fr",
    "plugin:dreamsim/smooth|sr",
    "plugin:gw/euclid_128_cgw2|fr",
    "plugin:gw/euclid_128_cgw2|sr",
    "plugin:lpips/depth|fr",
    "plugin:lpips/depth|sr",
    "plugin:now_metrical|mm",
    "plugin:now_original|mm",
    "plugin:realy|mm",
    "plugin:stirling|mm",
    "plugin:ulrssm/faces_xyz|pair_fr",
    "plugin:ulrssm/faces_xyz|pair_sr",
    "plugin:ulrssm/faces_xyz|tpl_fr",
    "plugin:ulrssm/faces_xyz|tpl_sr"
   ],
   "plugins_partial": [],
   "seconds": 18.805590391159058
  }
 },
 "points_vs_sources": {
  "spearman_e2_nocrop": {
   "n": 144,
   "max_abs_diff_point": 8.326672684688674e-17,
   "worst": "faceverse_neutral oracle_size sr: righe qui 99000, fonte 99000",
   "n_equal_1e-12": 144,
   "n_same_rows": 144
  },
  "spearman_e3_crop": {
   "n": 30,
   "max_abs_diff_point": 8.326672684688674e-17,
   "worst": "faceverse_neutral oracle_size sr: righe qui 49500, fonte 49500",
   "n_equal_1e-12": 30,
   "n_same_rows": 30
  },
  "spearman_e3_all_cross": {
   "n": 30,
   "max_abs_diff_point": 8.326672684688674e-17,
   "worst": "faceverse_neutral oracle_size sr: righe qui 148500, fonte 148500",
   "n_equal_1e-12": 30,
   "n_same_rows": 30
  },
  "g0_arcface_nocrop": {
   "n": 2,
   "max_abs_diff_point": 1.2186759101862066e-09,
   "worst": "faceverse_neutral arcface_normals sr: righe qui 99000, fonte 99000",
   "n_equal_1e-12": 0,
   "n_same_rows": 2
  },
  "g0_arcface_crop": {
   "n": 2,
   "max_abs_diff_point": 1.409874772484443e-09,
   "worst": "faceverse_neutral arcface_normals sr: righe qui 49500, fonte 49500",
   "n_equal_1e-12": 0,
   "n_same_rows": 2
  },
  "topo_pairs_e3_media_20_ordinate_senza_crop": {
   "n": 0
  },
  "topo_pairs_e3_media_5_stessa_topologia": {
   "n": 0
  },
  "e12_hifi3d_fr_spettrali_reti": {
   "n": 0
  }
 },
 "plugins": [
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/clip/depth",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/clip/flat",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/clip/smooth",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/dinov2/depth",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/dinov2/flat",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/dinov2/smooth",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/dreamsim/depth",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/dreamsim/flat",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/dreamsim/smooth",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/gw/euclid_128_cgw2",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/gw/euclid_256_cg",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/gw/euclid_256_cgw2",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/lpips/depth",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/lpips/flat",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/lpips/smooth",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_metrical",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_metrical/lmk_det",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_metrical/lmk_s1",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_metrical/lmk_s2",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_metrical/lmk_s3",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_metrical/lmk_s5",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_original",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_original/lmk_det",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_original/lmk_s1",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_original/lmk_s2",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_original/lmk_s3",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/now_original/lmk_s5",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/realy",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/realy/lmk_det",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/realy/lmk_s1",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/realy/lmk_s2",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/realy/lmk_s3",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/realy/lmk_s5",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/stirling",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/stirling/lmk_det",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/stirling/lmk_s1",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/stirling/lmk_s2",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/stirling/lmk_s3",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/stirling/lmk_s5",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/ulrssm/faces_wks",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/ulrssm/faces_xyz",
  "/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/competitors_new/ulrssm/zeroshot_faust"
 ],
 "plugin_kinds": {
  "clip/depth": "non dichiarata",
  "clip/flat": "non dichiarata",
  "clip/smooth": "dichiarata",
  "dinov2/depth": "dichiarata",
  "dinov2/flat": "non dichiarata",
  "dinov2/smooth": "non dichiarata",
  "dreamsim/depth": "non dichiarata",
  "dreamsim/flat": "non dichiarata",
  "dreamsim/smooth": "dichiarata",
  "gw/euclid_128_cgw2": "dichiarata",
  "gw/euclid_256_cg": "non dichiarata",
  "gw/euclid_256_cgw2": "non dichiarata",
  "lpips/depth": "dichiarata",
  "lpips/flat": "non dichiarata",
  "lpips/smooth": "non dichiarata",
  "now_metrical": "dichiarata",
  "now_metrical/lmk_det": "lmk_det",
  "now_metrical/lmk_s1": "non dichiarata",
  "now_metrical/lmk_s2": "non dichiarata",
  "now_metrical/lmk_s3": "non dichiarata",
  "now_metrical/lmk_s5": "non dichiarata",
  "now_original": "dichiarata",
  "now_original/lmk_det": "lmk_det",
  "now_original/lmk_s1": "non dichiarata",
  "now_original/lmk_s2": "non dichiarata",
  "now_original/lmk_s3": "non dichiarata",
  "now_original/lmk_s5": "non dichiarata",
  "realy": "dichiarata",
  "realy/lmk_det": "lmk_det",
  "realy/lmk_s1": "non dichiarata",
  "realy/lmk_s2": "non dichiarata",
  "realy/lmk_s3": "non dichiarata",
  "realy/lmk_s5": "non dichiarata",
  "stirling": "dichiarata",
  "stirling/lmk_det": "lmk_det",
  "stirling/lmk_s1": "non dichiarata",
  "stirling/lmk_s2": "non dichiarata",
  "stirling/lmk_s3": "non dichiarata",
  "stirling/lmk_s5": "non dichiarata",
  "ulrssm/faces_wks": "non dichiarata",
  "ulrssm/faces_xyz": "dichiarata",
  "ulrssm/zeroshot_faust": "non dichiarata"
 },
 "plugin_dirs_ignored_without_fm_plugin_json": [
  "clip",
  "dinov2",
  "dreamsim",
  "gw",
  "lpips",
  "percettivi",
  "ulrssm"
 ]
}
```

