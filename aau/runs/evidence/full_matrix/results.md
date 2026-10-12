# Matrice completa e confrontabile: Spearman con FR e SR, delta appaiati

Codice `aau/full_matrix/` (`fm_eval.py`, definizioni nel docstring), numeri in questa cartella. Ogni metodo di un dominio e' valutato sulle STESSE righe (maschera comune: righe con tutte le colonne del dominio finite), con le STESSE 1000 repliche per soggetto (seme della T1 del dominio, lo stesso per tutti gli insiemi). IC 95% percentile. Delta appaiati = differenza replica per replica. FaMoS TEST sta in una sezione descrittiva a parte in fondo. Ava-256 non e' toccato.

## File e colonne (nomi stabili)

- `spearman.csv`: `metodo` (etichetta), `colonna` (id stabile), `famiglia`, `nostro`, `dichiarata` (lettura dichiarata per quella GT), `post_hoc` (d_F calibrata, varianti A e B dei fit e composizione, template FaMoS costruito qui), `dominio`, `gt` (fr, sr), `insieme_righe` (nocrop, crop, all_cross, con_perturbazione, coppia_topologie, stessa_topologia, coppia_ordinata, media_20_ordinate_senza_crop, media_5_stessa_topologia), `perturbazione` (`-`, una topologia, `a~b` coppia non ordinata, `a>b` ordinata, `tutte`; `con_perturbazione` P = P contro le altre topologie SENZA crop, la scomposizione di nocrop; per P = crop coincide con `crop`; `stessa_topologia` = soggetti diversi con la stessa topologia, crop - crop compresa), `rho`, `ic_lo`, `ic_hi`, `n_righe`, `n_soggetti`, `seme`, `passo` (`comune`; la sensibilita' del NICP; le righe proprie di un plug-in con buchi).
- `deltas.csv`: le stesse chiavi + `tipo` (vs_nostro = braccio - concorrente; scala_allineamento = a - b; migliore_vs_migliore), `confronto`, `ruolo` (scala: principale o secondario), `piu_allineato` (scala: a o b), `metodo_a`, `colonna_a`, `post_hoc_a`, `metodo_b`, `colonna_b`, `post_hoc_b`, `rho_a`, `rho_b`, `delta`, `ic_lo`, `ic_hi`, `p_le0` (frazione delle repliche con delta <= 0).
- `non_applicabili.csv`: `dominio`, `metodo`, `colonna`, `insieme_righe`, `motivo` (anche le CAUTELE su celle calcolate).
- `controls.json`: controlli di riproduzione, maschera e righe tolte per colonna, fallimenti del NICP per topologia, Spearman di ogni colonna con la taglia oracolo e stimata (`spearman_con_taglia`), plug-in.

Letture dichiarate: nostri FR = d_F cal. (post hoc, emendamento 4), d_F non cal. (preregistrata), ctrlfr; SR = d_P (ctrlfr non e' un braccio per SR: e' in `spearman.csv` ma non e' un riferimento dei delta SR). Fit 3DMM: coefficienti (FR e SR), mesh FR con FR, mesh SR con SR, composizione di B con FR (k calibrato solo per B: per preregistrato e vA la composizione non esiste). Geometrici, percettivi, spettrali, reti e riferimenti: entrambe le GT. Tutte le letture sono comunque in `spearman.csv` (`dichiarata` = False).

## Come aggiungere un concorrente (plug-in)

Una cartella `aau/runs/evidence/competitors_new/<metodo>/` (o `<metodo>/<tag>/`) con, per dominio (`hifi3d`, `facescape`, `faceverse`, `faceverse_neutral`, `famos`), `<dominio>.npz` (`subjects`, `topologies`, matrici `D` o `D_<lettura>` (n, n) sulle mesh elencate; FaMoS: `subjects` = view_id, `topologies` = `scan_gallery`) oppure `<dominio>.csv` / `pairs_<dominio>.csv` (`subject_a`, `topology_a`, `subject_b`, `topology_b`, una colonna per lettura), e il file di marcatura OBBLIGATORIO `fm_plugin.json` (anche `{}`; le cartelle senza, es. `pilot/` o `prep/`, si ignorano; facoltativi `label`, `family`, `readings`: {lettura: {`label`, `gt`}}; senza, una lettura `fr` / `*_fr` e' dichiarata per FR, `sr` / `*_sr` per SR, le altre per entrambe). Poi solo lo stadio finale: `FM_STEP=eval sbatch --cpus-per-task=64 --mem=192G --time=02:00:00 aau/full_matrix/fm.sbatch`. Un plug-in finito su tutta la maschera comune entra nello stesso passo e non cambia le righe degli altri; uno con buchi va in un passo a parte (`passo` = righe proprie), con i bracci di riferimento sulle sue righe.

## Non applicabili e cautele

| dominio | metodo | insieme | motivo |
| --- | --- | --- | --- |
| hifi3d | NICP per coppia in mm | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| hifi3d | NICP per coppia cs | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| facescape | ArcFace-render, ombreggiato 3 viste (E12) | tutti | render ombreggiato calcolato solo da E12 (HIFI3D, FaceVerse con espressioni) |
| facescape | OpenShape PointBERT | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| facescape | Uni3D-g | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| facescape | NICP per coppia in mm | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| facescape | NICP per coppia cs | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| faceverse | OpenShape PointBERT | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| faceverse | Uni3D-g | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| faceverse | NICP per coppia in mm | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| faceverse | NICP per coppia cs | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| faceverse_neutral | C3M e123, d_F cal. | tutti | embedding di C3M e123 mai calcolato sulla vista neutra (store assente; bp_e3 e faceverse_neutral lo saltano) |
| faceverse_neutral | C3M e123, d_F non cal. | tutti | embedding di C3M e123 mai calcolato sulla vista neutra (store assente; bp_e3 e faceverse_neutral lo saltano) |
| faceverse_neutral | C3M e123, d_P | tutti | embedding di C3M e123 mai calcolato sulla vista neutra (store assente; bp_e3 e faceverse_neutral lo saltano) |
| faceverse_neutral | NICP su template cs | tutti | template cs mai iscritto sulla neutra (faceverse_neutral/blmm ha solo mm) |
| faceverse_neutral | ArcFace-render, ombreggiato 3 viste (E12) | tutti | render ombreggiato calcolato solo da E12 (HIFI3D, FaceVerse con espressioni) |
| faceverse_neutral | OpenShape PointBERT | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| faceverse_neutral | Uni3D-g | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| faceverse_neutral | NICP per coppia in mm | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| faceverse_neutral | NICP per coppia cs | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| famos | NICP su template cs | tutti | template cs non calcolato su FaMoS (qui solo mm) |
| famos | ArcFace-render, ombreggiato 3 viste (E12) | tutti | render ombreggiato calcolato solo da E12 (HIFI3D, FaceVerse con espressioni) |
| famos | OpenShape PointBERT | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| famos | Uni3D-g | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| famos | tutti | crop | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | tutti | all_cross | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | tutti | con_perturbazione | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | tutti | coppia_topologie | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | tutti | stessa_topologia | FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> scan) |
| famos | ArcFace-render, normal map 3 viste (G0) | nocrop | CAUTELA (calcolato, non escluso): sulle scansioni parziali il detector trova il volto in 27/45 render ({'-030.00': 11, '+000.00': 11, '+030.00': 5}); il ritaglio a +30 gradi e' tarato su 5 render su 15, fuori dal regime validato in G0 |

## Miglior nostro contro miglior concorrente (famiglie sintetiche; letture dichiarate, scelti sul punto: post hoc)

| dominio | GT | insieme | righe | miglior nostro | rho | miglior concorrente | rho | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | FR | nocrop | 98224 | ctrlfr s1234 | 0.757 [0.685, 0.817] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.712 [0.627, 0.781] | +0.046 [+0.006, +0.083] |
| hifi3d | FR | crop | 49306 | factorized s1234, d_F cal. | 0.663 [0.589, 0.721] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.671 [0.582, 0.740] | -0.008 [-0.048, +0.032] |
| hifi3d | FR | all_cross | 147530 | factorized s1234, d_F cal. | 0.709 [0.634, 0.767] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.697 [0.610, 0.766] | +0.011 [-0.026, +0.048] |
| hifi3d | FR | con_perturbazione original | 39406 | ctrlfr s1234 | 0.767 [0.696, 0.828] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.715 [0.631, 0.784] | +0.052 [+0.012, +0.089] |
| hifi3d | FR | con_perturbazione remesh | 39406 | ctrlfr s1234 | 0.747 [0.673, 0.805] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.708 [0.622, 0.778] | +0.038 [-0.001, +0.075] |
| hifi3d | FR | con_perturbazione down8k | 39406 | ctrlfr s1234 | 0.766 [0.693, 0.827] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.713 [0.627, 0.783] | +0.053 [+0.014, +0.092] |
| hifi3d | FR | con_perturbazione up60k | 38824 | ctrlfr s1234 | 0.755 [0.680, 0.815] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.715 [0.632, 0.784] | +0.039 [+0.001, +0.077] |
| hifi3d | FR | con_perturbazione noisy | 39406 | ctrlfr s1234 | 0.753 [0.683, 0.814] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.706 [0.621, 0.776] | +0.047 [+0.007, +0.084] |
| hifi3d | FR | stessa_topologia tutte | 29700 | ctrlfr s1234 | 0.780 [0.710, 0.839] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.716 [0.630, 0.786] | +0.064 [+0.025, +0.103] |
| hifi3d | SR | nocrop | 98224 | factorized s1234, d_P | 0.622 [0.550, 0.685] | GNM (visto) vB, coefficienti | 0.674 [0.610, 0.734] | -0.053 [-0.136, +0.032] |
| hifi3d | SR | crop | 49306 | factorized s1234, d_P | 0.550 [0.472, 0.623] | GNM (visto) vB, coefficienti | 0.651 [0.587, 0.711] | -0.101 [-0.184, -0.015] |
| hifi3d | SR | all_cross | 147530 | factorized s1234, d_P | 0.557 [0.485, 0.624] | GNM (visto) vB, coefficienti | 0.664 [0.600, 0.724] | -0.107 [-0.190, -0.024] |
| hifi3d | SR | con_perturbazione original | 39406 | factorized s1234, d_P | 0.629 [0.558, 0.692] | GNM (visto) vB, coefficienti | 0.676 [0.610, 0.735] | -0.047 [-0.131, +0.039] |
| hifi3d | SR | con_perturbazione remesh | 39406 | factorized s1234, d_P | 0.626 [0.553, 0.690] | GNM (visto) vB, coefficienti | 0.674 [0.607, 0.733] | -0.047 [-0.131, +0.037] |
| hifi3d | SR | con_perturbazione down8k | 39406 | factorized s1234, d_P | 0.625 [0.555, 0.688] | GNM (visto) vB, coefficienti | 0.672 [0.606, 0.732] | -0.046 [-0.127, +0.038] |
| hifi3d | SR | con_perturbazione up60k | 38824 | factorized s1234, d_P | 0.626 [0.554, 0.690] | GNM (visto) vB, coefficienti | 0.673 [0.607, 0.733] | -0.047 [-0.128, +0.038] |
| hifi3d | SR | con_perturbazione noisy | 39406 | factorized s1234, d_P | 0.604 [0.533, 0.668] | GNM (visto) vB, coefficienti | 0.680 [0.614, 0.738] | -0.075 [-0.161, +0.014] |
| hifi3d | SR | stessa_topologia tutte | 29700 | factorized s1234, d_P | 0.631 [0.560, 0.695] | GNM (visto) vB, coefficienti | 0.673 [0.609, 0.732] | -0.042 [-0.124, +0.042] |
| facescape | FR | nocrop | 88724 | C3M e205, d_F cal. | 0.711 [0.651, 0.763] | GNM (visto) vB, composizione S_B + k d_P | 0.800 [0.741, 0.847] | -0.089 [-0.147, -0.037] |
| facescape | FR | crop | 46931 | C3M e123, d_F non cal. | 0.539 [0.460, 0.617] | GNM (visto) vB, composizione S_B + k d_P | 0.761 [0.700, 0.812] | -0.222 [-0.282, -0.163] |
| facescape | FR | all_cross | 135655 | C3M e205, d_F non cal. | 0.469 [0.403, 0.533] | GNM (visto) vB, composizione S_B + k d_P | 0.783 [0.724, 0.831] | -0.313 [-0.371, -0.259] |
| facescape | FR | con_perturbazione original | 37031 | C3M e205, d_F cal. | 0.725 [0.665, 0.777] | GNM (visto) vB, composizione S_B + k d_P | 0.800 [0.741, 0.847] | -0.075 [-0.132, -0.023] |
| facescape | FR | con_perturbazione remesh | 37031 | C3M e205, d_F cal. | 0.729 [0.670, 0.780] | GNM (visto) vB, composizione S_B + k d_P | 0.801 [0.743, 0.849] | -0.072 [-0.129, -0.022] |
| facescape | FR | con_perturbazione down8k | 37031 | factorized s2345, d_F non cal. | 0.704 [0.623, 0.779] | GNM (visto) vB, composizione S_B + k d_P | 0.794 [0.732, 0.842] | -0.090 [-0.158, -0.034] |
| facescape | FR | con_perturbazione up60k | 29324 | C3M e205, d_F cal. | 0.742 [0.684, 0.792] | GNM (visto) vB, composizione S_B + k d_P | 0.803 [0.742, 0.852] | -0.061 [-0.118, -0.011] |
| facescape | FR | con_perturbazione noisy | 37031 | C3M e205, d_F cal. | 0.694 [0.633, 0.749] | GNM (visto) vB, composizione S_B + k d_P | 0.801 [0.742, 0.850] | -0.107 [-0.168, -0.052] |
| facescape | FR | stessa_topologia tutte | 29700 | C3M e205, d_F cal. | 0.728 [0.669, 0.776] | GNM (visto) vB, composizione S_B + k d_P | 0.814 [0.759, 0.859] | -0.086 [-0.142, -0.037] |
| facescape | SR | nocrop | 88724 | C3M e205, d_P | 0.756 [0.695, 0.807] | GNM (visto) vB, mesh d'identita' SR | 0.854 [0.803, 0.894] | -0.098 [-0.147, -0.051] |
| facescape | SR | crop | 46931 | C3M e123, d_P | 0.576 [0.506, 0.644] | GNM (visto) vB, mesh d'identita' SR | 0.830 [0.778, 0.873] | -0.255 [-0.315, -0.195] |
| facescape | SR | all_cross | 135655 | C3M e205, d_P | 0.499 [0.435, 0.557] | GNM (visto) vB, mesh d'identita' SR | 0.839 [0.786, 0.881] | -0.340 [-0.390, -0.291] |
| facescape | SR | con_perturbazione original | 37031 | C3M e205, d_P | 0.760 [0.701, 0.810] | GNM (visto) vB, mesh d'identita' SR | 0.853 [0.803, 0.892] | -0.093 [-0.141, -0.044] |
| facescape | SR | con_perturbazione remesh | 37031 | C3M e205, d_P | 0.768 [0.711, 0.815] | GNM (visto) vB, mesh d'identita' SR | 0.855 [0.806, 0.895] | -0.088 [-0.135, -0.042] |
| facescape | SR | con_perturbazione down8k | 37031 | factorized s2345, d_P | 0.761 [0.694, 0.821] | GNM (visto) vB, mesh d'identita' SR | 0.853 [0.802, 0.893] | -0.091 [-0.139, -0.047] |
| facescape | SR | con_perturbazione up60k | 29324 | C3M e205, d_P | 0.776 [0.718, 0.824] | GNM (visto) vB, mesh d'identita' SR | 0.856 [0.805, 0.896] | -0.080 [-0.127, -0.036] |
| facescape | SR | con_perturbazione noisy | 37031 | C3M e205, d_P | 0.759 [0.698, 0.809] | GNM (visto) vB, mesh d'identita' SR | 0.854 [0.803, 0.894] | -0.094 [-0.148, -0.044] |
| facescape | SR | stessa_topologia tutte | 29700 | C3M e205, d_P | 0.756 [0.699, 0.805] | GNM (visto) vB, mesh d'identita' SR | 0.855 [0.807, 0.895] | -0.099 [-0.146, -0.056] |
| faceverse | FR | nocrop | 99000 | factorized s2345, d_F cal. | 0.318 [0.249, 0.379] | GNM (visto) vB, composizione S_B + k d_P | 0.338 [0.255, 0.407] | -0.020 [-0.084, +0.053] |
| faceverse | FR | crop | 49500 | C3M e205, d_F non cal. | 0.297 [0.227, 0.365] | GNM (visto) vB, composizione S_B + k d_P | 0.325 [0.242, 0.394] | -0.028 [-0.111, +0.057] |
| faceverse | FR | all_cross | 148500 | C3M e205, d_F non cal. | 0.302 [0.233, 0.370] | GNM (visto) vB, composizione S_B + k d_P | 0.334 [0.253, 0.402] | -0.031 [-0.107, +0.048] |
| faceverse | FR | con_perturbazione original | 39600 | C3M e205, d_F non cal. | 0.322 [0.247, 0.392] | ICP + Chamfer in mm (per coppia) | 0.351 [0.275, 0.424] | -0.030 [-0.102, +0.045] |
| faceverse | FR | con_perturbazione remesh | 39600 | factorized s2345, d_F cal. | 0.322 [0.254, 0.385] | ICP + Chamfer in mm (per coppia) | 0.338 [0.257, 0.413] | -0.015 [-0.083, +0.053] |
| faceverse | FR | con_perturbazione down8k | 39600 | factorized s2345, d_F non cal. | 0.306 [0.230, 0.373] | GNM (visto) vB, composizione S_B + k d_P | 0.341 [0.257, 0.411] | -0.035 [-0.104, +0.041] |
| faceverse | FR | con_perturbazione up60k | 39600 | factorized s2345, d_F cal. | 0.321 [0.251, 0.383] | ICP + Chamfer in mm (per coppia) | 0.335 [0.255, 0.411] | -0.014 [-0.085, +0.056] |
| faceverse | FR | con_perturbazione noisy | 39600 | factorized s2345, d_F non cal. | 0.330 [0.253, 0.400] | GNM (visto) vB, composizione S_B + k d_P | 0.345 [0.263, 0.413] | -0.015 [-0.081, +0.055] |
| faceverse | FR | stessa_topologia tutte | 29700 | C3M e205, d_F cal. | 0.326 [0.253, 0.395] | ICP + Chamfer in mm (per coppia) | 0.343 [0.269, 0.413] | -0.017 [-0.077, +0.046] |
| faceverse | SR | nocrop | 99000 | C3M e205, d_P | 0.314 [0.242, 0.382] | GNM (visto) vB, mesh d'identita' SR | 0.327 [0.239, 0.402] | -0.013 [-0.089, +0.063] |
| faceverse | SR | crop | 49500 | C3M e205, d_P | 0.308 [0.241, 0.376] | GNM (visto) vB, mesh d'identita' SR | 0.316 [0.229, 0.388] | -0.007 [-0.079, +0.068] |
| faceverse | SR | all_cross | 148500 | C3M e205, d_P | 0.312 [0.244, 0.378] | GNM (visto) vB, mesh d'identita' SR | 0.323 [0.238, 0.397] | -0.011 [-0.084, +0.064] |
| faceverse | SR | con_perturbazione original | 39600 | C3M e205, d_P | 0.330 [0.254, 0.403] | GNM (visto) vB, mesh d'identita' SR | 0.325 [0.237, 0.400] | +0.005 [-0.072, +0.086] |
| faceverse | SR | con_perturbazione remesh | 39600 | C3M e205, d_P | 0.327 [0.247, 0.400] | GNM (visto) vB, mesh d'identita' SR | 0.325 [0.235, 0.401] | +0.002 [-0.082, +0.088] |
| faceverse | SR | con_perturbazione down8k | 39600 | factorized s2345, d_P | 0.303 [0.225, 0.372] | GNM (visto) vB, mesh d'identita' SR | 0.338 [0.250, 0.412] | -0.035 [-0.115, +0.045] |
| faceverse | SR | con_perturbazione up60k | 39600 | C3M e123, d_P | 0.313 [0.235, 0.388] | GNM (visto) vB, mesh d'identita' SR | 0.322 [0.233, 0.397] | -0.008 [-0.091, +0.068] |
| faceverse | SR | con_perturbazione noisy | 39600 | factorized s2345, d_P | 0.320 [0.240, 0.389] | GNM (visto) vB, mesh d'identita' SR | 0.329 [0.240, 0.404] | -0.009 [-0.082, +0.066] |
| faceverse | SR | stessa_topologia tutte | 29700 | C3M e205, d_P | 0.334 [0.263, 0.403] | GNM (visto) vB, mesh d'identita' SR | 0.322 [0.236, 0.395] | +0.012 [-0.063, +0.087] |
| faceverse_neutral | FR | nocrop | 99000 | factorized s2345, d_F non cal. | 0.373 [0.293, 0.451] | ICP + Chamfer in mm (per coppia) | 0.409 [0.326, 0.481] | -0.035 [-0.123, +0.042] |
| faceverse_neutral | FR | crop | 49500 | C3M e205, d_F non cal. | 0.345 [0.262, 0.420] | GNM (visto) vB, composizione S_B + k d_P | 0.385 [0.299, 0.462] | -0.040 [-0.134, +0.049] |
| faceverse_neutral | FR | all_cross | 148500 | C3M e205, d_F non cal. | 0.353 [0.269, 0.429] | GNM (visto) vB, composizione S_B + k d_P | 0.383 [0.296, 0.459] | -0.030 [-0.118, +0.059] |
| faceverse_neutral | FR | con_perturbazione original | 39600 | factorized s2345, d_F non cal. | 0.375 [0.294, 0.453] | ICP + Chamfer in mm (per coppia) | 0.415 [0.332, 0.486] | -0.040 [-0.125, +0.038] |
| faceverse_neutral | FR | con_perturbazione remesh | 39600 | factorized s2345, d_F non cal. | 0.378 [0.297, 0.454] | ICP + Chamfer in mm (per coppia) | 0.414 [0.331, 0.486] | -0.037 [-0.125, +0.043] |
| faceverse_neutral | FR | con_perturbazione down8k | 39600 | factorized s2345, d_F non cal. | 0.373 [0.292, 0.450] | ICP + Chamfer in mm (per coppia) | 0.402 [0.314, 0.475] | -0.029 [-0.116, +0.052] |
| faceverse_neutral | FR | con_perturbazione up60k | 39600 | factorized s2345, d_F non cal. | 0.374 [0.293, 0.452] | ICP + Chamfer in mm (per coppia) | 0.412 [0.328, 0.484] | -0.038 [-0.122, +0.040] |
| faceverse_neutral | FR | con_perturbazione noisy | 39600 | factorized s2345, d_F non cal. | 0.367 [0.289, 0.444] | ICP + Chamfer in mm (per coppia) | 0.404 [0.325, 0.476] | -0.037 [-0.122, +0.042] |
| faceverse_neutral | FR | stessa_topologia tutte | 29700 | C3M e205, d_F cal. | 0.395 [0.312, 0.470] | ICP + Chamfer in mm (per coppia) | 0.419 [0.337, 0.492] | -0.024 [-0.095, +0.050] |
| faceverse_neutral | SR | nocrop | 99000 | C3M e205, d_P | 0.382 [0.304, 0.452] | GNM (visto) vB, mesh d'identita' SR | 0.399 [0.315, 0.473] | -0.017 [-0.097, +0.065] |
| faceverse_neutral | SR | crop | 49500 | C3M e205, d_P | 0.372 [0.294, 0.442] | GNM (visto) vB, mesh d'identita' SR | 0.402 [0.321, 0.473] | -0.030 [-0.103, +0.049] |
| faceverse_neutral | SR | all_cross | 148500 | C3M e205, d_P | 0.378 [0.304, 0.448] | GNM (visto) vB, mesh d'identita' SR | 0.400 [0.317, 0.473] | -0.021 [-0.098, +0.057] |
| faceverse_neutral | SR | con_perturbazione original | 39600 | C3M e205, d_P | 0.398 [0.321, 0.469] | ICP + Chamfer cs (per coppia, scala tolta) | 0.400 [0.323, 0.471] | -0.002 [-0.072, +0.074] |
| faceverse_neutral | SR | con_perturbazione remesh | 39600 | C3M e205, d_P | 0.393 [0.313, 0.464] | ICP + Chamfer cs (per coppia, scala tolta) | 0.403 [0.325, 0.475] | -0.011 [-0.079, +0.066] |
| faceverse_neutral | SR | con_perturbazione down8k | 39600 | C3M e205, d_P | 0.387 [0.307, 0.459] | GNM (visto) vB, mesh d'identita' SR | 0.400 [0.314, 0.475] | -0.012 [-0.093, +0.070] |
| faceverse_neutral | SR | con_perturbazione up60k | 39600 | C3M e205, d_P | 0.389 [0.310, 0.463] | ICP + Chamfer cs (per coppia, scala tolta) | 0.400 [0.324, 0.471] | -0.011 [-0.082, +0.069] |
| faceverse_neutral | SR | con_perturbazione noisy | 39600 | factorized s2345, d_P | 0.365 [0.286, 0.437] | GNM (visto) vB, mesh d'identita' SR | 0.398 [0.313, 0.472] | -0.032 [-0.112, +0.048] |
| faceverse_neutral | SR | stessa_topologia tutte | 29700 | C3M e205, d_P | 0.417 [0.339, 0.490] | ICP + Chamfer cs (per coppia, scala tolta) | 0.405 [0.329, 0.472] | +0.012 [-0.056, +0.090] |

## Scala dell'allineamento

Delta appaiati a - b (stesse righe, stesse repliche) lungo i gradi di liberta' dell'allineamento. PRINCIPALI: 1 (togliere la taglia: la modalita' cs scala ogni mesh alla cs_ref del dominio, `blmm.work_coords`; non e' una scala per coppia), 2 (ICP cs - NICP cs), 3b (NICP su template cs - NICP per coppia cs), 4 e 5 con FR contro ICP mm, con SR contro ICP cs e NICP cs (con SR il riferimento corretto e' senza taglia). SECONDARI: 2b e 3 in mm, CONFUSI DALLA TAGLIA (in mm il NICP su template segue la taglia, il NICP per coppia la assorbe: vedi la tabella sotto), e 4 e 5 con SR contro ICP mm. 4: fit B con la mesh della GT (FR con FR, SR con SR); 5: nostro preregistrato (d_F non cal. con FR, d_P con SR), entrambi i semi.

Spearman (punto, righe nocrop della maschera) di ogni colonna della scala con la taglia oracolo / la taglia stimata (`est_cs`); tutte le colonne e gli insiemi crop e all_cross in `controls.json`, `spearman_con_taglia`:

| dominio | ICP + Chamfer in mm (per coppia) | ICP + Chamfer cs (per coppia, scala tolta) | NICP per coppia in mm | NICP per coppia cs | NICP su template in mm | NICP su template cs | GNM (visto) vB, mesh d'identita' FR | GNM (visto) vB, mesh d'identita' SR | factorized s1234, d_F non cal. | factorized s1234, d_P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | +0.51 / +0.56 | +0.04 / +0.04 | +0.22 / +0.14 | +0.04 / +0.02 | +0.77 / +0.86 | +0.04 / +0.04 | +0.70 / +0.77 | +0.08 / +0.08 | +0.37 / +0.37 | +0.08 / +0.05 |
| facescape | +0.08 / +0.08 | +0.02 / +0.04 | +0.05 / +0.02 | +0.01 / +0.02 | +0.42 / +0.35 | +0.07 / +0.04 | +0.29 / +0.18 | +0.11 / +0.08 | +0.11 / +0.09 | +0.09 / +0.06 |
| faceverse | +0.10 / +0.31 | -0.01 / +0.10 | -0.06 / +0.03 | -0.05 / +0.07 | +0.20 / +0.44 | +0.01 / +0.10 | +0.15 / +0.23 | +0.03 / +0.07 | +0.03 / +0.14 | +0.01 / +0.12 |
| faceverse_neutral | +0.11 / +0.29 | -0.02 / +0.11 | -0.08 / -0.01 | -0.07 / +0.06 | +0.22 / +0.47 | - | +0.16 / +0.21 | -0.01 / +0.06 | +0.02 / +0.14 | -0.01 / +0.12 |
| famos | +0.58 / +0.69 | +0.30 / +0.31 | +0.37 / +0.54 | +0.21 / +0.35 | +0.65 / +0.88 | - | +0.60 / +0.70 | +0.05 / +0.13 | +0.24 / +0.38 | +0.03 / +0.14 |

### Confronti principali

GT FR:

| dominio | passo | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.285 [+0.187, +0.383] | +0.204 [+0.134, +0.275] | +0.292 [+0.211, +0.372] | +0.291 [+0.196, +0.385] | +0.289 [+0.188, +0.392] | +0.276 [+0.180, +0.375] | +0.289 [+0.187, +0.393] | +0.289 [+0.191, +0.386] | +0.257 [+0.158, +0.356] |
| hifi3d | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.017 [-0.046, +0.012] | -0.044 [-0.075, -0.014] | -0.061 [-0.090, -0.032] | -0.011 [-0.035, +0.017] | -0.012 [-0.042, +0.018] | -0.026 [-0.057, +0.004] | -0.012 [-0.042, +0.019] | -0.014 [-0.041, +0.013] | - |
| hifi3d | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.187 [-0.275, -0.105] | -0.227 [-0.301, -0.161] | -0.231 [-0.304, -0.166] | -0.166 [-0.251, -0.085] | -0.200 [-0.287, -0.118] | -0.183 [-0.272, -0.099] | -0.194 [-0.285, -0.109] | -0.201 [-0.295, -0.116] | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.022 [-0.028, +0.071] | +0.207 [+0.148, +0.262] | +0.095 [+0.044, +0.145] | +0.031 [-0.021, +0.082] | -0.006 [-0.056, +0.044] | +0.044 [-0.002, +0.093] | +0.000 [-0.049, +0.048] | +0.004 [-0.047, +0.055] | +0.027 [-0.022, +0.077] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | +0.043 [-0.007, +0.092] | +0.221 [+0.161, +0.279] | +0.114 [+0.062, +0.163] | +0.053 [+0.002, +0.100] | +0.014 [-0.035, +0.062] | +0.069 [+0.021, +0.118] | +0.023 [-0.027, +0.073] | +0.021 [-0.032, +0.072] | +0.046 [-0.005, +0.096] |
| hifi3d | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.001 [-0.055, +0.055] | +0.145 [+0.079, +0.201] | +0.028 [-0.029, +0.081] | +0.012 [-0.041, +0.065] | -0.026 [-0.081, +0.029] | +0.023 [-0.032, +0.077] | -0.020 [-0.075, +0.037] | -0.024 [-0.081, +0.032] | +0.001 [-0.055, +0.056] |
| hifi3d | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.012 [-0.075, +0.051] | +0.099 [+0.032, +0.161] | +0.005 [-0.056, +0.065] | -0.001 [-0.063, +0.061] | -0.039 [-0.101, +0.025] | +0.014 [-0.048, +0.075] | -0.033 [-0.096, +0.032] | -0.034 [-0.097, +0.030] | -0.010 [-0.074, +0.053] |
| facescape | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.038 [+0.018, +0.058] | +0.068 [+0.031, +0.102] | +0.065 [+0.043, +0.086] | +0.048 [+0.029, +0.068] | +0.046 [+0.022, +0.070] | +0.021 [+0.005, +0.038] | +0.042 [+0.019, +0.065] | +0.047 [+0.025, +0.070] | +0.007 [-0.019, +0.033] |
| facescape | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.069 [+0.043, +0.095] | -0.004 [-0.036, +0.031] | +0.011 [-0.014, +0.038] | +0.063 [+0.038, +0.088] | +0.073 [+0.042, +0.103] | +0.076 [+0.052, +0.100] | +0.074 [+0.044, +0.104] | +0.068 [+0.042, +0.094] | - |
| facescape | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | +0.054 [-0.032, +0.136] | -0.175 [-0.245, -0.100] | -0.112 [-0.179, -0.047] | +0.056 [-0.024, +0.135] | -0.039 [-0.128, +0.046] | +0.122 [+0.036, +0.206] | -0.014 [-0.107, +0.076] | +0.032 [-0.050, +0.115] | - |
| facescape | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.225 [+0.162, +0.292] | +0.336 [+0.281, +0.387] | +0.283 [+0.225, +0.343] | +0.229 [+0.163, +0.299] | +0.131 [+0.064, +0.199] | +0.276 [+0.220, +0.335] | +0.159 [+0.090, +0.231] | +0.193 [+0.125, +0.265] | +0.124 [+0.060, +0.190] |
| facescape | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | +0.177 [+0.116, +0.241] | +0.269 [+0.216, +0.322] | +0.229 [+0.173, +0.287] | +0.180 [+0.116, +0.247] | +0.072 [+0.014, +0.136] | +0.235 [+0.181, +0.293] | +0.114 [+0.053, +0.180] | +0.147 [+0.080, +0.215] | +0.074 [+0.011, +0.136] |
| facescape | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | +0.234 [+0.183, +0.285] | +0.195 [+0.128, +0.264] | +0.062 [+0.020, +0.106] | +0.244 [+0.190, +0.299] | +0.149 [+0.097, +0.199] | +0.289 [+0.240, +0.336] | +0.171 [+0.121, +0.225] | +0.201 [+0.145, +0.254] | +0.118 [+0.063, +0.170] |
| facescape | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | +0.245 [+0.185, +0.301] | +0.174 [+0.103, +0.247] | +0.038 [-0.005, +0.083] | +0.251 [+0.187, +0.310] | +0.163 [+0.103, +0.221] | +0.309 [+0.253, +0.361] | +0.184 [+0.122, +0.243] | +0.194 [+0.130, +0.253] | +0.125 [+0.063, +0.183] |
| faceverse | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.048 [-0.004, +0.102] | -0.002 [-0.051, +0.052] | +0.037 [-0.010, +0.089] | +0.044 [-0.007, +0.099] | +0.048 [-0.001, +0.100] | +0.048 [-0.004, +0.101] | +0.050 [-0.004, +0.108] | +0.046 [-0.007, +0.102] | +0.050 [-0.000, +0.103] |
| faceverse | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.078 [+0.022, +0.129] | +0.042 [-0.003, +0.084] | +0.057 [+0.005, +0.104] | +0.095 [+0.039, +0.147] | +0.081 [+0.023, +0.133] | +0.061 [+0.001, +0.115] | +0.080 [+0.023, +0.134] | +0.074 [+0.018, +0.128] | - |
| faceverse | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.076 [-0.184, +0.029] | -0.108 [-0.198, -0.019] | -0.090 [-0.185, +0.003] | -0.064 [-0.168, +0.042] | -0.069 [-0.183, +0.038] | -0.099 [-0.219, +0.019] | -0.061 [-0.173, +0.047] | -0.090 [-0.199, +0.015] | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | -0.045 [-0.112, +0.019] | +0.009 [-0.049, +0.066] | -0.023 [-0.085, +0.037] | -0.046 [-0.115, +0.025] | -0.052 [-0.124, +0.020] | -0.033 [-0.106, +0.037] | -0.051 [-0.119, +0.015] | -0.045 [-0.110, +0.019] | -0.051 [-0.117, +0.010] |
| faceverse | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.073 [-0.151, +0.002] | -0.027 [-0.093, +0.043] | -0.054 [-0.126, +0.016] | -0.075 [-0.148, -0.002] | -0.078 [-0.162, +0.003] | -0.055 [-0.140, +0.025] | -0.078 [-0.157, -0.004] | -0.080 [-0.158, -0.004] | -0.079 [-0.155, -0.005] |
| faceverse | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.040 [-0.118, +0.036] | -0.007 [-0.077, +0.067] | -0.026 [-0.098, +0.046] | -0.057 [-0.135, +0.020] | -0.040 [-0.120, +0.039] | -0.036 [-0.116, +0.047] | -0.035 [-0.117, +0.042] | -0.033 [-0.110, +0.043] | -0.043 [-0.118, +0.030] |
| faceverse | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.020 [-0.105, +0.056] | -0.016 [-0.090, +0.055] | -0.016 [-0.096, +0.058] | -0.036 [-0.115, +0.038] | -0.022 [-0.108, +0.056] | -0.018 [-0.111, +0.068] | -0.016 [-0.104, +0.063] | -0.010 [-0.096, +0.070] | -0.034 [-0.116, +0.043] |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.035 [-0.018, +0.090] | -0.017 [-0.066, +0.034] | +0.025 [-0.024, +0.074] | +0.037 [-0.015, +0.092] | +0.034 [-0.020, +0.089] | +0.035 [-0.019, +0.091] | +0.036 [-0.018, +0.091] | +0.031 [-0.021, +0.086] | +0.037 [-0.016, +0.092] |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.088 [+0.022, +0.148] | +0.043 [-0.014, +0.099] | +0.061 [+0.002, +0.119] | +0.092 [+0.026, +0.153] | +0.093 [+0.028, +0.155] | +0.084 [+0.018, +0.148] | +0.090 [+0.023, +0.150] | +0.081 [+0.019, +0.137] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | -0.088 [-0.160, -0.015] | +0.006 [-0.058, +0.069] | -0.052 [-0.119, +0.017] | -0.093 [-0.164, -0.021] | -0.096 [-0.169, -0.025] | -0.078 [-0.154, -0.003] | -0.092 [-0.164, -0.021] | -0.083 [-0.155, -0.014] | -0.097 [-0.167, -0.026] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.094 [-0.175, -0.012] | -0.011 [-0.082, +0.061] | -0.061 [-0.138, +0.014] | -0.101 [-0.182, -0.019] | -0.096 [-0.176, -0.016] | -0.086 [-0.170, +0.002] | -0.098 [-0.180, -0.017] | -0.091 [-0.173, -0.008] | -0.106 [-0.187, -0.026] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.072 [-0.151, +0.013] | -0.011 [-0.086, +0.066] | -0.047 [-0.122, +0.033] | -0.074 [-0.153, +0.010] | -0.074 [-0.152, +0.009] | -0.067 [-0.147, +0.020] | -0.070 [-0.150, +0.014] | -0.076 [-0.156, +0.006] | -0.070 [-0.148, +0.014] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.035 [-0.123, +0.042] | -0.006 [-0.080, +0.067] | -0.022 [-0.101, +0.053] | -0.040 [-0.125, +0.038] | -0.037 [-0.125, +0.043] | -0.029 [-0.116, +0.052] | -0.038 [-0.122, +0.040] | -0.037 [-0.122, +0.042] | -0.045 [-0.129, +0.030] |

GT SR:

| dominio | passo | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.215 [-0.287, -0.147] | -0.073 [-0.127, -0.017] | -0.098 [-0.162, -0.035] | -0.189 [-0.258, -0.122] | -0.235 [-0.312, -0.164] | -0.214 [-0.286, -0.146] | -0.239 [-0.317, -0.169] | -0.218 [-0.291, -0.147] | -0.252 [-0.325, -0.186] |
| hifi3d | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.014 [-0.043, +0.021] | -0.071 [-0.106, -0.036] | -0.092 [-0.122, -0.060] | -0.005 [-0.032, +0.027] | -0.004 [-0.034, +0.032] | -0.028 [-0.059, +0.008] | -0.005 [-0.034, +0.031] | -0.011 [-0.038, +0.020] | - |
| hifi3d | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.294 [-0.390, -0.203] | -0.365 [-0.428, -0.307] | -0.366 [-0.425, -0.305] | -0.258 [-0.351, -0.166] | -0.322 [-0.416, -0.232] | -0.284 [-0.383, -0.192] | -0.310 [-0.409, -0.220] | -0.312 [-0.408, -0.222] | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.027 [-0.022, +0.080] | +0.231 [+0.168, +0.297] | +0.162 [+0.114, +0.217] | +0.052 [+0.003, +0.108] | -0.011 [-0.063, +0.043] | +0.049 [+0.005, +0.099] | -0.008 [-0.057, +0.046] | +0.013 [-0.038, +0.069] | -0.026 [-0.078, +0.029] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.017 [-0.033, +0.069] | +0.227 [+0.167, +0.292] | +0.161 [+0.108, +0.216] | +0.044 [-0.006, +0.098] | -0.023 [-0.077, +0.030] | +0.040 [-0.012, +0.091] | -0.013 [-0.064, +0.040] | +0.000 [-0.050, +0.053] | -0.037 [-0.090, +0.017] |
| hifi3d | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.041 [-0.005, +0.091] | +0.210 [+0.146, +0.269] | +0.132 [+0.088, +0.176] | +0.073 [+0.025, +0.122] | +0.009 [-0.040, +0.059] | +0.068 [+0.023, +0.112] | +0.012 [-0.035, +0.061] | +0.010 [-0.040, +0.062] | -0.001 [-0.050, +0.050] |
| hifi3d | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.033 [-0.027, +0.088] | +0.175 [+0.110, +0.240] | +0.116 [+0.059, +0.166] | +0.061 [-0.000, +0.118] | -0.001 [-0.063, +0.055] | +0.059 [+0.005, +0.112] | +0.002 [-0.059, +0.057] | +0.005 [-0.054, +0.063] | -0.014 [-0.076, +0.044] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.012 [-0.045, +0.072] | +0.159 [+0.088, +0.236] | +0.070 [+0.012, +0.135] | +0.047 [-0.010, +0.108] | -0.015 [-0.077, +0.048] | +0.021 [-0.034, +0.081] | -0.012 [-0.071, +0.049] | +0.001 [-0.056, +0.062] | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.003 [-0.056, +0.064] | +0.156 [+0.082, +0.228] | +0.068 [+0.005, +0.131] | +0.039 [-0.019, +0.100] | -0.027 [-0.087, +0.036] | +0.012 [-0.046, +0.074] | -0.018 [-0.079, +0.046] | -0.011 [-0.070, +0.051] | - |
| hifi3d | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.027 [-0.029, +0.083] | +0.139 [+0.068, +0.205] | +0.040 [-0.016, +0.095] | +0.068 [+0.014, +0.125] | +0.005 [-0.053, +0.064] | +0.040 [-0.013, +0.096] | +0.007 [-0.048, +0.067] | -0.001 [-0.058, +0.057] | - |
| hifi3d | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.018 [-0.046, +0.082] | +0.103 [+0.027, +0.177] | +0.023 [-0.039, +0.086] | +0.056 [-0.008, +0.120] | -0.005 [-0.072, +0.061] | +0.031 [-0.031, +0.093] | -0.003 [-0.069, +0.063] | -0.006 [-0.070, +0.058] | - |
| facescape | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.001 [-0.011, +0.009] | +0.037 [+0.008, +0.066] | +0.034 [+0.021, +0.047] | +0.011 [+0.001, +0.021] | -0.001 [-0.012, +0.011] | -0.011 [-0.022, -0.000] | -0.002 [-0.014, +0.009] | +0.004 [-0.007, +0.014] | -0.052 [-0.067, -0.038] |
| facescape | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.080 [+0.055, +0.106] | -0.005 [-0.040, +0.033] | +0.012 [-0.014, +0.041] | +0.073 [+0.049, +0.097] | +0.084 [+0.055, +0.114] | +0.086 [+0.062, +0.109] | +0.085 [+0.057, +0.115] | +0.078 [+0.053, +0.103] | - |
| facescape | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | +0.038 [-0.052, +0.129] | -0.213 [-0.283, -0.136] | -0.144 [-0.214, -0.073] | +0.038 [-0.050, +0.129] | -0.071 [-0.166, +0.026] | +0.118 [+0.029, +0.205] | -0.043 [-0.140, +0.057] | +0.016 [-0.073, +0.109] | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.376 [+0.323, +0.429] | +0.562 [+0.512, +0.612] | +0.478 [+0.427, +0.527] | +0.390 [+0.335, +0.444] | +0.276 [+0.222, +0.332] | +0.427 [+0.383, +0.472] | +0.299 [+0.245, +0.356] | +0.348 [+0.292, +0.403] | +0.196 [+0.138, +0.253] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.343 [+0.280, +0.402] | +0.538 [+0.477, +0.596] | +0.452 [+0.391, +0.507] | +0.359 [+0.296, +0.419] | +0.242 [+0.181, +0.301] | +0.395 [+0.339, +0.444] | +0.268 [+0.208, +0.329] | +0.309 [+0.243, +0.370] | +0.164 [+0.097, +0.225] |
| facescape | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.269 [+0.217, +0.317] | +0.247 [+0.184, +0.310] | +0.121 [+0.078, +0.161] | +0.294 [+0.238, +0.343] | +0.171 [+0.121, +0.221] | +0.320 [+0.273, +0.362] | +0.196 [+0.141, +0.247] | +0.248 [+0.195, +0.298] | +0.082 [+0.029, +0.135] |
| facescape | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.275 [+0.222, +0.328] | +0.205 [+0.137, +0.277] | +0.094 [+0.051, +0.138] | +0.296 [+0.241, +0.352] | +0.183 [+0.128, +0.236] | +0.335 [+0.288, +0.380] | +0.207 [+0.152, +0.263] | +0.233 [+0.176, +0.288] | +0.085 [+0.025, +0.145] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.456 [+0.392, +0.519] | +0.556 [+0.494, +0.617] | +0.490 [+0.428, +0.551] | +0.463 [+0.399, +0.527] | +0.361 [+0.289, +0.430] | +0.512 [+0.457, +0.567] | +0.384 [+0.314, +0.451] | +0.427 [+0.362, +0.492] | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.422 [+0.350, +0.490] | +0.533 [+0.467, +0.599] | +0.464 [+0.395, +0.530] | +0.432 [+0.359, +0.500] | +0.326 [+0.246, +0.401] | +0.481 [+0.417, +0.540] | +0.353 [+0.276, +0.425] | +0.387 [+0.311, +0.456] | - |
| facescape | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.349 [+0.291, +0.401] | +0.242 [+0.169, +0.313] | +0.133 [+0.082, +0.185] | +0.367 [+0.306, +0.422] | +0.256 [+0.195, +0.311] | +0.405 [+0.351, +0.456] | +0.281 [+0.219, +0.338] | +0.326 [+0.266, +0.381] | - |
| facescape | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.355 [+0.290, +0.412] | +0.200 [+0.122, +0.283] | +0.107 [+0.052, +0.161] | +0.369 [+0.303, +0.430] | +0.267 [+0.200, +0.331] | +0.421 [+0.365, +0.473] | +0.292 [+0.225, +0.356] | +0.311 [+0.243, +0.371] | - |
| faceverse | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.004 [-0.040, +0.050] | -0.037 [-0.077, +0.005] | -0.003 [-0.042, +0.038] | -0.003 [-0.046, +0.042] | +0.005 [-0.036, +0.050] | +0.008 [-0.037, +0.054] | +0.003 [-0.040, +0.049] | +0.003 [-0.044, +0.053] | +0.007 [-0.035, +0.052] |
| faceverse | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.071 [+0.016, +0.124] | +0.030 [-0.015, +0.073] | +0.048 [-0.003, +0.094] | +0.086 [+0.031, +0.137] | +0.074 [+0.018, +0.127] | +0.054 [-0.003, +0.107] | +0.076 [+0.019, +0.130] | +0.065 [+0.007, +0.119] | - |
| faceverse | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.088 [-0.191, +0.022] | -0.136 [-0.218, -0.045] | -0.107 [-0.202, -0.014] | -0.078 [-0.177, +0.027] | -0.083 [-0.190, +0.031] | -0.111 [-0.224, +0.008] | -0.071 [-0.182, +0.043] | -0.101 [-0.205, +0.009] | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.021 [-0.039, +0.079] | +0.028 [-0.031, +0.091] | +0.034 [-0.025, +0.086] | +0.003 [-0.055, +0.060] | +0.017 [-0.045, +0.077] | +0.045 [-0.021, +0.108] | +0.014 [-0.045, +0.069] | +0.026 [-0.035, +0.085] | +0.014 [-0.043, +0.069] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | -0.013 [-0.089, +0.057] | -0.010 [-0.080, +0.062] | -0.002 [-0.074, +0.067] | -0.030 [-0.104, +0.039] | -0.017 [-0.097, +0.056] | +0.012 [-0.069, +0.090] | -0.020 [-0.089, +0.048] | -0.015 [-0.094, +0.056] | -0.019 [-0.091, +0.051] |
| faceverse | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | -0.020 [-0.093, +0.055] | -0.036 [-0.106, +0.042] | -0.016 [-0.085, +0.053] | -0.039 [-0.108, +0.039] | -0.014 [-0.091, +0.062] | -0.017 [-0.091, +0.060] | -0.017 [-0.089, +0.057] | -0.014 [-0.089, +0.067] | -0.020 [-0.090, +0.054] |
| faceverse | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.007 [-0.065, +0.078] | -0.031 [-0.099, +0.038] | +0.003 [-0.063, +0.067] | -0.009 [-0.079, +0.060] | +0.010 [-0.060, +0.080] | +0.010 [-0.069, +0.088] | +0.004 [-0.071, +0.078] | +0.017 [-0.056, +0.089] | -0.004 [-0.073, +0.065] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.092 [+0.018, +0.161] | +0.058 [-0.006, +0.122] | +0.081 [+0.012, +0.145] | +0.089 [+0.012, +0.159] | +0.090 [+0.017, +0.160] | +0.099 [+0.023, +0.170] | +0.090 [+0.016, +0.158] | +0.091 [+0.016, +0.162] | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.057 [-0.021, +0.135] | +0.020 [-0.054, +0.097] | +0.046 [-0.031, +0.122] | +0.056 [-0.026, +0.136] | +0.057 [-0.023, +0.136] | +0.066 [-0.014, +0.152] | +0.055 [-0.021, +0.131] | +0.050 [-0.031, +0.130] | - |
| faceverse | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.051 [-0.027, +0.124] | -0.006 [-0.076, +0.063] | +0.032 [-0.037, +0.102] | +0.047 [-0.029, +0.115] | +0.059 [-0.022, +0.135] | +0.037 [-0.042, +0.115] | +0.058 [-0.020, +0.135] | +0.052 [-0.025, +0.125] | - |
| faceverse | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.078 [-0.005, +0.152] | -0.001 [-0.077, +0.069] | +0.051 [-0.027, +0.121] | +0.077 [-0.002, +0.150] | +0.084 [-0.004, +0.161] | +0.064 [-0.026, +0.146] | +0.080 [-0.009, +0.162] | +0.082 [+0.000, +0.159] | - |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.014 [-0.058, +0.030] | -0.057 [-0.100, -0.017] | -0.019 [-0.059, +0.021] | -0.012 [-0.056, +0.033] | -0.015 [-0.058, +0.030] | -0.014 [-0.058, +0.031] | -0.014 [-0.057, +0.031] | -0.018 [-0.060, +0.026] | -0.012 [-0.055, +0.032] |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.074 [+0.014, +0.133] | +0.034 [-0.024, +0.092] | +0.048 [-0.008, +0.103] | +0.079 [+0.018, +0.139] | +0.078 [+0.018, +0.138] | +0.070 [+0.008, +0.131] | +0.076 [+0.015, +0.134] | +0.069 [+0.012, +0.124] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.002 [-0.057, +0.062] | +0.046 [-0.005, +0.099] | +0.030 [-0.025, +0.084] | -0.000 [-0.060, +0.060] | -0.006 [-0.064, +0.053] | +0.010 [-0.053, +0.073] | -0.001 [-0.059, +0.059] | +0.002 [-0.056, +0.060] | -0.005 [-0.061, +0.054] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | -0.015 [-0.080, +0.054] | +0.015 [-0.043, +0.077] | +0.008 [-0.053, +0.074] | -0.020 [-0.085, +0.050] | -0.022 [-0.088, +0.045] | -0.005 [-0.070, +0.069] | -0.018 [-0.082, +0.050] | -0.013 [-0.079, +0.057] | -0.025 [-0.089, +0.042] |
| faceverse_neutral | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | -0.063 [-0.140, +0.017] | -0.052 [-0.123, +0.025] | -0.047 [-0.117, +0.029] | -0.063 [-0.140, +0.020] | -0.067 [-0.143, +0.014] | -0.058 [-0.133, +0.025] | -0.061 [-0.138, +0.022] | -0.073 [-0.149, +0.007] | -0.058 [-0.135, +0.023] |
| faceverse_neutral | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | -0.025 [-0.096, +0.051] | -0.040 [-0.111, +0.027] | -0.018 [-0.089, +0.051] | -0.027 [-0.098, +0.048] | -0.027 [-0.099, +0.049] | -0.018 [-0.092, +0.059] | -0.027 [-0.100, +0.047] | -0.030 [-0.103, +0.043] | -0.031 [-0.103, +0.041] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.076 [+0.005, +0.152] | +0.080 [+0.015, +0.149] | +0.078 [+0.011, +0.149] | +0.079 [+0.006, +0.154] | +0.072 [-0.001, +0.148] | +0.080 [+0.006, +0.158] | +0.075 [+0.005, +0.150] | +0.071 [+0.000, +0.141] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.059 [-0.025, +0.143] | +0.049 [-0.031, +0.128] | +0.056 [-0.025, +0.139] | +0.059 [-0.028, +0.145] | +0.057 [-0.030, +0.140] | +0.064 [-0.018, +0.152] | +0.057 [-0.027, +0.141] | +0.055 [-0.027, +0.137] | - |
| faceverse_neutral | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.010 [-0.068, +0.095] | -0.017 [-0.093, +0.062] | +0.001 [-0.075, +0.084] | +0.016 [-0.063, +0.098] | +0.012 [-0.067, +0.095] | +0.012 [-0.070, +0.096] | +0.015 [-0.067, +0.099] | -0.004 [-0.078, +0.074] | - |
| faceverse_neutral | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.049 [-0.041, +0.133] | -0.006 [-0.092, +0.078] | +0.030 [-0.058, +0.112] | +0.052 [-0.037, +0.137] | +0.051 [-0.039, +0.138] | +0.051 [-0.041, +0.137] | +0.049 [-0.042, +0.133] | +0.038 [-0.048, +0.120] | - |

### Confronti secondari (2b e 3 confusi dalla taglia; 4 e 5 con SR contro un riferimento con la taglia)

GT FR:

| dominio | passo | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.148 [+0.071, +0.231] | +0.002 [-0.069, +0.074] | +0.094 [+0.020, +0.173] | +0.162 [+0.089, +0.242] | +0.153 [+0.073, +0.238] | +0.135 [+0.057, +0.219] | +0.155 [+0.073, +0.242] | +0.150 [+0.074, +0.230] |
| hifi3d | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | +0.119 [+0.002, +0.233] | -0.079 [-0.189, +0.041] | +0.051 [-0.067, +0.166] | +0.141 [+0.028, +0.257] | +0.101 [-0.018, +0.215] | +0.121 [+0.000, +0.237] | +0.108 [-0.010, +0.219] | +0.104 [-0.015, +0.221] |
| facescape | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.079 [+0.049, +0.109] | -0.031 [-0.071, +0.007] | +0.034 [+0.003, +0.068] | +0.083 [+0.054, +0.113] | +0.085 [+0.048, +0.121] | +0.076 [+0.051, +0.101] | +0.084 [+0.050, +0.119] | +0.083 [+0.052, +0.114] |
| facescape | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | +0.168 [+0.076, +0.260] | -0.254 [-0.324, -0.182] | -0.085 [-0.156, -0.014] | +0.196 [+0.104, +0.284] | +0.094 [-0.004, +0.192] | +0.189 [+0.109, +0.271] | +0.117 [+0.020, +0.214] | +0.126 [+0.032, +0.220] |
| faceverse | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.136 [+0.066, +0.209] | +0.046 [-0.014, +0.107] | +0.103 [+0.038, +0.169] | +0.147 [+0.077, +0.219] | +0.141 [+0.070, +0.213] | +0.124 [+0.052, +0.198] | +0.138 [+0.064, +0.210] | +0.132 [+0.060, +0.204] |
| faceverse | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | +0.011 [-0.115, +0.135] | -0.101 [-0.207, -0.001] | -0.032 [-0.144, +0.080] | +0.011 [-0.124, +0.139] | +0.028 [-0.096, +0.155] | +0.003 [-0.127, +0.128] | +0.012 [-0.115, +0.133] | -0.002 [-0.128, +0.126] |
| faceverse_neutral | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.131 [+0.049, +0.212] | +0.022 [-0.045, +0.084] | +0.090 [+0.014, +0.162] | +0.137 [+0.055, +0.218] | +0.135 [+0.051, +0.216] | +0.129 [+0.045, +0.215] | +0.132 [+0.051, +0.214] | +0.123 [+0.044, +0.200] |
| faceverse_neutral | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.054 [-0.185, +0.072] | -0.133 [-0.239, -0.029] | -0.088 [-0.205, +0.029] | -0.054 [-0.183, +0.074] | -0.055 [-0.188, +0.072] | -0.045 [-0.175, +0.084] | -0.056 [-0.188, +0.071] | -0.063 [-0.191, +0.062] |

GT SR:

| dominio | passo | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | -0.138 [-0.214, -0.058] | -0.150 [-0.217, -0.083] | -0.141 [-0.213, -0.068] | -0.110 [-0.184, -0.033] | -0.149 [-0.231, -0.067] | -0.155 [-0.233, -0.075] | -0.147 [-0.227, -0.063] | -0.126 [-0.201, -0.048] | - |
| hifi3d | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.420 [-0.526, -0.313] | -0.370 [-0.458, -0.278] | -0.401 [-0.498, -0.301] | -0.394 [-0.498, -0.287] | -0.448 [-0.555, -0.340] | -0.417 [-0.522, -0.307] | -0.436 [-0.545, -0.329] | -0.424 [-0.528, -0.316] | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.242 [+0.165, +0.323] | +0.304 [+0.239, +0.375] | +0.260 [+0.190, +0.333] | +0.241 [+0.164, +0.322] | +0.224 [+0.145, +0.308] | +0.263 [+0.189, +0.341] | +0.232 [+0.151, +0.315] | +0.230 [+0.153, +0.313] | +0.226 [+0.150, +0.306] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.232 [+0.158, +0.310] | +0.300 [+0.237, +0.370] | +0.259 [+0.193, +0.329] | +0.233 [+0.159, +0.311] | +0.212 [+0.136, +0.292] | +0.254 [+0.182, +0.329] | +0.226 [+0.149, +0.307] | +0.218 [+0.144, +0.297] | +0.215 [+0.143, +0.291] |
| hifi3d | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.256 [+0.181, +0.332] | +0.283 [+0.217, +0.348] | +0.231 [+0.160, +0.300] | +0.262 [+0.190, +0.336] | +0.244 [+0.167, +0.322] | +0.282 [+0.208, +0.355] | +0.251 [+0.172, +0.333] | +0.228 [+0.150, +0.305] | +0.251 [+0.176, +0.325] |
| hifi3d | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.248 [+0.170, +0.333] | +0.247 [+0.175, +0.319] | +0.214 [+0.138, +0.290] | +0.250 [+0.171, +0.334] | +0.234 [+0.153, +0.322] | +0.273 [+0.199, +0.354] | +0.241 [+0.160, +0.330] | +0.223 [+0.139, +0.307] | +0.238 [+0.159, +0.323] |
| facescape | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.071 [+0.043, +0.100] | -0.042 [-0.076, -0.004] | +0.026 [-0.001, +0.057] | +0.074 [+0.048, +0.101] | +0.074 [+0.042, +0.109] | +0.069 [+0.046, +0.093] | +0.074 [+0.043, +0.107] | +0.073 [+0.045, +0.103] | - |
| facescape | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.007 [-0.084, +0.074] | -0.300 [-0.364, -0.233] | -0.181 [-0.239, -0.116] | +0.016 [-0.065, +0.097] | -0.094 [-0.177, -0.010] | +0.037 [-0.030, +0.109] | -0.071 [-0.155, +0.019] | -0.054 [-0.131, +0.031] | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.377 [+0.324, +0.430] | +0.525 [+0.473, +0.569] | +0.444 [+0.390, +0.494] | +0.379 [+0.323, +0.433] | +0.277 [+0.222, +0.335] | +0.438 [+0.391, +0.484] | +0.302 [+0.245, +0.358] | +0.345 [+0.287, +0.401] | +0.247 [+0.189, +0.305] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.343 [+0.280, +0.402] | +0.502 [+0.438, +0.553] | +0.418 [+0.355, +0.473] | +0.348 [+0.283, +0.408] | +0.242 [+0.178, +0.303] | +0.406 [+0.348, +0.456] | +0.270 [+0.208, +0.330] | +0.305 [+0.237, +0.364] | +0.216 [+0.145, +0.279] |
| facescape | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.269 [+0.217, +0.317] | +0.210 [+0.143, +0.276] | +0.087 [+0.041, +0.130] | +0.283 [+0.230, +0.331] | +0.172 [+0.118, +0.222] | +0.331 [+0.281, +0.375] | +0.198 [+0.143, +0.249] | +0.244 [+0.191, +0.294] | +0.134 [+0.077, +0.186] |
| facescape | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.276 [+0.221, +0.328] | +0.168 [+0.096, +0.240] | +0.060 [+0.013, +0.104] | +0.285 [+0.229, +0.340] | +0.183 [+0.128, +0.236] | +0.347 [+0.298, +0.394] | +0.210 [+0.153, +0.263] | +0.229 [+0.172, +0.283] | +0.137 [+0.075, +0.195] |
| faceverse | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.086 [+0.022, +0.152] | +0.003 [-0.053, +0.057] | +0.055 [-0.002, +0.113] | +0.093 [+0.032, +0.158] | +0.091 [+0.025, +0.159] | +0.077 [+0.007, +0.149] | +0.088 [+0.021, +0.154] | +0.083 [+0.015, +0.148] | - |
| faceverse | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.072 [-0.194, +0.051] | -0.163 [-0.264, -0.061] | -0.107 [-0.217, +0.005] | -0.076 [-0.203, +0.048] | -0.058 [-0.182, +0.067] | -0.076 [-0.202, +0.051] | -0.072 [-0.194, +0.045] | -0.078 [-0.200, +0.045] | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.018 [-0.050, +0.083] | +0.065 [+0.002, +0.126] | +0.036 [-0.030, +0.096] | +0.006 [-0.057, +0.071] | +0.011 [-0.058, +0.079] | +0.037 [-0.037, +0.109] | +0.011 [-0.058, +0.075] | +0.023 [-0.048, +0.093] | +0.007 [-0.059, +0.070] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | -0.017 [-0.092, +0.063] | +0.027 [-0.047, +0.104] | +0.001 [-0.073, +0.078] | -0.027 [-0.104, +0.051] | -0.022 [-0.104, +0.062] | +0.004 [-0.079, +0.085] | -0.023 [-0.097, +0.056] | -0.018 [-0.095, +0.066] | -0.025 [-0.102, +0.053] |
| faceverse | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | -0.023 [-0.097, +0.056] | +0.001 [-0.070, +0.073] | -0.014 [-0.082, +0.058] | -0.037 [-0.111, +0.040] | -0.020 [-0.094, +0.057] | -0.025 [-0.105, +0.060] | -0.020 [-0.096, +0.057] | -0.016 [-0.089, +0.062] | -0.027 [-0.100, +0.049] |
| faceverse | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.004 [-0.080, +0.079] | +0.006 [-0.070, +0.073] | +0.006 [-0.071, +0.075] | -0.007 [-0.085, +0.065] | +0.005 [-0.074, +0.079] | +0.002 [-0.089, +0.083] | +0.002 [-0.085, +0.078] | +0.014 [-0.069, +0.090] | -0.011 [-0.092, +0.062] |
| faceverse_neutral | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.069 [-0.002, +0.138] | -0.024 [-0.084, +0.038] | +0.034 [-0.032, +0.101] | +0.075 [+0.004, +0.144] | +0.072 [-0.001, +0.141] | +0.066 [-0.008, +0.140] | +0.070 [-0.003, +0.141] | +0.063 [-0.004, +0.129] | - |
| faceverse_neutral | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.158 [-0.273, -0.039] | -0.201 [-0.299, -0.100] | -0.179 [-0.287, -0.071] | -0.157 [-0.272, -0.038] | -0.160 [-0.277, -0.039] | -0.149 [-0.266, -0.027] | -0.162 [-0.278, -0.041] | -0.165 [-0.281, -0.048] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.016 [-0.050, +0.078] | +0.103 [+0.049, +0.156] | +0.049 [-0.012, +0.108] | +0.011 [-0.052, +0.073] | +0.009 [-0.057, +0.073] | +0.024 [-0.045, +0.090] | +0.013 [-0.051, +0.075] | +0.020 [-0.045, +0.082] | +0.007 [-0.054, +0.068] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | -0.001 [-0.077, +0.078] | +0.072 [-0.001, +0.146] | +0.027 [-0.047, +0.103] | -0.008 [-0.085, +0.071] | -0.007 [-0.082, +0.070] | +0.008 [-0.070, +0.091] | -0.005 [-0.081, +0.073] | +0.004 [-0.074, +0.084] | -0.014 [-0.090, +0.064] |
| faceverse_neutral | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | -0.049 [-0.127, +0.038] | +0.005 [-0.058, +0.079] | -0.028 [-0.100, +0.053] | -0.051 [-0.128, +0.038] | -0.052 [-0.128, +0.035] | -0.044 [-0.121, +0.043] | -0.047 [-0.125, +0.042] | -0.055 [-0.132, +0.031] | -0.046 [-0.123, +0.040] |
| faceverse_neutral | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | -0.011 [-0.094, +0.065] | +0.017 [-0.054, +0.085] | +0.001 [-0.075, +0.073] | -0.015 [-0.097, +0.061] | -0.012 [-0.097, +0.063] | -0.005 [-0.089, +0.074] | -0.013 [-0.097, +0.062] | -0.013 [-0.094, +0.065] | -0.019 [-0.098, +0.054] |

## Conteggio

Per ogni passo e GT, in nocrop, crop e all_cross: numero di famiglie sintetiche (HIFI3D, FaceScape, FaceVerse con espressioni, FaceVerse neutra) con IC 95% appaiato che esclude lo 0 nel verso "piu' allineamento = peggio" / nel verso opposto / non risolte. Lato piu' allineato: b nei passi 1-3 (cs: taglia tolta per mesh; NICP: deformazione; NICP per coppia: deformazione specifica della coppia contro l'iscrizione su un template comune), a nel 4 (fit del modello per mesh contro la rigida), b nel 5 (ICP: allineamento esplicito; il braccio non allinea). Le righe 4 e 5 contano ogni modello / seme a parte.

| GT | passo | ruolo | nocrop: peggio / meglio / n.r. | crop: peggio / meglio / n.r. | all_cross: peggio / meglio / n.r. |
| --- | --- | --- | --- | --- | --- |
| FR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |
| FR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 3 / 0 / 1 (su 4) | 0 / 1 / 3 (su 4) | 2 / 1 / 1 (su 4) |
| FR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 4 / 0 / 0 (su 4) | 0 / 0 / 4 (su 4) | 4 / 0 / 0 (su 4) |
| FR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 2 / 0 / 2 (su 4) | 0 / 3 / 1 (su 4) | 0 / 1 / 3 (su 4) |
| FR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 1 / 2 (su 3) | 0 / 3 / 0 (su 3) | 0 / 2 / 1 (su 3) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | principale | 1 / 1 / 2 (su 4) | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | principale | 1 / 1 / 2 (su 4) | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | principale | 1 / 0 / 3 (su 4) | 2 / 0 / 2 (su 4) | 1 / 0 / 3 (su 4) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | principale | 1 / 0 / 3 (su 4) | 2 / 0 / 2 (su 4) | 0 / 0 / 4 (su 4) |
| SR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 0 / 1 / 3 (su 4) | 1 / 2 / 1 (su 4) | 1 / 1 / 2 (su 4) |
| SR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 3 / 0 / 1 (su 4) | 0 / 1 / 3 (su 4) | 0 / 1 / 3 (su 4) |
| SR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 2 / 1 / 1 (su 4) | 0 / 2 / 2 (su 4) | 0 / 1 / 3 (su 4) |
| SR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 0 / 2 / 2 (su 4) | 0 / 4 / 0 (su 4) | 0 / 3 / 1 (su 4) |
| SR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 1 / 2 (su 3) | 0 / 3 / 0 (su 3) | 0 / 3 / 0 (su 3) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 1 / 3 (su 4) | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 1 / 3 (su 4) | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | principale | 1 / 0 / 3 (su 4) | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | principale | 1 / 0 / 3 (su 4) | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 3 / 1 (su 4) | 0 / 3 / 1 (su 4) | 0 / 4 / 0 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 1 / 3 (su 4) | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | principale | 1 / 0 / 3 (su 4) | 2 / 0 / 2 (su 4) | 1 / 0 / 3 (su 4) |
| SR | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | principale | 1 / 0 / 3 (su 4) | 2 / 0 / 2 (su 4) | 1 / 0 / 3 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | secondario | 0 / 2 / 2 (su 4) | 0 / 4 / 0 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | secondario | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | secondario | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | secondario | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |

## Fallimenti del NICP per coppia: selezione delle righe

La maschera comune toglie le righe dove il NICP per coppia (mm o cs) fallisce (matrice singolare). La selezione dipende dalla topologia, non dalla GT: le righe perse sono soprattutto quelle con up60k, che quindi pesa meno negli insiemi nocrop e all_cross; e il NICP non e' penalizzato per i propri fallimenti. Righe perse per topologia (righe con quella topologia su un lato, senza la maschera del NICP):

| dominio | righe perse | original | remesh | down8k | up60k | noisy | crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 970 | 194 / 49500 (0.4%) | 194 / 49500 (0.4%) | 194 / 49500 (0.4%) | 970 / 49500 (2.0%) | 194 / 49500 (0.4%) | 194 / 49500 (0.4%) |
| facescape | 12845 | 2569 / 49500 (5.2%) | 2569 / 49500 (5.2%) | 2569 / 49500 (5.2%) | 12845 / 49500 (25.9%) | 2569 / 49500 (5.2%) | 2569 / 49500 (5.2%) |
| faceverse | 0 | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) |
| faceverse_neutral | 0 | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) | 0 / 49500 (0.0%) |

Sensibilita' (`passo` = "sensibilita': NICP fallito = distanza massima" in entrambi i CSV): stesse colonne, maschera senza il NICP (le righe perse rientrano per tutti), NICP fallito = distanza massima della colonna sul dominio (il fallimento conta come la coppia piu' lontana). Il NICP con e senza la sostituzione e i passi della scala che lo usano:

| dominio | GT | insieme | metodo / passo | maschera comune | fallimento = distanza massima |
| --- | --- | --- | --- | --- | --- |
| hifi3d | FR | nocrop | NICP per coppia in mm | 0.495 [0.423, 0.569] (98224) | 0.486 [0.415, 0.557] (99000) |
| hifi3d | FR | nocrop | NICP per coppia cs | 0.376 [0.296, 0.465] (98224) | 0.368 [0.290, 0.457] (99000) |
| hifi3d | FR | nocrop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.017 [-0.046, +0.012] | -0.009 [-0.036, +0.020] |
| hifi3d | FR | nocrop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.187 [-0.275, -0.105] | -0.179 [-0.266, -0.097] |
| hifi3d | FR | crop | NICP per coppia in mm | 0.416 [0.350, 0.484] (49306) | 0.412 [0.347, 0.480] (49500) |
| hifi3d | FR | crop | NICP per coppia cs | 0.257 [0.188, 0.333] (49306) | 0.254 [0.187, 0.329] (49500) |
| hifi3d | FR | crop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.044 [-0.075, -0.014] | -0.041 [-0.072, -0.012] |
| hifi3d | FR | crop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.227 [-0.301, -0.161] | -0.225 [-0.298, -0.160] |
| hifi3d | SR | nocrop | NICP per coppia in mm | 0.503 [0.437, 0.574] (98224) | 0.494 [0.429, 0.563] (99000) |
| hifi3d | SR | nocrop | NICP per coppia cs | 0.595 [0.546, 0.648] (98224) | 0.585 [0.535, 0.637] (99000) |
| hifi3d | SR | nocrop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.014 [-0.043, +0.021] | -0.004 [-0.032, +0.029] |
| hifi3d | SR | nocrop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.294 [-0.390, -0.203] | -0.283 [-0.381, -0.192] |
| hifi3d | SR | crop | NICP per coppia in mm | 0.417 [0.357, 0.481] (49306) | 0.413 [0.355, 0.478] (49500) |
| hifi3d | SR | crop | NICP per coppia cs | 0.411 [0.363, 0.460] (49306) | 0.407 [0.360, 0.456] (49500) |
| hifi3d | SR | crop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.071 [-0.106, -0.036] | -0.068 [-0.103, -0.033] |
| hifi3d | SR | crop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.365 [-0.428, -0.307] | -0.362 [-0.425, -0.304] |
| facescape | FR | nocrop | NICP per coppia in mm | 0.374 [0.305, 0.446] (88724) | 0.297 [0.240, 0.356] (99000) |
| facescape | FR | nocrop | NICP per coppia cs | 0.346 [0.276, 0.419] (88724) | 0.274 [0.216, 0.337] (99000) |
| facescape | FR | nocrop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.069 [+0.043, +0.095] | +0.155 [+0.127, +0.180] |
| facescape | FR | nocrop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | +0.054 [-0.032, +0.136] | +0.130 [+0.050, +0.210] |
| facescape | FR | crop | NICP per coppia in mm | 0.324 [0.261, 0.387] (46931) | 0.289 [0.232, 0.345] (49500) |
| facescape | FR | crop | NICP per coppia cs | 0.228 [0.167, 0.291] (46931) | 0.203 [0.147, 0.262] (49500) |
| facescape | FR | crop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.004 [-0.036, +0.031] | +0.018 [-0.011, +0.048] |
| facescape | FR | crop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.175 [-0.245, -0.100] | -0.149 [-0.216, -0.081] |
| facescape | SR | nocrop | NICP per coppia in mm | 0.406 [0.336, 0.472] (88724) | 0.322 [0.265, 0.378] (99000) |
| facescape | SR | nocrop | NICP per coppia cs | 0.398 [0.329, 0.464] (88724) | 0.316 [0.260, 0.369] (99000) |
| facescape | SR | nocrop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.080 [+0.055, +0.106] | +0.177 [+0.152, +0.200] |
| facescape | SR | nocrop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | +0.038 [-0.052, +0.129] | +0.123 [+0.040, +0.210] |
| facescape | SR | crop | NICP per coppia in mm | 0.348 [0.282, 0.409] (46931) | 0.310 [0.251, 0.364] (49500) |
| facescape | SR | crop | NICP per coppia cs | 0.274 [0.213, 0.331] (46931) | 0.244 [0.189, 0.295] (49500) |
| facescape | SR | crop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.005 [-0.040, +0.033] | +0.020 [-0.012, +0.054] |
| facescape | SR | crop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.213 [-0.283, -0.136] | -0.183 [-0.246, -0.112] |

## Tabelle per dominio (famiglie sintetiche)

### HIFI3D: Spearman per insieme di righe (seme 990708)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (98224) | crop (49306) | all_cross (147530) | con_perturbazione original (39406) | con_perturbazione remesh (39406) | con_perturbazione down8k (39406) | con_perturbazione up60k (38824) | con_perturbazione noisy (39406) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.749 [0.673, 0.806] | 0.663 [0.589, 0.721] | 0.709 [0.634, 0.767] | 0.754 [0.679, 0.811] | 0.741 [0.668, 0.798] | 0.752 [0.678, 0.808] | 0.751 [0.674, 0.808] | 0.747 [0.670, 0.806] | 0.762 [0.688, 0.819] |
| **factorized s1234, d_F non cal.** | 0.642 [0.570, 0.706] | 0.563 [0.485, 0.630] | 0.584 [0.512, 0.647] | 0.649 [0.577, 0.712] | 0.642 [0.571, 0.705] | 0.644 [0.572, 0.707] | 0.646 [0.573, 0.710] | 0.633 [0.560, 0.698] | 0.648 [0.580, 0.712] |
| factorized s1234, d_P | 0.426 [0.333, 0.510] | 0.376 [0.281, 0.457] | 0.383 [0.291, 0.464] | 0.432 [0.339, 0.517] | 0.429 [0.337, 0.514] | 0.427 [0.335, 0.510] | 0.429 [0.335, 0.513] | 0.413 [0.317, 0.501] | 0.429 [0.337, 0.513] |
| **factorized s2345, d_F cal.*** | 0.731 [0.657, 0.792] | 0.552 [0.482, 0.614] | 0.661 [0.587, 0.722] | 0.737 [0.663, 0.799] | 0.717 [0.645, 0.778] | 0.738 [0.663, 0.798] | 0.731 [0.659, 0.793] | 0.731 [0.657, 0.793] | 0.746 [0.672, 0.807] |
| **factorized s2345, d_F non cal.** | 0.631 [0.560, 0.697] | 0.516 [0.436, 0.584] | 0.561 [0.486, 0.627] | 0.636 [0.565, 0.701] | 0.629 [0.559, 0.694] | 0.634 [0.564, 0.700] | 0.634 [0.562, 0.699] | 0.623 [0.551, 0.689] | 0.637 [0.566, 0.702] |
| factorized s2345, d_P | 0.417 [0.315, 0.512] | 0.347 [0.251, 0.436] | 0.370 [0.276, 0.457] | 0.421 [0.319, 0.517] | 0.419 [0.317, 0.513] | 0.420 [0.318, 0.515] | 0.418 [0.315, 0.513] | 0.409 [0.306, 0.501] | 0.418 [0.314, 0.513] |
| **ctrlfr s1234** | 0.757 [0.685, 0.817] | 0.236 [0.186, 0.281] | 0.515 [0.444, 0.577] | 0.767 [0.696, 0.828] | 0.747 [0.673, 0.805] | 0.766 [0.693, 0.827] | 0.755 [0.680, 0.815] | 0.753 [0.683, 0.814] | 0.780 [0.710, 0.839] |
| **ctrlfr s2345** | 0.746 [0.670, 0.809] | 0.279 [0.223, 0.329] | 0.529 [0.455, 0.593] | 0.757 [0.682, 0.820] | 0.732 [0.656, 0.795] | 0.755 [0.680, 0.817] | 0.754 [0.676, 0.817] | 0.733 [0.654, 0.798] | 0.768 [0.694, 0.829] |
| **C3M e123, d_F cal.*** | 0.748 [0.672, 0.808] | 0.506 [0.427, 0.574] | 0.641 [0.563, 0.703] | 0.759 [0.683, 0.817] | 0.732 [0.655, 0.792] | 0.756 [0.680, 0.815] | 0.750 [0.671, 0.809] | 0.747 [0.670, 0.806] | 0.766 [0.689, 0.823] |
| **C3M e123, d_F non cal.** | 0.528 [0.444, 0.606] | 0.413 [0.330, 0.491] | 0.418 [0.343, 0.488] | 0.535 [0.451, 0.611] | 0.531 [0.447, 0.609] | 0.534 [0.452, 0.612] | 0.529 [0.442, 0.606] | 0.519 [0.436, 0.600] | 0.538 [0.457, 0.614] |
| C3M e123, d_P | 0.400 [0.304, 0.492] | 0.316 [0.231, 0.397] | 0.313 [0.229, 0.390] | 0.406 [0.310, 0.496] | 0.404 [0.307, 0.494] | 0.406 [0.310, 0.495] | 0.400 [0.304, 0.491] | 0.390 [0.295, 0.479] | 0.406 [0.311, 0.492] |
| **C3M e205, d_F cal.*** | 0.739 [0.657, 0.801] | 0.472 [0.396, 0.538] | 0.629 [0.549, 0.692] | 0.750 [0.669, 0.810] | 0.725 [0.644, 0.786] | 0.745 [0.663, 0.806] | 0.736 [0.653, 0.799] | 0.744 [0.663, 0.805] | 0.757 [0.676, 0.815] |
| **C3M e205, d_F non cal.** | 0.508 [0.418, 0.589] | 0.405 [0.321, 0.484] | 0.413 [0.334, 0.488] | 0.516 [0.427, 0.598] | 0.511 [0.421, 0.593] | 0.514 [0.423, 0.596] | 0.509 [0.418, 0.589] | 0.499 [0.410, 0.581] | 0.519 [0.434, 0.597] |
| C3M e205, d_P | 0.404 [0.306, 0.496] | 0.329 [0.240, 0.410] | 0.325 [0.241, 0.406] | 0.412 [0.312, 0.505] | 0.408 [0.309, 0.498] | 0.411 [0.312, 0.505] | 0.404 [0.303, 0.497] | 0.393 [0.297, 0.484] | 0.411 [0.315, 0.502] |
| **ICP + Chamfer in mm (per coppia)** | 0.643 [0.572, 0.702] | 0.417 [0.361, 0.474] | 0.556 [0.491, 0.614] | 0.637 [0.565, 0.696] | 0.668 [0.596, 0.729] | 0.621 [0.552, 0.680] | 0.666 [0.595, 0.727] | 0.657 [0.587, 0.715] | 0.647 [0.580, 0.705] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.358 [0.276, 0.438] | 0.213 [0.155, 0.273] | 0.264 [0.198, 0.326] | 0.346 [0.267, 0.423] | 0.379 [0.292, 0.463] | 0.345 [0.264, 0.422] | 0.377 [0.289, 0.460] | 0.368 [0.286, 0.448] | 0.390 [0.304, 0.473] |
| **NICP per coppia in mm** | 0.495 [0.423, 0.569] | 0.416 [0.350, 0.484] | 0.462 [0.392, 0.532] | 0.475 [0.404, 0.545] | 0.515 [0.439, 0.592] | 0.486 [0.417, 0.558] | 0.512 [0.437, 0.588] | 0.508 [0.436, 0.580] | - |
| **NICP per coppia cs** | 0.376 [0.296, 0.465] | 0.257 [0.188, 0.333] | 0.325 [0.253, 0.410] | 0.356 [0.281, 0.443] | 0.391 [0.310, 0.484] | 0.371 [0.291, 0.459] | 0.389 [0.306, 0.482] | 0.382 [0.303, 0.474] | - |
| **NICP su template in mm** | 0.614 [0.506, 0.698] | 0.337 [0.241, 0.422] | 0.513 [0.409, 0.597] | 0.616 [0.508, 0.698] | 0.616 [0.508, 0.699] | 0.608 [0.499, 0.693] | 0.620 [0.512, 0.702] | 0.612 [0.504, 0.696] | 0.623 [0.516, 0.705] |
| **NICP su template cs** | 0.189 [0.103, 0.269] | 0.029 [-0.003, 0.064] | 0.095 [0.052, 0.138] | 0.191 [0.104, 0.271] | 0.191 [0.108, 0.266] | 0.188 [0.100, 0.269] | 0.194 [0.107, 0.276] | 0.181 [0.097, 0.262] | 0.203 [0.112, 0.287] |
| **varifold in mm (massa unitaria)** | 0.494 [0.411, 0.567] | 0.279 [0.225, 0.328] | 0.380 [0.307, 0.445] | 0.550 [0.468, 0.620] | 0.550 [0.468, 0.619] | 0.551 [0.469, 0.621] | 0.551 [0.469, 0.621] | 0.562 [0.477, 0.630] | 0.698 [0.623, 0.761] |
| **GNM (visto) preregistrato, coefficienti** | 0.619 [0.503, 0.710] | 0.443 [0.334, 0.534] | 0.550 [0.440, 0.639] | 0.623 [0.510, 0.713] | 0.618 [0.502, 0.709] | 0.617 [0.501, 0.707] | 0.624 [0.510, 0.715] | 0.612 [0.498, 0.702] | 0.624 [0.511, 0.713] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.608 [0.500, 0.693] | 0.537 [0.426, 0.623] | 0.583 [0.474, 0.667] | 0.612 [0.505, 0.696] | 0.608 [0.499, 0.692] | 0.607 [0.498, 0.692] | 0.610 [0.502, 0.695] | 0.602 [0.492, 0.688] | 0.617 [0.510, 0.701] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.482 [0.369, 0.582] | 0.262 [0.157, 0.349] | 0.378 [0.274, 0.467] | 0.478 [0.363, 0.577] | 0.481 [0.368, 0.582] | 0.480 [0.367, 0.582] | 0.488 [0.375, 0.588] | 0.485 [0.369, 0.583] | 0.456 [0.344, 0.550] |
| **GNM (visto) vA, coefficienti*** | 0.603 [0.492, 0.693] | 0.466 [0.356, 0.561] | 0.541 [0.433, 0.630] | 0.606 [0.496, 0.696] | 0.604 [0.496, 0.693] | 0.601 [0.491, 0.690] | 0.610 [0.499, 0.698] | 0.597 [0.483, 0.686] | 0.597 [0.487, 0.685] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.607 [0.500, 0.692] | 0.544 [0.435, 0.631] | 0.585 [0.476, 0.671] | 0.611 [0.504, 0.697] | 0.607 [0.499, 0.692] | 0.606 [0.497, 0.691] | 0.609 [0.503, 0.694] | 0.601 [0.492, 0.688] | 0.617 [0.509, 0.701] |
| GNM (visto) vA, mesh d'identita' SR* | 0.291 [0.182, 0.389] | 0.158 [0.072, 0.245] | 0.232 [0.140, 0.322] | 0.289 [0.180, 0.392] | 0.293 [0.185, 0.396] | 0.290 [0.179, 0.388] | 0.300 [0.191, 0.398] | 0.284 [0.180, 0.381] | 0.277 [0.175, 0.373] |
| **GNM (visto) vB, coefficienti*** | 0.550 [0.465, 0.628] | 0.524 [0.446, 0.600] | 0.539 [0.456, 0.617] | 0.551 [0.466, 0.630] | 0.551 [0.468, 0.629] | 0.547 [0.462, 0.626] | 0.547 [0.463, 0.626] | 0.554 [0.468, 0.632] | 0.546 [0.463, 0.623] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.664 [0.570, 0.738] | 0.625 [0.531, 0.701] | 0.651 [0.553, 0.724] | 0.668 [0.574, 0.741] | 0.662 [0.567, 0.736] | 0.665 [0.570, 0.738] | 0.667 [0.573, 0.741] | 0.661 [0.566, 0.735] | 0.674 [0.581, 0.747] |
| GNM (visto) vB, mesh d'identita' SR* | 0.411 [0.319, 0.499] | 0.380 [0.284, 0.470] | 0.395 [0.301, 0.481] | 0.412 [0.319, 0.500] | 0.411 [0.317, 0.501] | 0.409 [0.318, 0.497] | 0.410 [0.318, 0.499] | 0.410 [0.319, 0.500] | 0.410 [0.318, 0.498] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.698 [0.612, 0.769] | 0.661 [0.573, 0.733] | 0.683 [0.598, 0.755] | 0.701 [0.616, 0.771] | 0.696 [0.608, 0.768] | 0.698 [0.612, 0.769] | 0.700 [0.615, 0.771] | 0.696 [0.611, 0.767] | 0.705 [0.620, 0.775] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.566 [0.444, 0.663] | 0.401 [0.302, 0.486] | 0.506 [0.396, 0.598] | 0.570 [0.449, 0.670] | 0.557 [0.435, 0.658] | 0.572 [0.447, 0.672] | 0.578 [0.457, 0.675] | 0.554 [0.438, 0.650] | 0.578 [0.457, 0.675] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.639 [0.532, 0.725] | 0.546 [0.438, 0.631] | 0.607 [0.502, 0.692] | 0.644 [0.539, 0.730] | 0.635 [0.522, 0.723] | 0.644 [0.535, 0.730] | 0.645 [0.538, 0.730] | 0.627 [0.518, 0.714] | 0.648 [0.542, 0.733] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.276 [0.178, 0.379] | 0.172 [0.103, 0.239] | 0.236 [0.150, 0.325] | 0.277 [0.173, 0.382] | 0.286 [0.184, 0.393] | 0.279 [0.177, 0.388] | 0.288 [0.185, 0.397] | 0.253 [0.164, 0.344] | 0.278 [0.177, 0.379] |
| **FLAME 2023 Open vA, coefficienti*** | 0.580 [0.460, 0.676] | 0.423 [0.321, 0.511] | 0.524 [0.413, 0.617] | 0.585 [0.464, 0.684] | 0.572 [0.451, 0.671] | 0.586 [0.464, 0.685] | 0.592 [0.472, 0.687] | 0.568 [0.447, 0.663] | 0.592 [0.474, 0.688] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.641 [0.536, 0.725] | 0.565 [0.457, 0.652] | 0.616 [0.512, 0.700] | 0.646 [0.539, 0.730] | 0.639 [0.528, 0.724] | 0.645 [0.539, 0.730] | 0.646 [0.542, 0.729] | 0.630 [0.522, 0.715] | 0.649 [0.542, 0.731] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.272 [0.173, 0.373] | 0.179 [0.106, 0.252] | 0.236 [0.150, 0.325] | 0.274 [0.170, 0.378] | 0.284 [0.184, 0.387] | 0.276 [0.172, 0.383] | 0.283 [0.180, 0.387] | 0.247 [0.158, 0.337] | 0.274 [0.175, 0.371] |
| **FLAME 2023 Open vB, coefficienti*** | 0.585 [0.503, 0.661] | 0.542 [0.460, 0.616] | 0.570 [0.489, 0.646] | 0.582 [0.499, 0.660] | 0.581 [0.500, 0.658] | 0.584 [0.502, 0.661] | 0.589 [0.507, 0.665] | 0.592 [0.509, 0.668] | 0.577 [0.497, 0.652] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.686 [0.597, 0.758] | 0.639 [0.543, 0.713] | 0.670 [0.579, 0.741] | 0.690 [0.601, 0.762] | 0.682 [0.593, 0.755] | 0.690 [0.599, 0.762] | 0.689 [0.600, 0.759] | 0.679 [0.589, 0.754] | 0.693 [0.603, 0.766] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.458 [0.365, 0.542] | 0.433 [0.342, 0.516] | 0.448 [0.355, 0.531] | 0.459 [0.365, 0.543] | 0.461 [0.367, 0.545] | 0.454 [0.359, 0.541] | 0.459 [0.365, 0.544] | 0.459 [0.366, 0.544] | 0.456 [0.364, 0.539] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.712 [0.627, 0.781] | 0.671 [0.582, 0.740] | 0.697 [0.610, 0.766] | 0.715 [0.631, 0.784] | 0.708 [0.622, 0.778] | 0.713 [0.627, 0.783] | 0.715 [0.632, 0.784] | 0.706 [0.621, 0.776] | 0.716 [0.630, 0.786] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.134 [0.042, 0.233] | 0.140 [0.049, 0.239] | 0.135 [0.045, 0.234] | 0.133 [0.042, 0.236] | 0.140 [0.049, 0.239] | 0.138 [0.047, 0.239] | 0.136 [0.045, 0.238] | 0.127 [0.028, 0.237] | 0.114 [0.044, 0.193] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.166 [0.079, 0.261] | 0.182 [0.087, 0.284] | 0.171 [0.081, 0.269] | 0.174 [0.081, 0.272] | 0.175 [0.083, 0.272] | 0.174 [0.079, 0.276] | 0.175 [0.080, 0.276] | 0.148 [0.067, 0.243] | 0.148 [0.073, 0.227] |
| **ShapeDNA k=50** | 0.085 [0.051, 0.118] | 0.009 [-0.020, 0.037] | 0.057 [0.032, 0.083] | 0.143 [0.090, 0.193] | 0.136 [0.086, 0.185] | 0.134 [0.083, 0.183] | 0.121 [0.076, 0.166] | 0.003 [-0.042, 0.049] | 0.250 [0.163, 0.340] |
| **HKS globale** | 0.036 [0.006, 0.070] | -0.008 [-0.020, 0.005] | 0.018 [0.001, 0.037] | 0.059 [0.011, 0.113] | 0.050 [0.009, 0.095] | 0.058 [0.009, 0.111] | 0.052 [0.010, 0.101] | 0.011 [-0.030, 0.053] | 0.104 [0.020, 0.195] |
| **WKS globale** | 0.019 [0.003, 0.036] | 0.034 [0.008, 0.066] | 0.023 [0.006, 0.041] | 0.043 [0.017, 0.071] | 0.044 [0.019, 0.070] | 0.037 [0.012, 0.061] | 0.027 [0.006, 0.049] | -0.004 [-0.053, 0.048] | 0.199 [0.110, 0.290] |
| **OpenShape PointBERT** | 0.128 [0.083, 0.178] | 0.137 [0.085, 0.191] | 0.119 [0.075, 0.163] | 0.180 [0.118, 0.244] | 0.159 [0.099, 0.221] | 0.173 [0.112, 0.236] | 0.164 [0.102, 0.227] | 0.152 [0.067, 0.245] | 0.275 [0.182, 0.366] |
| **Uni3D-g** | 0.101 [0.048, 0.153] | 0.168 [0.089, 0.246] | 0.113 [0.056, 0.170] | 0.139 [0.065, 0.211] | 0.144 [0.076, 0.211] | 0.152 [0.081, 0.221] | 0.143 [0.069, 0.215] | 0.075 [-0.010, 0.160] | 0.276 [0.159, 0.389] |
| **taglia stimata dalla mesh (non oracolo)** | 0.600 [0.498, 0.690] | 0.056 [0.026, 0.092] | 0.353 [0.276, 0.427] | 0.602 [0.498, 0.692] | 0.587 [0.485, 0.678] | 0.601 [0.498, 0.692] | 0.605 [0.501, 0.695] | 0.605 [0.503, 0.695] | 0.620 [0.518, 0.710] |
| **taglia oracolo (S di FR)** | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.737 [0.656, 0.803] | 0.737 [0.657, 0.802] | 0.736 [0.658, 0.802] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (98224) | crop (49306) | all_cross (147530) | con_perturbazione original (39406) | con_perturbazione remesh (39406) | con_perturbazione down8k (39406) | con_perturbazione up60k (38824) | con_perturbazione noisy (39406) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.318 [0.227, 0.407] | 0.297 [0.215, 0.378] | 0.303 [0.216, 0.389] | 0.322 [0.232, 0.411] | 0.320 [0.230, 0.408] | 0.320 [0.229, 0.410] | 0.319 [0.229, 0.407] | 0.310 [0.217, 0.403] | 0.332 [0.240, 0.421] |
| factorized s1234, d_F non cal. | 0.551 [0.463, 0.630] | 0.488 [0.400, 0.568] | 0.503 [0.418, 0.579] | 0.558 [0.470, 0.636] | 0.555 [0.468, 0.634] | 0.554 [0.466, 0.634] | 0.554 [0.465, 0.633] | 0.535 [0.446, 0.613] | 0.566 [0.480, 0.644] |
| **factorized s1234, d_P** | 0.622 [0.550, 0.685] | 0.550 [0.472, 0.623] | 0.557 [0.485, 0.624] | 0.629 [0.558, 0.692] | 0.626 [0.553, 0.690] | 0.625 [0.555, 0.688] | 0.626 [0.554, 0.690] | 0.604 [0.533, 0.668] | 0.631 [0.560, 0.695] |
| factorized s2345, d_F cal.* | 0.312 [0.220, 0.401] | 0.255 [0.182, 0.326] | 0.286 [0.199, 0.366] | 0.315 [0.223, 0.405] | 0.311 [0.220, 0.398] | 0.316 [0.222, 0.406] | 0.312 [0.220, 0.401] | 0.308 [0.216, 0.397] | 0.323 [0.229, 0.412] |
| factorized s2345, d_F non cal. | 0.539 [0.445, 0.621] | 0.444 [0.354, 0.525] | 0.482 [0.394, 0.560] | 0.543 [0.450, 0.626] | 0.541 [0.448, 0.622] | 0.543 [0.449, 0.625] | 0.541 [0.446, 0.622] | 0.529 [0.433, 0.610] | 0.551 [0.460, 0.632] |
| **factorized s2345, d_P** | 0.613 [0.533, 0.688] | 0.514 [0.435, 0.590] | 0.540 [0.463, 0.613] | 0.617 [0.537, 0.691] | 0.616 [0.537, 0.690] | 0.616 [0.537, 0.690] | 0.616 [0.537, 0.689] | 0.600 [0.519, 0.675] | 0.618 [0.540, 0.689] |
| ctrlfr s1234 | 0.339 [0.242, 0.425] | 0.163 [0.120, 0.203] | 0.244 [0.174, 0.306] | 0.344 [0.245, 0.432] | 0.339 [0.243, 0.425] | 0.342 [0.243, 0.430] | 0.335 [0.236, 0.422] | 0.334 [0.237, 0.418] | 0.358 [0.261, 0.445] |
| ctrlfr s2345 | 0.348 [0.251, 0.435] | 0.188 [0.144, 0.230] | 0.258 [0.189, 0.321] | 0.354 [0.255, 0.442] | 0.346 [0.251, 0.432] | 0.351 [0.254, 0.439] | 0.349 [0.251, 0.437] | 0.339 [0.246, 0.425] | 0.363 [0.266, 0.451] |
| C3M e123, d_F cal.* | 0.292 [0.192, 0.381] | 0.220 [0.149, 0.285] | 0.251 [0.167, 0.330] | 0.297 [0.196, 0.387] | 0.289 [0.191, 0.377] | 0.293 [0.194, 0.384] | 0.291 [0.191, 0.382] | 0.291 [0.193, 0.381] | 0.306 [0.210, 0.396] |
| C3M e123, d_F non cal. | 0.555 [0.462, 0.641] | 0.420 [0.333, 0.506] | 0.441 [0.357, 0.521] | 0.565 [0.471, 0.650] | 0.558 [0.464, 0.643] | 0.558 [0.466, 0.644] | 0.557 [0.461, 0.646] | 0.547 [0.453, 0.632] | 0.576 [0.486, 0.660] |
| **C3M e123, d_P** | 0.595 [0.511, 0.671] | 0.462 [0.381, 0.541] | 0.461 [0.383, 0.531] | 0.604 [0.519, 0.681] | 0.599 [0.515, 0.675] | 0.598 [0.516, 0.672] | 0.599 [0.513, 0.676] | 0.585 [0.499, 0.665] | 0.607 [0.525, 0.681] |
| C3M e205, d_F cal.* | 0.286 [0.182, 0.376] | 0.219 [0.152, 0.283] | 0.250 [0.161, 0.330] | 0.291 [0.185, 0.381] | 0.284 [0.181, 0.372] | 0.287 [0.180, 0.378] | 0.283 [0.179, 0.376] | 0.289 [0.185, 0.377] | 0.303 [0.201, 0.392] |
| C3M e205, d_F non cal. | 0.556 [0.461, 0.642] | 0.442 [0.351, 0.525] | 0.456 [0.370, 0.536] | 0.566 [0.471, 0.654] | 0.560 [0.464, 0.645] | 0.558 [0.461, 0.644] | 0.558 [0.460, 0.648] | 0.552 [0.455, 0.639] | 0.581 [0.493, 0.666] |
| **C3M e205, d_P** | 0.590 [0.507, 0.669] | 0.481 [0.399, 0.561] | 0.473 [0.393, 0.546] | 0.600 [0.517, 0.680] | 0.594 [0.511, 0.673] | 0.592 [0.508, 0.671] | 0.594 [0.508, 0.674] | 0.585 [0.500, 0.662] | 0.607 [0.528, 0.682] |
| **ICP + Chamfer in mm (per coppia)** | 0.365 [0.280, 0.452] | 0.267 [0.209, 0.329] | 0.326 [0.252, 0.403] | 0.367 [0.285, 0.454] | 0.382 [0.295, 0.474] | 0.344 [0.261, 0.428] | 0.375 [0.286, 0.466] | 0.377 [0.292, 0.465] | 0.380 [0.298, 0.466] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.580 [0.529, 0.631] | 0.339 [0.290, 0.389] | 0.425 [0.378, 0.471] | 0.557 [0.507, 0.606] | 0.617 [0.564, 0.667] | 0.558 [0.504, 0.608] | 0.614 [0.561, 0.667] | 0.594 [0.545, 0.643] | 0.631 [0.581, 0.682] |
| **NICP per coppia in mm** | 0.503 [0.437, 0.574] | 0.417 [0.357, 0.481] | 0.467 [0.403, 0.535] | 0.477 [0.414, 0.545] | 0.532 [0.464, 0.603] | 0.498 [0.431, 0.569] | 0.522 [0.456, 0.596] | 0.503 [0.437, 0.573] | - |
| **NICP per coppia cs** | 0.595 [0.546, 0.648] | 0.411 [0.363, 0.460] | 0.517 [0.470, 0.567] | 0.562 [0.514, 0.613] | 0.621 [0.571, 0.676] | 0.585 [0.534, 0.640] | 0.619 [0.567, 0.673] | 0.606 [0.558, 0.657] | - |
| **NICP su template in mm** | 0.083 [-0.018, 0.185] | 0.046 [-0.014, 0.111] | 0.066 [-0.019, 0.154] | 0.083 [-0.017, 0.186] | 0.083 [-0.018, 0.185] | 0.081 [-0.019, 0.183] | 0.086 [-0.017, 0.187] | 0.079 [-0.021, 0.182] | 0.086 [-0.014, 0.189] |
| **NICP su template cs** | 0.301 [0.200, 0.393] | 0.045 [0.009, 0.085] | 0.151 [0.095, 0.200] | 0.304 [0.202, 0.394] | 0.299 [0.200, 0.385] | 0.301 [0.197, 0.392] | 0.309 [0.203, 0.406] | 0.294 [0.192, 0.384] | 0.326 [0.221, 0.418] |
| **varifold in mm (massa unitaria)** | 0.195 [0.126, 0.261] | 0.165 [0.122, 0.211] | 0.164 [0.109, 0.218] | 0.223 [0.147, 0.293] | 0.223 [0.148, 0.292] | 0.223 [0.147, 0.293] | 0.222 [0.147, 0.292] | 0.254 [0.179, 0.328] | 0.312 [0.226, 0.396] |
| **GNM (visto) preregistrato, coefficienti** | 0.156 [0.047, 0.259] | 0.106 [0.020, 0.190] | 0.136 [0.040, 0.229] | 0.163 [0.052, 0.266] | 0.159 [0.046, 0.262] | 0.155 [0.046, 0.256] | 0.160 [0.048, 0.267] | 0.145 [0.040, 0.247] | 0.161 [0.051, 0.261] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.093 [-0.004, 0.189] | 0.080 [-0.005, 0.169] | 0.088 [-0.004, 0.179] | 0.098 [-0.001, 0.193] | 0.093 [-0.004, 0.187] | 0.093 [-0.005, 0.188] | 0.093 [-0.007, 0.189] | 0.088 [-0.010, 0.180] | 0.095 [-0.002, 0.190] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.325 [0.205, 0.441] | 0.199 [0.112, 0.289] | 0.264 [0.165, 0.364] | 0.326 [0.204, 0.441] | 0.328 [0.207, 0.442] | 0.326 [0.208, 0.439] | 0.329 [0.205, 0.444] | 0.315 [0.195, 0.427] | 0.321 [0.206, 0.431] |
| **GNM (visto) vA, coefficienti*** | 0.194 [0.088, 0.295] | 0.148 [0.060, 0.240] | 0.171 [0.078, 0.266] | 0.199 [0.092, 0.300] | 0.198 [0.093, 0.299] | 0.191 [0.086, 0.290] | 0.200 [0.093, 0.305] | 0.181 [0.077, 0.281] | 0.197 [0.096, 0.294] |
| GNM (visto) vA, mesh d'identita' FR* | 0.105 [0.006, 0.198] | 0.091 [0.005, 0.179] | 0.099 [0.007, 0.191] | 0.109 [0.011, 0.203] | 0.105 [0.007, 0.198] | 0.105 [0.007, 0.199] | 0.105 [0.005, 0.199] | 0.098 [-0.000, 0.193] | 0.107 [0.010, 0.202] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.339 [0.229, 0.443] | 0.213 [0.127, 0.307] | 0.280 [0.188, 0.377] | 0.338 [0.228, 0.446] | 0.342 [0.233, 0.451] | 0.339 [0.232, 0.444] | 0.346 [0.235, 0.453] | 0.333 [0.226, 0.437] | 0.335 [0.232, 0.438] |
| **GNM (visto) vB, coefficienti*** | 0.674 [0.610, 0.734] | 0.651 [0.587, 0.711] | 0.664 [0.600, 0.724] | 0.676 [0.610, 0.735] | 0.674 [0.607, 0.733] | 0.672 [0.606, 0.732] | 0.673 [0.607, 0.733] | 0.680 [0.614, 0.738] | 0.673 [0.609, 0.732] |
| GNM (visto) vB, mesh d'identita' FR* | 0.217 [0.127, 0.306] | 0.205 [0.122, 0.291] | 0.212 [0.128, 0.300] | 0.221 [0.130, 0.309] | 0.215 [0.126, 0.304] | 0.218 [0.129, 0.307] | 0.217 [0.125, 0.307] | 0.214 [0.126, 0.303] | 0.220 [0.131, 0.309] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.607 [0.532, 0.679] | 0.570 [0.490, 0.647] | 0.587 [0.511, 0.661] | 0.609 [0.535, 0.681] | 0.606 [0.531, 0.679] | 0.607 [0.530, 0.679] | 0.607 [0.532, 0.679] | 0.607 [0.533, 0.678] | 0.605 [0.531, 0.677] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.348 [0.257, 0.438] | 0.332 [0.244, 0.420] | 0.341 [0.252, 0.429] | 0.352 [0.261, 0.440] | 0.346 [0.255, 0.434] | 0.350 [0.259, 0.440] | 0.348 [0.257, 0.438] | 0.344 [0.256, 0.434] | 0.352 [0.261, 0.441] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.159 [0.044, 0.268] | 0.096 [0.017, 0.168] | 0.137 [0.035, 0.228] | 0.161 [0.048, 0.269] | 0.161 [0.044, 0.275] | 0.166 [0.049, 0.279] | 0.166 [0.051, 0.279] | 0.139 [0.035, 0.242] | 0.163 [0.051, 0.271] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.125 [0.025, 0.224] | 0.095 [0.010, 0.181] | 0.114 [0.021, 0.211] | 0.127 [0.026, 0.228] | 0.126 [0.026, 0.229] | 0.129 [0.026, 0.232] | 0.128 [0.028, 0.227] | 0.113 [0.017, 0.208] | 0.125 [0.027, 0.226] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.287 [0.177, 0.398] | 0.199 [0.118, 0.281] | 0.253 [0.157, 0.353] | 0.301 [0.190, 0.414] | 0.284 [0.174, 0.396] | 0.292 [0.181, 0.404] | 0.305 [0.193, 0.417] | 0.254 [0.155, 0.352] | 0.299 [0.188, 0.408] |
| **FLAME 2023 Open vA, coefficienti*** | 0.155 [0.043, 0.262] | 0.096 [0.016, 0.170] | 0.135 [0.035, 0.228] | 0.158 [0.046, 0.264] | 0.158 [0.042, 0.267] | 0.163 [0.048, 0.273] | 0.162 [0.048, 0.273] | 0.137 [0.031, 0.237] | 0.159 [0.048, 0.269] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.118 [0.020, 0.217] | 0.094 [0.008, 0.181] | 0.110 [0.016, 0.205] | 0.120 [0.023, 0.219] | 0.120 [0.021, 0.221] | 0.123 [0.022, 0.224] | 0.121 [0.023, 0.218] | 0.108 [0.010, 0.206] | 0.119 [0.021, 0.217] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.299 [0.193, 0.405] | 0.220 [0.137, 0.302] | 0.268 [0.172, 0.366] | 0.314 [0.204, 0.423] | 0.296 [0.187, 0.405] | 0.304 [0.195, 0.412] | 0.316 [0.208, 0.423] | 0.268 [0.169, 0.366] | 0.310 [0.204, 0.418] |
| **FLAME 2023 Open vB, coefficienti*** | 0.585 [0.513, 0.652] | 0.551 [0.482, 0.615] | 0.573 [0.503, 0.636] | 0.586 [0.513, 0.650] | 0.581 [0.509, 0.649] | 0.590 [0.517, 0.656] | 0.585 [0.514, 0.651] | 0.587 [0.514, 0.655] | 0.582 [0.513, 0.647] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.231 [0.141, 0.320] | 0.208 [0.125, 0.294] | 0.223 [0.136, 0.311] | 0.232 [0.142, 0.322] | 0.230 [0.141, 0.319] | 0.234 [0.143, 0.325] | 0.231 [0.142, 0.320] | 0.226 [0.138, 0.314] | 0.230 [0.140, 0.319] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.597 [0.518, 0.673] | 0.567 [0.482, 0.647] | 0.585 [0.505, 0.662] | 0.600 [0.520, 0.675] | 0.594 [0.514, 0.671] | 0.598 [0.519, 0.673] | 0.601 [0.520, 0.677] | 0.595 [0.515, 0.669] | 0.594 [0.514, 0.670] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.346 [0.255, 0.437] | 0.320 [0.232, 0.408] | 0.337 [0.248, 0.426] | 0.348 [0.258, 0.437] | 0.344 [0.253, 0.436] | 0.349 [0.257, 0.438] | 0.347 [0.255, 0.437] | 0.341 [0.252, 0.430] | 0.345 [0.256, 0.435] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.220 [0.149, 0.293] | 0.225 [0.154, 0.301] | 0.221 [0.150, 0.293] | 0.221 [0.149, 0.293] | 0.229 [0.159, 0.301] | 0.226 [0.153, 0.300] | 0.225 [0.154, 0.298] | 0.213 [0.130, 0.293] | 0.183 [0.130, 0.238] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.237 [0.171, 0.302] | 0.249 [0.174, 0.322] | 0.241 [0.172, 0.308] | 0.246 [0.173, 0.319] | 0.245 [0.175, 0.314] | 0.248 [0.177, 0.319] | 0.250 [0.178, 0.321] | 0.223 [0.153, 0.296] | 0.208 [0.150, 0.266] |
| **ShapeDNA k=50** | 0.126 [0.097, 0.151] | 0.038 [0.008, 0.067] | 0.088 [0.066, 0.107] | 0.204 [0.160, 0.244] | 0.193 [0.151, 0.231] | 0.195 [0.151, 0.232] | 0.173 [0.133, 0.208] | 0.038 [-0.008, 0.082] | 0.375 [0.299, 0.443] |
| **HKS globale** | 0.069 [0.039, 0.100] | 0.004 [-0.008, 0.016] | 0.039 [0.022, 0.056] | 0.108 [0.060, 0.158] | 0.093 [0.053, 0.135] | 0.106 [0.058, 0.155] | 0.097 [0.053, 0.142] | 0.041 [-0.003, 0.082] | 0.192 [0.105, 0.278] |
| **WKS globale** | 0.030 [0.016, 0.044] | 0.062 [0.036, 0.089] | 0.037 [0.022, 0.054] | 0.065 [0.042, 0.090] | 0.065 [0.042, 0.089] | 0.053 [0.030, 0.073] | 0.041 [0.024, 0.060] | 0.016 [-0.025, 0.060] | 0.304 [0.226, 0.384] |
| **OpenShape PointBERT** | 0.182 [0.141, 0.227] | 0.183 [0.140, 0.226] | 0.165 [0.126, 0.205] | 0.255 [0.206, 0.309] | 0.235 [0.185, 0.290] | 0.247 [0.194, 0.304] | 0.232 [0.176, 0.289] | 0.200 [0.112, 0.291] | 0.381 [0.301, 0.460] |
| **Uni3D-g** | 0.136 [0.096, 0.177] | 0.241 [0.181, 0.297] | 0.154 [0.110, 0.196] | 0.195 [0.138, 0.255] | 0.191 [0.139, 0.243] | 0.203 [0.148, 0.259] | 0.198 [0.141, 0.259] | 0.083 [-0.001, 0.170] | 0.388 [0.298, 0.478] |
| **taglia stimata dalla mesh (non oracolo)** | 0.023 [-0.070, 0.111] | 0.009 [-0.004, 0.022] | 0.017 [-0.038, 0.071] | 0.023 [-0.070, 0.111] | 0.023 [-0.067, 0.108] | 0.023 [-0.070, 0.110] | 0.022 [-0.072, 0.111] | 0.023 [-0.069, 0.112] | 0.026 [-0.068, 0.116] |
| **taglia oracolo (S di FR)** | 0.074 [-0.025, 0.165] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.165] | 0.073 [-0.029, 0.163] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.166] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.106 [+0.062, +0.154] | +0.245 [+0.199, +0.294] | +0.153 [+0.110, +0.200] | +0.117 [+0.074, +0.166] | +0.072 [+0.029, +0.119] | +0.131 [+0.088, +0.176] | +0.084 [+0.040, +0.132] | +0.089 [+0.042, +0.139] | +0.115 [+0.070, +0.166] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.390 [+0.291, +0.493] | +0.450 [+0.364, +0.530] | +0.445 [+0.356, +0.533] | +0.408 [+0.311, +0.508] | +0.362 [+0.259, +0.467] | +0.406 [+0.309, +0.505] | +0.374 [+0.271, +0.479] | +0.379 [+0.277, +0.483] | +0.372 [+0.269, +0.477] |
| NICP per coppia in mm | +0.253 [+0.170, +0.342] | +0.247 [+0.168, +0.332] | +0.247 [+0.167, +0.334] | +0.279 [+0.197, +0.366] | +0.226 [+0.141, +0.318] | +0.265 [+0.181, +0.355] | +0.239 [+0.153, +0.329] | +0.239 [+0.153, +0.333] | - |
| NICP per coppia cs | +0.373 [+0.261, +0.481] | +0.406 [+0.306, +0.503] | +0.384 [+0.279, +0.488] | +0.398 [+0.290, +0.502] | +0.350 [+0.235, +0.460] | +0.381 [+0.269, +0.487] | +0.362 [+0.246, +0.474] | +0.364 [+0.251, +0.475] | - |
| NICP su template in mm | +0.134 [+0.085, +0.188] | +0.325 [+0.270, +0.382] | +0.195 [+0.148, +0.247] | +0.139 [+0.089, +0.193] | +0.125 [+0.075, +0.177] | +0.144 [+0.094, +0.199] | +0.131 [+0.083, +0.183] | +0.135 [+0.084, +0.188] | +0.139 [+0.090, +0.192] |
| NICP su template cs | +0.560 [+0.465, +0.657] | +0.633 [+0.560, +0.693] | +0.614 [+0.541, +0.686] | +0.564 [+0.471, +0.660] | +0.550 [+0.458, +0.645] | +0.563 [+0.469, +0.659] | +0.556 [+0.459, +0.656] | +0.566 [+0.467, +0.668] | +0.559 [+0.462, +0.659] |
| varifold in mm (massa unitaria) | +0.254 [+0.211, +0.301] | +0.384 [+0.331, +0.428] | +0.328 [+0.288, +0.369] | +0.204 [+0.164, +0.247] | +0.191 [+0.153, +0.233] | +0.200 [+0.160, +0.244] | +0.200 [+0.160, +0.242] | +0.185 [+0.140, +0.233] | +0.064 [+0.031, +0.102] |
| GNM (visto) preregistrato, coefficienti | +0.130 [+0.078, +0.188] | +0.220 [+0.161, +0.280] | +0.159 [+0.111, +0.208] | +0.131 [+0.081, +0.190] | +0.123 [+0.070, +0.182] | +0.135 [+0.083, +0.194] | +0.127 [+0.074, +0.186] | +0.135 [+0.084, +0.192] | +0.138 [+0.088, +0.195] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.141 [+0.088, +0.195] | +0.126 [+0.070, +0.184] | +0.125 [+0.076, +0.176] | +0.142 [+0.090, +0.195] | +0.133 [+0.081, +0.187] | +0.144 [+0.091, +0.199] | +0.140 [+0.090, +0.194] | +0.145 [+0.093, +0.202] | +0.145 [+0.093, +0.199] |
| GNM (visto) vA, coefficienti | +0.145 [+0.091, +0.207] | +0.196 [+0.131, +0.262] | +0.167 [+0.114, +0.223] | +0.148 [+0.093, +0.208] | +0.136 [+0.082, +0.197] | +0.151 [+0.096, +0.213] | +0.141 [+0.087, +0.203] | +0.150 [+0.093, +0.210] | +0.165 [+0.111, +0.226] |
| GNM (visto) vA, mesh d'identita' FR | +0.142 [+0.090, +0.195] | +0.119 [+0.064, +0.177] | +0.124 [+0.074, +0.174] | +0.143 [+0.091, +0.196] | +0.134 [+0.082, +0.189] | +0.145 [+0.093, +0.199] | +0.141 [+0.092, +0.195] | +0.146 [+0.094, +0.202] | +0.145 [+0.095, +0.200] |
| GNM (visto) vB, coefficienti | +0.199 [+0.095, +0.299] | +0.139 [+0.044, +0.235] | +0.169 [+0.067, +0.269] | +0.203 [+0.099, +0.306] | +0.189 [+0.087, +0.288] | +0.205 [+0.101, +0.305] | +0.203 [+0.099, +0.302] | +0.193 [+0.088, +0.296] | +0.216 [+0.114, +0.318] |
| GNM (visto) vB, mesh d'identita' FR | +0.084 [+0.042, +0.128] | +0.038 [-0.004, +0.080] | +0.058 [+0.018, +0.096] | +0.086 [+0.044, +0.130] | +0.078 [+0.036, +0.120] | +0.087 [+0.044, +0.131] | +0.084 [+0.042, +0.127] | +0.086 [+0.043, +0.132] | +0.088 [+0.046, +0.131] |
| GNM (visto) vB, composizione S_B + k d_P | +0.051 [+0.009, +0.094] | +0.002 [-0.040, +0.044] | +0.026 [-0.012, +0.064] | +0.054 [+0.012, +0.098] | +0.045 [+0.004, +0.087] | +0.054 [+0.012, +0.098] | +0.051 [+0.010, +0.094] | +0.051 [+0.009, +0.095] | +0.057 [+0.016, +0.101] |
| FLAME 2023 Open preregistrato, coefficienti | +0.182 [+0.119, +0.254] | +0.262 [+0.210, +0.320] | +0.202 [+0.145, +0.265] | +0.184 [+0.121, +0.258] | +0.183 [+0.118, +0.255] | +0.180 [+0.113, +0.256] | +0.172 [+0.109, +0.244] | +0.192 [+0.129, +0.261] | +0.184 [+0.119, +0.260] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.109 [+0.062, +0.161] | +0.117 [+0.069, +0.173] | +0.101 [+0.057, +0.150] | +0.110 [+0.063, +0.161] | +0.105 [+0.058, +0.158] | +0.108 [+0.061, +0.161] | +0.105 [+0.057, +0.159] | +0.120 [+0.073, +0.172] | +0.114 [+0.067, +0.167] |
| FLAME 2023 Open vA, coefficienti | +0.168 [+0.108, +0.238] | +0.239 [+0.189, +0.296] | +0.185 [+0.131, +0.247] | +0.170 [+0.109, +0.241] | +0.168 [+0.106, +0.240] | +0.165 [+0.102, +0.238] | +0.158 [+0.098, +0.229] | +0.179 [+0.118, +0.244] | +0.170 [+0.110, +0.244] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.107 [+0.061, +0.158] | +0.097 [+0.050, +0.152] | +0.093 [+0.050, +0.141] | +0.109 [+0.062, +0.159] | +0.102 [+0.056, +0.154] | +0.106 [+0.058, +0.158] | +0.104 [+0.058, +0.154] | +0.117 [+0.072, +0.169] | +0.113 [+0.067, +0.164] |
| FLAME 2023 Open vB, coefficienti | +0.163 [+0.072, +0.254] | +0.120 [+0.039, +0.209] | +0.139 [+0.051, +0.230] | +0.172 [+0.081, +0.263] | +0.159 [+0.070, +0.248] | +0.167 [+0.076, +0.259] | +0.162 [+0.073, +0.253] | +0.154 [+0.061, +0.245] | +0.185 [+0.096, +0.273] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.063 [+0.023, +0.101] | +0.024 [-0.018, +0.065] | +0.039 [+0.000, +0.074] | +0.064 [+0.025, +0.102] | +0.058 [+0.019, +0.097] | +0.062 [+0.021, +0.101] | +0.061 [+0.022, +0.099] | +0.068 [+0.028, +0.107] | +0.069 [+0.029, +0.107] |
| FLAME 2023 Open vB, composizione S_B + k d_P | +0.037 [-0.002, +0.075] | -0.008 [-0.048, +0.032] | +0.011 [-0.026, +0.048] | +0.039 [+0.001, +0.077] | +0.032 [-0.006, +0.070] | +0.038 [-0.001, +0.079] | +0.035 [-0.003, +0.074] | +0.040 [-0.001, +0.079] | +0.046 [+0.007, +0.084] |
| ArcFace-render, normal map 3 viste (G0) | +0.615 [+0.498, +0.720] | +0.523 [+0.403, +0.626] | +0.574 [+0.455, +0.679] | +0.621 [+0.502, +0.728] | +0.601 [+0.484, +0.706] | +0.614 [+0.492, +0.720] | +0.614 [+0.494, +0.722] | +0.619 [+0.498, +0.726] | +0.648 [+0.549, +0.739] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.583 [+0.471, +0.685] | +0.480 [+0.358, +0.588] | +0.537 [+0.420, +0.640] | +0.580 [+0.460, +0.685] | +0.566 [+0.447, +0.668] | +0.578 [+0.460, +0.684] | +0.575 [+0.456, +0.682] | +0.599 [+0.491, +0.698] | +0.614 [+0.509, +0.709] |
| ShapeDNA k=50 | +0.663 [+0.588, +0.731] | +0.654 [+0.574, +0.722] | +0.652 [+0.578, +0.715] | +0.611 [+0.529, +0.691] | +0.605 [+0.524, +0.680] | +0.617 [+0.537, +0.695] | +0.629 [+0.551, +0.703] | +0.744 [+0.658, +0.822] | +0.512 [+0.399, +0.619] |
| HKS globale | +0.712 [+0.635, +0.779] | +0.670 [+0.594, +0.733] | +0.691 [+0.616, +0.752] | +0.695 [+0.610, +0.770] | +0.690 [+0.610, +0.762] | +0.694 [+0.610, +0.769] | +0.698 [+0.616, +0.771] | +0.736 [+0.643, +0.807] | +0.658 [+0.555, +0.758] |
| WKS globale | +0.730 [+0.657, +0.788] | +0.629 [+0.556, +0.688] | +0.686 [+0.612, +0.746] | +0.711 [+0.636, +0.772] | +0.697 [+0.624, +0.757] | +0.714 [+0.641, +0.777] | +0.723 [+0.650, +0.784] | +0.751 [+0.669, +0.829] | +0.563 [+0.460, +0.670] |
| OpenShape PointBERT | +0.620 [+0.536, +0.693] | +0.526 [+0.446, +0.601] | +0.590 [+0.509, +0.662] | +0.574 [+0.486, +0.657] | +0.582 [+0.493, +0.664] | +0.578 [+0.486, +0.662] | +0.586 [+0.498, +0.671] | +0.595 [+0.479, +0.701] | +0.487 [+0.383, +0.597] |
| Uni3D-g | +0.647 [+0.566, +0.720] | +0.494 [+0.399, +0.585] | +0.596 [+0.512, +0.669] | +0.615 [+0.523, +0.699] | +0.597 [+0.513, +0.678] | +0.600 [+0.511, +0.683] | +0.607 [+0.517, +0.694] | +0.671 [+0.559, +0.778] | +0.486 [+0.364, +0.611] |
| taglia stimata dalla mesh (non oracolo) | +0.149 [+0.091, +0.208] | +0.606 [+0.551, +0.651] | +0.356 [+0.311, +0.399] | +0.152 [+0.095, +0.213] | +0.153 [+0.096, +0.213] | +0.150 [+0.094, +0.211] | +0.146 [+0.089, +0.204] | +0.141 [+0.085, +0.201] | +0.142 [+0.085, +0.201] |
| taglia oracolo (S di FR) | +0.012 [-0.034, +0.062] | -0.074 [-0.121, -0.021] | -0.028 [-0.072, +0.021] | +0.018 [-0.029, +0.068] | +0.004 [-0.042, +0.056] | +0.015 [-0.031, +0.065] | +0.014 [-0.033, +0.065] | +0.010 [-0.037, +0.060] | +0.025 [-0.021, +0.075] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.001 [-0.055, +0.055] | +0.145 [+0.079, +0.201] | +0.028 [-0.029, +0.081] | +0.012 [-0.041, +0.065] | -0.026 [-0.081, +0.029] | +0.023 [-0.032, +0.077] | -0.020 [-0.075, +0.037] | -0.024 [-0.081, +0.032] | +0.001 [-0.055, +0.056] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.284 [+0.206, +0.366] | +0.350 [+0.283, +0.418] | +0.320 [+0.254, +0.387] | +0.303 [+0.227, +0.383] | +0.263 [+0.182, +0.347] | +0.299 [+0.225, +0.376] | +0.269 [+0.188, +0.353] | +0.265 [+0.184, +0.351] | +0.258 [+0.178, +0.344] |
| NICP per coppia in mm | +0.147 [+0.086, +0.216] | +0.147 [+0.079, +0.215] | +0.122 [+0.060, +0.187] | +0.174 [+0.114, +0.241] | +0.127 [+0.064, +0.199] | +0.158 [+0.099, +0.227] | +0.135 [+0.072, +0.206] | +0.125 [+0.062, +0.196] | - |
| NICP per coppia cs | +0.267 [+0.181, +0.353] | +0.306 [+0.228, +0.390] | +0.258 [+0.177, +0.337] | +0.292 [+0.210, +0.377] | +0.251 [+0.162, +0.340] | +0.273 [+0.188, +0.356] | +0.258 [+0.169, +0.346] | +0.251 [+0.164, +0.340] | - |
| NICP su template in mm | +0.028 [-0.057, +0.117] | +0.226 [+0.135, +0.316] | +0.070 [-0.018, +0.155] | +0.033 [-0.052, +0.123] | +0.026 [-0.058, +0.114] | +0.036 [-0.049, +0.127] | +0.026 [-0.058, +0.113] | +0.021 [-0.064, +0.112] | +0.026 [-0.058, +0.115] |
| NICP su template cs | +0.454 [+0.371, +0.537] | +0.533 [+0.457, +0.596] | +0.489 [+0.423, +0.548] | +0.458 [+0.377, +0.541] | +0.452 [+0.378, +0.531] | +0.456 [+0.371, +0.540] | +0.452 [+0.370, +0.537] | +0.452 [+0.366, +0.540] | +0.446 [+0.364, +0.529] |
| varifold in mm (massa unitaria) | +0.148 [+0.083, +0.214] | +0.284 [+0.226, +0.340] | +0.203 [+0.142, +0.262] | +0.098 [+0.036, +0.161] | +0.093 [+0.029, +0.157] | +0.093 [+0.029, +0.155] | +0.095 [+0.033, +0.156] | +0.071 [+0.014, +0.125] | -0.050 [-0.109, +0.010] |
| GNM (visto) preregistrato, coefficienti | +0.024 [-0.050, +0.107] | +0.120 [+0.030, +0.216] | +0.033 [-0.045, +0.110] | +0.026 [-0.047, +0.106] | +0.025 [-0.052, +0.108] | +0.027 [-0.046, +0.110] | +0.022 [-0.052, +0.104] | +0.021 [-0.058, +0.105] | +0.025 [-0.049, +0.106] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.035 [-0.052, +0.121] | +0.026 [-0.071, +0.123] | +0.000 [-0.089, +0.088] | +0.037 [-0.047, +0.124] | +0.035 [-0.051, +0.122] | +0.037 [-0.048, +0.126] | +0.036 [-0.050, +0.123] | +0.031 [-0.055, +0.123] | +0.031 [-0.054, +0.122] |
| GNM (visto) vA, coefficienti | +0.039 [-0.036, +0.118] | +0.097 [+0.009, +0.191] | +0.042 [-0.037, +0.123] | +0.043 [-0.030, +0.123] | +0.038 [-0.036, +0.117] | +0.043 [-0.031, +0.124] | +0.037 [-0.038, +0.112] | +0.036 [-0.042, +0.119] | +0.052 [-0.025, +0.132] |
| GNM (visto) vA, mesh d'identita' FR | +0.035 [-0.049, +0.122] | +0.019 [-0.075, +0.116] | -0.001 [-0.089, +0.086] | +0.037 [-0.045, +0.123] | +0.036 [-0.049, +0.123] | +0.038 [-0.045, +0.126] | +0.037 [-0.048, +0.122] | +0.032 [-0.054, +0.122] | +0.032 [-0.051, +0.121] |
| GNM (visto) vB, coefficienti | +0.093 [+0.011, +0.180] | +0.039 [-0.041, +0.121] | +0.044 [-0.037, +0.127] | +0.098 [+0.018, +0.185] | +0.091 [+0.011, +0.177] | +0.097 [+0.015, +0.185] | +0.099 [+0.015, +0.187] | +0.079 [-0.003, +0.169] | +0.103 [+0.023, +0.190] |
| GNM (visto) vB, mesh d'identita' FR | -0.022 [-0.094, +0.049] | -0.062 [-0.146, +0.021] | -0.067 [-0.142, +0.007] | -0.019 [-0.090, +0.053] | -0.020 [-0.092, +0.052] | -0.021 [-0.091, +0.052] | -0.021 [-0.091, +0.050] | -0.028 [-0.103, +0.045] | -0.026 [-0.096, +0.047] |
| GNM (visto) vB, composizione S_B + k d_P | -0.056 [-0.118, +0.007] | -0.098 [-0.173, -0.026] | -0.100 [-0.165, -0.037] | -0.052 [-0.113, +0.009] | -0.054 [-0.118, +0.007] | -0.054 [-0.115, +0.008] | -0.053 [-0.114, +0.006] | -0.063 [-0.128, +0.002] | -0.057 [-0.118, +0.006] |
| FLAME 2023 Open preregistrato, coefficienti | +0.076 [-0.005, +0.162] | +0.162 [+0.079, +0.245] | +0.077 [+0.001, +0.156] | +0.079 [+0.000, +0.167] | +0.085 [+0.002, +0.174] | +0.072 [-0.011, +0.159] | +0.068 [-0.013, +0.155] | +0.078 [-0.005, +0.162] | +0.071 [-0.012, +0.156] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.003 [-0.073, +0.081] | +0.017 [-0.069, +0.108] | -0.024 [-0.104, +0.053] | +0.004 [-0.072, +0.082] | +0.007 [-0.071, +0.088] | +0.000 [-0.075, +0.081] | +0.001 [-0.074, +0.079] | +0.006 [-0.074, +0.086] | +0.000 [-0.075, +0.079] |
| FLAME 2023 Open vA, coefficienti | +0.062 [-0.018, +0.147] | +0.140 [+0.055, +0.223] | +0.060 [-0.018, +0.139] | +0.064 [-0.015, +0.151] | +0.070 [-0.011, +0.157] | +0.058 [-0.023, +0.145] | +0.054 [-0.026, +0.139] | +0.065 [-0.017, +0.149] | +0.056 [-0.024, +0.141] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.001 [-0.075, +0.078] | -0.003 [-0.091, +0.089] | -0.032 [-0.114, +0.048] | +0.003 [-0.073, +0.079] | +0.004 [-0.073, +0.082] | -0.001 [-0.076, +0.076] | -0.000 [-0.076, +0.078] | +0.003 [-0.079, +0.085] | -0.001 [-0.076, +0.076] |
| FLAME 2023 Open vB, coefficienti | +0.057 [-0.024, +0.141] | +0.020 [-0.062, +0.107] | +0.014 [-0.062, +0.094] | +0.067 [-0.014, +0.152] | +0.061 [-0.019, +0.145] | +0.060 [-0.021, +0.141] | +0.057 [-0.022, +0.140] | +0.041 [-0.041, +0.126] | +0.071 [-0.008, +0.153] |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.044 [-0.112, +0.022] | -0.076 [-0.154, +0.002] | -0.086 [-0.157, -0.020] | -0.041 [-0.109, +0.024] | -0.040 [-0.109, +0.025] | -0.046 [-0.112, +0.019] | -0.043 [-0.111, +0.020] | -0.046 [-0.116, +0.023] | -0.045 [-0.114, +0.021] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.069 [-0.127, -0.012] | -0.108 [-0.179, -0.040] | -0.114 [-0.176, -0.054] | -0.067 [-0.124, -0.010] | -0.066 [-0.125, -0.008] | -0.069 [-0.126, -0.011] | -0.069 [-0.127, -0.012] | -0.073 [-0.135, -0.014] | -0.068 [-0.127, -0.010] |
| ArcFace-render, normal map 3 viste (G0) | +0.509 [+0.400, +0.606] | +0.423 [+0.297, +0.530] | +0.448 [+0.334, +0.546] | +0.515 [+0.405, +0.614] | +0.502 [+0.394, +0.602] | +0.506 [+0.397, +0.605] | +0.510 [+0.400, +0.611] | +0.506 [+0.390, +0.608] | +0.535 [+0.441, +0.619] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.476 [+0.375, +0.568] | +0.380 [+0.256, +0.488] | +0.412 [+0.303, +0.509] | +0.474 [+0.367, +0.572] | +0.467 [+0.361, +0.565] | +0.470 [+0.363, +0.568] | +0.471 [+0.363, +0.571] | +0.485 [+0.387, +0.581] | +0.501 [+0.407, +0.588] |
| ShapeDNA k=50 | +0.557 [+0.490, +0.620] | +0.554 [+0.466, +0.627] | +0.526 [+0.457, +0.589] | +0.506 [+0.431, +0.576] | +0.507 [+0.432, +0.576] | +0.510 [+0.437, +0.578] | +0.525 [+0.453, +0.591] | +0.630 [+0.556, +0.704] | +0.399 [+0.298, +0.491] |
| HKS globale | +0.606 [+0.536, +0.668] | +0.571 [+0.490, +0.639] | +0.566 [+0.495, +0.628] | +0.590 [+0.513, +0.659] | +0.592 [+0.518, +0.660] | +0.586 [+0.509, +0.657] | +0.594 [+0.518, +0.664] | +0.622 [+0.545, +0.690] | +0.544 [+0.448, +0.631] |
| WKS globale | +0.624 [+0.558, +0.684] | +0.529 [+0.461, +0.593] | +0.561 [+0.495, +0.621] | +0.606 [+0.542, +0.666] | +0.599 [+0.532, +0.658] | +0.607 [+0.539, +0.671] | +0.619 [+0.553, +0.679] | +0.637 [+0.564, +0.706] | +0.449 [+0.357, +0.543] |
| OpenShape PointBERT | +0.514 [+0.444, +0.574] | +0.426 [+0.353, +0.492] | +0.465 [+0.398, +0.522] | +0.469 [+0.396, +0.538] | +0.483 [+0.405, +0.551] | +0.471 [+0.397, +0.540] | +0.482 [+0.408, +0.549] | +0.481 [+0.387, +0.574] | +0.374 [+0.283, +0.458] |
| Uni3D-g | +0.541 [+0.469, +0.601] | +0.394 [+0.301, +0.477] | +0.470 [+0.396, +0.531] | +0.510 [+0.432, +0.580] | +0.499 [+0.425, +0.567] | +0.492 [+0.416, +0.562] | +0.503 [+0.423, +0.574] | +0.558 [+0.453, +0.649] | +0.372 [+0.266, +0.477] |
| taglia stimata dalla mesh (non oracolo) | +0.042 [-0.052, +0.137] | +0.507 [+0.434, +0.568] | +0.231 [+0.153, +0.302] | +0.047 [-0.047, +0.141] | +0.055 [-0.038, +0.148] | +0.043 [-0.051, +0.138] | +0.041 [-0.054, +0.135] | +0.028 [-0.068, +0.121] | +0.028 [-0.063, +0.123] |
| taglia oracolo (S di FR) | -0.094 [-0.160, -0.022] | -0.174 [-0.252, -0.093] | -0.153 [-0.223, -0.079] | -0.088 [-0.153, -0.017] | -0.094 [-0.160, -0.021] | -0.092 [-0.160, -0.021] | -0.091 [-0.157, -0.018] | -0.104 [-0.171, -0.033] | -0.088 [-0.154, -0.016] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.256 [+0.181, +0.332] | +0.283 [+0.217, +0.348] | +0.231 [+0.160, +0.300] | +0.262 [+0.190, +0.336] | +0.244 [+0.167, +0.322] | +0.282 [+0.208, +0.355] | +0.251 [+0.172, +0.333] | +0.228 [+0.150, +0.305] | +0.251 [+0.176, +0.325] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.041 [-0.005, +0.091] | +0.210 [+0.146, +0.269] | +0.132 [+0.088, +0.176] | +0.073 [+0.025, +0.122] | +0.009 [-0.040, +0.059] | +0.068 [+0.023, +0.112] | +0.012 [-0.035, +0.061] | +0.010 [-0.040, +0.062] | -0.001 [-0.050, +0.050] |
| NICP per coppia in mm | +0.119 [+0.052, +0.195] | +0.133 [+0.058, +0.211] | +0.090 [+0.019, +0.164] | +0.152 [+0.086, +0.223] | +0.095 [+0.026, +0.172] | +0.127 [+0.063, +0.201] | +0.104 [+0.036, +0.182] | +0.101 [+0.029, +0.179] | - |
| NICP per coppia cs | +0.027 [-0.029, +0.083] | +0.139 [+0.068, +0.205] | +0.040 [-0.016, +0.095] | +0.068 [+0.014, +0.125] | +0.005 [-0.053, +0.064] | +0.040 [-0.013, +0.096] | +0.007 [-0.048, +0.067] | -0.001 [-0.058, +0.057] | - |
| NICP su template in mm | +0.539 [+0.441, +0.638] | +0.503 [+0.418, +0.583] | +0.491 [+0.396, +0.579] | +0.546 [+0.447, +0.644] | +0.543 [+0.446, +0.641] | +0.544 [+0.444, +0.643] | +0.541 [+0.442, +0.641] | +0.525 [+0.419, +0.625] | +0.545 [+0.448, +0.641] |
| NICP su template cs | +0.321 [+0.250, +0.401] | +0.504 [+0.427, +0.574] | +0.406 [+0.353, +0.459] | +0.325 [+0.254, +0.407] | +0.327 [+0.261, +0.405] | +0.324 [+0.248, +0.408] | +0.317 [+0.243, +0.401] | +0.311 [+0.239, +0.391] | +0.304 [+0.231, +0.387] |
| varifold in mm (massa unitaria) | +0.427 [+0.359, +0.496] | +0.384 [+0.328, +0.440] | +0.393 [+0.333, +0.451] | +0.407 [+0.338, +0.477] | +0.403 [+0.333, +0.474] | +0.403 [+0.331, +0.472] | +0.404 [+0.334, +0.477] | +0.351 [+0.283, +0.418] | +0.319 [+0.246, +0.395] |
| GNM (visto) preregistrato, coefficienti | +0.465 [+0.375, +0.567] | +0.444 [+0.355, +0.527] | +0.421 [+0.337, +0.507] | +0.467 [+0.377, +0.568] | +0.467 [+0.376, +0.569] | +0.471 [+0.380, +0.572] | +0.467 [+0.376, +0.568] | +0.459 [+0.365, +0.560] | +0.470 [+0.379, +0.569] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.297 [+0.212, +0.395] | +0.350 [+0.265, +0.435] | +0.293 [+0.219, +0.372] | +0.303 [+0.216, +0.405] | +0.298 [+0.212, +0.399] | +0.300 [+0.215, +0.400] | +0.298 [+0.213, +0.400] | +0.289 [+0.207, +0.386] | +0.310 [+0.227, +0.406] |
| GNM (visto) vA, coefficienti | +0.428 [+0.339, +0.528] | +0.402 [+0.305, +0.492] | +0.386 [+0.300, +0.477] | +0.430 [+0.341, +0.531] | +0.428 [+0.336, +0.529] | +0.434 [+0.347, +0.534] | +0.427 [+0.340, +0.528] | +0.424 [+0.333, +0.522] | +0.434 [+0.348, +0.529] |
| GNM (visto) vA, mesh d'identita' SR | +0.282 [+0.197, +0.378] | +0.336 [+0.253, +0.420] | +0.277 [+0.201, +0.355] | +0.292 [+0.206, +0.388] | +0.284 [+0.196, +0.380] | +0.287 [+0.200, +0.380] | +0.280 [+0.197, +0.375] | +0.271 [+0.186, +0.362] | +0.296 [+0.214, +0.389] |
| GNM (visto) vB, coefficienti | -0.053 [-0.136, +0.032] | -0.101 [-0.184, -0.015] | -0.107 [-0.190, -0.024] | -0.047 [-0.131, +0.039] | -0.047 [-0.131, +0.037] | -0.046 [-0.127, +0.038] | -0.047 [-0.128, +0.038] | -0.075 [-0.161, +0.014] | -0.042 [-0.124, +0.042] |
| GNM (visto) vB, mesh d'identita' SR | +0.015 [-0.040, +0.074] | -0.021 [-0.088, +0.043] | -0.030 [-0.085, +0.025] | +0.021 [-0.033, +0.079] | +0.020 [-0.035, +0.080] | +0.019 [-0.034, +0.076] | +0.020 [-0.035, +0.077] | -0.003 [-0.060, +0.058] | +0.025 [-0.028, +0.082] |
| FLAME 2023 Open preregistrato, coefficienti | +0.463 [+0.378, +0.556] | +0.453 [+0.378, +0.529] | +0.420 [+0.340, +0.502] | +0.468 [+0.381, +0.562] | +0.465 [+0.380, +0.559] | +0.460 [+0.372, +0.556] | +0.460 [+0.377, +0.557] | +0.465 [+0.376, +0.556] | +0.468 [+0.381, +0.561] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.335 [+0.255, +0.421] | +0.351 [+0.283, +0.419] | +0.304 [+0.233, +0.377] | +0.328 [+0.243, +0.418] | +0.342 [+0.257, +0.435] | +0.333 [+0.249, +0.420] | +0.322 [+0.238, +0.410] | +0.350 [+0.277, +0.426] | +0.332 [+0.250, +0.416] |
| FLAME 2023 Open vA, coefficienti | +0.466 [+0.380, +0.561] | +0.453 [+0.378, +0.530] | +0.422 [+0.340, +0.506] | +0.471 [+0.383, +0.566] | +0.469 [+0.382, +0.564] | +0.463 [+0.375, +0.561] | +0.464 [+0.380, +0.561] | +0.468 [+0.377, +0.559] | +0.472 [+0.386, +0.566] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.323 [+0.244, +0.408] | +0.330 [+0.260, +0.401] | +0.289 [+0.218, +0.363] | +0.316 [+0.233, +0.407] | +0.330 [+0.246, +0.421] | +0.321 [+0.238, +0.409] | +0.310 [+0.228, +0.399] | +0.337 [+0.265, +0.415] | +0.320 [+0.239, +0.407] |
| FLAME 2023 Open vB, coefficienti | +0.036 [-0.048, +0.123] | -0.001 [-0.095, +0.090] | -0.016 [-0.105, +0.074] | +0.044 [-0.041, +0.133] | +0.045 [-0.042, +0.132] | +0.036 [-0.048, +0.121] | +0.041 [-0.042, +0.128] | +0.017 [-0.073, +0.107] | +0.048 [-0.039, +0.135] |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.024 [-0.032, +0.082] | -0.017 [-0.083, +0.045] | -0.029 [-0.086, +0.028] | +0.029 [-0.027, +0.087] | +0.032 [-0.027, +0.092] | +0.028 [-0.028, +0.085] | +0.025 [-0.033, +0.082] | +0.010 [-0.048, +0.067] | +0.036 [-0.020, +0.094] |
| ArcFace-render, normal map 3 viste (G0) | +0.401 [+0.302, +0.498] | +0.324 [+0.211, +0.426] | +0.335 [+0.235, +0.432] | +0.409 [+0.308, +0.507] | +0.397 [+0.300, +0.492] | +0.400 [+0.300, +0.497] | +0.401 [+0.301, +0.498] | +0.392 [+0.285, +0.497] | +0.448 [+0.362, +0.530] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.385 [+0.291, +0.472] | +0.301 [+0.196, +0.401] | +0.316 [+0.222, +0.406] | +0.383 [+0.287, +0.475] | +0.381 [+0.286, +0.470] | +0.377 [+0.283, +0.468] | +0.376 [+0.280, +0.469] | +0.381 [+0.279, +0.471] | +0.422 [+0.337, +0.506] |
| ShapeDNA k=50 | +0.496 [+0.425, +0.560] | +0.512 [+0.428, +0.585] | +0.469 [+0.397, +0.533] | +0.425 [+0.350, +0.493] | +0.433 [+0.353, +0.503] | +0.431 [+0.357, +0.499] | +0.453 [+0.376, +0.521] | +0.566 [+0.486, +0.641] | +0.255 [+0.169, +0.338] |
| HKS globale | +0.552 [+0.476, +0.618] | +0.546 [+0.467, +0.618] | +0.518 [+0.447, +0.583] | +0.521 [+0.439, +0.596] | +0.533 [+0.450, +0.606] | +0.519 [+0.436, +0.594] | +0.529 [+0.446, +0.601] | +0.564 [+0.476, +0.641] | +0.438 [+0.334, +0.542] |
| WKS globale | +0.592 [+0.524, +0.655] | +0.488 [+0.421, +0.550] | +0.520 [+0.453, +0.580] | +0.564 [+0.499, +0.626] | +0.561 [+0.497, +0.624] | +0.572 [+0.504, +0.635] | +0.585 [+0.516, +0.649] | +0.588 [+0.512, +0.662] | +0.327 [+0.247, +0.414] |
| OpenShape PointBERT | +0.440 [+0.374, +0.504] | +0.366 [+0.298, +0.430] | +0.392 [+0.326, +0.451] | +0.374 [+0.311, +0.437] | +0.391 [+0.320, +0.460] | +0.378 [+0.311, +0.440] | +0.394 [+0.323, +0.463] | +0.404 [+0.305, +0.509] | +0.249 [+0.175, +0.325] |
| Uni3D-g | +0.486 [+0.412, +0.551] | +0.309 [+0.220, +0.387] | +0.403 [+0.331, +0.465] | +0.434 [+0.354, +0.506] | +0.435 [+0.358, +0.503] | +0.422 [+0.347, +0.491] | +0.428 [+0.347, +0.499] | +0.521 [+0.417, +0.627] | +0.242 [+0.150, +0.330] |
| taglia stimata dalla mesh (non oracolo) | +0.599 [+0.493, +0.704] | +0.540 [+0.462, +0.610] | +0.540 [+0.457, +0.618] | +0.607 [+0.502, +0.711] | +0.603 [+0.499, +0.708] | +0.603 [+0.497, +0.708] | +0.604 [+0.498, +0.709] | +0.581 [+0.474, +0.683] | +0.604 [+0.500, +0.709] |
| taglia oracolo (S di FR) | +0.548 [+0.444, +0.651] | +0.476 [+0.366, +0.583] | +0.483 [+0.378, +0.587] | +0.555 [+0.454, +0.658] | +0.552 [+0.447, +0.657] | +0.551 [+0.446, +0.656] | +0.553 [+0.447, +0.658] | +0.531 [+0.423, +0.637] | +0.556 [+0.454, +0.658] |

### FaceScape (dev): Spearman per insieme di righe (seme 796786)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (88724) | crop (46931) | all_cross (135655) | con_perturbazione original (37031) | con_perturbazione remesh (37031) | con_perturbazione down8k (37031) | con_perturbazione up60k (29324) | con_perturbazione noisy (37031) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.661 [0.586, 0.727] | 0.355 [0.277, 0.433] | 0.365 [0.314, 0.415] | 0.669 [0.596, 0.736] | 0.665 [0.592, 0.730] | 0.661 [0.585, 0.730] | 0.660 [0.588, 0.721] | 0.660 [0.585, 0.728] | 0.688 [0.621, 0.749] |
| **factorized s1234, d_F non cal.** | 0.687 [0.612, 0.758] | 0.487 [0.401, 0.574] | 0.438 [0.371, 0.504] | 0.695 [0.620, 0.768] | 0.697 [0.622, 0.767] | 0.684 [0.606, 0.758] | 0.695 [0.623, 0.762] | 0.687 [0.611, 0.759] | 0.700 [0.629, 0.766] |
| factorized s1234, d_P | 0.677 [0.596, 0.750] | 0.457 [0.368, 0.545] | 0.438 [0.368, 0.507] | 0.687 [0.609, 0.761] | 0.682 [0.604, 0.754] | 0.674 [0.592, 0.750] | 0.685 [0.608, 0.755] | 0.678 [0.596, 0.752] | 0.676 [0.599, 0.745] |
| **factorized s2345, d_F cal.*** | 0.669 [0.597, 0.733] | 0.282 [0.214, 0.348] | 0.345 [0.303, 0.387] | 0.678 [0.605, 0.742] | 0.668 [0.596, 0.732] | 0.682 [0.611, 0.745] | 0.662 [0.595, 0.724] | 0.660 [0.586, 0.725] | 0.702 [0.633, 0.764] |
| **factorized s2345, d_F non cal.** | 0.698 [0.619, 0.770] | 0.467 [0.378, 0.552] | 0.415 [0.353, 0.475] | 0.702 [0.620, 0.776] | 0.711 [0.632, 0.783] | 0.704 [0.623, 0.779] | 0.709 [0.630, 0.779] | 0.680 [0.597, 0.753] | 0.707 [0.628, 0.778] |
| factorized s2345, d_P | 0.684 [0.601, 0.760] | 0.425 [0.336, 0.515] | 0.415 [0.351, 0.481] | 0.689 [0.607, 0.767] | 0.694 [0.612, 0.770] | 0.691 [0.607, 0.768] | 0.696 [0.611, 0.770] | 0.667 [0.579, 0.744] | 0.679 [0.593, 0.754] |
| **ctrlfr s1234** | 0.661 [0.593, 0.721] | 0.266 [0.199, 0.333] | 0.325 [0.286, 0.361] | 0.690 [0.625, 0.747] | 0.685 [0.618, 0.742] | 0.694 [0.629, 0.752] | 0.683 [0.616, 0.741] | 0.630 [0.560, 0.688] | 0.725 [0.665, 0.778] |
| **ctrlfr s2345** | 0.653 [0.585, 0.712] | 0.236 [0.176, 0.305] | 0.315 [0.276, 0.352] | 0.677 [0.611, 0.736] | 0.650 [0.584, 0.710] | 0.681 [0.615, 0.739] | 0.677 [0.609, 0.735] | 0.661 [0.589, 0.725] | 0.700 [0.635, 0.752] |
| **C3M e123, d_F cal.*** | 0.665 [0.596, 0.727] | 0.473 [0.402, 0.547] | 0.413 [0.355, 0.469] | 0.671 [0.601, 0.733] | 0.688 [0.620, 0.750] | 0.659 [0.589, 0.722] | 0.684 [0.613, 0.747] | 0.647 [0.575, 0.713] | 0.677 [0.612, 0.735] |
| **C3M e123, d_F non cal.** | 0.682 [0.613, 0.747] | 0.539 [0.460, 0.617] | 0.461 [0.391, 0.528] | 0.685 [0.617, 0.749] | 0.701 [0.634, 0.763] | 0.682 [0.612, 0.747] | 0.697 [0.629, 0.762] | 0.670 [0.597, 0.737] | 0.691 [0.625, 0.751] |
| C3M e123, d_P | 0.675 [0.603, 0.742] | 0.514 [0.434, 0.595] | 0.447 [0.377, 0.514] | 0.679 [0.606, 0.745] | 0.694 [0.623, 0.759] | 0.675 [0.602, 0.742] | 0.690 [0.619, 0.757] | 0.664 [0.590, 0.734] | 0.674 [0.606, 0.737] |
| **C3M e205, d_F cal.*** | 0.711 [0.651, 0.763] | 0.451 [0.381, 0.519] | 0.420 [0.370, 0.468] | 0.725 [0.665, 0.777] | 0.729 [0.670, 0.780] | 0.697 [0.634, 0.751] | 0.742 [0.684, 0.792] | 0.694 [0.633, 0.749] | 0.728 [0.669, 0.776] |
| **C3M e205, d_F non cal.** | 0.694 [0.626, 0.755] | 0.516 [0.439, 0.590] | 0.469 [0.403, 0.533] | 0.700 [0.633, 0.760] | 0.705 [0.637, 0.766] | 0.686 [0.616, 0.748] | 0.715 [0.650, 0.775] | 0.690 [0.623, 0.753] | 0.707 [0.642, 0.767] |
| C3M e205, d_P | 0.685 [0.616, 0.749] | 0.498 [0.419, 0.575] | 0.454 [0.387, 0.517] | 0.691 [0.621, 0.755] | 0.696 [0.626, 0.759] | 0.677 [0.605, 0.742] | 0.707 [0.639, 0.769] | 0.684 [0.612, 0.750] | 0.688 [0.625, 0.750] |
| **ICP + Chamfer in mm (per coppia)** | 0.453 [0.389, 0.514] | 0.292 [0.252, 0.332] | 0.376 [0.319, 0.429] | 0.451 [0.388, 0.510] | 0.548 [0.478, 0.613] | 0.395 [0.342, 0.448] | 0.524 [0.454, 0.590] | 0.486 [0.421, 0.546] | 0.582 [0.514, 0.645] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.416 [0.344, 0.485] | 0.225 [0.161, 0.288] | 0.311 [0.246, 0.373] | 0.403 [0.333, 0.472] | 0.502 [0.423, 0.581] | 0.374 [0.312, 0.434] | 0.483 [0.403, 0.558] | 0.439 [0.366, 0.512] | 0.575 [0.495, 0.653] |
| **NICP per coppia in mm** | 0.374 [0.305, 0.446] | 0.324 [0.261, 0.387] | 0.342 [0.277, 0.410] | 0.368 [0.298, 0.441] | 0.463 [0.382, 0.545] | 0.320 [0.262, 0.381] | 0.440 [0.364, 0.518] | 0.404 [0.331, 0.478] | - |
| **NICP per coppia cs** | 0.346 [0.276, 0.419] | 0.228 [0.167, 0.291] | 0.300 [0.236, 0.369] | 0.340 [0.271, 0.413] | 0.429 [0.347, 0.514] | 0.297 [0.237, 0.358] | 0.409 [0.328, 0.488] | 0.371 [0.299, 0.447] | - |
| **NICP su template in mm** | 0.542 [0.468, 0.605] | 0.070 [0.018, 0.124] | 0.257 [0.219, 0.292] | 0.564 [0.487, 0.626] | 0.557 [0.481, 0.619] | 0.509 [0.438, 0.570] | 0.558 [0.480, 0.624] | 0.530 [0.457, 0.592] | 0.579 [0.509, 0.639] |
| **NICP su template cs** | 0.400 [0.323, 0.474] | 0.053 [0.010, 0.096] | 0.188 [0.149, 0.229] | 0.396 [0.320, 0.471] | 0.390 [0.314, 0.465] | 0.420 [0.338, 0.501] | 0.395 [0.314, 0.473] | 0.404 [0.328, 0.480] | 0.459 [0.381, 0.537] |
| **varifold in mm (massa unitaria)** | 0.264 [0.223, 0.299] | 0.218 [0.172, 0.265] | 0.161 [0.134, 0.186] | 0.368 [0.316, 0.412] | 0.363 [0.312, 0.408] | 0.367 [0.315, 0.412] | 0.382 [0.327, 0.429] | 0.258 [0.179, 0.335] | 0.639 [0.561, 0.706] |
| **GNM (visto) preregistrato, coefficienti** | 0.488 [0.399, 0.561] | 0.328 [0.260, 0.386] | 0.417 [0.341, 0.482] | 0.488 [0.395, 0.564] | 0.493 [0.405, 0.567] | 0.488 [0.398, 0.563] | 0.505 [0.416, 0.581] | 0.471 [0.384, 0.545] | 0.504 [0.418, 0.575] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.465 [0.386, 0.538] | 0.375 [0.313, 0.433] | 0.429 [0.362, 0.494] | 0.468 [0.389, 0.543] | 0.474 [0.397, 0.547] | 0.456 [0.381, 0.526] | 0.478 [0.399, 0.555] | 0.452 [0.373, 0.524] | 0.499 [0.424, 0.569] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.523 [0.424, 0.614] | 0.399 [0.316, 0.476] | 0.465 [0.376, 0.550] | 0.525 [0.427, 0.618] | 0.528 [0.430, 0.622] | 0.528 [0.431, 0.622] | 0.538 [0.437, 0.633] | 0.502 [0.403, 0.594] | 0.535 [0.440, 0.625] |
| **GNM (visto) vA, coefficienti*** | 0.478 [0.387, 0.556] | 0.360 [0.288, 0.422] | 0.424 [0.343, 0.494] | 0.474 [0.380, 0.555] | 0.487 [0.396, 0.566] | 0.482 [0.391, 0.560] | 0.493 [0.401, 0.574] | 0.459 [0.368, 0.537] | 0.487 [0.401, 0.559] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.484 [0.410, 0.554] | 0.403 [0.341, 0.459] | 0.452 [0.383, 0.514] | 0.488 [0.410, 0.561] | 0.494 [0.419, 0.563] | 0.477 [0.404, 0.544] | 0.497 [0.420, 0.570] | 0.469 [0.392, 0.541] | 0.516 [0.442, 0.584] |
| GNM (visto) vA, mesh d'identita' SR* | 0.507 [0.402, 0.609] | 0.418 [0.335, 0.498] | 0.466 [0.369, 0.558] | 0.510 [0.404, 0.611] | 0.516 [0.411, 0.618] | 0.513 [0.410, 0.617] | 0.522 [0.415, 0.624] | 0.483 [0.376, 0.585] | 0.518 [0.420, 0.615] |
| **GNM (visto) vB, coefficienti*** | 0.565 [0.491, 0.636] | 0.549 [0.469, 0.622] | 0.559 [0.482, 0.631] | 0.565 [0.492, 0.638] | 0.564 [0.490, 0.637] | 0.569 [0.491, 0.638] | 0.568 [0.491, 0.639] | 0.565 [0.491, 0.636] | 0.562 [0.484, 0.634] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.678 [0.615, 0.740] | 0.628 [0.564, 0.684] | 0.659 [0.597, 0.719] | 0.680 [0.618, 0.741] | 0.679 [0.616, 0.740] | 0.671 [0.609, 0.735] | 0.683 [0.618, 0.747] | 0.679 [0.614, 0.742] | 0.706 [0.647, 0.763] |
| GNM (visto) vB, mesh d'identita' SR* | 0.767 [0.675, 0.843] | 0.755 [0.681, 0.822] | 0.758 [0.674, 0.830] | 0.766 [0.675, 0.841] | 0.768 [0.677, 0.843] | 0.765 [0.674, 0.841] | 0.768 [0.678, 0.843] | 0.767 [0.674, 0.842] | 0.767 [0.680, 0.842] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.800 [0.741, 0.847] | 0.761 [0.700, 0.812] | 0.783 [0.724, 0.831] | 0.800 [0.741, 0.847] | 0.801 [0.743, 0.849] | 0.794 [0.732, 0.842] | 0.803 [0.742, 0.852] | 0.801 [0.742, 0.850] | 0.814 [0.759, 0.859] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.385 [0.299, 0.467] | 0.283 [0.223, 0.346] | 0.345 [0.275, 0.415] | 0.388 [0.304, 0.472] | 0.390 [0.305, 0.472] | 0.404 [0.316, 0.488] | 0.395 [0.306, 0.478] | 0.350 [0.265, 0.432] | 0.436 [0.350, 0.519] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.449 [0.366, 0.519] | 0.343 [0.281, 0.402] | 0.408 [0.337, 0.471] | 0.452 [0.373, 0.522] | 0.448 [0.365, 0.517] | 0.455 [0.374, 0.525] | 0.462 [0.376, 0.536] | 0.431 [0.345, 0.502] | 0.495 [0.416, 0.564] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.404 [0.306, 0.501] | 0.362 [0.283, 0.446] | 0.387 [0.295, 0.480] | 0.409 [0.310, 0.511] | 0.413 [0.314, 0.511] | 0.423 [0.319, 0.523] | 0.416 [0.316, 0.513] | 0.361 [0.266, 0.464] | 0.445 [0.343, 0.545] |
| **FLAME 2023 Open vA, coefficienti*** | 0.389 [0.304, 0.469] | 0.285 [0.225, 0.346] | 0.348 [0.278, 0.417] | 0.392 [0.308, 0.475] | 0.394 [0.310, 0.474] | 0.408 [0.320, 0.490] | 0.399 [0.310, 0.483] | 0.354 [0.270, 0.435] | 0.440 [0.355, 0.520] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.453 [0.375, 0.522] | 0.355 [0.293, 0.414] | 0.416 [0.348, 0.477] | 0.457 [0.380, 0.527] | 0.453 [0.373, 0.521] | 0.457 [0.378, 0.524] | 0.467 [0.386, 0.539] | 0.437 [0.356, 0.509] | 0.496 [0.418, 0.565] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.409 [0.309, 0.511] | 0.375 [0.291, 0.462] | 0.395 [0.301, 0.491] | 0.415 [0.314, 0.521] | 0.419 [0.318, 0.520] | 0.427 [0.324, 0.532] | 0.422 [0.321, 0.522] | 0.366 [0.266, 0.471] | 0.448 [0.346, 0.553] |
| **FLAME 2023 Open vB, coefficienti*** | 0.549 [0.467, 0.625] | 0.514 [0.431, 0.591] | 0.535 [0.455, 0.612] | 0.550 [0.468, 0.627] | 0.548 [0.462, 0.625] | 0.553 [0.469, 0.629] | 0.554 [0.469, 0.631] | 0.546 [0.462, 0.621] | 0.548 [0.466, 0.625] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.630 [0.555, 0.695] | 0.562 [0.492, 0.625] | 0.605 [0.533, 0.668] | 0.631 [0.557, 0.697] | 0.621 [0.545, 0.685] | 0.630 [0.555, 0.696] | 0.638 [0.562, 0.705] | 0.633 [0.556, 0.700] | 0.656 [0.582, 0.717] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.711 [0.614, 0.798] | 0.703 [0.611, 0.789] | 0.707 [0.609, 0.792] | 0.712 [0.615, 0.797] | 0.712 [0.617, 0.797] | 0.711 [0.614, 0.798] | 0.714 [0.618, 0.799] | 0.707 [0.613, 0.793] | 0.713 [0.617, 0.799] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.731 [0.657, 0.794] | 0.685 [0.610, 0.750] | 0.713 [0.641, 0.777] | 0.732 [0.659, 0.795] | 0.726 [0.652, 0.789] | 0.731 [0.657, 0.794] | 0.736 [0.661, 0.800] | 0.732 [0.658, 0.795] | 0.746 [0.674, 0.806] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.300 [0.231, 0.370] | 0.308 [0.235, 0.383] | 0.303 [0.231, 0.374] | 0.312 [0.239, 0.384] | 0.330 [0.257, 0.405] | 0.310 [0.238, 0.383] | 0.315 [0.240, 0.391] | 0.287 [0.215, 0.363] | 0.274 [0.211, 0.336] |
| **ShapeDNA k=50** | 0.091 [0.061, 0.122] | 0.011 [-0.034, 0.054] | 0.043 [0.028, 0.059] | 0.159 [0.115, 0.206] | 0.144 [0.103, 0.187] | 0.157 [0.113, 0.204] | 0.141 [0.099, 0.183] | -0.018 [-0.082, 0.044] | 0.287 [0.225, 0.352] |
| **HKS globale** | 0.034 [0.013, 0.056] | 0.011 [-0.026, 0.046] | 0.020 [0.007, 0.036] | 0.057 [0.021, 0.093] | 0.050 [0.022, 0.079] | 0.056 [0.021, 0.092] | 0.051 [0.017, 0.087] | 0.005 [-0.050, 0.053] | 0.109 [0.046, 0.175] |
| **WKS globale** | 0.016 [0.000, 0.034] | 0.010 [-0.030, 0.050] | 0.011 [-0.001, 0.023] | 0.055 [0.031, 0.080] | 0.048 [0.026, 0.071] | 0.054 [0.029, 0.079] | -0.002 [-0.024, 0.021] | -0.030 [-0.106, 0.042] | 0.171 [0.113, 0.232] |
| **taglia stimata dalla mesh (non oracolo)** | 0.204 [0.136, 0.266] | 0.012 [-0.026, 0.051] | 0.088 [0.059, 0.114] | 0.210 [0.139, 0.274] | 0.160 [0.101, 0.215] | 0.212 [0.140, 0.276] | 0.233 [0.160, 0.300] | 0.216 [0.146, 0.281] | 0.296 [0.208, 0.376] |
| **taglia oracolo (S di FR)** | 0.464 [0.345, 0.571] | 0.464 [0.344, 0.573] | 0.464 [0.345, 0.572] | 0.464 [0.344, 0.572] | 0.464 [0.344, 0.572] | 0.464 [0.344, 0.572] | 0.465 [0.345, 0.574] | 0.464 [0.344, 0.572] | 0.464 [0.344, 0.573] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (88724) | crop (46931) | all_cross (135655) | con_perturbazione original (37031) | con_perturbazione remesh (37031) | con_perturbazione down8k (37031) | con_perturbazione up60k (29324) | con_perturbazione noisy (37031) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.692 [0.623, 0.753] | 0.385 [0.315, 0.458] | 0.386 [0.335, 0.432] | 0.701 [0.632, 0.762] | 0.696 [0.626, 0.757] | 0.695 [0.625, 0.757] | 0.689 [0.621, 0.747] | 0.693 [0.623, 0.753] | 0.712 [0.648, 0.769] |
| factorized s1234, d_F non cal. | 0.745 [0.677, 0.801] | 0.535 [0.452, 0.612] | 0.474 [0.409, 0.534] | 0.753 [0.686, 0.809] | 0.753 [0.687, 0.808] | 0.743 [0.675, 0.802] | 0.750 [0.682, 0.805] | 0.747 [0.680, 0.804] | 0.754 [0.691, 0.804] |
| **factorized s1234, d_P** | 0.747 [0.677, 0.804] | 0.516 [0.430, 0.596] | 0.482 [0.416, 0.544] | 0.756 [0.687, 0.813] | 0.751 [0.683, 0.807] | 0.746 [0.676, 0.805] | 0.753 [0.682, 0.809] | 0.753 [0.682, 0.809] | 0.742 [0.678, 0.794] |
| factorized s2345, d_F cal.* | 0.687 [0.620, 0.748] | 0.310 [0.246, 0.376] | 0.358 [0.314, 0.398] | 0.696 [0.629, 0.757] | 0.686 [0.620, 0.747] | 0.702 [0.634, 0.762] | 0.681 [0.614, 0.741] | 0.678 [0.610, 0.739] | 0.712 [0.647, 0.770] |
| factorized s2345, d_F non cal. | 0.751 [0.686, 0.809] | 0.505 [0.417, 0.584] | 0.445 [0.388, 0.502] | 0.755 [0.690, 0.813] | 0.763 [0.701, 0.819] | 0.758 [0.693, 0.816] | 0.760 [0.696, 0.819] | 0.734 [0.663, 0.793] | 0.757 [0.695, 0.812] |
| **factorized s2345, d_P** | 0.753 [0.687, 0.813] | 0.474 [0.387, 0.557] | 0.456 [0.395, 0.515] | 0.759 [0.693, 0.818] | 0.762 [0.698, 0.820] | 0.761 [0.694, 0.821] | 0.764 [0.700, 0.823] | 0.739 [0.668, 0.799] | 0.745 [0.680, 0.800] |
| ctrlfr s1234 | 0.619 [0.543, 0.688] | 0.247 [0.180, 0.314] | 0.306 [0.263, 0.347] | 0.645 [0.569, 0.711] | 0.642 [0.565, 0.711] | 0.650 [0.575, 0.716] | 0.639 [0.563, 0.707] | 0.588 [0.512, 0.656] | 0.678 [0.608, 0.741] |
| ctrlfr s2345 | 0.627 [0.554, 0.691] | 0.243 [0.182, 0.311] | 0.306 [0.265, 0.344] | 0.650 [0.575, 0.715] | 0.628 [0.556, 0.692] | 0.655 [0.582, 0.721] | 0.647 [0.574, 0.713] | 0.634 [0.557, 0.700] | 0.668 [0.598, 0.731] |
| C3M e123, d_F cal.* | 0.687 [0.624, 0.746] | 0.501 [0.428, 0.572] | 0.431 [0.373, 0.485] | 0.692 [0.630, 0.750] | 0.709 [0.647, 0.767] | 0.682 [0.618, 0.741] | 0.703 [0.640, 0.760] | 0.674 [0.605, 0.736] | 0.692 [0.634, 0.747] |
| C3M e123, d_F non cal. | 0.741 [0.679, 0.796] | 0.584 [0.513, 0.652] | 0.500 [0.434, 0.560] | 0.743 [0.682, 0.796] | 0.759 [0.702, 0.811] | 0.740 [0.678, 0.794] | 0.754 [0.696, 0.807] | 0.736 [0.670, 0.792] | 0.746 [0.688, 0.795] |
| **C3M e123, d_P** | 0.746 [0.686, 0.799] | 0.576 [0.506, 0.644] | 0.494 [0.429, 0.554] | 0.748 [0.689, 0.800] | 0.764 [0.709, 0.814] | 0.745 [0.684, 0.798] | 0.760 [0.702, 0.813] | 0.743 [0.678, 0.801] | 0.740 [0.682, 0.788] |
| C3M e205, d_F cal.* | 0.708 [0.649, 0.763] | 0.465 [0.400, 0.534] | 0.423 [0.371, 0.471] | 0.721 [0.662, 0.775] | 0.727 [0.669, 0.778] | 0.696 [0.634, 0.752] | 0.734 [0.674, 0.784] | 0.694 [0.633, 0.753] | 0.720 [0.664, 0.771] |
| C3M e205, d_F non cal. | 0.751 [0.691, 0.801] | 0.556 [0.480, 0.625] | 0.506 [0.439, 0.565] | 0.756 [0.697, 0.806] | 0.764 [0.707, 0.813] | 0.743 [0.683, 0.796] | 0.771 [0.712, 0.820] | 0.752 [0.690, 0.802] | 0.763 [0.708, 0.811] |
| **C3M e205, d_P** | 0.756 [0.695, 0.807] | 0.555 [0.479, 0.622] | 0.499 [0.435, 0.557] | 0.760 [0.701, 0.810] | 0.768 [0.711, 0.815] | 0.747 [0.686, 0.800] | 0.776 [0.718, 0.824] | 0.759 [0.698, 0.809] | 0.756 [0.699, 0.805] |
| **ICP + Chamfer in mm (per coppia)** | 0.477 [0.414, 0.533] | 0.306 [0.267, 0.343] | 0.396 [0.339, 0.446] | 0.473 [0.411, 0.529] | 0.578 [0.513, 0.638] | 0.415 [0.361, 0.464] | 0.554 [0.488, 0.614] | 0.509 [0.447, 0.566] | 0.608 [0.544, 0.666] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.478 [0.412, 0.536] | 0.269 [0.211, 0.326] | 0.362 [0.304, 0.416] | 0.462 [0.399, 0.519] | 0.579 [0.512, 0.639] | 0.426 [0.371, 0.477] | 0.557 [0.488, 0.619] | 0.505 [0.440, 0.563] | 0.660 [0.594, 0.717] |
| **NICP per coppia in mm** | 0.406 [0.336, 0.472] | 0.348 [0.282, 0.409] | 0.370 [0.303, 0.433] | 0.399 [0.331, 0.464] | 0.504 [0.424, 0.577] | 0.345 [0.287, 0.402] | 0.480 [0.403, 0.552] | 0.436 [0.366, 0.502] | - |
| **NICP per coppia cs** | 0.398 [0.329, 0.464] | 0.274 [0.213, 0.331] | 0.350 [0.282, 0.411] | 0.389 [0.321, 0.452] | 0.495 [0.415, 0.569] | 0.340 [0.282, 0.394] | 0.472 [0.394, 0.544] | 0.427 [0.357, 0.494] | - |
| **NICP su template in mm** | 0.399 [0.320, 0.473] | 0.047 [-0.003, 0.097] | 0.189 [0.149, 0.225] | 0.415 [0.332, 0.496] | 0.410 [0.328, 0.489] | 0.383 [0.308, 0.456] | 0.410 [0.323, 0.490] | 0.381 [0.305, 0.454] | 0.425 [0.346, 0.503] |
| **NICP su template cs** | 0.436 [0.363, 0.508] | 0.061 [0.017, 0.107] | 0.205 [0.166, 0.244] | 0.427 [0.354, 0.501] | 0.424 [0.351, 0.497] | 0.458 [0.380, 0.530] | 0.429 [0.351, 0.507] | 0.443 [0.370, 0.516] | 0.494 [0.417, 0.566] |
| **varifold in mm (massa unitaria)** | 0.281 [0.246, 0.313] | 0.228 [0.182, 0.273] | 0.171 [0.145, 0.194] | 0.384 [0.342, 0.422] | 0.380 [0.337, 0.419] | 0.384 [0.342, 0.422] | 0.399 [0.354, 0.438] | 0.309 [0.236, 0.377] | 0.659 [0.595, 0.717] |
| **GNM (visto) preregistrato, coefficienti** | 0.460 [0.376, 0.538] | 0.298 [0.227, 0.367] | 0.389 [0.319, 0.461] | 0.464 [0.375, 0.543] | 0.466 [0.382, 0.544] | 0.466 [0.381, 0.545] | 0.479 [0.395, 0.559] | 0.433 [0.349, 0.511] | 0.475 [0.395, 0.553] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.371 [0.305, 0.436] | 0.299 [0.241, 0.359] | 0.342 [0.283, 0.401] | 0.376 [0.311, 0.443] | 0.377 [0.312, 0.444] | 0.368 [0.304, 0.433] | 0.385 [0.319, 0.451] | 0.353 [0.287, 0.419] | 0.392 [0.322, 0.463] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.576 [0.483, 0.663] | 0.412 [0.324, 0.495] | 0.503 [0.412, 0.586] | 0.578 [0.482, 0.665] | 0.583 [0.491, 0.671] | 0.587 [0.490, 0.675] | 0.594 [0.498, 0.680] | 0.548 [0.454, 0.632] | 0.587 [0.494, 0.668] |
| **GNM (visto) vA, coefficienti*** | 0.477 [0.389, 0.562] | 0.345 [0.269, 0.420] | 0.418 [0.340, 0.496] | 0.477 [0.388, 0.561] | 0.488 [0.400, 0.573] | 0.485 [0.396, 0.570] | 0.495 [0.405, 0.581] | 0.448 [0.360, 0.533] | 0.485 [0.402, 0.567] |
| GNM (visto) vA, mesh d'identita' FR* | 0.392 [0.327, 0.460] | 0.327 [0.269, 0.388] | 0.366 [0.305, 0.427] | 0.398 [0.332, 0.467] | 0.399 [0.335, 0.468] | 0.390 [0.326, 0.457] | 0.406 [0.339, 0.474] | 0.373 [0.306, 0.440] | 0.414 [0.346, 0.486] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.570 [0.465, 0.663] | 0.441 [0.351, 0.526] | 0.514 [0.417, 0.602] | 0.572 [0.466, 0.666] | 0.580 [0.475, 0.675] | 0.581 [0.473, 0.675] | 0.586 [0.480, 0.677] | 0.541 [0.439, 0.632] | 0.581 [0.481, 0.669] |
| **GNM (visto) vB, coefficienti*** | 0.618 [0.548, 0.683] | 0.601 [0.534, 0.664] | 0.611 [0.543, 0.676] | 0.618 [0.549, 0.682] | 0.618 [0.548, 0.683] | 0.618 [0.550, 0.682] | 0.621 [0.553, 0.687] | 0.618 [0.547, 0.683] | 0.615 [0.547, 0.677] |
| GNM (visto) vB, mesh d'identita' FR* | 0.608 [0.536, 0.683] | 0.566 [0.497, 0.635] | 0.592 [0.524, 0.664] | 0.611 [0.539, 0.685] | 0.606 [0.534, 0.681] | 0.604 [0.531, 0.679] | 0.614 [0.541, 0.692] | 0.607 [0.531, 0.684] | 0.628 [0.555, 0.705] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.854 [0.803, 0.894] | 0.830 [0.778, 0.873] | 0.839 [0.786, 0.881] | 0.853 [0.803, 0.892] | 0.855 [0.806, 0.895] | 0.853 [0.802, 0.893] | 0.856 [0.805, 0.896] | 0.854 [0.803, 0.894] | 0.855 [0.807, 0.895] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.783 [0.724, 0.839] | 0.744 [0.682, 0.801] | 0.766 [0.705, 0.822] | 0.785 [0.727, 0.839] | 0.783 [0.724, 0.838] | 0.780 [0.721, 0.836] | 0.788 [0.730, 0.843] | 0.783 [0.721, 0.840] | 0.796 [0.739, 0.849] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.396 [0.311, 0.476] | 0.279 [0.215, 0.348] | 0.351 [0.277, 0.425] | 0.404 [0.319, 0.488] | 0.401 [0.317, 0.482] | 0.416 [0.329, 0.496] | 0.409 [0.324, 0.492] | 0.353 [0.265, 0.438] | 0.447 [0.361, 0.527] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.379 [0.305, 0.453] | 0.294 [0.231, 0.355] | 0.346 [0.279, 0.411] | 0.384 [0.311, 0.461] | 0.382 [0.305, 0.459] | 0.389 [0.315, 0.463] | 0.393 [0.316, 0.470] | 0.352 [0.278, 0.423] | 0.417 [0.339, 0.495] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.469 [0.372, 0.561] | 0.405 [0.323, 0.489] | 0.444 [0.352, 0.532] | 0.481 [0.379, 0.573] | 0.478 [0.381, 0.567] | 0.491 [0.390, 0.581] | 0.486 [0.382, 0.579] | 0.416 [0.314, 0.512] | 0.517 [0.415, 0.607] |
| **FLAME 2023 Open vA, coefficienti*** | 0.397 [0.312, 0.476] | 0.279 [0.215, 0.346] | 0.351 [0.279, 0.424] | 0.406 [0.321, 0.488] | 0.402 [0.319, 0.483] | 0.417 [0.330, 0.496] | 0.410 [0.327, 0.491] | 0.354 [0.268, 0.437] | 0.448 [0.364, 0.528] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.373 [0.301, 0.445] | 0.297 [0.236, 0.359] | 0.344 [0.278, 0.409] | 0.378 [0.305, 0.452] | 0.376 [0.303, 0.451] | 0.381 [0.308, 0.453] | 0.387 [0.313, 0.460] | 0.348 [0.277, 0.417] | 0.406 [0.331, 0.482] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.480 [0.380, 0.575] | 0.424 [0.340, 0.509] | 0.458 [0.364, 0.547] | 0.491 [0.388, 0.586] | 0.489 [0.391, 0.581] | 0.500 [0.398, 0.594] | 0.498 [0.395, 0.593] | 0.425 [0.319, 0.525] | 0.525 [0.422, 0.618] |
| **FLAME 2023 Open vB, coefficienti*** | 0.592 [0.512, 0.664] | 0.554 [0.476, 0.629] | 0.577 [0.496, 0.648] | 0.591 [0.510, 0.664] | 0.590 [0.508, 0.661] | 0.594 [0.514, 0.666] | 0.599 [0.518, 0.669] | 0.591 [0.514, 0.660] | 0.594 [0.515, 0.665] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.590 [0.516, 0.662] | 0.531 [0.462, 0.600] | 0.568 [0.497, 0.638] | 0.590 [0.517, 0.662] | 0.584 [0.512, 0.656] | 0.591 [0.515, 0.663] | 0.599 [0.523, 0.671] | 0.589 [0.516, 0.660] | 0.610 [0.534, 0.683] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.820 [0.757, 0.870] | 0.807 [0.742, 0.859] | 0.814 [0.749, 0.865] | 0.822 [0.760, 0.870] | 0.821 [0.757, 0.871] | 0.821 [0.757, 0.872] | 0.825 [0.762, 0.874] | 0.814 [0.751, 0.866] | 0.824 [0.760, 0.874] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.742 [0.677, 0.802] | 0.696 [0.629, 0.757] | 0.725 [0.660, 0.784] | 0.743 [0.679, 0.801] | 0.739 [0.673, 0.795] | 0.744 [0.678, 0.803] | 0.749 [0.683, 0.806] | 0.739 [0.673, 0.801] | 0.757 [0.693, 0.812] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.348 [0.284, 0.414] | 0.369 [0.302, 0.440] | 0.355 [0.290, 0.422] | 0.365 [0.301, 0.434] | 0.381 [0.314, 0.452] | 0.364 [0.300, 0.433] | 0.369 [0.301, 0.437] | 0.321 [0.252, 0.393] | 0.322 [0.265, 0.381] |
| **ShapeDNA k=50** | 0.117 [0.089, 0.144] | -0.002 [-0.045, 0.036] | 0.052 [0.038, 0.067] | 0.195 [0.155, 0.237] | 0.177 [0.138, 0.213] | 0.193 [0.153, 0.234] | 0.173 [0.132, 0.211] | 0.027 [-0.035, 0.083] | 0.327 [0.262, 0.388] |
| **HKS globale** | 0.042 [0.020, 0.066] | -0.006 [-0.043, 0.028] | 0.020 [0.007, 0.036] | 0.068 [0.031, 0.107] | 0.058 [0.028, 0.089] | 0.068 [0.032, 0.106] | 0.061 [0.027, 0.098] | 0.018 [-0.033, 0.068] | 0.129 [0.068, 0.191] |
| **WKS globale** | 0.017 [0.002, 0.034] | 0.005 [-0.035, 0.043] | 0.010 [-0.002, 0.023] | 0.056 [0.032, 0.082] | 0.050 [0.028, 0.073] | 0.056 [0.033, 0.083] | 0.004 [-0.016, 0.024] | -0.033 [-0.112, 0.042] | 0.171 [0.109, 0.236] |
| **taglia stimata dalla mesh (non oracolo)** | 0.094 [0.044, 0.144] | 0.012 [-0.029, 0.055] | 0.041 [0.019, 0.062] | 0.098 [0.044, 0.151] | 0.075 [0.033, 0.114] | 0.099 [0.045, 0.152] | 0.111 [0.053, 0.169] | 0.096 [0.044, 0.147] | 0.126 [0.056, 0.196] |
| **taglia oracolo (S di FR)** | 0.119 [0.022, 0.229] | 0.118 [0.022, 0.228] | 0.119 [0.022, 0.228] | 0.118 [0.022, 0.228] | 0.118 [0.022, 0.228] | 0.118 [0.022, 0.228] | 0.121 [0.021, 0.230] | 0.118 [0.022, 0.228] | 0.118 [0.023, 0.229] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.207 [+0.160, +0.257] | +0.062 [-0.002, +0.130] | -0.011 [-0.049, +0.025] | +0.218 [+0.166, +0.271] | +0.117 [+0.067, +0.169] | +0.266 [+0.221, +0.311] | +0.136 [+0.087, +0.188] | +0.174 [+0.122, +0.225] | +0.106 [+0.055, +0.155] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.245 [+0.192, +0.300] | +0.130 [+0.066, +0.197] | +0.054 [+0.013, +0.096] | +0.266 [+0.211, +0.322] | +0.163 [+0.109, +0.225] | +0.287 [+0.239, +0.337] | +0.178 [+0.122, +0.237] | +0.222 [+0.164, +0.278] | +0.113 [+0.055, +0.176] |
| NICP per coppia in mm | +0.286 [+0.229, +0.345] | +0.031 [-0.039, +0.097] | +0.023 [-0.024, +0.069] | +0.301 [+0.243, +0.362] | +0.202 [+0.143, +0.266] | +0.341 [+0.287, +0.395] | +0.220 [+0.160, +0.279] | +0.257 [+0.195, +0.320] | - |
| NICP per coppia cs | +0.314 [+0.260, +0.370] | +0.126 [+0.055, +0.194] | +0.064 [+0.020, +0.110] | +0.329 [+0.268, +0.388] | +0.236 [+0.177, +0.299] | +0.364 [+0.310, +0.414] | +0.251 [+0.192, +0.313] | +0.289 [+0.229, +0.350] | - |
| NICP su template in mm | +0.119 [+0.030, +0.210] | +0.285 [+0.189, +0.379] | +0.108 [+0.050, +0.167] | +0.105 [+0.017, +0.200] | +0.109 [+0.022, +0.202] | +0.152 [+0.066, +0.246] | +0.102 [+0.012, +0.198] | +0.131 [+0.039, +0.225] | +0.109 [+0.026, +0.198] |
| NICP su template cs | +0.261 [+0.180, +0.339] | +0.301 [+0.224, +0.393] | +0.177 [+0.130, +0.225] | +0.273 [+0.196, +0.351] | +0.275 [+0.195, +0.350] | +0.241 [+0.157, +0.326] | +0.265 [+0.180, +0.347] | +0.257 [+0.178, +0.337] | +0.229 [+0.151, +0.310] |
| varifold in mm (massa unitaria) | +0.396 [+0.349, +0.437] | +0.136 [+0.076, +0.200] | +0.204 [+0.169, +0.235] | +0.301 [+0.255, +0.342] | +0.302 [+0.258, +0.342] | +0.294 [+0.246, +0.336] | +0.278 [+0.231, +0.319] | +0.402 [+0.337, +0.471] | +0.049 [-0.000, +0.102] |
| GNM (visto) preregistrato, coefficienti | +0.173 [+0.086, +0.264] | +0.026 [-0.054, +0.121] | -0.053 [-0.125, +0.028] | +0.181 [+0.091, +0.281] | +0.173 [+0.087, +0.263] | +0.173 [+0.086, +0.268] | +0.155 [+0.064, +0.248] | +0.190 [+0.103, +0.282] | +0.184 [+0.097, +0.276] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.196 [+0.108, +0.288] | -0.021 [-0.107, +0.076] | -0.065 [-0.142, +0.010] | +0.201 [+0.114, +0.292] | +0.191 [+0.104, +0.280] | +0.205 [+0.117, +0.294] | +0.182 [+0.093, +0.273] | +0.209 [+0.114, +0.304] | +0.189 [+0.104, +0.277] |
| GNM (visto) vA, coefficienti | +0.183 [+0.092, +0.279] | -0.005 [-0.086, +0.093] | -0.059 [-0.130, +0.026] | +0.194 [+0.098, +0.295] | +0.178 [+0.084, +0.273] | +0.179 [+0.085, +0.277] | +0.167 [+0.077, +0.261] | +0.201 [+0.112, +0.300] | +0.201 [+0.115, +0.295] |
| GNM (visto) vA, mesh d'identita' FR | +0.177 [+0.089, +0.266] | -0.049 [-0.136, +0.049] | -0.087 [-0.163, -0.012] | +0.181 [+0.096, +0.271] | +0.172 [+0.084, +0.258] | +0.184 [+0.097, +0.274] | +0.164 [+0.071, +0.253] | +0.192 [+0.100, +0.282] | +0.172 [+0.088, +0.259] |
| GNM (visto) vB, coefficienti | +0.095 [+0.016, +0.163] | -0.194 [-0.291, -0.112] | -0.194 [-0.265, -0.132] | +0.104 [+0.024, +0.173] | +0.101 [+0.022, +0.168] | +0.092 [+0.017, +0.162] | +0.092 [+0.016, +0.163] | +0.096 [+0.015, +0.164] | +0.126 [+0.052, +0.191] |
| GNM (visto) vB, mesh d'identita' FR | -0.018 [-0.085, +0.043] | -0.274 [-0.347, -0.197] | -0.295 [-0.354, -0.239] | -0.011 [-0.080, +0.049] | -0.013 [-0.081, +0.047] | -0.010 [-0.077, +0.050] | -0.023 [-0.092, +0.037] | -0.018 [-0.089, +0.043] | -0.018 [-0.081, +0.039] |
| GNM (visto) vB, composizione S_B + k d_P | -0.139 [-0.194, -0.090] | -0.406 [-0.469, -0.343] | -0.418 [-0.461, -0.374] | -0.131 [-0.186, -0.083] | -0.136 [-0.190, -0.086] | -0.133 [-0.189, -0.081] | -0.143 [-0.199, -0.096] | -0.141 [-0.197, -0.092] | -0.126 [-0.179, -0.077] |
| FLAME 2023 Open preregistrato, coefficienti | +0.276 [+0.189, +0.374] | +0.071 [-0.014, +0.164] | +0.019 [-0.051, +0.097] | +0.281 [+0.190, +0.379] | +0.275 [+0.187, +0.369] | +0.257 [+0.168, +0.358] | +0.265 [+0.177, +0.363] | +0.310 [+0.220, +0.414] | +0.253 [+0.162, +0.347] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.212 [+0.123, +0.305] | +0.011 [-0.072, +0.104] | -0.043 [-0.114, +0.037] | +0.217 [+0.125, +0.312] | +0.217 [+0.128, +0.312] | +0.206 [+0.117, +0.299] | +0.198 [+0.106, +0.294] | +0.230 [+0.136, +0.333] | +0.193 [+0.108, +0.285] |
| FLAME 2023 Open vA, coefficienti | +0.272 [+0.186, +0.368] | +0.070 [-0.013, +0.162] | +0.017 [-0.054, +0.090] | +0.277 [+0.188, +0.372] | +0.271 [+0.183, +0.364] | +0.253 [+0.164, +0.353] | +0.261 [+0.173, +0.357] | +0.306 [+0.216, +0.410] | +0.249 [+0.161, +0.343] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.207 [+0.117, +0.300] | -0.001 [-0.087, +0.092] | -0.051 [-0.124, +0.027] | +0.212 [+0.121, +0.303] | +0.212 [+0.122, +0.305] | +0.204 [+0.117, +0.297] | +0.194 [+0.102, +0.284] | +0.224 [+0.130, +0.328] | +0.192 [+0.106, +0.282] |
| FLAME 2023 Open vB, coefficienti | +0.111 [+0.038, +0.178] | -0.159 [-0.252, -0.070] | -0.170 [-0.237, -0.106] | +0.119 [+0.042, +0.189] | +0.117 [+0.044, +0.184] | +0.108 [+0.035, +0.176] | +0.106 [+0.032, +0.172] | +0.115 [+0.040, +0.182] | +0.140 [+0.071, +0.207] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.031 [-0.036, +0.089] | -0.207 [-0.278, -0.133] | -0.240 [-0.297, -0.182] | +0.038 [-0.023, +0.096] | +0.045 [-0.023, +0.103] | +0.031 [-0.035, +0.091] | +0.022 [-0.046, +0.083] | +0.027 [-0.042, +0.090] | +0.032 [-0.031, +0.091] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.070 [-0.127, -0.017] | -0.330 [-0.392, -0.262] | -0.349 [-0.402, -0.296] | -0.063 [-0.121, -0.009] | -0.060 [-0.120, -0.007] | -0.070 [-0.128, -0.018] | -0.076 [-0.134, -0.021] | -0.071 [-0.128, -0.016] | -0.058 [-0.113, -0.005] |
| ArcFace-render, normal map 3 viste (G0) | +0.360 [+0.285, +0.432] | +0.047 [-0.048, +0.135] | +0.062 [-0.004, +0.130] | +0.357 [+0.283, +0.433] | +0.336 [+0.258, +0.411] | +0.351 [+0.276, +0.425] | +0.345 [+0.269, +0.419] | +0.373 [+0.282, +0.458] | +0.414 [+0.349, +0.481] |
| ShapeDNA k=50 | +0.570 [+0.502, +0.628] | +0.344 [+0.252, +0.433] | +0.322 [+0.276, +0.364] | +0.510 [+0.443, +0.570] | +0.521 [+0.457, +0.577] | +0.504 [+0.436, +0.565] | +0.519 [+0.452, +0.579] | +0.679 [+0.587, +0.771] | +0.401 [+0.338, +0.469] |
| HKS globale | +0.627 [+0.557, +0.687] | +0.344 [+0.247, +0.442] | +0.344 [+0.295, +0.392] | +0.612 [+0.541, +0.675] | +0.615 [+0.544, +0.677] | +0.605 [+0.533, +0.670] | +0.609 [+0.537, +0.673] | +0.655 [+0.578, +0.732] | +0.579 [+0.499, +0.656] |
| WKS globale | +0.644 [+0.572, +0.707] | +0.345 [+0.258, +0.440] | +0.354 [+0.305, +0.402] | +0.614 [+0.547, +0.677] | +0.617 [+0.551, +0.679] | +0.607 [+0.537, +0.671] | +0.662 [+0.589, +0.725] | +0.691 [+0.591, +0.786] | +0.517 [+0.443, +0.591] |
| taglia stimata dalla mesh (non oracolo) | +0.456 [+0.354, +0.560] | +0.343 [+0.263, +0.426] | +0.276 [+0.216, +0.336] | +0.459 [+0.352, +0.563] | +0.505 [+0.407, +0.602] | +0.449 [+0.343, +0.555] | +0.427 [+0.319, +0.530] | +0.444 [+0.340, +0.552] | +0.392 [+0.275, +0.506] |
| taglia oracolo (S di FR) | +0.196 [+0.047, +0.355] | -0.109 [-0.267, +0.056] | -0.100 [-0.233, +0.045] | +0.205 [+0.055, +0.364] | +0.201 [+0.053, +0.359] | +0.197 [+0.046, +0.357] | +0.195 [+0.048, +0.351] | +0.196 [+0.047, +0.357] | +0.224 [+0.072, +0.377] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.234 [+0.183, +0.285] | +0.195 [+0.128, +0.264] | +0.062 [+0.020, +0.106] | +0.244 [+0.190, +0.299] | +0.149 [+0.097, +0.199] | +0.289 [+0.240, +0.336] | +0.171 [+0.121, +0.225] | +0.201 [+0.145, +0.254] | +0.118 [+0.063, +0.170] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.272 [+0.220, +0.321] | +0.262 [+0.199, +0.328] | +0.127 [+0.088, +0.170] | +0.293 [+0.238, +0.344] | +0.195 [+0.142, +0.247] | +0.310 [+0.262, +0.355] | +0.213 [+0.159, +0.266] | +0.248 [+0.190, +0.302] | +0.126 [+0.069, +0.180] |
| NICP per coppia in mm | +0.313 [+0.253, +0.375] | +0.163 [+0.095, +0.235] | +0.096 [+0.047, +0.146] | +0.327 [+0.265, +0.389] | +0.234 [+0.171, +0.297] | +0.364 [+0.309, +0.417] | +0.255 [+0.195, +0.318] | +0.283 [+0.217, +0.350] | - |
| NICP per coppia cs | +0.341 [+0.282, +0.400] | +0.259 [+0.191, +0.327] | +0.138 [+0.088, +0.188] | +0.356 [+0.294, +0.415] | +0.268 [+0.205, +0.329] | +0.386 [+0.333, +0.438] | +0.287 [+0.228, +0.348] | +0.316 [+0.251, +0.377] | - |
| NICP su template in mm | +0.146 [+0.055, +0.241] | +0.417 [+0.320, +0.513] | +0.181 [+0.110, +0.258] | +0.131 [+0.040, +0.230] | +0.140 [+0.052, +0.234] | +0.175 [+0.083, +0.269] | +0.137 [+0.042, +0.234] | +0.157 [+0.059, +0.253] | +0.121 [+0.031, +0.214] |
| NICP su template cs | +0.287 [+0.206, +0.369] | +0.434 [+0.352, +0.522] | +0.250 [+0.191, +0.307] | +0.299 [+0.220, +0.380] | +0.307 [+0.226, +0.384] | +0.264 [+0.177, +0.351] | +0.300 [+0.216, +0.386] | +0.283 [+0.199, +0.366] | +0.241 [+0.159, +0.324] |
| varifold in mm (massa unitaria) | +0.423 [+0.376, +0.467] | +0.269 [+0.209, +0.327] | +0.277 [+0.229, +0.325] | +0.328 [+0.280, +0.369] | +0.334 [+0.291, +0.373] | +0.317 [+0.269, +0.360] | +0.313 [+0.269, +0.352] | +0.429 [+0.365, +0.494] | +0.061 [+0.014, +0.108] |
| GNM (visto) preregistrato, coefficienti | +0.200 [+0.103, +0.295] | +0.159 [+0.076, +0.250] | +0.021 [-0.060, +0.106] | +0.207 [+0.108, +0.305] | +0.204 [+0.108, +0.296] | +0.195 [+0.098, +0.293] | +0.191 [+0.091, +0.289] | +0.216 [+0.117, +0.315] | +0.196 [+0.105, +0.290] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.223 [+0.121, +0.321] | +0.112 [+0.022, +0.202] | +0.009 [-0.083, +0.097] | +0.227 [+0.124, +0.328] | +0.223 [+0.122, +0.321] | +0.228 [+0.130, +0.325] | +0.218 [+0.114, +0.317] | +0.235 [+0.129, +0.341] | +0.201 [+0.102, +0.295] |
| GNM (visto) vA, coefficienti | +0.210 [+0.111, +0.308] | +0.127 [+0.042, +0.222] | +0.014 [-0.068, +0.100] | +0.221 [+0.121, +0.322] | +0.210 [+0.114, +0.303] | +0.202 [+0.103, +0.301] | +0.202 [+0.108, +0.299] | +0.228 [+0.132, +0.328] | +0.213 [+0.119, +0.307] |
| GNM (visto) vA, mesh d'identita' FR | +0.203 [+0.106, +0.300] | +0.084 [-0.002, +0.172] | -0.014 [-0.102, +0.073] | +0.208 [+0.108, +0.307] | +0.203 [+0.106, +0.297] | +0.207 [+0.111, +0.302] | +0.199 [+0.101, +0.294] | +0.218 [+0.116, +0.322] | +0.184 [+0.087, +0.278] |
| GNM (visto) vB, coefficienti | +0.122 [+0.045, +0.190] | -0.061 [-0.157, +0.021] | -0.121 [-0.195, -0.056] | +0.130 [+0.051, +0.200] | +0.133 [+0.056, +0.202] | +0.115 [+0.040, +0.186] | +0.127 [+0.049, +0.195] | +0.122 [+0.043, +0.192] | +0.139 [+0.063, +0.202] |
| GNM (visto) vB, mesh d'identita' FR | +0.009 [-0.065, +0.081] | -0.141 [-0.211, -0.073] | -0.221 [-0.290, -0.154] | +0.015 [-0.061, +0.086] | +0.018 [-0.057, +0.088] | +0.013 [-0.065, +0.086] | +0.012 [-0.063, +0.083] | +0.008 [-0.074, +0.080] | -0.005 [-0.078, +0.064] |
| GNM (visto) vB, composizione S_B + k d_P | -0.112 [-0.175, -0.058] | -0.274 [-0.338, -0.210] | -0.345 [-0.398, -0.291] | -0.105 [-0.167, -0.052] | -0.105 [-0.166, -0.051] | -0.110 [-0.170, -0.053] | -0.108 [-0.168, -0.056] | -0.114 [-0.176, -0.059] | -0.114 [-0.174, -0.062] |
| FLAME 2023 Open preregistrato, coefficienti | +0.303 [+0.212, +0.403] | +0.204 [+0.125, +0.286] | +0.093 [+0.017, +0.176] | +0.307 [+0.212, +0.406] | +0.307 [+0.216, +0.405] | +0.280 [+0.187, +0.380] | +0.300 [+0.205, +0.401] | +0.337 [+0.241, +0.444] | +0.265 [+0.173, +0.362] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.238 [+0.139, +0.337] | +0.144 [+0.064, +0.226] | +0.030 [-0.053, +0.118] | +0.243 [+0.144, +0.343] | +0.249 [+0.150, +0.348] | +0.229 [+0.127, +0.328] | +0.233 [+0.131, +0.331] | +0.256 [+0.153, +0.369] | +0.205 [+0.105, +0.303] |
| FLAME 2023 Open vA, coefficienti | +0.299 [+0.210, +0.397] | +0.202 [+0.126, +0.284] | +0.090 [+0.014, +0.171] | +0.303 [+0.210, +0.402] | +0.303 [+0.212, +0.399] | +0.276 [+0.184, +0.375] | +0.296 [+0.201, +0.395] | +0.333 [+0.242, +0.440] | +0.261 [+0.169, +0.357] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.234 [+0.136, +0.334] | +0.132 [+0.051, +0.214] | +0.022 [-0.062, +0.109] | +0.239 [+0.138, +0.339] | +0.244 [+0.148, +0.341] | +0.227 [+0.126, +0.322] | +0.229 [+0.126, +0.325] | +0.250 [+0.145, +0.361] | +0.204 [+0.106, +0.303] |
| FLAME 2023 Open vB, coefficienti | +0.138 [+0.061, +0.211] | -0.027 [-0.116, +0.060] | -0.097 [-0.168, -0.023] | +0.145 [+0.065, +0.220] | +0.149 [+0.069, +0.222] | +0.131 [+0.053, +0.204] | +0.141 [+0.062, +0.212] | +0.141 [+0.061, +0.216] | +0.152 [+0.079, +0.223] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.057 [-0.018, +0.125] | -0.075 [-0.141, -0.006] | -0.167 [-0.233, -0.105] | +0.065 [-0.008, +0.130] | +0.076 [+0.002, +0.143] | +0.054 [-0.022, +0.122] | +0.057 [-0.019, +0.127] | +0.054 [-0.028, +0.124] | +0.044 [-0.030, +0.114] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.044 [-0.110, +0.013] | -0.198 [-0.265, -0.130] | -0.275 [-0.333, -0.220] | -0.036 [-0.102, +0.019] | -0.029 [-0.094, +0.026] | -0.048 [-0.111, +0.008] | -0.040 [-0.107, +0.018] | -0.045 [-0.111, +0.015] | -0.046 [-0.108, +0.010] |
| ArcFace-render, normal map 3 viste (G0) | +0.387 [+0.311, +0.459] | +0.179 [+0.091, +0.265] | +0.135 [+0.060, +0.208] | +0.384 [+0.310, +0.459] | +0.367 [+0.292, +0.442] | +0.373 [+0.299, +0.447] | +0.380 [+0.304, +0.453] | +0.400 [+0.303, +0.487] | +0.426 [+0.356, +0.495] |
| ShapeDNA k=50 | +0.597 [+0.532, +0.658] | +0.476 [+0.383, +0.562] | +0.395 [+0.335, +0.455] | +0.536 [+0.469, +0.600] | +0.553 [+0.489, +0.612] | +0.527 [+0.458, +0.595] | +0.555 [+0.488, +0.619] | +0.705 [+0.612, +0.797] | +0.413 [+0.346, +0.484] |
| HKS globale | +0.654 [+0.586, +0.718] | +0.476 [+0.382, +0.572] | +0.418 [+0.354, +0.483] | +0.639 [+0.567, +0.705] | +0.647 [+0.578, +0.710] | +0.627 [+0.558, +0.695] | +0.644 [+0.571, +0.708] | +0.682 [+0.598, +0.761] | +0.591 [+0.513, +0.670] |
| WKS globale | +0.671 [+0.600, +0.739] | +0.477 [+0.388, +0.566] | +0.428 [+0.364, +0.494] | +0.640 [+0.573, +0.709] | +0.649 [+0.583, +0.716] | +0.630 [+0.559, +0.701] | +0.697 [+0.624, +0.767] | +0.717 [+0.612, +0.822] | +0.529 [+0.452, +0.611] |
| taglia stimata dalla mesh (non oracolo) | +0.483 [+0.371, +0.592] | +0.475 [+0.394, +0.560] | +0.350 [+0.277, +0.425] | +0.485 [+0.371, +0.595] | +0.537 [+0.432, +0.634] | +0.472 [+0.356, +0.581] | +0.462 [+0.349, +0.571] | +0.471 [+0.357, +0.579] | +0.404 [+0.285, +0.522] |
| taglia oracolo (S di FR) | +0.223 [+0.070, +0.387] | +0.023 [-0.141, +0.197] | -0.026 [-0.170, +0.133] | +0.231 [+0.080, +0.394] | +0.233 [+0.081, +0.398] | +0.220 [+0.066, +0.386] | +0.230 [+0.076, +0.391] | +0.223 [+0.069, +0.390] | +0.236 [+0.085, +0.398] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.269 [+0.217, +0.317] | +0.210 [+0.143, +0.276] | +0.087 [+0.041, +0.130] | +0.283 [+0.230, +0.331] | +0.172 [+0.118, +0.222] | +0.331 [+0.281, +0.375] | +0.198 [+0.143, +0.249] | +0.244 [+0.191, +0.294] | +0.134 [+0.077, +0.186] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.269 [+0.217, +0.317] | +0.247 [+0.184, +0.310] | +0.121 [+0.078, +0.161] | +0.294 [+0.238, +0.343] | +0.171 [+0.121, +0.221] | +0.320 [+0.273, +0.362] | +0.196 [+0.141, +0.247] | +0.248 [+0.195, +0.298] | +0.082 [+0.029, +0.135] |
| NICP per coppia in mm | +0.340 [+0.283, +0.393] | +0.168 [+0.100, +0.235] | +0.113 [+0.061, +0.161] | +0.357 [+0.299, +0.413] | +0.246 [+0.186, +0.303] | +0.400 [+0.348, +0.450] | +0.272 [+0.213, +0.329] | +0.318 [+0.257, +0.373] | - |
| NICP per coppia cs | +0.349 [+0.291, +0.401] | +0.242 [+0.169, +0.313] | +0.133 [+0.082, +0.185] | +0.367 [+0.306, +0.422] | +0.256 [+0.195, +0.311] | +0.405 [+0.351, +0.456] | +0.281 [+0.219, +0.338] | +0.326 [+0.266, +0.381] | - |
| NICP su template in mm | +0.348 [+0.259, +0.423] | +0.468 [+0.369, +0.560] | +0.294 [+0.232, +0.351] | +0.341 [+0.254, +0.421] | +0.340 [+0.255, +0.416] | +0.363 [+0.281, +0.441] | +0.343 [+0.253, +0.424] | +0.372 [+0.284, +0.450] | +0.317 [+0.233, +0.390] |
| NICP su template cs | +0.311 [+0.236, +0.389] | +0.454 [+0.363, +0.538] | +0.277 [+0.217, +0.333] | +0.329 [+0.251, +0.408] | +0.326 [+0.251, +0.403] | +0.288 [+0.205, +0.371] | +0.324 [+0.241, +0.406] | +0.310 [+0.234, +0.384] | +0.247 [+0.171, +0.328] |
| varifold in mm (massa unitaria) | +0.465 [+0.418, +0.503] | +0.287 [+0.222, +0.344] | +0.312 [+0.265, +0.354] | +0.372 [+0.330, +0.408] | +0.370 [+0.329, +0.404] | +0.362 [+0.315, +0.401] | +0.354 [+0.313, +0.388] | +0.444 [+0.388, +0.503] | +0.083 [+0.041, +0.126] |
| GNM (visto) preregistrato, coefficienti | +0.286 [+0.197, +0.376] | +0.218 [+0.131, +0.303] | +0.093 [+0.017, +0.168] | +0.292 [+0.202, +0.387] | +0.285 [+0.196, +0.374] | +0.280 [+0.187, +0.370] | +0.274 [+0.184, +0.363] | +0.320 [+0.232, +0.409] | +0.267 [+0.180, +0.351] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.170 [+0.075, +0.264] | +0.104 [+0.010, +0.199] | -0.021 [-0.107, +0.068] | +0.178 [+0.086, +0.272] | +0.167 [+0.074, +0.258] | +0.159 [+0.063, +0.256] | +0.159 [+0.069, +0.249] | +0.206 [+0.107, +0.301] | +0.154 [+0.065, +0.248] |
| GNM (visto) vA, coefficienti | +0.269 [+0.174, +0.360] | +0.170 [+0.074, +0.267] | +0.064 [-0.020, +0.149] | +0.280 [+0.184, +0.376] | +0.262 [+0.167, +0.353] | +0.261 [+0.164, +0.354] | +0.258 [+0.163, +0.349] | +0.305 [+0.210, +0.399] | +0.257 [+0.165, +0.344] |
| GNM (visto) vA, mesh d'identita' SR | +0.176 [+0.077, +0.278] | +0.074 [-0.021, +0.168] | -0.032 [-0.124, +0.063] | +0.184 [+0.085, +0.288] | +0.171 [+0.074, +0.272] | +0.165 [+0.064, +0.270] | +0.166 [+0.069, +0.267] | +0.212 [+0.109, +0.317] | +0.161 [+0.062, +0.261] |
| GNM (visto) vB, coefficienti | +0.129 [+0.051, +0.198] | -0.086 [-0.178, +0.000] | -0.129 [-0.206, -0.063] | +0.138 [+0.061, +0.206] | +0.132 [+0.055, +0.202] | +0.127 [+0.051, +0.198] | +0.131 [+0.053, +0.206] | +0.136 [+0.057, +0.206] | +0.126 [+0.056, +0.193] |
| GNM (visto) vB, mesh d'identita' SR | -0.107 [-0.155, -0.065] | -0.315 [-0.378, -0.253] | -0.357 [-0.406, -0.312] | -0.097 [-0.145, -0.054] | -0.105 [-0.153, -0.062] | -0.107 [-0.156, -0.063] | -0.103 [-0.150, -0.063] | -0.100 [-0.149, -0.058] | -0.113 [-0.161, -0.070] |
| FLAME 2023 Open preregistrato, coefficienti | +0.350 [+0.261, +0.442] | +0.237 [+0.158, +0.316] | +0.131 [+0.048, +0.210] | +0.352 [+0.256, +0.445] | +0.349 [+0.259, +0.439] | +0.330 [+0.238, +0.422] | +0.343 [+0.251, +0.433] | +0.400 [+0.307, +0.490] | +0.294 [+0.203, +0.383] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.277 [+0.177, +0.374] | +0.110 [+0.017, +0.203] | +0.038 [-0.053, +0.131] | +0.275 [+0.168, +0.380] | +0.273 [+0.176, +0.365] | +0.255 [+0.152, +0.356] | +0.266 [+0.166, +0.362] | +0.338 [+0.236, +0.439] | +0.225 [+0.124, +0.329] |
| FLAME 2023 Open vA, coefficienti | +0.349 [+0.261, +0.440] | +0.237 [+0.157, +0.317] | +0.131 [+0.049, +0.209] | +0.350 [+0.256, +0.442] | +0.348 [+0.259, +0.437] | +0.329 [+0.237, +0.422] | +0.343 [+0.251, +0.432] | +0.399 [+0.306, +0.490] | +0.294 [+0.203, +0.382] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.267 [+0.166, +0.365] | +0.091 [-0.007, +0.185] | +0.024 [-0.069, +0.117] | +0.265 [+0.156, +0.369] | +0.262 [+0.163, +0.358] | +0.246 [+0.141, +0.348] | +0.255 [+0.156, +0.352] | +0.328 [+0.224, +0.432] | +0.217 [+0.113, +0.321] |
| FLAME 2023 Open vB, coefficienti | +0.154 [+0.085, +0.220] | -0.039 [-0.127, +0.044] | -0.094 [-0.162, -0.032] | +0.165 [+0.094, +0.232] | +0.161 [+0.090, +0.227] | +0.151 [+0.080, +0.218] | +0.154 [+0.084, +0.221] | +0.163 [+0.093, +0.227] | +0.148 [+0.081, +0.211] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.074 [-0.129, -0.022] | -0.292 [-0.361, -0.221] | -0.331 [-0.387, -0.277] | -0.065 [-0.120, -0.013] | -0.070 [-0.126, -0.016] | -0.075 [-0.132, -0.020] | -0.072 [-0.126, -0.020] | -0.061 [-0.115, -0.010] | -0.082 [-0.139, -0.027] |
| ArcFace-render, normal map 3 viste (G0) | +0.399 [+0.328, +0.461] | +0.146 [+0.057, +0.229] | +0.128 [+0.054, +0.195] | +0.391 [+0.317, +0.454] | +0.369 [+0.296, +0.434] | +0.382 [+0.310, +0.443] | +0.384 [+0.308, +0.449] | +0.433 [+0.343, +0.513] | +0.419 [+0.350, +0.479] |
| ShapeDNA k=50 | +0.629 [+0.567, +0.682] | +0.517 [+0.424, +0.598] | +0.430 [+0.372, +0.486] | +0.561 [+0.502, +0.618] | +0.574 [+0.516, +0.626] | +0.553 [+0.490, +0.613] | +0.580 [+0.519, +0.636] | +0.726 [+0.638, +0.806] | +0.414 [+0.351, +0.478] |
| HKS globale | +0.704 [+0.638, +0.761] | +0.522 [+0.435, +0.604] | +0.463 [+0.398, +0.521] | +0.688 [+0.618, +0.750] | +0.692 [+0.621, +0.751] | +0.678 [+0.603, +0.743] | +0.691 [+0.618, +0.756] | +0.735 [+0.659, +0.806] | +0.613 [+0.531, +0.690] |
| WKS globale | +0.729 [+0.665, +0.785] | +0.511 [+0.422, +0.594] | +0.473 [+0.411, +0.530] | +0.700 [+0.640, +0.755] | +0.701 [+0.642, +0.755] | +0.690 [+0.629, +0.747] | +0.749 [+0.679, +0.806] | +0.787 [+0.695, +0.881] | +0.571 [+0.496, +0.646] |
| taglia stimata dalla mesh (non oracolo) | +0.652 [+0.574, +0.729] | +0.503 [+0.421, +0.583] | +0.441 [+0.374, +0.501] | +0.659 [+0.579, +0.735] | +0.676 [+0.603, +0.741] | +0.647 [+0.566, +0.723] | +0.642 [+0.560, +0.723] | +0.658 [+0.578, +0.738] | +0.616 [+0.523, +0.704] |
| taglia oracolo (S di FR) | +0.628 [+0.503, +0.741] | +0.397 [+0.255, +0.532] | +0.364 [+0.238, +0.482] | +0.638 [+0.515, +0.753] | +0.632 [+0.507, +0.745] | +0.627 [+0.503, +0.743] | +0.632 [+0.507, +0.749] | +0.635 [+0.511, +0.746] | +0.624 [+0.505, +0.735] |

### FaceVerse con espressioni: Spearman per insieme di righe (seme 271049)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (99000) | crop (49500) | all_cross (148500) | con_perturbazione original (39600) | con_perturbazione remesh (39600) | con_perturbazione down8k (39600) | con_perturbazione up60k (39600) | con_perturbazione noisy (39600) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.303 [0.229, 0.374] | 0.252 [0.175, 0.326] | 0.284 [0.213, 0.354] | 0.300 [0.222, 0.372] | 0.309 [0.235, 0.380] | 0.289 [0.218, 0.359] | 0.306 [0.230, 0.377] | 0.312 [0.236, 0.380] | 0.303 [0.225, 0.374] |
| **factorized s1234, d_F non cal.** | 0.297 [0.229, 0.362] | 0.265 [0.185, 0.339] | 0.286 [0.217, 0.352] | 0.294 [0.220, 0.366] | 0.298 [0.224, 0.366] | 0.288 [0.219, 0.353] | 0.300 [0.230, 0.361] | 0.306 [0.235, 0.373] | 0.300 [0.232, 0.365] |
| factorized s1234, d_P | 0.283 [0.210, 0.346] | 0.252 [0.173, 0.324] | 0.271 [0.198, 0.335] | 0.280 [0.205, 0.350] | 0.283 [0.208, 0.351] | 0.276 [0.206, 0.338] | 0.285 [0.211, 0.344] | 0.291 [0.218, 0.355] | 0.286 [0.212, 0.352] |
| **factorized s2345, d_F cal.*** | 0.318 [0.249, 0.379] | 0.234 [0.171, 0.292] | 0.288 [0.224, 0.345] | 0.316 [0.238, 0.383] | 0.322 [0.254, 0.385] | 0.301 [0.229, 0.362] | 0.321 [0.251, 0.383] | 0.329 [0.259, 0.393] | 0.309 [0.241, 0.369] |
| **factorized s2345, d_F non cal.** | 0.317 [0.243, 0.383] | 0.256 [0.187, 0.323] | 0.296 [0.226, 0.362] | 0.315 [0.234, 0.386] | 0.315 [0.241, 0.388] | 0.306 [0.230, 0.373] | 0.319 [0.247, 0.382] | 0.330 [0.253, 0.400] | 0.309 [0.238, 0.375] |
| factorized s2345, d_P | 0.308 [0.234, 0.373] | 0.251 [0.182, 0.316] | 0.287 [0.219, 0.352] | 0.307 [0.230, 0.375] | 0.305 [0.230, 0.376] | 0.300 [0.223, 0.367] | 0.308 [0.235, 0.371] | 0.318 [0.244, 0.388] | 0.300 [0.230, 0.364] |
| **ctrlfr s1234** | 0.282 [0.215, 0.343] | 0.256 [0.190, 0.314] | 0.273 [0.208, 0.327] | 0.291 [0.214, 0.361] | 0.279 [0.209, 0.344] | 0.265 [0.197, 0.327] | 0.296 [0.230, 0.356] | 0.297 [0.222, 0.360] | 0.302 [0.237, 0.363] |
| **ctrlfr s2345** | 0.258 [0.187, 0.324] | 0.239 [0.167, 0.303] | 0.251 [0.185, 0.315] | 0.258 [0.184, 0.330] | 0.260 [0.183, 0.330] | 0.242 [0.169, 0.314] | 0.256 [0.186, 0.317] | 0.282 [0.206, 0.351] | 0.276 [0.206, 0.341] |
| **C3M e123, d_F cal.*** | 0.289 [0.220, 0.353] | 0.253 [0.186, 0.316] | 0.277 [0.209, 0.339] | 0.304 [0.227, 0.372] | 0.299 [0.222, 0.373] | 0.286 [0.213, 0.353] | 0.291 [0.220, 0.356] | 0.280 [0.206, 0.348] | 0.312 [0.235, 0.380] |
| **C3M e123, d_F non cal.** | 0.300 [0.224, 0.374] | 0.289 [0.218, 0.356] | 0.296 [0.225, 0.365] | 0.307 [0.226, 0.385] | 0.306 [0.222, 0.380] | 0.289 [0.211, 0.365] | 0.302 [0.222, 0.377] | 0.305 [0.227, 0.381] | 0.314 [0.238, 0.386] |
| C3M e123, d_P | 0.303 [0.228, 0.376] | 0.287 [0.216, 0.352] | 0.297 [0.224, 0.363] | 0.310 [0.231, 0.384] | 0.310 [0.228, 0.387] | 0.294 [0.217, 0.370] | 0.302 [0.224, 0.378] | 0.306 [0.230, 0.381] | 0.312 [0.236, 0.385] |
| **C3M e205, d_F cal.*** | 0.290 [0.224, 0.352] | 0.250 [0.185, 0.307] | 0.276 [0.213, 0.336] | 0.313 [0.245, 0.379] | 0.302 [0.229, 0.369] | 0.288 [0.219, 0.353] | 0.292 [0.226, 0.356] | 0.278 [0.210, 0.340] | 0.326 [0.253, 0.395] |
| **C3M e205, d_F non cal.** | 0.305 [0.232, 0.376] | 0.297 [0.227, 0.365] | 0.302 [0.233, 0.370] | 0.322 [0.247, 0.392] | 0.309 [0.231, 0.384] | 0.291 [0.216, 0.363] | 0.304 [0.229, 0.376] | 0.311 [0.238, 0.382] | 0.326 [0.251, 0.395] |
| C3M e205, d_P | 0.310 [0.237, 0.381] | 0.297 [0.227, 0.364] | 0.305 [0.235, 0.372] | 0.326 [0.253, 0.395] | 0.316 [0.236, 0.391] | 0.298 [0.221, 0.371] | 0.305 [0.231, 0.378] | 0.313 [0.239, 0.384] | 0.325 [0.253, 0.393] |
| **ICP + Chamfer in mm (per coppia)** | 0.337 [0.262, 0.410] | 0.271 [0.207, 0.332] | 0.312 [0.242, 0.379] | 0.351 [0.275, 0.424] | 0.338 [0.257, 0.413] | 0.324 [0.247, 0.398] | 0.335 [0.255, 0.411] | 0.340 [0.261, 0.419] | 0.343 [0.269, 0.413] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.290 [0.218, 0.358] | 0.273 [0.209, 0.332] | 0.275 [0.211, 0.334] | 0.307 [0.235, 0.376] | 0.289 [0.214, 0.360] | 0.276 [0.201, 0.347] | 0.286 [0.212, 0.357] | 0.294 [0.220, 0.367] | 0.293 [0.223, 0.358] |
| **NICP per coppia in mm** | 0.201 [0.128, 0.277] | 0.225 [0.155, 0.299] | 0.209 [0.140, 0.278] | 0.204 [0.124, 0.282] | 0.197 [0.122, 0.273] | 0.200 [0.120, 0.279] | 0.198 [0.122, 0.274] | 0.208 [0.131, 0.285] | - |
| **NICP per coppia cs** | 0.212 [0.131, 0.292] | 0.231 [0.160, 0.303] | 0.217 [0.144, 0.294] | 0.213 [0.133, 0.290] | 0.209 [0.128, 0.294] | 0.215 [0.131, 0.300] | 0.205 [0.125, 0.287] | 0.220 [0.139, 0.301] | - |
| **NICP su template in mm** | 0.212 [0.117, 0.308] | 0.124 [0.053, 0.192] | 0.177 [0.096, 0.256] | 0.215 [0.118, 0.311] | 0.225 [0.132, 0.320] | 0.203 [0.111, 0.298] | 0.209 [0.113, 0.307] | 0.207 [0.108, 0.305] | 0.213 [0.116, 0.307] |
| **NICP su template cs** | 0.135 [0.061, 0.204] | 0.123 [0.068, 0.175] | 0.127 [0.066, 0.184] | 0.149 [0.073, 0.217] | 0.140 [0.065, 0.206] | 0.116 [0.033, 0.192] | 0.144 [0.069, 0.214] | 0.130 [0.053, 0.201] | 0.146 [0.072, 0.209] |
| **varifold in mm (massa unitaria)** | 0.260 [0.201, 0.313] | 0.237 [0.185, 0.285] | 0.248 [0.195, 0.296] | 0.283 [0.218, 0.341] | 0.272 [0.212, 0.324] | 0.272 [0.203, 0.336] | 0.282 [0.220, 0.342] | 0.300 [0.227, 0.367] | 0.285 [0.229, 0.333] |
| **GNM (visto) preregistrato, coefficienti** | 0.220 [0.148, 0.286] | 0.216 [0.145, 0.283] | 0.218 [0.146, 0.283] | 0.237 [0.159, 0.308] | 0.203 [0.130, 0.271] | 0.200 [0.126, 0.268] | 0.241 [0.165, 0.308] | 0.219 [0.135, 0.292] | 0.221 [0.150, 0.286] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.227 [0.140, 0.311] | 0.216 [0.130, 0.297] | 0.223 [0.136, 0.305] | 0.243 [0.153, 0.331] | 0.222 [0.132, 0.304] | 0.220 [0.132, 0.303] | 0.225 [0.135, 0.312] | 0.225 [0.134, 0.310] | 0.227 [0.140, 0.311] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.228 [0.145, 0.303] | 0.223 [0.150, 0.292] | 0.226 [0.148, 0.298] | 0.235 [0.149, 0.312] | 0.215 [0.134, 0.291] | 0.233 [0.153, 0.308] | 0.227 [0.144, 0.303] | 0.229 [0.143, 0.311] | 0.229 [0.147, 0.303] |
| **GNM (visto) vA, coefficienti*** | 0.223 [0.145, 0.291] | 0.222 [0.151, 0.289] | 0.222 [0.148, 0.286] | 0.238 [0.157, 0.310] | 0.205 [0.124, 0.275] | 0.205 [0.130, 0.274] | 0.243 [0.165, 0.312] | 0.222 [0.137, 0.296] | 0.225 [0.150, 0.289] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.230 [0.141, 0.316] | 0.219 [0.132, 0.302] | 0.226 [0.139, 0.312] | 0.246 [0.155, 0.334] | 0.226 [0.135, 0.308] | 0.223 [0.135, 0.307] | 0.228 [0.138, 0.315] | 0.228 [0.136, 0.314] | 0.230 [0.142, 0.314] |
| GNM (visto) vA, mesh d'identita' SR* | 0.219 [0.137, 0.297] | 0.215 [0.142, 0.287] | 0.218 [0.140, 0.292] | 0.225 [0.140, 0.304] | 0.208 [0.127, 0.285] | 0.226 [0.147, 0.302] | 0.220 [0.137, 0.296] | 0.220 [0.134, 0.302] | 0.220 [0.140, 0.296] |
| **GNM (visto) vB, coefficienti*** | 0.226 [0.134, 0.299] | 0.220 [0.126, 0.301] | 0.224 [0.133, 0.297] | 0.214 [0.117, 0.295] | 0.234 [0.139, 0.313] | 0.232 [0.135, 0.312] | 0.220 [0.126, 0.300] | 0.231 [0.138, 0.307] | 0.224 [0.133, 0.299] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.292 [0.205, 0.363] | 0.281 [0.192, 0.352] | 0.289 [0.199, 0.359] | 0.306 [0.210, 0.383] | 0.286 [0.202, 0.360] | 0.291 [0.200, 0.362] | 0.284 [0.193, 0.360] | 0.295 [0.209, 0.368] | 0.291 [0.202, 0.363] |
| GNM (visto) vB, mesh d'identita' SR* | 0.319 [0.235, 0.390] | 0.305 [0.226, 0.377] | 0.314 [0.233, 0.384] | 0.321 [0.236, 0.396] | 0.314 [0.225, 0.389] | 0.326 [0.244, 0.399] | 0.304 [0.220, 0.378] | 0.328 [0.244, 0.404] | 0.312 [0.231, 0.383] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.338 [0.255, 0.407] | 0.325 [0.242, 0.394] | 0.334 [0.253, 0.402] | 0.348 [0.258, 0.420] | 0.332 [0.248, 0.404] | 0.341 [0.257, 0.411] | 0.326 [0.240, 0.401] | 0.345 [0.263, 0.413] | 0.335 [0.253, 0.403] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.174 [0.109, 0.237] | 0.162 [0.098, 0.225] | 0.170 [0.107, 0.233] | 0.180 [0.114, 0.242] | 0.171 [0.103, 0.236] | 0.178 [0.115, 0.241] | 0.171 [0.104, 0.239] | 0.172 [0.104, 0.237] | 0.176 [0.110, 0.241] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.211 [0.137, 0.279] | 0.193 [0.117, 0.265] | 0.205 [0.131, 0.275] | 0.223 [0.150, 0.291] | 0.207 [0.132, 0.279] | 0.215 [0.144, 0.278] | 0.206 [0.127, 0.286] | 0.206 [0.133, 0.277] | 0.212 [0.135, 0.281] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.166 [0.091, 0.235] | 0.161 [0.089, 0.230] | 0.164 [0.091, 0.232] | 0.160 [0.083, 0.232] | 0.163 [0.088, 0.231] | 0.174 [0.097, 0.242] | 0.166 [0.090, 0.234] | 0.166 [0.084, 0.236] | 0.169 [0.096, 0.238] |
| **FLAME 2023 Open vA, coefficienti*** | 0.174 [0.109, 0.237] | 0.162 [0.098, 0.225] | 0.170 [0.107, 0.233] | 0.180 [0.114, 0.242] | 0.171 [0.103, 0.236] | 0.178 [0.115, 0.241] | 0.171 [0.104, 0.239] | 0.172 [0.104, 0.237] | 0.176 [0.110, 0.241] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.211 [0.137, 0.279] | 0.193 [0.117, 0.265] | 0.205 [0.131, 0.275] | 0.223 [0.150, 0.291] | 0.207 [0.132, 0.279] | 0.215 [0.144, 0.278] | 0.206 [0.127, 0.286] | 0.206 [0.133, 0.277] | 0.212 [0.135, 0.281] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.166 [0.091, 0.235] | 0.161 [0.089, 0.230] | 0.164 [0.091, 0.232] | 0.160 [0.083, 0.232] | 0.163 [0.088, 0.231] | 0.174 [0.097, 0.242] | 0.166 [0.090, 0.234] | 0.166 [0.084, 0.236] | 0.169 [0.096, 0.238] |
| **FLAME 2023 Open vB, coefficienti*** | 0.203 [0.121, 0.275] | 0.179 [0.084, 0.268] | 0.195 [0.111, 0.272] | 0.171 [0.081, 0.255] | 0.200 [0.112, 0.281] | 0.229 [0.135, 0.305] | 0.199 [0.112, 0.279] | 0.218 [0.135, 0.293] | 0.194 [0.112, 0.270] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.264 [0.183, 0.336] | 0.245 [0.166, 0.317] | 0.258 [0.180, 0.327] | 0.276 [0.192, 0.348] | 0.260 [0.180, 0.334] | 0.269 [0.188, 0.341] | 0.257 [0.174, 0.335] | 0.260 [0.176, 0.333] | 0.263 [0.186, 0.333] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.290 [0.216, 0.361] | 0.270 [0.188, 0.348] | 0.283 [0.209, 0.355] | 0.289 [0.210, 0.363] | 0.285 [0.207, 0.357] | 0.301 [0.226, 0.374] | 0.277 [0.198, 0.349] | 0.296 [0.216, 0.368] | 0.285 [0.212, 0.358] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.302 [0.222, 0.372] | 0.283 [0.204, 0.355] | 0.295 [0.217, 0.365] | 0.310 [0.228, 0.382] | 0.297 [0.219, 0.368] | 0.309 [0.233, 0.377] | 0.291 [0.205, 0.364] | 0.301 [0.221, 0.373] | 0.300 [0.222, 0.367] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.127 [0.081, 0.175] | 0.160 [0.106, 0.217] | 0.137 [0.090, 0.186] | 0.143 [0.091, 0.194] | 0.145 [0.097, 0.196] | 0.143 [0.090, 0.199] | 0.147 [0.093, 0.197] | 0.102 [0.039, 0.164] | 0.150 [0.098, 0.204] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.117 [0.070, 0.166] | 0.161 [0.102, 0.222] | 0.131 [0.080, 0.183] | 0.138 [0.081, 0.195] | 0.133 [0.077, 0.188] | 0.144 [0.087, 0.202] | 0.139 [0.081, 0.195] | 0.089 [0.022, 0.156] | 0.146 [0.089, 0.202] |
| **ShapeDNA k=50** | 0.063 [0.036, 0.087] | 0.057 [0.010, 0.104] | 0.060 [0.032, 0.088] | 0.074 [0.031, 0.117] | 0.081 [0.036, 0.123] | 0.088 [0.047, 0.129] | 0.091 [0.048, 0.132] | 0.078 [0.009, 0.149] | 0.127 [0.069, 0.184] |
| **HKS globale** | 0.032 [0.009, 0.053] | 0.042 [0.004, 0.082] | 0.035 [0.012, 0.058] | 0.045 [0.013, 0.076] | 0.041 [0.009, 0.075] | 0.043 [0.010, 0.076] | 0.049 [0.014, 0.080] | 0.030 [-0.028, 0.084] | 0.067 [0.024, 0.114] |
| **WKS globale** | 0.039 [0.014, 0.065] | 0.065 [0.027, 0.104] | 0.044 [0.020, 0.069] | 0.036 [-0.003, 0.073] | 0.042 [0.005, 0.080] | 0.050 [0.010, 0.090] | 0.051 [0.016, 0.087] | 0.091 [0.015, 0.164] | 0.079 [0.030, 0.127] |
| **taglia stimata dalla mesh (non oracolo)** | 0.126 [0.037, 0.214] | 0.009 [-0.022, 0.043] | 0.077 [0.021, 0.136] | 0.123 [0.029, 0.217] | 0.120 [0.032, 0.208] | 0.115 [0.027, 0.205] | 0.130 [0.034, 0.224] | 0.143 [0.052, 0.233] | 0.130 [0.040, 0.217] |
| **taglia oracolo (S di FR)** | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (99000) | crop (49500) | all_cross (148500) | con_perturbazione original (39600) | con_perturbazione remesh (39600) | con_perturbazione down8k (39600) | con_perturbazione up60k (39600) | con_perturbazione noisy (39600) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.269 [0.195, 0.341] | 0.220 [0.143, 0.294] | 0.251 [0.174, 0.325] | 0.260 [0.179, 0.337] | 0.281 [0.207, 0.355] | 0.256 [0.179, 0.329] | 0.273 [0.194, 0.349] | 0.275 [0.201, 0.352] | 0.267 [0.184, 0.344] |
| factorized s1234, d_F non cal. | 0.287 [0.215, 0.353] | 0.251 [0.172, 0.326] | 0.274 [0.202, 0.341] | 0.281 [0.207, 0.355] | 0.295 [0.219, 0.363] | 0.276 [0.202, 0.341] | 0.292 [0.218, 0.356] | 0.292 [0.219, 0.359] | 0.289 [0.216, 0.355] |
| **factorized s1234, d_P** | 0.286 [0.215, 0.352] | 0.251 [0.174, 0.324] | 0.273 [0.201, 0.339] | 0.282 [0.207, 0.355] | 0.294 [0.217, 0.362] | 0.277 [0.202, 0.340] | 0.290 [0.218, 0.355] | 0.290 [0.214, 0.355] | 0.288 [0.215, 0.354] |
| factorized s2345, d_F cal.* | 0.286 [0.206, 0.353] | 0.208 [0.134, 0.271] | 0.258 [0.180, 0.321] | 0.280 [0.196, 0.352] | 0.298 [0.219, 0.369] | 0.271 [0.190, 0.339] | 0.288 [0.210, 0.356] | 0.296 [0.215, 0.366] | 0.278 [0.198, 0.346] |
| factorized s2345, d_F non cal. | 0.309 [0.227, 0.376] | 0.249 [0.173, 0.317] | 0.288 [0.209, 0.357] | 0.305 [0.217, 0.379] | 0.315 [0.231, 0.388] | 0.297 [0.215, 0.364] | 0.309 [0.231, 0.376] | 0.318 [0.234, 0.389] | 0.300 [0.223, 0.367] |
| **factorized s2345, d_P** | 0.313 [0.237, 0.377] | 0.257 [0.181, 0.323] | 0.293 [0.219, 0.357] | 0.312 [0.229, 0.381] | 0.318 [0.239, 0.389] | 0.303 [0.225, 0.372] | 0.312 [0.237, 0.376] | 0.320 [0.240, 0.389] | 0.304 [0.231, 0.369] |
| ctrlfr s1234 | 0.273 [0.201, 0.339] | 0.239 [0.163, 0.306] | 0.261 [0.193, 0.324] | 0.278 [0.197, 0.352] | 0.274 [0.200, 0.342] | 0.258 [0.185, 0.324] | 0.285 [0.211, 0.349] | 0.284 [0.201, 0.350] | 0.287 [0.213, 0.356] |
| ctrlfr s2345 | 0.259 [0.188, 0.325] | 0.243 [0.171, 0.305] | 0.253 [0.188, 0.316] | 0.255 [0.178, 0.332] | 0.267 [0.192, 0.338] | 0.243 [0.170, 0.312] | 0.258 [0.189, 0.323] | 0.280 [0.211, 0.349] | 0.275 [0.205, 0.341] |
| C3M e123, d_F cal.* | 0.270 [0.193, 0.341] | 0.232 [0.159, 0.302] | 0.258 [0.182, 0.328] | 0.281 [0.196, 0.358] | 0.284 [0.201, 0.359] | 0.267 [0.186, 0.340] | 0.273 [0.194, 0.347] | 0.258 [0.182, 0.332] | 0.292 [0.208, 0.367] |
| C3M e123, d_F non cal. | 0.303 [0.221, 0.381] | 0.289 [0.215, 0.359] | 0.299 [0.224, 0.369] | 0.312 [0.224, 0.390] | 0.315 [0.226, 0.394] | 0.291 [0.208, 0.367] | 0.306 [0.223, 0.383] | 0.302 [0.221, 0.377] | 0.320 [0.240, 0.392] |
| **C3M e123, d_P** | 0.313 [0.236, 0.386] | 0.294 [0.224, 0.364] | 0.306 [0.234, 0.372] | 0.321 [0.240, 0.398] | 0.324 [0.240, 0.399] | 0.302 [0.223, 0.374] | 0.313 [0.235, 0.388] | 0.309 [0.234, 0.382] | 0.324 [0.250, 0.394] |
| C3M e205, d_F cal.* | 0.275 [0.204, 0.340] | 0.239 [0.170, 0.300] | 0.263 [0.196, 0.325] | 0.293 [0.219, 0.365] | 0.291 [0.216, 0.364] | 0.275 [0.203, 0.344] | 0.275 [0.202, 0.343] | 0.261 [0.193, 0.326] | 0.311 [0.232, 0.384] |
| C3M e205, d_F non cal. | 0.306 [0.230, 0.376] | 0.305 [0.232, 0.371] | 0.305 [0.232, 0.372] | 0.321 [0.240, 0.395] | 0.317 [0.233, 0.393] | 0.293 [0.216, 0.367] | 0.305 [0.228, 0.379] | 0.304 [0.231, 0.375] | 0.331 [0.256, 0.401] |
| **C3M e205, d_P** | 0.314 [0.242, 0.382] | 0.308 [0.241, 0.376] | 0.312 [0.244, 0.378] | 0.330 [0.254, 0.403] | 0.327 [0.247, 0.400] | 0.303 [0.230, 0.376] | 0.310 [0.238, 0.383] | 0.309 [0.236, 0.379] | 0.334 [0.263, 0.403] |
| **ICP + Chamfer in mm (per coppia)** | 0.309 [0.234, 0.380] | 0.251 [0.186, 0.311] | 0.287 [0.217, 0.352] | 0.319 [0.240, 0.388] | 0.313 [0.233, 0.384] | 0.301 [0.223, 0.375] | 0.310 [0.233, 0.388] | 0.306 [0.227, 0.380] | 0.315 [0.240, 0.383] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.306 [0.233, 0.376] | 0.288 [0.223, 0.347] | 0.290 [0.225, 0.353] | 0.321 [0.249, 0.390] | 0.308 [0.232, 0.377] | 0.293 [0.213, 0.368] | 0.308 [0.232, 0.380] | 0.303 [0.224, 0.378] | 0.308 [0.239, 0.375] |
| **NICP per coppia in mm** | 0.223 [0.147, 0.298] | 0.248 [0.180, 0.319] | 0.231 [0.158, 0.302] | 0.226 [0.146, 0.304] | 0.223 [0.144, 0.297] | 0.224 [0.137, 0.305] | 0.223 [0.143, 0.301] | 0.223 [0.142, 0.300] | - |
| **NICP per coppia cs** | 0.235 [0.158, 0.310] | 0.258 [0.190, 0.328] | 0.242 [0.172, 0.313] | 0.235 [0.158, 0.311] | 0.234 [0.152, 0.314] | 0.239 [0.155, 0.321] | 0.232 [0.151, 0.315] | 0.238 [0.156, 0.314] | - |
| **NICP su template in mm** | 0.152 [0.058, 0.246] | 0.085 [0.017, 0.153] | 0.125 [0.047, 0.205] | 0.150 [0.055, 0.246] | 0.165 [0.072, 0.259] | 0.148 [0.053, 0.244] | 0.151 [0.057, 0.245] | 0.145 [0.050, 0.245] | 0.153 [0.061, 0.249] |
| **NICP su template cs** | 0.147 [0.070, 0.216] | 0.122 [0.066, 0.173] | 0.135 [0.071, 0.191] | 0.158 [0.082, 0.232] | 0.151 [0.073, 0.217] | 0.129 [0.046, 0.204] | 0.161 [0.084, 0.229] | 0.137 [0.057, 0.208] | 0.153 [0.078, 0.219] |
| **varifold in mm (massa unitaria)** | 0.250 [0.193, 0.306] | 0.227 [0.174, 0.280] | 0.238 [0.185, 0.287] | 0.269 [0.206, 0.331] | 0.263 [0.205, 0.319] | 0.265 [0.200, 0.327] | 0.270 [0.207, 0.331] | 0.287 [0.215, 0.354] | 0.271 [0.216, 0.322] |
| **GNM (visto) preregistrato, coefficienti** | 0.182 [0.104, 0.252] | 0.177 [0.105, 0.248] | 0.181 [0.107, 0.248] | 0.200 [0.120, 0.273] | 0.166 [0.088, 0.235] | 0.167 [0.091, 0.239] | 0.202 [0.126, 0.274] | 0.177 [0.096, 0.250] | 0.183 [0.109, 0.251] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.171 [0.082, 0.260] | 0.161 [0.074, 0.248] | 0.168 [0.080, 0.253] | 0.183 [0.089, 0.273] | 0.169 [0.079, 0.258] | 0.167 [0.079, 0.251] | 0.171 [0.078, 0.263] | 0.165 [0.076, 0.255] | 0.170 [0.081, 0.257] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.209 [0.129, 0.285] | 0.203 [0.126, 0.278] | 0.207 [0.129, 0.283] | 0.216 [0.132, 0.295] | 0.198 [0.118, 0.276] | 0.218 [0.141, 0.293] | 0.212 [0.128, 0.291] | 0.203 [0.116, 0.283] | 0.210 [0.130, 0.288] |
| **GNM (visto) vA, coefficienti*** | 0.184 [0.103, 0.254] | 0.182 [0.109, 0.254] | 0.183 [0.107, 0.250] | 0.200 [0.119, 0.273] | 0.167 [0.085, 0.238] | 0.170 [0.093, 0.243] | 0.203 [0.123, 0.274] | 0.178 [0.095, 0.256] | 0.185 [0.109, 0.252] |
| GNM (visto) vA, mesh d'identita' FR* | 0.174 [0.085, 0.263] | 0.163 [0.077, 0.252] | 0.170 [0.082, 0.257] | 0.186 [0.090, 0.276] | 0.173 [0.083, 0.261] | 0.170 [0.081, 0.255] | 0.173 [0.080, 0.265] | 0.167 [0.077, 0.260] | 0.173 [0.083, 0.261] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.205 [0.124, 0.281] | 0.199 [0.122, 0.274] | 0.203 [0.124, 0.278] | 0.210 [0.126, 0.288] | 0.194 [0.115, 0.272] | 0.214 [0.135, 0.288] | 0.209 [0.124, 0.290] | 0.198 [0.109, 0.278] | 0.206 [0.124, 0.282] |
| **GNM (visto) vB, coefficienti*** | 0.222 [0.124, 0.302] | 0.224 [0.128, 0.303] | 0.223 [0.126, 0.300] | 0.209 [0.109, 0.293] | 0.231 [0.136, 0.313] | 0.231 [0.128, 0.314] | 0.221 [0.121, 0.302] | 0.220 [0.124, 0.297] | 0.222 [0.128, 0.301] |
| GNM (visto) vB, mesh d'identita' FR* | 0.244 [0.149, 0.326] | 0.235 [0.139, 0.316] | 0.241 [0.147, 0.321] | 0.251 [0.150, 0.340] | 0.239 [0.141, 0.322] | 0.246 [0.149, 0.331] | 0.240 [0.140, 0.329] | 0.242 [0.148, 0.327] | 0.243 [0.148, 0.325] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.327 [0.239, 0.402] | 0.316 [0.229, 0.388] | 0.323 [0.238, 0.397] | 0.325 [0.237, 0.400] | 0.325 [0.235, 0.401] | 0.338 [0.250, 0.412] | 0.322 [0.233, 0.397] | 0.329 [0.240, 0.404] | 0.322 [0.236, 0.395] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.306 [0.209, 0.381] | 0.295 [0.199, 0.372] | 0.302 [0.206, 0.378] | 0.309 [0.211, 0.389] | 0.301 [0.202, 0.382] | 0.312 [0.219, 0.387] | 0.301 [0.207, 0.377] | 0.306 [0.210, 0.382] | 0.303 [0.207, 0.380] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.152 [0.079, 0.218] | 0.141 [0.071, 0.208] | 0.148 [0.077, 0.214] | 0.160 [0.087, 0.227] | 0.147 [0.074, 0.216] | 0.153 [0.078, 0.221] | 0.152 [0.079, 0.222] | 0.148 [0.073, 0.217] | 0.154 [0.081, 0.221] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.170 [0.089, 0.245] | 0.155 [0.074, 0.234] | 0.165 [0.086, 0.241] | 0.181 [0.103, 0.257] | 0.166 [0.086, 0.241] | 0.172 [0.093, 0.246] | 0.166 [0.078, 0.247] | 0.165 [0.084, 0.244] | 0.170 [0.088, 0.245] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.155 [0.075, 0.228] | 0.153 [0.071, 0.231] | 0.154 [0.076, 0.228] | 0.151 [0.066, 0.227] | 0.150 [0.069, 0.225] | 0.163 [0.079, 0.238] | 0.160 [0.079, 0.236] | 0.151 [0.063, 0.227] | 0.160 [0.078, 0.234] |
| **FLAME 2023 Open vA, coefficienti*** | 0.152 [0.079, 0.218] | 0.141 [0.071, 0.208] | 0.148 [0.077, 0.214] | 0.160 [0.087, 0.227] | 0.147 [0.074, 0.216] | 0.153 [0.078, 0.221] | 0.152 [0.079, 0.222] | 0.148 [0.073, 0.217] | 0.154 [0.081, 0.221] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.170 [0.089, 0.245] | 0.155 [0.074, 0.234] | 0.165 [0.086, 0.241] | 0.181 [0.103, 0.257] | 0.166 [0.086, 0.241] | 0.172 [0.093, 0.246] | 0.166 [0.078, 0.247] | 0.165 [0.084, 0.244] | 0.170 [0.088, 0.245] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.155 [0.075, 0.228] | 0.153 [0.071, 0.231] | 0.154 [0.076, 0.228] | 0.151 [0.066, 0.227] | 0.150 [0.069, 0.225] | 0.163 [0.079, 0.238] | 0.160 [0.079, 0.236] | 0.151 [0.063, 0.227] | 0.160 [0.078, 0.234] |
| **FLAME 2023 Open vB, coefficienti*** | 0.193 [0.108, 0.271] | 0.178 [0.078, 0.270] | 0.188 [0.099, 0.265] | 0.162 [0.068, 0.248] | 0.189 [0.098, 0.272] | 0.218 [0.127, 0.299] | 0.199 [0.110, 0.283] | 0.198 [0.113, 0.275] | 0.188 [0.101, 0.265] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.218 [0.125, 0.302] | 0.202 [0.110, 0.286] | 0.213 [0.121, 0.294] | 0.227 [0.129, 0.315] | 0.217 [0.121, 0.302] | 0.223 [0.129, 0.307] | 0.216 [0.122, 0.305] | 0.210 [0.114, 0.293] | 0.217 [0.125, 0.300] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.293 [0.209, 0.368] | 0.278 [0.189, 0.359] | 0.288 [0.203, 0.366] | 0.292 [0.207, 0.367] | 0.291 [0.206, 0.370] | 0.305 [0.223, 0.385] | 0.288 [0.202, 0.365] | 0.288 [0.201, 0.368] | 0.290 [0.202, 0.370] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.270 [0.182, 0.349] | 0.254 [0.163, 0.336] | 0.265 [0.179, 0.344] | 0.275 [0.181, 0.358] | 0.268 [0.176, 0.351] | 0.278 [0.187, 0.358] | 0.266 [0.176, 0.349] | 0.263 [0.173, 0.346] | 0.268 [0.181, 0.349] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.138 [0.093, 0.185] | 0.173 [0.119, 0.233] | 0.149 [0.103, 0.199] | 0.153 [0.102, 0.206] | 0.158 [0.112, 0.209] | 0.157 [0.105, 0.212] | 0.158 [0.107, 0.209] | 0.113 [0.048, 0.177] | 0.162 [0.112, 0.220] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.129 [0.081, 0.177] | 0.180 [0.123, 0.241] | 0.145 [0.094, 0.196] | 0.151 [0.095, 0.206] | 0.147 [0.096, 0.202] | 0.163 [0.109, 0.221] | 0.150 [0.096, 0.207] | 0.092 [0.022, 0.160] | 0.164 [0.112, 0.220] |
| **ShapeDNA k=50** | 0.061 [0.033, 0.086] | 0.068 [0.023, 0.110] | 0.062 [0.033, 0.089] | 0.076 [0.031, 0.118] | 0.081 [0.035, 0.122] | 0.083 [0.039, 0.126] | 0.086 [0.040, 0.127] | 0.076 [0.005, 0.144] | 0.125 [0.065, 0.181] |
| **HKS globale** | 0.029 [0.004, 0.051] | 0.041 [0.001, 0.081] | 0.032 [0.009, 0.056] | 0.042 [0.008, 0.076] | 0.038 [0.002, 0.074] | 0.034 [-0.002, 0.070] | 0.039 [0.005, 0.070] | 0.036 [-0.027, 0.093] | 0.061 [0.015, 0.109] |
| **WKS globale** | 0.039 [0.012, 0.066] | 0.062 [0.022, 0.102] | 0.044 [0.020, 0.067] | 0.037 [-0.001, 0.073] | 0.044 [0.007, 0.083] | 0.047 [0.006, 0.089] | 0.049 [0.016, 0.085] | 0.091 [0.015, 0.166] | 0.071 [0.021, 0.120] |
| **taglia stimata dalla mesh (non oracolo)** | 0.080 [-0.002, 0.170] | -0.000 [-0.028, 0.032] | 0.046 [-0.004, 0.099] | 0.071 [-0.015, 0.163] | 0.080 [-0.001, 0.170] | 0.077 [-0.005, 0.161] | 0.079 [-0.008, 0.170] | 0.093 [0.001, 0.183] | 0.083 [-0.001, 0.172] |
| **taglia oracolo (S di FR)** | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.035 [-0.105, +0.036] | -0.019 [-0.083, +0.050] | -0.028 [-0.092, +0.040] | -0.052 [-0.119, +0.021] | -0.029 [-0.105, +0.046] | -0.035 [-0.108, +0.041] | -0.029 [-0.105, +0.044] | -0.028 [-0.099, +0.043] | -0.040 [-0.109, +0.030] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.013 [-0.064, +0.099] | -0.021 [-0.097, +0.059] | +0.010 [-0.066, +0.089] | -0.008 [-0.087, +0.079] | +0.019 [-0.062, +0.108] | +0.013 [-0.067, +0.101] | +0.020 [-0.058, +0.110] | +0.018 [-0.063, +0.108] | +0.009 [-0.066, +0.091] |
| NICP per coppia in mm | +0.102 [+0.015, +0.190] | +0.027 [-0.053, +0.103] | +0.075 [-0.006, +0.154] | +0.096 [+0.002, +0.183] | +0.112 [+0.020, +0.204] | +0.089 [-0.001, +0.181] | +0.108 [+0.021, +0.196] | +0.104 [+0.015, +0.186] | - |
| NICP per coppia cs | +0.091 [-0.001, +0.177] | +0.022 [-0.060, +0.101] | +0.067 [-0.017, +0.149] | +0.087 [-0.000, +0.178] | +0.100 [+0.007, +0.188] | +0.074 [-0.013, +0.162] | +0.101 [+0.011, +0.191] | +0.092 [+0.002, +0.178] | - |
| NICP su template in mm | +0.091 [-0.010, +0.189] | +0.128 [+0.049, +0.208] | +0.107 [+0.018, +0.192] | +0.085 [-0.021, +0.188] | +0.084 [-0.020, +0.188] | +0.086 [-0.014, +0.185] | +0.097 [-0.001, +0.191] | +0.105 [+0.007, +0.207] | +0.090 [-0.005, +0.185] |
| NICP su template cs | +0.167 [+0.063, +0.271] | +0.130 [+0.036, +0.215] | +0.157 [+0.065, +0.249] | +0.151 [+0.044, +0.257] | +0.169 [+0.065, +0.274] | +0.173 [+0.060, +0.286] | +0.162 [+0.054, +0.261] | +0.182 [+0.083, +0.285] | +0.157 [+0.057, +0.259] |
| varifold in mm (massa unitaria) | +0.043 [-0.019, +0.107] | +0.015 [-0.045, +0.082] | +0.036 [-0.020, +0.098] | +0.016 [-0.045, +0.080] | +0.037 [-0.026, +0.106] | +0.017 [-0.047, +0.083] | +0.025 [-0.041, +0.093] | +0.012 [-0.060, +0.087] | +0.018 [-0.041, +0.080] |
| GNM (visto) preregistrato, coefficienti | +0.083 [-0.014, +0.183] | +0.037 [-0.064, +0.148] | +0.066 [-0.029, +0.164] | +0.063 [-0.044, +0.168] | +0.106 [+0.005, +0.221] | +0.089 [-0.012, +0.188] | +0.065 [-0.029, +0.155] | +0.093 [-0.006, +0.203] | +0.081 [-0.015, +0.181] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.076 [-0.016, +0.177] | +0.036 [-0.046, +0.128] | +0.061 [-0.025, +0.156] | +0.057 [-0.042, +0.162] | +0.087 [-0.008, +0.192] | +0.069 [-0.018, +0.167] | +0.081 [-0.009, +0.174] | +0.087 [-0.005, +0.190] | +0.076 [-0.012, +0.174] |
| GNM (visto) vA, coefficienti | +0.080 [-0.019, +0.186] | +0.030 [-0.068, +0.142] | +0.062 [-0.035, +0.162] | +0.061 [-0.042, +0.169] | +0.104 [+0.003, +0.219] | +0.084 [-0.021, +0.186] | +0.063 [-0.032, +0.154] | +0.089 [-0.014, +0.198] | +0.078 [-0.021, +0.183] |
| GNM (visto) vA, mesh d'identita' FR | +0.073 [-0.020, +0.173] | +0.034 [-0.050, +0.128] | +0.058 [-0.028, +0.154] | +0.054 [-0.045, +0.160] | +0.083 [-0.014, +0.192] | +0.066 [-0.023, +0.164] | +0.078 [-0.012, +0.172] | +0.084 [-0.007, +0.189] | +0.073 [-0.015, +0.173] |
| GNM (visto) vB, coefficienti | +0.077 [-0.019, +0.178] | +0.032 [-0.061, +0.133] | +0.061 [-0.029, +0.160] | +0.086 [-0.014, +0.189] | +0.074 [-0.020, +0.175] | +0.057 [-0.041, +0.168] | +0.086 [-0.012, +0.193] | +0.081 [-0.008, +0.177] | +0.078 [-0.017, +0.181] |
| GNM (visto) vB, mesh d'identita' FR | +0.010 [-0.062, +0.090] | -0.028 [-0.097, +0.046] | -0.004 [-0.074, +0.075] | -0.006 [-0.087, +0.082] | +0.023 [-0.053, +0.105] | -0.002 [-0.074, +0.078] | +0.022 [-0.049, +0.100] | +0.017 [-0.055, +0.097] | +0.011 [-0.060, +0.090] |
| GNM (visto) vB, composizione S_B + k d_P | -0.036 [-0.105, +0.045] | -0.073 [-0.143, +0.008] | -0.049 [-0.117, +0.030] | -0.049 [-0.127, +0.041] | -0.023 [-0.097, +0.059] | -0.052 [-0.123, +0.031] | -0.020 [-0.090, +0.060] | -0.033 [-0.104, +0.047] | -0.032 [-0.102, +0.050] |
| FLAME 2023 Open preregistrato, coefficienti | +0.128 [+0.036, +0.221] | +0.091 [-0.006, +0.189] | +0.114 [+0.023, +0.206] | +0.119 [+0.020, +0.220] | +0.138 [+0.044, +0.239] | +0.111 [+0.018, +0.204] | +0.135 [+0.047, +0.222] | +0.139 [+0.044, +0.233] | +0.127 [+0.032, +0.222] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.091 [+0.005, +0.177] | +0.059 [-0.026, +0.147] | +0.079 [-0.004, +0.165] | +0.077 [-0.015, +0.168] | +0.102 [+0.008, +0.199] | +0.074 [-0.010, +0.158] | +0.100 [+0.013, +0.182] | +0.106 [+0.021, +0.192] | +0.091 [+0.004, +0.181] |
| FLAME 2023 Open vA, coefficienti | +0.128 [+0.036, +0.221] | +0.091 [-0.006, +0.189] | +0.114 [+0.023, +0.206] | +0.119 [+0.020, +0.220] | +0.138 [+0.044, +0.239] | +0.111 [+0.018, +0.204] | +0.135 [+0.047, +0.222] | +0.139 [+0.044, +0.233] | +0.127 [+0.032, +0.222] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.091 [+0.005, +0.177] | +0.059 [-0.026, +0.147] | +0.079 [-0.004, +0.165] | +0.077 [-0.015, +0.168] | +0.102 [+0.008, +0.199] | +0.074 [-0.010, +0.158] | +0.100 [+0.013, +0.182] | +0.106 [+0.021, +0.192] | +0.091 [+0.004, +0.181] |
| FLAME 2023 Open vB, coefficienti | +0.100 [+0.002, +0.195] | +0.074 [-0.029, +0.183] | +0.090 [-0.005, +0.186] | +0.128 [+0.025, +0.240] | +0.109 [+0.018, +0.208] | +0.060 [-0.036, +0.162] | +0.107 [+0.016, +0.203] | +0.093 [+0.001, +0.181] | +0.108 [+0.016, +0.206] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.038 [-0.042, +0.129] | +0.007 [-0.072, +0.095] | +0.027 [-0.053, +0.115] | +0.023 [-0.064, +0.117] | +0.049 [-0.039, +0.146] | +0.020 [-0.061, +0.109] | +0.049 [-0.029, +0.134] | +0.052 [-0.028, +0.142] | +0.039 [-0.041, +0.132] |
| FLAME 2023 Open vB, composizione S_B + k d_P | +0.001 [-0.079, +0.089] | -0.031 [-0.116, +0.061] | -0.011 [-0.091, +0.076] | -0.010 [-0.102, +0.082] | +0.012 [-0.073, +0.105] | -0.020 [-0.104, +0.068] | +0.015 [-0.063, +0.103] | +0.010 [-0.071, +0.098] | +0.003 [-0.083, +0.095] |
| ArcFace-render, normal map 3 viste (G0) | +0.176 [+0.098, +0.252] | +0.092 [+0.000, +0.181] | +0.147 [+0.068, +0.226] | +0.156 [+0.065, +0.242] | +0.163 [+0.082, +0.240] | +0.146 [+0.057, +0.228] | +0.160 [+0.081, +0.232] | +0.210 [+0.108, +0.305] | +0.152 [+0.064, +0.234] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.185 [+0.096, +0.264] | +0.091 [-0.011, +0.187] | +0.153 [+0.063, +0.236] | +0.162 [+0.067, +0.247] | +0.176 [+0.084, +0.262] | +0.145 [+0.051, +0.230] | +0.167 [+0.070, +0.250] | +0.223 [+0.109, +0.326] | +0.157 [+0.061, +0.248] |
| ShapeDNA k=50 | +0.240 [+0.162, +0.308] | +0.195 [+0.104, +0.284] | +0.224 [+0.148, +0.297] | +0.225 [+0.135, +0.313] | +0.228 [+0.144, +0.308] | +0.201 [+0.121, +0.283] | +0.215 [+0.132, +0.291] | +0.234 [+0.138, +0.327] | +0.176 [+0.083, +0.266] |
| HKS globale | +0.271 [+0.197, +0.345] | +0.211 [+0.122, +0.295] | +0.249 [+0.174, +0.324] | +0.255 [+0.172, +0.335] | +0.267 [+0.188, +0.350] | +0.246 [+0.166, +0.324] | +0.258 [+0.178, +0.339] | +0.282 [+0.189, +0.371] | +0.236 [+0.149, +0.327] |
| WKS globale | +0.263 [+0.186, +0.331] | +0.187 [+0.106, +0.268] | +0.240 [+0.165, +0.312] | +0.264 [+0.181, +0.342] | +0.267 [+0.188, +0.343] | +0.239 [+0.160, +0.317] | +0.255 [+0.176, +0.326] | +0.221 [+0.120, +0.320] | +0.223 [+0.138, +0.305] |
| taglia stimata dalla mesh (non oracolo) | +0.176 [+0.086, +0.256] | +0.243 [+0.161, +0.320] | +0.207 [+0.133, +0.274] | +0.177 [+0.079, +0.266] | +0.189 [+0.094, +0.276] | +0.174 [+0.084, +0.254] | +0.176 [+0.084, +0.261] | +0.169 [+0.077, +0.252] | +0.173 [+0.080, +0.255] |
| taglia oracolo (S di FR) | +0.094 [-0.029, +0.208] | +0.043 [-0.075, +0.158] | +0.075 [-0.045, +0.188] | +0.091 [-0.036, +0.206] | +0.100 [-0.029, +0.213] | +0.080 [-0.043, +0.196] | +0.097 [-0.027, +0.216] | +0.103 [-0.019, +0.216] | +0.093 [-0.025, +0.208] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.040 [-0.118, +0.036] | -0.007 [-0.077, +0.067] | -0.026 [-0.098, +0.046] | -0.057 [-0.135, +0.020] | -0.040 [-0.120, +0.039] | -0.036 [-0.116, +0.047] | -0.035 [-0.117, +0.042] | -0.033 [-0.110, +0.043] | -0.043 [-0.118, +0.030] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.007 [-0.069, +0.085] | -0.008 [-0.082, +0.069] | +0.011 [-0.062, +0.083] | -0.013 [-0.088, +0.063] | +0.008 [-0.072, +0.087] | +0.012 [-0.066, +0.089] | +0.015 [-0.062, +0.089] | +0.013 [-0.065, +0.094] | +0.007 [-0.068, +0.082] |
| NICP per coppia in mm | +0.096 [+0.019, +0.169] | +0.039 [-0.032, +0.117] | +0.076 [+0.003, +0.151] | +0.090 [+0.010, +0.167] | +0.101 [+0.019, +0.181] | +0.088 [+0.004, +0.172] | +0.103 [+0.027, +0.182] | +0.098 [+0.018, +0.172] | - |
| NICP per coppia cs | +0.085 [+0.007, +0.164] | +0.034 [-0.039, +0.107] | +0.068 [-0.006, +0.141] | +0.082 [+0.003, +0.158] | +0.089 [+0.004, +0.171] | +0.073 [-0.010, +0.157] | +0.095 [+0.017, +0.173] | +0.086 [+0.007, +0.164] | - |
| NICP su template in mm | +0.085 [-0.028, +0.196] | +0.140 [+0.046, +0.226] | +0.108 [+0.007, +0.208] | +0.079 [-0.042, +0.197] | +0.073 [-0.044, +0.186] | +0.085 [-0.030, +0.194] | +0.091 [-0.025, +0.199] | +0.100 [-0.015, +0.215] | +0.087 [-0.022, +0.196] |
| NICP su template cs | +0.162 [+0.063, +0.263] | +0.142 [+0.054, +0.225] | +0.159 [+0.072, +0.246] | +0.145 [+0.048, +0.250] | +0.158 [+0.059, +0.261] | +0.172 [+0.067, +0.284] | +0.156 [+0.058, +0.253] | +0.176 [+0.076, +0.280] | +0.154 [+0.064, +0.253] |
| varifold in mm (massa unitaria) | +0.037 [-0.026, +0.097] | +0.027 [-0.039, +0.097] | +0.038 [-0.022, +0.099] | +0.011 [-0.056, +0.072] | +0.026 [-0.042, +0.097] | +0.016 [-0.051, +0.080] | +0.019 [-0.049, +0.083] | +0.006 [-0.069, +0.076] | +0.015 [-0.044, +0.079] |
| GNM (visto) preregistrato, coefficienti | +0.077 [-0.019, +0.178] | +0.049 [-0.058, +0.163] | +0.067 [-0.028, +0.168] | +0.057 [-0.046, +0.164] | +0.095 [-0.003, +0.206] | +0.088 [-0.019, +0.188] | +0.060 [-0.032, +0.148] | +0.088 [-0.013, +0.195] | +0.079 [-0.017, +0.181] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.070 [-0.032, +0.174] | +0.049 [-0.046, +0.151] | +0.062 [-0.037, +0.163] | +0.051 [-0.057, +0.165] | +0.076 [-0.031, +0.187] | +0.068 [-0.027, +0.168] | +0.075 [-0.027, +0.170] | +0.082 [-0.018, +0.190] | +0.073 [-0.027, +0.179] |
| GNM (visto) vA, coefficienti | +0.074 [-0.024, +0.177] | +0.043 [-0.063, +0.158] | +0.063 [-0.031, +0.164] | +0.056 [-0.048, +0.161] | +0.093 [-0.010, +0.207] | +0.083 [-0.022, +0.184] | +0.057 [-0.036, +0.150] | +0.084 [-0.016, +0.193] | +0.075 [-0.021, +0.177] |
| GNM (visto) vA, mesh d'identita' FR | +0.067 [-0.035, +0.170] | +0.046 [-0.050, +0.149] | +0.059 [-0.040, +0.161] | +0.048 [-0.059, +0.164] | +0.072 [-0.037, +0.184] | +0.065 [-0.032, +0.165] | +0.073 [-0.029, +0.169] | +0.079 [-0.022, +0.188] | +0.070 [-0.031, +0.177] |
| GNM (visto) vB, coefficienti | +0.071 [-0.006, +0.157] | +0.045 [-0.039, +0.138] | +0.062 [-0.015, +0.145] | +0.081 [-0.003, +0.175] | +0.063 [-0.019, +0.152] | +0.056 [-0.031, +0.147] | +0.080 [+0.000, +0.169] | +0.076 [-0.005, +0.161] | +0.076 [-0.003, +0.163] |
| GNM (visto) vB, mesh d'identita' FR | +0.004 [-0.078, +0.084] | -0.016 [-0.093, +0.068] | -0.003 [-0.082, +0.077] | -0.011 [-0.107, +0.085] | +0.012 [-0.075, +0.095] | -0.003 [-0.084, +0.076] | +0.016 [-0.063, +0.095] | +0.011 [-0.072, +0.093] | +0.009 [-0.072, +0.091] |
| GNM (visto) vB, composizione S_B + k d_P | -0.041 [-0.115, +0.034] | -0.060 [-0.133, +0.021] | -0.048 [-0.120, +0.026] | -0.054 [-0.137, +0.033] | -0.034 [-0.110, +0.045] | -0.053 [-0.125, +0.022] | -0.026 [-0.099, +0.047] | -0.038 [-0.114, +0.037] | -0.035 [-0.110, +0.043] |
| FLAME 2023 Open preregistrato, coefficienti | +0.122 [+0.037, +0.209] | +0.103 [+0.007, +0.201] | +0.115 [+0.029, +0.204] | +0.114 [+0.019, +0.206] | +0.127 [+0.027, +0.226] | +0.110 [+0.024, +0.198] | +0.129 [+0.043, +0.212] | +0.134 [+0.043, +0.223] | +0.124 [+0.035, +0.216] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.086 [-0.008, +0.173] | +0.071 [-0.023, +0.166] | +0.080 [-0.010, +0.168] | +0.071 [-0.029, +0.166] | +0.091 [-0.008, +0.190] | +0.073 [-0.016, +0.160] | +0.094 [-0.003, +0.180] | +0.100 [+0.007, +0.192] | +0.089 [-0.003, +0.178] |
| FLAME 2023 Open vA, coefficienti | +0.122 [+0.037, +0.209] | +0.103 [+0.007, +0.201] | +0.115 [+0.029, +0.204] | +0.114 [+0.019, +0.206] | +0.127 [+0.027, +0.226] | +0.110 [+0.024, +0.198] | +0.129 [+0.043, +0.212] | +0.134 [+0.043, +0.223] | +0.124 [+0.035, +0.216] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.086 [-0.008, +0.173] | +0.071 [-0.023, +0.166] | +0.080 [-0.010, +0.168] | +0.071 [-0.029, +0.166] | +0.091 [-0.008, +0.190] | +0.073 [-0.016, +0.160] | +0.094 [-0.003, +0.180] | +0.100 [+0.007, +0.192] | +0.089 [-0.003, +0.178] |
| FLAME 2023 Open vB, coefficienti | +0.094 [+0.015, +0.178] | +0.086 [-0.010, +0.188] | +0.091 [+0.009, +0.176] | +0.123 [+0.034, +0.221] | +0.098 [+0.013, +0.190] | +0.059 [-0.025, +0.142] | +0.101 [+0.020, +0.187] | +0.088 [+0.004, +0.167] | +0.106 [+0.024, +0.192] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.033 [-0.053, +0.118] | +0.020 [-0.069, +0.113] | +0.028 [-0.055, +0.114] | +0.018 [-0.075, +0.112] | +0.038 [-0.053, +0.132] | +0.019 [-0.066, +0.104] | +0.043 [-0.046, +0.126] | +0.046 [-0.039, +0.134] | +0.037 [-0.050, +0.126] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.005 [-0.086, +0.078] | -0.018 [-0.102, +0.074] | -0.010 [-0.090, +0.074] | -0.016 [-0.101, +0.073] | +0.001 [-0.085, +0.086] | -0.021 [-0.103, +0.061] | +0.010 [-0.070, +0.086] | +0.005 [-0.074, +0.089] | +0.000 [-0.080, +0.087] |
| ArcFace-render, normal map 3 viste (G0) | +0.170 [+0.091, +0.244] | +0.105 [+0.015, +0.195] | +0.148 [+0.068, +0.223] | +0.151 [+0.062, +0.237] | +0.152 [+0.068, +0.229] | +0.145 [+0.058, +0.228] | +0.154 [+0.075, +0.223] | +0.204 [+0.111, +0.297] | +0.150 [+0.068, +0.226] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.179 [+0.095, +0.260] | +0.104 [+0.007, +0.198] | +0.154 [+0.068, +0.237] | +0.156 [+0.065, +0.240] | +0.165 [+0.076, +0.250] | +0.144 [+0.048, +0.233] | +0.162 [+0.076, +0.244] | +0.217 [+0.113, +0.316] | +0.154 [+0.067, +0.238] |
| ShapeDNA k=50 | +0.234 [+0.164, +0.300] | +0.207 [+0.122, +0.292] | +0.226 [+0.152, +0.294] | +0.220 [+0.141, +0.296] | +0.217 [+0.136, +0.295] | +0.200 [+0.123, +0.278] | +0.209 [+0.138, +0.282] | +0.229 [+0.143, +0.311] | +0.173 [+0.088, +0.259] |
| HKS globale | +0.265 [+0.194, +0.334] | +0.223 [+0.135, +0.311] | +0.250 [+0.174, +0.322] | +0.249 [+0.171, +0.329] | +0.256 [+0.178, +0.337] | +0.245 [+0.167, +0.318] | +0.252 [+0.174, +0.324] | +0.277 [+0.190, +0.357] | +0.233 [+0.144, +0.317] |
| WKS globale | +0.258 [+0.182, +0.326] | +0.199 [+0.115, +0.282] | +0.241 [+0.167, +0.310] | +0.258 [+0.181, +0.337] | +0.256 [+0.172, +0.334] | +0.238 [+0.157, +0.314] | +0.249 [+0.175, +0.315] | +0.215 [+0.125, +0.306] | +0.221 [+0.138, +0.303] |
| taglia stimata dalla mesh (non oracolo) | +0.171 [+0.058, +0.269] | +0.256 [+0.170, +0.333] | +0.209 [+0.119, +0.288] | +0.171 [+0.053, +0.278] | +0.178 [+0.060, +0.277] | +0.173 [+0.064, +0.271] | +0.170 [+0.058, +0.266] | +0.163 [+0.056, +0.263] | +0.171 [+0.063, +0.268] |
| taglia oracolo (S di FR) | +0.088 [-0.050, +0.213] | +0.056 [-0.073, +0.175] | +0.076 [-0.060, +0.199] | +0.085 [-0.055, +0.213] | +0.089 [-0.055, +0.216] | +0.079 [-0.056, +0.204] | +0.091 [-0.047, +0.213] | +0.097 [-0.039, +0.217] | +0.091 [-0.045, +0.212] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.023 [-0.097, +0.056] | +0.001 [-0.070, +0.073] | -0.014 [-0.082, +0.058] | -0.037 [-0.111, +0.040] | -0.020 [-0.094, +0.057] | -0.025 [-0.105, +0.060] | -0.020 [-0.096, +0.057] | -0.016 [-0.089, +0.062] | -0.027 [-0.100, +0.049] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.020 [-0.093, +0.055] | -0.036 [-0.106, +0.042] | -0.016 [-0.085, +0.053] | -0.039 [-0.108, +0.039] | -0.014 [-0.091, +0.062] | -0.017 [-0.091, +0.060] | -0.017 [-0.089, +0.057] | -0.014 [-0.089, +0.067] | -0.020 [-0.090, +0.054] |
| NICP per coppia in mm | +0.063 [-0.016, +0.141] | +0.004 [-0.068, +0.081] | +0.042 [-0.034, +0.114] | +0.056 [-0.022, +0.135] | +0.071 [-0.012, +0.149] | +0.053 [-0.031, +0.135] | +0.067 [-0.014, +0.148] | +0.066 [-0.013, +0.143] | - |
| NICP per coppia cs | +0.051 [-0.027, +0.124] | -0.006 [-0.076, +0.063] | +0.032 [-0.037, +0.102] | +0.047 [-0.029, +0.115] | +0.059 [-0.022, +0.135] | +0.037 [-0.042, +0.115] | +0.058 [-0.020, +0.135] | +0.052 [-0.025, +0.125] | - |
| NICP su template in mm | +0.134 [+0.018, +0.239] | +0.167 [+0.074, +0.249] | +0.149 [+0.049, +0.238] | +0.132 [+0.009, +0.244] | +0.128 [+0.013, +0.236] | +0.129 [+0.011, +0.238] | +0.139 [+0.028, +0.244] | +0.145 [+0.021, +0.253] | +0.135 [+0.025, +0.236] |
| NICP su template cs | +0.139 [+0.042, +0.237] | +0.130 [+0.044, +0.212] | +0.139 [+0.052, +0.224] | +0.124 [+0.028, +0.224] | +0.143 [+0.046, +0.242] | +0.148 [+0.042, +0.254] | +0.129 [+0.029, +0.223] | +0.153 [+0.050, +0.253] | +0.135 [+0.043, +0.233] |
| varifold in mm (massa unitaria) | +0.036 [-0.021, +0.092] | +0.024 [-0.038, +0.092] | +0.035 [-0.019, +0.091] | +0.013 [-0.046, +0.071] | +0.030 [-0.033, +0.092] | +0.011 [-0.050, +0.073] | +0.020 [-0.038, +0.080] | +0.003 [-0.070, +0.072] | +0.017 [-0.038, +0.076] |
| GNM (visto) preregistrato, coefficienti | +0.104 [+0.015, +0.200] | +0.074 [-0.027, +0.180] | +0.093 [+0.003, +0.188] | +0.082 [-0.016, +0.186] | +0.127 [+0.030, +0.241] | +0.110 [+0.018, +0.204] | +0.088 [+0.008, +0.171] | +0.113 [+0.019, +0.222] | +0.105 [+0.015, +0.202] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.077 [-0.016, +0.179] | +0.049 [-0.041, +0.147] | +0.067 [-0.022, +0.165] | +0.066 [-0.032, +0.175] | +0.096 [-0.000, +0.199] | +0.059 [-0.031, +0.156] | +0.078 [-0.013, +0.174] | +0.087 [-0.009, +0.193] | +0.078 [-0.012, +0.181] |
| GNM (visto) vA, coefficienti | +0.102 [+0.014, +0.199] | +0.069 [-0.031, +0.176] | +0.090 [+0.001, +0.188] | +0.082 [-0.018, +0.185] | +0.126 [+0.030, +0.238] | +0.107 [+0.014, +0.202] | +0.087 [+0.006, +0.169] | +0.111 [+0.014, +0.219] | +0.103 [+0.014, +0.201] |
| GNM (visto) vA, mesh d'identita' SR | +0.081 [-0.010, +0.183] | +0.052 [-0.039, +0.153] | +0.071 [-0.019, +0.170] | +0.072 [-0.027, +0.186] | +0.099 [+0.004, +0.204] | +0.063 [-0.028, +0.160] | +0.082 [-0.011, +0.181] | +0.092 [-0.004, +0.199] | +0.083 [-0.008, +0.183] |
| GNM (visto) vB, coefficienti | +0.064 [-0.022, +0.158] | +0.027 [-0.066, +0.128] | +0.051 [-0.034, +0.144] | +0.073 [-0.017, +0.174] | +0.062 [-0.024, +0.162] | +0.046 [-0.043, +0.144] | +0.069 [-0.016, +0.165] | +0.070 [-0.014, +0.158] | +0.066 [-0.021, +0.161] |
| GNM (visto) vB, mesh d'identita' SR | -0.041 [-0.112, +0.033] | -0.065 [-0.141, +0.021] | -0.050 [-0.119, +0.028] | -0.043 [-0.118, +0.040] | -0.031 [-0.105, +0.044] | -0.061 [-0.134, +0.016] | -0.031 [-0.102, +0.042] | -0.039 [-0.113, +0.033] | -0.034 [-0.106, +0.043] |
| FLAME 2023 Open preregistrato, coefficienti | +0.134 [+0.046, +0.224] | +0.110 [+0.019, +0.211] | +0.125 [+0.038, +0.211] | +0.122 [+0.029, +0.221] | +0.146 [+0.051, +0.246] | +0.124 [+0.037, +0.212] | +0.138 [+0.050, +0.218] | +0.141 [+0.053, +0.233] | +0.135 [+0.046, +0.224] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.131 [+0.040, +0.233] | +0.099 [-0.001, +0.209] | +0.119 [+0.027, +0.221] | +0.130 [+0.027, +0.242] | +0.143 [+0.049, +0.251] | +0.114 [+0.022, +0.214] | +0.130 [+0.037, +0.226] | +0.139 [+0.049, +0.241] | +0.129 [+0.035, +0.234] |
| FLAME 2023 Open vA, coefficienti | +0.134 [+0.046, +0.224] | +0.110 [+0.019, +0.211] | +0.125 [+0.038, +0.211] | +0.122 [+0.029, +0.221] | +0.146 [+0.051, +0.246] | +0.124 [+0.037, +0.212] | +0.138 [+0.050, +0.218] | +0.141 [+0.053, +0.233] | +0.135 [+0.046, +0.224] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.131 [+0.040, +0.233] | +0.099 [-0.001, +0.209] | +0.119 [+0.027, +0.221] | +0.130 [+0.027, +0.242] | +0.143 [+0.049, +0.251] | +0.114 [+0.022, +0.214] | +0.130 [+0.037, +0.226] | +0.139 [+0.049, +0.241] | +0.129 [+0.035, +0.234] |
| FLAME 2023 Open vB, coefficienti | +0.093 [+0.008, +0.182] | +0.073 [-0.031, +0.179] | +0.086 [-0.001, +0.179] | +0.120 [+0.017, +0.220] | +0.105 [+0.016, +0.194] | +0.059 [-0.023, +0.148] | +0.092 [+0.000, +0.180] | +0.092 [+0.001, +0.174] | +0.101 [+0.014, +0.193] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.006 [-0.090, +0.084] | -0.027 [-0.117, +0.066] | -0.014 [-0.099, +0.077] | -0.010 [-0.098, +0.084] | +0.003 [-0.083, +0.095] | -0.029 [-0.114, +0.066] | +0.003 [-0.079, +0.089] | +0.001 [-0.084, +0.091] | -0.001 [-0.086, +0.088] |
| ArcFace-render, normal map 3 viste (G0) | +0.148 [+0.068, +0.220] | +0.078 [-0.011, +0.169] | +0.125 [+0.046, +0.203] | +0.128 [+0.042, +0.213] | +0.136 [+0.051, +0.211] | +0.120 [+0.032, +0.202] | +0.133 [+0.055, +0.206] | +0.176 [+0.083, +0.271] | +0.126 [+0.045, +0.204] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.157 [+0.074, +0.234] | +0.072 [-0.019, +0.162] | +0.129 [+0.045, +0.208] | +0.131 [+0.045, +0.210] | +0.147 [+0.059, +0.230] | +0.114 [+0.024, +0.198] | +0.140 [+0.052, +0.220] | +0.198 [+0.095, +0.295] | +0.124 [+0.035, +0.209] |
| ShapeDNA k=50 | +0.225 [+0.153, +0.295] | +0.183 [+0.100, +0.268] | +0.211 [+0.138, +0.281] | +0.206 [+0.121, +0.287] | +0.213 [+0.126, +0.291] | +0.193 [+0.115, +0.272] | +0.204 [+0.125, +0.278] | +0.214 [+0.131, +0.298] | +0.163 [+0.072, +0.251] |
| HKS globale | +0.257 [+0.185, +0.326] | +0.211 [+0.125, +0.297] | +0.241 [+0.166, +0.310] | +0.240 [+0.161, +0.316] | +0.256 [+0.171, +0.332] | +0.242 [+0.162, +0.316] | +0.252 [+0.172, +0.325] | +0.254 [+0.172, +0.331] | +0.228 [+0.136, +0.310] |
| WKS globale | +0.247 [+0.173, +0.313] | +0.189 [+0.109, +0.267] | +0.230 [+0.157, +0.298] | +0.245 [+0.167, +0.323] | +0.250 [+0.165, +0.330] | +0.230 [+0.145, +0.306] | +0.241 [+0.166, +0.309] | +0.198 [+0.107, +0.287] | +0.217 [+0.134, +0.300] |
| taglia stimata dalla mesh (non oracolo) | +0.206 [+0.100, +0.299] | +0.252 [+0.167, +0.328] | +0.228 [+0.147, +0.298] | +0.211 [+0.099, +0.308] | +0.213 [+0.107, +0.303] | +0.199 [+0.094, +0.295] | +0.212 [+0.107, +0.303] | +0.197 [+0.090, +0.293] | +0.205 [+0.106, +0.297] |
| taglia oracolo (S di FR) | +0.267 [+0.144, +0.380] | +0.233 [+0.122, +0.339] | +0.255 [+0.136, +0.363] | +0.263 [+0.132, +0.379] | +0.275 [+0.149, +0.389] | +0.258 [+0.139, +0.368] | +0.271 [+0.145, +0.378] | +0.271 [+0.152, +0.379] | +0.270 [+0.149, +0.376] |

### FaceVerse neutra: Spearman per insieme di righe (seme 271049)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (99000) | crop (49500) | all_cross (148500) | con_perturbazione original (39600) | con_perturbazione remesh (39600) | con_perturbazione down8k (39600) | con_perturbazione up60k (39600) | con_perturbazione noisy (39600) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.341 [0.263, 0.416] | 0.282 [0.206, 0.360] | 0.319 [0.242, 0.395] | 0.343 [0.264, 0.420] | 0.343 [0.264, 0.419] | 0.337 [0.258, 0.412] | 0.346 [0.264, 0.424] | 0.336 [0.256, 0.411] | 0.349 [0.266, 0.427] |
| **factorized s1234, d_F non cal.** | 0.337 [0.256, 0.409] | 0.306 [0.221, 0.390] | 0.326 [0.249, 0.400] | 0.341 [0.259, 0.414] | 0.340 [0.260, 0.413] | 0.335 [0.255, 0.408] | 0.342 [0.259, 0.416] | 0.328 [0.246, 0.397] | 0.349 [0.269, 0.428] |
| factorized s1234, d_P | 0.321 [0.237, 0.391] | 0.293 [0.210, 0.374] | 0.310 [0.229, 0.384] | 0.325 [0.241, 0.398] | 0.324 [0.241, 0.397] | 0.319 [0.236, 0.393] | 0.326 [0.240, 0.402] | 0.309 [0.224, 0.382] | 0.335 [0.251, 0.412] |
| **factorized s2345, d_F cal.*** | 0.370 [0.298, 0.440] | 0.272 [0.207, 0.335] | 0.335 [0.267, 0.400] | 0.372 [0.298, 0.442] | 0.376 [0.303, 0.445] | 0.368 [0.294, 0.437] | 0.371 [0.296, 0.441] | 0.366 [0.295, 0.433] | 0.370 [0.298, 0.441] |
| **factorized s2345, d_F non cal.** | 0.373 [0.293, 0.451] | 0.310 [0.231, 0.388] | 0.352 [0.273, 0.427] | 0.375 [0.294, 0.453] | 0.378 [0.297, 0.454] | 0.373 [0.292, 0.450] | 0.374 [0.293, 0.452] | 0.367 [0.289, 0.444] | 0.374 [0.295, 0.450] |
| factorized s2345, d_P | 0.361 [0.276, 0.439] | 0.306 [0.226, 0.382] | 0.341 [0.259, 0.417] | 0.363 [0.279, 0.441] | 0.365 [0.279, 0.444] | 0.361 [0.278, 0.439] | 0.362 [0.277, 0.440] | 0.354 [0.269, 0.433] | 0.364 [0.277, 0.441] |
| **ctrlfr s1234** | 0.327 [0.248, 0.399] | 0.296 [0.225, 0.363] | 0.316 [0.242, 0.388] | 0.337 [0.257, 0.411] | 0.334 [0.256, 0.407] | 0.334 [0.258, 0.407] | 0.339 [0.261, 0.415] | 0.317 [0.238, 0.391] | 0.364 [0.285, 0.439] |
| **ctrlfr s2345** | 0.299 [0.214, 0.374] | 0.269 [0.192, 0.340] | 0.289 [0.207, 0.360] | 0.308 [0.222, 0.383] | 0.305 [0.221, 0.379] | 0.306 [0.221, 0.379] | 0.304 [0.217, 0.380] | 0.286 [0.200, 0.359] | 0.330 [0.242, 0.405] |
| **C3M e205, d_F cal.*** | 0.336 [0.264, 0.400] | 0.299 [0.229, 0.364] | 0.324 [0.255, 0.387] | 0.358 [0.283, 0.424] | 0.350 [0.274, 0.418] | 0.352 [0.275, 0.420] | 0.352 [0.276, 0.422] | 0.317 [0.250, 0.380] | 0.395 [0.312, 0.470] |
| **C3M e205, d_F non cal.** | 0.357 [0.273, 0.433] | 0.345 [0.262, 0.420] | 0.353 [0.269, 0.429] | 0.371 [0.287, 0.449] | 0.366 [0.279, 0.444] | 0.362 [0.275, 0.439] | 0.368 [0.278, 0.451] | 0.343 [0.259, 0.418] | 0.393 [0.309, 0.472] |
| C3M e205, d_P | 0.363 [0.275, 0.442] | 0.345 [0.258, 0.419] | 0.357 [0.272, 0.434] | 0.377 [0.289, 0.455] | 0.372 [0.282, 0.451] | 0.368 [0.281, 0.447] | 0.371 [0.281, 0.452] | 0.340 [0.259, 0.415] | 0.394 [0.304, 0.474] |
| **ICP + Chamfer in mm (per coppia)** | 0.409 [0.326, 0.481] | 0.317 [0.248, 0.379] | 0.373 [0.297, 0.442] | 0.415 [0.332, 0.486] | 0.414 [0.331, 0.486] | 0.402 [0.314, 0.475] | 0.412 [0.328, 0.484] | 0.404 [0.325, 0.476] | 0.419 [0.337, 0.492] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.374 [0.293, 0.449] | 0.333 [0.262, 0.399] | 0.348 [0.274, 0.417] | 0.378 [0.297, 0.452] | 0.380 [0.298, 0.455] | 0.367 [0.281, 0.443] | 0.376 [0.296, 0.451] | 0.373 [0.292, 0.447] | 0.382 [0.302, 0.455] |
| **NICP per coppia in mm** | 0.278 [0.194, 0.360] | 0.295 [0.214, 0.371] | 0.283 [0.205, 0.362] | 0.278 [0.196, 0.358] | 0.280 [0.195, 0.363] | 0.272 [0.184, 0.359] | 0.279 [0.195, 0.361] | 0.281 [0.201, 0.359] | - |
| **NICP per coppia cs** | 0.286 [0.198, 0.372] | 0.290 [0.209, 0.368] | 0.287 [0.202, 0.368] | 0.286 [0.199, 0.370] | 0.287 [0.199, 0.373] | 0.282 [0.191, 0.369] | 0.287 [0.198, 0.373] | 0.292 [0.204, 0.373] | - |
| **NICP su template in mm** | 0.223 [0.125, 0.319] | 0.162 [0.098, 0.225] | 0.195 [0.112, 0.278] | 0.224 [0.125, 0.320] | 0.225 [0.124, 0.322] | 0.228 [0.130, 0.323] | 0.223 [0.125, 0.320] | 0.218 [0.121, 0.318] | 0.228 [0.129, 0.325] |
| **varifold in mm (massa unitaria)** | 0.285 [0.223, 0.346] | 0.255 [0.203, 0.304] | 0.270 [0.214, 0.324] | 0.323 [0.255, 0.389] | 0.307 [0.243, 0.368] | 0.323 [0.255, 0.390] | 0.322 [0.254, 0.389] | 0.338 [0.262, 0.413] | 0.346 [0.281, 0.408] |
| **GNM (visto) preregistrato, coefficienti** | 0.240 [0.164, 0.309] | 0.239 [0.164, 0.306] | 0.240 [0.164, 0.306] | 0.244 [0.168, 0.314] | 0.236 [0.159, 0.303] | 0.244 [0.168, 0.310] | 0.244 [0.166, 0.312] | 0.236 [0.161, 0.305] | 0.240 [0.165, 0.308] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.232 [0.140, 0.318] | 0.234 [0.147, 0.316] | 0.232 [0.142, 0.317] | 0.235 [0.144, 0.320] | 0.230 [0.135, 0.317] | 0.237 [0.144, 0.323] | 0.232 [0.138, 0.318] | 0.225 [0.135, 0.312] | 0.232 [0.140, 0.318] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.240 [0.157, 0.320] | 0.244 [0.166, 0.320] | 0.241 [0.160, 0.319] | 0.246 [0.162, 0.325] | 0.238 [0.155, 0.319] | 0.243 [0.161, 0.322] | 0.244 [0.160, 0.324] | 0.229 [0.147, 0.308] | 0.244 [0.162, 0.323] |
| **GNM (visto) vA, coefficienti*** | 0.230 [0.152, 0.304] | 0.233 [0.157, 0.305] | 0.231 [0.153, 0.304] | 0.235 [0.155, 0.308] | 0.225 [0.146, 0.299] | 0.233 [0.154, 0.305] | 0.234 [0.154, 0.308] | 0.226 [0.151, 0.301] | 0.231 [0.154, 0.303] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.236 [0.142, 0.321] | 0.238 [0.148, 0.323] | 0.237 [0.145, 0.322] | 0.239 [0.145, 0.324] | 0.235 [0.140, 0.321] | 0.242 [0.147, 0.326] | 0.237 [0.141, 0.321] | 0.230 [0.138, 0.318] | 0.237 [0.143, 0.324] |
| GNM (visto) vA, mesh d'identita' SR* | 0.222 [0.135, 0.301] | 0.228 [0.145, 0.305] | 0.224 [0.137, 0.301] | 0.228 [0.140, 0.307] | 0.222 [0.134, 0.303] | 0.225 [0.138, 0.304] | 0.226 [0.138, 0.307] | 0.211 [0.125, 0.290] | 0.227 [0.141, 0.304] |
| **GNM (visto) vB, coefficienti*** | 0.299 [0.197, 0.384] | 0.292 [0.194, 0.376] | 0.297 [0.196, 0.380] | 0.298 [0.196, 0.383] | 0.299 [0.195, 0.385] | 0.301 [0.199, 0.384] | 0.299 [0.198, 0.383] | 0.303 [0.201, 0.387] | 0.295 [0.195, 0.378] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.321 [0.225, 0.400] | 0.323 [0.232, 0.400] | 0.321 [0.226, 0.400] | 0.322 [0.224, 0.401] | 0.318 [0.222, 0.398] | 0.324 [0.229, 0.404] | 0.320 [0.224, 0.398] | 0.321 [0.225, 0.399] | 0.322 [0.226, 0.402] |
| GNM (visto) vB, mesh d'identita' SR* | 0.373 [0.288, 0.449] | 0.379 [0.295, 0.453] | 0.375 [0.291, 0.450] | 0.375 [0.292, 0.453] | 0.372 [0.286, 0.448] | 0.373 [0.289, 0.450] | 0.373 [0.288, 0.450] | 0.372 [0.287, 0.446] | 0.375 [0.291, 0.451] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.382 [0.293, 0.458] | 0.385 [0.299, 0.462] | 0.383 [0.296, 0.459] | 0.383 [0.293, 0.460] | 0.380 [0.290, 0.458] | 0.384 [0.296, 0.461] | 0.381 [0.292, 0.456] | 0.383 [0.294, 0.459] | 0.383 [0.295, 0.461] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.171 [0.102, 0.241] | 0.173 [0.105, 0.242] | 0.172 [0.104, 0.242] | 0.169 [0.099, 0.241] | 0.175 [0.104, 0.245] | 0.173 [0.103, 0.242] | 0.174 [0.102, 0.245] | 0.165 [0.095, 0.235] | 0.174 [0.104, 0.246] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.214 [0.134, 0.291] | 0.214 [0.135, 0.289] | 0.214 [0.133, 0.289] | 0.212 [0.131, 0.289] | 0.220 [0.140, 0.300] | 0.216 [0.135, 0.293] | 0.214 [0.131, 0.291] | 0.208 [0.122, 0.286] | 0.215 [0.133, 0.293] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.169 [0.089, 0.244] | 0.170 [0.097, 0.244] | 0.169 [0.090, 0.243] | 0.166 [0.085, 0.241] | 0.170 [0.089, 0.245] | 0.172 [0.092, 0.246] | 0.173 [0.091, 0.249] | 0.163 [0.082, 0.235] | 0.172 [0.092, 0.246] |
| **FLAME 2023 Open vA, coefficienti*** | 0.179 [0.108, 0.249] | 0.181 [0.112, 0.248] | 0.179 [0.110, 0.247] | 0.177 [0.106, 0.248] | 0.183 [0.110, 0.252] | 0.180 [0.110, 0.250] | 0.181 [0.109, 0.251] | 0.173 [0.103, 0.245] | 0.181 [0.109, 0.251] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.220 [0.134, 0.299] | 0.219 [0.135, 0.297] | 0.220 [0.136, 0.298] | 0.218 [0.133, 0.298] | 0.227 [0.141, 0.306] | 0.223 [0.138, 0.301] | 0.220 [0.133, 0.299] | 0.214 [0.124, 0.296] | 0.220 [0.135, 0.300] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.175 [0.094, 0.248] | 0.177 [0.101, 0.250] | 0.175 [0.095, 0.249] | 0.172 [0.090, 0.246] | 0.176 [0.094, 0.250] | 0.178 [0.099, 0.251] | 0.179 [0.096, 0.256] | 0.169 [0.089, 0.242] | 0.178 [0.096, 0.252] |
| **FLAME 2023 Open vB, coefficienti*** | 0.249 [0.143, 0.341] | 0.223 [0.122, 0.317] | 0.240 [0.137, 0.331] | 0.247 [0.141, 0.342] | 0.250 [0.141, 0.342] | 0.255 [0.146, 0.346] | 0.248 [0.143, 0.340] | 0.246 [0.142, 0.339] | 0.239 [0.135, 0.330] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.315 [0.223, 0.393] | 0.306 [0.220, 0.384] | 0.312 [0.222, 0.391] | 0.314 [0.223, 0.392] | 0.318 [0.229, 0.396] | 0.316 [0.224, 0.395] | 0.314 [0.222, 0.394] | 0.313 [0.220, 0.395] | 0.313 [0.221, 0.393] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.358 [0.275, 0.438] | 0.348 [0.265, 0.429] | 0.355 [0.272, 0.434] | 0.356 [0.272, 0.437] | 0.359 [0.275, 0.439] | 0.359 [0.276, 0.438] | 0.361 [0.278, 0.441] | 0.357 [0.274, 0.438] | 0.356 [0.273, 0.435] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.367 [0.280, 0.444] | 0.357 [0.268, 0.433] | 0.363 [0.277, 0.440] | 0.365 [0.278, 0.443] | 0.369 [0.284, 0.446] | 0.368 [0.282, 0.445] | 0.367 [0.280, 0.445] | 0.365 [0.276, 0.443] | 0.364 [0.278, 0.442] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.153 [0.110, 0.198] | 0.181 [0.130, 0.233] | 0.162 [0.116, 0.208] | 0.173 [0.125, 0.221] | 0.174 [0.124, 0.222] | 0.180 [0.129, 0.230] | 0.172 [0.124, 0.221] | 0.124 [0.064, 0.192] | 0.178 [0.127, 0.230] |
| **ShapeDNA k=50** | 0.073 [0.036, 0.109] | 0.059 [0.012, 0.104] | 0.067 [0.030, 0.101] | 0.095 [0.037, 0.146] | 0.099 [0.044, 0.150] | 0.096 [0.040, 0.147] | 0.098 [0.042, 0.151] | 0.094 [0.034, 0.159] | 0.142 [0.064, 0.217] |
| **HKS globale** | 0.036 [0.009, 0.063] | 0.038 [-0.004, 0.081] | 0.035 [0.006, 0.066] | 0.045 [0.006, 0.085] | 0.044 [0.006, 0.084] | 0.045 [0.006, 0.086] | 0.046 [0.007, 0.086] | 0.057 [-0.004, 0.111] | 0.073 [0.014, 0.133] |
| **WKS globale** | 0.051 [0.023, 0.076] | 0.056 [0.019, 0.089] | 0.049 [0.021, 0.074] | 0.063 [0.019, 0.105] | 0.063 [0.023, 0.099] | 0.057 [0.017, 0.093] | 0.057 [0.021, 0.090] | 0.111 [0.050, 0.168] | 0.076 [0.006, 0.139] |
| **taglia stimata dalla mesh (non oracolo)** | 0.143 [0.045, 0.237] | 0.022 [0.002, 0.049] | 0.089 [0.030, 0.149] | 0.144 [0.044, 0.242] | 0.140 [0.044, 0.230] | 0.144 [0.044, 0.241] | 0.145 [0.044, 0.242] | 0.144 [0.052, 0.234] | 0.143 [0.044, 0.241] |
| **taglia oracolo (S di FR)** | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (99000) | crop (49500) | all_cross (148500) | con_perturbazione original (39600) | con_perturbazione remesh (39600) | con_perturbazione down8k (39600) | con_perturbazione up60k (39600) | con_perturbazione noisy (39600) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.308 [0.225, 0.387] | 0.255 [0.180, 0.330] | 0.288 [0.211, 0.364] | 0.310 [0.226, 0.388] | 0.310 [0.226, 0.389] | 0.304 [0.221, 0.383] | 0.312 [0.227, 0.392] | 0.303 [0.221, 0.381] | 0.315 [0.230, 0.395] |
| factorized s1234, d_F non cal. | 0.335 [0.254, 0.407] | 0.303 [0.223, 0.383] | 0.324 [0.244, 0.398] | 0.338 [0.258, 0.415] | 0.338 [0.257, 0.412] | 0.333 [0.254, 0.404] | 0.340 [0.257, 0.417] | 0.326 [0.247, 0.395] | 0.346 [0.265, 0.422] |
| **factorized s1234, d_P** | 0.333 [0.251, 0.405] | 0.304 [0.224, 0.382] | 0.323 [0.246, 0.393] | 0.338 [0.256, 0.409] | 0.337 [0.256, 0.407] | 0.332 [0.250, 0.402] | 0.339 [0.256, 0.413] | 0.323 [0.244, 0.392] | 0.347 [0.266, 0.420] |
| factorized s2345, d_F cal.* | 0.336 [0.252, 0.409] | 0.247 [0.175, 0.311] | 0.303 [0.226, 0.371] | 0.337 [0.252, 0.411] | 0.341 [0.257, 0.414] | 0.333 [0.248, 0.407] | 0.336 [0.252, 0.410] | 0.333 [0.249, 0.404] | 0.336 [0.251, 0.411] |
| factorized s2345, d_F non cal. | 0.369 [0.285, 0.444] | 0.307 [0.222, 0.385] | 0.347 [0.265, 0.421] | 0.370 [0.285, 0.445] | 0.373 [0.289, 0.448] | 0.367 [0.283, 0.443] | 0.369 [0.285, 0.445] | 0.363 [0.282, 0.437] | 0.369 [0.286, 0.444] |
| **factorized s2345, d_P** | 0.372 [0.291, 0.444] | 0.316 [0.236, 0.389] | 0.352 [0.273, 0.424] | 0.374 [0.292, 0.447] | 0.376 [0.294, 0.450] | 0.371 [0.290, 0.443] | 0.373 [0.291, 0.446] | 0.365 [0.286, 0.437] | 0.374 [0.293, 0.446] |
| ctrlfr s1234 | 0.326 [0.243, 0.400] | 0.291 [0.213, 0.359] | 0.314 [0.234, 0.388] | 0.336 [0.252, 0.412] | 0.330 [0.245, 0.405] | 0.334 [0.250, 0.409] | 0.337 [0.254, 0.415] | 0.317 [0.236, 0.389] | 0.359 [0.276, 0.436] |
| ctrlfr s2345 | 0.305 [0.224, 0.377] | 0.274 [0.191, 0.343] | 0.294 [0.215, 0.364] | 0.311 [0.229, 0.383] | 0.312 [0.231, 0.383] | 0.313 [0.231, 0.382] | 0.311 [0.228, 0.383] | 0.293 [0.214, 0.368] | 0.333 [0.249, 0.407] |
| C3M e205, d_F cal.* | 0.329 [0.255, 0.398] | 0.297 [0.225, 0.364] | 0.318 [0.246, 0.385] | 0.351 [0.275, 0.422] | 0.345 [0.266, 0.415] | 0.345 [0.265, 0.417] | 0.342 [0.261, 0.417] | 0.308 [0.236, 0.376] | 0.387 [0.303, 0.463] |
| C3M e205, d_F non cal. | 0.371 [0.287, 0.444] | 0.367 [0.286, 0.440] | 0.369 [0.289, 0.444] | 0.387 [0.304, 0.460] | 0.382 [0.297, 0.456] | 0.375 [0.290, 0.450] | 0.381 [0.295, 0.459] | 0.354 [0.272, 0.428] | 0.411 [0.328, 0.487] |
| **C3M e205, d_P** | 0.382 [0.304, 0.452] | 0.372 [0.294, 0.442] | 0.378 [0.304, 0.448] | 0.398 [0.321, 0.469] | 0.393 [0.313, 0.464] | 0.387 [0.307, 0.459] | 0.389 [0.310, 0.463] | 0.357 [0.281, 0.426] | 0.417 [0.339, 0.490] |
| **ICP + Chamfer in mm (per coppia)** | 0.383 [0.298, 0.456] | 0.299 [0.233, 0.363] | 0.351 [0.274, 0.420] | 0.389 [0.305, 0.461] | 0.389 [0.305, 0.463] | 0.376 [0.290, 0.452] | 0.387 [0.301, 0.460] | 0.378 [0.296, 0.450] | 0.393 [0.310, 0.467] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.397 [0.320, 0.467] | 0.356 [0.286, 0.420] | 0.370 [0.301, 0.435] | 0.400 [0.323, 0.471] | 0.403 [0.325, 0.475] | 0.389 [0.308, 0.460] | 0.400 [0.324, 0.471] | 0.396 [0.318, 0.466] | 0.405 [0.329, 0.472] |
| **NICP per coppia in mm** | 0.314 [0.232, 0.391] | 0.323 [0.245, 0.400] | 0.316 [0.237, 0.393] | 0.314 [0.232, 0.391] | 0.317 [0.234, 0.396] | 0.309 [0.224, 0.390] | 0.317 [0.235, 0.394] | 0.315 [0.235, 0.390] | - |
| **NICP per coppia cs** | 0.323 [0.242, 0.401] | 0.322 [0.243, 0.397] | 0.322 [0.242, 0.398] | 0.321 [0.241, 0.400] | 0.325 [0.242, 0.404] | 0.320 [0.236, 0.399] | 0.325 [0.243, 0.403] | 0.327 [0.247, 0.404] | - |
| **NICP su template in mm** | 0.156 [0.061, 0.253] | 0.122 [0.062, 0.181] | 0.137 [0.059, 0.212] | 0.156 [0.061, 0.254] | 0.157 [0.060, 0.254] | 0.160 [0.066, 0.258] | 0.155 [0.060, 0.251] | 0.150 [0.055, 0.248] | 0.161 [0.068, 0.254] |
| **varifold in mm (massa unitaria)** | 0.276 [0.211, 0.337] | 0.250 [0.197, 0.302] | 0.263 [0.204, 0.319] | 0.313 [0.243, 0.377] | 0.297 [0.232, 0.358] | 0.313 [0.243, 0.377] | 0.312 [0.242, 0.376] | 0.333 [0.251, 0.407] | 0.333 [0.267, 0.394] |
| **GNM (visto) preregistrato, coefficienti** | 0.207 [0.127, 0.277] | 0.204 [0.127, 0.276] | 0.206 [0.127, 0.275] | 0.210 [0.127, 0.282] | 0.203 [0.121, 0.273] | 0.211 [0.130, 0.281] | 0.211 [0.130, 0.279] | 0.202 [0.122, 0.273] | 0.207 [0.127, 0.277] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.174 [0.078, 0.265] | 0.176 [0.085, 0.263] | 0.175 [0.079, 0.264] | 0.178 [0.079, 0.270] | 0.172 [0.073, 0.265] | 0.180 [0.085, 0.269] | 0.175 [0.075, 0.266] | 0.166 [0.071, 0.257] | 0.174 [0.077, 0.265] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.228 [0.145, 0.308] | 0.230 [0.151, 0.305] | 0.228 [0.147, 0.305] | 0.233 [0.150, 0.313] | 0.226 [0.144, 0.304] | 0.231 [0.148, 0.312] | 0.234 [0.151, 0.315] | 0.217 [0.134, 0.296] | 0.233 [0.151, 0.311] |
| **GNM (visto) vA, coefficienti*** | 0.211 [0.130, 0.284] | 0.211 [0.132, 0.287] | 0.211 [0.131, 0.284] | 0.215 [0.133, 0.288] | 0.206 [0.125, 0.279] | 0.213 [0.134, 0.285] | 0.215 [0.133, 0.288] | 0.207 [0.126, 0.283] | 0.211 [0.131, 0.284] |
| GNM (visto) vA, mesh d'identita' FR* | 0.179 [0.080, 0.273] | 0.180 [0.087, 0.270] | 0.179 [0.080, 0.272] | 0.182 [0.083, 0.277] | 0.177 [0.078, 0.271] | 0.185 [0.088, 0.278] | 0.180 [0.079, 0.275] | 0.171 [0.076, 0.263] | 0.179 [0.079, 0.273] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.222 [0.138, 0.300] | 0.225 [0.147, 0.302] | 0.223 [0.141, 0.299] | 0.226 [0.142, 0.305] | 0.221 [0.138, 0.300] | 0.224 [0.142, 0.303] | 0.228 [0.145, 0.306] | 0.212 [0.128, 0.290] | 0.227 [0.144, 0.301] |
| **GNM (visto) vB, coefficienti*** | 0.304 [0.202, 0.390] | 0.296 [0.197, 0.383] | 0.301 [0.201, 0.386] | 0.302 [0.201, 0.388] | 0.304 [0.202, 0.392] | 0.305 [0.204, 0.393] | 0.303 [0.201, 0.390] | 0.307 [0.205, 0.393] | 0.299 [0.200, 0.384] |
| GNM (visto) vB, mesh d'identita' FR* | 0.275 [0.171, 0.366] | 0.276 [0.179, 0.363] | 0.275 [0.175, 0.364] | 0.276 [0.171, 0.366] | 0.272 [0.170, 0.363] | 0.279 [0.175, 0.370] | 0.274 [0.170, 0.365] | 0.273 [0.172, 0.363] | 0.275 [0.173, 0.366] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.399 [0.315, 0.473] | 0.402 [0.321, 0.473] | 0.400 [0.317, 0.473] | 0.400 [0.316, 0.475] | 0.398 [0.313, 0.472] | 0.400 [0.314, 0.475] | 0.399 [0.315, 0.474] | 0.398 [0.313, 0.472] | 0.400 [0.317, 0.474] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.358 [0.259, 0.439] | 0.360 [0.264, 0.439] | 0.358 [0.261, 0.439] | 0.359 [0.259, 0.441] | 0.355 [0.256, 0.438] | 0.361 [0.263, 0.441] | 0.357 [0.259, 0.438] | 0.357 [0.258, 0.439] | 0.358 [0.260, 0.440] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.150 [0.073, 0.223] | 0.152 [0.078, 0.222] | 0.151 [0.076, 0.222] | 0.151 [0.072, 0.225] | 0.152 [0.073, 0.223] | 0.154 [0.078, 0.225] | 0.151 [0.073, 0.224] | 0.145 [0.067, 0.217] | 0.153 [0.076, 0.227] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.172 [0.086, 0.255] | 0.172 [0.090, 0.251] | 0.172 [0.087, 0.253] | 0.172 [0.084, 0.255] | 0.176 [0.090, 0.258] | 0.175 [0.089, 0.256] | 0.170 [0.084, 0.253] | 0.167 [0.078, 0.249] | 0.172 [0.086, 0.255] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.162 [0.078, 0.240] | 0.164 [0.082, 0.239] | 0.163 [0.081, 0.239] | 0.161 [0.079, 0.241] | 0.162 [0.078, 0.240] | 0.167 [0.084, 0.244] | 0.163 [0.081, 0.243] | 0.157 [0.073, 0.236] | 0.166 [0.082, 0.245] |
| **FLAME 2023 Open vA, coefficienti*** | 0.156 [0.077, 0.229] | 0.157 [0.085, 0.228] | 0.156 [0.079, 0.229] | 0.156 [0.076, 0.231] | 0.157 [0.078, 0.230] | 0.159 [0.084, 0.231] | 0.156 [0.076, 0.231] | 0.150 [0.070, 0.223] | 0.158 [0.079, 0.233] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.174 [0.081, 0.260] | 0.173 [0.088, 0.256] | 0.173 [0.083, 0.258] | 0.173 [0.082, 0.260] | 0.178 [0.087, 0.264] | 0.177 [0.086, 0.262] | 0.172 [0.079, 0.259] | 0.169 [0.076, 0.257] | 0.174 [0.081, 0.259] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.171 [0.087, 0.249] | 0.173 [0.091, 0.250] | 0.172 [0.090, 0.248] | 0.170 [0.089, 0.249] | 0.171 [0.087, 0.246] | 0.176 [0.094, 0.254] | 0.173 [0.091, 0.251] | 0.166 [0.085, 0.244] | 0.175 [0.091, 0.253] |
| **FLAME 2023 Open vB, coefficienti*** | 0.268 [0.159, 0.360] | 0.238 [0.132, 0.335] | 0.258 [0.150, 0.350] | 0.265 [0.154, 0.360] | 0.269 [0.158, 0.364] | 0.275 [0.165, 0.366] | 0.266 [0.157, 0.358] | 0.265 [0.155, 0.360] | 0.256 [0.150, 0.347] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.274 [0.173, 0.366] | 0.265 [0.169, 0.354] | 0.271 [0.172, 0.360] | 0.274 [0.175, 0.365] | 0.277 [0.176, 0.366] | 0.276 [0.173, 0.365] | 0.272 [0.170, 0.363] | 0.273 [0.170, 0.367] | 0.271 [0.170, 0.363] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.382 [0.297, 0.459] | 0.371 [0.284, 0.452] | 0.378 [0.293, 0.457] | 0.380 [0.295, 0.459] | 0.382 [0.297, 0.459] | 0.384 [0.298, 0.462] | 0.382 [0.297, 0.461] | 0.382 [0.297, 0.461] | 0.380 [0.295, 0.460] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.345 [0.250, 0.430] | 0.333 [0.238, 0.420] | 0.341 [0.246, 0.426] | 0.344 [0.246, 0.432] | 0.346 [0.251, 0.429] | 0.347 [0.251, 0.428] | 0.343 [0.247, 0.430] | 0.344 [0.246, 0.433] | 0.342 [0.246, 0.427] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.166 [0.122, 0.213] | 0.196 [0.145, 0.249] | 0.175 [0.128, 0.225] | 0.187 [0.138, 0.237] | 0.189 [0.141, 0.239] | 0.196 [0.148, 0.250] | 0.186 [0.137, 0.236] | 0.131 [0.065, 0.204] | 0.193 [0.144, 0.245] |
| **ShapeDNA k=50** | 0.070 [0.030, 0.105] | 0.062 [0.020, 0.102] | 0.066 [0.027, 0.100] | 0.089 [0.031, 0.142] | 0.093 [0.035, 0.148] | 0.091 [0.034, 0.143] | 0.091 [0.033, 0.147] | 0.093 [0.028, 0.154] | 0.137 [0.057, 0.211] |
| **HKS globale** | 0.037 [0.009, 0.065] | 0.037 [-0.007, 0.083] | 0.036 [0.005, 0.068] | 0.046 [0.005, 0.087] | 0.046 [0.005, 0.088] | 0.047 [0.005, 0.087] | 0.046 [0.005, 0.088] | 0.065 [-0.001, 0.128] | 0.074 [0.012, 0.137] |
| **WKS globale** | 0.049 [0.022, 0.076] | 0.059 [0.022, 0.092] | 0.048 [0.022, 0.074] | 0.062 [0.019, 0.103] | 0.058 [0.019, 0.095] | 0.057 [0.018, 0.094] | 0.051 [0.016, 0.086] | 0.111 [0.047, 0.175] | 0.068 [-0.001, 0.133] |
| **taglia stimata dalla mesh (non oracolo)** | 0.090 [-0.001, 0.181] | 0.015 [0.001, 0.030] | 0.054 [0.003, 0.106] | 0.090 [-0.004, 0.184] | 0.088 [-0.000, 0.176] | 0.090 [-0.005, 0.184] | 0.090 [-0.004, 0.184] | 0.091 [0.003, 0.179] | 0.090 [-0.007, 0.186] |
| **taglia oracolo (S di FR)** | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.068 [-0.139, +0.011] | -0.034 [-0.098, +0.035] | -0.054 [-0.119, +0.017] | -0.072 [-0.143, +0.007] | -0.071 [-0.142, +0.006] | -0.065 [-0.137, +0.013] | -0.066 [-0.140, +0.014] | -0.068 [-0.139, +0.009] | -0.071 [-0.142, +0.007] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.033 [-0.114, +0.053] | -0.051 [-0.126, +0.031] | -0.029 [-0.103, +0.054] | -0.035 [-0.120, +0.055] | -0.037 [-0.118, +0.050] | -0.030 [-0.113, +0.060] | -0.031 [-0.113, +0.058] | -0.037 [-0.116, +0.046] | -0.033 [-0.115, +0.055] |
| NICP per coppia in mm | +0.063 [-0.027, +0.152] | -0.012 [-0.092, +0.065] | +0.036 [-0.045, +0.119] | +0.065 [-0.025, +0.155] | +0.063 [-0.028, +0.153] | +0.065 [-0.031, +0.157] | +0.066 [-0.026, +0.157] | +0.055 [-0.032, +0.140] | - |
| NICP per coppia cs | +0.054 [-0.043, +0.147] | -0.008 [-0.089, +0.077] | +0.032 [-0.055, +0.120] | +0.057 [-0.041, +0.151] | +0.056 [-0.043, +0.152] | +0.055 [-0.045, +0.150] | +0.059 [-0.039, +0.154] | +0.044 [-0.050, +0.132] | - |
| NICP su template in mm | +0.117 [+0.013, +0.219] | +0.121 [+0.040, +0.200] | +0.124 [+0.036, +0.211] | +0.119 [+0.015, +0.223] | +0.118 [+0.015, +0.222] | +0.109 [+0.006, +0.210] | +0.122 [+0.019, +0.225] | +0.117 [+0.015, +0.219] | +0.120 [+0.019, +0.220] |
| varifold in mm (massa unitaria) | +0.056 [-0.009, +0.128] | +0.028 [-0.033, +0.094] | +0.049 [-0.013, +0.117] | +0.020 [-0.046, +0.094] | +0.037 [-0.026, +0.108] | +0.014 [-0.050, +0.087] | +0.023 [-0.042, +0.098] | -0.002 [-0.078, +0.077] | +0.003 [-0.060, +0.074] |
| GNM (visto) preregistrato, coefficienti | +0.100 [-0.009, +0.201] | +0.043 [-0.057, +0.150] | +0.079 [-0.024, +0.179] | +0.099 [-0.010, +0.199] | +0.108 [+0.000, +0.208] | +0.093 [-0.017, +0.195] | +0.102 [-0.010, +0.206] | +0.100 [-0.001, +0.198] | +0.108 [+0.003, +0.213] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.109 [+0.007, +0.214] | +0.049 [-0.045, +0.144] | +0.087 [-0.011, +0.185] | +0.108 [+0.003, +0.210] | +0.114 [+0.013, +0.218] | +0.100 [+0.001, +0.204] | +0.114 [+0.009, +0.220] | +0.111 [+0.008, +0.213] | +0.116 [+0.014, +0.219] |
| GNM (visto) vA, coefficienti | +0.110 [+0.005, +0.208] | +0.049 [-0.058, +0.148] | +0.088 [-0.014, +0.187] | +0.108 [+0.003, +0.207] | +0.118 [+0.012, +0.219] | +0.104 [-0.002, +0.203] | +0.112 [+0.004, +0.213] | +0.109 [+0.009, +0.212] | +0.117 [+0.013, +0.220] |
| GNM (visto) vA, mesh d'identita' FR | +0.104 [+0.002, +0.210] | +0.045 [-0.049, +0.143] | +0.082 [-0.016, +0.184] | +0.103 [-0.001, +0.208] | +0.109 [+0.009, +0.214] | +0.095 [-0.007, +0.203] | +0.109 [+0.003, +0.215] | +0.105 [+0.003, +0.209] | +0.112 [+0.009, +0.215] |
| GNM (visto) vB, coefficienti | +0.041 [-0.060, +0.152] | -0.009 [-0.101, +0.089] | +0.022 [-0.075, +0.128] | +0.044 [-0.058, +0.155] | +0.044 [-0.057, +0.156] | +0.036 [-0.065, +0.146] | +0.046 [-0.059, +0.156] | +0.033 [-0.067, +0.146] | +0.053 [-0.048, +0.162] |
| GNM (visto) vB, mesh d'identita' FR | +0.020 [-0.070, +0.118] | -0.040 [-0.117, +0.046] | -0.002 [-0.089, +0.092] | +0.021 [-0.068, +0.122] | +0.025 [-0.066, +0.123] | +0.013 [-0.077, +0.111] | +0.026 [-0.065, +0.125] | +0.015 [-0.070, +0.111] | +0.027 [-0.062, +0.124] |
| GNM (visto) vB, composizione S_B + k d_P | -0.041 [-0.126, +0.052] | -0.103 [-0.179, -0.022] | -0.064 [-0.145, +0.023] | -0.040 [-0.127, +0.053] | -0.037 [-0.122, +0.056] | -0.047 [-0.132, +0.046] | -0.035 [-0.121, +0.059] | -0.047 [-0.128, +0.047] | -0.035 [-0.115, +0.057] |
| FLAME 2023 Open preregistrato, coefficienti | +0.170 [+0.067, +0.274] | +0.110 [+0.011, +0.212] | +0.148 [+0.051, +0.251] | +0.173 [+0.067, +0.279] | +0.169 [+0.065, +0.276] | +0.164 [+0.064, +0.268] | +0.172 [+0.067, +0.280] | +0.171 [+0.073, +0.271] | +0.175 [+0.073, +0.283] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.127 [+0.017, +0.232] | +0.069 [-0.033, +0.167] | +0.105 [-0.002, +0.207] | +0.130 [+0.018, +0.237] | +0.123 [+0.011, +0.228] | +0.121 [+0.013, +0.223] | +0.131 [+0.017, +0.239] | +0.128 [+0.022, +0.230] | +0.133 [+0.020, +0.242] |
| FLAME 2023 Open vA, coefficienti | +0.162 [+0.057, +0.267] | +0.102 [+0.005, +0.202] | +0.140 [+0.044, +0.240] | +0.166 [+0.060, +0.272] | +0.161 [+0.056, +0.269] | +0.157 [+0.057, +0.261] | +0.164 [+0.058, +0.273] | +0.163 [+0.063, +0.264] | +0.167 [+0.064, +0.276] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.120 [+0.012, +0.221] | +0.064 [-0.032, +0.163] | +0.099 [-0.006, +0.199] | +0.124 [+0.016, +0.228] | +0.116 [+0.010, +0.218] | +0.114 [+0.009, +0.215] | +0.126 [+0.012, +0.234] | +0.122 [+0.017, +0.227] | +0.128 [+0.019, +0.233] |
| FLAME 2023 Open vB, coefficienti | +0.092 [-0.014, +0.204] | +0.060 [-0.042, +0.164] | +0.079 [-0.022, +0.185] | +0.096 [-0.013, +0.209] | +0.093 [-0.015, +0.206] | +0.082 [-0.026, +0.192] | +0.097 [-0.011, +0.212] | +0.090 [-0.017, +0.204] | +0.110 [+0.006, +0.222] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.026 [-0.070, +0.133] | -0.024 [-0.111, +0.075] | +0.007 [-0.086, +0.110] | +0.029 [-0.070, +0.135] | +0.025 [-0.072, +0.130] | +0.021 [-0.072, +0.125] | +0.032 [-0.067, +0.138] | +0.023 [-0.074, +0.131] | +0.036 [-0.066, +0.144] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.026 [-0.123, +0.079] | -0.074 [-0.165, +0.022] | -0.044 [-0.136, +0.054] | -0.022 [-0.122, +0.084] | -0.026 [-0.123, +0.079] | -0.031 [-0.125, +0.074] | -0.022 [-0.118, +0.085] | -0.029 [-0.122, +0.079] | -0.016 [-0.111, +0.088] |
| ArcFace-render, normal map 3 viste (G0) | +0.187 [+0.095, +0.272] | +0.101 [+0.009, +0.188] | +0.157 [+0.068, +0.238] | +0.170 [+0.078, +0.255] | +0.169 [+0.077, +0.253] | +0.157 [+0.065, +0.243] | +0.173 [+0.079, +0.259] | +0.212 [+0.097, +0.320] | +0.171 [+0.075, +0.257] |
| ShapeDNA k=50 | +0.267 [+0.182, +0.348] | +0.224 [+0.138, +0.318] | +0.252 [+0.169, +0.334] | +0.248 [+0.157, +0.339] | +0.244 [+0.155, +0.330] | +0.241 [+0.151, +0.331] | +0.247 [+0.157, +0.338] | +0.242 [+0.142, +0.343] | +0.206 [+0.102, +0.313] |
| HKS globale | +0.305 [+0.222, +0.387] | +0.245 [+0.155, +0.335] | +0.284 [+0.201, +0.368] | +0.298 [+0.206, +0.388] | +0.299 [+0.208, +0.390] | +0.292 [+0.202, +0.380] | +0.300 [+0.207, +0.392] | +0.279 [+0.185, +0.372] | +0.276 [+0.165, +0.380] |
| WKS globale | +0.290 [+0.210, +0.370] | +0.227 [+0.148, +0.311] | +0.271 [+0.194, +0.346] | +0.279 [+0.198, +0.368] | +0.280 [+0.200, +0.358] | +0.280 [+0.199, +0.361] | +0.288 [+0.208, +0.368] | +0.225 [+0.122, +0.329] | +0.273 [+0.180, +0.371] |
| taglia stimata dalla mesh (non oracolo) | +0.197 [+0.097, +0.286] | +0.260 [+0.183, +0.342] | +0.230 [+0.150, +0.304] | +0.198 [+0.096, +0.289] | +0.203 [+0.104, +0.290] | +0.193 [+0.094, +0.282] | +0.201 [+0.097, +0.293] | +0.191 [+0.096, +0.280] | +0.205 [+0.103, +0.297] |
| taglia oracolo (S di FR) | +0.132 [+0.005, +0.251] | +0.073 [-0.046, +0.197] | +0.110 [-0.015, +0.230] | +0.134 [+0.007, +0.254] | +0.134 [+0.009, +0.254] | +0.128 [+0.002, +0.249] | +0.136 [+0.007, +0.259] | +0.127 [-0.003, +0.246] | +0.139 [+0.015, +0.260] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.072 [-0.151, +0.013] | -0.011 [-0.086, +0.066] | -0.047 [-0.122, +0.033] | -0.074 [-0.153, +0.010] | -0.074 [-0.152, +0.009] | -0.067 [-0.147, +0.020] | -0.070 [-0.150, +0.014] | -0.076 [-0.156, +0.006] | -0.070 [-0.148, +0.014] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.037 [-0.118, +0.051] | -0.028 [-0.104, +0.050] | -0.022 [-0.097, +0.060] | -0.037 [-0.122, +0.054] | -0.040 [-0.120, +0.048] | -0.031 [-0.117, +0.058] | -0.034 [-0.117, +0.056] | -0.045 [-0.124, +0.039] | -0.033 [-0.116, +0.055] |
| NICP per coppia in mm | +0.059 [-0.025, +0.140] | +0.011 [-0.063, +0.090] | +0.043 [-0.035, +0.120] | +0.063 [-0.022, +0.146] | +0.060 [-0.022, +0.144] | +0.063 [-0.027, +0.147] | +0.063 [-0.021, +0.146] | +0.046 [-0.031, +0.121] | - |
| NICP per coppia cs | +0.051 [-0.034, +0.141] | +0.016 [-0.064, +0.101] | +0.039 [-0.040, +0.122] | +0.055 [-0.031, +0.144] | +0.053 [-0.033, +0.143] | +0.053 [-0.036, +0.144] | +0.056 [-0.029, +0.147] | +0.036 [-0.047, +0.121] | - |
| NICP su template in mm | +0.113 [-0.004, +0.225] | +0.144 [+0.044, +0.241] | +0.131 [+0.024, +0.230] | +0.117 [-0.003, +0.228] | +0.115 [-0.001, +0.229] | +0.107 [-0.014, +0.220] | +0.119 [-0.000, +0.231] | +0.109 [-0.008, +0.221] | +0.121 [+0.006, +0.231] |
| varifold in mm (massa unitaria) | +0.052 [-0.015, +0.121] | +0.051 [-0.014, +0.121] | +0.056 [-0.009, +0.123] | +0.018 [-0.053, +0.088] | +0.033 [-0.035, +0.103] | +0.012 [-0.057, +0.083] | +0.020 [-0.051, +0.092] | -0.011 [-0.085, +0.064] | +0.003 [-0.065, +0.073] |
| GNM (visto) preregistrato, coefficienti | +0.097 [-0.010, +0.198] | +0.067 [-0.043, +0.176] | +0.086 [-0.019, +0.190] | +0.097 [-0.012, +0.201] | +0.104 [-0.002, +0.207] | +0.091 [-0.018, +0.193] | +0.099 [-0.010, +0.204] | +0.092 [-0.012, +0.191] | +0.109 [+0.004, +0.212] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.105 [-0.004, +0.214] | +0.072 [-0.034, +0.176] | +0.094 [-0.009, +0.199] | +0.106 [-0.004, +0.213] | +0.110 [+0.002, +0.219] | +0.098 [-0.011, +0.208] | +0.111 [+0.000, +0.222] | +0.102 [-0.009, +0.210] | +0.117 [+0.012, +0.225] |
| GNM (visto) vA, coefficienti | +0.107 [-0.006, +0.206] | +0.072 [-0.036, +0.176] | +0.095 [-0.014, +0.195] | +0.106 [-0.006, +0.209] | +0.115 [+0.005, +0.216] | +0.103 [-0.011, +0.201] | +0.108 [-0.006, +0.212] | +0.101 [-0.002, +0.199] | +0.118 [+0.008, +0.221] |
| GNM (visto) vA, mesh d'identita' FR | +0.100 [-0.007, +0.211] | +0.068 [-0.039, +0.174] | +0.089 [-0.014, +0.196] | +0.101 [-0.008, +0.210] | +0.105 [-0.006, +0.216] | +0.093 [-0.014, +0.205] | +0.106 [-0.005, +0.218] | +0.097 [-0.012, +0.207] | +0.113 [+0.006, +0.222] |
| GNM (visto) vB, coefficienti | +0.037 [-0.054, +0.134] | +0.014 [-0.079, +0.106] | +0.029 [-0.060, +0.123] | +0.042 [-0.049, +0.139] | +0.041 [-0.050, +0.138] | +0.035 [-0.057, +0.129] | +0.043 [-0.050, +0.139] | +0.025 [-0.064, +0.117] | +0.054 [-0.035, +0.149] |
| GNM (visto) vB, mesh d'identita' FR | +0.016 [-0.079, +0.116] | -0.017 [-0.101, +0.073] | +0.005 [-0.083, +0.096] | +0.019 [-0.076, +0.117] | +0.022 [-0.073, +0.120] | +0.011 [-0.085, +0.109] | +0.023 [-0.074, +0.124] | +0.007 [-0.085, +0.109] | +0.028 [-0.063, +0.124] |
| GNM (visto) vB, composizione S_B + k d_P | -0.045 [-0.130, +0.045] | -0.079 [-0.159, +0.002] | -0.057 [-0.138, +0.028] | -0.042 [-0.129, +0.048] | -0.040 [-0.127, +0.050] | -0.049 [-0.136, +0.041] | -0.039 [-0.126, +0.051] | -0.055 [-0.140, +0.036] | -0.034 [-0.118, +0.052] |
| FLAME 2023 Open preregistrato, coefficienti | +0.166 [+0.060, +0.266] | +0.133 [+0.027, +0.233] | +0.154 [+0.052, +0.252] | +0.171 [+0.064, +0.273] | +0.165 [+0.057, +0.266] | +0.162 [+0.058, +0.261] | +0.169 [+0.061, +0.271] | +0.163 [+0.059, +0.262] | +0.176 [+0.071, +0.277] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.123 [+0.012, +0.228] | +0.092 [-0.011, +0.196] | +0.112 [+0.009, +0.217] | +0.128 [+0.016, +0.234] | +0.120 [+0.008, +0.227] | +0.119 [+0.012, +0.223] | +0.128 [+0.015, +0.239] | +0.120 [+0.009, +0.221] | +0.134 [+0.026, +0.240] |
| FLAME 2023 Open vA, coefficienti | +0.158 [+0.053, +0.257] | +0.125 [+0.022, +0.224] | +0.147 [+0.044, +0.244] | +0.164 [+0.057, +0.264] | +0.157 [+0.049, +0.258] | +0.155 [+0.049, +0.252] | +0.161 [+0.052, +0.263] | +0.155 [+0.052, +0.252] | +0.168 [+0.063, +0.270] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.117 [+0.007, +0.227] | +0.087 [-0.014, +0.189] | +0.106 [+0.002, +0.209] | +0.122 [+0.012, +0.231] | +0.113 [+0.005, +0.222] | +0.113 [+0.003, +0.218] | +0.122 [+0.009, +0.234] | +0.114 [+0.003, +0.222] | +0.129 [+0.020, +0.237] |
| FLAME 2023 Open vB, coefficienti | +0.088 [-0.012, +0.185] | +0.083 [-0.017, +0.181] | +0.086 [-0.011, +0.182] | +0.094 [-0.007, +0.193] | +0.090 [-0.009, +0.188] | +0.080 [-0.016, +0.173] | +0.094 [-0.007, +0.193] | +0.082 [-0.017, +0.181] | +0.111 [+0.013, +0.208] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.022 [-0.076, +0.121] | -0.000 [-0.092, +0.095] | +0.014 [-0.078, +0.110] | +0.027 [-0.070, +0.128] | +0.022 [-0.076, +0.121] | +0.019 [-0.075, +0.117] | +0.029 [-0.070, +0.129] | +0.015 [-0.083, +0.117] | +0.036 [-0.063, +0.136] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.030 [-0.121, +0.067] | -0.051 [-0.139, +0.038] | -0.037 [-0.128, +0.053] | -0.024 [-0.118, +0.073] | -0.029 [-0.121, +0.065] | -0.033 [-0.122, +0.062] | -0.025 [-0.117, +0.072] | -0.037 [-0.129, +0.061] | -0.015 [-0.110, +0.078] |
| ArcFace-render, normal map 3 viste (G0) | +0.184 [+0.093, +0.269] | +0.125 [+0.025, +0.218] | +0.164 [+0.074, +0.248] | +0.168 [+0.071, +0.256] | +0.166 [+0.071, +0.252] | +0.155 [+0.061, +0.242] | +0.170 [+0.076, +0.258] | +0.204 [+0.089, +0.306] | +0.171 [+0.079, +0.259] |
| ShapeDNA k=50 | +0.263 [+0.182, +0.341] | +0.247 [+0.161, +0.338] | +0.259 [+0.179, +0.338] | +0.246 [+0.156, +0.336] | +0.241 [+0.153, +0.328] | +0.239 [+0.149, +0.326] | +0.244 [+0.153, +0.334] | +0.234 [+0.131, +0.333] | +0.207 [+0.106, +0.309] |
| HKS globale | +0.301 [+0.217, +0.378] | +0.268 [+0.177, +0.363] | +0.291 [+0.209, +0.371] | +0.296 [+0.207, +0.380] | +0.296 [+0.208, +0.379] | +0.290 [+0.203, +0.373] | +0.297 [+0.207, +0.380] | +0.271 [+0.178, +0.358] | +0.277 [+0.174, +0.376] |
| WKS globale | +0.286 [+0.205, +0.366] | +0.250 [+0.164, +0.336] | +0.277 [+0.197, +0.357] | +0.277 [+0.186, +0.363] | +0.277 [+0.189, +0.353] | +0.278 [+0.192, +0.361] | +0.285 [+0.196, +0.366] | +0.217 [+0.124, +0.314] | +0.273 [+0.175, +0.372] |
| taglia stimata dalla mesh (non oracolo) | +0.193 [+0.075, +0.297] | +0.283 [+0.196, +0.375] | +0.237 [+0.140, +0.323] | +0.196 [+0.076, +0.301] | +0.200 [+0.087, +0.300] | +0.191 [+0.072, +0.296] | +0.198 [+0.076, +0.302] | +0.183 [+0.070, +0.285] | +0.206 [+0.088, +0.315] |
| taglia oracolo (S di FR) | +0.128 [-0.015, +0.260] | +0.097 [-0.040, +0.230] | +0.117 [-0.024, +0.252] | +0.132 [-0.013, +0.266] | +0.131 [-0.012, +0.262] | +0.126 [-0.018, +0.257] | +0.133 [-0.010, +0.270] | +0.119 [-0.027, +0.249] | +0.140 [+0.000, +0.272] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.049 [-0.127, +0.038] | +0.005 [-0.058, +0.079] | -0.028 [-0.100, +0.053] | -0.051 [-0.128, +0.038] | -0.052 [-0.128, +0.035] | -0.044 [-0.121, +0.043] | -0.047 [-0.125, +0.042] | -0.055 [-0.132, +0.031] | -0.046 [-0.123, +0.040] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.063 [-0.140, +0.017] | -0.052 [-0.123, +0.025] | -0.047 [-0.117, +0.029] | -0.063 [-0.140, +0.020] | -0.067 [-0.143, +0.014] | -0.058 [-0.133, +0.025] | -0.061 [-0.138, +0.022] | -0.073 [-0.149, +0.007] | -0.058 [-0.135, +0.023] |
| NICP per coppia in mm | +0.019 [-0.063, +0.103] | -0.018 [-0.094, +0.059] | +0.006 [-0.070, +0.087] | +0.024 [-0.061, +0.108] | +0.020 [-0.063, +0.104] | +0.022 [-0.063, +0.113] | +0.023 [-0.061, +0.106] | +0.008 [-0.070, +0.087] | - |
| NICP per coppia cs | +0.010 [-0.068, +0.095] | -0.017 [-0.093, +0.062] | +0.001 [-0.075, +0.084] | +0.016 [-0.063, +0.098] | +0.012 [-0.067, +0.095] | +0.012 [-0.070, +0.096] | +0.015 [-0.067, +0.099] | -0.004 [-0.078, +0.074] | - |
| NICP su template in mm | +0.178 [+0.061, +0.281] | +0.183 [+0.088, +0.267] | +0.186 [+0.085, +0.273] | +0.181 [+0.064, +0.284] | +0.180 [+0.063, +0.283] | +0.172 [+0.054, +0.276] | +0.185 [+0.067, +0.288] | +0.173 [+0.060, +0.277] | +0.186 [+0.071, +0.286] |
| varifold in mm (massa unitaria) | +0.057 [-0.006, +0.125] | +0.054 [-0.007, +0.121] | +0.060 [+0.002, +0.123] | +0.025 [-0.039, +0.094] | +0.040 [-0.023, +0.108] | +0.019 [-0.044, +0.086] | +0.027 [-0.037, +0.096] | -0.010 [-0.083, +0.067] | +0.014 [-0.050, +0.080] |
| GNM (visto) preregistrato, coefficienti | +0.126 [+0.028, +0.222] | +0.100 [-0.000, +0.201] | +0.117 [+0.022, +0.211] | +0.128 [+0.027, +0.225] | +0.134 [+0.036, +0.230] | +0.121 [+0.023, +0.219] | +0.128 [+0.028, +0.227] | +0.120 [+0.025, +0.216] | +0.140 [+0.044, +0.237] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.106 [+0.006, +0.206] | +0.074 [-0.020, +0.173] | +0.095 [-0.001, +0.192] | +0.105 [+0.006, +0.206] | +0.111 [+0.011, +0.211] | +0.101 [+0.002, +0.204] | +0.106 [+0.002, +0.209] | +0.106 [+0.008, +0.203] | +0.114 [+0.016, +0.215] |
| GNM (visto) vA, coefficienti | +0.123 [+0.018, +0.219] | +0.093 [-0.006, +0.194] | +0.112 [+0.012, +0.207] | +0.123 [+0.017, +0.220] | +0.131 [+0.026, +0.226] | +0.119 [+0.014, +0.217] | +0.124 [+0.020, +0.223] | +0.115 [+0.012, +0.212] | +0.136 [+0.036, +0.233] |
| GNM (visto) vA, mesh d'identita' SR | +0.112 [+0.014, +0.217] | +0.079 [-0.015, +0.182] | +0.100 [+0.008, +0.201] | +0.112 [+0.013, +0.219] | +0.116 [+0.018, +0.220] | +0.108 [+0.009, +0.214] | +0.112 [+0.013, +0.218] | +0.110 [+0.016, +0.211] | +0.120 [+0.024, +0.226] |
| GNM (visto) vB, coefficienti | +0.030 [-0.059, +0.130] | +0.008 [-0.083, +0.103] | +0.022 [-0.065, +0.118] | +0.036 [-0.054, +0.138] | +0.033 [-0.056, +0.133] | +0.026 [-0.061, +0.126] | +0.036 [-0.054, +0.137] | +0.016 [-0.074, +0.116] | +0.048 [-0.041, +0.146] |
| GNM (visto) vB, mesh d'identita' SR | -0.065 [-0.142, +0.012] | -0.098 [-0.172, -0.020] | -0.077 [-0.152, +0.000] | -0.062 [-0.141, +0.015] | -0.061 [-0.137, +0.016] | -0.068 [-0.146, +0.011] | -0.060 [-0.139, +0.018] | -0.075 [-0.149, +0.002] | -0.053 [-0.129, +0.024] |
| FLAME 2023 Open preregistrato, coefficienti | +0.183 [+0.077, +0.284] | +0.153 [+0.052, +0.254] | +0.172 [+0.068, +0.270] | +0.187 [+0.080, +0.291] | +0.185 [+0.076, +0.287] | +0.178 [+0.073, +0.277] | +0.189 [+0.082, +0.292] | +0.178 [+0.072, +0.278] | +0.194 [+0.088, +0.296] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.171 [+0.071, +0.279] | +0.141 [+0.040, +0.253] | +0.160 [+0.060, +0.267] | +0.176 [+0.075, +0.285] | +0.175 [+0.071, +0.285] | +0.165 [+0.065, +0.273] | +0.176 [+0.071, +0.286] | +0.166 [+0.066, +0.272] | +0.181 [+0.079, +0.291] |
| FLAME 2023 Open vA, coefficienti | +0.178 [+0.071, +0.276] | +0.147 [+0.048, +0.246] | +0.167 [+0.063, +0.261] | +0.182 [+0.073, +0.281] | +0.180 [+0.069, +0.279] | +0.173 [+0.068, +0.270] | +0.183 [+0.075, +0.285] | +0.173 [+0.067, +0.272] | +0.189 [+0.082, +0.288] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.162 [+0.063, +0.267] | +0.131 [+0.032, +0.235] | +0.151 [+0.055, +0.255] | +0.168 [+0.067, +0.270] | +0.166 [+0.066, +0.273] | +0.156 [+0.058, +0.259] | +0.167 [+0.066, +0.271] | +0.156 [+0.059, +0.259] | +0.172 [+0.071, +0.278] |
| FLAME 2023 Open vB, coefficienti | +0.066 [-0.032, +0.167] | +0.066 [-0.032, +0.168] | +0.065 [-0.030, +0.164] | +0.072 [-0.027, +0.175] | +0.068 [-0.028, +0.170] | +0.056 [-0.039, +0.155] | +0.073 [-0.028, +0.177] | +0.058 [-0.042, +0.163] | +0.091 [-0.005, +0.191] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.049 [-0.129, +0.037] | -0.066 [-0.150, +0.025] | -0.055 [-0.137, +0.028] | -0.043 [-0.124, +0.045] | -0.045 [-0.125, +0.042] | -0.052 [-0.134, +0.037] | -0.043 [-0.125, +0.043] | -0.059 [-0.142, +0.027] | -0.033 [-0.115, +0.052] |
| ArcFace-render, normal map 3 viste (G0) | +0.167 [+0.074, +0.252] | +0.108 [+0.009, +0.205] | +0.148 [+0.056, +0.236] | +0.151 [+0.057, +0.241] | +0.148 [+0.054, +0.235] | +0.135 [+0.042, +0.221] | +0.153 [+0.060, +0.243] | +0.191 [+0.074, +0.299] | +0.155 [+0.059, +0.245] |
| ShapeDNA k=50 | +0.264 [+0.183, +0.341] | +0.242 [+0.153, +0.333] | +0.257 [+0.175, +0.334] | +0.248 [+0.156, +0.337] | +0.243 [+0.154, +0.331] | +0.241 [+0.149, +0.325] | +0.248 [+0.153, +0.338] | +0.229 [+0.131, +0.324] | +0.210 [+0.108, +0.311] |
| HKS globale | +0.296 [+0.212, +0.370] | +0.268 [+0.175, +0.356] | +0.287 [+0.204, +0.364] | +0.291 [+0.201, +0.375] | +0.291 [+0.200, +0.374] | +0.285 [+0.196, +0.368] | +0.293 [+0.202, +0.377] | +0.257 [+0.167, +0.353] | +0.273 [+0.172, +0.375] |
| WKS globale | +0.285 [+0.204, +0.357] | +0.245 [+0.161, +0.325] | +0.275 [+0.197, +0.348] | +0.276 [+0.192, +0.355] | +0.279 [+0.194, +0.353] | +0.275 [+0.191, +0.349] | +0.288 [+0.204, +0.363] | +0.211 [+0.119, +0.304] | +0.279 [+0.181, +0.377] |
| taglia stimata dalla mesh (non oracolo) | +0.244 [+0.134, +0.340] | +0.289 [+0.205, +0.368] | +0.269 [+0.185, +0.343] | +0.247 [+0.137, +0.346] | +0.249 [+0.140, +0.343] | +0.242 [+0.132, +0.338] | +0.249 [+0.140, +0.348] | +0.232 [+0.123, +0.323] | +0.257 [+0.145, +0.358] |
| taglia oracolo (S di FR) | +0.315 [+0.186, +0.427] | +0.286 [+0.167, +0.394] | +0.304 [+0.181, +0.414] | +0.319 [+0.189, +0.433] | +0.318 [+0.189, +0.430] | +0.313 [+0.183, +0.427] | +0.321 [+0.194, +0.434] | +0.304 [+0.176, +0.415] | +0.328 [+0.204, +0.440] |

## FaMoS TEST (descrittivo)

**Non usare per scegliere il setting di Ava-256** ne' per nessuna selezione: 15 persone, 105 righe (blocco scan gallery -> scan), IC larghissimi; nessuna crop, nessuna perturbazione. Cautele: ArcFace-render (ritaglio tarato su pochi render, vedi le cautele sopra), NICP su template con un template costruito qui dalle persone TRAIN (post hoc).

| dominio | GT | insieme | righe | miglior nostro | rho | miglior concorrente | rho | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | FR | nocrop | 105 | ctrlfr s1234 | 0.831 [0.575, 0.949] | GNM (visto) vB, composizione S_B + k d_P | 0.753 [0.393, 0.938] | +0.078 [-0.082, +0.312] |
| famos | SR | nocrop | 105 | C3M e205, d_P | 0.756 [0.512, 0.889] | FLAME 2023 Open vB, mesh d'identita' SR | 0.791 [0.626, 0.889] | -0.035 [-0.235, +0.116] |

Scala dell'allineamento, FaMoS (tutti i passi, principali e secondari):

GT FR:

| dominio | passo | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.138 [-0.125, +0.418] | - | - | - | - | - | - | - | - |
| famos | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.018 [-0.201, +0.248] | - | - | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | -0.054 [-0.232, +0.095] | - | - | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.071 [-0.235, +0.093] | - | - | - | - | - | - | - | - |
| famos | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.123 [-0.312, +0.101] | - | - | - | - | - | - | - | - |
| famos | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.127 [-0.309, +0.049] | - | - | - | - | - | - | - | - |

GT SR:

| dominio | passo | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.063 [-0.283, +0.161] | - | - | - | - | - | - | - | - |
| famos | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.164 [-0.368, +0.026] | - | - | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.155 [-0.048, +0.452] | - | - | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.179 [-0.026, +0.477] | - | - | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.129 [-0.091, +0.417] | - | - | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.117 [-0.084, +0.361] | - | - | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | -0.009 [-0.187, +0.209] | - | - | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.015 [-0.130, +0.213] | - | - | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | -0.035 [-0.207, +0.186] | - | - | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | -0.047 [-0.208, +0.135] | - | - | - | - | - | - | - | - |

GT FR:

| dominio | passo | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.020 [-0.223, +0.265] | - | - | - | - | - | - | - |
| famos | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.032 [-0.319, +0.286] | - | - | - | - | - | - | - |

GT SR:

| dominio | passo | nocrop | crop | all_cross | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | -0.231 [-0.464, -0.041] | - | - | - | - | - | - | - | - |
| famos | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.406 [-0.806, -0.087] | - | - | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.218 [-0.105, +0.613] | - | - | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.242 [-0.092, +0.631] | - | - | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.192 [-0.170, +0.587] | - | - | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.180 [-0.142, +0.547] | - | - | - | - | - | - | - | - |

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
| **NICP per coppia cs** | 0.584 [0.243, 0.818] |
| **NICP su template in mm*** | 0.688 [0.338, 0.891] |
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
| **NICP per coppia cs** | 0.775 [0.540, 0.860] |
| **NICP su template in mm*** | 0.373 [-0.133, 0.731] |
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

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop |
| --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.062 [-0.223, +0.068] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.077 [-0.197, +0.382] |
| NICP per coppia in mm | -0.041 [-0.339, +0.211] |
| NICP per coppia cs | +0.094 [-0.173, +0.394] |
| NICP su template in mm | -0.010 [-0.332, +0.216] |
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

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop |
| --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.123 [-0.312, +0.101] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.015 [-0.233, +0.271] |
| NICP per coppia in mm | -0.103 [-0.329, +0.118] |
| NICP per coppia cs | +0.033 [-0.151, +0.253] |
| NICP su template in mm | -0.071 [-0.358, +0.247] |
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

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop |
| --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.192 [-0.170, +0.587] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.129 [-0.091, +0.417] |
| NICP per coppia in mm | -0.039 [-0.284, +0.221] |
| NICP per coppia cs | -0.035 [-0.207, +0.186] |
| NICP su template in mm | +0.368 [-0.125, +0.939] |
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

## Controlli

```
{
 "views": {
  "hifi3d": {
   "load": {
    "varifold_check_rel_max_vs_shards": 1.2287317409875875e-05,
    "fits_nocrop_vs_saved_max_abs": {
     "gnm_orig_coef": 6.661338147750939e-15,
     "gnm_orig_fr": 9.573360854053803e-10,
     "gnm_orig_sr": 1.9452120539331652e-09,
     "gnm_va_coef": 7.105427357601002e-15,
     "gnm_va_fr": 1.0308147183057947e-09,
     "gnm_va_sr": 1.971354249979962e-09,
     "gnm_vb_coef": 5.329070518200751e-15,
     "gnm_vb_fr": 5.845428585615764e-10,
     "gnm_vb_sr": 7.487273856465748e-10,
     "flame2023_orig_coef": 7.167599846980011e-13,
     "flame2023_orig_fr": 3.564512573994705e-11,
     "flame2023_orig_sr": 2.6642802200549376e-10,
     "flame2023_va_coef": 4.961309141293668e-13,
     "flame2023_va_fr": 2.2332080629183793e-11,
     "flame2023_va_sr": 1.1988965029074983e-10,
     "flame2023_vb_coef": 1.4022116801015727e-13,
     "flame2023_vb_fr": 1.821182094019491e-11,
     "flame2023_vb_sr": 1.9997031808216548e-11
    }
   },
   "t1_control": {
    "n_compared": 82,
    "max_abs_diff_point_ci": 8.673617379884035e-17,
    "rows_t1_mask": 98224,
    "seed": 990708
   },
   "rows": 178200,
   "gt_spread_within_subject_pair": 0.0,
   "mask": {
    "rows_mask": 177230,
    "rows_mask_cross": 147530,
    "rows_mask_same": 29700,
    "rows_valid_gt": 178200,
    "columns_missing_on_same": [
     "mm_nicp_p2tri",
     "cs_nicp_p2tri"
    ],
    "rows_dropped_by_column_cross": {
     "mm_nicp_p2tri": 970,
     "cs_nicp_p2tri": 970
    }
   },
   "spearman_con_taglia": {
    "nocrop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.7094,
      "est_cs": 0.7743
     },
     "factorized_s1234|form": {
      "oracle_size": 0.3726,
      "est_cs": 0.3679
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.081,
      "est_cs": 0.0456
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.6826,
      "est_cs": 0.7763
     },
     "factorized_s2345|form": {
      "oracle_size": 0.3618,
      "est_cs": 0.3778
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0729,
      "est_cs": 0.0523
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.7045,
      "est_cs": 0.725
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.6842,
      "est_cs": 0.714
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.7384,
      "est_cs": 0.7566
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.2395,
      "est_cs": 0.2181
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.081,
      "est_cs": 0.056
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.7303,
      "est_cs": 0.756
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.2179,
      "est_cs": 0.1899
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.0912,
      "est_cs": 0.0602
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.5089,
      "est_cs": 0.5615
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": 0.0392,
      "est_cs": 0.0358
     },
     "mm_nicp_p2tri": {
      "oracle_size": 0.2214,
      "est_cs": 0.1362
     },
     "cs_nicp_p2tri": {
      "oracle_size": 0.0439,
      "est_cs": 0.0214
     },
     "mm_nicp_template": {
      "oracle_size": 0.7682,
      "est_cs": 0.8551
     },
     "cs_nicp_template": {
      "oracle_size": 0.0394,
      "est_cs": 0.0439
     },
     "varifold": {
      "oracle_size": 0.4762,
      "est_cs": 0.5731
     },
     "gnm_orig_coef": {
      "oracle_size": 0.6945,
      "est_cs": 0.7187
     },
     "gnm_orig_fr": {
      "oracle_size": 0.746,
      "est_cs": 0.8239
     },
     "gnm_orig_sr": {
      "oracle_size": 0.3838,
      "est_cs": 0.3593
     },
     "gnm_va_coef": {
      "oracle_size": 0.63,
      "est_cs": 0.6636
     },
     "gnm_va_fr": {
      "oracle_size": 0.7319,
      "est_cs": 0.8123
     },
     "gnm_va_sr": {
      "oracle_size": 0.1269,
      "est_cs": 0.1261
     },
     "gnm_vb_coef": {
      "oracle_size": 0.1693,
      "est_cs": 0.128
     },
     "gnm_vb_fr": {
      "oracle_size": 0.6996,
      "est_cs": 0.7745
     },
     "gnm_vb_sr": {
      "oracle_size": 0.0836,
      "est_cs": 0.0774
     },
     "gnm_vb_comp": {
      "oracle_size": 0.6211,
      "est_cs": 0.675
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.6194,
      "est_cs": 0.5811
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.7617,
      "est_cs": 0.7735
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.1399,
      "est_cs": 0.1222
     },
     "flame2023_va_coef": {
      "oracle_size": 0.6413,
      "est_cs": 0.6152
     },
     "flame2023_va_fr": {
      "oracle_size": 0.7706,
      "est_cs": 0.7936
     },
     "flame2023_va_sr": {
      "oracle_size": 0.1246,
      "est_cs": 0.1148
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.2906,
      "est_cs": 0.2426
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.7158,
      "est_cs": 0.7636
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.142,
      "est_cs": 0.1417
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.638,
      "est_cs": 0.6746
     },
     "arcface_normals": {
      "oracle_size": 0.029,
      "est_cs": 0.0378
     },
     "arcface_shaded": {
      "oracle_size": 0.0466,
      "est_cs": 0.0578
     },
     "shapedna_k50": {
      "oracle_size": 0.0212,
      "est_cs": 0.0127
     },
     "hks": {
      "oracle_size": 0.0067,
      "est_cs": 0.0045
     },
     "wks": {
      "oracle_size": 0.0033,
      "est_cs": -0.0046
     },
     "openshape": {
      "oracle_size": 0.0303,
      "est_cs": 0.0082
     },
     "uni3d": {
      "oracle_size": 0.0374,
      "est_cs": 0.0228
     },
     "est_cs": {
      "oracle_size": 0.8043,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.8043
     }
    },
    "crop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.6081,
      "est_cs": 0.4498
     },
     "factorized_s1234|form": {
      "oracle_size": 0.3159,
      "est_cs": 0.2434
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.0652,
      "est_cs": 0.0514
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.4917,
      "est_cs": 0.5994
     },
     "factorized_s2345|form": {
      "oracle_size": 0.2857,
      "est_cs": 0.3317
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0516,
      "est_cs": 0.0665
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.161,
      "est_cs": 0.8734
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.1971,
      "est_cs": 0.8695
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.4677,
      "est_cs": 0.6522
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.186,
      "est_cs": 0.1706
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0635,
      "est_cs": 0.0236
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.4235,
      "est_cs": 0.685
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.1631,
      "est_cs": 0.1427
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.066,
      "est_cs": 0.0156
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.3047,
      "est_cs": 0.0788
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": 0.0254,
      "est_cs": 0.0519
     },
     "mm_nicp_p2tri": {
      "oracle_size": 0.1872,
      "est_cs": -0.051
     },
     "cs_nicp_p2tri": {
      "oracle_size": 0.0298,
      "est_cs": 0.1613
     },
     "mm_nicp_template": {
      "oracle_size": 0.3799,
      "est_cs": 0.7599
     },
     "cs_nicp_template": {
      "oracle_size": 0.0085,
      "est_cs": 0.0084
     },
     "varifold": {
      "oracle_size": 0.2362,
      "est_cs": 0.8138
     },
     "gnm_orig_coef": {
      "oracle_size": 0.4825,
      "est_cs": 0.4926
     },
     "gnm_orig_fr": {
      "oracle_size": 0.6403,
      "est_cs": 0.4582
     },
     "gnm_orig_sr": {
      "oracle_size": 0.1926,
      "est_cs": 0.072
     },
     "gnm_va_coef": {
      "oracle_size": 0.4701,
      "est_cs": 0.3711
     },
     "gnm_va_fr": {
      "oracle_size": 0.6398,
      "est_cs": 0.4357
     },
     "gnm_va_sr": {
      "oracle_size": 0.0531,
      "est_cs": -0.1314
     },
     "gnm_vb_coef": {
      "oracle_size": 0.1546,
      "est_cs": 0.1051
     },
     "gnm_vb_fr": {
      "oracle_size": 0.6486,
      "est_cs": 0.3827
     },
     "gnm_vb_sr": {
      "oracle_size": 0.0734,
      "est_cs": -0.0573
     },
     "gnm_vb_comp": {
      "oracle_size": 0.581,
      "est_cs": 0.3286
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.4366,
      "est_cs": 0.4146
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.6417,
      "est_cs": 0.4464
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0734,
      "est_cs": -0.0701
     },
     "flame2023_va_coef": {
      "oracle_size": 0.4661,
      "est_cs": 0.4196
     },
     "flame2023_va_fr": {
      "oracle_size": 0.6708,
      "est_cs": 0.4216
     },
     "flame2023_va_sr": {
      "oracle_size": 0.066,
      "est_cs": -0.1422
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.2638,
      "est_cs": 0.2125
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.6639,
      "est_cs": 0.364
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.1332,
      "est_cs": -0.0376
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.6017,
      "est_cs": 0.313
     },
     "arcface_normals": {
      "oracle_size": 0.0304,
      "est_cs": 0.0189
     },
     "arcface_shaded": {
      "oracle_size": 0.0608,
      "est_cs": 0.0217
     },
     "shapedna_k50": {
      "oracle_size": -0.0169,
      "est_cs": 0.0608
     },
     "hks": {
      "oracle_size": -0.0142,
      "est_cs": 0.0804
     },
     "wks": {
      "oracle_size": -0.001,
      "est_cs": 0.0004
     },
     "openshape": {
      "oracle_size": 0.0337,
      "est_cs": 0.0835
     },
     "uni3d": {
      "oracle_size": 0.0499,
      "est_cs": 0.021
     },
     "est_cs": {
      "oracle_size": 0.0522,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.0522
     }
    },
    "all_cross": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.6664,
      "est_cs": 0.6453
     },
     "factorized_s1234|form": {
      "oracle_size": 0.3354,
      "est_cs": 0.3857
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.0721,
      "est_cs": 0.1657
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.6098,
      "est_cs": 0.7
     },
     "factorized_s2345|form": {
      "oracle_size": 0.3185,
      "est_cs": 0.4194
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0644,
      "est_cs": 0.1759
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.4604,
      "est_cs": 0.7966
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.4674,
      "est_cs": 0.788
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.6244,
      "est_cs": 0.7317
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.1865,
      "est_cs": 0.3367
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0634,
      "est_cs": 0.2305
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.6089,
      "est_cs": 0.7395
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.1725,
      "est_cs": 0.3087
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.0718,
      "est_cs": 0.2228
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.4307,
      "est_cs": 0.3343
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": 0.0289,
      "est_cs": 0.2232
     },
     "mm_nicp_p2tri": {
      "oracle_size": 0.2068,
      "est_cs": 0.0187
     },
     "cs_nicp_p2tri": {
      "oracle_size": 0.037,
      "est_cs": 0.1455
     },
     "mm_nicp_template": {
      "oracle_size": 0.6271,
      "est_cs": 0.8108
     },
     "cs_nicp_template": {
      "oracle_size": 0.0194,
      "est_cs": 0.2776
     },
     "varifold": {
      "oracle_size": 0.3558,
      "est_cs": 0.6857
     },
     "gnm_orig_coef": {
      "oracle_size": 0.6117,
      "est_cs": 0.6344
     },
     "gnm_orig_fr": {
      "oracle_size": 0.71,
      "est_cs": 0.6407
     },
     "gnm_orig_sr": {
      "oracle_size": 0.2946,
      "est_cs": 0.3494
     },
     "gnm_va_coef": {
      "oracle_size": 0.56,
      "est_cs": 0.572
     },
     "gnm_va_fr": {
      "oracle_size": 0.7,
      "est_cs": 0.6287
     },
     "gnm_va_sr": {
      "oracle_size": 0.0964,
      "est_cs": 0.1622
     },
     "gnm_vb_coef": {
      "oracle_size": 0.1639,
      "est_cs": 0.1351
     },
     "gnm_vb_fr": {
      "oracle_size": 0.6822,
      "est_cs": 0.5763
     },
     "gnm_vb_sr": {
      "oracle_size": 0.0789,
      "est_cs": 0.0864
     },
     "gnm_vb_comp": {
      "oracle_size": 0.606,
      "est_cs": 0.518
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.5518,
      "est_cs": 0.5128
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.7205,
      "est_cs": 0.6067
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.1146,
      "est_cs": 0.1288
     },
     "flame2023_va_coef": {
      "oracle_size": 0.5766,
      "est_cs": 0.5322
     },
     "flame2023_va_fr": {
      "oracle_size": 0.7366,
      "est_cs": 0.605
     },
     "flame2023_va_sr": {
      "oracle_size": 0.1025,
      "est_cs": 0.1002
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.281,
      "est_cs": 0.2328
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.6984,
      "est_cs": 0.556
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.1385,
      "est_cs": 0.102
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.6254,
      "est_cs": 0.4974
     },
     "arcface_normals": {
      "oracle_size": 0.0294,
      "est_cs": 0.0569
     },
     "arcface_shaded": {
      "oracle_size": 0.0513,
      "est_cs": 0.0529
     },
     "shapedna_k50": {
      "oracle_size": 0.0114,
      "est_cs": 0.0679
     },
     "hks": {
      "oracle_size": -0.0006,
      "est_cs": 0.0627
     },
     "wks": {
      "oracle_size": 0.0026,
      "est_cs": -0.0613
     },
     "openshape": {
      "oracle_size": 0.0289,
      "est_cs": 0.0669
     },
     "uni3d": {
      "oracle_size": 0.0405,
      "est_cs": 0.0245
     },
     "est_cs": {
      "oracle_size": 0.4564,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.4564
     }
    }
   },
   "nicp_failures": {
    "rows_lost": 970,
    "by_topology": {
     "original": {
      "lost": 194,
      "rows_without_nicp_mask": 49500
     },
     "remesh": {
      "lost": 194,
      "rows_without_nicp_mask": 49500
     },
     "down8k": {
      "lost": 194,
      "rows_without_nicp_mask": 49500
     },
     "up60k": {
      "lost": 970,
      "rows_without_nicp_mask": 49500
     },
     "noisy": {
      "lost": 194,
      "rows_without_nicp_mask": 49500
     },
     "crop": {
      "lost": 194,
      "rows_without_nicp_mask": 49500
     }
    }
   },
   "seed": 990708,
   "n_methods": 50,
   "plugins_full": [],
   "plugins_partial": [],
   "seconds": 576.3498060703278
  },
  "facescape": {
   "load": {
    "varifold_check_rel_max_vs_shards": 1.6609071940010267e-07,
    "fits_nocrop_vs_saved_max_abs": {
     "gnm_orig_coef": 1.3100631690576847e-14,
     "gnm_orig_fr": 1.2755672629261028e-09,
     "gnm_orig_sr": 1.6633164151569346e-09,
     "gnm_va_coef": 2.6201263381153694e-14,
     "gnm_va_fr": 1.4376322493570015e-09,
     "gnm_va_sr": 1.3281761246375368e-09,
     "gnm_vb_coef": 3.552713678800501e-15,
     "gnm_vb_fr": 5.160437632767412e-10,
     "gnm_vb_sr": 8.463807571468251e-10,
     "flame2023_orig_coef": 1.2396195181452185e-12,
     "flame2023_orig_fr": 1.731326193521454e-11,
     "flame2023_orig_sr": 9.242636517248215e-11,
     "flame2023_va_coef": 7.002176616310862e-13,
     "flame2023_va_fr": 1.9528434425097885e-11,
     "flame2023_va_sr": 3.7713859812882333e-11,
     "flame2023_vb_coef": 2.071676163950542e-13,
     "flame2023_vb_fr": 2.0292129088161914e-11,
     "flame2023_vb_sr": 1.9912488324891342e-11
    }
   },
   "t1_control": {
    "n_compared": 82,
    "max_abs_diff_point_ci": 8.673617379884035e-17,
    "rows_t1_mask": 88725,
    "seed": 796786
   },
   "rows": 178200,
   "gt_spread_within_subject_pair": 0.0,
   "mask": {
    "rows_mask": 165355,
    "rows_mask_cross": 135655,
    "rows_mask_same": 29700,
    "rows_valid_gt": 178200,
    "columns_missing_on_same": [
     "mm_nicp_p2tri",
     "cs_nicp_p2tri"
    ],
    "rows_dropped_by_column_cross": {
     "mm_nicp_p2tri": 12843,
     "cs_nicp_p2tri": 12842
    }
   },
   "spearman_con_taglia": {
    "nocrop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.1353,
      "est_cs": 0.2094
     },
     "factorized_s1234|form": {
      "oracle_size": 0.1092,
      "est_cs": 0.0893
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.088,
      "est_cs": 0.0566
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.1668,
      "est_cs": 0.2688
     },
     "factorized_s2345|form": {
      "oracle_size": 0.1302,
      "est_cs": 0.1164
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.1031,
      "est_cs": 0.0697
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.2637,
      "est_cs": 0.159
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.2225,
      "est_cs": 0.1741
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.1551,
      "est_cs": 0.1995
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.1106,
      "est_cs": 0.0849
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0909,
      "est_cs": 0.072
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.2088,
      "est_cs": 0.1842
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.1182,
      "est_cs": 0.0717
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.097,
      "est_cs": 0.0627
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.0848,
      "est_cs": 0.0755
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": 0.0178,
      "est_cs": 0.0433
     },
     "mm_nicp_p2tri": {
      "oracle_size": 0.0504,
      "est_cs": 0.0231
     },
     "cs_nicp_p2tri": {
      "oracle_size": 0.0078,
      "est_cs": 0.0186
     },
     "mm_nicp_template": {
      "oracle_size": 0.4175,
      "est_cs": 0.3485
     },
     "cs_nicp_template": {
      "oracle_size": 0.0719,
      "est_cs": 0.0379
     },
     "varifold": {
      "oracle_size": 0.0578,
      "est_cs": 0.1376
     },
     "gnm_orig_coef": {
      "oracle_size": 0.1657,
      "est_cs": 0.0913
     },
     "gnm_orig_fr": {
      "oracle_size": 0.2583,
      "est_cs": 0.153
     },
     "gnm_orig_sr": {
      "oracle_size": 0.081,
      "est_cs": 0.0309
     },
     "gnm_va_coef": {
      "oracle_size": 0.1248,
      "est_cs": 0.0816
     },
     "gnm_va_fr": {
      "oracle_size": 0.2622,
      "est_cs": 0.1557
     },
     "gnm_va_sr": {
      "oracle_size": 0.0693,
      "est_cs": 0.0263
     },
     "gnm_vb_coef": {
      "oracle_size": 0.0943,
      "est_cs": 0.0695
     },
     "gnm_vb_fr": {
      "oracle_size": 0.2913,
      "est_cs": 0.1773
     },
     "gnm_vb_sr": {
      "oracle_size": 0.1143,
      "est_cs": 0.0845
     },
     "gnm_vb_comp": {
      "oracle_size": 0.2532,
      "est_cs": 0.1577
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.0856,
      "est_cs": 0.0506
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.2296,
      "est_cs": 0.1497
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0255,
      "est_cs": 0.0159
     },
     "flame2023_va_coef": {
      "oracle_size": 0.0891,
      "est_cs": 0.0548
     },
     "flame2023_va_fr": {
      "oracle_size": 0.244,
      "est_cs": 0.165
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0205,
      "est_cs": 0.0149
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.0929,
      "est_cs": 0.0657
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.2302,
      "est_cs": 0.1756
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0555,
      "est_cs": 0.0751
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1883,
      "est_cs": 0.1548
     },
     "arcface_normals": {
      "oracle_size": 0.0008,
      "est_cs": 0.014
     },
     "shapedna_k50": {
      "oracle_size": -0.0228,
      "est_cs": -0.0061
     },
     "hks": {
      "oracle_size": -0.0029,
      "est_cs": -0.0098
     },
     "wks": {
      "oracle_size": 0.0008,
      "est_cs": -0.0297
     },
     "est_cs": {
      "oracle_size": 0.2791,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.2791
     }
    },
    "crop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.0406,
      "est_cs": 0.4973
     },
     "factorized_s1234|form": {
      "oracle_size": 0.0476,
      "est_cs": 0.1536
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.0224,
      "est_cs": 0.018
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.0356,
      "est_cs": 0.6173
     },
     "factorized_s2345|form": {
      "oracle_size": 0.0737,
      "est_cs": 0.2421
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.044,
      "est_cs": 0.0471
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.0995,
      "est_cs": 0.5523
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.0551,
      "est_cs": 0.5557
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.0832,
      "est_cs": 0.414
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.0741,
      "est_cs": 0.0578
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0425,
      "est_cs": 0.0173
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.0939,
      "est_cs": 0.4692
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.0766,
      "est_cs": 0.0877
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.046,
      "est_cs": 0.0554
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.0541,
      "est_cs": -0.067
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": -0.0188,
      "est_cs": 0.0201
     },
     "mm_nicp_p2tri": {
      "oracle_size": 0.0509,
      "est_cs": -0.0814
     },
     "cs_nicp_p2tri": {
      "oracle_size": -0.0134,
      "est_cs": 0.019
     },
     "mm_nicp_template": {
      "oracle_size": 0.0662,
      "est_cs": 0.7259
     },
     "cs_nicp_template": {
      "oracle_size": 0.0048,
      "est_cs": 0.0927
     },
     "varifold": {
      "oracle_size": 0.0494,
      "est_cs": 0.4918
     },
     "gnm_orig_coef": {
      "oracle_size": 0.1281,
      "est_cs": 0.2514
     },
     "gnm_orig_fr": {
      "oracle_size": 0.2217,
      "est_cs": 0.3556
     },
     "gnm_orig_sr": {
      "oracle_size": 0.1023,
      "est_cs": 0.0475
     },
     "gnm_va_coef": {
      "oracle_size": 0.1115,
      "est_cs": 0.2173
     },
     "gnm_va_fr": {
      "oracle_size": 0.2284,
      "est_cs": 0.3366
     },
     "gnm_va_sr": {
      "oracle_size": 0.0944,
      "est_cs": -0.0057
     },
     "gnm_vb_coef": {
      "oracle_size": 0.0872,
      "est_cs": 0.0878
     },
     "gnm_vb_fr": {
      "oracle_size": 0.2742,
      "est_cs": 0.2845
     },
     "gnm_vb_sr": {
      "oracle_size": 0.1245,
      "est_cs": 0.043
     },
     "gnm_vb_comp": {
      "oracle_size": 0.2458,
      "est_cs": 0.2163
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.0775,
      "est_cs": 0.1627
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.1774,
      "est_cs": 0.3375
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0414,
      "est_cs": -0.0321
     },
     "flame2023_va_coef": {
      "oracle_size": 0.079,
      "est_cs": 0.173
     },
     "flame2023_va_fr": {
      "oracle_size": 0.1947,
      "est_cs": 0.3518
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0379,
      "est_cs": -0.0593
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.0871,
      "est_cs": 0.1589
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.203,
      "est_cs": 0.3043
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0575,
      "est_cs": 0.0019
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1748,
      "est_cs": 0.2218
     },
     "arcface_normals": {
      "oracle_size": -0.0142,
      "est_cs": -0.0166
     },
     "shapedna_k50": {
      "oracle_size": 0.0178,
      "est_cs": -0.0142
     },
     "hks": {
      "oracle_size": 0.0294,
      "est_cs": 0.0091
     },
     "wks": {
      "oracle_size": 0.0072,
      "est_cs": 0.018
     },
     "est_cs": {
      "oracle_size": 0.0106,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.0106
     }
    },
    "all_cross": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.0654,
      "est_cs": 0.7246
     },
     "factorized_s1234|form": {
      "oracle_size": 0.0634,
      "est_cs": 0.6258
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.05,
      "est_cs": 0.5904
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.0776,
      "est_cs": 0.7606
     },
     "factorized_s2345|form": {
      "oracle_size": 0.0736,
      "est_cs": 0.6682
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0584,
      "est_cs": 0.6255
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.1259,
      "est_cs": 0.7373
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.1011,
      "est_cs": 0.7442
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.0864,
      "est_cs": 0.6882
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.0686,
      "est_cs": 0.5979
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0521,
      "est_cs": 0.5988
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.1124,
      "est_cs": 0.7001
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.0758,
      "est_cs": 0.587
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.0581,
      "est_cs": 0.5923
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.0684,
      "est_cs": -0.0192
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": 0.0011,
      "est_cs": 0.4004
     },
     "mm_nicp_p2tri": {
      "oracle_size": 0.0479,
      "est_cs": -0.1637
     },
     "cs_nicp_p2tri": {
      "oracle_size": -0.0006,
      "est_cs": 0.1351
     },
     "mm_nicp_template": {
      "oracle_size": 0.2001,
      "est_cs": 0.7783
     },
     "cs_nicp_template": {
      "oracle_size": 0.0326,
      "est_cs": 0.6425
     },
     "varifold": {
      "oracle_size": 0.038,
      "est_cs": 0.2216
     },
     "gnm_orig_coef": {
      "oracle_size": 0.1484,
      "est_cs": 0.263
     },
     "gnm_orig_fr": {
      "oracle_size": 0.2434,
      "est_cs": 0.2237
     },
     "gnm_orig_sr": {
      "oracle_size": 0.0853,
      "est_cs": 0.2233
     },
     "gnm_va_coef": {
      "oracle_size": 0.1164,
      "est_cs": 0.2478
     },
     "gnm_va_fr": {
      "oracle_size": 0.2488,
      "est_cs": 0.214
     },
     "gnm_va_sr": {
      "oracle_size": 0.0757,
      "est_cs": 0.1883
     },
     "gnm_vb_coef": {
      "oracle_size": 0.0917,
      "est_cs": 0.0774
     },
     "gnm_vb_fr": {
      "oracle_size": 0.2846,
      "est_cs": 0.1689
     },
     "gnm_vb_sr": {
      "oracle_size": 0.1174,
      "est_cs": 0.1372
     },
     "gnm_vb_comp": {
      "oracle_size": 0.2493,
      "est_cs": 0.1701
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.0818,
      "est_cs": 0.1476
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.2095,
      "est_cs": 0.2003
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0302,
      "est_cs": 0.093
     },
     "flame2023_va_coef": {
      "oracle_size": 0.0845,
      "est_cs": 0.152
     },
     "flame2023_va_fr": {
      "oracle_size": 0.2251,
      "est_cs": 0.2039
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0257,
      "est_cs": 0.0864
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.0905,
      "est_cs": 0.1193
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.2203,
      "est_cs": 0.1652
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0562,
      "est_cs": 0.0883
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1832,
      "est_cs": 0.1491
     },
     "arcface_normals": {
      "oracle_size": -0.0045,
      "est_cs": -0.0382
     },
     "shapedna_k50": {
      "oracle_size": -0.0076,
      "est_cs": 0.2311
     },
     "hks": {
      "oracle_size": 0.0056,
      "est_cs": 0.2191
     },
     "wks": {
      "oracle_size": 0.0016,
      "est_cs": 0.0609
     },
     "est_cs": {
      "oracle_size": 0.1205,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.1205
     }
    }
   },
   "nicp_failures": {
    "rows_lost": 12845,
    "by_topology": {
     "original": {
      "lost": 2569,
      "rows_without_nicp_mask": 49500
     },
     "remesh": {
      "lost": 2569,
      "rows_without_nicp_mask": 49500
     },
     "down8k": {
      "lost": 2569,
      "rows_without_nicp_mask": 49500
     },
     "up60k": {
      "lost": 12845,
      "rows_without_nicp_mask": 49500
     },
     "noisy": {
      "lost": 2569,
      "rows_without_nicp_mask": 49500
     },
     "crop": {
      "lost": 2569,
      "rows_without_nicp_mask": 49500
     }
    }
   },
   "seed": 796786,
   "n_methods": 47,
   "plugins_full": [],
   "plugins_partial": [],
   "seconds": 519.6985790729523
  },
  "faceverse": {
   "load": {
    "varifold_check_rel_max_vs_shards": 1.9919494615550093e-05,
    "fits_nocrop_vs_saved_max_abs": {
     "gnm_orig_coef": 5.329070518200751e-15,
     "gnm_orig_fr": 5.51555384697977e-10,
     "gnm_orig_sr": 7.483947628283971e-10,
     "gnm_va_coef": 5.329070518200751e-15,
     "gnm_va_fr": 4.3281753181467764e-10,
     "gnm_va_sr": 6.213387304665474e-10,
     "gnm_vb_coef": 3.552713678800501e-15,
     "gnm_vb_fr": 1.4801770920058743e-10,
     "gnm_vb_sr": 2.1676349604149436e-10,
     "flame2023_orig_coef": 5.805356195764944e-13,
     "flame2023_orig_fr": 9.098166664500695e-12,
     "flame2023_orig_sr": 1.4411888349386004e-11,
     "flame2023_va_coef": 5.275779813018744e-13,
     "flame2023_va_fr": 8.950340468771856e-12,
     "flame2023_va_sr": 1.9881429835777453e-11,
     "flame2023_vb_coef": 4.618527782440651e-14,
     "flame2023_vb_fr": 3.36863870131765e-12,
     "flame2023_vb_sr": 3.689715200039245e-12
    }
   },
   "t1_control": {
    "n_compared": 82,
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
   "spearman_con_taglia": {
    "nocrop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.1182,
      "est_cs": 0.256
     },
     "factorized_s1234|form": {
      "oracle_size": 0.032,
      "est_cs": 0.144
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.0085,
      "est_cs": 0.117
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.1277,
      "est_cs": 0.2635
     },
     "factorized_s2345|form": {
      "oracle_size": 0.0467,
      "est_cs": 0.1485
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0226,
      "est_cs": 0.1199
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.0182,
      "est_cs": 0.1951
     },
     "ctrlfr_s2345|z": {
      "oracle_size": -0.0057,
      "est_cs": 0.2353
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.0723,
      "est_cs": 0.2092
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.0171,
      "est_cs": 0.1239
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0167,
      "est_cs": 0.1194
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.0575,
      "est_cs": 0.2144
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.0171,
      "est_cs": 0.1295
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.0232,
      "est_cs": 0.128
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.1013,
      "est_cs": 0.3069
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": -0.0129,
      "est_cs": 0.1042
     },
     "mm_nicp_p2tri": {
      "oracle_size": -0.0586,
      "est_cs": 0.0289
     },
     "cs_nicp_p2tri": {
      "oracle_size": -0.0453,
      "est_cs": 0.0749
     },
     "mm_nicp_template": {
      "oracle_size": 0.1965,
      "est_cs": 0.4441
     },
     "cs_nicp_template": {
      "oracle_size": 0.0063,
      "est_cs": 0.0967
     },
     "varifold": {
      "oracle_size": 0.0596,
      "est_cs": 0.277
     },
     "gnm_orig_coef": {
      "oracle_size": 0.1469,
      "est_cs": 0.08
     },
     "gnm_orig_fr": {
      "oracle_size": 0.1902,
      "est_cs": 0.2292
     },
     "gnm_orig_sr": {
      "oracle_size": 0.1079,
      "est_cs": 0.0277
     },
     "gnm_va_coef": {
      "oracle_size": 0.1509,
      "est_cs": 0.0869
     },
     "gnm_va_fr": {
      "oracle_size": 0.1935,
      "est_cs": 0.2339
     },
     "gnm_va_sr": {
      "oracle_size": 0.0994,
      "est_cs": 0.0115
     },
     "gnm_vb_coef": {
      "oracle_size": 0.0034,
      "est_cs": 0.0911
     },
     "gnm_vb_fr": {
      "oracle_size": 0.1548,
      "est_cs": 0.2283
     },
     "gnm_vb_sr": {
      "oracle_size": 0.0316,
      "est_cs": 0.0669
     },
     "gnm_vb_comp": {
      "oracle_size": 0.1216,
      "est_cs": 0.1829
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.1033,
      "est_cs": 0.0264
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.1453,
      "est_cs": 0.1337
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0837,
      "est_cs": -0.0161
     },
     "flame2023_va_coef": {
      "oracle_size": 0.1033,
      "est_cs": 0.0264
     },
     "flame2023_va_fr": {
      "oracle_size": 0.1453,
      "est_cs": 0.1337
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0837,
      "est_cs": -0.0161
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.0317,
      "est_cs": 0.0568
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.156,
      "est_cs": 0.1555
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0586,
      "est_cs": 0.0461
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1307,
      "est_cs": 0.1249
     },
     "arcface_normals": {
      "oracle_size": -0.0169,
      "est_cs": -0.0072
     },
     "arcface_shaded": {
      "oracle_size": 0.0044,
      "est_cs": 0.0117
     },
     "shapedna_k50": {
      "oracle_size": 0.0069,
      "est_cs": 0.0123
     },
     "hks": {
      "oracle_size": 0.0174,
      "est_cs": 0.0059
     },
     "wks": {
      "oracle_size": 0.0018,
      "est_cs": 0.0172
     },
     "est_cs": {
      "oracle_size": 0.1323,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.1323
     }
    },
    "crop": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.1162,
      "est_cs": 0.2413
     },
     "factorized_s1234|form": {
      "oracle_size": 0.048,
      "est_cs": 0.0721
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.0249,
      "est_cs": 0.0211
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.1123,
      "est_cs": 0.2876
     },
     "factorized_s2345|form": {
      "oracle_size": 0.0552,
      "est_cs": 0.1055
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0355,
      "est_cs": 0.048
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.0508,
      "est_cs": 0.2497
     },
     "ctrlfr_s2345|z": {
      "oracle_size": 0.0068,
      "est_cs": 0.2007
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.0861,
      "est_cs": 0.1934
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.0364,
      "est_cs": 0.0359
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0364,
      "est_cs": 0.0201
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.058,
      "est_cs": 0.2264
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.0208,
      "est_cs": 0.0512
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.0271,
      "est_cs": 0.0355
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.0884,
      "est_cs": 0.2177
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": -0.0011,
      "est_cs": 0.0542
     },
     "mm_nicp_p2tri": {
      "oracle_size": -0.0384,
      "est_cs": 0.0003
     },
     "cs_nicp_p2tri": {
      "oracle_size": -0.0289,
      "est_cs": 0.0265
     },
     "mm_nicp_template": {
      "oracle_size": 0.1261,
      "est_cs": 0.5816
     },
     "cs_nicp_template": {
      "oracle_size": 0.0044,
      "est_cs": 0.0739
     },
     "varifold": {
      "oracle_size": 0.0678,
      "est_cs": 0.4977
     },
     "gnm_orig_coef": {
      "oracle_size": 0.1573,
      "est_cs": 0.0194
     },
     "gnm_orig_fr": {
      "oracle_size": 0.1867,
      "est_cs": 0.0926
     },
     "gnm_orig_sr": {
      "oracle_size": 0.1077,
      "est_cs": -0.03
     },
     "gnm_va_coef": {
      "oracle_size": 0.1599,
      "est_cs": 0.016
     },
     "gnm_va_fr": {
      "oracle_size": 0.1906,
      "est_cs": 0.0908
     },
     "gnm_va_sr": {
      "oracle_size": 0.1001,
      "est_cs": -0.0399
     },
     "gnm_vb_coef": {
      "oracle_size": -0.0217,
      "est_cs": -0.0614
     },
     "gnm_vb_fr": {
      "oracle_size": 0.1534,
      "est_cs": 0.0708
     },
     "gnm_vb_sr": {
      "oracle_size": 0.0279,
      "est_cs": -0.0526
     },
     "gnm_vb_comp": {
      "oracle_size": 0.1205,
      "est_cs": 0.0292
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.1058,
      "est_cs": 0.0108
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.1426,
      "est_cs": 0.0646
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0882,
      "est_cs": -0.0471
     },
     "flame2023_va_coef": {
      "oracle_size": 0.1058,
      "est_cs": 0.0108
     },
     "flame2023_va_fr": {
      "oracle_size": 0.1426,
      "est_cs": 0.0646
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0882,
      "est_cs": -0.0471
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.0033,
      "est_cs": -0.0459
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.1483,
      "est_cs": 0.0454
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.04,
      "est_cs": -0.0667
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1212,
      "est_cs": 0.0089
     },
     "arcface_normals": {
      "oracle_size": -0.0242,
      "est_cs": 0.0259
     },
     "arcface_shaded": {
      "oracle_size": -0.011,
      "est_cs": 0.0397
     },
     "shapedna_k50": {
      "oracle_size": -0.0072,
      "est_cs": 0.0502
     },
     "hks": {
      "oracle_size": 0.0205,
      "est_cs": 0.0568
     },
     "wks": {
      "oracle_size": 0.0143,
      "est_cs": 0.0788
     },
     "est_cs": {
      "oracle_size": 0.0106,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.0106
     }
    },
    "all_cross": {
     "factorized_s1234|form_cal": {
      "oracle_size": 0.1165,
      "est_cs": 0.2579
     },
     "factorized_s1234|form": {
      "oracle_size": 0.0371,
      "est_cs": 0.1294
     },
     "factorized_s1234|shape": {
      "oracle_size": 0.0138,
      "est_cs": 0.1065
     },
     "factorized_s2345|form_cal": {
      "oracle_size": 0.1214,
      "est_cs": 0.2861
     },
     "factorized_s2345|form": {
      "oracle_size": 0.0492,
      "est_cs": 0.151
     },
     "factorized_s2345|shape": {
      "oracle_size": 0.0267,
      "est_cs": 0.1255
     },
     "ctrlfr_s1234|z": {
      "oracle_size": 0.0287,
      "est_cs": 0.2115
     },
     "ctrlfr_s2345|z": {
      "oracle_size": -0.0017,
      "est_cs": 0.2214
     },
     "factorizedc3m_e123|form_cal": {
      "oracle_size": 0.0766,
      "est_cs": 0.1992
     },
     "factorizedc3m_e123|form": {
      "oracle_size": 0.0235,
      "est_cs": 0.0952
     },
     "factorizedc3m_e123|shape": {
      "oracle_size": 0.0233,
      "est_cs": 0.1007
     },
     "factorizedc3m_e205|form_cal": {
      "oracle_size": 0.0576,
      "est_cs": 0.2082
     },
     "factorizedc3m_e205|form": {
      "oracle_size": 0.0183,
      "est_cs": 0.0953
     },
     "factorizedc3m_e205|shape": {
      "oracle_size": 0.0247,
      "est_cs": 0.1045
     },
     "mm_rigid_icp_chamfer": {
      "oracle_size": 0.0957,
      "est_cs": 0.2726
     },
     "cs_rigid_icp_chamfer": {
      "oracle_size": -0.0083,
      "est_cs": 0.1675
     },
     "mm_nicp_p2tri": {
      "oracle_size": -0.0516,
      "est_cs": -0.0011
     },
     "cs_nicp_p2tri": {
      "oracle_size": -0.0396,
      "est_cs": 0.0777
     },
     "mm_nicp_template": {
      "oracle_size": 0.1696,
      "est_cs": 0.5273
     },
     "cs_nicp_template": {
      "oracle_size": 0.0042,
      "est_cs": 0.162
     },
     "varifold": {
      "oracle_size": 0.0616,
      "est_cs": 0.3762
     },
     "gnm_orig_coef": {
      "oracle_size": 0.1501,
      "est_cs": 0.073
     },
     "gnm_orig_fr": {
      "oracle_size": 0.189,
      "est_cs": 0.1684
     },
     "gnm_orig_sr": {
      "oracle_size": 0.1076,
      "est_cs": 0.0234
     },
     "gnm_va_coef": {
      "oracle_size": 0.1536,
      "est_cs": 0.0767
     },
     "gnm_va_fr": {
      "oracle_size": 0.1925,
      "est_cs": 0.1703
     },
     "gnm_va_sr": {
      "oracle_size": 0.0995,
      "est_cs": 0.0106
     },
     "gnm_vb_coef": {
      "oracle_size": -0.005,
      "est_cs": 0.0397
     },
     "gnm_vb_fr": {
      "oracle_size": 0.1543,
      "est_cs": 0.1585
     },
     "gnm_vb_sr": {
      "oracle_size": 0.0304,
      "est_cs": 0.0318
     },
     "gnm_vb_comp": {
      "oracle_size": 0.1213,
      "est_cs": 0.1209
     },
     "flame2023_orig_coef": {
      "oracle_size": 0.1041,
      "est_cs": 0.0252
     },
     "flame2023_orig_fr": {
      "oracle_size": 0.1443,
      "est_cs": 0.1061
     },
     "flame2023_orig_sr": {
      "oracle_size": 0.0852,
      "est_cs": -0.0181
     },
     "flame2023_va_coef": {
      "oracle_size": 0.1041,
      "est_cs": 0.0252
     },
     "flame2023_va_fr": {
      "oracle_size": 0.1443,
      "est_cs": 0.1061
     },
     "flame2023_va_sr": {
      "oracle_size": 0.0852,
      "est_cs": -0.0181
     },
     "flame2023_vb_coef": {
      "oracle_size": 0.0221,
      "est_cs": 0.035
     },
     "flame2023_vb_fr": {
      "oracle_size": 0.1533,
      "est_cs": 0.1135
     },
     "flame2023_vb_sr": {
      "oracle_size": 0.0524,
      "est_cs": 0.0126
     },
     "flame2023_vb_comp": {
      "oracle_size": 0.1275,
      "est_cs": 0.0827
     },
     "arcface_normals": {
      "oracle_size": -0.019,
      "est_cs": -0.0354
     },
     "arcface_shaded": {
      "oracle_size": -0.0004,
      "est_cs": -0.0164
     },
     "shapedna_k50": {
      "oracle_size": 0.0023,
      "est_cs": -0.009
     },
     "hks": {
      "oracle_size": 0.0184,
      "est_cs": -0.0214
     },
     "wks": {
      "oracle_size": 0.0044,
      "est_cs": 0.0193
     },
     "est_cs": {
      "oracle_size": 0.0813,
      "est_cs": 1.0
     },
     "oracle_size": {
      "oracle_size": 1.0,
      "est_cs": 0.0813
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
   "n_methods": 48,
   "plugins_full": [],
   "plugins_partial": [],
   "seconds": 424.8509407043457
  },
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
   "n_methods": 43,
   "plugins_full": [],
   "plugins_partial": [],
   "seconds": 181.36453199386597
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
   "famos_cs_nicp_vs_famos_frame": 0.0,
   "mask": {
    "rows_mask": 105,
    "rows_mask_cross": 105,
    "rows_mask_same": 0,
    "rows_valid_gt": 105,
    "columns_missing_on_same": [],
    "rows_dropped_by_column_cross": {}
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
     "cs_nicp_p2tri": {
      "oracle_size": 0.2108,
      "est_cs": 0.3501
     },
     "mm_nicp_template": {
      "oracle_size": 0.6496,
      "est_cs": 0.8822
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
   "n_methods": 46,
   "plugins_full": [],
   "plugins_partial": [],
   "seconds": 8.208912372589111
  }
 },
 "points_vs_sources": {
  "spearman_e2_nocrop": {
   "n": 392,
   "max_abs_diff_point": 1.541486785461288e-05,
   "worst": "facescape factorizedc3m_e123|form fr: righe qui 88724, fonte 88725",
   "n_equal_1e-12": 310,
   "n_same_rows": 310
  },
  "spearman_e3_crop": {
   "n": 144,
   "max_abs_diff_point": 0.011865102426030338,
   "worst": "facescape mm_rigid_icp_chamfer sr: righe qui 46931, fonte 49500",
   "n_equal_1e-12": 68,
   "n_same_rows": 68
  },
  "spearman_e3_all_cross": {
   "n": 144,
   "max_abs_diff_point": 0.009232840918389384,
   "worst": "facescape ctrlfr_s1234|z fr: righe qui 135655, fonte 148500",
   "n_equal_1e-12": 68,
   "n_same_rows": 68
  },
  "g0_arcface_nocrop": {
   "n": 6,
   "max_abs_diff_point": 7.316718607652817e-06,
   "worst": "facescape arcface_normals fr: righe qui 88724, fonte 88725",
   "n_equal_1e-12": 0,
   "n_same_rows": 4
  },
  "g0_arcface_crop": {
   "n": 6,
   "max_abs_diff_point": 0.0018215850297232938,
   "worst": "facescape arcface_normals sr: righe qui 46931, fonte 49500",
   "n_equal_1e-12": 0,
   "n_same_rows": 2
  },
  "topo_pairs_e3_media_20_ordinate_senza_crop": {
   "n": 36,
   "max_abs_diff_point": 0.004888122426251695,
   "worst": "facescape ctrlfr_s2345|z fr: righe qui 88724, fonte 99000",
   "n_equal_1e-12": 12,
   "n_same_rows": 12
  },
  "topo_pairs_e3_media_5_stessa_topologia": {
   "n": 36,
   "max_abs_diff_point": 1.1102230246251565e-16,
   "worst": "hifi3d factorized_s2345|shape sr: righe qui 24750, fonte 24750",
   "n_equal_1e-12": 36,
   "n_same_rows": 36
  },
  "e12_hifi3d_fr_spettrali_reti": {
   "n": 5,
   "max_abs_diff_point": 0.0006342917860109737,
   "worst": "hifi3d shapedna_k50 fr: righe qui 98224, fonte 99000",
   "n_equal_1e-12": 0,
   "n_same_rows": 0
  }
 },
 "plugins": [],
 "plugin_dirs_ignored_without_fm_plugin_json": []
}
```

