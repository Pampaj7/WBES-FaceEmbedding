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
| hifi3d | 178200 | 147530 / 29700 | mm_nicp_p2tri 970, cs_nicp_p2tri 970 | 18 / 28 | 0 | gw/euclid_128_cgw2, gw/euclid_256_cg, gw/euclid_256_cgw2, now_metrical, now_original, realy, stirling, ulrssm/zeroshot_faust | now_metrical/lmk_det, now_metrical/lmk_s1, now_metrical/lmk_s2, now_metrical/lmk_s3, now_metrical/lmk_s5, now_original/lmk_det, now_original/lmk_s1, now_original/lmk_s2, now_original/lmk_s3, now_original/lmk_s5, realy/lmk_det, realy/lmk_s1, realy/lmk_s2, realy/lmk_s3, realy/lmk_s5, stirling/lmk_det, stirling/lmk_s1, stirling/lmk_s2, stirling/lmk_s3, stirling/lmk_s5 (20 colonne): 11803 righe a topologie diverse + 0 stessa, perse 135727 + 29700 |
| facescape | 178200 | 135655 / 29700 | mm_nicp_p2tri 12843, cs_nicp_p2tri 12842 | 18 / 28 | 0 | gw/euclid_128_cgw2, gw/euclid_256_cg, gw/euclid_256_cgw2, now_metrical, now_original, realy, stirling, ulrssm/zeroshot_faust | now_metrical/lmk_det, now_metrical/lmk_s1, now_metrical/lmk_s2, now_metrical/lmk_s3, now_metrical/lmk_s5, now_original/lmk_det, now_original/lmk_s1, now_original/lmk_s2, now_original/lmk_s3, now_original/lmk_s5, realy/lmk_det, realy/lmk_s1, realy/lmk_s2, realy/lmk_s3, realy/lmk_s5, stirling/lmk_det, stirling/lmk_s1, stirling/lmk_s2, stirling/lmk_s3, stirling/lmk_s5 (20 colonne): 10851 righe a topologie diverse + 0 stessa, perse 124804 + 29700 |
| faceverse | 178200 | 148500 / 29700 | 0 | 18 / 28 | 0 | gw/euclid_128_cgw2, gw/euclid_256_cg, gw/euclid_256_cgw2, now_metrical, now_original, realy, stirling, ulrssm/zeroshot_faust | - |
| faceverse_neutral | 178200 | 148500 / 29700 | 0 | 18 / 28 | 0 | gw/euclid_128_cgw2, gw/euclid_256_cg, gw/euclid_256_cgw2, now_metrical, now_original, realy, stirling | - |
| famos | 105 | 105 / 0 | 0 | 18 / 28 | 0 | - | - |

## Non applicabili e cautele

| dominio | metodo | insieme | motivo |
| --- | --- | --- | --- |
| hifi3d | NICP per coppia in mm | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| hifi3d | NICP per coppia cs | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| hifi3d | Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido), rilevatore (insightface 1k3d68 su render) (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido), sigma 1 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido), sigma 2 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido), sigma 3 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido), sigma 5 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW originale (Procrustes + ICP con scala), rilevatore (insightface 1k3d68 su render) (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW originale (Procrustes + ICP con scala), sigma 1 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW originale (Procrustes + ICP con scala), sigma 2 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW originale (Procrustes + ICP con scala), sigma 3 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | NoW originale (Procrustes + ICP con scala), sigma 5 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP), rilevatore (insightface 1k3d68 su render) (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP), sigma 1 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP), sigma 2 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP), sigma 3 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP), sigma 5 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE), rilevatore (insightface 1k3d68 su render) (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE), sigma 1 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE), sigma 2 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE), sigma 3 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE), sigma 5 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | ULRSSM [zeroshot_faust] (non dichiarata) per coppia, mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| hifi3d | ULRSSM [zeroshot_faust] (non dichiarata) per coppia, a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | ArcFace-render, ombreggiato 3 viste (E12) | tutti | render ombreggiato calcolato solo da E12 (HIFI3D, FaceVerse con espressioni) |
| facescape | OpenShape PointBERT | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| facescape | Uni3D-g | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| facescape | NICP per coppia in mm | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| facescape | NICP per coppia cs | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| facescape | Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido), rilevatore (insightface 1k3d68 su render) (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido), sigma 1 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido), sigma 2 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido), sigma 3 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido), sigma 5 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW originale (Procrustes + ICP con scala), rilevatore (insightface 1k3d68 su render) (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW originale (Procrustes + ICP con scala), sigma 1 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW originale (Procrustes + ICP con scala), sigma 2 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW originale (Procrustes + ICP con scala), sigma 3 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | NoW originale (Procrustes + ICP con scala), sigma 5 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | REALY (gICP con scala + rICP per regione + nICP), rilevatore (insightface 1k3d68 su render) (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | REALY (gICP con scala + rICP per regione + nICP), sigma 1 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | REALY (gICP con scala + rICP per regione + nICP), sigma 2 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | REALY (gICP con scala + rICP per regione + nICP), sigma 3 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | REALY (gICP con scala + rICP per regione + nICP), sigma 5 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE), rilevatore (insightface 1k3d68 su render) (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE), sigma 1 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE), sigma 2 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE), sigma 3 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE), sigma 5 mm (non dichiarata, sensibilita' ai landmark), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | ULRSSM [zeroshot_faust] (non dichiarata) per coppia, mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| facescape | ULRSSM [zeroshot_faust] (non dichiarata) per coppia, a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | plug-in now_metrical/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_metrical/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_metrical/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_metrical/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_metrical/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_original/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_original/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_original/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_original/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in now_original/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in realy/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in realy/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in realy/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in realy/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in realy/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in stirling/lmk_det | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in stirling/lmk_s1 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in stirling/lmk_s2 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in stirling/lmk_s3 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | plug-in stirling/lmk_s5 | tutti | nessun file per questo dominio al momento del calcolo |
| faceverse | OpenShape PointBERT | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| faceverse | Uni3D-g | tutti | OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d) |
| faceverse | NICP per coppia in mm | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| faceverse | NICP per coppia cs | stessa_topologia | distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato) |
| faceverse | Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | ULRSSM [zeroshot_faust] (non dichiarata) per coppia, mm (FR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
| faceverse | ULRSSM [zeroshot_faust] (non dichiarata) per coppia, a CS_ref (SR) | stessa_topologia | plug-in senza distanze fra soggetti diversi con la stessa topologia |
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
| faceverse_neutral | C3M e123, d_F cal. | tutti | embedding di C3M e123 mai calcolato sulla vista neutra (store assente; bp_e3 e faceverse_neutral lo saltano) |
| faceverse_neutral | C3M e123, d_F non cal. | tutti | embedding di C3M e123 mai calcolato sulla vista neutra (store assente; bp_e3 e faceverse_neutral lo saltano) |
| faceverse_neutral | C3M e123, d_P | tutti | embedding di C3M e123 mai calcolato sulla vista neutra (store assente; bp_e3 e faceverse_neutral lo saltano) |
| faceverse_neutral | NICP su template cs | tutti | template cs mai iscritto sulla neutra (faceverse_neutral/blmm ha solo mm) |
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

## Miglior nostro contro miglior concorrente (famiglie sintetiche, senza crop; letture dichiarate di varianti dichiarate, scelti sul punto: post hoc)

| dominio | GT | insieme | righe | miglior nostro | rho | miglior concorrente | rho | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | FR | nocrop | 98224 | ctrlfr s1234 | 0.757 [0.685, 0.817] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.712 [0.627, 0.781] | +0.046 [+0.006, +0.083] |
| hifi3d | FR | con_perturbazione original | 39406 | ctrlfr s1234 | 0.767 [0.696, 0.828] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.715 [0.631, 0.784] | +0.052 [+0.012, +0.089] |
| hifi3d | FR | con_perturbazione remesh | 39406 | ctrlfr s1234 | 0.747 [0.673, 0.805] | ULRSSM [faces_xyz] per coppia, mm (FR) | 0.710 [0.631, 0.775] | +0.036 [+0.003, +0.074] |
| hifi3d | FR | con_perturbazione down8k | 39406 | ctrlfr s1234 | 0.766 [0.693, 0.827] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.713 [0.627, 0.783] | +0.053 [+0.014, +0.092] |
| hifi3d | FR | con_perturbazione up60k | 38824 | ctrlfr s1234 | 0.755 [0.680, 0.815] | FLAME 2023 Open vB, composizione S_B + k d_P | 0.715 [0.632, 0.784] | +0.039 [+0.001, +0.077] |
| hifi3d | FR | con_perturbazione noisy | 39406 | ctrlfr s1234 | 0.753 [0.683, 0.814] | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.719 [0.651, 0.775] | +0.034 [-0.006, +0.074] |
| hifi3d | FR | stessa_topologia senza_crop | 24750 | ctrlfr s1234 | 0.779 [0.707, 0.840] | ULRSSM [faces_xyz] per coppia, mm (FR) | 0.780 [0.710, 0.837] | -0.001 [-0.036, +0.036] |
| hifi3d | SR | nocrop | 98224 | factorized s1234, d_P | 0.622 [0.550, 0.685] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.715 [0.645, 0.771] | -0.093 [-0.182, -0.013] |
| hifi3d | SR | con_perturbazione original | 39406 | factorized s1234, d_P | 0.629 [0.558, 0.692] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.720 [0.652, 0.775] | -0.091 [-0.178, -0.012] |
| hifi3d | SR | con_perturbazione remesh | 39406 | factorized s1234, d_P | 0.626 [0.553, 0.690] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.718 [0.650, 0.775] | -0.092 [-0.182, -0.013] |
| hifi3d | SR | con_perturbazione down8k | 39406 | factorized s1234, d_P | 0.625 [0.555, 0.688] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.715 [0.646, 0.772] | -0.090 [-0.177, -0.011] |
| hifi3d | SR | con_perturbazione up60k | 38824 | factorized s1234, d_P | 0.626 [0.554, 0.690] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.722 [0.653, 0.779] | -0.096 [-0.182, -0.017] |
| hifi3d | SR | con_perturbazione noisy | 39406 | factorized s1234, d_P | 0.604 [0.533, 0.668] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.717 [0.648, 0.774] | -0.112 [-0.206, -0.031] |
| hifi3d | SR | stessa_topologia senza_crop | 24750 | factorized s1234, d_P | 0.634 [0.562, 0.699] | GNM (visto) vB, coefficienti | 0.673 [0.607, 0.733] | -0.039 [-0.121, +0.045] |
| facescape | FR | nocrop | 88724 | C3M e205, d_F cal. | 0.711 [0.651, 0.763] | GNM (visto) vB, composizione S_B + k d_P | 0.800 [0.741, 0.847] | -0.089 [-0.147, -0.037] |
| facescape | FR | con_perturbazione original | 37031 | C3M e205, d_F cal. | 0.725 [0.665, 0.777] | GNM (visto) vB, composizione S_B + k d_P | 0.800 [0.741, 0.847] | -0.075 [-0.132, -0.023] |
| facescape | FR | con_perturbazione remesh | 37031 | C3M e205, d_F cal. | 0.729 [0.670, 0.780] | GNM (visto) vB, composizione S_B + k d_P | 0.801 [0.743, 0.849] | -0.072 [-0.129, -0.022] |
| facescape | FR | con_perturbazione down8k | 37031 | factorized s2345, d_F non cal. | 0.704 [0.623, 0.779] | GNM (visto) vB, composizione S_B + k d_P | 0.794 [0.732, 0.842] | -0.090 [-0.158, -0.034] |
| facescape | FR | con_perturbazione up60k | 29324 | C3M e205, d_F cal. | 0.742 [0.684, 0.792] | GNM (visto) vB, composizione S_B + k d_P | 0.803 [0.742, 0.852] | -0.061 [-0.118, -0.011] |
| facescape | FR | con_perturbazione noisy | 37031 | C3M e205, d_F cal. | 0.694 [0.633, 0.749] | GNM (visto) vB, composizione S_B + k d_P | 0.801 [0.742, 0.850] | -0.107 [-0.168, -0.052] |
| facescape | FR | stessa_topologia senza_crop | 24750 | C3M e205, d_F cal. | 0.759 [0.701, 0.809] | GNM (visto) vB, composizione S_B + k d_P | 0.807 [0.749, 0.855] | -0.048 [-0.106, +0.003] |
| facescape | SR | nocrop | 88724 | C3M e205, d_P | 0.756 [0.695, 0.807] | GNM (visto) vB, mesh d'identita' SR | 0.854 [0.803, 0.894] | -0.098 [-0.147, -0.051] |
| facescape | SR | con_perturbazione original | 37031 | C3M e205, d_P | 0.760 [0.701, 0.810] | GNM (visto) vB, mesh d'identita' SR | 0.853 [0.803, 0.892] | -0.093 [-0.141, -0.044] |
| facescape | SR | con_perturbazione remesh | 37031 | C3M e205, d_P | 0.768 [0.711, 0.815] | GNM (visto) vB, mesh d'identita' SR | 0.855 [0.806, 0.895] | -0.088 [-0.135, -0.042] |
| facescape | SR | con_perturbazione down8k | 37031 | factorized s2345, d_P | 0.761 [0.694, 0.821] | GNM (visto) vB, mesh d'identita' SR | 0.853 [0.802, 0.893] | -0.091 [-0.139, -0.047] |
| facescape | SR | con_perturbazione up60k | 29324 | C3M e205, d_P | 0.776 [0.718, 0.824] | GNM (visto) vB, mesh d'identita' SR | 0.856 [0.805, 0.896] | -0.080 [-0.127, -0.036] |
| facescape | SR | con_perturbazione noisy | 37031 | C3M e205, d_P | 0.759 [0.698, 0.809] | GNM (visto) vB, mesh d'identita' SR | 0.854 [0.803, 0.894] | -0.094 [-0.148, -0.044] |
| facescape | SR | stessa_topologia senza_crop | 24750 | C3M e205, d_P | 0.794 [0.739, 0.842] | GNM (visto) vB, mesh d'identita' SR | 0.854 [0.805, 0.894] | -0.060 [-0.107, -0.013] |
| faceverse | FR | nocrop | 99000 | factorized s2345, d_F cal. | 0.318 [0.249, 0.379] | GNM (visto) vB, composizione S_B + k d_P | 0.338 [0.255, 0.407] | -0.020 [-0.084, +0.053] |
| faceverse | FR | con_perturbazione original | 39600 | C3M e205, d_F non cal. | 0.322 [0.247, 0.392] | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.352 [0.273, 0.423] | -0.031 [-0.102, +0.046] |
| faceverse | FR | con_perturbazione remesh | 39600 | factorized s2345, d_F cal. | 0.322 [0.254, 0.385] | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.341 [0.268, 0.410] | -0.019 [-0.081, +0.053] |
| faceverse | FR | con_perturbazione down8k | 39600 | factorized s2345, d_F non cal. | 0.306 [0.230, 0.373] | GNM (visto) vB, composizione S_B + k d_P | 0.341 [0.257, 0.411] | -0.035 [-0.104, +0.041] |
| faceverse | FR | con_perturbazione up60k | 39600 | factorized s2345, d_F cal. | 0.321 [0.251, 0.383] | ICP + Chamfer in mm (per coppia) | 0.335 [0.255, 0.411] | -0.014 [-0.085, +0.056] |
| faceverse | FR | con_perturbazione noisy | 39600 | factorized s2345, d_F non cal. | 0.330 [0.253, 0.400] | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.356 [0.275, 0.434] | -0.027 [-0.102, +0.045] |
| faceverse | FR | stessa_topologia senza_crop | 24750 | C3M e205, d_F cal. | 0.333 [0.260, 0.402] | ICP + Chamfer in mm (per coppia) | 0.339 [0.263, 0.410] | -0.006 [-0.070, +0.061] |
| faceverse | SR | nocrop | 99000 | C3M e205, d_P | 0.314 [0.242, 0.382] | GNM (visto) vB, mesh d'identita' SR | 0.327 [0.239, 0.402] | -0.013 [-0.089, +0.063] |
| faceverse | SR | con_perturbazione original | 39600 | C3M e205, d_P | 0.330 [0.254, 0.403] | GNM (visto) vB, mesh d'identita' SR | 0.325 [0.237, 0.400] | +0.005 [-0.072, +0.086] |
| faceverse | SR | con_perturbazione remesh | 39600 | C3M e205, d_P | 0.327 [0.247, 0.400] | GNM (visto) vB, mesh d'identita' SR | 0.325 [0.235, 0.401] | +0.002 [-0.082, +0.088] |
| faceverse | SR | con_perturbazione down8k | 39600 | factorized s2345, d_P | 0.303 [0.225, 0.372] | GNM (visto) vB, mesh d'identita' SR | 0.338 [0.250, 0.412] | -0.035 [-0.115, +0.045] |
| faceverse | SR | con_perturbazione up60k | 39600 | C3M e123, d_P | 0.313 [0.235, 0.388] | GNM (visto) vB, mesh d'identita' SR | 0.322 [0.233, 0.397] | -0.008 [-0.091, +0.068] |
| faceverse | SR | con_perturbazione noisy | 39600 | factorized s2345, d_P | 0.320 [0.240, 0.389] | GNM (visto) vB, mesh d'identita' SR | 0.329 [0.240, 0.404] | -0.009 [-0.082, +0.066] |
| faceverse | SR | stessa_topologia senza_crop | 24750 | C3M e205, d_P | 0.336 [0.260, 0.405] | GNM (visto) vB, mesh d'identita' SR | 0.324 [0.236, 0.400] | +0.011 [-0.068, +0.090] |
| faceverse_neutral | FR | nocrop | 99000 | factorized s2345, d_F non cal. | 0.373 [0.293, 0.451] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.479 [0.393, 0.557] | -0.106 [-0.209, -0.005] |
| faceverse_neutral | FR | con_perturbazione original | 39600 | factorized s2345, d_F non cal. | 0.375 [0.294, 0.453] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.480 [0.394, 0.559] | -0.105 [-0.208, -0.003] |
| faceverse_neutral | FR | con_perturbazione remesh | 39600 | factorized s2345, d_F non cal. | 0.378 [0.297, 0.454] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.480 [0.392, 0.558] | -0.102 [-0.206, +0.002] |
| faceverse_neutral | FR | con_perturbazione down8k | 39600 | factorized s2345, d_F non cal. | 0.373 [0.292, 0.450] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.477 [0.392, 0.555] | -0.105 [-0.206, -0.004] |
| faceverse_neutral | FR | con_perturbazione up60k | 39600 | factorized s2345, d_F non cal. | 0.374 [0.293, 0.452] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.481 [0.394, 0.559] | -0.107 [-0.209, -0.005] |
| faceverse_neutral | FR | con_perturbazione noisy | 39600 | factorized s2345, d_F non cal. | 0.367 [0.289, 0.444] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.488 [0.402, 0.565] | -0.121 [-0.225, -0.018] |
| faceverse_neutral | FR | stessa_topologia senza_crop | 24750 | C3M e205, d_F cal. | 0.399 [0.317, 0.474] | ULRSSM [faces_xyz] via template, GT FR (FR) | 0.417 [0.334, 0.495] | -0.018 [-0.100, +0.066] |
| faceverse_neutral | SR | nocrop | 99000 | C3M e205, d_P | 0.382 [0.304, 0.452] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.521 [0.447, 0.590] | -0.139 [-0.225, -0.053] |
| faceverse_neutral | SR | con_perturbazione original | 39600 | C3M e205, d_P | 0.398 [0.321, 0.469] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.523 [0.448, 0.590] | -0.125 [-0.213, -0.039] |
| faceverse_neutral | SR | con_perturbazione remesh | 39600 | C3M e205, d_P | 0.393 [0.313, 0.464] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.523 [0.449, 0.591] | -0.130 [-0.218, -0.041] |
| faceverse_neutral | SR | con_perturbazione down8k | 39600 | C3M e205, d_P | 0.387 [0.307, 0.459] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.520 [0.446, 0.588] | -0.133 [-0.220, -0.047] |
| faceverse_neutral | SR | con_perturbazione up60k | 39600 | C3M e205, d_P | 0.389 [0.310, 0.463] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.523 [0.449, 0.591] | -0.134 [-0.222, -0.046] |
| faceverse_neutral | SR | con_perturbazione noisy | 39600 | factorized s2345, d_P | 0.365 [0.286, 0.437] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.530 [0.454, 0.599] | -0.165 [-0.256, -0.074] |
| faceverse_neutral | SR | stessa_topologia senza_crop | 24750 | C3M e205, d_P | 0.417 [0.336, 0.491] | ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.411 [0.333, 0.483] | +0.005 [-0.070, +0.079] |

Ogni lettura nostra di riferimento contro il miglior concorrente della riga sopra (nocrop; delta `vs_nostro`, stesse righe e repliche; gli altri insiemi in `deltas.csv`):

| dominio | GT | insieme | miglior concorrente | lettura nostra | rho nostro | rho concorrente | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | FR | nocrop | FLAME 2023 Open vB, composizione S_B + k d_P | factorized s1234, d_F cal. | 0.749 | 0.712 | +0.037 [-0.002, +0.075] |
| hifi3d | FR | nocrop | FLAME 2023 Open vB, composizione S_B + k d_P | factorized s2345, d_F cal. | 0.731 | 0.712 | +0.019 [-0.019, +0.057] |
| hifi3d | FR | nocrop | FLAME 2023 Open vB, composizione S_B + k d_P | factorized s1234, d_F non cal. | 0.642 | 0.712 | -0.069 [-0.127, -0.012] |
| hifi3d | FR | nocrop | FLAME 2023 Open vB, composizione S_B + k d_P | factorized s2345, d_F non cal. | 0.631 | 0.712 | -0.080 [-0.143, -0.019] |
| hifi3d | FR | nocrop | FLAME 2023 Open vB, composizione S_B + k d_P | ctrlfr s1234 | 0.757 | 0.712 | +0.046 [+0.006, +0.083] |
| hifi3d | FR | nocrop | FLAME 2023 Open vB, composizione S_B + k d_P | ctrlfr s2345 | 0.746 | 0.712 | +0.034 [-0.003, +0.071] |
| hifi3d | SR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_P | 0.622 | 0.715 | -0.093 [-0.182, -0.013] |
| hifi3d | SR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_P | 0.613 | 0.715 | -0.102 [-0.193, -0.014] |
| facescape | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | factorized s1234, d_F cal. | 0.661 | 0.800 | -0.139 [-0.194, -0.090] |
| facescape | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | factorized s2345, d_F cal. | 0.669 | 0.800 | -0.131 [-0.184, -0.086] |
| facescape | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | factorized s1234, d_F non cal. | 0.687 | 0.800 | -0.112 [-0.175, -0.058] |
| facescape | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | factorized s2345, d_F non cal. | 0.698 | 0.800 | -0.101 [-0.169, -0.046] |
| facescape | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | ctrlfr s1234 | 0.661 | 0.800 | -0.138 [-0.203, -0.082] |
| facescape | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | ctrlfr s2345 | 0.653 | 0.800 | -0.147 [-0.208, -0.092] |
| facescape | SR | nocrop | GNM (visto) vB, mesh d'identita' SR | factorized s1234, d_P | 0.747 | 0.854 | -0.107 [-0.155, -0.065] |
| facescape | SR | nocrop | GNM (visto) vB, mesh d'identita' SR | factorized s2345, d_P | 0.753 | 0.854 | -0.101 [-0.148, -0.058] |
| faceverse | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | factorized s1234, d_F cal. | 0.303 | 0.338 | -0.036 [-0.105, +0.045] |
| faceverse | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | factorized s2345, d_F cal. | 0.318 | 0.338 | -0.020 [-0.084, +0.053] |
| faceverse | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | factorized s1234, d_F non cal. | 0.297 | 0.338 | -0.041 [-0.115, +0.034] |
| faceverse | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | factorized s2345, d_F non cal. | 0.317 | 0.338 | -0.021 [-0.090, +0.051] |
| faceverse | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | ctrlfr s1234 | 0.282 | 0.338 | -0.056 [-0.134, +0.022] |
| faceverse | FR | nocrop | GNM (visto) vB, composizione S_B + k d_P | ctrlfr s2345 | 0.258 | 0.338 | -0.080 [-0.159, +0.001] |
| faceverse | SR | nocrop | GNM (visto) vB, mesh d'identita' SR | factorized s1234, d_P | 0.286 | 0.327 | -0.041 [-0.112, +0.033] |
| faceverse | SR | nocrop | GNM (visto) vB, mesh d'identita' SR | factorized s2345, d_P | 0.313 | 0.327 | -0.014 [-0.085, +0.058] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_F cal. | 0.341 | 0.479 | -0.138 [-0.242, -0.033] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_F cal. | 0.370 | 0.479 | -0.109 [-0.204, -0.006] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_F non cal. | 0.337 | 0.479 | -0.142 [-0.236, -0.037] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_F non cal. | 0.373 | 0.479 | -0.106 [-0.209, -0.005] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | ctrlfr s1234 | 0.327 | 0.479 | -0.152 [-0.248, -0.052] |
| faceverse_neutral | FR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | ctrlfr s2345 | 0.299 | 0.479 | -0.180 [-0.286, -0.076] |
| faceverse_neutral | SR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_P | 0.333 | 0.521 | -0.188 [-0.274, -0.096] |
| faceverse_neutral | SR | nocrop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_P | 0.372 | 0.521 | -0.149 [-0.240, -0.058] |

## Scala dell'allineamento (senza crop)

Delta appaiati a - b (stesse righe, stesse repliche) lungo i gradi di liberta' dell'allineamento. PRINCIPALI: 1 (togliere la taglia: la modalita' cs scala ogni mesh alla cs_ref del dominio, `blmm.work_coords`; non e' una scala per coppia), 1b (togliere la taglia PER COPPIA: NoW originale, Procrustes + ICP con scala stimata per coppia, - NoW metrical, la stessa catena senza scala; landmark esatti; lato piu' allineato a), 2 (ICP cs - NICP cs), 3b (NICP su template cs - NICP per coppia cs), 4 e 5 con FR contro ICP mm, con SR contro ICP cs e NICP cs (con SR il riferimento corretto e' senza taglia). SECONDARI: 2b e 3 in mm, CONFUSI DALLA TAGLIA (in mm il NICP su template segue la taglia, il NICP per coppia la assorbe: vedi la tabella sotto), e 4 e 5 con SR contro ICP mm. 4: fit B con la mesh della GT (FR con FR, SR con SR); 5: nostro preregistrato (d_F non cal. con FR, d_P con SR), entrambi i semi.

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

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.285 [+0.187, +0.383] | +0.291 [+0.196, +0.385] | +0.289 [+0.188, +0.392] | +0.276 [+0.180, +0.375] | +0.289 [+0.187, +0.393] | +0.289 [+0.191, +0.386] | +0.275 [+0.175, +0.376] |
| hifi3d | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.281 [-0.372, -0.194] | -0.282 [-0.372, -0.195] | -0.282 [-0.373, -0.195] | -0.287 [-0.380, -0.196] | -0.285 [-0.377, -0.194] | -0.287 [-0.378, -0.199] | - |
| hifi3d | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.017 [-0.046, +0.012] | -0.011 [-0.035, +0.017] | -0.012 [-0.042, +0.018] | -0.026 [-0.057, +0.004] | -0.012 [-0.042, +0.019] | -0.014 [-0.041, +0.013] | - |
| hifi3d | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.187 [-0.275, -0.105] | -0.166 [-0.251, -0.085] | -0.200 [-0.287, -0.118] | -0.183 [-0.272, -0.099] | -0.194 [-0.285, -0.109] | -0.201 [-0.295, -0.116] | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.022 [-0.028, +0.071] | +0.031 [-0.021, +0.082] | -0.006 [-0.056, +0.044] | +0.044 [-0.002, +0.093] | +0.000 [-0.049, +0.048] | +0.004 [-0.047, +0.055] | +0.007 [-0.044, +0.056] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | +0.043 [-0.007, +0.092] | +0.053 [+0.002, +0.100] | +0.014 [-0.035, +0.062] | +0.069 [+0.021, +0.118] | +0.023 [-0.027, +0.073] | +0.021 [-0.032, +0.072] | +0.031 [-0.021, +0.082] |
| hifi3d | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.001 [-0.055, +0.055] | +0.012 [-0.041, +0.065] | -0.026 [-0.081, +0.029] | +0.023 [-0.032, +0.077] | -0.020 [-0.075, +0.037] | -0.024 [-0.081, +0.032] | -0.006 [-0.062, +0.050] |
| hifi3d | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.012 [-0.075, +0.051] | -0.001 [-0.063, +0.061] | -0.039 [-0.101, +0.025] | +0.014 [-0.048, +0.075] | -0.033 [-0.096, +0.032] | -0.034 [-0.097, +0.030] | -0.020 [-0.083, +0.044] |
| facescape | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.038 [+0.018, +0.058] | +0.048 [+0.029, +0.068] | +0.046 [+0.022, +0.070] | +0.021 [+0.005, +0.038] | +0.042 [+0.019, +0.065] | +0.047 [+0.025, +0.070] | +0.038 [+0.013, +0.062] |
| facescape | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.052 [-0.091, -0.017] | -0.057 [-0.096, -0.022] | -0.054 [-0.092, -0.018] | -0.053 [-0.094, -0.014] | -0.056 [-0.098, -0.019] | -0.061 [-0.102, -0.024] | - |
| facescape | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.069 [+0.043, +0.095] | +0.063 [+0.038, +0.088] | +0.073 [+0.042, +0.103] | +0.076 [+0.052, +0.100] | +0.074 [+0.044, +0.104] | +0.068 [+0.042, +0.094] | - |
| facescape | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | +0.054 [-0.032, +0.136] | +0.056 [-0.024, +0.135] | -0.039 [-0.128, +0.046] | +0.122 [+0.036, +0.206] | -0.014 [-0.107, +0.076] | +0.032 [-0.050, +0.115] | - |
| facescape | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.225 [+0.162, +0.292] | +0.229 [+0.163, +0.299] | +0.131 [+0.064, +0.199] | +0.276 [+0.220, +0.335] | +0.159 [+0.090, +0.231] | +0.193 [+0.125, +0.265] | +0.078 [+0.012, +0.148] |
| facescape | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | +0.177 [+0.116, +0.241] | +0.180 [+0.116, +0.247] | +0.072 [+0.014, +0.136] | +0.235 [+0.181, +0.293] | +0.114 [+0.053, +0.180] | +0.147 [+0.080, +0.215] | +0.031 [-0.032, +0.095] |
| facescape | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | +0.234 [+0.183, +0.285] | +0.244 [+0.190, +0.299] | +0.149 [+0.097, +0.199] | +0.289 [+0.240, +0.336] | +0.171 [+0.121, +0.225] | +0.201 [+0.145, +0.254] | +0.113 [+0.060, +0.167] |
| facescape | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | +0.245 [+0.185, +0.301] | +0.251 [+0.187, +0.310] | +0.163 [+0.103, +0.221] | +0.309 [+0.253, +0.361] | +0.184 [+0.122, +0.243] | +0.194 [+0.130, +0.253] | +0.111 [+0.051, +0.171] |
| faceverse | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.048 [-0.004, +0.102] | +0.044 [-0.007, +0.099] | +0.048 [-0.001, +0.100] | +0.048 [-0.004, +0.101] | +0.050 [-0.004, +0.108] | +0.046 [-0.007, +0.102] | +0.049 [-0.002, +0.105] |
| faceverse | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.038 [-0.078, +0.000] | -0.039 [-0.082, +0.002] | -0.039 [-0.080, +0.001] | -0.032 [-0.070, +0.005] | -0.038 [-0.084, +0.002] | -0.036 [-0.080, +0.003] | - |
| faceverse | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.078 [+0.022, +0.129] | +0.095 [+0.039, +0.147] | +0.081 [+0.023, +0.133] | +0.061 [+0.001, +0.115] | +0.080 [+0.023, +0.134] | +0.074 [+0.018, +0.128] | - |
| faceverse | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.076 [-0.184, +0.029] | -0.064 [-0.168, +0.042] | -0.069 [-0.183, +0.038] | -0.099 [-0.219, +0.019] | -0.061 [-0.173, +0.047] | -0.090 [-0.199, +0.015] | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | -0.045 [-0.112, +0.019] | -0.046 [-0.115, +0.025] | -0.052 [-0.124, +0.020] | -0.033 [-0.106, +0.037] | -0.051 [-0.119, +0.015] | -0.045 [-0.110, +0.019] | -0.046 [-0.113, +0.018] |
| faceverse | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.073 [-0.151, +0.002] | -0.075 [-0.148, -0.002] | -0.078 [-0.162, +0.003] | -0.055 [-0.140, +0.025] | -0.078 [-0.157, -0.004] | -0.080 [-0.158, -0.004] | -0.071 [-0.150, +0.003] |
| faceverse | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.040 [-0.118, +0.036] | -0.057 [-0.135, +0.020] | -0.040 [-0.120, +0.039] | -0.036 [-0.116, +0.047] | -0.035 [-0.117, +0.042] | -0.033 [-0.110, +0.043] | -0.037 [-0.115, +0.040] |
| faceverse | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.020 [-0.105, +0.056] | -0.036 [-0.115, +0.038] | -0.022 [-0.108, +0.056] | -0.018 [-0.111, +0.068] | -0.016 [-0.104, +0.063] | -0.010 [-0.096, +0.070] | -0.021 [-0.108, +0.057] |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.035 [-0.018, +0.090] | +0.037 [-0.015, +0.092] | +0.034 [-0.020, +0.089] | +0.035 [-0.019, +0.091] | +0.036 [-0.018, +0.091] | +0.031 [-0.021, +0.086] | +0.039 [-0.015, +0.095] |
| faceverse_neutral | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.021 [-0.060, +0.019] | -0.020 [-0.059, +0.019] | -0.020 [-0.060, +0.020] | -0.020 [-0.061, +0.021] | -0.020 [-0.060, +0.022] | -0.018 [-0.059, +0.022] | - |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.088 [+0.022, +0.148] | +0.092 [+0.026, +0.153] | +0.093 [+0.028, +0.155] | +0.084 [+0.018, +0.148] | +0.090 [+0.023, +0.150] | +0.081 [+0.019, +0.137] | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | -0.088 [-0.160, -0.015] | -0.093 [-0.164, -0.021] | -0.096 [-0.169, -0.025] | -0.078 [-0.154, -0.003] | -0.092 [-0.164, -0.021] | -0.083 [-0.155, -0.014] | -0.095 [-0.167, -0.022] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.094 [-0.175, -0.012] | -0.101 [-0.182, -0.019] | -0.096 [-0.176, -0.016] | -0.086 [-0.170, +0.002] | -0.098 [-0.180, -0.017] | -0.091 [-0.173, -0.008] | -0.100 [-0.183, -0.019] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.072 [-0.151, +0.013] | -0.074 [-0.153, +0.010] | -0.074 [-0.152, +0.009] | -0.067 [-0.147, +0.020] | -0.070 [-0.150, +0.014] | -0.076 [-0.156, +0.006] | -0.065 [-0.145, +0.021] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.035 [-0.123, +0.042] | -0.040 [-0.125, +0.038] | -0.037 [-0.125, +0.043] | -0.029 [-0.116, +0.052] | -0.038 [-0.122, +0.040] | -0.037 [-0.122, +0.042] | -0.035 [-0.123, +0.044] |

GT SR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.215 [-0.287, -0.147] | -0.189 [-0.258, -0.122] | -0.235 [-0.312, -0.164] | -0.214 [-0.286, -0.146] | -0.239 [-0.317, -0.169] | -0.218 [-0.291, -0.147] | -0.245 [-0.320, -0.177] |
| hifi3d | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.190 [+0.132, +0.252] | +0.186 [+0.129, +0.249] | +0.189 [+0.130, +0.251] | +0.200 [+0.141, +0.264] | +0.194 [+0.134, +0.259] | +0.195 [+0.136, +0.259] | - |
| hifi3d | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.014 [-0.043, +0.021] | -0.005 [-0.032, +0.027] | -0.004 [-0.034, +0.032] | -0.028 [-0.059, +0.008] | -0.005 [-0.034, +0.031] | -0.011 [-0.038, +0.020] | - |
| hifi3d | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.294 [-0.390, -0.203] | -0.258 [-0.351, -0.166] | -0.322 [-0.416, -0.232] | -0.284 [-0.383, -0.192] | -0.310 [-0.409, -0.220] | -0.312 [-0.408, -0.222] | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.027 [-0.022, +0.080] | +0.052 [+0.003, +0.108] | -0.011 [-0.063, +0.043] | +0.049 [+0.005, +0.099] | -0.008 [-0.057, +0.046] | +0.013 [-0.038, +0.069] | -0.016 [-0.067, +0.038] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.017 [-0.033, +0.069] | +0.044 [-0.006, +0.098] | -0.023 [-0.077, +0.030] | +0.040 [-0.012, +0.091] | -0.013 [-0.064, +0.040] | +0.000 [-0.050, +0.053] | -0.024 [-0.077, +0.030] |
| hifi3d | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.041 [-0.005, +0.091] | +0.073 [+0.025, +0.122] | +0.009 [-0.040, +0.059] | +0.068 [+0.023, +0.112] | +0.012 [-0.035, +0.061] | +0.010 [-0.040, +0.062] | +0.012 [-0.039, +0.063] |
| hifi3d | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.033 [-0.027, +0.088] | +0.061 [-0.000, +0.118] | -0.001 [-0.063, +0.055] | +0.059 [+0.005, +0.112] | +0.002 [-0.059, +0.057] | +0.005 [-0.054, +0.063] | -0.003 [-0.065, +0.053] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.012 [-0.045, +0.072] | +0.047 [-0.010, +0.108] | -0.015 [-0.077, +0.048] | +0.021 [-0.034, +0.081] | -0.012 [-0.071, +0.049] | +0.001 [-0.056, +0.062] | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.003 [-0.056, +0.064] | +0.039 [-0.019, +0.100] | -0.027 [-0.087, +0.036] | +0.012 [-0.046, +0.074] | -0.018 [-0.079, +0.046] | -0.011 [-0.070, +0.051] | - |
| hifi3d | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.027 [-0.029, +0.083] | +0.068 [+0.014, +0.125] | +0.005 [-0.053, +0.064] | +0.040 [-0.013, +0.096] | +0.007 [-0.048, +0.067] | -0.001 [-0.058, +0.057] | - |
| hifi3d | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.018 [-0.046, +0.082] | +0.056 [-0.008, +0.120] | -0.005 [-0.072, +0.061] | +0.031 [-0.031, +0.093] | -0.003 [-0.069, +0.063] | -0.006 [-0.070, +0.058] | - |
| facescape | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.001 [-0.011, +0.009] | +0.011 [+0.001, +0.021] | -0.001 [-0.012, +0.011] | -0.011 [-0.022, -0.000] | -0.002 [-0.014, +0.009] | +0.004 [-0.007, +0.014] | -0.018 [-0.032, -0.005] |
| facescape | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.024 [+0.008, +0.040] | +0.019 [+0.004, +0.034] | +0.022 [+0.007, +0.039] | +0.027 [+0.009, +0.045] | +0.022 [+0.007, +0.040] | +0.020 [+0.004, +0.037] | - |
| facescape | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.080 [+0.055, +0.106] | +0.073 [+0.049, +0.097] | +0.084 [+0.055, +0.114] | +0.086 [+0.062, +0.109] | +0.085 [+0.057, +0.115] | +0.078 [+0.053, +0.103] | - |
| facescape | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | +0.038 [-0.052, +0.129] | +0.038 [-0.050, +0.129] | -0.071 [-0.166, +0.026] | +0.118 [+0.029, +0.205] | -0.043 [-0.140, +0.057] | +0.016 [-0.073, +0.109] | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.376 [+0.323, +0.429] | +0.390 [+0.335, +0.444] | +0.276 [+0.222, +0.332] | +0.427 [+0.383, +0.472] | +0.299 [+0.245, +0.356] | +0.348 [+0.292, +0.403] | +0.198 [+0.140, +0.256] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.343 [+0.280, +0.402] | +0.359 [+0.296, +0.419] | +0.242 [+0.181, +0.301] | +0.395 [+0.339, +0.444] | +0.268 [+0.208, +0.329] | +0.309 [+0.243, +0.370] | +0.166 [+0.101, +0.224] |
| facescape | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.269 [+0.217, +0.317] | +0.294 [+0.238, +0.343] | +0.171 [+0.121, +0.221] | +0.320 [+0.273, +0.362] | +0.196 [+0.141, +0.247] | +0.248 [+0.195, +0.298] | +0.132 [+0.078, +0.183] |
| facescape | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.275 [+0.222, +0.328] | +0.296 [+0.241, +0.352] | +0.183 [+0.128, +0.236] | +0.335 [+0.288, +0.380] | +0.207 [+0.152, +0.263] | +0.233 [+0.176, +0.288] | +0.126 [+0.071, +0.183] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.456 [+0.392, +0.519] | +0.463 [+0.399, +0.527] | +0.361 [+0.289, +0.430] | +0.512 [+0.457, +0.567] | +0.384 [+0.314, +0.451] | +0.427 [+0.362, +0.492] | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.422 [+0.350, +0.490] | +0.432 [+0.359, +0.500] | +0.326 [+0.246, +0.401] | +0.481 [+0.417, +0.540] | +0.353 [+0.276, +0.425] | +0.387 [+0.311, +0.456] | - |
| facescape | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.349 [+0.291, +0.401] | +0.367 [+0.306, +0.422] | +0.256 [+0.195, +0.311] | +0.405 [+0.351, +0.456] | +0.281 [+0.219, +0.338] | +0.326 [+0.266, +0.381] | - |
| facescape | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.355 [+0.290, +0.412] | +0.369 [+0.303, +0.430] | +0.267 [+0.200, +0.331] | +0.421 [+0.365, +0.473] | +0.292 [+0.225, +0.356] | +0.311 [+0.243, +0.371] | - |
| faceverse | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.004 [-0.040, +0.050] | -0.003 [-0.046, +0.042] | +0.005 [-0.036, +0.050] | +0.008 [-0.037, +0.054] | +0.003 [-0.040, +0.049] | +0.003 [-0.044, +0.053] | +0.006 [-0.039, +0.054] |
| faceverse | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.003 [-0.041, +0.037] | -0.000 [-0.040, +0.040] | -0.003 [-0.039, +0.036] | -0.001 [-0.039, +0.041] | -0.003 [-0.044, +0.038] | +0.000 [-0.043, +0.040] | - |
| faceverse | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.071 [+0.016, +0.124] | +0.086 [+0.031, +0.137] | +0.074 [+0.018, +0.127] | +0.054 [-0.003, +0.107] | +0.076 [+0.019, +0.130] | +0.065 [+0.007, +0.119] | - |
| faceverse | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.088 [-0.191, +0.022] | -0.078 [-0.177, +0.027] | -0.083 [-0.190, +0.031] | -0.111 [-0.224, +0.008] | -0.071 [-0.182, +0.043] | -0.101 [-0.205, +0.009] | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.021 [-0.039, +0.079] | +0.003 [-0.055, +0.060] | +0.017 [-0.045, +0.077] | +0.045 [-0.021, +0.108] | +0.014 [-0.045, +0.069] | +0.026 [-0.035, +0.085] | +0.020 [-0.040, +0.076] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | -0.013 [-0.089, +0.057] | -0.030 [-0.104, +0.039] | -0.017 [-0.097, +0.056] | +0.012 [-0.069, +0.090] | -0.020 [-0.089, +0.048] | -0.015 [-0.094, +0.056] | -0.011 [-0.087, +0.060] |
| faceverse | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | -0.020 [-0.093, +0.055] | -0.039 [-0.108, +0.039] | -0.014 [-0.091, +0.062] | -0.017 [-0.091, +0.060] | -0.017 [-0.089, +0.057] | -0.014 [-0.089, +0.067] | -0.012 [-0.082, +0.063] |
| faceverse | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.007 [-0.065, +0.078] | -0.009 [-0.079, +0.060] | +0.010 [-0.060, +0.080] | +0.010 [-0.069, +0.088] | +0.004 [-0.071, +0.078] | +0.017 [-0.056, +0.089] | +0.009 [-0.064, +0.080] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.092 [+0.018, +0.161] | +0.089 [+0.012, +0.159] | +0.090 [+0.017, +0.160] | +0.099 [+0.023, +0.170] | +0.090 [+0.016, +0.158] | +0.091 [+0.016, +0.162] | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.057 [-0.021, +0.135] | +0.056 [-0.026, +0.136] | +0.057 [-0.023, +0.136] | +0.066 [-0.014, +0.152] | +0.055 [-0.021, +0.131] | +0.050 [-0.031, +0.130] | - |
| faceverse | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.051 [-0.027, +0.124] | +0.047 [-0.029, +0.115] | +0.059 [-0.022, +0.135] | +0.037 [-0.042, +0.115] | +0.058 [-0.020, +0.135] | +0.052 [-0.025, +0.125] | - |
| faceverse | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.078 [-0.005, +0.152] | +0.077 [-0.002, +0.150] | +0.084 [-0.004, +0.161] | +0.064 [-0.026, +0.146] | +0.080 [-0.009, +0.162] | +0.082 [+0.000, +0.159] | - |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.014 [-0.058, +0.030] | -0.012 [-0.056, +0.033] | -0.015 [-0.058, +0.030] | -0.014 [-0.058, +0.031] | -0.014 [-0.057, +0.031] | -0.018 [-0.060, +0.026] | -0.011 [-0.055, +0.034] |
| faceverse_neutral | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.018 [-0.017, +0.056] | +0.019 [-0.017, +0.057] | +0.019 [-0.018, +0.058] | +0.020 [-0.017, +0.060] | +0.021 [-0.016, +0.061] | +0.023 [-0.013, +0.063] | - |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.074 [+0.014, +0.133] | +0.079 [+0.018, +0.139] | +0.078 [+0.018, +0.138] | +0.070 [+0.008, +0.131] | +0.076 [+0.015, +0.134] | +0.069 [+0.012, +0.124] | - |
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
| hifi3d | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.148 [+0.071, +0.231] | +0.162 [+0.089, +0.242] | +0.153 [+0.073, +0.238] | +0.135 [+0.057, +0.219] | +0.155 [+0.073, +0.242] | +0.150 [+0.074, +0.230] |
| hifi3d | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | +0.119 [+0.002, +0.233] | +0.141 [+0.028, +0.257] | +0.101 [-0.018, +0.215] | +0.121 [+0.000, +0.237] | +0.108 [-0.010, +0.219] | +0.104 [-0.015, +0.221] |
| facescape | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.079 [+0.049, +0.109] | +0.083 [+0.054, +0.113] | +0.085 [+0.048, +0.121] | +0.076 [+0.051, +0.101] | +0.084 [+0.050, +0.119] | +0.083 [+0.052, +0.114] |
| facescape | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | +0.168 [+0.076, +0.260] | +0.196 [+0.104, +0.284] | +0.094 [-0.004, +0.192] | +0.189 [+0.109, +0.271] | +0.117 [+0.020, +0.214] | +0.126 [+0.032, +0.220] |
| faceverse | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.136 [+0.066, +0.209] | +0.147 [+0.077, +0.219] | +0.141 [+0.070, +0.213] | +0.124 [+0.052, +0.198] | +0.138 [+0.064, +0.210] | +0.132 [+0.060, +0.204] |
| faceverse | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | +0.011 [-0.115, +0.135] | +0.011 [-0.124, +0.139] | +0.028 [-0.096, +0.155] | +0.003 [-0.127, +0.128] | +0.012 [-0.115, +0.133] | -0.002 [-0.128, +0.126] |
| faceverse_neutral | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.131 [+0.049, +0.212] | +0.137 [+0.055, +0.218] | +0.135 [+0.051, +0.216] | +0.129 [+0.045, +0.215] | +0.132 [+0.051, +0.214] | +0.123 [+0.044, +0.200] |
| faceverse_neutral | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.054 [-0.185, +0.072] | -0.054 [-0.183, +0.074] | -0.055 [-0.188, +0.072] | -0.045 [-0.175, +0.084] | -0.056 [-0.188, +0.071] | -0.063 [-0.191, +0.062] |

GT SR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | -0.138 [-0.214, -0.058] | -0.110 [-0.184, -0.033] | -0.149 [-0.231, -0.067] | -0.155 [-0.233, -0.075] | -0.147 [-0.227, -0.063] | -0.126 [-0.201, -0.048] | - |
| hifi3d | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.420 [-0.526, -0.313] | -0.394 [-0.498, -0.287] | -0.448 [-0.555, -0.340] | -0.417 [-0.522, -0.307] | -0.436 [-0.545, -0.329] | -0.424 [-0.528, -0.316] | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.242 [+0.165, +0.323] | +0.241 [+0.164, +0.322] | +0.224 [+0.145, +0.308] | +0.263 [+0.189, +0.341] | +0.232 [+0.151, +0.315] | +0.230 [+0.153, +0.313] | +0.229 [+0.154, +0.313] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.232 [+0.158, +0.310] | +0.233 [+0.159, +0.311] | +0.212 [+0.136, +0.292] | +0.254 [+0.182, +0.329] | +0.226 [+0.149, +0.307] | +0.218 [+0.144, +0.297] | +0.221 [+0.146, +0.301] |
| hifi3d | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.256 [+0.181, +0.332] | +0.262 [+0.190, +0.336] | +0.244 [+0.167, +0.322] | +0.282 [+0.208, +0.355] | +0.251 [+0.172, +0.333] | +0.228 [+0.150, +0.305] | +0.257 [+0.181, +0.334] |
| hifi3d | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.248 [+0.170, +0.333] | +0.250 [+0.171, +0.334] | +0.234 [+0.153, +0.322] | +0.273 [+0.199, +0.354] | +0.241 [+0.160, +0.330] | +0.223 [+0.139, +0.307] | +0.242 [+0.163, +0.329] |
| facescape | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.071 [+0.043, +0.100] | +0.074 [+0.048, +0.101] | +0.074 [+0.042, +0.109] | +0.069 [+0.046, +0.093] | +0.074 [+0.043, +0.107] | +0.073 [+0.045, +0.103] | - |
| facescape | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.007 [-0.084, +0.074] | +0.016 [-0.065, +0.097] | -0.094 [-0.177, -0.010] | +0.037 [-0.030, +0.109] | -0.071 [-0.155, +0.019] | -0.054 [-0.131, +0.031] | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.377 [+0.324, +0.430] | +0.379 [+0.323, +0.433] | +0.277 [+0.222, +0.335] | +0.438 [+0.391, +0.484] | +0.302 [+0.245, +0.358] | +0.345 [+0.287, +0.401] | +0.216 [+0.159, +0.273] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.343 [+0.280, +0.402] | +0.348 [+0.283, +0.408] | +0.242 [+0.178, +0.303] | +0.406 [+0.348, +0.456] | +0.270 [+0.208, +0.330] | +0.305 [+0.237, +0.364] | +0.184 [+0.118, +0.242] |
| facescape | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.269 [+0.217, +0.317] | +0.283 [+0.230, +0.331] | +0.172 [+0.118, +0.222] | +0.331 [+0.281, +0.375] | +0.198 [+0.143, +0.249] | +0.244 [+0.191, +0.294] | +0.150 [+0.095, +0.200] |
| facescape | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.276 [+0.221, +0.328] | +0.285 [+0.229, +0.340] | +0.183 [+0.128, +0.236] | +0.347 [+0.298, +0.394] | +0.210 [+0.153, +0.263] | +0.229 [+0.172, +0.283] | +0.144 [+0.087, +0.200] |
| faceverse | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.086 [+0.022, +0.152] | +0.093 [+0.032, +0.158] | +0.091 [+0.025, +0.159] | +0.077 [+0.007, +0.149] | +0.088 [+0.021, +0.154] | +0.083 [+0.015, +0.148] | - |
| faceverse | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.072 [-0.194, +0.051] | -0.076 [-0.203, +0.048] | -0.058 [-0.182, +0.067] | -0.076 [-0.202, +0.051] | -0.072 [-0.194, +0.045] | -0.078 [-0.200, +0.045] | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.018 [-0.050, +0.083] | +0.006 [-0.057, +0.071] | +0.011 [-0.058, +0.079] | +0.037 [-0.037, +0.109] | +0.011 [-0.058, +0.075] | +0.023 [-0.048, +0.093] | +0.014 [-0.053, +0.080] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | -0.017 [-0.092, +0.063] | -0.027 [-0.104, +0.051] | -0.022 [-0.104, +0.062] | +0.004 [-0.079, +0.085] | -0.023 [-0.097, +0.056] | -0.018 [-0.095, +0.066] | -0.017 [-0.092, +0.062] |
| faceverse | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | -0.023 [-0.097, +0.056] | -0.037 [-0.111, +0.040] | -0.020 [-0.094, +0.057] | -0.025 [-0.105, +0.060] | -0.020 [-0.096, +0.057] | -0.016 [-0.089, +0.062] | -0.017 [-0.093, +0.059] |
| faceverse | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.004 [-0.080, +0.079] | -0.007 [-0.085, +0.065] | +0.005 [-0.074, +0.079] | +0.002 [-0.089, +0.083] | +0.002 [-0.085, +0.078] | +0.014 [-0.069, +0.090] | +0.004 [-0.080, +0.078] |
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
| FR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 2 / 0 / 2 (su 4) |
| FR | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | principale | 2 / 0 / 2 (su 4) |
| FR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 3 / 0 / 1 (su 4) |
| FR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 4 / 0 / 0 (su 4) |
| FR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 2 / 0 / 2 (su 4) |
| FR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 1 / 2 (su 3) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | principale | 1 / 1 / 2 (su 4) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | principale | 1 / 1 / 2 (su 4) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | principale | 1 / 0 / 3 (su 4) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | principale | 1 / 0 / 3 (su 4) |
| SR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 0 / 1 / 3 (su 4) |
| SR | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | principale | 0 / 2 / 2 (su 4) |
| SR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 3 / 0 / 1 (su 4) |
| SR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 2 / 1 / 1 (su 4) |
| SR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 0 / 2 / 2 (su 4) |
| SR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 1 / 2 (su 3) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 1 / 3 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 1 / 3 (su 4) |
| SR | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | principale | 1 / 0 / 3 (su 4) |
| SR | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | principale | 1 / 0 / 3 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 3 / 1 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 1 / 3 (su 4) |
| SR | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | principale | 1 / 0 / 3 (su 4) |
| SR | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | principale | 1 / 0 / 3 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | secondario | 0 / 2 / 2 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | secondario | 0 / 2 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | secondario | 2 / 0 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | secondario | 2 / 0 / 2 (su 4) |

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
| hifi3d | SR | nocrop | NICP per coppia in mm | 0.503 [0.437, 0.574] (98224) | 0.494 [0.429, 0.563] (99000) |
| hifi3d | SR | nocrop | NICP per coppia cs | 0.595 [0.546, 0.648] (98224) | 0.585 [0.535, 0.637] (99000) |
| hifi3d | SR | nocrop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.014 [-0.043, +0.021] | -0.004 [-0.032, +0.029] |
| hifi3d | SR | nocrop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.294 [-0.390, -0.203] | -0.283 [-0.381, -0.192] |
| facescape | FR | nocrop | NICP per coppia in mm | 0.374 [0.305, 0.446] (88724) | 0.297 [0.240, 0.356] (99000) |
| facescape | FR | nocrop | NICP per coppia cs | 0.346 [0.276, 0.419] (88724) | 0.274 [0.216, 0.337] (99000) |
| facescape | FR | nocrop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.069 [+0.043, +0.095] | +0.155 [+0.127, +0.180] |
| facescape | FR | nocrop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | +0.054 [-0.032, +0.136] | +0.130 [+0.050, +0.210] |
| facescape | SR | nocrop | NICP per coppia in mm | 0.406 [0.336, 0.472] (88724) | 0.322 [0.265, 0.378] (99000) |
| facescape | SR | nocrop | NICP per coppia cs | 0.398 [0.329, 0.464] (88724) | 0.316 [0.260, 0.369] (99000) |
| facescape | SR | nocrop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.080 [+0.055, +0.106] | +0.177 [+0.152, +0.200] |
| facescape | SR | nocrop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | +0.038 [-0.052, +0.129] | +0.123 [+0.040, +0.210] |

## Protocolli: landmark esatti e landmark rilevati (lmk_det, colonna aggiuntiva marcata)

NoW originale, NoW metrical, REALY e Stirling: lettura dichiarata coi landmark esatti (passo `comune`). Il rilevatore (`lmk_det`, insightface 1k3d68 sui render: la variante che si calcola su Ava-256) e il rumore sui landmark (`lmk_s1`-`lmk_s5`, sigma in mm, non dichiarati) esistono solo su un sottocampione di coppie (HIFI3D e FaceScape): passo `righe proprie`, con i landmark esatti e i bracci di riferimento sulle stesse righe e repliche (le righe perse sono nella tabella della maschera). Spearman (senza crop):

HIFI3D:

| GT | insieme | protocollo | righe | landmark esatti (dichiarata) | **lmk_det (rilevatore)** | lmk_s1 | lmk_s2 | lmk_s3 | lmk_s5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FR | nocrop | now_metrical | 7880 | 0.694 [0.623, 0.749] | 0.684 [0.615, 0.742] | 0.693 [0.623, 0.748] | 0.693 [0.623, 0.748] | 0.692 [0.621, 0.748] | 0.689 [0.618, 0.745] |
| FR | nocrop | now_original | 7880 | 0.411 [0.324, 0.501] | 0.412 [0.321, 0.504] | 0.410 [0.322, 0.500] | 0.410 [0.321, 0.500] | 0.409 [0.319, 0.499] | 0.408 [0.319, 0.497] |
| FR | nocrop | realy | 7880 | 0.489 [0.388, 0.576] | 0.419 [0.330, 0.500] | 0.480 [0.382, 0.564] | 0.452 [0.360, 0.534] | 0.382 [0.291, 0.467] | 0.197 [0.124, 0.262] |
| FR | nocrop | stirling | 7880 | 0.458 [0.359, 0.550] | 0.375 [0.282, 0.461] | 0.447 [0.349, 0.535] | 0.407 [0.315, 0.486] | 0.346 [0.264, 0.417] | 0.223 [0.158, 0.285] |
| FR | con_perturbazione original | now_metrical | 3184 | 0.706 [0.632, 0.762] | 0.695 [0.623, 0.751] | 0.705 [0.632, 0.761] | 0.705 [0.631, 0.761] | 0.703 [0.629, 0.759] | 0.700 [0.626, 0.756] |
| FR | con_perturbazione original | now_original | 3184 | 0.421 [0.317, 0.515] | 0.422 [0.318, 0.519] | 0.420 [0.317, 0.513] | 0.420 [0.316, 0.513] | 0.419 [0.317, 0.513] | 0.417 [0.315, 0.511] |
| FR | con_perturbazione original | realy | 3184 | 0.490 [0.380, 0.583] | 0.431 [0.323, 0.520] | 0.477 [0.371, 0.565] | 0.445 [0.336, 0.539] | 0.368 [0.255, 0.464] | 0.184 [0.092, 0.268] |
| FR | con_perturbazione original | stirling | 3184 | 0.472 [0.361, 0.572] | 0.406 [0.301, 0.504] | 0.460 [0.348, 0.558] | 0.418 [0.314, 0.508] | 0.357 [0.255, 0.440] | 0.232 [0.150, 0.302] |
| FR | con_perturbazione remesh | now_metrical | 3118 | 0.688 [0.611, 0.752] | 0.680 [0.607, 0.744] | 0.689 [0.611, 0.752] | 0.689 [0.612, 0.751] | 0.689 [0.612, 0.752] | 0.688 [0.612, 0.752] |
| FR | con_perturbazione remesh | now_original | 3118 | 0.406 [0.314, 0.489] | 0.408 [0.315, 0.491] | 0.406 [0.313, 0.489] | 0.406 [0.314, 0.490] | 0.406 [0.313, 0.490] | 0.405 [0.312, 0.488] |
| FR | con_perturbazione remesh | realy | 3118 | 0.499 [0.403, 0.590] | 0.438 [0.350, 0.522] | 0.490 [0.396, 0.578] | 0.465 [0.371, 0.556] | 0.403 [0.307, 0.498] | 0.212 [0.134, 0.297] |
| FR | con_perturbazione remesh | stirling | 3118 | 0.458 [0.355, 0.547] | 0.379 [0.283, 0.465] | 0.449 [0.351, 0.538] | 0.403 [0.308, 0.486] | 0.332 [0.245, 0.412] | 0.215 [0.139, 0.293] |
| FR | con_perturbazione down8k | now_metrical | 3092 | 0.699 [0.624, 0.760] | 0.697 [0.622, 0.759] | 0.699 [0.622, 0.760] | 0.698 [0.620, 0.760] | 0.696 [0.616, 0.759] | 0.695 [0.615, 0.758] |
| FR | con_perturbazione down8k | now_original | 3092 | 0.416 [0.317, 0.508] | 0.419 [0.318, 0.512] | 0.415 [0.314, 0.507] | 0.414 [0.313, 0.505] | 0.413 [0.312, 0.505] | 0.414 [0.312, 0.507] |
| FR | con_perturbazione down8k | realy | 3092 | 0.479 [0.369, 0.578] | 0.427 [0.328, 0.523] | 0.477 [0.366, 0.577] | 0.455 [0.347, 0.549] | 0.381 [0.274, 0.476] | 0.184 [0.094, 0.273] |
| FR | con_perturbazione down8k | stirling | 3092 | 0.461 [0.349, 0.558] | 0.375 [0.274, 0.466] | 0.450 [0.346, 0.542] | 0.414 [0.316, 0.499] | 0.357 [0.263, 0.437] | 0.232 [0.150, 0.303] |
| FR | con_perturbazione up60k | now_metrical | 3160 | 0.704 [0.624, 0.764] | 0.698 [0.619, 0.758] | 0.703 [0.625, 0.763] | 0.701 [0.625, 0.762] | 0.700 [0.623, 0.760] | 0.696 [0.618, 0.756] |
| FR | con_perturbazione up60k | now_original | 3160 | 0.415 [0.305, 0.508] | 0.420 [0.311, 0.512] | 0.414 [0.302, 0.508] | 0.413 [0.302, 0.507] | 0.412 [0.302, 0.504] | 0.409 [0.301, 0.501] |
| FR | con_perturbazione up60k | realy | 3160 | 0.485 [0.364, 0.579] | 0.427 [0.328, 0.516] | 0.482 [0.367, 0.576] | 0.459 [0.357, 0.549] | 0.387 [0.284, 0.476] | 0.203 [0.116, 0.290] |
| FR | con_perturbazione up60k | stirling | 3160 | 0.462 [0.340, 0.566] | 0.390 [0.286, 0.488] | 0.461 [0.344, 0.561] | 0.429 [0.324, 0.521] | 0.376 [0.274, 0.460] | 0.234 [0.156, 0.311] |
| FR | con_perturbazione noisy | now_metrical | 3206 | 0.718 [0.643, 0.775] | 0.711 [0.640, 0.769] | 0.719 [0.644, 0.776] | 0.718 [0.644, 0.775] | 0.718 [0.642, 0.775] | 0.717 [0.642, 0.774] |
| FR | con_perturbazione noisy | now_original | 3206 | 0.425 [0.325, 0.527] | 0.426 [0.321, 0.528] | 0.424 [0.324, 0.527] | 0.423 [0.324, 0.527] | 0.423 [0.323, 0.525] | 0.423 [0.322, 0.526] |
| FR | con_perturbazione noisy | realy | 3206 | 0.502 [0.392, 0.599] | 0.409 [0.306, 0.497] | 0.485 [0.379, 0.581] | 0.448 [0.343, 0.542] | 0.379 [0.273, 0.475] | 0.203 [0.110, 0.290] |
| FR | con_perturbazione noisy | stirling | 3206 | 0.457 [0.346, 0.563] | 0.341 [0.239, 0.440] | 0.434 [0.329, 0.535] | 0.385 [0.289, 0.479] | 0.321 [0.231, 0.406] | 0.215 [0.130, 0.296] |
| SR | nocrop | now_metrical | 7880 | 0.433 [0.341, 0.519] | 0.447 [0.357, 0.532] | 0.433 [0.340, 0.519] | 0.432 [0.340, 0.519] | 0.432 [0.340, 0.519] | 0.429 [0.338, 0.516] |
| SR | nocrop | now_original | 7880 | 0.620 [0.554, 0.678] | 0.614 [0.545, 0.673] | 0.620 [0.554, 0.677] | 0.619 [0.554, 0.678] | 0.619 [0.552, 0.677] | 0.616 [0.549, 0.674] |
| SR | nocrop | realy | 7880 | 0.703 [0.631, 0.764] | 0.620 [0.556, 0.674] | 0.680 [0.607, 0.740] | 0.624 [0.549, 0.688] | 0.515 [0.431, 0.586] | 0.289 [0.223, 0.353] |
| SR | nocrop | stirling | 7880 | 0.633 [0.547, 0.706] | 0.531 [0.452, 0.601] | 0.619 [0.538, 0.686] | 0.562 [0.486, 0.624] | 0.476 [0.405, 0.534] | 0.303 [0.235, 0.360] |
| SR | con_perturbazione original | now_metrical | 3184 | 0.439 [0.341, 0.526] | 0.452 [0.352, 0.537] | 0.439 [0.340, 0.525] | 0.439 [0.340, 0.526] | 0.439 [0.338, 0.527] | 0.436 [0.336, 0.525] |
| SR | con_perturbazione original | now_original | 3184 | 0.622 [0.553, 0.684] | 0.617 [0.545, 0.679] | 0.621 [0.552, 0.683] | 0.621 [0.553, 0.684] | 0.621 [0.553, 0.683] | 0.619 [0.552, 0.680] |
| SR | con_perturbazione original | realy | 3184 | 0.697 [0.615, 0.760] | 0.633 [0.559, 0.693] | 0.673 [0.592, 0.738] | 0.615 [0.528, 0.683] | 0.500 [0.406, 0.579] | 0.273 [0.184, 0.352] |
| SR | con_perturbazione original | stirling | 3184 | 0.637 [0.547, 0.708] | 0.567 [0.481, 0.639] | 0.627 [0.539, 0.694] | 0.571 [0.490, 0.635] | 0.485 [0.401, 0.549] | 0.298 [0.213, 0.366] |
| SR | con_perturbazione remesh | now_metrical | 3118 | 0.435 [0.341, 0.524] | 0.447 [0.352, 0.532] | 0.435 [0.341, 0.524] | 0.434 [0.340, 0.524] | 0.434 [0.339, 0.523] | 0.431 [0.334, 0.520] |
| SR | con_perturbazione remesh | now_original | 3118 | 0.617 [0.540, 0.686] | 0.610 [0.533, 0.680] | 0.617 [0.541, 0.686] | 0.617 [0.541, 0.686] | 0.617 [0.541, 0.685] | 0.615 [0.538, 0.683] |
| SR | con_perturbazione remesh | realy | 3118 | 0.712 [0.641, 0.778] | 0.656 [0.591, 0.710] | 0.692 [0.622, 0.755] | 0.644 [0.565, 0.712] | 0.549 [0.457, 0.628] | 0.319 [0.240, 0.397] |
| SR | con_perturbazione remesh | stirling | 3118 | 0.628 [0.529, 0.705] | 0.548 [0.456, 0.631] | 0.611 [0.511, 0.685] | 0.545 [0.452, 0.620] | 0.455 [0.367, 0.528] | 0.289 [0.203, 0.363] |
| SR | con_perturbazione down8k | now_metrical | 3092 | 0.424 [0.320, 0.525] | 0.444 [0.338, 0.544] | 0.423 [0.319, 0.523] | 0.422 [0.315, 0.524] | 0.421 [0.312, 0.523] | 0.417 [0.307, 0.521] |
| SR | con_perturbazione down8k | now_original | 3092 | 0.628 [0.546, 0.696] | 0.626 [0.544, 0.698] | 0.627 [0.543, 0.695] | 0.625 [0.541, 0.695] | 0.623 [0.538, 0.694] | 0.622 [0.536, 0.692] |
| SR | con_perturbazione down8k | realy | 3092 | 0.697 [0.610, 0.771] | 0.653 [0.579, 0.713] | 0.684 [0.600, 0.754] | 0.636 [0.548, 0.709] | 0.522 [0.428, 0.606] | 0.299 [0.212, 0.379] |
| SR | con_perturbazione down8k | stirling | 3092 | 0.634 [0.541, 0.714] | 0.548 [0.450, 0.627] | 0.621 [0.532, 0.697] | 0.569 [0.489, 0.637] | 0.486 [0.402, 0.558] | 0.304 [0.224, 0.377] |
| SR | con_perturbazione up60k | now_metrical | 3160 | 0.446 [0.349, 0.531] | 0.459 [0.363, 0.544] | 0.445 [0.349, 0.531] | 0.445 [0.350, 0.531] | 0.446 [0.351, 0.531] | 0.444 [0.350, 0.530] |
| SR | con_perturbazione up60k | now_original | 3160 | 0.636 [0.561, 0.697] | 0.630 [0.554, 0.691] | 0.636 [0.560, 0.697] | 0.635 [0.561, 0.697] | 0.635 [0.560, 0.697] | 0.632 [0.557, 0.695] |
| SR | con_perturbazione up60k | realy | 3160 | 0.729 [0.656, 0.788] | 0.645 [0.575, 0.701] | 0.708 [0.634, 0.768] | 0.645 [0.570, 0.708] | 0.525 [0.437, 0.598] | 0.272 [0.186, 0.351] |
| SR | con_perturbazione up60k | stirling | 3160 | 0.649 [0.567, 0.721] | 0.550 [0.463, 0.622] | 0.639 [0.561, 0.709] | 0.591 [0.515, 0.655] | 0.514 [0.441, 0.576] | 0.336 [0.252, 0.408] |
| SR | con_perturbazione noisy | now_metrical | 3206 | 0.452 [0.353, 0.550] | 0.473 [0.374, 0.570] | 0.452 [0.353, 0.550] | 0.452 [0.353, 0.549] | 0.453 [0.354, 0.552] | 0.450 [0.350, 0.549] |
| SR | con_perturbazione noisy | now_original | 3206 | 0.646 [0.576, 0.708] | 0.644 [0.571, 0.709] | 0.646 [0.575, 0.708] | 0.646 [0.574, 0.707] | 0.646 [0.574, 0.708] | 0.645 [0.569, 0.707] |
| SR | con_perturbazione noisy | realy | 3206 | 0.701 [0.623, 0.768] | 0.567 [0.482, 0.637] | 0.664 [0.584, 0.736] | 0.599 [0.504, 0.678] | 0.492 [0.396, 0.577] | 0.286 [0.200, 0.368] |
| SR | con_perturbazione noisy | stirling | 3206 | 0.650 [0.553, 0.725] | 0.465 [0.371, 0.552] | 0.622 [0.531, 0.695] | 0.557 [0.467, 0.626] | 0.458 [0.368, 0.535] | 0.297 [0.211, 0.379] |

FaceScape (dev):

| GT | insieme | protocollo | righe | landmark esatti (dichiarata) | **lmk_det (rilevatore)** | lmk_s1 | lmk_s2 | lmk_s3 | lmk_s5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FR | nocrop | now_metrical | 7088 | 0.674 [0.601, 0.733] | 0.676 [0.607, 0.733] | 0.674 [0.600, 0.734] | 0.674 [0.601, 0.733] | 0.673 [0.600, 0.733] | 0.672 [0.598, 0.730] |
| FR | nocrop | now_original | 7088 | 0.621 [0.532, 0.701] | 0.615 [0.525, 0.693] | 0.621 [0.532, 0.700] | 0.620 [0.530, 0.699] | 0.620 [0.528, 0.699] | 0.619 [0.525, 0.699] |
| FR | nocrop | realy | 7088 | 0.613 [0.527, 0.692] | 0.463 [0.374, 0.557] | 0.587 [0.499, 0.668] | 0.503 [0.415, 0.588] | 0.382 [0.292, 0.470] | 0.216 [0.141, 0.293] |
| FR | nocrop | stirling | 7088 | 0.579 [0.478, 0.672] | 0.485 [0.381, 0.585] | 0.575 [0.475, 0.664] | 0.526 [0.430, 0.614] | 0.443 [0.351, 0.528] | 0.259 [0.177, 0.338] |
| FR | con_perturbazione original | now_metrical | 3004 | 0.654 [0.569, 0.722] | 0.652 [0.569, 0.719] | 0.653 [0.567, 0.723] | 0.653 [0.567, 0.724] | 0.653 [0.568, 0.723] | 0.653 [0.567, 0.723] |
| FR | con_perturbazione original | now_original | 3004 | 0.603 [0.503, 0.687] | 0.594 [0.495, 0.682] | 0.602 [0.503, 0.688] | 0.602 [0.502, 0.687] | 0.602 [0.501, 0.688] | 0.602 [0.503, 0.689] |
| FR | con_perturbazione original | realy | 3004 | 0.611 [0.520, 0.696] | 0.494 [0.388, 0.592] | 0.581 [0.480, 0.672] | 0.485 [0.379, 0.585] | 0.375 [0.276, 0.472] | 0.209 [0.129, 0.293] |
| FR | con_perturbazione original | stirling | 3004 | 0.566 [0.457, 0.667] | 0.516 [0.398, 0.625] | 0.566 [0.460, 0.662] | 0.515 [0.420, 0.611] | 0.431 [0.343, 0.523] | 0.244 [0.157, 0.335] |
| FR | con_perturbazione remesh | now_metrical | 2970 | 0.679 [0.595, 0.743] | 0.679 [0.598, 0.744] | 0.678 [0.593, 0.743] | 0.678 [0.593, 0.744] | 0.677 [0.593, 0.744] | 0.675 [0.590, 0.741] |
| FR | con_perturbazione remesh | now_original | 2970 | 0.621 [0.519, 0.703] | 0.615 [0.515, 0.700] | 0.620 [0.518, 0.702] | 0.620 [0.518, 0.703] | 0.619 [0.516, 0.703] | 0.618 [0.516, 0.702] |
| FR | con_perturbazione remesh | realy | 2970 | 0.611 [0.517, 0.697] | 0.468 [0.379, 0.561] | 0.586 [0.492, 0.673] | 0.508 [0.414, 0.599] | 0.371 [0.265, 0.472] | 0.208 [0.117, 0.302] |
| FR | con_perturbazione remesh | stirling | 2970 | 0.577 [0.475, 0.670] | 0.498 [0.390, 0.596] | 0.573 [0.466, 0.664] | 0.521 [0.410, 0.612] | 0.433 [0.313, 0.527] | 0.236 [0.134, 0.326] |
| FR | con_perturbazione down8k | now_metrical | 2931 | 0.743 [0.672, 0.801] | 0.758 [0.689, 0.814] | 0.743 [0.672, 0.801] | 0.742 [0.673, 0.801] | 0.742 [0.672, 0.800] | 0.739 [0.668, 0.797] |
| FR | con_perturbazione down8k | now_original | 2931 | 0.693 [0.599, 0.770] | 0.695 [0.601, 0.770] | 0.692 [0.598, 0.770] | 0.691 [0.596, 0.768] | 0.690 [0.595, 0.767] | 0.688 [0.593, 0.764] |
| FR | con_perturbazione down8k | realy | 2931 | 0.644 [0.557, 0.724] | 0.516 [0.426, 0.611] | 0.622 [0.530, 0.704] | 0.527 [0.429, 0.612] | 0.405 [0.298, 0.496] | 0.242 [0.148, 0.331] |
| FR | con_perturbazione down8k | stirling | 2931 | 0.631 [0.525, 0.723] | 0.545 [0.433, 0.644] | 0.631 [0.528, 0.718] | 0.583 [0.479, 0.674] | 0.491 [0.379, 0.589] | 0.290 [0.187, 0.393] |
| FR | con_perturbazione up60k | now_metrical | 2306 | 0.712 [0.631, 0.773] | 0.708 [0.627, 0.771] | 0.711 [0.630, 0.773] | 0.710 [0.628, 0.772] | 0.709 [0.627, 0.772] | 0.706 [0.625, 0.768] |
| FR | con_perturbazione up60k | now_original | 2306 | 0.657 [0.561, 0.736] | 0.646 [0.547, 0.730] | 0.656 [0.560, 0.734] | 0.655 [0.557, 0.734] | 0.654 [0.555, 0.734] | 0.651 [0.550, 0.734] |
| FR | con_perturbazione up60k | realy | 2306 | 0.614 [0.515, 0.709] | 0.500 [0.398, 0.601] | 0.599 [0.502, 0.694] | 0.532 [0.430, 0.625] | 0.403 [0.299, 0.505] | 0.228 [0.129, 0.324] |
| FR | con_perturbazione up60k | stirling | 2306 | 0.604 [0.490, 0.698] | 0.534 [0.413, 0.635] | 0.597 [0.486, 0.695] | 0.548 [0.437, 0.651] | 0.464 [0.350, 0.569] | 0.285 [0.178, 0.385] |
| FR | con_perturbazione noisy | now_metrical | 2965 | 0.694 [0.620, 0.755] | 0.705 [0.634, 0.764] | 0.694 [0.620, 0.756] | 0.694 [0.619, 0.755] | 0.694 [0.620, 0.755] | 0.693 [0.620, 0.753] |
| FR | con_perturbazione noisy | now_original | 2965 | 0.622 [0.529, 0.710] | 0.624 [0.529, 0.711] | 0.622 [0.529, 0.711] | 0.623 [0.529, 0.712] | 0.623 [0.528, 0.712] | 0.624 [0.528, 0.712] |
| FR | con_perturbazione noisy | realy | 2965 | 0.606 [0.510, 0.687] | 0.449 [0.338, 0.557] | 0.563 [0.467, 0.652] | 0.482 [0.382, 0.571] | 0.365 [0.263, 0.460] | 0.197 [0.106, 0.282] |
| FR | con_perturbazione noisy | stirling | 2965 | 0.568 [0.462, 0.663] | 0.415 [0.297, 0.527] | 0.560 [0.457, 0.652] | 0.511 [0.411, 0.600] | 0.435 [0.330, 0.533] | 0.262 [0.159, 0.356] |
| SR | nocrop | now_metrical | 7088 | 0.706 [0.649, 0.758] | 0.706 [0.649, 0.758] | 0.706 [0.649, 0.758] | 0.706 [0.649, 0.758] | 0.706 [0.649, 0.758] | 0.705 [0.647, 0.756] |
| SR | nocrop | now_original | 7088 | 0.726 [0.672, 0.776] | 0.722 [0.664, 0.771] | 0.726 [0.669, 0.776] | 0.726 [0.668, 0.775] | 0.725 [0.667, 0.774] | 0.723 [0.664, 0.772] |
| SR | nocrop | realy | 7088 | 0.714 [0.643, 0.775] | 0.517 [0.439, 0.599] | 0.683 [0.607, 0.747] | 0.584 [0.503, 0.658] | 0.448 [0.364, 0.532] | 0.262 [0.194, 0.335] |
| SR | nocrop | stirling | 7088 | 0.678 [0.606, 0.743] | 0.551 [0.460, 0.634] | 0.672 [0.600, 0.736] | 0.612 [0.535, 0.682] | 0.515 [0.431, 0.588] | 0.310 [0.227, 0.383] |
| SR | con_perturbazione original | now_metrical | 3004 | 0.697 [0.629, 0.755] | 0.695 [0.627, 0.754] | 0.696 [0.629, 0.755] | 0.697 [0.629, 0.756] | 0.698 [0.629, 0.757] | 0.698 [0.628, 0.758] |
| SR | con_perturbazione original | now_original | 3004 | 0.720 [0.657, 0.776] | 0.713 [0.651, 0.768] | 0.719 [0.656, 0.775] | 0.720 [0.656, 0.775] | 0.720 [0.656, 0.775] | 0.719 [0.656, 0.773] |
| SR | con_perturbazione original | realy | 3004 | 0.717 [0.641, 0.782] | 0.554 [0.466, 0.639] | 0.680 [0.598, 0.751] | 0.570 [0.481, 0.653] | 0.438 [0.347, 0.530] | 0.247 [0.165, 0.325] |
| SR | con_perturbazione original | stirling | 3004 | 0.680 [0.603, 0.749] | 0.593 [0.495, 0.683] | 0.674 [0.597, 0.744] | 0.609 [0.527, 0.685] | 0.506 [0.420, 0.587] | 0.289 [0.201, 0.380] |
| SR | con_perturbazione remesh | now_metrical | 2970 | 0.713 [0.650, 0.769] | 0.711 [0.647, 0.768] | 0.712 [0.649, 0.768] | 0.712 [0.649, 0.767] | 0.713 [0.649, 0.767] | 0.712 [0.649, 0.766] |
| SR | con_perturbazione remesh | now_original | 2970 | 0.727 [0.665, 0.780] | 0.721 [0.657, 0.773] | 0.727 [0.664, 0.780] | 0.726 [0.663, 0.780] | 0.726 [0.661, 0.779] | 0.725 [0.660, 0.777] |
| SR | con_perturbazione remesh | realy | 2970 | 0.707 [0.631, 0.773] | 0.534 [0.454, 0.617] | 0.677 [0.600, 0.745] | 0.586 [0.504, 0.666] | 0.440 [0.346, 0.531] | 0.258 [0.177, 0.343] |
| SR | con_perturbazione remesh | stirling | 2970 | 0.676 [0.598, 0.747] | 0.577 [0.479, 0.660] | 0.672 [0.588, 0.741] | 0.610 [0.518, 0.687] | 0.509 [0.414, 0.593] | 0.293 [0.193, 0.378] |
| SR | con_perturbazione down8k | now_metrical | 2931 | 0.769 [0.712, 0.820] | 0.782 [0.727, 0.831] | 0.769 [0.710, 0.820] | 0.768 [0.710, 0.819] | 0.768 [0.710, 0.819] | 0.765 [0.708, 0.817] |
| SR | con_perturbazione down8k | now_original | 2931 | 0.794 [0.740, 0.843] | 0.799 [0.745, 0.845] | 0.793 [0.739, 0.843] | 0.792 [0.738, 0.842] | 0.791 [0.735, 0.839] | 0.787 [0.731, 0.835] |
| SR | con_perturbazione down8k | realy | 2931 | 0.737 [0.663, 0.800] | 0.577 [0.498, 0.654] | 0.712 [0.633, 0.778] | 0.604 [0.514, 0.683] | 0.468 [0.370, 0.559] | 0.284 [0.189, 0.375] |
| SR | con_perturbazione down8k | stirling | 2931 | 0.721 [0.639, 0.790] | 0.614 [0.516, 0.697] | 0.718 [0.641, 0.786] | 0.661 [0.576, 0.734] | 0.556 [0.459, 0.640] | 0.332 [0.228, 0.430] |
| SR | con_perturbazione up60k | now_metrical | 2306 | 0.735 [0.669, 0.791] | 0.732 [0.667, 0.789] | 0.734 [0.667, 0.790] | 0.734 [0.669, 0.790] | 0.734 [0.670, 0.790] | 0.732 [0.667, 0.788] |
| SR | con_perturbazione up60k | now_original | 2306 | 0.748 [0.686, 0.804] | 0.741 [0.676, 0.798] | 0.748 [0.684, 0.803] | 0.747 [0.683, 0.802] | 0.746 [0.683, 0.802] | 0.744 [0.678, 0.799] |
| SR | con_perturbazione up60k | realy | 2306 | 0.709 [0.629, 0.778] | 0.556 [0.468, 0.642] | 0.687 [0.605, 0.760] | 0.602 [0.504, 0.684] | 0.461 [0.364, 0.549] | 0.261 [0.157, 0.346] |
| SR | con_perturbazione up60k | stirling | 2306 | 0.689 [0.605, 0.762] | 0.604 [0.503, 0.696] | 0.682 [0.591, 0.760] | 0.626 [0.535, 0.710] | 0.532 [0.431, 0.621] | 0.332 [0.234, 0.430] |
| SR | con_perturbazione noisy | now_metrical | 2965 | 0.720 [0.653, 0.778] | 0.731 [0.662, 0.785] | 0.720 [0.654, 0.778] | 0.720 [0.654, 0.778] | 0.721 [0.654, 0.779] | 0.719 [0.654, 0.776] |
| SR | con_perturbazione noisy | now_original | 2965 | 0.736 [0.669, 0.792] | 0.741 [0.677, 0.796] | 0.736 [0.670, 0.792] | 0.736 [0.669, 0.792] | 0.736 [0.670, 0.792] | 0.734 [0.668, 0.789] |
| SR | con_perturbazione noisy | realy | 2965 | 0.721 [0.642, 0.785] | 0.502 [0.403, 0.598] | 0.678 [0.595, 0.748] | 0.576 [0.485, 0.659] | 0.440 [0.343, 0.530] | 0.259 [0.180, 0.339] |
| SR | con_perturbazione noisy | stirling | 2965 | 0.680 [0.600, 0.747] | 0.469 [0.360, 0.564] | 0.665 [0.586, 0.730] | 0.602 [0.518, 0.675] | 0.513 [0.424, 0.588] | 0.322 [0.227, 0.404] |

## Varianti non dichiarate (passo `varianti non dichiarate`, descrittivo)

Stesse righe della maschera comune e stesse repliche; mai nel miglior concorrente ne' nella scala. GW `euclid_256_cgw2` e `euclid_256_cg`, percettivi nelle varianti di render non scelte dal pilota, ULRSSM `faces_wks` e `zeroshot_faust`. I delta `vs_nostro` sono in `deltas.csv` con lo stesso `passo`.

GT FR:

| metodo | hifi3d nocrop | hifi3d stessa_topologia senza_crop | facescape nocrop | facescape stessa_topologia senza_crop | faceverse nocrop | faceverse stessa_topologia senza_crop | faceverse_neutral nocrop | faceverse_neutral stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), mm (FR) | 0.149 [0.106, 0.195] | 0.341 [0.259, 0.428] | 0.091 [0.059, 0.126] | 0.257 [0.171, 0.350] | 0.023 [-0.002, 0.046] | 0.052 [0.004, 0.094] | 0.035 [0.009, 0.061] | 0.070 [0.016, 0.126] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.134 [0.089, 0.181] | 0.333 [0.239, 0.431] | 0.100 [0.062, 0.144] | 0.275 [0.195, 0.363] | 0.034 [0.009, 0.060] | 0.070 [0.024, 0.115] | 0.043 [0.018, 0.071] | 0.106 [0.054, 0.161] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.063 [0.037, 0.091] | 0.334 [0.239, 0.434] | 0.045 [0.025, 0.068] | 0.311 [0.220, 0.411] | 0.026 [0.008, 0.044] | 0.100 [0.043, 0.155] | 0.036 [0.019, 0.054] | 0.155 [0.082, 0.222] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.084 [0.055, 0.115] | 0.360 [0.257, 0.468] | 0.060 [0.037, 0.087] | 0.329 [0.229, 0.435] | 0.046 [0.019, 0.071] | 0.132 [0.069, 0.188] | 0.067 [0.040, 0.094] | 0.196 [0.112, 0.276] |
| DreamSim ensemble su render [profondita'] (non dichiarata), mm (FR) | 0.187 [0.143, 0.233] | 0.374 [0.282, 0.460] | 0.182 [0.146, 0.222] | 0.459 [0.383, 0.545] | 0.041 [0.009, 0.071] | 0.052 [0.014, 0.088] | 0.076 [0.026, 0.126] | 0.088 [0.031, 0.152] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.155 [0.109, 0.201] | 0.420 [0.312, 0.522] | 0.116 [0.086, 0.150] | 0.397 [0.321, 0.481] | 0.048 [0.025, 0.072] | 0.106 [0.058, 0.154] | 0.082 [0.050, 0.110] | 0.187 [0.116, 0.249] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | 0.434 [0.354, 0.504] | - | 0.155 [0.131, 0.177] | - | 0.112 [0.065, 0.158] | - | 0.141 [0.089, 0.189] | - |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | 0.471 [0.388, 0.546] | - | 0.187 [0.161, 0.210] | - | 0.205 [0.136, 0.267] | - | 0.249 [0.171, 0.320] | - |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.355 [0.316, 0.389] | 0.738 [0.666, 0.798] | 0.173 [0.137, 0.205] | 0.456 [0.373, 0.534] | 0.091 [0.063, 0.115] | 0.267 [0.198, 0.327] | 0.136 [0.103, 0.166] | 0.401 [0.314, 0.481] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.416 [0.361, 0.465] | 0.765 [0.695, 0.823] | 0.216 [0.177, 0.253] | 0.517 [0.427, 0.599] | 0.112 [0.080, 0.140] | 0.277 [0.209, 0.336] | 0.166 [0.126, 0.201] | 0.408 [0.320, 0.487] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, mm (FR) | 0.082 [0.034, 0.129] | 0.232 [0.137, 0.325] | 0.079 [0.047, 0.111] | 0.191 [0.127, 0.262] | 0.086 [0.049, 0.127] | 0.116 [0.069, 0.166] | 0.070 [0.027, 0.117] | 0.107 [0.055, 0.161] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT FR (FR) | 0.033 [-0.039, 0.112] | 0.094 [0.020, 0.180] | 0.054 [-0.025, 0.142] | 0.094 [0.010, 0.190] | 0.074 [0.024, 0.126] | 0.128 [0.052, 0.202] | 0.046 [-0.005, 0.101] | 0.092 [0.007, 0.179] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, mm (FR) | 0.156 [0.124, 0.187] | - | 0.137 [0.103, 0.171] | - | 0.075 [0.036, 0.114] | - | 0.092 [0.049, 0.136] | 0.159 [0.096, 0.220] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT FR (FR) | 0.085 [0.026, 0.140] | 0.131 [0.059, 0.208] | 0.050 [0.008, 0.092] | 0.060 [0.017, 0.105] | 0.087 [0.026, 0.146] | 0.145 [0.067, 0.209] | 0.089 [0.021, 0.150] | 0.135 [0.044, 0.213] |

GT SR:

| metodo | hifi3d nocrop | hifi3d stessa_topologia senza_crop | facescape nocrop | facescape stessa_topologia senza_crop | faceverse nocrop | faceverse stessa_topologia senza_crop | faceverse_neutral nocrop | faceverse_neutral stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.143 [0.106, 0.188] | 0.333 [0.254, 0.426] | 0.105 [0.074, 0.141] | 0.298 [0.216, 0.388] | 0.021 [-0.006, 0.044] | 0.048 [-0.002, 0.091] | 0.040 [0.015, 0.066] | 0.083 [0.027, 0.140] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.152 [0.114, 0.191] | 0.396 [0.313, 0.483] | 0.091 [0.052, 0.136] | 0.272 [0.193, 0.364] | 0.034 [0.012, 0.056] | 0.084 [0.037, 0.129] | 0.055 [0.028, 0.086] | 0.141 [0.094, 0.194] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.077 [0.050, 0.107] | 0.414 [0.333, 0.497] | 0.046 [0.027, 0.067] | 0.334 [0.243, 0.437] | 0.022 [0.004, 0.040] | 0.105 [0.049, 0.158] | 0.035 [0.019, 0.050] | 0.173 [0.107, 0.238] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.105 [0.080, 0.130] | 0.451 [0.373, 0.530] | 0.063 [0.039, 0.091] | 0.342 [0.242, 0.446] | 0.047 [0.020, 0.071] | 0.139 [0.078, 0.194] | 0.072 [0.046, 0.099] | 0.204 [0.128, 0.276] |
| DreamSim ensemble su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.215 [0.174, 0.264] | 0.438 [0.350, 0.536] | 0.179 [0.142, 0.219] | 0.462 [0.382, 0.549] | 0.047 [0.015, 0.076] | 0.055 [0.017, 0.090] | 0.087 [0.037, 0.139] | 0.100 [0.040, 0.162] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.165 [0.128, 0.205] | 0.471 [0.384, 0.561] | 0.114 [0.086, 0.147] | 0.421 [0.338, 0.502] | 0.047 [0.024, 0.070] | 0.115 [0.067, 0.162] | 0.091 [0.059, 0.121] | 0.223 [0.156, 0.284] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | 0.224 [0.186, 0.259] | - | 0.187 [0.158, 0.216] | - | 0.091 [0.056, 0.123] | - | 0.121 [0.079, 0.158] | - |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | 0.263 [0.225, 0.303] | - | 0.239 [0.210, 0.267] | - | 0.191 [0.132, 0.239] | - | 0.243 [0.180, 0.297] | - |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.259 [0.230, 0.289] | 0.552 [0.491, 0.613] | 0.172 [0.139, 0.204] | 0.440 [0.367, 0.512] | 0.087 [0.060, 0.111] | 0.253 [0.178, 0.315] | 0.130 [0.099, 0.160] | 0.375 [0.290, 0.452] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.284 [0.253, 0.316] | 0.621 [0.557, 0.686] | 0.220 [0.179, 0.257] | 0.506 [0.424, 0.585] | 0.111 [0.078, 0.141] | 0.267 [0.193, 0.331] | 0.157 [0.120, 0.191] | 0.381 [0.294, 0.459] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, a CS_ref (SR) | 0.102 [0.063, 0.140] | 0.279 [0.198, 0.358] | 0.082 [0.054, 0.112] | 0.210 [0.151, 0.274] | 0.095 [0.057, 0.135] | 0.128 [0.080, 0.178] | 0.082 [0.042, 0.124] | 0.119 [0.064, 0.175] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT SR (d_P) (SR) | 0.049 [-0.018, 0.119] | 0.126 [0.054, 0.203] | 0.033 [-0.027, 0.095] | 0.084 [0.013, 0.154] | 0.088 [0.038, 0.140] | 0.141 [0.068, 0.211] | 0.057 [0.005, 0.111] | 0.100 [0.017, 0.182] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, a CS_ref (SR) | 0.178 [0.145, 0.210] | - | 0.143 [0.109, 0.176] | - | 0.086 [0.047, 0.122] | - | 0.101 [0.059, 0.144] | 0.175 [0.113, 0.235] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT SR (d_P) (SR) | 0.123 [0.080, 0.164] | 0.218 [0.148, 0.290] | 0.042 [-0.002, 0.091] | 0.073 [0.028, 0.124] | 0.091 [0.039, 0.139] | 0.167 [0.099, 0.228] | 0.116 [0.057, 0.172] | 0.178 [0.091, 0.256] |

## Tabelle per dominio (famiglie sintetiche, senza crop)

### HIFI3D: Spearman per insieme di righe (seme 990708)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (98224) | con_perturbazione original (39406) | con_perturbazione remesh (39406) | con_perturbazione down8k (39406) | con_perturbazione up60k (38824) | con_perturbazione noisy (39406) | stessa_topologia senza_crop (24750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.749 [0.673, 0.806] | 0.754 [0.679, 0.811] | 0.741 [0.668, 0.798] | 0.752 [0.678, 0.808] | 0.751 [0.674, 0.808] | 0.747 [0.670, 0.806] | 0.760 [0.685, 0.817] |
| **factorized s1234, d_F non cal.** | 0.642 [0.570, 0.706] | 0.649 [0.577, 0.712] | 0.642 [0.571, 0.705] | 0.644 [0.572, 0.707] | 0.646 [0.573, 0.710] | 0.633 [0.560, 0.698] | 0.653 [0.582, 0.717] |
| factorized s1234, d_P | 0.426 [0.333, 0.510] | 0.432 [0.339, 0.517] | 0.429 [0.337, 0.514] | 0.427 [0.335, 0.510] | 0.429 [0.335, 0.513] | 0.413 [0.317, 0.501] | 0.434 [0.341, 0.518] |
| **factorized s2345, d_F cal.*** | 0.731 [0.657, 0.792] | 0.737 [0.663, 0.799] | 0.717 [0.645, 0.778] | 0.738 [0.663, 0.798] | 0.731 [0.659, 0.793] | 0.731 [0.657, 0.793] | 0.744 [0.671, 0.805] |
| **factorized s2345, d_F non cal.** | 0.631 [0.560, 0.697] | 0.636 [0.565, 0.701] | 0.629 [0.559, 0.694] | 0.634 [0.564, 0.700] | 0.634 [0.562, 0.699] | 0.623 [0.551, 0.689] | 0.639 [0.570, 0.705] |
| factorized s2345, d_P | 0.417 [0.315, 0.512] | 0.421 [0.319, 0.517] | 0.419 [0.317, 0.513] | 0.420 [0.318, 0.515] | 0.418 [0.315, 0.513] | 0.409 [0.306, 0.501] | 0.423 [0.319, 0.518] |
| **ctrlfr s1234** | 0.757 [0.685, 0.817] | 0.767 [0.696, 0.828] | 0.747 [0.673, 0.805] | 0.766 [0.693, 0.827] | 0.755 [0.680, 0.815] | 0.753 [0.683, 0.814] | 0.779 [0.707, 0.840] |
| **ctrlfr s2345** | 0.746 [0.670, 0.809] | 0.757 [0.682, 0.820] | 0.732 [0.656, 0.795] | 0.755 [0.680, 0.817] | 0.754 [0.676, 0.817] | 0.733 [0.654, 0.798] | 0.765 [0.688, 0.827] |
| **C3M e123, d_F cal.*** | 0.748 [0.672, 0.808] | 0.759 [0.683, 0.817] | 0.732 [0.655, 0.792] | 0.756 [0.680, 0.815] | 0.750 [0.671, 0.809] | 0.747 [0.670, 0.806] | 0.767 [0.691, 0.825] |
| **C3M e123, d_F non cal.** | 0.528 [0.444, 0.606] | 0.535 [0.451, 0.611] | 0.531 [0.447, 0.609] | 0.534 [0.452, 0.612] | 0.529 [0.442, 0.606] | 0.519 [0.436, 0.600] | 0.545 [0.460, 0.620] |
| C3M e123, d_P | 0.400 [0.304, 0.492] | 0.406 [0.310, 0.496] | 0.404 [0.307, 0.494] | 0.406 [0.310, 0.495] | 0.400 [0.304, 0.491] | 0.390 [0.295, 0.479] | 0.414 [0.318, 0.502] |
| **C3M e205, d_F cal.*** | 0.739 [0.657, 0.801] | 0.750 [0.669, 0.810] | 0.725 [0.644, 0.786] | 0.745 [0.663, 0.806] | 0.736 [0.653, 0.799] | 0.744 [0.663, 0.805] | 0.757 [0.676, 0.818] |
| **C3M e205, d_F non cal.** | 0.508 [0.418, 0.589] | 0.516 [0.427, 0.598] | 0.511 [0.421, 0.593] | 0.514 [0.423, 0.596] | 0.509 [0.418, 0.589] | 0.499 [0.410, 0.581] | 0.525 [0.436, 0.605] |
| C3M e205, d_P | 0.404 [0.306, 0.496] | 0.412 [0.312, 0.505] | 0.408 [0.309, 0.498] | 0.411 [0.312, 0.505] | 0.404 [0.303, 0.497] | 0.393 [0.297, 0.484] | 0.419 [0.320, 0.512] |
| **ICP + Chamfer in mm (per coppia)** | 0.643 [0.572, 0.702] | 0.637 [0.565, 0.696] | 0.668 [0.596, 0.729] | 0.621 [0.552, 0.680] | 0.666 [0.595, 0.727] | 0.657 [0.587, 0.715] | 0.659 [0.588, 0.719] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.358 [0.276, 0.438] | 0.346 [0.267, 0.423] | 0.379 [0.292, 0.463] | 0.345 [0.264, 0.422] | 0.377 [0.289, 0.460] | 0.368 [0.286, 0.448] | 0.384 [0.299, 0.467] |
| **NICP per coppia in mm** | 0.495 [0.423, 0.569] | 0.475 [0.404, 0.545] | 0.515 [0.439, 0.592] | 0.486 [0.417, 0.558] | 0.512 [0.437, 0.588] | 0.508 [0.436, 0.580] | - |
| **NICP per coppia cs** | 0.376 [0.296, 0.465] | 0.356 [0.281, 0.443] | 0.391 [0.310, 0.484] | 0.371 [0.291, 0.459] | 0.389 [0.306, 0.482] | 0.382 [0.303, 0.474] | - |
| **NICP su template in mm** | 0.614 [0.506, 0.698] | 0.616 [0.508, 0.698] | 0.616 [0.508, 0.699] | 0.608 [0.499, 0.693] | 0.620 [0.512, 0.702] | 0.612 [0.504, 0.696] | 0.616 [0.508, 0.700] |
| **NICP su template cs** | 0.189 [0.103, 0.269] | 0.191 [0.104, 0.271] | 0.191 [0.108, 0.266] | 0.188 [0.100, 0.269] | 0.194 [0.107, 0.276] | 0.181 [0.097, 0.262] | 0.201 [0.108, 0.288] |
| **varifold in mm (massa unitaria)** | 0.494 [0.411, 0.567] | 0.550 [0.468, 0.620] | 0.550 [0.468, 0.619] | 0.551 [0.469, 0.621] | 0.551 [0.469, 0.621] | 0.562 [0.477, 0.630] | 0.693 [0.618, 0.756] |
| **GNM (visto) preregistrato, coefficienti** | 0.619 [0.503, 0.710] | 0.623 [0.510, 0.713] | 0.618 [0.502, 0.709] | 0.617 [0.501, 0.707] | 0.624 [0.510, 0.715] | 0.612 [0.498, 0.702] | 0.622 [0.507, 0.713] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.608 [0.500, 0.693] | 0.612 [0.505, 0.696] | 0.608 [0.499, 0.692] | 0.607 [0.498, 0.692] | 0.610 [0.502, 0.695] | 0.602 [0.492, 0.688] | 0.609 [0.501, 0.694] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.482 [0.369, 0.582] | 0.478 [0.363, 0.577] | 0.481 [0.368, 0.582] | 0.480 [0.367, 0.582] | 0.488 [0.375, 0.588] | 0.485 [0.369, 0.583] | 0.486 [0.371, 0.586] |
| **GNM (visto) vA, coefficienti*** | 0.603 [0.492, 0.693] | 0.606 [0.496, 0.696] | 0.604 [0.496, 0.693] | 0.601 [0.491, 0.690] | 0.610 [0.499, 0.698] | 0.597 [0.483, 0.686] | 0.606 [0.495, 0.696] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.607 [0.500, 0.692] | 0.611 [0.504, 0.697] | 0.607 [0.499, 0.692] | 0.606 [0.497, 0.691] | 0.609 [0.503, 0.694] | 0.601 [0.492, 0.688] | 0.608 [0.501, 0.693] |
| GNM (visto) vA, mesh d'identita' SR* | 0.291 [0.182, 0.389] | 0.289 [0.180, 0.392] | 0.293 [0.185, 0.396] | 0.290 [0.179, 0.388] | 0.300 [0.191, 0.398] | 0.284 [0.180, 0.381] | 0.294 [0.185, 0.393] |
| **GNM (visto) vB, coefficienti*** | 0.550 [0.465, 0.628] | 0.551 [0.466, 0.630] | 0.551 [0.468, 0.629] | 0.547 [0.462, 0.626] | 0.547 [0.463, 0.626] | 0.554 [0.468, 0.632] | 0.549 [0.465, 0.627] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.664 [0.570, 0.738] | 0.668 [0.574, 0.741] | 0.662 [0.567, 0.736] | 0.665 [0.570, 0.738] | 0.667 [0.573, 0.741] | 0.661 [0.566, 0.735] | 0.666 [0.571, 0.740] |
| GNM (visto) vB, mesh d'identita' SR* | 0.411 [0.319, 0.499] | 0.412 [0.319, 0.500] | 0.411 [0.317, 0.501] | 0.409 [0.318, 0.497] | 0.410 [0.318, 0.499] | 0.410 [0.319, 0.500] | 0.411 [0.319, 0.500] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.698 [0.612, 0.769] | 0.701 [0.616, 0.771] | 0.696 [0.608, 0.768] | 0.698 [0.612, 0.769] | 0.700 [0.615, 0.771] | 0.696 [0.611, 0.767] | 0.699 [0.613, 0.771] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.566 [0.444, 0.663] | 0.570 [0.449, 0.670] | 0.557 [0.435, 0.658] | 0.572 [0.447, 0.672] | 0.578 [0.457, 0.675] | 0.554 [0.438, 0.650] | 0.580 [0.459, 0.679] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.639 [0.532, 0.725] | 0.644 [0.539, 0.730] | 0.635 [0.522, 0.723] | 0.644 [0.535, 0.730] | 0.645 [0.538, 0.730] | 0.627 [0.518, 0.714] | 0.645 [0.537, 0.730] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.276 [0.178, 0.379] | 0.277 [0.173, 0.382] | 0.286 [0.184, 0.393] | 0.279 [0.177, 0.388] | 0.288 [0.185, 0.397] | 0.253 [0.164, 0.344] | 0.289 [0.185, 0.397] |
| **FLAME 2023 Open vA, coefficienti*** | 0.580 [0.460, 0.676] | 0.585 [0.464, 0.684] | 0.572 [0.451, 0.671] | 0.586 [0.464, 0.685] | 0.592 [0.472, 0.687] | 0.568 [0.447, 0.663] | 0.593 [0.472, 0.691] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.641 [0.536, 0.725] | 0.646 [0.539, 0.730] | 0.639 [0.528, 0.724] | 0.645 [0.539, 0.730] | 0.646 [0.542, 0.729] | 0.630 [0.522, 0.715] | 0.645 [0.540, 0.730] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.272 [0.173, 0.373] | 0.274 [0.170, 0.378] | 0.284 [0.184, 0.387] | 0.276 [0.172, 0.383] | 0.283 [0.180, 0.387] | 0.247 [0.158, 0.337] | 0.283 [0.182, 0.386] |
| **FLAME 2023 Open vB, coefficienti*** | 0.585 [0.503, 0.661] | 0.582 [0.499, 0.660] | 0.581 [0.500, 0.658] | 0.584 [0.502, 0.661] | 0.589 [0.507, 0.665] | 0.592 [0.509, 0.668] | 0.583 [0.501, 0.658] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.686 [0.597, 0.758] | 0.690 [0.601, 0.762] | 0.682 [0.593, 0.755] | 0.690 [0.599, 0.762] | 0.689 [0.600, 0.759] | 0.679 [0.589, 0.754] | 0.690 [0.599, 0.762] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.458 [0.365, 0.542] | 0.459 [0.365, 0.543] | 0.461 [0.367, 0.545] | 0.454 [0.359, 0.541] | 0.459 [0.365, 0.544] | 0.459 [0.366, 0.544] | 0.459 [0.367, 0.544] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.712 [0.627, 0.781] | 0.715 [0.631, 0.784] | 0.708 [0.622, 0.778] | 0.713 [0.627, 0.783] | 0.715 [0.632, 0.784] | 0.706 [0.621, 0.776] | 0.715 [0.629, 0.784] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.134 [0.042, 0.233] | 0.133 [0.042, 0.236] | 0.140 [0.049, 0.239] | 0.138 [0.047, 0.239] | 0.136 [0.045, 0.238] | 0.127 [0.028, 0.237] | 0.112 [0.043, 0.189] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.166 [0.079, 0.261] | 0.174 [0.081, 0.272] | 0.175 [0.083, 0.272] | 0.174 [0.079, 0.276] | 0.175 [0.080, 0.276] | 0.148 [0.067, 0.243] | 0.143 [0.071, 0.219] |
| **ShapeDNA k=50** | 0.085 [0.051, 0.118] | 0.143 [0.090, 0.193] | 0.136 [0.086, 0.185] | 0.134 [0.083, 0.183] | 0.121 [0.076, 0.166] | 0.003 [-0.042, 0.049] | 0.259 [0.171, 0.350] |
| **HKS globale** | 0.036 [0.006, 0.070] | 0.059 [0.011, 0.113] | 0.050 [0.009, 0.095] | 0.058 [0.009, 0.111] | 0.052 [0.010, 0.101] | 0.011 [-0.030, 0.053] | 0.113 [0.027, 0.204] |
| **WKS globale** | 0.019 [0.003, 0.036] | 0.043 [0.017, 0.071] | 0.044 [0.019, 0.070] | 0.037 [0.012, 0.061] | 0.027 [0.006, 0.049] | -0.004 [-0.053, 0.048] | 0.207 [0.116, 0.299] |
| **OpenShape PointBERT** | 0.128 [0.083, 0.178] | 0.180 [0.118, 0.244] | 0.159 [0.099, 0.221] | 0.173 [0.112, 0.236] | 0.164 [0.102, 0.227] | 0.152 [0.067, 0.245] | 0.262 [0.169, 0.357] |
| **Uni3D-g** | 0.101 [0.048, 0.153] | 0.139 [0.065, 0.211] | 0.144 [0.076, 0.211] | 0.152 [0.081, 0.221] | 0.143 [0.069, 0.215] | 0.075 [-0.010, 0.160] | 0.269 [0.149, 0.383] |
| **taglia stimata dalla mesh (non oracolo)** | 0.600 [0.498, 0.690] | 0.602 [0.498, 0.692] | 0.587 [0.485, 0.678] | 0.601 [0.498, 0.692] | 0.605 [0.501, 0.695] | 0.605 [0.503, 0.695] | 0.616 [0.513, 0.707] |
| **taglia oracolo (S di FR)** | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.737 [0.656, 0.803] | 0.737 [0.657, 0.802] | 0.736 [0.658, 0.802] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.151 [0.106, 0.197] | 0.216 [0.152, 0.279] | 0.197 [0.139, 0.253] | 0.212 [0.148, 0.273] | 0.210 [0.148, 0.271] | 0.151 [0.054, 0.246] | 0.351 [0.248, 0.454] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.118 [0.075, 0.166] | 0.173 [0.113, 0.237] | 0.155 [0.101, 0.213] | 0.168 [0.108, 0.232] | 0.168 [0.109, 0.231] | 0.108 [0.015, 0.200] | 0.279 [0.181, 0.383] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.119 [0.085, 0.151] | 0.189 [0.135, 0.238] | 0.161 [0.110, 0.209] | 0.149 [0.103, 0.195] | 0.154 [0.110, 0.197] | 0.140 [0.074, 0.201] | 0.368 [0.269, 0.460] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.100 [0.066, 0.134] | 0.148 [0.096, 0.204] | 0.133 [0.085, 0.184] | 0.122 [0.083, 0.166] | 0.123 [0.080, 0.168] | 0.157 [0.088, 0.227] | 0.275 [0.182, 0.382] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.182 [0.135, 0.228] | 0.254 [0.188, 0.317] | 0.231 [0.170, 0.287] | 0.246 [0.181, 0.308] | 0.248 [0.183, 0.310] | 0.228 [0.143, 0.316] | 0.432 [0.320, 0.538] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.145 [0.099, 0.194] | 0.204 [0.140, 0.271] | 0.183 [0.124, 0.244] | 0.198 [0.135, 0.264] | 0.200 [0.137, 0.265] | 0.181 [0.097, 0.269] | 0.348 [0.241, 0.461] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.466 [0.382, 0.542] | 0.536 [0.451, 0.612] | 0.551 [0.463, 0.629] | 0.534 [0.448, 0.610] | 0.536 [0.449, 0.612] | 0.282 [0.208, 0.354] | - |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.153 [0.103, 0.200] | 0.205 [0.141, 0.266] | 0.207 [0.142, 0.266] | 0.206 [0.144, 0.266] | 0.203 [0.140, 0.262] | 0.185 [0.092, 0.276] | - |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.615 [0.531, 0.688] | 0.639 [0.556, 0.711] | 0.628 [0.545, 0.698] | 0.645 [0.562, 0.718] | 0.639 [0.555, 0.710] | 0.648 [0.563, 0.724] | 0.713 [0.635, 0.781] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.231 [0.171, 0.291] | 0.262 [0.196, 0.330] | 0.259 [0.193, 0.326] | 0.268 [0.200, 0.337] | 0.263 [0.197, 0.330] | 0.245 [0.179, 0.317] | 0.349 [0.267, 0.434] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.697 [0.628, 0.753] | 0.702 [0.634, 0.758] | 0.698 [0.628, 0.755] | 0.710 [0.640, 0.765] | 0.700 [0.631, 0.756] | 0.719 [0.651, 0.775] | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.416 [0.336, 0.495] | 0.419 [0.340, 0.500] | 0.415 [0.335, 0.495] | 0.423 [0.340, 0.501] | 0.415 [0.334, 0.494] | 0.432 [0.351, 0.513] | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.495 [0.402, 0.575] | 0.498 [0.406, 0.579] | 0.498 [0.406, 0.577] | 0.493 [0.400, 0.574] | 0.498 [0.404, 0.578] | 0.498 [0.405, 0.578] | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.463 [0.375, 0.548] | 0.471 [0.381, 0.556] | 0.465 [0.377, 0.550] | 0.464 [0.374, 0.548] | 0.463 [0.374, 0.548] | 0.475 [0.384, 0.562] | - |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.691 [0.609, 0.757] | 0.704 [0.624, 0.769] | 0.710 [0.631, 0.775] | 0.705 [0.626, 0.770] | 0.710 [0.631, 0.775] | 0.704 [0.633, 0.763] | 0.780 [0.710, 0.837] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.297 [0.215, 0.367] | 0.317 [0.231, 0.390] | 0.320 [0.232, 0.393] | 0.324 [0.236, 0.399] | 0.320 [0.231, 0.393] | 0.387 [0.288, 0.475] | 0.411 [0.311, 0.501] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.629 [0.547, 0.697] | 0.649 [0.564, 0.721] | 0.657 [0.573, 0.729] | 0.667 [0.589, 0.730] | 0.658 [0.572, 0.731] | 0.557 [0.482, 0.619] | 0.752 [0.671, 0.818] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.248 [0.163, 0.334] | 0.245 [0.157, 0.334] | 0.232 [0.145, 0.322] | 0.287 [0.200, 0.375] | 0.246 [0.155, 0.339] | 0.302 [0.206, 0.388] | 0.331 [0.241, 0.424] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (98224) | con_perturbazione original (39406) | con_perturbazione remesh (39406) | con_perturbazione down8k (39406) | con_perturbazione up60k (38824) | con_perturbazione noisy (39406) | stessa_topologia senza_crop (24750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.318 [0.227, 0.407] | 0.322 [0.232, 0.411] | 0.320 [0.230, 0.408] | 0.320 [0.229, 0.410] | 0.319 [0.229, 0.407] | 0.310 [0.217, 0.403] | 0.325 [0.233, 0.415] |
| factorized s1234, d_F non cal. | 0.551 [0.463, 0.630] | 0.558 [0.470, 0.636] | 0.555 [0.468, 0.634] | 0.554 [0.466, 0.634] | 0.554 [0.465, 0.633] | 0.535 [0.446, 0.613] | 0.563 [0.474, 0.642] |
| **factorized s1234, d_P** | 0.622 [0.550, 0.685] | 0.629 [0.558, 0.692] | 0.626 [0.553, 0.690] | 0.625 [0.555, 0.688] | 0.626 [0.554, 0.690] | 0.604 [0.533, 0.668] | 0.634 [0.562, 0.699] |
| factorized s2345, d_F cal.* | 0.312 [0.220, 0.401] | 0.315 [0.223, 0.405] | 0.311 [0.220, 0.398] | 0.316 [0.222, 0.406] | 0.312 [0.220, 0.401] | 0.308 [0.216, 0.397] | 0.317 [0.224, 0.409] |
| factorized s2345, d_F non cal. | 0.539 [0.445, 0.621] | 0.543 [0.450, 0.626] | 0.541 [0.448, 0.622] | 0.543 [0.449, 0.625] | 0.541 [0.446, 0.622] | 0.529 [0.433, 0.610] | 0.547 [0.453, 0.629] |
| **factorized s2345, d_P** | 0.613 [0.533, 0.688] | 0.617 [0.537, 0.691] | 0.616 [0.537, 0.690] | 0.616 [0.537, 0.690] | 0.616 [0.537, 0.689] | 0.600 [0.519, 0.675] | 0.620 [0.541, 0.693] |
| ctrlfr s1234 | 0.339 [0.242, 0.425] | 0.344 [0.245, 0.432] | 0.339 [0.243, 0.425] | 0.342 [0.243, 0.430] | 0.335 [0.236, 0.422] | 0.334 [0.237, 0.418] | 0.347 [0.248, 0.436] |
| ctrlfr s2345 | 0.348 [0.251, 0.435] | 0.354 [0.255, 0.442] | 0.346 [0.251, 0.432] | 0.351 [0.254, 0.439] | 0.349 [0.251, 0.437] | 0.339 [0.246, 0.425] | 0.356 [0.258, 0.445] |
| C3M e123, d_F cal.* | 0.292 [0.192, 0.381] | 0.297 [0.196, 0.387] | 0.289 [0.191, 0.377] | 0.293 [0.194, 0.384] | 0.291 [0.191, 0.382] | 0.291 [0.193, 0.381] | 0.300 [0.200, 0.391] |
| C3M e123, d_F non cal. | 0.555 [0.462, 0.641] | 0.565 [0.471, 0.650] | 0.558 [0.464, 0.643] | 0.558 [0.466, 0.644] | 0.557 [0.461, 0.646] | 0.547 [0.453, 0.632] | 0.574 [0.482, 0.660] |
| **C3M e123, d_P** | 0.595 [0.511, 0.671] | 0.604 [0.519, 0.681] | 0.599 [0.515, 0.675] | 0.598 [0.516, 0.672] | 0.599 [0.513, 0.676] | 0.585 [0.499, 0.665] | 0.613 [0.529, 0.689] |
| C3M e205, d_F cal.* | 0.286 [0.182, 0.376] | 0.291 [0.185, 0.381] | 0.284 [0.181, 0.372] | 0.287 [0.180, 0.378] | 0.283 [0.179, 0.376] | 0.289 [0.185, 0.377] | 0.295 [0.189, 0.385] |
| C3M e205, d_F non cal. | 0.556 [0.461, 0.642] | 0.566 [0.471, 0.654] | 0.560 [0.464, 0.645] | 0.558 [0.461, 0.644] | 0.558 [0.460, 0.648] | 0.552 [0.455, 0.639] | 0.578 [0.484, 0.665] |
| **C3M e205, d_P** | 0.590 [0.507, 0.669] | 0.600 [0.517, 0.680] | 0.594 [0.511, 0.673] | 0.592 [0.508, 0.671] | 0.594 [0.508, 0.674] | 0.585 [0.500, 0.662] | 0.611 [0.530, 0.689] |
| **ICP + Chamfer in mm (per coppia)** | 0.365 [0.280, 0.452] | 0.367 [0.285, 0.454] | 0.382 [0.295, 0.474] | 0.344 [0.261, 0.428] | 0.375 [0.286, 0.466] | 0.377 [0.292, 0.465] | 0.377 [0.292, 0.466] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.580 [0.529, 0.631] | 0.557 [0.507, 0.606] | 0.617 [0.564, 0.667] | 0.558 [0.504, 0.608] | 0.614 [0.561, 0.667] | 0.594 [0.545, 0.643] | 0.623 [0.571, 0.673] |
| **NICP per coppia in mm** | 0.503 [0.437, 0.574] | 0.477 [0.414, 0.545] | 0.532 [0.464, 0.603] | 0.498 [0.431, 0.569] | 0.522 [0.456, 0.596] | 0.503 [0.437, 0.573] | - |
| **NICP per coppia cs** | 0.595 [0.546, 0.648] | 0.562 [0.514, 0.613] | 0.621 [0.571, 0.676] | 0.585 [0.534, 0.640] | 0.619 [0.567, 0.673] | 0.606 [0.558, 0.657] | - |
| **NICP su template in mm** | 0.083 [-0.018, 0.185] | 0.083 [-0.017, 0.186] | 0.083 [-0.018, 0.185] | 0.081 [-0.019, 0.183] | 0.086 [-0.017, 0.187] | 0.079 [-0.021, 0.182] | 0.084 [-0.018, 0.186] |
| **NICP su template cs** | 0.301 [0.200, 0.393] | 0.304 [0.202, 0.394] | 0.299 [0.200, 0.385] | 0.301 [0.197, 0.392] | 0.309 [0.203, 0.406] | 0.294 [0.192, 0.384] | 0.322 [0.213, 0.419] |
| **varifold in mm (massa unitaria)** | 0.195 [0.126, 0.261] | 0.223 [0.147, 0.293] | 0.223 [0.148, 0.292] | 0.223 [0.147, 0.293] | 0.222 [0.147, 0.292] | 0.254 [0.179, 0.328] | 0.303 [0.220, 0.386] |
| **GNM (visto) preregistrato, coefficienti** | 0.156 [0.047, 0.259] | 0.163 [0.052, 0.266] | 0.159 [0.046, 0.262] | 0.155 [0.046, 0.256] | 0.160 [0.048, 0.267] | 0.145 [0.040, 0.247] | 0.157 [0.046, 0.260] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.093 [-0.004, 0.189] | 0.098 [-0.001, 0.193] | 0.093 [-0.004, 0.187] | 0.093 [-0.005, 0.188] | 0.093 [-0.007, 0.189] | 0.088 [-0.010, 0.180] | 0.094 [-0.004, 0.188] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.325 [0.205, 0.441] | 0.326 [0.204, 0.441] | 0.328 [0.207, 0.442] | 0.326 [0.208, 0.439] | 0.329 [0.205, 0.444] | 0.315 [0.195, 0.427] | 0.328 [0.207, 0.445] |
| **GNM (visto) vA, coefficienti*** | 0.194 [0.088, 0.295] | 0.199 [0.092, 0.300] | 0.198 [0.093, 0.299] | 0.191 [0.086, 0.290] | 0.200 [0.093, 0.305] | 0.181 [0.077, 0.281] | 0.195 [0.090, 0.297] |
| GNM (visto) vA, mesh d'identita' FR* | 0.105 [0.006, 0.198] | 0.109 [0.011, 0.203] | 0.105 [0.007, 0.198] | 0.105 [0.007, 0.199] | 0.105 [0.005, 0.199] | 0.098 [-0.000, 0.193] | 0.105 [0.007, 0.199] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.339 [0.229, 0.443] | 0.338 [0.228, 0.446] | 0.342 [0.233, 0.451] | 0.339 [0.232, 0.444] | 0.346 [0.235, 0.453] | 0.333 [0.226, 0.437] | 0.343 [0.233, 0.450] |
| **GNM (visto) vB, coefficienti*** | 0.674 [0.610, 0.734] | 0.676 [0.610, 0.735] | 0.674 [0.607, 0.733] | 0.672 [0.606, 0.732] | 0.673 [0.607, 0.733] | 0.680 [0.614, 0.738] | 0.673 [0.607, 0.733] |
| GNM (visto) vB, mesh d'identita' FR* | 0.217 [0.127, 0.306] | 0.221 [0.130, 0.309] | 0.215 [0.126, 0.304] | 0.218 [0.129, 0.307] | 0.217 [0.125, 0.307] | 0.214 [0.126, 0.303] | 0.218 [0.129, 0.307] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.607 [0.532, 0.679] | 0.609 [0.535, 0.681] | 0.606 [0.531, 0.679] | 0.607 [0.530, 0.679] | 0.607 [0.532, 0.679] | 0.607 [0.533, 0.678] | 0.607 [0.532, 0.679] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.348 [0.257, 0.438] | 0.352 [0.261, 0.440] | 0.346 [0.255, 0.434] | 0.350 [0.259, 0.440] | 0.348 [0.257, 0.438] | 0.344 [0.256, 0.434] | 0.348 [0.258, 0.437] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.159 [0.044, 0.268] | 0.161 [0.048, 0.269] | 0.161 [0.044, 0.275] | 0.166 [0.049, 0.279] | 0.166 [0.051, 0.279] | 0.139 [0.035, 0.242] | 0.166 [0.051, 0.280] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.125 [0.025, 0.224] | 0.127 [0.026, 0.228] | 0.126 [0.026, 0.229] | 0.129 [0.026, 0.232] | 0.128 [0.028, 0.227] | 0.113 [0.017, 0.208] | 0.128 [0.028, 0.228] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.287 [0.177, 0.398] | 0.301 [0.190, 0.414] | 0.284 [0.174, 0.396] | 0.292 [0.181, 0.404] | 0.305 [0.193, 0.417] | 0.254 [0.155, 0.352] | 0.302 [0.189, 0.418] |
| **FLAME 2023 Open vA, coefficienti*** | 0.155 [0.043, 0.262] | 0.158 [0.046, 0.264] | 0.158 [0.042, 0.267] | 0.163 [0.048, 0.273] | 0.162 [0.048, 0.273] | 0.137 [0.031, 0.237] | 0.162 [0.049, 0.275] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.118 [0.020, 0.217] | 0.120 [0.023, 0.219] | 0.120 [0.021, 0.221] | 0.123 [0.022, 0.224] | 0.121 [0.023, 0.218] | 0.108 [0.010, 0.206] | 0.121 [0.023, 0.219] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.299 [0.193, 0.405] | 0.314 [0.204, 0.423] | 0.296 [0.187, 0.405] | 0.304 [0.195, 0.412] | 0.316 [0.208, 0.423] | 0.268 [0.169, 0.366] | 0.313 [0.202, 0.421] |
| **FLAME 2023 Open vB, coefficienti*** | 0.585 [0.513, 0.652] | 0.586 [0.513, 0.650] | 0.581 [0.509, 0.649] | 0.590 [0.517, 0.656] | 0.585 [0.514, 0.651] | 0.587 [0.514, 0.655] | 0.585 [0.513, 0.652] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.231 [0.141, 0.320] | 0.232 [0.142, 0.322] | 0.230 [0.141, 0.319] | 0.234 [0.143, 0.325] | 0.231 [0.142, 0.320] | 0.226 [0.138, 0.314] | 0.232 [0.142, 0.322] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.597 [0.518, 0.673] | 0.600 [0.520, 0.675] | 0.594 [0.514, 0.671] | 0.598 [0.519, 0.673] | 0.601 [0.520, 0.677] | 0.595 [0.515, 0.669] | 0.599 [0.519, 0.674] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.346 [0.255, 0.437] | 0.348 [0.258, 0.437] | 0.344 [0.253, 0.436] | 0.349 [0.257, 0.438] | 0.347 [0.255, 0.437] | 0.341 [0.252, 0.430] | 0.348 [0.258, 0.439] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.220 [0.149, 0.293] | 0.221 [0.149, 0.293] | 0.229 [0.159, 0.301] | 0.226 [0.153, 0.300] | 0.225 [0.154, 0.298] | 0.213 [0.130, 0.293] | 0.179 [0.128, 0.231] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.237 [0.171, 0.302] | 0.246 [0.173, 0.319] | 0.245 [0.175, 0.314] | 0.248 [0.177, 0.319] | 0.250 [0.178, 0.321] | 0.223 [0.153, 0.296] | 0.203 [0.147, 0.258] |
| **ShapeDNA k=50** | 0.126 [0.097, 0.151] | 0.204 [0.160, 0.244] | 0.193 [0.151, 0.231] | 0.195 [0.151, 0.232] | 0.173 [0.133, 0.208] | 0.038 [-0.008, 0.082] | 0.379 [0.301, 0.447] |
| **HKS globale** | 0.069 [0.039, 0.100] | 0.108 [0.060, 0.158] | 0.093 [0.053, 0.135] | 0.106 [0.058, 0.155] | 0.097 [0.053, 0.142] | 0.041 [-0.003, 0.082] | 0.203 [0.117, 0.292] |
| **WKS globale** | 0.030 [0.016, 0.044] | 0.065 [0.042, 0.090] | 0.065 [0.042, 0.089] | 0.053 [0.030, 0.073] | 0.041 [0.024, 0.060] | 0.016 [-0.025, 0.060] | 0.314 [0.230, 0.392] |
| **OpenShape PointBERT** | 0.182 [0.141, 0.227] | 0.255 [0.206, 0.309] | 0.235 [0.185, 0.290] | 0.247 [0.194, 0.304] | 0.232 [0.176, 0.289] | 0.200 [0.112, 0.291] | 0.375 [0.292, 0.459] |
| **Uni3D-g** | 0.136 [0.096, 0.177] | 0.195 [0.138, 0.255] | 0.191 [0.139, 0.243] | 0.203 [0.148, 0.259] | 0.198 [0.141, 0.259] | 0.083 [-0.001, 0.170] | 0.377 [0.288, 0.467] |
| **taglia stimata dalla mesh (non oracolo)** | 0.023 [-0.070, 0.111] | 0.023 [-0.070, 0.111] | 0.023 [-0.067, 0.108] | 0.023 [-0.070, 0.110] | 0.022 [-0.072, 0.111] | 0.023 [-0.069, 0.112] | 0.024 [-0.071, 0.113] |
| **taglia oracolo (S di FR)** | 0.074 [-0.025, 0.165] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.165] | 0.073 [-0.029, 0.163] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.166] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.166 [0.121, 0.210] | 0.247 [0.191, 0.305] | 0.224 [0.170, 0.276] | 0.238 [0.181, 0.295] | 0.238 [0.181, 0.295] | 0.131 [0.032, 0.232] | 0.413 [0.324, 0.505] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.173 [0.131, 0.215] | 0.257 [0.201, 0.314] | 0.232 [0.182, 0.281] | 0.246 [0.191, 0.301] | 0.250 [0.195, 0.303] | 0.138 [0.045, 0.230] | 0.425 [0.333, 0.516] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.137 [0.105, 0.170] | 0.209 [0.164, 0.254] | 0.182 [0.136, 0.227] | 0.178 [0.134, 0.224] | 0.166 [0.127, 0.206] | 0.185 [0.119, 0.256] | 0.404 [0.328, 0.481] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.149 [0.122, 0.179] | 0.228 [0.185, 0.272] | 0.199 [0.160, 0.238] | 0.194 [0.155, 0.234] | 0.183 [0.148, 0.217] | 0.217 [0.149, 0.284] | 0.435 [0.352, 0.512] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.200 [0.159, 0.244] | 0.281 [0.228, 0.338] | 0.256 [0.208, 0.306] | 0.274 [0.221, 0.330] | 0.273 [0.219, 0.330] | 0.238 [0.150, 0.321] | 0.485 [0.398, 0.578] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.206 [0.168, 0.246] | 0.291 [0.241, 0.344] | 0.261 [0.215, 0.308] | 0.282 [0.232, 0.336] | 0.282 [0.232, 0.336] | 0.252 [0.171, 0.329] | 0.500 [0.411, 0.593] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.077 [0.010, 0.146] | 0.093 [0.019, 0.172] | 0.095 [0.018, 0.175] | 0.092 [0.018, 0.171] | 0.091 [0.016, 0.169] | 0.054 [0.002, 0.110] | - |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.228 [0.183, 0.271] | 0.308 [0.253, 0.360] | 0.309 [0.254, 0.361] | 0.306 [0.250, 0.358] | 0.303 [0.248, 0.357] | 0.267 [0.171, 0.356] | - |
| LPIPS alex su render [profondita'], mm (FR) | 0.212 [0.132, 0.292] | 0.228 [0.147, 0.309] | 0.225 [0.145, 0.306] | 0.231 [0.149, 0.313] | 0.226 [0.144, 0.308] | 0.220 [0.135, 0.310] | 0.271 [0.183, 0.362] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.349 [0.292, 0.410] | 0.394 [0.335, 0.457] | 0.390 [0.331, 0.454] | 0.402 [0.342, 0.466] | 0.394 [0.335, 0.457] | 0.385 [0.317, 0.454] | 0.523 [0.453, 0.594] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.430 [0.349, 0.509] | 0.437 [0.356, 0.517] | 0.431 [0.349, 0.511] | 0.432 [0.347, 0.513] | 0.428 [0.345, 0.510] | 0.446 [0.363, 0.527] | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.620 [0.555, 0.677] | 0.623 [0.559, 0.680] | 0.620 [0.554, 0.678] | 0.632 [0.567, 0.690] | 0.622 [0.557, 0.681] | 0.641 [0.576, 0.699] | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.715 [0.645, 0.771] | 0.720 [0.652, 0.775] | 0.718 [0.650, 0.775] | 0.715 [0.646, 0.772] | 0.722 [0.653, 0.779] | 0.717 [0.648, 0.774] | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.631 [0.553, 0.695] | 0.640 [0.562, 0.705] | 0.633 [0.556, 0.699] | 0.631 [0.552, 0.698] | 0.631 [0.553, 0.696] | 0.646 [0.568, 0.713] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.269 [0.181, 0.356] | 0.273 [0.186, 0.362] | 0.278 [0.190, 0.366] | 0.274 [0.186, 0.362] | 0.274 [0.185, 0.362] | 0.304 [0.217, 0.391] | 0.315 [0.220, 0.412] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.479 [0.412, 0.541] | 0.514 [0.448, 0.576] | 0.517 [0.451, 0.579] | 0.524 [0.457, 0.588] | 0.520 [0.452, 0.582] | 0.629 [0.557, 0.696] | 0.666 [0.596, 0.730] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.298 [0.226, 0.369] | 0.303 [0.227, 0.377] | 0.310 [0.232, 0.385] | 0.317 [0.241, 0.392] | 0.306 [0.227, 0.383] | 0.287 [0.221, 0.351] | 0.355 [0.268, 0.441] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.482 [0.410, 0.551] | 0.486 [0.410, 0.560] | 0.476 [0.400, 0.552] | 0.535 [0.463, 0.604] | 0.492 [0.416, 0.568] | 0.554 [0.484, 0.621] | 0.600 [0.526, 0.674] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.106 [+0.062, +0.154] | +0.117 [+0.074, +0.166] | +0.072 [+0.029, +0.119] | +0.131 [+0.088, +0.176] | +0.084 [+0.040, +0.132] | +0.089 [+0.042, +0.139] | +0.100 [+0.055, +0.150] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.390 [+0.291, +0.493] | +0.408 [+0.311, +0.508] | +0.362 [+0.259, +0.467] | +0.406 [+0.309, +0.505] | +0.374 [+0.271, +0.479] | +0.379 [+0.277, +0.483] | +0.375 [+0.273, +0.479] |
| NICP per coppia in mm | +0.253 [+0.170, +0.342] | +0.279 [+0.197, +0.366] | +0.226 [+0.141, +0.318] | +0.265 [+0.181, +0.355] | +0.239 [+0.153, +0.329] | +0.239 [+0.153, +0.333] | - |
| NICP per coppia cs | +0.373 [+0.261, +0.481] | +0.398 [+0.290, +0.502] | +0.350 [+0.235, +0.460] | +0.381 [+0.269, +0.487] | +0.362 [+0.246, +0.474] | +0.364 [+0.251, +0.475] | - |
| NICP su template in mm | +0.134 [+0.085, +0.188] | +0.139 [+0.089, +0.193] | +0.125 [+0.075, +0.177] | +0.144 [+0.094, +0.199] | +0.131 [+0.083, +0.183] | +0.135 [+0.084, +0.188] | +0.143 [+0.093, +0.197] |
| NICP su template cs | +0.560 [+0.465, +0.657] | +0.564 [+0.471, +0.660] | +0.550 [+0.458, +0.645] | +0.563 [+0.469, +0.659] | +0.556 [+0.459, +0.656] | +0.566 [+0.467, +0.668] | +0.558 [+0.459, +0.659] |
| varifold in mm (massa unitaria) | +0.254 [+0.211, +0.301] | +0.204 [+0.164, +0.247] | +0.191 [+0.153, +0.233] | +0.200 [+0.160, +0.244] | +0.200 [+0.160, +0.242] | +0.185 [+0.140, +0.233] | +0.066 [+0.033, +0.105] |
| GNM (visto) preregistrato, coefficienti | +0.130 [+0.078, +0.188] | +0.131 [+0.081, +0.190] | +0.123 [+0.070, +0.182] | +0.135 [+0.083, +0.194] | +0.127 [+0.074, +0.186] | +0.135 [+0.084, +0.192] | +0.138 [+0.086, +0.197] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.141 [+0.088, +0.195] | +0.142 [+0.090, +0.195] | +0.133 [+0.081, +0.187] | +0.144 [+0.091, +0.199] | +0.140 [+0.090, +0.194] | +0.145 [+0.093, +0.202] | +0.150 [+0.098, +0.205] |
| GNM (visto) vA, coefficienti | +0.145 [+0.091, +0.207] | +0.148 [+0.093, +0.208] | +0.136 [+0.082, +0.197] | +0.151 [+0.096, +0.213] | +0.141 [+0.087, +0.203] | +0.150 [+0.093, +0.210] | +0.154 [+0.099, +0.216] |
| GNM (visto) vA, mesh d'identita' FR | +0.142 [+0.090, +0.195] | +0.143 [+0.091, +0.196] | +0.134 [+0.082, +0.189] | +0.145 [+0.093, +0.199] | +0.141 [+0.092, +0.195] | +0.146 [+0.094, +0.202] | +0.151 [+0.099, +0.205] |
| GNM (visto) vB, coefficienti | +0.199 [+0.095, +0.299] | +0.203 [+0.099, +0.306] | +0.189 [+0.087, +0.288] | +0.205 [+0.101, +0.305] | +0.203 [+0.099, +0.302] | +0.193 [+0.088, +0.296] | +0.211 [+0.107, +0.312] |
| GNM (visto) vB, mesh d'identita' FR | +0.084 [+0.042, +0.128] | +0.086 [+0.044, +0.130] | +0.078 [+0.036, +0.120] | +0.087 [+0.044, +0.131] | +0.084 [+0.042, +0.127] | +0.086 [+0.043, +0.132] | +0.094 [+0.051, +0.139] |
| GNM (visto) vB, composizione S_B + k d_P | +0.051 [+0.009, +0.094] | +0.054 [+0.012, +0.098] | +0.045 [+0.004, +0.087] | +0.054 [+0.012, +0.098] | +0.051 [+0.010, +0.094] | +0.051 [+0.009, +0.095] | +0.060 [+0.018, +0.106] |
| FLAME 2023 Open preregistrato, coefficienti | +0.182 [+0.119, +0.254] | +0.184 [+0.121, +0.258] | +0.183 [+0.118, +0.255] | +0.180 [+0.113, +0.256] | +0.172 [+0.109, +0.244] | +0.192 [+0.129, +0.261] | +0.179 [+0.113, +0.255] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.109 [+0.062, +0.161] | +0.110 [+0.063, +0.161] | +0.105 [+0.058, +0.158] | +0.108 [+0.061, +0.161] | +0.105 [+0.057, +0.159] | +0.120 [+0.073, +0.172] | +0.115 [+0.067, +0.168] |
| FLAME 2023 Open vA, coefficienti | +0.168 [+0.108, +0.238] | +0.170 [+0.109, +0.241] | +0.168 [+0.106, +0.240] | +0.165 [+0.102, +0.238] | +0.158 [+0.098, +0.229] | +0.179 [+0.118, +0.244] | +0.166 [+0.103, +0.240] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.107 [+0.061, +0.158] | +0.109 [+0.062, +0.159] | +0.102 [+0.056, +0.154] | +0.106 [+0.058, +0.158] | +0.104 [+0.058, +0.154] | +0.117 [+0.072, +0.169] | +0.114 [+0.067, +0.166] |
| FLAME 2023 Open vB, coefficienti | +0.163 [+0.072, +0.254] | +0.172 [+0.081, +0.263] | +0.159 [+0.070, +0.248] | +0.167 [+0.076, +0.259] | +0.162 [+0.073, +0.253] | +0.154 [+0.061, +0.245] | +0.177 [+0.087, +0.265] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.063 [+0.023, +0.101] | +0.064 [+0.025, +0.102] | +0.058 [+0.019, +0.097] | +0.062 [+0.021, +0.101] | +0.061 [+0.022, +0.099] | +0.068 [+0.028, +0.107] | +0.070 [+0.029, +0.109] |
| FLAME 2023 Open vB, composizione S_B + k d_P | +0.037 [-0.002, +0.075] | +0.039 [+0.001, +0.077] | +0.032 [-0.006, +0.070] | +0.038 [-0.001, +0.079] | +0.035 [-0.003, +0.074] | +0.040 [-0.001, +0.079] | +0.045 [+0.006, +0.084] |
| ArcFace-render, normal map 3 viste (G0) | +0.615 [+0.498, +0.720] | +0.621 [+0.502, +0.728] | +0.601 [+0.484, +0.706] | +0.614 [+0.492, +0.720] | +0.614 [+0.494, +0.722] | +0.619 [+0.498, +0.726] | +0.648 [+0.549, +0.738] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.583 [+0.471, +0.685] | +0.580 [+0.460, +0.685] | +0.566 [+0.447, +0.668] | +0.578 [+0.460, +0.684] | +0.575 [+0.456, +0.682] | +0.599 [+0.491, +0.698] | +0.616 [+0.513, +0.709] |
| ShapeDNA k=50 | +0.663 [+0.588, +0.731] | +0.611 [+0.529, +0.691] | +0.605 [+0.524, +0.680] | +0.617 [+0.537, +0.695] | +0.629 [+0.551, +0.703] | +0.744 [+0.658, +0.822] | +0.500 [+0.386, +0.607] |
| HKS globale | +0.712 [+0.635, +0.779] | +0.695 [+0.610, +0.770] | +0.690 [+0.610, +0.762] | +0.694 [+0.610, +0.769] | +0.698 [+0.616, +0.771] | +0.736 [+0.643, +0.807] | +0.646 [+0.540, +0.751] |
| WKS globale | +0.730 [+0.657, +0.788] | +0.711 [+0.636, +0.772] | +0.697 [+0.624, +0.757] | +0.714 [+0.641, +0.777] | +0.723 [+0.650, +0.784] | +0.751 [+0.669, +0.829] | +0.553 [+0.446, +0.661] |
| OpenShape PointBERT | +0.620 [+0.536, +0.693] | +0.574 [+0.486, +0.657] | +0.582 [+0.493, +0.664] | +0.578 [+0.486, +0.662] | +0.586 [+0.498, +0.671] | +0.595 [+0.479, +0.701] | +0.497 [+0.389, +0.609] |
| Uni3D-g | +0.647 [+0.566, +0.720] | +0.615 [+0.523, +0.699] | +0.597 [+0.513, +0.678] | +0.600 [+0.511, +0.683] | +0.607 [+0.517, +0.694] | +0.671 [+0.559, +0.778] | +0.491 [+0.367, +0.615] |
| taglia stimata dalla mesh (non oracolo) | +0.149 [+0.091, +0.208] | +0.152 [+0.095, +0.213] | +0.153 [+0.096, +0.213] | +0.150 [+0.094, +0.211] | +0.146 [+0.089, +0.204] | +0.141 [+0.085, +0.201] | +0.144 [+0.086, +0.204] |
| taglia oracolo (S di FR) | +0.012 [-0.034, +0.062] | +0.018 [-0.029, +0.068] | +0.004 [-0.042, +0.056] | +0.015 [-0.031, +0.065] | +0.014 [-0.033, +0.065] | +0.010 [-0.037, +0.060] | +0.023 [-0.023, +0.074] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.598 [+0.521, +0.662] | +0.538 [+0.458, +0.613] | +0.543 [+0.467, +0.614] | +0.539 [+0.457, +0.614] | +0.541 [+0.460, +0.616] | +0.596 [+0.486, +0.703] | +0.408 [+0.300, +0.515] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.630 [+0.546, +0.701] | +0.565 [+0.475, +0.645] | +0.580 [+0.492, +0.659] | +0.602 [+0.514, +0.682] | +0.596 [+0.511, +0.672] | +0.607 [+0.498, +0.695] | +0.391 [+0.267, +0.509] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.566 [+0.487, +0.629] | +0.500 [+0.416, +0.576] | +0.510 [+0.430, +0.580] | +0.505 [+0.422, +0.580] | +0.502 [+0.419, +0.576] | +0.519 [+0.408, +0.621] | +0.328 [+0.213, +0.441] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.282 [+0.237, +0.333] | +0.218 [+0.176, +0.265] | +0.190 [+0.146, +0.237] | +0.217 [+0.175, +0.264] | +0.215 [+0.173, +0.261] | +0.465 [+0.409, +0.520] | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.133 [+0.089, +0.180] | +0.115 [+0.070, +0.161] | +0.113 [+0.070, +0.157] | +0.107 [+0.062, +0.153] | +0.112 [+0.068, +0.157] | +0.099 [+0.052, +0.149] | +0.047 [+0.005, +0.088] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | +0.051 [+0.018, +0.088] | +0.052 [+0.018, +0.088] | +0.043 [+0.009, +0.079] | +0.042 [+0.008, +0.077] | +0.051 [+0.017, +0.087] | +0.028 [-0.008, +0.064] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.333 [+0.228, +0.432] | +0.335 [+0.228, +0.435] | +0.325 [+0.219, +0.424] | +0.329 [+0.224, +0.428] | +0.335 [+0.230, +0.435] | +0.315 [+0.206, +0.420] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.254 [+0.134, +0.376] | +0.256 [+0.136, +0.379] | +0.243 [+0.124, +0.364] | +0.258 [+0.138, +0.380] | +0.253 [+0.133, +0.376] | +0.249 [+0.126, +0.373] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.285 [+0.167, +0.400] | +0.284 [+0.165, +0.398] | +0.276 [+0.157, +0.390] | +0.288 [+0.167, +0.406] | +0.287 [+0.168, +0.403] | +0.271 [+0.153, +0.388] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.058 [+0.024, +0.095] | +0.051 [+0.017, +0.085] | +0.030 [-0.002, +0.065] | +0.047 [+0.013, +0.082] | +0.041 [+0.008, +0.075] | +0.043 [+0.005, +0.080] | -0.020 [-0.055, +0.012] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.119 [+0.071, +0.177] | +0.105 [+0.053, +0.164] | +0.083 [+0.030, +0.145] | +0.085 [+0.038, +0.136] | +0.093 [+0.039, +0.155] | +0.190 [+0.140, +0.242] | +0.008 [-0.044, +0.064] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.001 [-0.055, +0.055] | +0.012 [-0.041, +0.065] | -0.026 [-0.081, +0.029] | +0.023 [-0.032, +0.077] | -0.020 [-0.075, +0.037] | -0.024 [-0.081, +0.032] | -0.006 [-0.062, +0.050] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.284 [+0.206, +0.366] | +0.303 [+0.227, +0.383] | +0.263 [+0.182, +0.347] | +0.299 [+0.225, +0.376] | +0.269 [+0.188, +0.353] | +0.265 [+0.184, +0.351] | +0.269 [+0.188, +0.352] |
| NICP per coppia in mm | +0.147 [+0.086, +0.216] | +0.174 [+0.114, +0.241] | +0.127 [+0.064, +0.199] | +0.158 [+0.099, +0.227] | +0.135 [+0.072, +0.206] | +0.125 [+0.062, +0.196] | - |
| NICP per coppia cs | +0.267 [+0.181, +0.353] | +0.292 [+0.210, +0.377] | +0.251 [+0.162, +0.340] | +0.273 [+0.188, +0.356] | +0.258 [+0.169, +0.346] | +0.251 [+0.164, +0.340] | - |
| NICP su template in mm | +0.028 [-0.057, +0.117] | +0.033 [-0.052, +0.123] | +0.026 [-0.058, +0.114] | +0.036 [-0.049, +0.127] | +0.026 [-0.058, +0.113] | +0.021 [-0.064, +0.112] | +0.037 [-0.047, +0.126] |
| NICP su template cs | +0.454 [+0.371, +0.537] | +0.458 [+0.377, +0.541] | +0.452 [+0.378, +0.531] | +0.456 [+0.371, +0.540] | +0.452 [+0.370, +0.537] | +0.452 [+0.366, +0.540] | +0.452 [+0.366, +0.539] |
| varifold in mm (massa unitaria) | +0.148 [+0.083, +0.214] | +0.098 [+0.036, +0.161] | +0.093 [+0.029, +0.157] | +0.093 [+0.029, +0.155] | +0.095 [+0.033, +0.156] | +0.071 [+0.014, +0.125] | -0.040 [-0.101, +0.019] |
| GNM (visto) preregistrato, coefficienti | +0.024 [-0.050, +0.107] | +0.026 [-0.047, +0.106] | +0.025 [-0.052, +0.108] | +0.027 [-0.046, +0.110] | +0.022 [-0.052, +0.104] | +0.021 [-0.058, +0.105] | +0.031 [-0.042, +0.114] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.035 [-0.052, +0.121] | +0.037 [-0.047, +0.124] | +0.035 [-0.051, +0.122] | +0.037 [-0.048, +0.126] | +0.036 [-0.050, +0.123] | +0.031 [-0.055, +0.123] | +0.044 [-0.041, +0.133] |
| GNM (visto) vA, coefficienti | +0.039 [-0.036, +0.118] | +0.043 [-0.030, +0.123] | +0.038 [-0.036, +0.117] | +0.043 [-0.031, +0.124] | +0.037 [-0.038, +0.112] | +0.036 [-0.042, +0.119] | +0.047 [-0.027, +0.125] |
| GNM (visto) vA, mesh d'identita' FR | +0.035 [-0.049, +0.122] | +0.037 [-0.045, +0.123] | +0.036 [-0.049, +0.123] | +0.038 [-0.045, +0.126] | +0.037 [-0.048, +0.122] | +0.032 [-0.054, +0.122] | +0.045 [-0.038, +0.133] |
| GNM (visto) vB, coefficienti | +0.093 [+0.011, +0.180] | +0.098 [+0.018, +0.185] | +0.091 [+0.011, +0.177] | +0.097 [+0.015, +0.185] | +0.099 [+0.015, +0.187] | +0.079 [-0.003, +0.169] | +0.104 [+0.023, +0.191] |
| GNM (visto) vB, mesh d'identita' FR | -0.022 [-0.094, +0.049] | -0.019 [-0.090, +0.053] | -0.020 [-0.092, +0.052] | -0.021 [-0.091, +0.052] | -0.021 [-0.091, +0.050] | -0.028 [-0.103, +0.045] | -0.013 [-0.084, +0.059] |
| GNM (visto) vB, composizione S_B + k d_P | -0.056 [-0.118, +0.007] | -0.052 [-0.113, +0.009] | -0.054 [-0.118, +0.007] | -0.054 [-0.115, +0.008] | -0.053 [-0.114, +0.006] | -0.063 [-0.128, +0.002] | -0.046 [-0.109, +0.016] |
| FLAME 2023 Open preregistrato, coefficienti | +0.076 [-0.005, +0.162] | +0.079 [+0.000, +0.167] | +0.085 [+0.002, +0.174] | +0.072 [-0.011, +0.159] | +0.068 [-0.013, +0.155] | +0.078 [-0.005, +0.162] | +0.073 [-0.010, +0.160] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.003 [-0.073, +0.081] | +0.004 [-0.072, +0.082] | +0.007 [-0.071, +0.088] | +0.000 [-0.075, +0.081] | +0.001 [-0.074, +0.079] | +0.006 [-0.074, +0.086] | +0.008 [-0.068, +0.087] |
| FLAME 2023 Open vA, coefficienti | +0.062 [-0.018, +0.147] | +0.064 [-0.015, +0.151] | +0.070 [-0.011, +0.157] | +0.058 [-0.023, +0.145] | +0.054 [-0.026, +0.139] | +0.065 [-0.017, +0.149] | +0.060 [-0.023, +0.145] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.001 [-0.075, +0.078] | +0.003 [-0.073, +0.079] | +0.004 [-0.073, +0.082] | -0.001 [-0.076, +0.076] | -0.000 [-0.076, +0.078] | +0.003 [-0.079, +0.085] | +0.008 [-0.067, +0.086] |
| FLAME 2023 Open vB, coefficienti | +0.057 [-0.024, +0.141] | +0.067 [-0.014, +0.152] | +0.061 [-0.019, +0.145] | +0.060 [-0.021, +0.141] | +0.057 [-0.022, +0.140] | +0.041 [-0.041, +0.126] | +0.070 [-0.010, +0.154] |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.044 [-0.112, +0.022] | -0.041 [-0.109, +0.024] | -0.040 [-0.109, +0.025] | -0.046 [-0.112, +0.019] | -0.043 [-0.111, +0.020] | -0.046 [-0.116, +0.023] | -0.037 [-0.104, +0.028] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.069 [-0.127, -0.012] | -0.067 [-0.124, -0.010] | -0.066 [-0.125, -0.008] | -0.069 [-0.126, -0.011] | -0.069 [-0.127, -0.012] | -0.073 [-0.135, -0.014] | -0.062 [-0.119, -0.004] |
| ArcFace-render, normal map 3 viste (G0) | +0.509 [+0.400, +0.606] | +0.515 [+0.405, +0.614] | +0.502 [+0.394, +0.602] | +0.506 [+0.397, +0.605] | +0.510 [+0.400, +0.611] | +0.506 [+0.390, +0.608] | +0.541 [+0.449, +0.625] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.476 [+0.375, +0.568] | +0.474 [+0.367, +0.572] | +0.467 [+0.361, +0.565] | +0.470 [+0.363, +0.568] | +0.471 [+0.363, +0.571] | +0.485 [+0.387, +0.581] | +0.510 [+0.420, +0.594] |
| ShapeDNA k=50 | +0.557 [+0.490, +0.620] | +0.506 [+0.431, +0.576] | +0.507 [+0.432, +0.576] | +0.510 [+0.437, +0.578] | +0.525 [+0.453, +0.591] | +0.630 [+0.556, +0.704] | +0.394 [+0.293, +0.490] |
| HKS globale | +0.606 [+0.536, +0.668] | +0.590 [+0.513, +0.659] | +0.592 [+0.518, +0.660] | +0.586 [+0.509, +0.657] | +0.594 [+0.518, +0.664] | +0.622 [+0.545, +0.690] | +0.540 [+0.443, +0.630] |
| WKS globale | +0.624 [+0.558, +0.684] | +0.606 [+0.542, +0.666] | +0.599 [+0.532, +0.658] | +0.607 [+0.539, +0.671] | +0.619 [+0.553, +0.679] | +0.637 [+0.564, +0.706] | +0.446 [+0.351, +0.540] |
| OpenShape PointBERT | +0.514 [+0.444, +0.574] | +0.469 [+0.396, +0.538] | +0.483 [+0.405, +0.551] | +0.471 [+0.397, +0.540] | +0.482 [+0.408, +0.549] | +0.481 [+0.387, +0.574] | +0.391 [+0.297, +0.479] |
| Uni3D-g | +0.541 [+0.469, +0.601] | +0.510 [+0.432, +0.580] | +0.499 [+0.425, +0.567] | +0.492 [+0.416, +0.562] | +0.503 [+0.423, +0.574] | +0.558 [+0.453, +0.649] | +0.384 [+0.280, +0.491] |
| taglia stimata dalla mesh (non oracolo) | +0.042 [-0.052, +0.137] | +0.047 [-0.047, +0.141] | +0.055 [-0.038, +0.148] | +0.043 [-0.051, +0.138] | +0.041 [-0.054, +0.135] | +0.028 [-0.068, +0.121] | +0.037 [-0.057, +0.131] |
| taglia oracolo (S di FR) | -0.094 [-0.160, -0.022] | -0.088 [-0.153, -0.017] | -0.094 [-0.160, -0.021] | -0.092 [-0.160, -0.021] | -0.091 [-0.157, -0.018] | -0.104 [-0.171, -0.033] | -0.083 [-0.148, -0.013] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.492 [+0.427, +0.550] | +0.432 [+0.360, +0.501] | +0.445 [+0.376, +0.514] | +0.432 [+0.360, +0.500] | +0.436 [+0.367, +0.504] | +0.482 [+0.385, +0.571] | +0.302 [+0.207, +0.400] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.524 [+0.445, +0.585] | +0.459 [+0.378, +0.531] | +0.481 [+0.401, +0.551] | +0.495 [+0.413, +0.566] | +0.492 [+0.416, +0.559] | +0.493 [+0.398, +0.568] | +0.285 [+0.173, +0.388] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.460 [+0.399, +0.515] | +0.394 [+0.327, +0.462] | +0.412 [+0.349, +0.474] | +0.398 [+0.331, +0.465] | +0.398 [+0.331, +0.466] | +0.405 [+0.315, +0.487] | +0.221 [+0.124, +0.321] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.176 [+0.102, +0.250] | +0.112 [+0.039, +0.187] | +0.091 [+0.017, +0.168] | +0.110 [+0.036, +0.184] | +0.111 [+0.038, +0.183] | +0.351 [+0.284, +0.414] | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.027 [-0.042, +0.097] | +0.010 [-0.057, +0.077] | +0.015 [-0.052, +0.082] | -0.001 [-0.068, +0.067] | +0.007 [-0.060, +0.074] | -0.015 [-0.085, +0.055] | -0.060 [-0.124, +0.001] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.055 [-0.105, -0.009] | -0.053 [-0.102, -0.007] | -0.055 [-0.106, -0.008] | -0.066 [-0.117, -0.019] | -0.054 [-0.105, -0.006] | -0.086 [-0.139, -0.038] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.227 [+0.151, +0.299] | +0.229 [+0.153, +0.304] | +0.227 [+0.153, +0.300] | +0.221 [+0.148, +0.293] | +0.231 [+0.156, +0.304] | +0.201 [+0.124, +0.278] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.148 [+0.043, +0.247] | +0.150 [+0.048, +0.249] | +0.144 [+0.040, +0.244] | +0.151 [+0.046, +0.251] | +0.148 [+0.044, +0.248] | +0.135 [+0.023, +0.238] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.179 [+0.089, +0.267] | +0.178 [+0.088, +0.267] | +0.177 [+0.087, +0.266] | +0.181 [+0.088, +0.271] | +0.183 [+0.091, +0.274] | +0.158 [+0.066, +0.246] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | -0.048 [-0.108, +0.012] | -0.055 [-0.115, +0.005] | -0.068 [-0.130, -0.007] | -0.061 [-0.120, +0.001] | -0.064 [-0.123, -0.004] | -0.071 [-0.126, -0.015] | -0.127 [-0.187, -0.068] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.013 [-0.047, +0.080] | -0.000 [-0.065, +0.072] | -0.015 [-0.081, +0.057] | -0.022 [-0.080, +0.042] | -0.012 [-0.079, +0.061] | +0.076 [+0.022, +0.133] | -0.099 [-0.164, -0.030] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.256 [+0.181, +0.332] | +0.262 [+0.190, +0.336] | +0.244 [+0.167, +0.322] | +0.282 [+0.208, +0.355] | +0.251 [+0.172, +0.333] | +0.228 [+0.150, +0.305] | +0.257 [+0.181, +0.334] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.041 [-0.005, +0.091] | +0.073 [+0.025, +0.122] | +0.009 [-0.040, +0.059] | +0.068 [+0.023, +0.112] | +0.012 [-0.035, +0.061] | +0.010 [-0.040, +0.062] | +0.012 [-0.039, +0.063] |
| NICP per coppia in mm | +0.119 [+0.052, +0.195] | +0.152 [+0.086, +0.223] | +0.095 [+0.026, +0.172] | +0.127 [+0.063, +0.201] | +0.104 [+0.036, +0.182] | +0.101 [+0.029, +0.179] | - |
| NICP per coppia cs | +0.027 [-0.029, +0.083] | +0.068 [+0.014, +0.125] | +0.005 [-0.053, +0.064] | +0.040 [-0.013, +0.096] | +0.007 [-0.048, +0.067] | -0.001 [-0.058, +0.057] | - |
| NICP su template in mm | +0.539 [+0.441, +0.638] | +0.546 [+0.447, +0.644] | +0.543 [+0.446, +0.641] | +0.544 [+0.444, +0.643] | +0.541 [+0.442, +0.641] | +0.525 [+0.419, +0.625] | +0.551 [+0.452, +0.649] |
| NICP su template cs | +0.321 [+0.250, +0.401] | +0.325 [+0.254, +0.407] | +0.327 [+0.261, +0.405] | +0.324 [+0.248, +0.408] | +0.317 [+0.243, +0.401] | +0.311 [+0.239, +0.391] | +0.312 [+0.233, +0.401] |
| varifold in mm (massa unitaria) | +0.427 [+0.359, +0.496] | +0.407 [+0.338, +0.477] | +0.403 [+0.333, +0.474] | +0.403 [+0.331, +0.472] | +0.404 [+0.334, +0.477] | +0.351 [+0.283, +0.418] | +0.331 [+0.256, +0.409] |
| GNM (visto) preregistrato, coefficienti | +0.465 [+0.375, +0.567] | +0.467 [+0.377, +0.568] | +0.467 [+0.376, +0.569] | +0.471 [+0.380, +0.572] | +0.467 [+0.376, +0.568] | +0.459 [+0.365, +0.560] | +0.477 [+0.386, +0.579] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.297 [+0.212, +0.395] | +0.303 [+0.216, +0.405] | +0.298 [+0.212, +0.399] | +0.300 [+0.215, +0.400] | +0.298 [+0.213, +0.400] | +0.289 [+0.207, +0.386] | +0.306 [+0.220, +0.407] |
| GNM (visto) vA, coefficienti | +0.428 [+0.339, +0.528] | +0.430 [+0.341, +0.531] | +0.428 [+0.336, +0.529] | +0.434 [+0.347, +0.534] | +0.427 [+0.340, +0.528] | +0.424 [+0.333, +0.522] | +0.439 [+0.350, +0.538] |
| GNM (visto) vA, mesh d'identita' SR | +0.282 [+0.197, +0.378] | +0.292 [+0.206, +0.388] | +0.284 [+0.196, +0.380] | +0.287 [+0.200, +0.380] | +0.280 [+0.197, +0.375] | +0.271 [+0.186, +0.362] | +0.291 [+0.205, +0.387] |
| GNM (visto) vB, coefficienti | -0.053 [-0.136, +0.032] | -0.047 [-0.131, +0.039] | -0.047 [-0.131, +0.037] | -0.046 [-0.127, +0.038] | -0.047 [-0.128, +0.038] | -0.075 [-0.161, +0.014] | -0.039 [-0.121, +0.045] |
| GNM (visto) vB, mesh d'identita' SR | +0.015 [-0.040, +0.074] | +0.021 [-0.033, +0.079] | +0.020 [-0.035, +0.080] | +0.019 [-0.034, +0.076] | +0.020 [-0.035, +0.077] | -0.003 [-0.060, +0.058] | +0.027 [-0.026, +0.086] |
| FLAME 2023 Open preregistrato, coefficienti | +0.463 [+0.378, +0.556] | +0.468 [+0.381, +0.562] | +0.465 [+0.380, +0.559] | +0.460 [+0.372, +0.556] | +0.460 [+0.377, +0.557] | +0.465 [+0.376, +0.556] | +0.468 [+0.382, +0.562] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.335 [+0.255, +0.421] | +0.328 [+0.243, +0.418] | +0.342 [+0.257, +0.435] | +0.333 [+0.249, +0.420] | +0.322 [+0.238, +0.410] | +0.350 [+0.277, +0.426] | +0.332 [+0.249, +0.422] |
| FLAME 2023 Open vA, coefficienti | +0.466 [+0.380, +0.561] | +0.471 [+0.383, +0.566] | +0.469 [+0.382, +0.564] | +0.463 [+0.375, +0.561] | +0.464 [+0.380, +0.561] | +0.468 [+0.377, +0.559] | +0.472 [+0.385, +0.567] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.323 [+0.244, +0.408] | +0.316 [+0.233, +0.407] | +0.330 [+0.246, +0.421] | +0.321 [+0.238, +0.409] | +0.310 [+0.228, +0.399] | +0.337 [+0.265, +0.415] | +0.321 [+0.239, +0.410] |
| FLAME 2023 Open vB, coefficienti | +0.036 [-0.048, +0.123] | +0.044 [-0.041, +0.133] | +0.045 [-0.042, +0.132] | +0.036 [-0.048, +0.121] | +0.041 [-0.042, +0.128] | +0.017 [-0.073, +0.107] | +0.049 [-0.037, +0.135] |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.024 [-0.032, +0.082] | +0.029 [-0.027, +0.087] | +0.032 [-0.027, +0.092] | +0.028 [-0.028, +0.085] | +0.025 [-0.033, +0.082] | +0.010 [-0.048, +0.067] | +0.035 [-0.020, +0.092] |
| ArcFace-render, normal map 3 viste (G0) | +0.401 [+0.302, +0.498] | +0.409 [+0.308, +0.507] | +0.397 [+0.300, +0.492] | +0.400 [+0.300, +0.497] | +0.401 [+0.301, +0.498] | +0.392 [+0.285, +0.497] | +0.456 [+0.370, +0.537] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.385 [+0.291, +0.472] | +0.383 [+0.287, +0.475] | +0.381 [+0.286, +0.470] | +0.377 [+0.283, +0.468] | +0.376 [+0.280, +0.469] | +0.381 [+0.279, +0.471] | +0.432 [+0.348, +0.511] |
| ShapeDNA k=50 | +0.496 [+0.425, +0.560] | +0.425 [+0.350, +0.493] | +0.433 [+0.353, +0.503] | +0.431 [+0.357, +0.499] | +0.453 [+0.376, +0.521] | +0.566 [+0.486, +0.641] | +0.255 [+0.165, +0.340] |
| HKS globale | +0.552 [+0.476, +0.618] | +0.521 [+0.439, +0.596] | +0.533 [+0.450, +0.606] | +0.519 [+0.436, +0.594] | +0.529 [+0.446, +0.601] | +0.564 [+0.476, +0.641] | +0.431 [+0.320, +0.538] |
| WKS globale | +0.592 [+0.524, +0.655] | +0.564 [+0.499, +0.626] | +0.561 [+0.497, +0.624] | +0.572 [+0.504, +0.635] | +0.585 [+0.516, +0.649] | +0.588 [+0.512, +0.662] | +0.320 [+0.239, +0.410] |
| OpenShape PointBERT | +0.440 [+0.374, +0.504] | +0.374 [+0.311, +0.437] | +0.391 [+0.320, +0.460] | +0.378 [+0.311, +0.440] | +0.394 [+0.323, +0.463] | +0.404 [+0.305, +0.509] | +0.259 [+0.177, +0.337] |
| Uni3D-g | +0.486 [+0.412, +0.551] | +0.434 [+0.354, +0.506] | +0.435 [+0.358, +0.503] | +0.422 [+0.347, +0.491] | +0.428 [+0.347, +0.499] | +0.521 [+0.417, +0.627] | +0.257 [+0.164, +0.348] |
| taglia stimata dalla mesh (non oracolo) | +0.599 [+0.493, +0.704] | +0.607 [+0.502, +0.711] | +0.603 [+0.499, +0.708] | +0.603 [+0.497, +0.708] | +0.604 [+0.498, +0.709] | +0.581 [+0.474, +0.683] | +0.611 [+0.506, +0.715] |
| taglia oracolo (S di FR) | +0.548 [+0.444, +0.651] | +0.555 [+0.454, +0.658] | +0.552 [+0.447, +0.657] | +0.551 [+0.446, +0.656] | +0.553 [+0.447, +0.658] | +0.531 [+0.423, +0.637] | +0.560 [+0.456, +0.663] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.449 [+0.379, +0.513] | +0.372 [+0.301, +0.443] | +0.394 [+0.326, +0.460] | +0.379 [+0.308, +0.447] | +0.376 [+0.307, +0.442] | +0.467 [+0.367, +0.568] | +0.209 [+0.121, +0.294] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.472 [+0.392, +0.543] | +0.401 [+0.314, +0.482] | +0.427 [+0.342, +0.506] | +0.431 [+0.349, +0.507] | +0.444 [+0.362, +0.518] | +0.388 [+0.286, +0.482] | +0.199 [+0.093, +0.305] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.415 [+0.349, +0.473] | +0.338 [+0.270, +0.402] | +0.365 [+0.298, +0.428] | +0.343 [+0.276, +0.405] | +0.344 [+0.277, +0.405] | +0.353 [+0.274, +0.437] | +0.134 [+0.045, +0.222] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.394 [+0.332, +0.449] | +0.321 [+0.257, +0.381] | +0.317 [+0.253, +0.378] | +0.320 [+0.256, +0.380] | +0.323 [+0.258, +0.384] | +0.337 [+0.251, +0.423] | - |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.273 [+0.204, +0.341] | +0.235 [+0.166, +0.305] | +0.236 [+0.166, +0.307] | +0.223 [+0.153, +0.294] | +0.232 [+0.165, +0.301] | +0.220 [+0.140, +0.306] | +0.112 [+0.036, +0.188] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | +0.192 [+0.127, +0.261] | +0.192 [+0.128, +0.261] | +0.195 [+0.130, +0.264] | +0.194 [+0.128, +0.263] | +0.198 [+0.132, +0.269] | +0.158 [+0.092, +0.230] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.002 [-0.046, +0.047] | +0.007 [-0.043, +0.053] | +0.006 [-0.043, +0.052] | -0.007 [-0.053, +0.036] | +0.004 [-0.043, +0.048] | -0.037 [-0.087, +0.011] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.093 [-0.182, -0.013] | -0.091 [-0.178, -0.012] | -0.092 [-0.182, -0.013] | -0.090 [-0.177, -0.011] | -0.096 [-0.182, -0.017] | -0.112 [-0.206, -0.031] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.009 [-0.077, +0.067] | -0.011 [-0.078, +0.065] | -0.007 [-0.077, +0.067] | -0.006 [-0.076, +0.071] | -0.005 [-0.073, +0.072] | -0.042 [-0.113, +0.033] | - |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.143 [+0.086, +0.198] | +0.116 [+0.058, +0.171] | +0.109 [+0.051, +0.165] | +0.101 [+0.045, +0.157] | +0.107 [+0.049, +0.164] | -0.025 [-0.093, +0.041] | -0.032 [-0.094, +0.029] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | +0.140 [+0.071, +0.220] | +0.144 [+0.068, +0.232] | +0.151 [+0.076, +0.238] | +0.090 [+0.023, +0.166] | +0.135 [+0.059, +0.221] | +0.050 [-0.021, +0.130] | +0.034 [-0.041, +0.119] |

### FaceScape (dev): Spearman per insieme di righe (seme 796786)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (88724) | con_perturbazione original (37031) | con_perturbazione remesh (37031) | con_perturbazione down8k (37031) | con_perturbazione up60k (29324) | con_perturbazione noisy (37031) | stessa_topologia senza_crop (24750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.661 [0.586, 0.727] | 0.669 [0.596, 0.736] | 0.665 [0.592, 0.730] | 0.661 [0.585, 0.730] | 0.660 [0.588, 0.721] | 0.660 [0.585, 0.728] | 0.705 [0.636, 0.770] |
| **factorized s1234, d_F non cal.** | 0.687 [0.612, 0.758] | 0.695 [0.620, 0.768] | 0.697 [0.622, 0.767] | 0.684 [0.606, 0.758] | 0.695 [0.623, 0.762] | 0.687 [0.611, 0.759] | 0.724 [0.652, 0.794] |
| factorized s1234, d_P | 0.677 [0.596, 0.750] | 0.687 [0.609, 0.761] | 0.682 [0.604, 0.754] | 0.674 [0.592, 0.750] | 0.685 [0.608, 0.755] | 0.678 [0.596, 0.752] | 0.717 [0.640, 0.789] |
| **factorized s2345, d_F cal.*** | 0.669 [0.597, 0.733] | 0.678 [0.605, 0.742] | 0.668 [0.596, 0.732] | 0.682 [0.611, 0.745] | 0.662 [0.595, 0.724] | 0.660 [0.586, 0.725] | 0.714 [0.644, 0.776] |
| **factorized s2345, d_F non cal.** | 0.698 [0.619, 0.770] | 0.702 [0.620, 0.776] | 0.711 [0.632, 0.783] | 0.704 [0.623, 0.779] | 0.709 [0.630, 0.779] | 0.680 [0.597, 0.753] | 0.723 [0.643, 0.797] |
| factorized s2345, d_P | 0.684 [0.601, 0.760] | 0.689 [0.607, 0.767] | 0.694 [0.612, 0.770] | 0.691 [0.607, 0.768] | 0.696 [0.611, 0.770] | 0.667 [0.579, 0.744] | 0.712 [0.627, 0.787] |
| **ctrlfr s1234** | 0.661 [0.593, 0.721] | 0.690 [0.625, 0.747] | 0.685 [0.618, 0.742] | 0.694 [0.629, 0.752] | 0.683 [0.616, 0.741] | 0.630 [0.560, 0.688] | 0.757 [0.693, 0.811] |
| **ctrlfr s2345** | 0.653 [0.585, 0.712] | 0.677 [0.611, 0.736] | 0.650 [0.584, 0.710] | 0.681 [0.615, 0.739] | 0.677 [0.609, 0.735] | 0.661 [0.589, 0.725] | 0.730 [0.664, 0.785] |
| **C3M e123, d_F cal.*** | 0.665 [0.596, 0.727] | 0.671 [0.601, 0.733] | 0.688 [0.620, 0.750] | 0.659 [0.589, 0.722] | 0.684 [0.613, 0.747] | 0.647 [0.575, 0.713] | 0.705 [0.637, 0.767] |
| **C3M e123, d_F non cal.** | 0.682 [0.613, 0.747] | 0.685 [0.617, 0.749] | 0.701 [0.634, 0.763] | 0.682 [0.612, 0.747] | 0.697 [0.629, 0.762] | 0.670 [0.597, 0.737] | 0.719 [0.655, 0.782] |
| C3M e123, d_P | 0.675 [0.603, 0.742] | 0.679 [0.606, 0.745] | 0.694 [0.623, 0.759] | 0.675 [0.602, 0.742] | 0.690 [0.619, 0.757] | 0.664 [0.590, 0.734] | 0.714 [0.647, 0.778] |
| **C3M e205, d_F cal.*** | 0.711 [0.651, 0.763] | 0.725 [0.665, 0.777] | 0.729 [0.670, 0.780] | 0.697 [0.634, 0.751] | 0.742 [0.684, 0.792] | 0.694 [0.633, 0.749] | 0.759 [0.701, 0.809] |
| **C3M e205, d_F non cal.** | 0.694 [0.626, 0.755] | 0.700 [0.633, 0.760] | 0.705 [0.637, 0.766] | 0.686 [0.616, 0.748] | 0.715 [0.650, 0.775] | 0.690 [0.623, 0.753] | 0.730 [0.665, 0.789] |
| C3M e205, d_P | 0.685 [0.616, 0.749] | 0.691 [0.621, 0.755] | 0.696 [0.626, 0.759] | 0.677 [0.605, 0.742] | 0.707 [0.639, 0.769] | 0.684 [0.612, 0.750] | 0.722 [0.655, 0.784] |
| **ICP + Chamfer in mm (per coppia)** | 0.453 [0.389, 0.514] | 0.451 [0.388, 0.510] | 0.548 [0.478, 0.613] | 0.395 [0.342, 0.448] | 0.524 [0.454, 0.590] | 0.486 [0.421, 0.546] | 0.611 [0.546, 0.673] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.416 [0.344, 0.485] | 0.403 [0.333, 0.472] | 0.502 [0.423, 0.581] | 0.374 [0.312, 0.434] | 0.483 [0.403, 0.558] | 0.439 [0.366, 0.512] | 0.574 [0.495, 0.650] |
| **NICP per coppia in mm** | 0.374 [0.305, 0.446] | 0.368 [0.298, 0.441] | 0.463 [0.382, 0.545] | 0.320 [0.262, 0.381] | 0.440 [0.364, 0.518] | 0.404 [0.331, 0.478] | - |
| **NICP per coppia cs** | 0.346 [0.276, 0.419] | 0.340 [0.271, 0.413] | 0.429 [0.347, 0.514] | 0.297 [0.237, 0.358] | 0.409 [0.328, 0.488] | 0.371 [0.299, 0.447] | - |
| **NICP su template in mm** | 0.542 [0.468, 0.605] | 0.564 [0.487, 0.626] | 0.557 [0.481, 0.619] | 0.509 [0.438, 0.570] | 0.558 [0.480, 0.624] | 0.530 [0.457, 0.592] | 0.569 [0.493, 0.635] |
| **NICP su template cs** | 0.400 [0.323, 0.474] | 0.396 [0.320, 0.471] | 0.390 [0.314, 0.465] | 0.420 [0.338, 0.501] | 0.395 [0.314, 0.473] | 0.404 [0.328, 0.480] | 0.459 [0.374, 0.542] |
| **varifold in mm (massa unitaria)** | 0.264 [0.223, 0.299] | 0.368 [0.316, 0.412] | 0.363 [0.312, 0.408] | 0.367 [0.315, 0.412] | 0.382 [0.327, 0.429] | 0.258 [0.179, 0.335] | 0.642 [0.560, 0.712] |
| **GNM (visto) preregistrato, coefficienti** | 0.488 [0.399, 0.561] | 0.488 [0.395, 0.564] | 0.493 [0.405, 0.567] | 0.488 [0.398, 0.563] | 0.505 [0.416, 0.581] | 0.471 [0.384, 0.545] | 0.505 [0.414, 0.577] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.465 [0.386, 0.538] | 0.468 [0.389, 0.543] | 0.474 [0.397, 0.547] | 0.456 [0.381, 0.526] | 0.478 [0.399, 0.555] | 0.452 [0.373, 0.524] | 0.483 [0.402, 0.557] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.523 [0.424, 0.614] | 0.525 [0.427, 0.618] | 0.528 [0.430, 0.622] | 0.528 [0.431, 0.622] | 0.538 [0.437, 0.633] | 0.502 [0.403, 0.594] | 0.541 [0.445, 0.631] |
| **GNM (visto) vA, coefficienti*** | 0.478 [0.387, 0.556] | 0.474 [0.380, 0.555] | 0.487 [0.396, 0.566] | 0.482 [0.391, 0.560] | 0.493 [0.401, 0.574] | 0.459 [0.368, 0.537] | 0.492 [0.403, 0.568] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.484 [0.410, 0.554] | 0.488 [0.410, 0.561] | 0.494 [0.419, 0.563] | 0.477 [0.404, 0.544] | 0.497 [0.420, 0.570] | 0.469 [0.392, 0.541] | 0.501 [0.423, 0.571] |
| GNM (visto) vA, mesh d'identita' SR* | 0.507 [0.402, 0.609] | 0.510 [0.404, 0.611] | 0.516 [0.411, 0.618] | 0.513 [0.410, 0.617] | 0.522 [0.415, 0.624] | 0.483 [0.376, 0.585] | 0.522 [0.421, 0.623] |
| **GNM (visto) vB, coefficienti*** | 0.565 [0.491, 0.636] | 0.565 [0.492, 0.638] | 0.564 [0.490, 0.637] | 0.569 [0.491, 0.638] | 0.568 [0.491, 0.639] | 0.565 [0.491, 0.636] | 0.564 [0.489, 0.636] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.678 [0.615, 0.740] | 0.680 [0.618, 0.741] | 0.679 [0.616, 0.740] | 0.671 [0.609, 0.735] | 0.683 [0.618, 0.747] | 0.679 [0.614, 0.742] | 0.690 [0.625, 0.753] |
| GNM (visto) vB, mesh d'identita' SR* | 0.767 [0.675, 0.843] | 0.766 [0.675, 0.841] | 0.768 [0.677, 0.843] | 0.765 [0.674, 0.841] | 0.768 [0.678, 0.843] | 0.767 [0.674, 0.842] | 0.767 [0.677, 0.842] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.800 [0.741, 0.847] | 0.800 [0.741, 0.847] | 0.801 [0.743, 0.849] | 0.794 [0.732, 0.842] | 0.803 [0.742, 0.852] | 0.801 [0.742, 0.850] | 0.807 [0.749, 0.855] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.385 [0.299, 0.467] | 0.388 [0.304, 0.472] | 0.390 [0.305, 0.472] | 0.404 [0.316, 0.488] | 0.395 [0.306, 0.478] | 0.350 [0.265, 0.432] | 0.422 [0.333, 0.508] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.449 [0.366, 0.519] | 0.452 [0.373, 0.522] | 0.448 [0.365, 0.517] | 0.455 [0.374, 0.525] | 0.462 [0.376, 0.536] | 0.431 [0.345, 0.502] | 0.475 [0.391, 0.548] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.404 [0.306, 0.501] | 0.409 [0.310, 0.511] | 0.413 [0.314, 0.511] | 0.423 [0.319, 0.523] | 0.416 [0.316, 0.513] | 0.361 [0.266, 0.464] | 0.437 [0.334, 0.539] |
| **FLAME 2023 Open vA, coefficienti*** | 0.389 [0.304, 0.469] | 0.392 [0.308, 0.475] | 0.394 [0.310, 0.474] | 0.408 [0.320, 0.490] | 0.399 [0.310, 0.483] | 0.354 [0.270, 0.435] | 0.426 [0.339, 0.509] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.453 [0.375, 0.522] | 0.457 [0.380, 0.527] | 0.453 [0.373, 0.521] | 0.457 [0.378, 0.524] | 0.467 [0.386, 0.539] | 0.437 [0.356, 0.509] | 0.476 [0.395, 0.550] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.409 [0.309, 0.511] | 0.415 [0.314, 0.521] | 0.419 [0.318, 0.520] | 0.427 [0.324, 0.532] | 0.422 [0.321, 0.522] | 0.366 [0.266, 0.471] | 0.441 [0.336, 0.546] |
| **FLAME 2023 Open vB, coefficienti*** | 0.549 [0.467, 0.625] | 0.550 [0.468, 0.627] | 0.548 [0.462, 0.625] | 0.553 [0.469, 0.629] | 0.554 [0.469, 0.631] | 0.546 [0.462, 0.621] | 0.551 [0.468, 0.628] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.630 [0.555, 0.695] | 0.631 [0.557, 0.697] | 0.621 [0.545, 0.685] | 0.630 [0.555, 0.696] | 0.638 [0.562, 0.705] | 0.633 [0.556, 0.700] | 0.642 [0.566, 0.708] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.711 [0.614, 0.798] | 0.712 [0.615, 0.797] | 0.712 [0.617, 0.797] | 0.711 [0.614, 0.798] | 0.714 [0.618, 0.799] | 0.707 [0.613, 0.793] | 0.713 [0.617, 0.799] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.731 [0.657, 0.794] | 0.732 [0.659, 0.795] | 0.726 [0.652, 0.789] | 0.731 [0.657, 0.794] | 0.736 [0.661, 0.800] | 0.732 [0.658, 0.795] | 0.740 [0.668, 0.802] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.300 [0.231, 0.370] | 0.312 [0.239, 0.384] | 0.330 [0.257, 0.405] | 0.310 [0.238, 0.383] | 0.315 [0.240, 0.391] | 0.287 [0.215, 0.363] | 0.265 [0.204, 0.324] |
| **ShapeDNA k=50** | 0.091 [0.061, 0.122] | 0.159 [0.115, 0.206] | 0.144 [0.103, 0.187] | 0.157 [0.113, 0.204] | 0.141 [0.099, 0.183] | -0.018 [-0.082, 0.044] | 0.320 [0.248, 0.399] |
| **HKS globale** | 0.034 [0.013, 0.056] | 0.057 [0.021, 0.093] | 0.050 [0.022, 0.079] | 0.056 [0.021, 0.092] | 0.051 [0.017, 0.087] | 0.005 [-0.050, 0.053] | 0.112 [0.045, 0.183] |
| **WKS globale** | 0.016 [0.000, 0.034] | 0.055 [0.031, 0.080] | 0.048 [0.026, 0.071] | 0.054 [0.029, 0.079] | -0.002 [-0.024, 0.021] | -0.030 [-0.106, 0.042] | 0.190 [0.119, 0.264] |
| **taglia stimata dalla mesh (non oracolo)** | 0.204 [0.136, 0.266] | 0.210 [0.139, 0.274] | 0.160 [0.101, 0.215] | 0.212 [0.140, 0.276] | 0.233 [0.160, 0.300] | 0.216 [0.146, 0.281] | 0.302 [0.214, 0.386] |
| **taglia oracolo (S di FR)** | 0.464 [0.345, 0.571] | 0.464 [0.344, 0.572] | 0.464 [0.344, 0.572] | 0.464 [0.344, 0.572] | 0.465 [0.345, 0.574] | 0.464 [0.344, 0.572] | 0.464 [0.344, 0.573] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.120 [0.083, 0.161] | 0.178 [0.126, 0.236] | 0.165 [0.117, 0.218] | 0.174 [0.122, 0.232] | 0.185 [0.133, 0.243] | 0.096 [0.036, 0.167] | 0.316 [0.224, 0.414] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.107 [0.070, 0.149] | 0.161 [0.109, 0.222] | 0.141 [0.097, 0.192] | 0.159 [0.106, 0.219] | 0.172 [0.119, 0.231] | 0.077 [0.017, 0.145] | 0.292 [0.203, 0.391] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.096 [0.060, 0.133] | 0.123 [0.080, 0.166] | 0.142 [0.096, 0.192] | 0.115 [0.072, 0.159] | 0.144 [0.095, 0.197] | 0.127 [0.057, 0.197] | 0.356 [0.262, 0.449] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.095 [0.061, 0.131] | 0.127 [0.085, 0.172] | 0.156 [0.111, 0.205] | 0.122 [0.081, 0.167] | 0.133 [0.086, 0.183] | 0.108 [0.038, 0.180] | 0.352 [0.252, 0.446] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.147 [0.114, 0.184] | 0.234 [0.186, 0.286] | 0.194 [0.150, 0.240] | 0.232 [0.183, 0.283] | 0.217 [0.168, 0.269] | 0.119 [0.058, 0.183] | 0.441 [0.353, 0.531] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.141 [0.108, 0.176] | 0.222 [0.173, 0.275] | 0.186 [0.141, 0.233] | 0.220 [0.171, 0.273] | 0.206 [0.156, 0.260] | 0.118 [0.061, 0.178] | 0.421 [0.332, 0.514] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.164 [0.139, 0.187] | 0.261 [0.225, 0.293] | 0.234 [0.198, 0.267] | 0.261 [0.224, 0.293] | 0.280 [0.241, 0.314] | 0.029 [-0.012, 0.077] | - |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.167 [0.133, 0.200] | 0.254 [0.210, 0.293] | 0.262 [0.218, 0.303] | 0.256 [0.212, 0.296] | 0.265 [0.218, 0.306] | 0.046 [-0.051, 0.141] | - |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.278 [0.209, 0.342] | 0.328 [0.250, 0.400] | 0.304 [0.238, 0.369] | 0.329 [0.251, 0.401] | 0.335 [0.253, 0.407] | 0.360 [0.269, 0.443] | 0.492 [0.385, 0.581] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.248 [0.185, 0.312] | 0.298 [0.225, 0.370] | 0.289 [0.223, 0.356] | 0.299 [0.226, 0.371] | 0.306 [0.230, 0.378] | 0.342 [0.246, 0.433] | 0.468 [0.369, 0.564] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.675 [0.610, 0.732] | 0.662 [0.592, 0.719] | 0.683 [0.614, 0.741] | 0.734 [0.672, 0.792] | 0.693 [0.626, 0.754] | 0.700 [0.632, 0.757] | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.623 [0.535, 0.700] | 0.605 [0.520, 0.684] | 0.629 [0.539, 0.708] | 0.682 [0.598, 0.755] | 0.637 [0.548, 0.717] | 0.639 [0.552, 0.718] | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.618 [0.534, 0.693] | 0.616 [0.532, 0.691] | 0.619 [0.536, 0.695] | 0.622 [0.539, 0.701] | 0.624 [0.543, 0.700] | 0.630 [0.544, 0.707] | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.575 [0.478, 0.666] | 0.569 [0.474, 0.662] | 0.580 [0.484, 0.673] | 0.603 [0.504, 0.694] | 0.583 [0.484, 0.679] | 0.585 [0.490, 0.679] | - |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.260 [0.226, 0.290] | 0.372 [0.329, 0.407] | 0.347 [0.304, 0.385] | 0.371 [0.329, 0.409] | 0.399 [0.354, 0.438] | 0.232 [0.155, 0.306] | 0.682 [0.613, 0.740] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.259 [0.220, 0.294] | 0.365 [0.317, 0.408] | 0.359 [0.310, 0.404] | 0.367 [0.320, 0.411] | 0.387 [0.336, 0.432] | 0.230 [0.146, 0.304] | 0.621 [0.544, 0.690] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.254 [0.227, 0.278] | 0.366 [0.331, 0.397] | 0.354 [0.317, 0.389] | 0.357 [0.323, 0.390] | 0.369 [0.332, 0.404] | 0.217 [0.158, 0.278] | 0.493 [0.442, 0.541] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.260 [0.227, 0.291] | 0.365 [0.322, 0.406] | 0.362 [0.315, 0.405] | 0.359 [0.314, 0.401] | 0.369 [0.322, 0.413] | 0.253 [0.177, 0.324] | 0.442 [0.388, 0.493] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (88724) | con_perturbazione original (37031) | con_perturbazione remesh (37031) | con_perturbazione down8k (37031) | con_perturbazione up60k (29324) | con_perturbazione noisy (37031) | stessa_topologia senza_crop (24750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.692 [0.623, 0.753] | 0.701 [0.632, 0.762] | 0.696 [0.626, 0.757] | 0.695 [0.625, 0.757] | 0.689 [0.621, 0.747] | 0.693 [0.623, 0.753] | 0.736 [0.672, 0.795] |
| factorized s1234, d_F non cal. | 0.745 [0.677, 0.801] | 0.753 [0.686, 0.809] | 0.753 [0.687, 0.808] | 0.743 [0.675, 0.802] | 0.750 [0.682, 0.805] | 0.747 [0.680, 0.804] | 0.784 [0.721, 0.838] |
| **factorized s1234, d_P** | 0.747 [0.677, 0.804] | 0.756 [0.687, 0.813] | 0.751 [0.683, 0.807] | 0.746 [0.676, 0.805] | 0.753 [0.682, 0.809] | 0.753 [0.682, 0.809] | 0.789 [0.724, 0.843] |
| factorized s2345, d_F cal.* | 0.687 [0.620, 0.748] | 0.696 [0.629, 0.757] | 0.686 [0.620, 0.747] | 0.702 [0.634, 0.762] | 0.681 [0.614, 0.741] | 0.678 [0.610, 0.739] | 0.730 [0.664, 0.790] |
| factorized s2345, d_F non cal. | 0.751 [0.686, 0.809] | 0.755 [0.690, 0.813] | 0.763 [0.701, 0.819] | 0.758 [0.693, 0.816] | 0.760 [0.696, 0.819] | 0.734 [0.663, 0.793] | 0.778 [0.715, 0.833] |
| **factorized s2345, d_P** | 0.753 [0.687, 0.813] | 0.759 [0.693, 0.818] | 0.762 [0.698, 0.820] | 0.761 [0.694, 0.821] | 0.764 [0.700, 0.823] | 0.739 [0.668, 0.799] | 0.783 [0.720, 0.839] |
| ctrlfr s1234 | 0.619 [0.543, 0.688] | 0.645 [0.569, 0.711] | 0.642 [0.565, 0.711] | 0.650 [0.575, 0.716] | 0.639 [0.563, 0.707] | 0.588 [0.512, 0.656] | 0.707 [0.635, 0.772] |
| ctrlfr s2345 | 0.627 [0.554, 0.691] | 0.650 [0.575, 0.715] | 0.628 [0.556, 0.692] | 0.655 [0.582, 0.721] | 0.647 [0.574, 0.713] | 0.634 [0.557, 0.700] | 0.700 [0.627, 0.765] |
| C3M e123, d_F cal.* | 0.687 [0.624, 0.746] | 0.692 [0.630, 0.750] | 0.709 [0.647, 0.767] | 0.682 [0.618, 0.741] | 0.703 [0.640, 0.760] | 0.674 [0.605, 0.736] | 0.727 [0.666, 0.783] |
| C3M e123, d_F non cal. | 0.741 [0.679, 0.796] | 0.743 [0.682, 0.796] | 0.759 [0.702, 0.811] | 0.740 [0.678, 0.794] | 0.754 [0.696, 0.807] | 0.736 [0.670, 0.792] | 0.780 [0.726, 0.830] |
| **C3M e123, d_P** | 0.746 [0.686, 0.799] | 0.748 [0.689, 0.800] | 0.764 [0.709, 0.814] | 0.745 [0.684, 0.798] | 0.760 [0.702, 0.813] | 0.743 [0.678, 0.801] | 0.786 [0.732, 0.836] |
| C3M e205, d_F cal.* | 0.708 [0.649, 0.763] | 0.721 [0.662, 0.775] | 0.727 [0.669, 0.778] | 0.696 [0.634, 0.752] | 0.734 [0.674, 0.784] | 0.694 [0.633, 0.753] | 0.753 [0.699, 0.805] |
| C3M e205, d_F non cal. | 0.751 [0.691, 0.801] | 0.756 [0.697, 0.806] | 0.764 [0.707, 0.813] | 0.743 [0.683, 0.796] | 0.771 [0.712, 0.820] | 0.752 [0.690, 0.802] | 0.789 [0.733, 0.837] |
| **C3M e205, d_P** | 0.756 [0.695, 0.807] | 0.760 [0.701, 0.810] | 0.768 [0.711, 0.815] | 0.747 [0.686, 0.800] | 0.776 [0.718, 0.824] | 0.759 [0.698, 0.809] | 0.794 [0.739, 0.842] |
| **ICP + Chamfer in mm (per coppia)** | 0.477 [0.414, 0.533] | 0.473 [0.411, 0.529] | 0.578 [0.513, 0.638] | 0.415 [0.361, 0.464] | 0.554 [0.488, 0.614] | 0.509 [0.447, 0.566] | 0.639 [0.576, 0.698] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.478 [0.412, 0.536] | 0.462 [0.399, 0.519] | 0.579 [0.512, 0.639] | 0.426 [0.371, 0.477] | 0.557 [0.488, 0.619] | 0.505 [0.440, 0.563] | 0.657 [0.591, 0.714] |
| **NICP per coppia in mm** | 0.406 [0.336, 0.472] | 0.399 [0.331, 0.464] | 0.504 [0.424, 0.577] | 0.345 [0.287, 0.402] | 0.480 [0.403, 0.552] | 0.436 [0.366, 0.502] | - |
| **NICP per coppia cs** | 0.398 [0.329, 0.464] | 0.389 [0.321, 0.452] | 0.495 [0.415, 0.569] | 0.340 [0.282, 0.394] | 0.472 [0.394, 0.544] | 0.427 [0.357, 0.494] | - |
| **NICP su template in mm** | 0.399 [0.320, 0.473] | 0.415 [0.332, 0.496] | 0.410 [0.328, 0.489] | 0.383 [0.308, 0.456] | 0.410 [0.323, 0.490] | 0.381 [0.305, 0.454] | 0.415 [0.334, 0.498] |
| **NICP su template cs** | 0.436 [0.363, 0.508] | 0.427 [0.354, 0.501] | 0.424 [0.351, 0.497] | 0.458 [0.380, 0.530] | 0.429 [0.351, 0.507] | 0.443 [0.370, 0.516] | 0.494 [0.414, 0.571] |
| **varifold in mm (massa unitaria)** | 0.281 [0.246, 0.313] | 0.384 [0.342, 0.422] | 0.380 [0.337, 0.419] | 0.384 [0.342, 0.422] | 0.399 [0.354, 0.438] | 0.309 [0.236, 0.377] | 0.663 [0.597, 0.721] |
| **GNM (visto) preregistrato, coefficienti** | 0.460 [0.376, 0.538] | 0.464 [0.375, 0.543] | 0.466 [0.382, 0.544] | 0.466 [0.381, 0.545] | 0.479 [0.395, 0.559] | 0.433 [0.349, 0.511] | 0.475 [0.390, 0.555] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.371 [0.305, 0.436] | 0.376 [0.311, 0.443] | 0.377 [0.312, 0.444] | 0.368 [0.304, 0.433] | 0.385 [0.319, 0.451] | 0.353 [0.287, 0.419] | 0.384 [0.315, 0.455] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.576 [0.483, 0.663] | 0.578 [0.482, 0.665] | 0.583 [0.491, 0.671] | 0.587 [0.490, 0.675] | 0.594 [0.498, 0.680] | 0.548 [0.454, 0.632] | 0.596 [0.500, 0.681] |
| **GNM (visto) vA, coefficienti*** | 0.477 [0.389, 0.562] | 0.477 [0.388, 0.561] | 0.488 [0.400, 0.573] | 0.485 [0.396, 0.570] | 0.495 [0.405, 0.581] | 0.448 [0.360, 0.533] | 0.490 [0.401, 0.575] |
| GNM (visto) vA, mesh d'identita' FR* | 0.392 [0.327, 0.460] | 0.398 [0.332, 0.467] | 0.399 [0.335, 0.468] | 0.390 [0.326, 0.457] | 0.406 [0.339, 0.474] | 0.373 [0.306, 0.440] | 0.405 [0.337, 0.479] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.570 [0.465, 0.663] | 0.572 [0.466, 0.666] | 0.580 [0.475, 0.675] | 0.581 [0.473, 0.675] | 0.586 [0.480, 0.677] | 0.541 [0.439, 0.632] | 0.587 [0.483, 0.680] |
| **GNM (visto) vB, coefficienti*** | 0.618 [0.548, 0.683] | 0.618 [0.549, 0.682] | 0.618 [0.548, 0.683] | 0.618 [0.550, 0.682] | 0.621 [0.553, 0.687] | 0.618 [0.547, 0.683] | 0.616 [0.547, 0.680] |
| GNM (visto) vB, mesh d'identita' FR* | 0.608 [0.536, 0.683] | 0.611 [0.539, 0.685] | 0.606 [0.534, 0.681] | 0.604 [0.531, 0.679] | 0.614 [0.541, 0.692] | 0.607 [0.531, 0.684] | 0.616 [0.543, 0.695] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.854 [0.803, 0.894] | 0.853 [0.803, 0.892] | 0.855 [0.806, 0.895] | 0.853 [0.802, 0.893] | 0.856 [0.805, 0.896] | 0.854 [0.803, 0.894] | 0.854 [0.805, 0.894] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.783 [0.724, 0.839] | 0.785 [0.727, 0.839] | 0.783 [0.724, 0.838] | 0.780 [0.721, 0.836] | 0.788 [0.730, 0.843] | 0.783 [0.721, 0.840] | 0.789 [0.731, 0.845] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.396 [0.311, 0.476] | 0.404 [0.319, 0.488] | 0.401 [0.317, 0.482] | 0.416 [0.329, 0.496] | 0.409 [0.324, 0.492] | 0.353 [0.265, 0.438] | 0.433 [0.349, 0.515] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.379 [0.305, 0.453] | 0.384 [0.311, 0.461] | 0.382 [0.305, 0.459] | 0.389 [0.315, 0.463] | 0.393 [0.316, 0.470] | 0.352 [0.278, 0.423] | 0.401 [0.325, 0.479] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.469 [0.372, 0.561] | 0.481 [0.379, 0.573] | 0.478 [0.381, 0.567] | 0.491 [0.390, 0.581] | 0.486 [0.382, 0.579] | 0.416 [0.314, 0.512] | 0.507 [0.409, 0.599] |
| **FLAME 2023 Open vA, coefficienti*** | 0.397 [0.312, 0.476] | 0.406 [0.321, 0.488] | 0.402 [0.319, 0.483] | 0.417 [0.330, 0.496] | 0.410 [0.327, 0.491] | 0.354 [0.268, 0.437] | 0.434 [0.349, 0.514] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.373 [0.301, 0.445] | 0.378 [0.305, 0.452] | 0.376 [0.303, 0.451] | 0.381 [0.308, 0.453] | 0.387 [0.313, 0.460] | 0.348 [0.277, 0.417] | 0.392 [0.316, 0.468] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.480 [0.380, 0.575] | 0.491 [0.388, 0.586] | 0.489 [0.391, 0.581] | 0.500 [0.398, 0.594] | 0.498 [0.395, 0.593] | 0.425 [0.319, 0.525] | 0.516 [0.412, 0.609] |
| **FLAME 2023 Open vB, coefficienti*** | 0.592 [0.512, 0.664] | 0.591 [0.510, 0.664] | 0.590 [0.508, 0.661] | 0.594 [0.514, 0.666] | 0.599 [0.518, 0.669] | 0.591 [0.514, 0.660] | 0.594 [0.514, 0.666] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.590 [0.516, 0.662] | 0.590 [0.517, 0.662] | 0.584 [0.512, 0.656] | 0.591 [0.515, 0.663] | 0.599 [0.523, 0.671] | 0.589 [0.516, 0.660] | 0.599 [0.522, 0.673] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.820 [0.757, 0.870] | 0.822 [0.760, 0.870] | 0.821 [0.757, 0.871] | 0.821 [0.757, 0.872] | 0.825 [0.762, 0.874] | 0.814 [0.751, 0.866] | 0.822 [0.759, 0.873] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.742 [0.677, 0.802] | 0.743 [0.679, 0.801] | 0.739 [0.673, 0.795] | 0.744 [0.678, 0.803] | 0.749 [0.683, 0.806] | 0.739 [0.673, 0.801] | 0.750 [0.685, 0.809] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.348 [0.284, 0.414] | 0.365 [0.301, 0.434] | 0.381 [0.314, 0.452] | 0.364 [0.300, 0.433] | 0.369 [0.301, 0.437] | 0.321 [0.252, 0.393] | 0.310 [0.255, 0.366] |
| **ShapeDNA k=50** | 0.117 [0.089, 0.144] | 0.195 [0.155, 0.237] | 0.177 [0.138, 0.213] | 0.193 [0.153, 0.234] | 0.173 [0.132, 0.211] | 0.027 [-0.035, 0.083] | 0.374 [0.300, 0.443] |
| **HKS globale** | 0.042 [0.020, 0.066] | 0.068 [0.031, 0.107] | 0.058 [0.028, 0.089] | 0.068 [0.032, 0.106] | 0.061 [0.027, 0.098] | 0.018 [-0.033, 0.068] | 0.136 [0.066, 0.208] |
| **WKS globale** | 0.017 [0.002, 0.034] | 0.056 [0.032, 0.082] | 0.050 [0.028, 0.073] | 0.056 [0.033, 0.083] | 0.004 [-0.016, 0.024] | -0.033 [-0.112, 0.042] | 0.191 [0.118, 0.268] |
| **taglia stimata dalla mesh (non oracolo)** | 0.094 [0.044, 0.144] | 0.098 [0.044, 0.151] | 0.075 [0.033, 0.114] | 0.099 [0.045, 0.152] | 0.111 [0.053, 0.169] | 0.096 [0.044, 0.147] | 0.133 [0.060, 0.207] |
| **taglia oracolo (S di FR)** | 0.119 [0.022, 0.229] | 0.118 [0.022, 0.228] | 0.118 [0.022, 0.228] | 0.118 [0.022, 0.228] | 0.121 [0.021, 0.230] | 0.118 [0.022, 0.228] | 0.118 [0.023, 0.229] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.123 [0.085, 0.164] | 0.184 [0.129, 0.242] | 0.165 [0.117, 0.215] | 0.180 [0.126, 0.237] | 0.189 [0.135, 0.247] | 0.098 [0.037, 0.167] | 0.328 [0.237, 0.427] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.114 [0.079, 0.154] | 0.172 [0.119, 0.231] | 0.148 [0.104, 0.195] | 0.169 [0.115, 0.228] | 0.182 [0.131, 0.239] | 0.089 [0.030, 0.153] | 0.316 [0.223, 0.416] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.097 [0.063, 0.132] | 0.120 [0.076, 0.163] | 0.140 [0.091, 0.190] | 0.113 [0.069, 0.157] | 0.139 [0.092, 0.190] | 0.152 [0.084, 0.222] | 0.345 [0.248, 0.445] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.103 [0.070, 0.136] | 0.130 [0.086, 0.173] | 0.160 [0.114, 0.210] | 0.125 [0.084, 0.168] | 0.140 [0.096, 0.186] | 0.145 [0.081, 0.211] | 0.351 [0.250, 0.449] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.154 [0.121, 0.190] | 0.250 [0.202, 0.300] | 0.206 [0.161, 0.254] | 0.248 [0.199, 0.299] | 0.231 [0.182, 0.282] | 0.099 [0.047, 0.158] | 0.478 [0.391, 0.567] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.152 [0.121, 0.186] | 0.246 [0.198, 0.297] | 0.205 [0.159, 0.253] | 0.245 [0.196, 0.297] | 0.227 [0.179, 0.278] | 0.104 [0.055, 0.158] | 0.472 [0.386, 0.563] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.136 [0.108, 0.163] | 0.216 [0.178, 0.254] | 0.195 [0.157, 0.232] | 0.216 [0.177, 0.254] | 0.229 [0.187, 0.271] | 0.024 [-0.025, 0.078] | - |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.178 [0.146, 0.209] | 0.266 [0.226, 0.301] | 0.272 [0.230, 0.309] | 0.268 [0.228, 0.306] | 0.274 [0.231, 0.314] | 0.072 [-0.022, 0.163] | - |
| LPIPS alex su render [profondita'], mm (FR) | 0.272 [0.205, 0.338] | 0.322 [0.248, 0.394] | 0.297 [0.232, 0.361] | 0.323 [0.249, 0.395] | 0.329 [0.251, 0.402] | 0.350 [0.258, 0.442] | 0.485 [0.389, 0.573] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.264 [0.203, 0.325] | 0.317 [0.251, 0.382] | 0.306 [0.246, 0.367] | 0.318 [0.252, 0.383] | 0.324 [0.255, 0.393] | 0.369 [0.276, 0.459] | 0.499 [0.412, 0.582] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.701 [0.642, 0.753] | 0.688 [0.629, 0.741] | 0.712 [0.652, 0.764] | 0.761 [0.706, 0.809] | 0.721 [0.662, 0.774] | 0.724 [0.665, 0.777] | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.725 [0.671, 0.775] | 0.707 [0.650, 0.761] | 0.734 [0.680, 0.785] | 0.788 [0.740, 0.833] | 0.744 [0.688, 0.795] | 0.744 [0.690, 0.795] | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.718 [0.654, 0.774] | 0.717 [0.650, 0.773] | 0.719 [0.652, 0.775] | 0.726 [0.662, 0.781] | 0.726 [0.663, 0.781] | 0.732 [0.667, 0.787] | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.672 [0.604, 0.736] | 0.668 [0.599, 0.731] | 0.679 [0.610, 0.743] | 0.705 [0.637, 0.767] | 0.681 [0.611, 0.745] | 0.686 [0.617, 0.749] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.253 [0.219, 0.284] | 0.363 [0.322, 0.399] | 0.343 [0.301, 0.382] | 0.362 [0.321, 0.399] | 0.389 [0.347, 0.425] | 0.219 [0.141, 0.299] | 0.654 [0.587, 0.713] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.283 [0.251, 0.312] | 0.400 [0.363, 0.435] | 0.395 [0.356, 0.431] | 0.402 [0.362, 0.437] | 0.425 [0.385, 0.458] | 0.246 [0.166, 0.320] | 0.676 [0.612, 0.733] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.260 [0.234, 0.284] | 0.373 [0.338, 0.405] | 0.366 [0.330, 0.399] | 0.365 [0.329, 0.397] | 0.380 [0.344, 0.415] | 0.216 [0.154, 0.275] | 0.501 [0.448, 0.549] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.301 [0.276, 0.322] | 0.421 [0.389, 0.448] | 0.420 [0.388, 0.446] | 0.414 [0.380, 0.442] | 0.429 [0.395, 0.459] | 0.293 [0.213, 0.363] | 0.505 [0.463, 0.542] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.207 [+0.160, +0.257] | +0.218 [+0.166, +0.271] | +0.117 [+0.067, +0.169] | +0.266 [+0.221, +0.311] | +0.136 [+0.087, +0.188] | +0.174 [+0.122, +0.225] | +0.094 [+0.040, +0.146] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.245 [+0.192, +0.300] | +0.266 [+0.211, +0.322] | +0.163 [+0.109, +0.225] | +0.287 [+0.239, +0.337] | +0.178 [+0.122, +0.237] | +0.222 [+0.164, +0.278] | +0.132 [+0.075, +0.193] |
| NICP per coppia in mm | +0.286 [+0.229, +0.345] | +0.301 [+0.243, +0.362] | +0.202 [+0.143, +0.266] | +0.341 [+0.287, +0.395] | +0.220 [+0.160, +0.279] | +0.257 [+0.195, +0.320] | - |
| NICP per coppia cs | +0.314 [+0.260, +0.370] | +0.329 [+0.268, +0.388] | +0.236 [+0.177, +0.299] | +0.364 [+0.310, +0.414] | +0.251 [+0.192, +0.313] | +0.289 [+0.229, +0.350] | - |
| NICP su template in mm | +0.119 [+0.030, +0.210] | +0.105 [+0.017, +0.200] | +0.109 [+0.022, +0.202] | +0.152 [+0.066, +0.246] | +0.102 [+0.012, +0.198] | +0.131 [+0.039, +0.225] | +0.136 [+0.050, +0.230] |
| NICP su template cs | +0.261 [+0.180, +0.339] | +0.273 [+0.196, +0.351] | +0.275 [+0.195, +0.350] | +0.241 [+0.157, +0.326] | +0.265 [+0.180, +0.347] | +0.257 [+0.178, +0.337] | +0.247 [+0.160, +0.331] |
| varifold in mm (massa unitaria) | +0.396 [+0.349, +0.437] | +0.301 [+0.255, +0.342] | +0.302 [+0.258, +0.342] | +0.294 [+0.246, +0.336] | +0.278 [+0.231, +0.319] | +0.402 [+0.337, +0.471] | +0.063 [+0.017, +0.119] |
| GNM (visto) preregistrato, coefficienti | +0.173 [+0.086, +0.264] | +0.181 [+0.091, +0.281] | +0.173 [+0.087, +0.263] | +0.173 [+0.086, +0.268] | +0.155 [+0.064, +0.248] | +0.190 [+0.103, +0.282] | +0.200 [+0.113, +0.293] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.196 [+0.108, +0.288] | +0.201 [+0.114, +0.292] | +0.191 [+0.104, +0.280] | +0.205 [+0.117, +0.294] | +0.182 [+0.093, +0.273] | +0.209 [+0.114, +0.304] | +0.222 [+0.132, +0.315] |
| GNM (visto) vA, coefficienti | +0.183 [+0.092, +0.279] | +0.194 [+0.098, +0.295] | +0.178 [+0.084, +0.273] | +0.179 [+0.085, +0.277] | +0.167 [+0.077, +0.261] | +0.201 [+0.112, +0.300] | +0.213 [+0.120, +0.310] |
| GNM (visto) vA, mesh d'identita' FR | +0.177 [+0.089, +0.266] | +0.181 [+0.096, +0.271] | +0.172 [+0.084, +0.258] | +0.184 [+0.097, +0.274] | +0.164 [+0.071, +0.253] | +0.192 [+0.100, +0.282] | +0.204 [+0.117, +0.296] |
| GNM (visto) vB, coefficienti | +0.095 [+0.016, +0.163] | +0.104 [+0.024, +0.173] | +0.101 [+0.022, +0.168] | +0.092 [+0.017, +0.162] | +0.092 [+0.016, +0.163] | +0.096 [+0.015, +0.164] | +0.141 [+0.065, +0.212] |
| GNM (visto) vB, mesh d'identita' FR | -0.018 [-0.085, +0.043] | -0.011 [-0.080, +0.049] | -0.013 [-0.081, +0.047] | -0.010 [-0.077, +0.050] | -0.023 [-0.092, +0.037] | -0.018 [-0.089, +0.043] | +0.015 [-0.052, +0.075] |
| GNM (visto) vB, composizione S_B + k d_P | -0.139 [-0.194, -0.090] | -0.131 [-0.186, -0.083] | -0.136 [-0.190, -0.086] | -0.133 [-0.189, -0.081] | -0.143 [-0.199, -0.096] | -0.141 [-0.197, -0.092] | -0.102 [-0.157, -0.054] |
| FLAME 2023 Open preregistrato, coefficienti | +0.276 [+0.189, +0.374] | +0.281 [+0.190, +0.379] | +0.275 [+0.187, +0.369] | +0.257 [+0.168, +0.358] | +0.265 [+0.177, +0.363] | +0.310 [+0.220, +0.414] | +0.283 [+0.193, +0.381] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.212 [+0.123, +0.305] | +0.217 [+0.125, +0.312] | +0.217 [+0.128, +0.312] | +0.206 [+0.117, +0.299] | +0.198 [+0.106, +0.294] | +0.230 [+0.136, +0.333] | +0.230 [+0.143, +0.326] |
| FLAME 2023 Open vA, coefficienti | +0.272 [+0.186, +0.368] | +0.277 [+0.188, +0.372] | +0.271 [+0.183, +0.364] | +0.253 [+0.164, +0.353] | +0.261 [+0.173, +0.357] | +0.306 [+0.216, +0.410] | +0.279 [+0.190, +0.375] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.207 [+0.117, +0.300] | +0.212 [+0.121, +0.303] | +0.212 [+0.122, +0.305] | +0.204 [+0.117, +0.297] | +0.194 [+0.102, +0.284] | +0.224 [+0.130, +0.328] | +0.229 [+0.140, +0.322] |
| FLAME 2023 Open vB, coefficienti | +0.111 [+0.038, +0.178] | +0.119 [+0.042, +0.189] | +0.117 [+0.044, +0.184] | +0.108 [+0.035, +0.176] | +0.106 [+0.032, +0.172] | +0.115 [+0.040, +0.182] | +0.154 [+0.083, +0.222] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.031 [-0.036, +0.089] | +0.038 [-0.023, +0.096] | +0.045 [-0.023, +0.103] | +0.031 [-0.035, +0.091] | +0.022 [-0.046, +0.083] | +0.027 [-0.042, +0.090] | +0.063 [-0.002, +0.122] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.070 [-0.127, -0.017] | -0.063 [-0.121, -0.009] | -0.060 [-0.120, -0.007] | -0.070 [-0.128, -0.018] | -0.076 [-0.134, -0.021] | -0.071 [-0.128, -0.016] | -0.035 [-0.090, +0.018] |
| ArcFace-render, normal map 3 viste (G0) | +0.360 [+0.285, +0.432] | +0.357 [+0.283, +0.433] | +0.336 [+0.258, +0.411] | +0.351 [+0.276, +0.425] | +0.345 [+0.269, +0.419] | +0.373 [+0.282, +0.458] | +0.440 [+0.372, +0.510] |
| ShapeDNA k=50 | +0.570 [+0.502, +0.628] | +0.510 [+0.443, +0.570] | +0.521 [+0.457, +0.577] | +0.504 [+0.436, +0.565] | +0.519 [+0.452, +0.579] | +0.679 [+0.587, +0.771] | +0.386 [+0.311, +0.462] |
| HKS globale | +0.627 [+0.557, +0.687] | +0.612 [+0.541, +0.675] | +0.615 [+0.544, +0.677] | +0.605 [+0.533, +0.670] | +0.609 [+0.537, +0.673] | +0.655 [+0.578, +0.732] | +0.593 [+0.506, +0.677] |
| WKS globale | +0.644 [+0.572, +0.707] | +0.614 [+0.547, +0.677] | +0.617 [+0.551, +0.679] | +0.607 [+0.537, +0.671] | +0.662 [+0.589, +0.725] | +0.691 [+0.591, +0.786] | +0.516 [+0.430, +0.605] |
| taglia stimata dalla mesh (non oracolo) | +0.456 [+0.354, +0.560] | +0.459 [+0.352, +0.563] | +0.505 [+0.407, +0.602] | +0.449 [+0.343, +0.555] | +0.427 [+0.319, +0.530] | +0.444 [+0.340, +0.552] | +0.403 [+0.285, +0.519] |
| taglia oracolo (S di FR) | +0.196 [+0.047, +0.355] | +0.205 [+0.055, +0.364] | +0.201 [+0.053, +0.359] | +0.197 [+0.046, +0.357] | +0.195 [+0.048, +0.351] | +0.196 [+0.047, +0.357] | +0.241 [+0.092, +0.398] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.541 [+0.477, +0.602] | +0.491 [+0.424, +0.557] | +0.501 [+0.434, +0.562] | +0.487 [+0.418, +0.553] | +0.475 [+0.413, +0.539] | +0.564 [+0.485, +0.641] | +0.389 [+0.299, +0.485] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.565 [+0.498, +0.631] | +0.546 [+0.479, +0.612] | +0.523 [+0.451, +0.590] | +0.546 [+0.476, +0.613] | +0.516 [+0.444, +0.585] | +0.534 [+0.446, +0.630] | +0.349 [+0.258, +0.438] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.513 [+0.449, +0.570] | +0.435 [+0.372, +0.495] | +0.471 [+0.407, +0.527] | +0.429 [+0.365, +0.488] | +0.443 [+0.385, +0.497] | +0.541 [+0.455, +0.623] | +0.264 [+0.185, +0.343] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.496 [+0.433, +0.555] | +0.408 [+0.344, +0.471] | +0.431 [+0.369, +0.491] | +0.400 [+0.335, +0.462] | +0.380 [+0.317, +0.439] | +0.632 [+0.558, +0.703] | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.383 [+0.316, +0.447] | +0.341 [+0.270, +0.409] | +0.361 [+0.297, +0.421] | +0.332 [+0.261, +0.400] | +0.325 [+0.257, +0.393] | +0.301 [+0.224, +0.380] | +0.213 [+0.132, +0.302] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.015 [-0.072, +0.045] | +0.007 [-0.055, +0.070] | -0.017 [-0.077, +0.043] | -0.073 [-0.132, -0.015] | -0.033 [-0.091, +0.028] | -0.040 [-0.104, +0.023] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.038 [-0.026, +0.104] | +0.064 [-0.004, +0.133] | +0.036 [-0.029, +0.106] | -0.021 [-0.085, +0.044] | +0.023 [-0.041, +0.092] | +0.021 [-0.048, +0.090] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.043 [-0.028, +0.114] | +0.052 [-0.019, +0.124] | +0.047 [-0.023, +0.121] | +0.039 [-0.031, +0.109] | +0.036 [-0.036, +0.106] | +0.030 [-0.044, +0.104] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.086 [+0.010, +0.167] | +0.100 [+0.023, +0.183] | +0.086 [+0.010, +0.167] | +0.058 [-0.020, +0.140] | +0.078 [+0.003, +0.158] | +0.075 [-0.002, +0.156] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.401 [+0.346, +0.449] | +0.297 [+0.246, +0.344] | +0.318 [+0.271, +0.363] | +0.290 [+0.237, +0.335] | +0.261 [+0.210, +0.305] | +0.429 [+0.352, +0.503] | +0.024 [-0.030, +0.076] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.407 [+0.348, +0.460] | +0.303 [+0.247, +0.356] | +0.311 [+0.261, +0.361] | +0.304 [+0.246, +0.356] | +0.291 [+0.236, +0.340] | +0.443 [+0.358, +0.524] | +0.212 [+0.160, +0.266] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.234 [+0.183, +0.285] | +0.244 [+0.190, +0.299] | +0.149 [+0.097, +0.199] | +0.289 [+0.240, +0.336] | +0.171 [+0.121, +0.225] | +0.201 [+0.145, +0.254] | +0.113 [+0.060, +0.167] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.272 [+0.220, +0.321] | +0.293 [+0.238, +0.344] | +0.195 [+0.142, +0.247] | +0.310 [+0.262, +0.355] | +0.213 [+0.159, +0.266] | +0.248 [+0.190, +0.302] | +0.151 [+0.095, +0.205] |
| NICP per coppia in mm | +0.313 [+0.253, +0.375] | +0.327 [+0.265, +0.389] | +0.234 [+0.171, +0.297] | +0.364 [+0.309, +0.417] | +0.255 [+0.195, +0.318] | +0.283 [+0.217, +0.350] | - |
| NICP per coppia cs | +0.341 [+0.282, +0.400] | +0.356 [+0.294, +0.415] | +0.268 [+0.205, +0.329] | +0.386 [+0.333, +0.438] | +0.287 [+0.228, +0.348] | +0.316 [+0.251, +0.377] | - |
| NICP su template in mm | +0.146 [+0.055, +0.241] | +0.131 [+0.040, +0.230] | +0.140 [+0.052, +0.234] | +0.175 [+0.083, +0.269] | +0.137 [+0.042, +0.234] | +0.157 [+0.059, +0.253] | +0.155 [+0.064, +0.254] |
| NICP su template cs | +0.287 [+0.206, +0.369] | +0.299 [+0.220, +0.380] | +0.307 [+0.226, +0.384] | +0.264 [+0.177, +0.351] | +0.300 [+0.216, +0.386] | +0.283 [+0.199, +0.366] | +0.266 [+0.180, +0.355] |
| varifold in mm (massa unitaria) | +0.423 [+0.376, +0.467] | +0.328 [+0.280, +0.369] | +0.334 [+0.291, +0.373] | +0.317 [+0.269, +0.360] | +0.313 [+0.269, +0.352] | +0.429 [+0.365, +0.494] | +0.083 [+0.035, +0.131] |
| GNM (visto) preregistrato, coefficienti | +0.200 [+0.103, +0.295] | +0.207 [+0.108, +0.305] | +0.204 [+0.108, +0.296] | +0.195 [+0.098, +0.293] | +0.191 [+0.091, +0.289] | +0.216 [+0.117, +0.315] | +0.219 [+0.121, +0.313] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.223 [+0.121, +0.321] | +0.227 [+0.124, +0.328] | +0.223 [+0.122, +0.321] | +0.228 [+0.130, +0.325] | +0.218 [+0.114, +0.317] | +0.235 [+0.129, +0.341] | +0.242 [+0.139, +0.344] |
| GNM (visto) vA, coefficienti | +0.210 [+0.111, +0.308] | +0.221 [+0.121, +0.322] | +0.210 [+0.114, +0.303] | +0.202 [+0.103, +0.301] | +0.202 [+0.108, +0.299] | +0.228 [+0.132, +0.328] | +0.232 [+0.134, +0.330] |
| GNM (visto) vA, mesh d'identita' FR | +0.203 [+0.106, +0.300] | +0.208 [+0.108, +0.307] | +0.203 [+0.106, +0.297] | +0.207 [+0.111, +0.302] | +0.199 [+0.101, +0.294] | +0.218 [+0.116, +0.322] | +0.223 [+0.123, +0.325] |
| GNM (visto) vB, coefficienti | +0.122 [+0.045, +0.190] | +0.130 [+0.051, +0.200] | +0.133 [+0.056, +0.202] | +0.115 [+0.040, +0.186] | +0.127 [+0.049, +0.195] | +0.122 [+0.043, +0.192] | +0.161 [+0.088, +0.228] |
| GNM (visto) vB, mesh d'identita' FR | +0.009 [-0.065, +0.081] | +0.015 [-0.061, +0.086] | +0.018 [-0.057, +0.088] | +0.013 [-0.065, +0.086] | +0.012 [-0.063, +0.083] | +0.008 [-0.074, +0.080] | +0.035 [-0.042, +0.106] |
| GNM (visto) vB, composizione S_B + k d_P | -0.112 [-0.175, -0.058] | -0.105 [-0.167, -0.052] | -0.105 [-0.166, -0.051] | -0.110 [-0.170, -0.053] | -0.108 [-0.168, -0.056] | -0.114 [-0.176, -0.059] | -0.083 [-0.142, -0.030] |
| FLAME 2023 Open preregistrato, coefficienti | +0.303 [+0.212, +0.403] | +0.307 [+0.212, +0.406] | +0.307 [+0.216, +0.405] | +0.280 [+0.187, +0.380] | +0.300 [+0.205, +0.401] | +0.337 [+0.241, +0.444] | +0.303 [+0.210, +0.402] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.238 [+0.139, +0.337] | +0.243 [+0.144, +0.343] | +0.249 [+0.150, +0.348] | +0.229 [+0.127, +0.328] | +0.233 [+0.131, +0.331] | +0.256 [+0.153, +0.369] | +0.250 [+0.150, +0.351] |
| FLAME 2023 Open vA, coefficienti | +0.299 [+0.210, +0.397] | +0.303 [+0.210, +0.402] | +0.303 [+0.212, +0.399] | +0.276 [+0.184, +0.375] | +0.296 [+0.201, +0.395] | +0.333 [+0.242, +0.440] | +0.298 [+0.207, +0.397] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.234 [+0.136, +0.334] | +0.239 [+0.138, +0.339] | +0.244 [+0.148, +0.341] | +0.227 [+0.126, +0.322] | +0.229 [+0.126, +0.325] | +0.250 [+0.145, +0.361] | +0.248 [+0.148, +0.349] |
| FLAME 2023 Open vB, coefficienti | +0.138 [+0.061, +0.211] | +0.145 [+0.065, +0.220] | +0.149 [+0.069, +0.222] | +0.131 [+0.053, +0.204] | +0.141 [+0.062, +0.212] | +0.141 [+0.061, +0.216] | +0.173 [+0.098, +0.245] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.057 [-0.018, +0.125] | +0.065 [-0.008, +0.130] | +0.076 [+0.002, +0.143] | +0.054 [-0.022, +0.122] | +0.057 [-0.019, +0.127] | +0.054 [-0.028, +0.124] | +0.082 [+0.009, +0.152] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.044 [-0.110, +0.013] | -0.036 [-0.102, +0.019] | -0.029 [-0.094, +0.026] | -0.048 [-0.111, +0.008] | -0.040 [-0.107, +0.018] | -0.045 [-0.111, +0.015] | -0.016 [-0.080, +0.040] |
| ArcFace-render, normal map 3 viste (G0) | +0.387 [+0.311, +0.459] | +0.384 [+0.310, +0.459] | +0.367 [+0.292, +0.442] | +0.373 [+0.299, +0.447] | +0.380 [+0.304, +0.453] | +0.400 [+0.303, +0.487] | +0.459 [+0.392, +0.532] |
| ShapeDNA k=50 | +0.597 [+0.532, +0.658] | +0.536 [+0.469, +0.600] | +0.553 [+0.489, +0.612] | +0.527 [+0.458, +0.595] | +0.555 [+0.488, +0.619] | +0.705 [+0.612, +0.797] | +0.405 [+0.332, +0.481] |
| HKS globale | +0.654 [+0.586, +0.718] | +0.639 [+0.567, +0.705] | +0.647 [+0.578, +0.710] | +0.627 [+0.558, +0.695] | +0.644 [+0.571, +0.708] | +0.682 [+0.598, +0.761] | +0.613 [+0.531, +0.693] |
| WKS globale | +0.671 [+0.600, +0.739] | +0.640 [+0.573, +0.709] | +0.649 [+0.583, +0.716] | +0.630 [+0.559, +0.701] | +0.697 [+0.624, +0.767] | +0.717 [+0.612, +0.822] | +0.535 [+0.445, +0.626] |
| taglia stimata dalla mesh (non oracolo) | +0.483 [+0.371, +0.592] | +0.485 [+0.371, +0.595] | +0.537 [+0.432, +0.634] | +0.472 [+0.356, +0.581] | +0.462 [+0.349, +0.571] | +0.471 [+0.357, +0.579] | +0.422 [+0.300, +0.542] |
| taglia oracolo (S di FR) | +0.223 [+0.070, +0.387] | +0.231 [+0.080, +0.394] | +0.233 [+0.081, +0.398] | +0.220 [+0.066, +0.386] | +0.230 [+0.076, +0.391] | +0.223 [+0.069, +0.390] | +0.261 [+0.110, +0.423] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.567 [+0.502, +0.631] | +0.518 [+0.451, +0.585] | +0.532 [+0.465, +0.596] | +0.510 [+0.441, +0.576] | +0.510 [+0.445, +0.577] | +0.591 [+0.509, +0.669] | +0.408 [+0.323, +0.500] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.591 [+0.522, +0.658] | +0.572 [+0.502, +0.641] | +0.554 [+0.487, +0.624] | +0.568 [+0.497, +0.637] | +0.551 [+0.475, +0.624] | +0.560 [+0.474, +0.656] | +0.369 [+0.281, +0.456] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.540 [+0.474, +0.602] | +0.462 [+0.394, +0.527] | +0.502 [+0.437, +0.563] | +0.452 [+0.386, +0.514] | +0.478 [+0.417, +0.538] | +0.568 [+0.474, +0.656] | +0.283 [+0.205, +0.364] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.523 [+0.456, +0.585] | +0.434 [+0.367, +0.500] | +0.463 [+0.398, +0.524] | +0.423 [+0.359, +0.488] | +0.415 [+0.350, +0.479] | +0.658 [+0.580, +0.732] | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.410 [+0.345, +0.472] | +0.367 [+0.299, +0.436] | +0.392 [+0.332, +0.452] | +0.355 [+0.290, +0.424] | +0.360 [+0.296, +0.425] | +0.327 [+0.248, +0.409] | +0.232 [+0.155, +0.317] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | +0.012 [-0.052, +0.075] | +0.033 [-0.031, +0.099] | +0.014 [-0.047, +0.076] | -0.051 [-0.113, +0.008] | +0.002 [-0.060, +0.067] | -0.013 [-0.079, +0.053] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.065 [-0.002, +0.124] | +0.090 [+0.021, +0.152] | +0.068 [+0.001, +0.130] | +0.002 [-0.063, +0.059] | +0.058 [-0.007, +0.118] | +0.048 [-0.022, +0.110] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.070 [-0.002, +0.138] | +0.079 [+0.005, +0.148] | +0.078 [+0.006, +0.146] | +0.061 [-0.009, +0.130] | +0.071 [-0.000, +0.137] | +0.057 [-0.020, +0.128] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.113 [+0.039, +0.187] | +0.126 [+0.051, +0.203] | +0.117 [+0.043, +0.192] | +0.081 [+0.007, +0.155] | +0.113 [+0.038, +0.189] | +0.102 [+0.025, +0.176] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.428 [+0.372, +0.478] | +0.324 [+0.270, +0.369] | +0.350 [+0.301, +0.392] | +0.312 [+0.257, +0.360] | +0.296 [+0.245, +0.340] | +0.455 [+0.377, +0.536] | +0.043 [-0.008, +0.094] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.433 [+0.372, +0.489] | +0.330 [+0.274, +0.382] | +0.343 [+0.290, +0.392] | +0.326 [+0.269, +0.380] | +0.327 [+0.273, +0.376] | +0.470 [+0.379, +0.555] | +0.231 [+0.182, +0.281] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.269 [+0.217, +0.317] | +0.283 [+0.230, +0.331] | +0.172 [+0.118, +0.222] | +0.331 [+0.281, +0.375] | +0.198 [+0.143, +0.249] | +0.244 [+0.191, +0.294] | +0.150 [+0.095, +0.200] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.269 [+0.217, +0.317] | +0.294 [+0.238, +0.343] | +0.171 [+0.121, +0.221] | +0.320 [+0.273, +0.362] | +0.196 [+0.141, +0.247] | +0.248 [+0.195, +0.298] | +0.132 [+0.078, +0.183] |
| NICP per coppia in mm | +0.340 [+0.283, +0.393] | +0.357 [+0.299, +0.413] | +0.246 [+0.186, +0.303] | +0.400 [+0.348, +0.450] | +0.272 [+0.213, +0.329] | +0.318 [+0.257, +0.373] | - |
| NICP per coppia cs | +0.349 [+0.291, +0.401] | +0.367 [+0.306, +0.422] | +0.256 [+0.195, +0.311] | +0.405 [+0.351, +0.456] | +0.281 [+0.219, +0.338] | +0.326 [+0.266, +0.381] | - |
| NICP su template in mm | +0.348 [+0.259, +0.423] | +0.341 [+0.254, +0.421] | +0.340 [+0.255, +0.416] | +0.363 [+0.281, +0.441] | +0.343 [+0.253, +0.424] | +0.372 [+0.284, +0.450] | +0.374 [+0.286, +0.452] |
| NICP su template cs | +0.311 [+0.236, +0.389] | +0.329 [+0.251, +0.408] | +0.326 [+0.251, +0.403] | +0.288 [+0.205, +0.371] | +0.324 [+0.241, +0.406] | +0.310 [+0.234, +0.384] | +0.295 [+0.213, +0.379] |
| varifold in mm (massa unitaria) | +0.465 [+0.418, +0.503] | +0.372 [+0.330, +0.408] | +0.370 [+0.329, +0.404] | +0.362 [+0.315, +0.401] | +0.354 [+0.313, +0.388] | +0.444 [+0.388, +0.503] | +0.126 [+0.084, +0.171] |
| GNM (visto) preregistrato, coefficienti | +0.286 [+0.197, +0.376] | +0.292 [+0.202, +0.387] | +0.285 [+0.196, +0.374] | +0.280 [+0.187, +0.370] | +0.274 [+0.184, +0.363] | +0.320 [+0.232, +0.409] | +0.314 [+0.223, +0.404] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.170 [+0.075, +0.264] | +0.178 [+0.086, +0.272] | +0.167 [+0.074, +0.258] | +0.159 [+0.063, +0.256] | +0.159 [+0.069, +0.249] | +0.206 [+0.107, +0.301] | +0.193 [+0.104, +0.286] |
| GNM (visto) vA, coefficienti | +0.269 [+0.174, +0.360] | +0.280 [+0.184, +0.376] | +0.262 [+0.167, +0.353] | +0.261 [+0.164, +0.354] | +0.258 [+0.163, +0.349] | +0.305 [+0.210, +0.399] | +0.299 [+0.204, +0.392] |
| GNM (visto) vA, mesh d'identita' SR | +0.176 [+0.077, +0.278] | +0.184 [+0.085, +0.288] | +0.171 [+0.074, +0.272] | +0.165 [+0.064, +0.270] | +0.166 [+0.069, +0.267] | +0.212 [+0.109, +0.317] | +0.202 [+0.104, +0.303] |
| GNM (visto) vB, coefficienti | +0.129 [+0.051, +0.198] | +0.138 [+0.061, +0.206] | +0.132 [+0.055, +0.202] | +0.127 [+0.051, +0.198] | +0.131 [+0.053, +0.206] | +0.136 [+0.057, +0.206] | +0.173 [+0.098, +0.240] |
| GNM (visto) vB, mesh d'identita' SR | -0.107 [-0.155, -0.065] | -0.097 [-0.145, -0.054] | -0.105 [-0.153, -0.062] | -0.107 [-0.156, -0.063] | -0.103 [-0.150, -0.063] | -0.100 [-0.149, -0.058] | -0.065 [-0.111, -0.022] |
| FLAME 2023 Open preregistrato, coefficienti | +0.350 [+0.261, +0.442] | +0.352 [+0.256, +0.445] | +0.349 [+0.259, +0.439] | +0.330 [+0.238, +0.422] | +0.343 [+0.251, +0.433] | +0.400 [+0.307, +0.490] | +0.356 [+0.266, +0.449] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.277 [+0.177, +0.374] | +0.275 [+0.168, +0.380] | +0.273 [+0.176, +0.365] | +0.255 [+0.152, +0.356] | +0.266 [+0.166, +0.362] | +0.338 [+0.236, +0.439] | +0.282 [+0.183, +0.382] |
| FLAME 2023 Open vA, coefficienti | +0.349 [+0.261, +0.440] | +0.350 [+0.256, +0.442] | +0.348 [+0.259, +0.437] | +0.329 [+0.237, +0.422] | +0.343 [+0.251, +0.432] | +0.399 [+0.306, +0.490] | +0.355 [+0.266, +0.446] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.267 [+0.166, +0.365] | +0.265 [+0.156, +0.369] | +0.262 [+0.163, +0.358] | +0.246 [+0.141, +0.348] | +0.255 [+0.156, +0.352] | +0.328 [+0.224, +0.432] | +0.273 [+0.170, +0.375] |
| FLAME 2023 Open vB, coefficienti | +0.154 [+0.085, +0.220] | +0.165 [+0.094, +0.232] | +0.161 [+0.090, +0.227] | +0.151 [+0.080, +0.218] | +0.154 [+0.084, +0.221] | +0.163 [+0.093, +0.227] | +0.195 [+0.129, +0.262] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.074 [-0.129, -0.022] | -0.065 [-0.120, -0.013] | -0.070 [-0.126, -0.016] | -0.075 [-0.132, -0.020] | -0.072 [-0.126, -0.020] | -0.061 [-0.115, -0.010] | -0.033 [-0.088, +0.019] |
| ArcFace-render, normal map 3 viste (G0) | +0.399 [+0.328, +0.461] | +0.391 [+0.317, +0.454] | +0.369 [+0.296, +0.434] | +0.382 [+0.310, +0.443] | +0.384 [+0.308, +0.449] | +0.433 [+0.343, +0.513] | +0.479 [+0.418, +0.538] |
| ShapeDNA k=50 | +0.629 [+0.567, +0.682] | +0.561 [+0.502, +0.618] | +0.574 [+0.516, +0.626] | +0.553 [+0.490, +0.613] | +0.580 [+0.519, +0.636] | +0.726 [+0.638, +0.806] | +0.415 [+0.347, +0.489] |
| HKS globale | +0.704 [+0.638, +0.761] | +0.688 [+0.618, +0.750] | +0.692 [+0.621, +0.751] | +0.678 [+0.603, +0.743] | +0.691 [+0.618, +0.756] | +0.735 [+0.659, +0.806] | +0.653 [+0.564, +0.739] |
| WKS globale | +0.729 [+0.665, +0.785] | +0.700 [+0.640, +0.755] | +0.701 [+0.642, +0.755] | +0.690 [+0.629, +0.747] | +0.749 [+0.679, +0.806] | +0.787 [+0.695, +0.881] | +0.598 [+0.513, +0.685] |
| taglia stimata dalla mesh (non oracolo) | +0.652 [+0.574, +0.729] | +0.659 [+0.579, +0.735] | +0.676 [+0.603, +0.741] | +0.647 [+0.566, +0.723] | +0.642 [+0.560, +0.723] | +0.658 [+0.578, +0.738] | +0.656 [+0.566, +0.745] |
| taglia oracolo (S di FR) | +0.628 [+0.503, +0.741] | +0.638 [+0.515, +0.753] | +0.632 [+0.507, +0.745] | +0.627 [+0.503, +0.743] | +0.632 [+0.507, +0.749] | +0.635 [+0.511, +0.746] | +0.671 [+0.547, +0.783] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.632 [+0.571, +0.688] | +0.584 [+0.517, +0.644] | +0.602 [+0.542, +0.657] | +0.577 [+0.508, +0.637] | +0.571 [+0.506, +0.628] | +0.664 [+0.584, +0.737] | +0.473 [+0.380, +0.560] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.644 [+0.588, +0.699] | +0.626 [+0.571, +0.681] | +0.590 [+0.535, +0.647] | +0.621 [+0.563, +0.677] | +0.612 [+0.554, +0.671] | +0.608 [+0.530, +0.693] | +0.438 [+0.358, +0.521] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.594 [+0.535, +0.648] | +0.511 [+0.447, +0.570] | +0.546 [+0.485, +0.604] | +0.501 [+0.437, +0.561] | +0.525 [+0.462, +0.583] | +0.649 [+0.572, +0.714] | +0.317 [+0.230, +0.403] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.569 [+0.513, +0.616] | +0.490 [+0.436, +0.538] | +0.479 [+0.427, +0.525] | +0.477 [+0.423, +0.525] | +0.478 [+0.428, +0.523] | +0.681 [+0.572, +0.782] | - |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.482 [+0.427, +0.536] | +0.439 [+0.378, +0.498] | +0.445 [+0.389, +0.499] | +0.428 [+0.366, +0.488] | +0.428 [+0.371, +0.486] | +0.385 [+0.312, +0.460] | +0.290 [+0.218, +0.361] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | +0.045 [-0.013, +0.100] | +0.068 [+0.009, +0.126] | +0.039 [-0.019, +0.094] | -0.015 [-0.074, +0.036] | +0.031 [-0.027, +0.086] | +0.029 [-0.032, +0.086] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.022 [-0.032, +0.072] | +0.049 [-0.005, +0.104] | +0.017 [-0.035, +0.067] | -0.042 [-0.095, +0.003] | +0.009 [-0.044, +0.058] | +0.010 [-0.046, +0.061] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.028 [-0.035, +0.093] | +0.040 [-0.025, +0.107] | +0.032 [-0.031, +0.097] | +0.020 [-0.047, +0.085] | +0.027 [-0.039, +0.091] | +0.022 [-0.043, +0.091] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.074 [+0.010, +0.136] | +0.089 [+0.023, +0.153] | +0.071 [+0.009, +0.134] | +0.041 [-0.025, +0.104] | +0.072 [+0.007, +0.137] | +0.068 [+0.002, +0.133] | - |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.463 [+0.415, +0.503] | +0.356 [+0.308, +0.394] | +0.355 [+0.310, +0.391] | +0.344 [+0.294, +0.386] | +0.328 [+0.282, +0.366] | +0.508 [+0.438, +0.578] | +0.113 [+0.066, +0.155] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | +0.446 [+0.392, +0.490] | +0.335 [+0.286, +0.377] | +0.331 [+0.282, +0.372] | +0.332 [+0.281, +0.377] | +0.324 [+0.276, +0.363] | +0.460 [+0.372, +0.547] | +0.284 [+0.243, +0.321] |

### FaceVerse con espressioni: Spearman per insieme di righe (seme 271049)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (99000) | con_perturbazione original (39600) | con_perturbazione remesh (39600) | con_perturbazione down8k (39600) | con_perturbazione up60k (39600) | con_perturbazione noisy (39600) | stessa_topologia senza_crop (24750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.303 [0.229, 0.374] | 0.300 [0.222, 0.372] | 0.309 [0.235, 0.380] | 0.289 [0.218, 0.359] | 0.306 [0.230, 0.377] | 0.312 [0.236, 0.380] | 0.306 [0.230, 0.378] |
| **factorized s1234, d_F non cal.** | 0.297 [0.229, 0.362] | 0.294 [0.220, 0.366] | 0.298 [0.224, 0.366] | 0.288 [0.219, 0.353] | 0.300 [0.230, 0.361] | 0.306 [0.235, 0.373] | 0.302 [0.233, 0.367] |
| factorized s1234, d_P | 0.283 [0.210, 0.346] | 0.280 [0.205, 0.350] | 0.283 [0.208, 0.351] | 0.276 [0.206, 0.338] | 0.285 [0.211, 0.344] | 0.291 [0.218, 0.355] | 0.288 [0.214, 0.353] |
| **factorized s2345, d_F cal.*** | 0.318 [0.249, 0.379] | 0.316 [0.238, 0.383] | 0.322 [0.254, 0.385] | 0.301 [0.229, 0.362] | 0.321 [0.251, 0.383] | 0.329 [0.259, 0.393] | 0.320 [0.250, 0.381] |
| **factorized s2345, d_F non cal.** | 0.317 [0.243, 0.383] | 0.315 [0.234, 0.386] | 0.315 [0.241, 0.388] | 0.306 [0.230, 0.373] | 0.319 [0.247, 0.382] | 0.330 [0.253, 0.400] | 0.318 [0.243, 0.386] |
| factorized s2345, d_P | 0.308 [0.234, 0.373] | 0.307 [0.230, 0.375] | 0.305 [0.230, 0.376] | 0.300 [0.223, 0.367] | 0.308 [0.235, 0.371] | 0.318 [0.244, 0.388] | 0.309 [0.234, 0.375] |
| **ctrlfr s1234** | 0.282 [0.215, 0.343] | 0.291 [0.214, 0.361] | 0.279 [0.209, 0.344] | 0.265 [0.197, 0.327] | 0.296 [0.230, 0.356] | 0.297 [0.222, 0.360] | 0.299 [0.231, 0.365] |
| **ctrlfr s2345** | 0.258 [0.187, 0.324] | 0.258 [0.184, 0.330] | 0.260 [0.183, 0.330] | 0.242 [0.169, 0.314] | 0.256 [0.186, 0.317] | 0.282 [0.206, 0.351] | 0.270 [0.197, 0.338] |
| **C3M e123, d_F cal.*** | 0.289 [0.220, 0.353] | 0.304 [0.227, 0.372] | 0.299 [0.222, 0.373] | 0.286 [0.213, 0.353] | 0.291 [0.220, 0.356] | 0.280 [0.206, 0.348] | 0.317 [0.235, 0.390] |
| **C3M e123, d_F non cal.** | 0.300 [0.224, 0.374] | 0.307 [0.226, 0.385] | 0.306 [0.222, 0.380] | 0.289 [0.211, 0.365] | 0.302 [0.222, 0.377] | 0.305 [0.227, 0.381] | 0.316 [0.233, 0.392] |
| C3M e123, d_P | 0.303 [0.228, 0.376] | 0.310 [0.231, 0.384] | 0.310 [0.228, 0.387] | 0.294 [0.217, 0.370] | 0.302 [0.224, 0.378] | 0.306 [0.230, 0.381] | 0.315 [0.234, 0.391] |
| **C3M e205, d_F cal.*** | 0.290 [0.224, 0.352] | 0.313 [0.245, 0.379] | 0.302 [0.229, 0.369] | 0.288 [0.219, 0.353] | 0.292 [0.226, 0.356] | 0.278 [0.210, 0.340] | 0.333 [0.260, 0.402] |
| **C3M e205, d_F non cal.** | 0.305 [0.232, 0.376] | 0.322 [0.247, 0.392] | 0.309 [0.231, 0.384] | 0.291 [0.216, 0.363] | 0.304 [0.229, 0.376] | 0.311 [0.238, 0.382] | 0.328 [0.252, 0.400] |
| C3M e205, d_P | 0.310 [0.237, 0.381] | 0.326 [0.253, 0.395] | 0.316 [0.236, 0.391] | 0.298 [0.221, 0.371] | 0.305 [0.231, 0.378] | 0.313 [0.239, 0.384] | 0.329 [0.253, 0.400] |
| **ICP + Chamfer in mm (per coppia)** | 0.337 [0.262, 0.410] | 0.351 [0.275, 0.424] | 0.338 [0.257, 0.413] | 0.324 [0.247, 0.398] | 0.335 [0.255, 0.411] | 0.340 [0.261, 0.419] | 0.339 [0.263, 0.410] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.290 [0.218, 0.358] | 0.307 [0.235, 0.376] | 0.289 [0.214, 0.360] | 0.276 [0.201, 0.347] | 0.286 [0.212, 0.357] | 0.294 [0.220, 0.367] | 0.290 [0.218, 0.359] |
| **NICP per coppia in mm** | 0.201 [0.128, 0.277] | 0.204 [0.124, 0.282] | 0.197 [0.122, 0.273] | 0.200 [0.120, 0.279] | 0.198 [0.122, 0.274] | 0.208 [0.131, 0.285] | - |
| **NICP per coppia cs** | 0.212 [0.131, 0.292] | 0.213 [0.133, 0.290] | 0.209 [0.128, 0.294] | 0.215 [0.131, 0.300] | 0.205 [0.125, 0.287] | 0.220 [0.139, 0.301] | - |
| **NICP su template in mm** | 0.212 [0.117, 0.308] | 0.215 [0.118, 0.311] | 0.225 [0.132, 0.320] | 0.203 [0.111, 0.298] | 0.209 [0.113, 0.307] | 0.207 [0.108, 0.305] | 0.212 [0.116, 0.308] |
| **NICP su template cs** | 0.135 [0.061, 0.204] | 0.149 [0.073, 0.217] | 0.140 [0.065, 0.206] | 0.116 [0.033, 0.192] | 0.144 [0.069, 0.214] | 0.130 [0.053, 0.201] | 0.140 [0.063, 0.207] |
| **varifold in mm (massa unitaria)** | 0.260 [0.201, 0.313] | 0.283 [0.218, 0.341] | 0.272 [0.212, 0.324] | 0.272 [0.203, 0.336] | 0.282 [0.220, 0.342] | 0.300 [0.227, 0.367] | 0.277 [0.220, 0.327] |
| **GNM (visto) preregistrato, coefficienti** | 0.220 [0.148, 0.286] | 0.237 [0.159, 0.308] | 0.203 [0.130, 0.271] | 0.200 [0.126, 0.268] | 0.241 [0.165, 0.308] | 0.219 [0.135, 0.292] | 0.221 [0.149, 0.287] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.227 [0.140, 0.311] | 0.243 [0.153, 0.331] | 0.222 [0.132, 0.304] | 0.220 [0.132, 0.303] | 0.225 [0.135, 0.312] | 0.225 [0.134, 0.310] | 0.228 [0.141, 0.313] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.228 [0.145, 0.303] | 0.235 [0.149, 0.312] | 0.215 [0.134, 0.291] | 0.233 [0.153, 0.308] | 0.227 [0.144, 0.303] | 0.229 [0.143, 0.311] | 0.230 [0.147, 0.305] |
| **GNM (visto) vA, coefficienti*** | 0.223 [0.145, 0.291] | 0.238 [0.157, 0.310] | 0.205 [0.124, 0.275] | 0.205 [0.130, 0.274] | 0.243 [0.165, 0.312] | 0.222 [0.137, 0.296] | 0.224 [0.145, 0.291] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.230 [0.141, 0.316] | 0.246 [0.155, 0.334] | 0.226 [0.135, 0.308] | 0.223 [0.135, 0.307] | 0.228 [0.138, 0.315] | 0.228 [0.136, 0.314] | 0.231 [0.143, 0.317] |
| GNM (visto) vA, mesh d'identita' SR* | 0.219 [0.137, 0.297] | 0.225 [0.140, 0.304] | 0.208 [0.127, 0.285] | 0.226 [0.147, 0.302] | 0.220 [0.137, 0.296] | 0.220 [0.134, 0.302] | 0.222 [0.140, 0.298] |
| **GNM (visto) vB, coefficienti*** | 0.226 [0.134, 0.299] | 0.214 [0.117, 0.295] | 0.234 [0.139, 0.313] | 0.232 [0.135, 0.312] | 0.220 [0.126, 0.300] | 0.231 [0.138, 0.307] | 0.224 [0.131, 0.299] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.292 [0.205, 0.363] | 0.306 [0.210, 0.383] | 0.286 [0.202, 0.360] | 0.291 [0.200, 0.362] | 0.284 [0.193, 0.360] | 0.295 [0.209, 0.368] | 0.293 [0.204, 0.365] |
| GNM (visto) vB, mesh d'identita' SR* | 0.319 [0.235, 0.390] | 0.321 [0.236, 0.396] | 0.314 [0.225, 0.389] | 0.326 [0.244, 0.399] | 0.304 [0.220, 0.378] | 0.328 [0.244, 0.404] | 0.316 [0.232, 0.386] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.338 [0.255, 0.407] | 0.348 [0.258, 0.420] | 0.332 [0.248, 0.404] | 0.341 [0.257, 0.411] | 0.326 [0.240, 0.401] | 0.345 [0.263, 0.413] | 0.338 [0.255, 0.406] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.174 [0.109, 0.237] | 0.180 [0.114, 0.242] | 0.171 [0.103, 0.236] | 0.178 [0.115, 0.241] | 0.171 [0.104, 0.239] | 0.172 [0.104, 0.237] | 0.178 [0.112, 0.242] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.211 [0.137, 0.279] | 0.223 [0.150, 0.291] | 0.207 [0.132, 0.279] | 0.215 [0.144, 0.278] | 0.206 [0.127, 0.286] | 0.206 [0.133, 0.277] | 0.215 [0.142, 0.284] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.166 [0.091, 0.235] | 0.160 [0.083, 0.232] | 0.163 [0.088, 0.231] | 0.174 [0.097, 0.242] | 0.166 [0.090, 0.234] | 0.166 [0.084, 0.236] | 0.170 [0.096, 0.240] |
| **FLAME 2023 Open vA, coefficienti*** | 0.174 [0.109, 0.237] | 0.180 [0.114, 0.242] | 0.171 [0.103, 0.236] | 0.178 [0.115, 0.241] | 0.171 [0.104, 0.239] | 0.172 [0.104, 0.237] | 0.178 [0.112, 0.242] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.211 [0.137, 0.279] | 0.223 [0.150, 0.291] | 0.207 [0.132, 0.279] | 0.215 [0.144, 0.278] | 0.206 [0.127, 0.286] | 0.206 [0.133, 0.277] | 0.215 [0.142, 0.284] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.166 [0.091, 0.235] | 0.160 [0.083, 0.232] | 0.163 [0.088, 0.231] | 0.174 [0.097, 0.242] | 0.166 [0.090, 0.234] | 0.166 [0.084, 0.236] | 0.170 [0.096, 0.240] |
| **FLAME 2023 Open vB, coefficienti*** | 0.203 [0.121, 0.275] | 0.171 [0.081, 0.255] | 0.200 [0.112, 0.281] | 0.229 [0.135, 0.305] | 0.199 [0.112, 0.279] | 0.218 [0.135, 0.293] | 0.202 [0.121, 0.276] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.264 [0.183, 0.336] | 0.276 [0.192, 0.348] | 0.260 [0.180, 0.334] | 0.269 [0.188, 0.341] | 0.257 [0.174, 0.335] | 0.260 [0.176, 0.333] | 0.268 [0.187, 0.340] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.290 [0.216, 0.361] | 0.289 [0.210, 0.363] | 0.285 [0.207, 0.357] | 0.301 [0.226, 0.374] | 0.277 [0.198, 0.349] | 0.296 [0.216, 0.368] | 0.291 [0.217, 0.361] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.302 [0.222, 0.372] | 0.310 [0.228, 0.382] | 0.297 [0.219, 0.368] | 0.309 [0.233, 0.377] | 0.291 [0.205, 0.364] | 0.301 [0.221, 0.373] | 0.304 [0.224, 0.373] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.127 [0.081, 0.175] | 0.143 [0.091, 0.194] | 0.145 [0.097, 0.196] | 0.143 [0.090, 0.199] | 0.147 [0.093, 0.197] | 0.102 [0.039, 0.164] | 0.141 [0.091, 0.193] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.117 [0.070, 0.166] | 0.138 [0.081, 0.195] | 0.133 [0.077, 0.188] | 0.144 [0.087, 0.202] | 0.139 [0.081, 0.195] | 0.089 [0.022, 0.156] | 0.135 [0.080, 0.189] |
| **ShapeDNA k=50** | 0.063 [0.036, 0.087] | 0.074 [0.031, 0.117] | 0.081 [0.036, 0.123] | 0.088 [0.047, 0.129] | 0.091 [0.048, 0.132] | 0.078 [0.009, 0.149] | 0.133 [0.071, 0.188] |
| **HKS globale** | 0.032 [0.009, 0.053] | 0.045 [0.013, 0.076] | 0.041 [0.009, 0.075] | 0.043 [0.010, 0.076] | 0.049 [0.014, 0.080] | 0.030 [-0.028, 0.084] | 0.067 [0.024, 0.114] |
| **WKS globale** | 0.039 [0.014, 0.065] | 0.036 [-0.003, 0.073] | 0.042 [0.005, 0.080] | 0.050 [0.010, 0.090] | 0.051 [0.016, 0.087] | 0.091 [0.015, 0.164] | 0.069 [0.016, 0.120] |
| **taglia stimata dalla mesh (non oracolo)** | 0.126 [0.037, 0.214] | 0.123 [0.029, 0.217] | 0.120 [0.032, 0.208] | 0.115 [0.027, 0.205] | 0.130 [0.034, 0.224] | 0.143 [0.052, 0.233] | 0.131 [0.042, 0.220] |
| **taglia oracolo (S di FR)** | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.049 [0.019, 0.079] | 0.068 [0.027, 0.108] | 0.049 [0.010, 0.089] | 0.059 [0.021, 0.095] | 0.069 [0.032, 0.106] | 0.081 [-0.012, 0.170] | 0.082 [0.032, 0.131] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.052 [0.023, 0.080] | 0.072 [0.033, 0.111] | 0.057 [0.017, 0.096] | 0.064 [0.025, 0.100] | 0.072 [0.036, 0.107] | 0.081 [-0.009, 0.172] | 0.086 [0.039, 0.133] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.065 [0.026, 0.101] | 0.077 [0.033, 0.116] | 0.069 [0.024, 0.113] | 0.080 [0.024, 0.131] | 0.085 [0.041, 0.127] | 0.065 [-0.004, 0.131] | 0.117 [0.062, 0.168] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.062 [0.022, 0.098] | 0.074 [0.027, 0.117] | 0.066 [0.021, 0.111] | 0.072 [0.016, 0.121] | 0.084 [0.038, 0.125] | 0.062 [-0.005, 0.128] | 0.111 [0.054, 0.162] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.061 [0.033, 0.087] | 0.089 [0.052, 0.124] | 0.073 [0.036, 0.109] | 0.074 [0.036, 0.111] | 0.092 [0.061, 0.125] | 0.071 [-0.005, 0.148] | 0.122 [0.070, 0.172] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.061 [0.034, 0.088] | 0.088 [0.050, 0.123] | 0.073 [0.037, 0.108] | 0.075 [0.038, 0.112] | 0.092 [0.060, 0.125] | 0.075 [0.001, 0.152] | 0.121 [0.069, 0.172] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.196 [0.125, 0.261] | 0.210 [0.129, 0.283] | 0.199 [0.127, 0.267] | 0.191 [0.119, 0.262] | 0.214 [0.139, 0.286] | 0.210 [0.129, 0.283] | - |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.178 [0.119, 0.227] | 0.195 [0.124, 0.258] | 0.189 [0.122, 0.245] | 0.193 [0.131, 0.249] | 0.195 [0.133, 0.247] | 0.188 [0.112, 0.256] | - |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.137 [0.092, 0.182] | 0.163 [0.107, 0.211] | 0.143 [0.094, 0.190] | 0.142 [0.089, 0.193] | 0.160 [0.107, 0.210] | 0.180 [0.112, 0.243] | 0.211 [0.146, 0.270] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.111 [0.071, 0.148] | 0.132 [0.080, 0.175] | 0.117 [0.069, 0.158] | 0.121 [0.074, 0.166] | 0.127 [0.082, 0.167] | 0.156 [0.094, 0.208] | 0.178 [0.118, 0.231] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.338 [0.264, 0.408] | 0.352 [0.273, 0.423] | 0.341 [0.268, 0.410] | 0.336 [0.258, 0.410] | 0.335 [0.260, 0.412] | 0.356 [0.275, 0.434] | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.300 [0.231, 0.366] | 0.313 [0.239, 0.382] | 0.302 [0.229, 0.373] | 0.304 [0.231, 0.374] | 0.297 [0.227, 0.368] | 0.320 [0.243, 0.397] | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.302 [0.235, 0.364] | 0.300 [0.224, 0.368] | 0.297 [0.228, 0.361] | 0.313 [0.236, 0.385] | 0.296 [0.228, 0.359] | 0.308 [0.235, 0.374] | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.201 [0.136, 0.269] | 0.211 [0.143, 0.279] | 0.204 [0.133, 0.278] | 0.197 [0.128, 0.268] | 0.190 [0.120, 0.259] | 0.204 [0.135, 0.274] | - |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.159 [0.125, 0.191] | 0.195 [0.146, 0.239] | 0.194 [0.151, 0.234] | 0.189 [0.144, 0.233] | 0.201 [0.155, 0.242] | 0.202 [0.126, 0.276] | 0.278 [0.220, 0.331] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.148 [0.117, 0.177] | 0.182 [0.135, 0.225] | 0.181 [0.140, 0.218] | 0.183 [0.141, 0.222] | 0.185 [0.140, 0.223] | 0.191 [0.108, 0.267] | 0.263 [0.209, 0.313] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.164 [0.117, 0.212] | 0.201 [0.139, 0.262] | 0.199 [0.148, 0.247] | 0.178 [0.120, 0.240] | 0.202 [0.144, 0.255] | 0.155 [0.078, 0.242] | 0.230 [0.171, 0.289] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.159 [0.115, 0.204] | 0.195 [0.138, 0.249] | 0.190 [0.144, 0.236] | 0.181 [0.130, 0.235] | 0.190 [0.137, 0.240] | 0.164 [0.084, 0.247] | 0.216 [0.163, 0.268] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | nocrop (99000) | con_perturbazione original (39600) | con_perturbazione remesh (39600) | con_perturbazione down8k (39600) | con_perturbazione up60k (39600) | con_perturbazione noisy (39600) | stessa_topologia senza_crop (24750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.269 [0.195, 0.341] | 0.260 [0.179, 0.337] | 0.281 [0.207, 0.355] | 0.256 [0.179, 0.329] | 0.273 [0.194, 0.349] | 0.275 [0.201, 0.352] | 0.272 [0.195, 0.347] |
| factorized s1234, d_F non cal. | 0.287 [0.215, 0.353] | 0.281 [0.207, 0.355] | 0.295 [0.219, 0.363] | 0.276 [0.202, 0.341] | 0.292 [0.218, 0.356] | 0.292 [0.219, 0.359] | 0.293 [0.221, 0.361] |
| **factorized s1234, d_P** | 0.286 [0.215, 0.352] | 0.282 [0.207, 0.355] | 0.294 [0.217, 0.362] | 0.277 [0.202, 0.340] | 0.290 [0.218, 0.355] | 0.290 [0.214, 0.355] | 0.293 [0.221, 0.360] |
| factorized s2345, d_F cal.* | 0.286 [0.206, 0.353] | 0.280 [0.196, 0.352] | 0.298 [0.219, 0.369] | 0.271 [0.190, 0.339] | 0.288 [0.210, 0.356] | 0.296 [0.215, 0.366] | 0.288 [0.206, 0.356] |
| factorized s2345, d_F non cal. | 0.309 [0.227, 0.376] | 0.305 [0.217, 0.379] | 0.315 [0.231, 0.388] | 0.297 [0.215, 0.364] | 0.309 [0.231, 0.376] | 0.318 [0.234, 0.389] | 0.310 [0.229, 0.379] |
| **factorized s2345, d_P** | 0.313 [0.237, 0.377] | 0.312 [0.229, 0.381] | 0.318 [0.239, 0.389] | 0.303 [0.225, 0.372] | 0.312 [0.237, 0.376] | 0.320 [0.240, 0.389] | 0.314 [0.239, 0.380] |
| ctrlfr s1234 | 0.273 [0.201, 0.339] | 0.278 [0.197, 0.352] | 0.274 [0.200, 0.342] | 0.258 [0.185, 0.324] | 0.285 [0.211, 0.349] | 0.284 [0.201, 0.350] | 0.286 [0.212, 0.356] |
| ctrlfr s2345 | 0.259 [0.188, 0.325] | 0.255 [0.178, 0.332] | 0.267 [0.192, 0.338] | 0.243 [0.170, 0.312] | 0.258 [0.189, 0.323] | 0.280 [0.211, 0.349] | 0.270 [0.196, 0.338] |
| C3M e123, d_F cal.* | 0.270 [0.193, 0.341] | 0.281 [0.196, 0.358] | 0.284 [0.201, 0.359] | 0.267 [0.186, 0.340] | 0.273 [0.194, 0.347] | 0.258 [0.182, 0.332] | 0.298 [0.210, 0.376] |
| C3M e123, d_F non cal. | 0.303 [0.221, 0.381] | 0.312 [0.224, 0.390] | 0.315 [0.226, 0.394] | 0.291 [0.208, 0.367] | 0.306 [0.223, 0.383] | 0.302 [0.221, 0.377] | 0.322 [0.235, 0.402] |
| **C3M e123, d_P** | 0.313 [0.236, 0.386] | 0.321 [0.240, 0.398] | 0.324 [0.240, 0.399] | 0.302 [0.223, 0.374] | 0.313 [0.235, 0.388] | 0.309 [0.234, 0.382] | 0.327 [0.247, 0.405] |
| C3M e205, d_F cal.* | 0.275 [0.204, 0.340] | 0.293 [0.219, 0.365] | 0.291 [0.216, 0.364] | 0.275 [0.203, 0.344] | 0.275 [0.202, 0.343] | 0.261 [0.193, 0.326] | 0.316 [0.237, 0.390] |
| C3M e205, d_F non cal. | 0.306 [0.230, 0.376] | 0.321 [0.240, 0.395] | 0.317 [0.233, 0.393] | 0.293 [0.216, 0.367] | 0.305 [0.228, 0.379] | 0.304 [0.231, 0.375] | 0.331 [0.250, 0.406] |
| **C3M e205, d_P** | 0.314 [0.242, 0.382] | 0.330 [0.254, 0.403] | 0.327 [0.247, 0.400] | 0.303 [0.230, 0.376] | 0.310 [0.238, 0.383] | 0.309 [0.236, 0.379] | 0.336 [0.260, 0.405] |
| **ICP + Chamfer in mm (per coppia)** | 0.309 [0.234, 0.380] | 0.319 [0.240, 0.388] | 0.313 [0.233, 0.384] | 0.301 [0.223, 0.375] | 0.310 [0.233, 0.388] | 0.306 [0.227, 0.380] | 0.310 [0.233, 0.380] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.306 [0.233, 0.376] | 0.321 [0.249, 0.390] | 0.308 [0.232, 0.377] | 0.293 [0.213, 0.368] | 0.308 [0.232, 0.380] | 0.303 [0.224, 0.378] | 0.305 [0.231, 0.375] |
| **NICP per coppia in mm** | 0.223 [0.147, 0.298] | 0.226 [0.146, 0.304] | 0.223 [0.144, 0.297] | 0.224 [0.137, 0.305] | 0.223 [0.143, 0.301] | 0.223 [0.142, 0.300] | - |
| **NICP per coppia cs** | 0.235 [0.158, 0.310] | 0.235 [0.158, 0.311] | 0.234 [0.152, 0.314] | 0.239 [0.155, 0.321] | 0.232 [0.151, 0.315] | 0.238 [0.156, 0.314] | - |
| **NICP su template in mm** | 0.152 [0.058, 0.246] | 0.150 [0.055, 0.246] | 0.165 [0.072, 0.259] | 0.148 [0.053, 0.244] | 0.151 [0.057, 0.245] | 0.145 [0.050, 0.245] | 0.152 [0.057, 0.246] |
| **NICP su template cs** | 0.147 [0.070, 0.216] | 0.158 [0.082, 0.232] | 0.151 [0.073, 0.217] | 0.129 [0.046, 0.204] | 0.161 [0.084, 0.229] | 0.137 [0.057, 0.208] | 0.149 [0.071, 0.219] |
| **varifold in mm (massa unitaria)** | 0.250 [0.193, 0.306] | 0.269 [0.206, 0.331] | 0.263 [0.205, 0.319] | 0.265 [0.200, 0.327] | 0.270 [0.207, 0.331] | 0.287 [0.215, 0.354] | 0.264 [0.208, 0.316] |
| **GNM (visto) preregistrato, coefficienti** | 0.182 [0.104, 0.252] | 0.200 [0.120, 0.273] | 0.166 [0.088, 0.235] | 0.167 [0.091, 0.239] | 0.202 [0.126, 0.274] | 0.177 [0.096, 0.250] | 0.184 [0.105, 0.252] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.171 [0.082, 0.260] | 0.183 [0.089, 0.273] | 0.169 [0.079, 0.258] | 0.167 [0.079, 0.251] | 0.171 [0.078, 0.263] | 0.165 [0.076, 0.255] | 0.173 [0.082, 0.260] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.209 [0.129, 0.285] | 0.216 [0.132, 0.295] | 0.198 [0.118, 0.276] | 0.218 [0.141, 0.293] | 0.212 [0.128, 0.291] | 0.203 [0.116, 0.283] | 0.212 [0.130, 0.286] |
| **GNM (visto) vA, coefficienti*** | 0.184 [0.103, 0.254] | 0.200 [0.119, 0.273] | 0.167 [0.085, 0.238] | 0.170 [0.093, 0.243] | 0.203 [0.123, 0.274] | 0.178 [0.095, 0.256] | 0.185 [0.104, 0.255] |
| GNM (visto) vA, mesh d'identita' FR* | 0.174 [0.085, 0.263] | 0.186 [0.090, 0.276] | 0.173 [0.083, 0.261] | 0.170 [0.081, 0.255] | 0.173 [0.080, 0.265] | 0.167 [0.077, 0.260] | 0.175 [0.086, 0.263] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.205 [0.124, 0.281] | 0.210 [0.126, 0.288] | 0.194 [0.115, 0.272] | 0.214 [0.135, 0.288] | 0.209 [0.124, 0.290] | 0.198 [0.109, 0.278] | 0.207 [0.126, 0.282] |
| **GNM (visto) vB, coefficienti*** | 0.222 [0.124, 0.302] | 0.209 [0.109, 0.293] | 0.231 [0.136, 0.313] | 0.231 [0.128, 0.314] | 0.221 [0.121, 0.302] | 0.220 [0.124, 0.297] | 0.219 [0.124, 0.298] |
| GNM (visto) vB, mesh d'identita' FR* | 0.244 [0.149, 0.326] | 0.251 [0.150, 0.340] | 0.239 [0.141, 0.322] | 0.246 [0.149, 0.331] | 0.240 [0.140, 0.329] | 0.242 [0.148, 0.327] | 0.244 [0.151, 0.327] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.327 [0.239, 0.402] | 0.325 [0.237, 0.400] | 0.325 [0.235, 0.401] | 0.338 [0.250, 0.412] | 0.322 [0.233, 0.397] | 0.329 [0.240, 0.404] | 0.324 [0.236, 0.400] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.306 [0.209, 0.381] | 0.309 [0.211, 0.389] | 0.301 [0.202, 0.382] | 0.312 [0.219, 0.387] | 0.301 [0.207, 0.377] | 0.306 [0.210, 0.382] | 0.305 [0.208, 0.380] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.152 [0.079, 0.218] | 0.160 [0.087, 0.227] | 0.147 [0.074, 0.216] | 0.153 [0.078, 0.221] | 0.152 [0.079, 0.222] | 0.148 [0.073, 0.217] | 0.156 [0.083, 0.225] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.170 [0.089, 0.245] | 0.181 [0.103, 0.257] | 0.166 [0.086, 0.241] | 0.172 [0.093, 0.246] | 0.166 [0.078, 0.247] | 0.165 [0.084, 0.244] | 0.174 [0.094, 0.250] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.155 [0.075, 0.228] | 0.151 [0.066, 0.227] | 0.150 [0.069, 0.225] | 0.163 [0.079, 0.238] | 0.160 [0.079, 0.236] | 0.151 [0.063, 0.227] | 0.160 [0.077, 0.235] |
| **FLAME 2023 Open vA, coefficienti*** | 0.152 [0.079, 0.218] | 0.160 [0.087, 0.227] | 0.147 [0.074, 0.216] | 0.153 [0.078, 0.221] | 0.152 [0.079, 0.222] | 0.148 [0.073, 0.217] | 0.156 [0.083, 0.225] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.170 [0.089, 0.245] | 0.181 [0.103, 0.257] | 0.166 [0.086, 0.241] | 0.172 [0.093, 0.246] | 0.166 [0.078, 0.247] | 0.165 [0.084, 0.244] | 0.174 [0.094, 0.250] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.155 [0.075, 0.228] | 0.151 [0.066, 0.227] | 0.150 [0.069, 0.225] | 0.163 [0.079, 0.238] | 0.160 [0.079, 0.236] | 0.151 [0.063, 0.227] | 0.160 [0.077, 0.235] |
| **FLAME 2023 Open vB, coefficienti*** | 0.193 [0.108, 0.271] | 0.162 [0.068, 0.248] | 0.189 [0.098, 0.272] | 0.218 [0.127, 0.299] | 0.199 [0.110, 0.283] | 0.198 [0.113, 0.275] | 0.192 [0.108, 0.268] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.218 [0.125, 0.302] | 0.227 [0.129, 0.315] | 0.217 [0.121, 0.302] | 0.223 [0.129, 0.307] | 0.216 [0.122, 0.305] | 0.210 [0.114, 0.293] | 0.221 [0.129, 0.308] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.293 [0.209, 0.368] | 0.292 [0.207, 0.367] | 0.291 [0.206, 0.370] | 0.305 [0.223, 0.385] | 0.288 [0.202, 0.365] | 0.288 [0.201, 0.368] | 0.294 [0.208, 0.370] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.270 [0.182, 0.349] | 0.275 [0.181, 0.358] | 0.268 [0.176, 0.351] | 0.278 [0.187, 0.358] | 0.266 [0.176, 0.349] | 0.263 [0.173, 0.346] | 0.272 [0.184, 0.353] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.138 [0.093, 0.185] | 0.153 [0.102, 0.206] | 0.158 [0.112, 0.209] | 0.157 [0.105, 0.212] | 0.158 [0.107, 0.209] | 0.113 [0.048, 0.177] | 0.152 [0.103, 0.207] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.129 [0.081, 0.177] | 0.151 [0.095, 0.206] | 0.147 [0.096, 0.202] | 0.163 [0.109, 0.221] | 0.150 [0.096, 0.207] | 0.092 [0.022, 0.160] | 0.152 [0.103, 0.208] |
| **ShapeDNA k=50** | 0.061 [0.033, 0.086] | 0.076 [0.031, 0.118] | 0.081 [0.035, 0.122] | 0.083 [0.039, 0.126] | 0.086 [0.040, 0.127] | 0.076 [0.005, 0.144] | 0.128 [0.061, 0.186] |
| **HKS globale** | 0.029 [0.004, 0.051] | 0.042 [0.008, 0.076] | 0.038 [0.002, 0.074] | 0.034 [-0.002, 0.070] | 0.039 [0.005, 0.070] | 0.036 [-0.027, 0.093] | 0.061 [0.013, 0.111] |
| **WKS globale** | 0.039 [0.012, 0.066] | 0.037 [-0.001, 0.073] | 0.044 [0.007, 0.083] | 0.047 [0.006, 0.089] | 0.049 [0.016, 0.085] | 0.091 [0.015, 0.166] | 0.063 [0.009, 0.112] |
| **taglia stimata dalla mesh (non oracolo)** | 0.080 [-0.002, 0.170] | 0.071 [-0.015, 0.163] | 0.080 [-0.001, 0.170] | 0.077 [-0.005, 0.161] | 0.079 [-0.008, 0.170] | 0.093 [0.001, 0.183] | 0.084 [-0.003, 0.174] |
| **taglia oracolo (S di FR)** | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.051 [0.023, 0.079] | 0.075 [0.038, 0.116] | 0.052 [0.016, 0.093] | 0.062 [0.026, 0.098] | 0.070 [0.034, 0.105] | 0.079 [-0.013, 0.171] | 0.090 [0.040, 0.138] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.052 [0.026, 0.079] | 0.078 [0.041, 0.116] | 0.058 [0.022, 0.096] | 0.063 [0.029, 0.099] | 0.072 [0.038, 0.105] | 0.076 [-0.016, 0.165] | 0.092 [0.043, 0.139] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.072 [0.034, 0.108] | 0.087 [0.043, 0.126] | 0.076 [0.030, 0.121] | 0.088 [0.032, 0.137] | 0.089 [0.046, 0.128] | 0.079 [0.007, 0.150] | 0.129 [0.071, 0.181] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.070 [0.030, 0.106] | 0.083 [0.037, 0.126] | 0.074 [0.030, 0.118] | 0.080 [0.024, 0.132] | 0.089 [0.046, 0.128] | 0.075 [0.004, 0.145] | 0.123 [0.064, 0.175] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.059 [0.032, 0.086] | 0.093 [0.056, 0.128] | 0.074 [0.037, 0.109] | 0.076 [0.039, 0.113] | 0.091 [0.058, 0.123] | 0.049 [-0.025, 0.125] | 0.127 [0.075, 0.174] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.060 [0.033, 0.086] | 0.094 [0.055, 0.130] | 0.076 [0.038, 0.111] | 0.078 [0.040, 0.113] | 0.093 [0.060, 0.125] | 0.052 [-0.022, 0.126] | 0.127 [0.075, 0.175] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.163 [0.098, 0.224] | 0.174 [0.095, 0.246] | 0.171 [0.105, 0.237] | 0.165 [0.097, 0.229] | 0.179 [0.110, 0.245] | 0.171 [0.093, 0.239] | - |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.183 [0.122, 0.233] | 0.201 [0.127, 0.265] | 0.197 [0.130, 0.252] | 0.197 [0.130, 0.252] | 0.204 [0.140, 0.259] | 0.189 [0.113, 0.254] | - |
| LPIPS alex su render [profondita'], mm (FR) | 0.119 [0.074, 0.164] | 0.142 [0.089, 0.193] | 0.130 [0.081, 0.181] | 0.125 [0.073, 0.178] | 0.138 [0.088, 0.186] | 0.147 [0.081, 0.212] | 0.184 [0.121, 0.247] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.112 [0.069, 0.152] | 0.133 [0.080, 0.181] | 0.122 [0.074, 0.168] | 0.121 [0.072, 0.167] | 0.126 [0.080, 0.170] | 0.151 [0.087, 0.209] | 0.178 [0.113, 0.237] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.310 [0.234, 0.385] | 0.319 [0.234, 0.397] | 0.314 [0.238, 0.386] | 0.314 [0.234, 0.394] | 0.313 [0.235, 0.393] | 0.321 [0.235, 0.402] | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.307 [0.236, 0.377] | 0.319 [0.243, 0.389] | 0.311 [0.237, 0.382] | 0.313 [0.233, 0.384] | 0.311 [0.236, 0.384] | 0.321 [0.240, 0.397] | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.312 [0.250, 0.368] | 0.313 [0.246, 0.376] | 0.310 [0.245, 0.373] | 0.325 [0.253, 0.389] | 0.306 [0.246, 0.364] | 0.312 [0.241, 0.377] | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.198 [0.134, 0.263] | 0.206 [0.140, 0.267] | 0.196 [0.126, 0.270] | 0.199 [0.131, 0.265] | 0.194 [0.127, 0.261] | 0.196 [0.126, 0.268] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.156 [0.123, 0.187] | 0.189 [0.140, 0.233] | 0.191 [0.149, 0.230] | 0.184 [0.142, 0.230] | 0.193 [0.149, 0.236] | 0.211 [0.128, 0.286] | 0.268 [0.212, 0.322] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.158 [0.125, 0.187] | 0.194 [0.146, 0.236] | 0.195 [0.153, 0.231] | 0.192 [0.149, 0.231] | 0.197 [0.155, 0.238] | 0.210 [0.121, 0.286] | 0.277 [0.223, 0.326] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.156 [0.110, 0.204] | 0.195 [0.136, 0.252] | 0.197 [0.147, 0.245] | 0.173 [0.113, 0.234] | 0.194 [0.138, 0.247] | 0.134 [0.062, 0.215] | 0.223 [0.165, 0.283] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.166 [0.122, 0.209] | 0.206 [0.153, 0.260] | 0.205 [0.160, 0.249] | 0.189 [0.134, 0.244] | 0.201 [0.150, 0.250] | 0.156 [0.080, 0.238] | 0.228 [0.174, 0.279] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.035 [-0.105, +0.036] | -0.052 [-0.119, +0.021] | -0.029 [-0.105, +0.046] | -0.035 [-0.108, +0.041] | -0.029 [-0.105, +0.044] | -0.028 [-0.099, +0.043] | -0.033 [-0.102, +0.038] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.013 [-0.064, +0.099] | -0.008 [-0.087, +0.079] | +0.019 [-0.062, +0.108] | +0.013 [-0.067, +0.101] | +0.020 [-0.058, +0.110] | +0.018 [-0.063, +0.108] | +0.016 [-0.060, +0.100] |
| NICP per coppia in mm | +0.102 [+0.015, +0.190] | +0.096 [+0.002, +0.183] | +0.112 [+0.020, +0.204] | +0.089 [-0.001, +0.181] | +0.108 [+0.021, +0.196] | +0.104 [+0.015, +0.186] | - |
| NICP per coppia cs | +0.091 [-0.001, +0.177] | +0.087 [-0.000, +0.178] | +0.100 [+0.007, +0.188] | +0.074 [-0.013, +0.162] | +0.101 [+0.011, +0.191] | +0.092 [+0.002, +0.178] | - |
| NICP su template in mm | +0.091 [-0.010, +0.189] | +0.085 [-0.021, +0.188] | +0.084 [-0.020, +0.188] | +0.086 [-0.014, +0.185] | +0.097 [-0.001, +0.191] | +0.105 [+0.007, +0.207] | +0.095 [-0.006, +0.194] |
| NICP su template cs | +0.167 [+0.063, +0.271] | +0.151 [+0.044, +0.257] | +0.169 [+0.065, +0.274] | +0.173 [+0.060, +0.286] | +0.162 [+0.054, +0.261] | +0.182 [+0.083, +0.285] | +0.167 [+0.064, +0.271] |
| varifold in mm (massa unitaria) | +0.043 [-0.019, +0.107] | +0.016 [-0.045, +0.080] | +0.037 [-0.026, +0.106] | +0.017 [-0.047, +0.083] | +0.025 [-0.041, +0.093] | +0.012 [-0.060, +0.087] | +0.029 [-0.031, +0.093] |
| GNM (visto) preregistrato, coefficienti | +0.083 [-0.014, +0.183] | +0.063 [-0.044, +0.168] | +0.106 [+0.005, +0.221] | +0.089 [-0.012, +0.188] | +0.065 [-0.029, +0.155] | +0.093 [-0.006, +0.203] | +0.085 [-0.016, +0.187] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.076 [-0.016, +0.177] | +0.057 [-0.042, +0.162] | +0.087 [-0.008, +0.192] | +0.069 [-0.018, +0.167] | +0.081 [-0.009, +0.174] | +0.087 [-0.005, +0.190] | +0.078 [-0.015, +0.180] |
| GNM (visto) vA, coefficienti | +0.080 [-0.019, +0.186] | +0.061 [-0.042, +0.169] | +0.104 [+0.003, +0.219] | +0.084 [-0.021, +0.186] | +0.063 [-0.032, +0.154] | +0.089 [-0.014, +0.198] | +0.082 [-0.020, +0.187] |
| GNM (visto) vA, mesh d'identita' FR | +0.073 [-0.020, +0.173] | +0.054 [-0.045, +0.160] | +0.083 [-0.014, +0.192] | +0.066 [-0.023, +0.164] | +0.078 [-0.012, +0.172] | +0.084 [-0.007, +0.189] | +0.075 [-0.018, +0.177] |
| GNM (visto) vB, coefficienti | +0.077 [-0.019, +0.178] | +0.086 [-0.014, +0.189] | +0.074 [-0.020, +0.175] | +0.057 [-0.041, +0.168] | +0.086 [-0.012, +0.193] | +0.081 [-0.008, +0.177] | +0.082 [-0.014, +0.183] |
| GNM (visto) vB, mesh d'identita' FR | +0.010 [-0.062, +0.090] | -0.006 [-0.087, +0.082] | +0.023 [-0.053, +0.105] | -0.002 [-0.074, +0.078] | +0.022 [-0.049, +0.100] | +0.017 [-0.055, +0.097] | +0.013 [-0.062, +0.094] |
| GNM (visto) vB, composizione S_B + k d_P | -0.036 [-0.105, +0.045] | -0.049 [-0.127, +0.041] | -0.023 [-0.097, +0.059] | -0.052 [-0.123, +0.031] | -0.020 [-0.090, +0.060] | -0.033 [-0.104, +0.047] | -0.032 [-0.104, +0.050] |
| FLAME 2023 Open preregistrato, coefficienti | +0.128 [+0.036, +0.221] | +0.119 [+0.020, +0.220] | +0.138 [+0.044, +0.239] | +0.111 [+0.018, +0.204] | +0.135 [+0.047, +0.222] | +0.139 [+0.044, +0.233] | +0.128 [+0.032, +0.220] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.091 [+0.005, +0.177] | +0.077 [-0.015, +0.168] | +0.102 [+0.008, +0.199] | +0.074 [-0.010, +0.158] | +0.100 [+0.013, +0.182] | +0.106 [+0.021, +0.192] | +0.091 [+0.004, +0.177] |
| FLAME 2023 Open vA, coefficienti | +0.128 [+0.036, +0.221] | +0.119 [+0.020, +0.220] | +0.138 [+0.044, +0.239] | +0.111 [+0.018, +0.204] | +0.135 [+0.047, +0.222] | +0.139 [+0.044, +0.233] | +0.128 [+0.032, +0.220] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.091 [+0.005, +0.177] | +0.077 [-0.015, +0.168] | +0.102 [+0.008, +0.199] | +0.074 [-0.010, +0.158] | +0.100 [+0.013, +0.182] | +0.106 [+0.021, +0.192] | +0.091 [+0.004, +0.177] |
| FLAME 2023 Open vB, coefficienti | +0.100 [+0.002, +0.195] | +0.128 [+0.025, +0.240] | +0.109 [+0.018, +0.208] | +0.060 [-0.036, +0.162] | +0.107 [+0.016, +0.203] | +0.093 [+0.001, +0.181] | +0.104 [+0.010, +0.197] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.038 [-0.042, +0.129] | +0.023 [-0.064, +0.117] | +0.049 [-0.039, +0.146] | +0.020 [-0.061, +0.109] | +0.049 [-0.029, +0.134] | +0.052 [-0.028, +0.142] | +0.038 [-0.043, +0.130] |
| FLAME 2023 Open vB, composizione S_B + k d_P | +0.001 [-0.079, +0.089] | -0.010 [-0.102, +0.082] | +0.012 [-0.073, +0.105] | -0.020 [-0.104, +0.068] | +0.015 [-0.063, +0.103] | +0.010 [-0.071, +0.098] | +0.002 [-0.083, +0.092] |
| ArcFace-render, normal map 3 viste (G0) | +0.176 [+0.098, +0.252] | +0.156 [+0.065, +0.242] | +0.163 [+0.082, +0.240] | +0.146 [+0.057, +0.228] | +0.160 [+0.081, +0.232] | +0.210 [+0.108, +0.305] | +0.165 [+0.078, +0.245] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.185 [+0.096, +0.264] | +0.162 [+0.067, +0.247] | +0.176 [+0.084, +0.262] | +0.145 [+0.051, +0.230] | +0.167 [+0.070, +0.250] | +0.223 [+0.109, +0.326] | +0.171 [+0.076, +0.258] |
| ShapeDNA k=50 | +0.240 [+0.162, +0.308] | +0.225 [+0.135, +0.313] | +0.228 [+0.144, +0.308] | +0.201 [+0.121, +0.283] | +0.215 [+0.132, +0.291] | +0.234 [+0.138, +0.327] | +0.173 [+0.083, +0.261] |
| HKS globale | +0.271 [+0.197, +0.345] | +0.255 [+0.172, +0.335] | +0.267 [+0.188, +0.350] | +0.246 [+0.166, +0.324] | +0.258 [+0.178, +0.339] | +0.282 [+0.189, +0.371] | +0.239 [+0.154, +0.328] |
| WKS globale | +0.263 [+0.186, +0.331] | +0.264 [+0.181, +0.342] | +0.267 [+0.188, +0.343] | +0.239 [+0.160, +0.317] | +0.255 [+0.176, +0.326] | +0.221 [+0.120, +0.320] | +0.237 [+0.149, +0.318] |
| taglia stimata dalla mesh (non oracolo) | +0.176 [+0.086, +0.256] | +0.177 [+0.079, +0.266] | +0.189 [+0.094, +0.276] | +0.174 [+0.084, +0.254] | +0.176 [+0.084, +0.261] | +0.169 [+0.077, +0.252] | +0.175 [+0.081, +0.259] |
| taglia oracolo (S di FR) | +0.094 [-0.029, +0.208] | +0.091 [-0.036, +0.206] | +0.100 [-0.029, +0.213] | +0.080 [-0.043, +0.196] | +0.097 [-0.027, +0.216] | +0.103 [-0.019, +0.216] | +0.097 [-0.026, +0.214] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.254 [+0.175, +0.325] | +0.232 [+0.149, +0.310] | +0.259 [+0.180, +0.335] | +0.230 [+0.153, +0.303] | +0.237 [+0.154, +0.317] | +0.231 [+0.111, +0.347] | +0.224 [+0.138, +0.304] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.237 [+0.162, +0.309] | +0.222 [+0.143, +0.300] | +0.240 [+0.164, +0.317] | +0.209 [+0.126, +0.293] | +0.221 [+0.140, +0.304] | +0.247 [+0.148, +0.338] | +0.189 [+0.107, +0.268] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.242 [+0.164, +0.311] | +0.210 [+0.130, +0.282] | +0.236 [+0.159, +0.308] | +0.215 [+0.138, +0.288] | +0.214 [+0.134, +0.287] | +0.241 [+0.130, +0.336] | +0.184 [+0.102, +0.262] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.107 [+0.033, +0.176] | +0.090 [+0.010, +0.165] | +0.110 [+0.033, +0.186] | +0.098 [+0.019, +0.171] | +0.092 [+0.013, +0.168] | +0.102 [+0.022, +0.176] | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.166 [+0.097, +0.232] | +0.137 [+0.066, +0.204] | +0.166 [+0.092, +0.237] | +0.147 [+0.070, +0.221] | +0.146 [+0.065, +0.220] | +0.131 [+0.053, +0.208] | +0.095 [+0.016, +0.172] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.035 [-0.101, +0.040] | -0.052 [-0.121, +0.024] | -0.032 [-0.104, +0.046] | -0.047 [-0.114, +0.028] | -0.029 [-0.095, +0.049] | -0.044 [-0.113, +0.033] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.003 [-0.070, +0.082] | -0.013 [-0.089, +0.069] | +0.006 [-0.068, +0.087] | -0.015 [-0.091, +0.068] | +0.009 [-0.068, +0.093] | -0.008 [-0.087, +0.073] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.001 [-0.084, +0.084] | -0.001 [-0.091, +0.088] | +0.011 [-0.073, +0.095] | -0.024 [-0.120, +0.068] | +0.010 [-0.077, +0.098] | +0.004 [-0.081, +0.084] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.102 [+0.008, +0.192] | +0.089 [-0.009, +0.181] | +0.104 [+0.002, +0.194] | +0.092 [-0.006, +0.182] | +0.116 [+0.019, +0.205] | +0.108 [+0.015, +0.199] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.144 [+0.075, +0.210] | +0.105 [+0.037, +0.172] | +0.115 [+0.048, +0.182] | +0.101 [+0.030, +0.168] | +0.106 [+0.033, +0.174] | +0.110 [+0.006, +0.210] | +0.028 [-0.044, +0.097] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.139 [+0.071, +0.204] | +0.098 [+0.026, +0.169] | +0.110 [+0.041, +0.179] | +0.111 [+0.032, +0.183] | +0.104 [+0.029, +0.177] | +0.157 [+0.064, +0.246] | +0.076 [+0.002, +0.149] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.040 [-0.118, +0.036] | -0.057 [-0.135, +0.020] | -0.040 [-0.120, +0.039] | -0.036 [-0.116, +0.047] | -0.035 [-0.117, +0.042] | -0.033 [-0.110, +0.043] | -0.037 [-0.115, +0.040] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.007 [-0.069, +0.085] | -0.013 [-0.088, +0.063] | +0.008 [-0.072, +0.087] | +0.012 [-0.066, +0.089] | +0.015 [-0.062, +0.089] | +0.013 [-0.065, +0.094] | +0.013 [-0.061, +0.089] |
| NICP per coppia in mm | +0.096 [+0.019, +0.169] | +0.090 [+0.010, +0.167] | +0.101 [+0.019, +0.181] | +0.088 [+0.004, +0.172] | +0.103 [+0.027, +0.182] | +0.098 [+0.018, +0.172] | - |
| NICP per coppia cs | +0.085 [+0.007, +0.164] | +0.082 [+0.003, +0.158] | +0.089 [+0.004, +0.171] | +0.073 [-0.010, +0.157] | +0.095 [+0.017, +0.173] | +0.086 [+0.007, +0.164] | - |
| NICP su template in mm | +0.085 [-0.028, +0.196] | +0.079 [-0.042, +0.197] | +0.073 [-0.044, +0.186] | +0.085 [-0.030, +0.194] | +0.091 [-0.025, +0.199] | +0.100 [-0.015, +0.215] | +0.091 [-0.025, +0.203] |
| NICP su template cs | +0.162 [+0.063, +0.263] | +0.145 [+0.048, +0.250] | +0.158 [+0.059, +0.261] | +0.172 [+0.067, +0.284] | +0.156 [+0.058, +0.253] | +0.176 [+0.076, +0.280] | +0.163 [+0.065, +0.267] |
| varifold in mm (massa unitaria) | +0.037 [-0.026, +0.097] | +0.011 [-0.056, +0.072] | +0.026 [-0.042, +0.097] | +0.016 [-0.051, +0.080] | +0.019 [-0.049, +0.083] | +0.006 [-0.069, +0.076] | +0.026 [-0.040, +0.086] |
| GNM (visto) preregistrato, coefficienti | +0.077 [-0.019, +0.178] | +0.057 [-0.046, +0.164] | +0.095 [-0.003, +0.206] | +0.088 [-0.019, +0.188] | +0.060 [-0.032, +0.148] | +0.088 [-0.013, +0.195] | +0.081 [-0.016, +0.186] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.070 [-0.032, +0.174] | +0.051 [-0.057, +0.165] | +0.076 [-0.031, +0.187] | +0.068 [-0.027, +0.168] | +0.075 [-0.027, +0.170] | +0.082 [-0.018, +0.190] | +0.074 [-0.027, +0.181] |
| GNM (visto) vA, coefficienti | +0.074 [-0.024, +0.177] | +0.056 [-0.048, +0.161] | +0.093 [-0.010, +0.207] | +0.083 [-0.022, +0.184] | +0.057 [-0.036, +0.150] | +0.084 [-0.016, +0.193] | +0.078 [-0.019, +0.183] |
| GNM (visto) vA, mesh d'identita' FR | +0.067 [-0.035, +0.170] | +0.048 [-0.059, +0.164] | +0.072 [-0.037, +0.184] | +0.065 [-0.032, +0.165] | +0.073 [-0.029, +0.169] | +0.079 [-0.022, +0.188] | +0.071 [-0.031, +0.180] |
| GNM (visto) vB, coefficienti | +0.071 [-0.006, +0.157] | +0.081 [-0.003, +0.175] | +0.063 [-0.019, +0.152] | +0.056 [-0.031, +0.147] | +0.080 [+0.000, +0.169] | +0.076 [-0.005, +0.161] | +0.079 [-0.000, +0.168] |
| GNM (visto) vB, mesh d'identita' FR | +0.004 [-0.078, +0.084] | -0.011 [-0.107, +0.085] | +0.012 [-0.075, +0.095] | -0.003 [-0.084, +0.076] | +0.016 [-0.063, +0.095] | +0.011 [-0.072, +0.093] | +0.009 [-0.077, +0.091] |
| GNM (visto) vB, composizione S_B + k d_P | -0.041 [-0.115, +0.034] | -0.054 [-0.137, +0.033] | -0.034 [-0.110, +0.045] | -0.053 [-0.125, +0.022] | -0.026 [-0.099, +0.047] | -0.038 [-0.114, +0.037] | -0.035 [-0.113, +0.041] |
| FLAME 2023 Open preregistrato, coefficienti | +0.122 [+0.037, +0.209] | +0.114 [+0.019, +0.206] | +0.127 [+0.027, +0.226] | +0.110 [+0.024, +0.198] | +0.129 [+0.043, +0.212] | +0.134 [+0.043, +0.223] | +0.124 [+0.037, +0.215] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.086 [-0.008, +0.173] | +0.071 [-0.029, +0.166] | +0.091 [-0.008, +0.190] | +0.073 [-0.016, +0.160] | +0.094 [-0.003, +0.180] | +0.100 [+0.007, +0.192] | +0.088 [-0.006, +0.177] |
| FLAME 2023 Open vA, coefficienti | +0.122 [+0.037, +0.209] | +0.114 [+0.019, +0.206] | +0.127 [+0.027, +0.226] | +0.110 [+0.024, +0.198] | +0.129 [+0.043, +0.212] | +0.134 [+0.043, +0.223] | +0.124 [+0.037, +0.215] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.086 [-0.008, +0.173] | +0.071 [-0.029, +0.166] | +0.091 [-0.008, +0.190] | +0.073 [-0.016, +0.160] | +0.094 [-0.003, +0.180] | +0.100 [+0.007, +0.192] | +0.088 [-0.006, +0.177] |
| FLAME 2023 Open vB, coefficienti | +0.094 [+0.015, +0.178] | +0.123 [+0.034, +0.221] | +0.098 [+0.013, +0.190] | +0.059 [-0.025, +0.142] | +0.101 [+0.020, +0.187] | +0.088 [+0.004, +0.167] | +0.101 [+0.020, +0.184] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.033 [-0.053, +0.118] | +0.018 [-0.075, +0.112] | +0.038 [-0.053, +0.132] | +0.019 [-0.066, +0.104] | +0.043 [-0.046, +0.126] | +0.046 [-0.039, +0.134] | +0.035 [-0.054, +0.125] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.005 [-0.086, +0.078] | -0.016 [-0.101, +0.073] | +0.001 [-0.085, +0.086] | -0.021 [-0.103, +0.061] | +0.010 [-0.070, +0.086] | +0.005 [-0.074, +0.089] | -0.002 [-0.084, +0.082] |
| ArcFace-render, normal map 3 viste (G0) | +0.170 [+0.091, +0.244] | +0.151 [+0.062, +0.237] | +0.152 [+0.068, +0.229] | +0.145 [+0.058, +0.228] | +0.154 [+0.075, +0.223] | +0.204 [+0.111, +0.297] | +0.161 [+0.076, +0.240] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.179 [+0.095, +0.260] | +0.156 [+0.065, +0.240] | +0.165 [+0.076, +0.250] | +0.144 [+0.048, +0.233] | +0.162 [+0.076, +0.244] | +0.217 [+0.113, +0.316] | +0.167 [+0.082, +0.251] |
| ShapeDNA k=50 | +0.234 [+0.164, +0.300] | +0.220 [+0.141, +0.296] | +0.217 [+0.136, +0.295] | +0.200 [+0.123, +0.278] | +0.209 [+0.138, +0.282] | +0.229 [+0.143, +0.311] | +0.170 [+0.087, +0.255] |
| HKS globale | +0.265 [+0.194, +0.334] | +0.249 [+0.171, +0.329] | +0.256 [+0.178, +0.337] | +0.245 [+0.167, +0.318] | +0.252 [+0.174, +0.324] | +0.277 [+0.190, +0.357] | +0.235 [+0.148, +0.314] |
| WKS globale | +0.258 [+0.182, +0.326] | +0.258 [+0.181, +0.337] | +0.256 [+0.172, +0.334] | +0.238 [+0.157, +0.314] | +0.249 [+0.175, +0.315] | +0.215 [+0.125, +0.306] | +0.233 [+0.145, +0.319] |
| taglia stimata dalla mesh (non oracolo) | +0.171 [+0.058, +0.269] | +0.171 [+0.053, +0.278] | +0.178 [+0.060, +0.277] | +0.173 [+0.064, +0.271] | +0.170 [+0.058, +0.266] | +0.163 [+0.056, +0.263] | +0.171 [+0.058, +0.272] |
| taglia oracolo (S di FR) | +0.088 [-0.050, +0.213] | +0.085 [-0.055, +0.213] | +0.089 [-0.055, +0.216] | +0.079 [-0.056, +0.204] | +0.091 [-0.047, +0.213] | +0.097 [-0.039, +0.217] | +0.093 [-0.047, +0.219] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.248 [+0.172, +0.315] | +0.226 [+0.153, +0.300] | +0.248 [+0.169, +0.322] | +0.229 [+0.155, +0.300] | +0.231 [+0.154, +0.305] | +0.225 [+0.109, +0.333] | +0.220 [+0.136, +0.296] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.231 [+0.163, +0.302] | +0.217 [+0.147, +0.288] | +0.229 [+0.154, +0.305] | +0.208 [+0.134, +0.284] | +0.215 [+0.142, +0.290] | +0.241 [+0.146, +0.326] | +0.185 [+0.109, +0.260] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.236 [+0.162, +0.298] | +0.205 [+0.135, +0.268] | +0.225 [+0.147, +0.295] | +0.213 [+0.140, +0.280] | +0.208 [+0.137, +0.275] | +0.236 [+0.129, +0.333] | +0.181 [+0.105, +0.252] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.101 [+0.019, +0.178] | +0.084 [-0.001, +0.171] | +0.099 [+0.009, +0.179] | +0.097 [+0.013, +0.179] | +0.086 [-0.000, +0.172] | +0.096 [+0.016, +0.178] | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.160 [+0.087, +0.227] | +0.132 [+0.059, +0.197] | +0.155 [+0.076, +0.229] | +0.146 [+0.068, +0.221] | +0.140 [+0.062, +0.214] | +0.126 [+0.037, +0.203] | +0.092 [+0.006, +0.174] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.041 [-0.111, +0.030] | -0.058 [-0.130, +0.017] | -0.043 [-0.116, +0.030] | -0.048 [-0.114, +0.030] | -0.035 [-0.106, +0.035] | -0.050 [-0.120, +0.025] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.003 [-0.068, +0.061] | -0.019 [-0.085, +0.044] | -0.005 [-0.077, +0.064] | -0.016 [-0.085, +0.057] | +0.004 [-0.063, +0.068] | -0.014 [-0.083, +0.055] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.005 [-0.080, +0.069] | -0.006 [-0.083, +0.075] | +0.000 [-0.077, +0.078] | -0.025 [-0.113, +0.060] | +0.005 [-0.069, +0.083] | -0.002 [-0.081, +0.072] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.096 [+0.003, +0.178] | +0.083 [-0.009, +0.171] | +0.093 [-0.006, +0.177] | +0.090 [-0.007, +0.180] | +0.110 [+0.010, +0.197] | +0.102 [+0.010, +0.187] | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.138 [+0.069, +0.203] | +0.099 [+0.031, +0.163] | +0.104 [+0.031, +0.173] | +0.099 [+0.026, +0.164] | +0.100 [+0.028, +0.164] | +0.105 [+0.007, +0.203] | +0.024 [-0.049, +0.092] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.133 [+0.064, +0.194] | +0.093 [+0.025, +0.156] | +0.099 [+0.027, +0.165] | +0.109 [+0.033, +0.179] | +0.098 [+0.026, +0.166] | +0.151 [+0.051, +0.236] | +0.073 [-0.002, +0.141] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.023 [-0.097, +0.056] | -0.037 [-0.111, +0.040] | -0.020 [-0.094, +0.057] | -0.025 [-0.105, +0.060] | -0.020 [-0.096, +0.057] | -0.016 [-0.089, +0.062] | -0.017 [-0.093, +0.059] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.020 [-0.093, +0.055] | -0.039 [-0.108, +0.039] | -0.014 [-0.091, +0.062] | -0.017 [-0.091, +0.060] | -0.017 [-0.089, +0.057] | -0.014 [-0.089, +0.067] | -0.012 [-0.082, +0.063] |
| NICP per coppia in mm | +0.063 [-0.016, +0.141] | +0.056 [-0.022, +0.135] | +0.071 [-0.012, +0.149] | +0.053 [-0.031, +0.135] | +0.067 [-0.014, +0.148] | +0.066 [-0.013, +0.143] | - |
| NICP per coppia cs | +0.051 [-0.027, +0.124] | +0.047 [-0.029, +0.115] | +0.059 [-0.022, +0.135] | +0.037 [-0.042, +0.115] | +0.058 [-0.020, +0.135] | +0.052 [-0.025, +0.125] | - |
| NICP su template in mm | +0.134 [+0.018, +0.239] | +0.132 [+0.009, +0.244] | +0.128 [+0.013, +0.236] | +0.129 [+0.011, +0.238] | +0.139 [+0.028, +0.244] | +0.145 [+0.021, +0.253] | +0.141 [+0.022, +0.249] |
| NICP su template cs | +0.139 [+0.042, +0.237] | +0.124 [+0.028, +0.224] | +0.143 [+0.046, +0.242] | +0.148 [+0.042, +0.254] | +0.129 [+0.029, +0.223] | +0.153 [+0.050, +0.253] | +0.143 [+0.046, +0.242] |
| varifold in mm (massa unitaria) | +0.036 [-0.021, +0.092] | +0.013 [-0.046, +0.071] | +0.030 [-0.033, +0.092] | +0.011 [-0.050, +0.073] | +0.020 [-0.038, +0.080] | +0.003 [-0.070, +0.072] | +0.029 [-0.028, +0.088] |
| GNM (visto) preregistrato, coefficienti | +0.104 [+0.015, +0.200] | +0.082 [-0.016, +0.186] | +0.127 [+0.030, +0.241] | +0.110 [+0.018, +0.204] | +0.088 [+0.008, +0.171] | +0.113 [+0.019, +0.222] | +0.109 [+0.019, +0.204] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.077 [-0.016, +0.179] | +0.066 [-0.032, +0.175] | +0.096 [-0.000, +0.199] | +0.059 [-0.031, +0.156] | +0.078 [-0.013, +0.174] | +0.087 [-0.009, +0.193] | +0.081 [-0.010, +0.186] |
| GNM (visto) vA, coefficienti | +0.102 [+0.014, +0.199] | +0.082 [-0.018, +0.185] | +0.126 [+0.030, +0.238] | +0.107 [+0.014, +0.202] | +0.087 [+0.006, +0.169] | +0.111 [+0.014, +0.219] | +0.108 [+0.018, +0.205] |
| GNM (visto) vA, mesh d'identita' SR | +0.081 [-0.010, +0.183] | +0.072 [-0.027, +0.186] | +0.099 [+0.004, +0.204] | +0.063 [-0.028, +0.160] | +0.082 [-0.011, +0.181] | +0.092 [-0.004, +0.199] | +0.086 [-0.007, +0.190] |
| GNM (visto) vB, coefficienti | +0.064 [-0.022, +0.158] | +0.073 [-0.017, +0.174] | +0.062 [-0.024, +0.162] | +0.046 [-0.043, +0.144] | +0.069 [-0.016, +0.165] | +0.070 [-0.014, +0.158] | +0.074 [-0.013, +0.171] |
| GNM (visto) vB, mesh d'identita' SR | -0.041 [-0.112, +0.033] | -0.043 [-0.118, +0.040] | -0.031 [-0.105, +0.044] | -0.061 [-0.134, +0.016] | -0.031 [-0.102, +0.042] | -0.039 [-0.113, +0.033] | -0.031 [-0.101, +0.044] |
| FLAME 2023 Open preregistrato, coefficienti | +0.134 [+0.046, +0.224] | +0.122 [+0.029, +0.221] | +0.146 [+0.051, +0.246] | +0.124 [+0.037, +0.212] | +0.138 [+0.050, +0.218] | +0.141 [+0.053, +0.233] | +0.136 [+0.047, +0.228] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.131 [+0.040, +0.233] | +0.130 [+0.027, +0.242] | +0.143 [+0.049, +0.251] | +0.114 [+0.022, +0.214] | +0.130 [+0.037, +0.226] | +0.139 [+0.049, +0.241] | +0.133 [+0.042, +0.237] |
| FLAME 2023 Open vA, coefficienti | +0.134 [+0.046, +0.224] | +0.122 [+0.029, +0.221] | +0.146 [+0.051, +0.246] | +0.124 [+0.037, +0.212] | +0.138 [+0.050, +0.218] | +0.141 [+0.053, +0.233] | +0.136 [+0.047, +0.228] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.131 [+0.040, +0.233] | +0.130 [+0.027, +0.242] | +0.143 [+0.049, +0.251] | +0.114 [+0.022, +0.214] | +0.130 [+0.037, +0.226] | +0.139 [+0.049, +0.241] | +0.133 [+0.042, +0.237] |
| FLAME 2023 Open vB, coefficienti | +0.093 [+0.008, +0.182] | +0.120 [+0.017, +0.220] | +0.105 [+0.016, +0.194] | +0.059 [-0.023, +0.148] | +0.092 [+0.000, +0.180] | +0.092 [+0.001, +0.174] | +0.101 [+0.017, +0.191] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.006 [-0.090, +0.084] | -0.010 [-0.098, +0.084] | +0.003 [-0.083, +0.095] | -0.029 [-0.114, +0.066] | +0.003 [-0.079, +0.089] | +0.001 [-0.084, +0.091] | -0.001 [-0.085, +0.087] |
| ArcFace-render, normal map 3 viste (G0) | +0.148 [+0.068, +0.220] | +0.128 [+0.042, +0.213] | +0.136 [+0.051, +0.211] | +0.120 [+0.032, +0.202] | +0.133 [+0.055, +0.206] | +0.176 [+0.083, +0.271] | +0.141 [+0.055, +0.218] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.157 [+0.074, +0.234] | +0.131 [+0.045, +0.210] | +0.147 [+0.059, +0.230] | +0.114 [+0.024, +0.198] | +0.140 [+0.052, +0.220] | +0.198 [+0.095, +0.295] | +0.141 [+0.054, +0.223] |
| ShapeDNA k=50 | +0.225 [+0.153, +0.295] | +0.206 [+0.121, +0.287] | +0.213 [+0.126, +0.291] | +0.193 [+0.115, +0.272] | +0.204 [+0.125, +0.278] | +0.214 [+0.131, +0.298] | +0.165 [+0.075, +0.253] |
| HKS globale | +0.257 [+0.185, +0.326] | +0.240 [+0.161, +0.316] | +0.256 [+0.171, +0.332] | +0.242 [+0.162, +0.316] | +0.252 [+0.172, +0.325] | +0.254 [+0.172, +0.331] | +0.232 [+0.140, +0.315] |
| WKS globale | +0.247 [+0.173, +0.313] | +0.245 [+0.167, +0.323] | +0.250 [+0.165, +0.330] | +0.230 [+0.145, +0.306] | +0.241 [+0.166, +0.309] | +0.198 [+0.107, +0.287] | +0.230 [+0.142, +0.318] |
| taglia stimata dalla mesh (non oracolo) | +0.206 [+0.100, +0.299] | +0.211 [+0.099, +0.308] | +0.213 [+0.107, +0.303] | +0.199 [+0.094, +0.295] | +0.212 [+0.107, +0.303] | +0.197 [+0.090, +0.293] | +0.209 [+0.104, +0.306] |
| taglia oracolo (S di FR) | +0.267 [+0.144, +0.380] | +0.263 [+0.132, +0.379] | +0.275 [+0.149, +0.389] | +0.258 [+0.139, +0.368] | +0.271 [+0.145, +0.378] | +0.271 [+0.152, +0.379] | +0.274 [+0.148, +0.386] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.234 [+0.160, +0.299] | +0.204 [+0.132, +0.276] | +0.235 [+0.156, +0.305] | +0.214 [+0.139, +0.284] | +0.218 [+0.141, +0.290] | +0.214 [+0.105, +0.321] | +0.200 [+0.119, +0.272] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.216 [+0.147, +0.286] | +0.199 [+0.129, +0.270] | +0.219 [+0.143, +0.294] | +0.197 [+0.125, +0.271] | +0.201 [+0.130, +0.276] | +0.214 [+0.115, +0.304] | +0.170 [+0.092, +0.246] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.226 [+0.152, +0.289] | +0.188 [+0.117, +0.255] | +0.218 [+0.140, +0.285] | +0.199 [+0.122, +0.265] | +0.198 [+0.124, +0.263] | +0.238 [+0.141, +0.329] | +0.166 [+0.088, +0.237] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.103 [+0.035, +0.170] | +0.081 [+0.008, +0.161] | +0.097 [+0.024, +0.165] | +0.080 [+0.006, +0.149] | +0.086 [+0.013, +0.159] | +0.100 [+0.017, +0.179] | - |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.175 [+0.106, +0.236] | +0.149 [+0.079, +0.213] | +0.171 [+0.098, +0.237] | +0.155 [+0.083, +0.222] | +0.164 [+0.094, +0.229] | +0.139 [+0.063, +0.209] | +0.115 [+0.039, +0.188] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.024 [-0.097, +0.050] | -0.037 [-0.116, +0.040] | -0.020 [-0.096, +0.057] | -0.037 [-0.111, +0.041] | -0.023 [-0.100, +0.049] | -0.031 [-0.107, +0.047] | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.021 [-0.084, +0.045] | -0.037 [-0.104, +0.031] | -0.017 [-0.084, +0.053] | -0.036 [-0.101, +0.034] | -0.020 [-0.084, +0.046] | -0.031 [-0.099, +0.035] | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.026 [-0.094, +0.041] | -0.031 [-0.105, +0.044] | -0.017 [-0.092, +0.056] | -0.048 [-0.124, +0.025] | -0.016 [-0.086, +0.049] | -0.022 [-0.100, +0.050] | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.088 [+0.000, +0.169] | +0.076 [-0.011, +0.160] | +0.097 [+0.004, +0.178] | +0.078 [-0.012, +0.158] | +0.096 [+0.003, +0.174] | +0.093 [+0.003, +0.178] | - |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.128 [+0.063, +0.190] | +0.088 [+0.023, +0.154] | +0.099 [+0.032, +0.163] | +0.084 [+0.017, +0.147] | +0.094 [+0.027, +0.154] | +0.080 [-0.019, +0.185] | +0.016 [-0.050, +0.082] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | +0.121 [+0.054, +0.178] | +0.076 [+0.011, +0.137] | +0.088 [+0.019, +0.148] | +0.087 [+0.013, +0.152] | +0.089 [+0.021, +0.149] | +0.134 [+0.038, +0.216] | +0.065 [-0.006, +0.126] |

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
| **C3M e205, d_F cal.*** | 0.336 [0.264, 0.400] | 0.358 [0.283, 0.424] | 0.350 [0.274, 0.418] | 0.352 [0.275, 0.420] | 0.352 [0.276, 0.422] | 0.317 [0.250, 0.380] | 0.399 [0.317, 0.474] |
| **C3M e205, d_F non cal.** | 0.357 [0.273, 0.433] | 0.371 [0.287, 0.449] | 0.366 [0.279, 0.444] | 0.362 [0.275, 0.439] | 0.368 [0.278, 0.451] | 0.343 [0.259, 0.418] | 0.396 [0.310, 0.477] |
| C3M e205, d_P | 0.363 [0.275, 0.442] | 0.377 [0.289, 0.455] | 0.372 [0.282, 0.451] | 0.368 [0.281, 0.447] | 0.371 [0.281, 0.452] | 0.340 [0.259, 0.415] | 0.397 [0.308, 0.478] |
| **ICP + Chamfer in mm (per coppia)** | 0.409 [0.326, 0.481] | 0.415 [0.332, 0.486] | 0.414 [0.331, 0.486] | 0.402 [0.314, 0.475] | 0.412 [0.328, 0.484] | 0.404 [0.325, 0.476] | 0.416 [0.333, 0.489] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.374 [0.293, 0.449] | 0.378 [0.297, 0.452] | 0.380 [0.298, 0.455] | 0.367 [0.281, 0.443] | 0.376 [0.296, 0.451] | 0.373 [0.292, 0.447] | 0.377 [0.297, 0.452] |
| **NICP per coppia in mm** | 0.278 [0.194, 0.360] | 0.278 [0.196, 0.358] | 0.280 [0.195, 0.363] | 0.272 [0.184, 0.359] | 0.279 [0.195, 0.361] | 0.281 [0.201, 0.359] | - |
| **NICP per coppia cs** | 0.286 [0.198, 0.372] | 0.286 [0.199, 0.370] | 0.287 [0.199, 0.373] | 0.282 [0.191, 0.369] | 0.287 [0.198, 0.373] | 0.292 [0.204, 0.373] | - |
| **NICP su template in mm** | 0.223 [0.125, 0.319] | 0.224 [0.125, 0.320] | 0.225 [0.124, 0.322] | 0.228 [0.130, 0.323] | 0.223 [0.125, 0.320] | 0.218 [0.121, 0.318] | 0.224 [0.126, 0.320] |
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
| C3M e205, d_F cal.* | 0.329 [0.255, 0.398] | 0.351 [0.275, 0.422] | 0.345 [0.266, 0.415] | 0.345 [0.265, 0.417] | 0.342 [0.261, 0.417] | 0.308 [0.236, 0.376] | 0.389 [0.303, 0.467] |
| C3M e205, d_F non cal. | 0.371 [0.287, 0.444] | 0.387 [0.304, 0.460] | 0.382 [0.297, 0.456] | 0.375 [0.290, 0.450] | 0.381 [0.295, 0.459] | 0.354 [0.272, 0.428] | 0.411 [0.323, 0.489] |
| **C3M e205, d_P** | 0.382 [0.304, 0.452] | 0.398 [0.321, 0.469] | 0.393 [0.313, 0.464] | 0.387 [0.307, 0.459] | 0.389 [0.310, 0.463] | 0.357 [0.281, 0.426] | 0.417 [0.336, 0.491] |
| **ICP + Chamfer in mm (per coppia)** | 0.383 [0.298, 0.456] | 0.389 [0.305, 0.461] | 0.389 [0.305, 0.463] | 0.376 [0.290, 0.452] | 0.387 [0.301, 0.460] | 0.378 [0.296, 0.450] | 0.390 [0.304, 0.463] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.397 [0.320, 0.467] | 0.400 [0.323, 0.471] | 0.403 [0.325, 0.475] | 0.389 [0.308, 0.460] | 0.400 [0.324, 0.471] | 0.396 [0.318, 0.466] | 0.400 [0.324, 0.472] |
| **NICP per coppia in mm** | 0.314 [0.232, 0.391] | 0.314 [0.232, 0.391] | 0.317 [0.234, 0.396] | 0.309 [0.224, 0.390] | 0.317 [0.235, 0.394] | 0.315 [0.235, 0.390] | - |
| **NICP per coppia cs** | 0.323 [0.242, 0.401] | 0.321 [0.241, 0.400] | 0.325 [0.242, 0.404] | 0.320 [0.236, 0.399] | 0.325 [0.243, 0.403] | 0.327 [0.247, 0.404] | - |
| **NICP su template in mm** | 0.156 [0.061, 0.253] | 0.156 [0.061, 0.254] | 0.157 [0.060, 0.254] | 0.160 [0.066, 0.258] | 0.155 [0.060, 0.251] | 0.150 [0.055, 0.248] | 0.156 [0.061, 0.254] |
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
| hifi3d | FR | crop | 49306 | factorized s1234, d_F cal. | 0.663 [0.589, 0.721] | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.703 [0.636, 0.758] | -0.040 [-0.076, -0.006] |
| hifi3d | FR | all_cross | 147530 | factorized s1234, d_F cal. | 0.709 [0.634, 0.767] | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.699 [0.631, 0.755] | +0.010 [-0.023, +0.046] |
| hifi3d | FR | stessa_topologia crop | 4950 | ctrlfr s2345 | 0.789 [0.723, 0.842] | ULRSSM [faces_xyz] per coppia, mm (FR) | 0.840 [0.783, 0.885] | -0.051 [-0.089, -0.016] |
| hifi3d | FR | stessa_topologia tutte | 29700 | ctrlfr s1234 | 0.780 [0.710, 0.839] | ULRSSM [faces_xyz] per coppia, mm (FR) | 0.788 [0.719, 0.844] | -0.008 [-0.042, +0.028] |
| hifi3d | SR | crop | 49306 | factorized s1234, d_P | 0.550 [0.472, 0.623] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.721 [0.653, 0.776] | -0.171 [-0.261, -0.090] |
| hifi3d | SR | all_cross | 147530 | factorized s1234, d_P | 0.557 [0.485, 0.624] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.716 [0.647, 0.771] | -0.159 [-0.245, -0.081] |
| hifi3d | SR | stessa_topologia crop | 4950 | factorized s1234, d_P | 0.666 [0.601, 0.727] | ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.760 [0.707, 0.811] | -0.094 [-0.153, -0.033] |
| hifi3d | SR | stessa_topologia tutte | 29700 | factorized s1234, d_P | 0.631 [0.560, 0.695] | ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.680 [0.613, 0.742] | -0.049 [-0.110, +0.010] |
| facescape | FR | crop | 46931 | C3M e123, d_F non cal. | 0.539 [0.460, 0.617] | GNM (visto) vB, composizione S_B + k d_P | 0.761 [0.700, 0.812] | -0.222 [-0.282, -0.163] |
| facescape | FR | all_cross | 135655 | C3M e205, d_F non cal. | 0.469 [0.403, 0.533] | GNM (visto) vB, composizione S_B + k d_P | 0.783 [0.724, 0.831] | -0.313 [-0.371, -0.259] |
| facescape | FR | stessa_topologia crop | 4950 | ctrlfr s1234 | 0.730 [0.675, 0.782] | GNM (visto) vB, composizione S_B + k d_P | 0.851 [0.804, 0.889] | -0.121 [-0.177, -0.066] |
| facescape | FR | stessa_topologia tutte | 29700 | C3M e205, d_F cal. | 0.728 [0.669, 0.776] | GNM (visto) vB, composizione S_B + k d_P | 0.814 [0.759, 0.859] | -0.086 [-0.142, -0.037] |
| facescape | SR | crop | 46931 | C3M e123, d_P | 0.576 [0.506, 0.644] | GNM (visto) vB, mesh d'identita' SR | 0.830 [0.778, 0.873] | -0.255 [-0.315, -0.195] |
| facescape | SR | all_cross | 135655 | C3M e205, d_P | 0.499 [0.435, 0.557] | GNM (visto) vB, mesh d'identita' SR | 0.839 [0.786, 0.881] | -0.340 [-0.390, -0.291] |
| facescape | SR | stessa_topologia crop | 4950 | C3M e205, d_P | 0.746 [0.673, 0.808] | GNM (visto) vB, mesh d'identita' SR | 0.872 [0.828, 0.910] | -0.126 [-0.194, -0.064] |
| facescape | SR | stessa_topologia tutte | 29700 | C3M e205, d_P | 0.756 [0.699, 0.805] | GNM (visto) vB, mesh d'identita' SR | 0.855 [0.807, 0.895] | -0.099 [-0.146, -0.056] |
| faceverse | FR | crop | 49500 | C3M e205, d_F non cal. | 0.297 [0.227, 0.365] | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.341 [0.266, 0.409] | -0.044 [-0.115, +0.028] |
| faceverse | FR | all_cross | 148500 | C3M e205, d_F non cal. | 0.302 [0.233, 0.370] | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.337 [0.266, 0.406] | -0.035 [-0.103, +0.035] |
| faceverse | FR | stessa_topologia crop | 4950 | ctrlfr s1234 | 0.316 [0.217, 0.404] | ULRSSM [faces_xyz] per coppia, mm (FR) | 0.383 [0.289, 0.464] | -0.066 [-0.153, +0.024] |
| faceverse | FR | stessa_topologia tutte | 29700 | C3M e205, d_F cal. | 0.326 [0.253, 0.395] | ICP + Chamfer in mm (per coppia) | 0.343 [0.269, 0.413] | -0.017 [-0.077, +0.046] |
| faceverse | SR | crop | 49500 | C3M e205, d_P | 0.308 [0.241, 0.376] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.332 [0.270, 0.390] | -0.023 [-0.097, +0.055] |
| faceverse | SR | all_cross | 148500 | C3M e205, d_P | 0.312 [0.244, 0.378] | GNM (visto) vB, mesh d'identita' SR | 0.323 [0.238, 0.397] | -0.011 [-0.084, +0.064] |
| faceverse | SR | stessa_topologia crop | 4950 | C3M e205, d_P | 0.332 [0.251, 0.421] | ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.379 [0.298, 0.460] | -0.047 [-0.141, +0.053] |
| faceverse | SR | stessa_topologia tutte | 29700 | C3M e205, d_P | 0.334 [0.263, 0.403] | GNM (visto) vB, mesh d'identita' SR | 0.322 [0.236, 0.395] | +0.012 [-0.063, +0.087] |
| faceverse_neutral | FR | crop | 49500 | C3M e205, d_F non cal. | 0.345 [0.262, 0.420] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.481 [0.394, 0.559] | -0.135 [-0.225, -0.039] |
| faceverse_neutral | FR | all_cross | 148500 | C3M e205, d_F non cal. | 0.353 [0.269, 0.429] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.479 [0.393, 0.557] | -0.126 [-0.220, -0.029] |
| faceverse_neutral | FR | stessa_topologia crop | 4950 | ctrlfr s1234 | 0.386 [0.304, 0.464] | ICP + Chamfer in mm (per coppia) | 0.444 [0.363, 0.515] | -0.058 [-0.143, +0.022] |
| faceverse_neutral | FR | stessa_topologia tutte | 29700 | C3M e205, d_F cal. | 0.395 [0.312, 0.470] | ICP + Chamfer in mm (per coppia) | 0.419 [0.337, 0.492] | -0.024 [-0.095, +0.050] |
| faceverse_neutral | SR | crop | 49500 | C3M e205, d_P | 0.372 [0.294, 0.442] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.523 [0.448, 0.591] | -0.151 [-0.235, -0.063] |
| faceverse_neutral | SR | all_cross | 148500 | C3M e205, d_P | 0.378 [0.304, 0.448] | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.521 [0.447, 0.589] | -0.143 [-0.229, -0.057] |
| faceverse_neutral | SR | stessa_topologia crop | 4950 | C3M e205, d_P | 0.419 [0.338, 0.498] | ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.469 [0.384, 0.553] | -0.050 [-0.140, +0.032] |
| faceverse_neutral | SR | stessa_topologia tutte | 29700 | C3M e205, d_P | 0.417 [0.339, 0.490] | ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.419 [0.340, 0.491] | -0.002 [-0.073, +0.069] |

Ogni lettura nostra di riferimento contro il miglior concorrente (crop, all_cross):

| dominio | GT | insieme | miglior concorrente | lettura nostra | rho nostro | rho concorrente | delta [IC] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s1234, d_F cal. | 0.663 | 0.703 | -0.040 [-0.076, -0.006] |
| hifi3d | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s2345, d_F cal. | 0.552 | 0.703 | -0.151 [-0.190, -0.113] |
| hifi3d | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s1234, d_F non cal. | 0.563 | 0.703 | -0.140 [-0.201, -0.084] |
| hifi3d | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s2345, d_F non cal. | 0.516 | 0.703 | -0.187 [-0.247, -0.127] |
| hifi3d | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | ctrlfr s1234 | 0.236 | 0.703 | -0.467 [-0.516, -0.412] |
| hifi3d | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | ctrlfr s2345 | 0.279 | 0.703 | -0.424 [-0.470, -0.377] |
| hifi3d | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s1234, d_F cal. | 0.709 | 0.699 | +0.010 [-0.023, +0.046] |
| hifi3d | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s2345, d_F cal. | 0.661 | 0.699 | -0.038 [-0.073, -0.004] |
| hifi3d | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s1234, d_F non cal. | 0.584 | 0.699 | -0.115 [-0.166, -0.068] |
| hifi3d | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s2345, d_F non cal. | 0.561 | 0.699 | -0.137 [-0.191, -0.084] |
| hifi3d | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | ctrlfr s1234 | 0.515 | 0.699 | -0.183 [-0.224, -0.147] |
| hifi3d | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | ctrlfr s2345 | 0.529 | 0.699 | -0.170 [-0.211, -0.129] |
| hifi3d | SR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_P | 0.550 | 0.721 | -0.171 [-0.261, -0.090] |
| hifi3d | SR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_P | 0.514 | 0.721 | -0.207 [-0.302, -0.112] |
| hifi3d | SR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_P | 0.557 | 0.716 | -0.159 [-0.245, -0.081] |
| hifi3d | SR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_P | 0.540 | 0.716 | -0.176 [-0.265, -0.089] |
| facescape | FR | crop | GNM (visto) vB, composizione S_B + k d_P | factorized s1234, d_F cal. | 0.355 | 0.761 | -0.406 [-0.469, -0.343] |
| facescape | FR | crop | GNM (visto) vB, composizione S_B + k d_P | factorized s2345, d_F cal. | 0.282 | 0.761 | -0.479 [-0.540, -0.418] |
| facescape | FR | crop | GNM (visto) vB, composizione S_B + k d_P | factorized s1234, d_F non cal. | 0.487 | 0.761 | -0.274 [-0.338, -0.210] |
| facescape | FR | crop | GNM (visto) vB, composizione S_B + k d_P | factorized s2345, d_F non cal. | 0.467 | 0.761 | -0.294 [-0.362, -0.232] |
| facescape | FR | crop | GNM (visto) vB, composizione S_B + k d_P | ctrlfr s1234 | 0.266 | 0.761 | -0.495 [-0.550, -0.438] |
| facescape | FR | crop | GNM (visto) vB, composizione S_B + k d_P | ctrlfr s2345 | 0.236 | 0.761 | -0.525 [-0.583, -0.464] |
| facescape | FR | all_cross | GNM (visto) vB, composizione S_B + k d_P | factorized s1234, d_F cal. | 0.365 | 0.783 | -0.418 [-0.461, -0.374] |
| facescape | FR | all_cross | GNM (visto) vB, composizione S_B + k d_P | factorized s2345, d_F cal. | 0.345 | 0.783 | -0.438 [-0.477, -0.397] |
| facescape | FR | all_cross | GNM (visto) vB, composizione S_B + k d_P | factorized s1234, d_F non cal. | 0.438 | 0.783 | -0.345 [-0.398, -0.291] |
| facescape | FR | all_cross | GNM (visto) vB, composizione S_B + k d_P | factorized s2345, d_F non cal. | 0.415 | 0.783 | -0.368 [-0.418, -0.321] |
| facescape | FR | all_cross | GNM (visto) vB, composizione S_B + k d_P | ctrlfr s1234 | 0.325 | 0.783 | -0.458 [-0.500, -0.413] |
| facescape | FR | all_cross | GNM (visto) vB, composizione S_B + k d_P | ctrlfr s2345 | 0.315 | 0.783 | -0.468 [-0.508, -0.422] |
| facescape | SR | crop | GNM (visto) vB, mesh d'identita' SR | factorized s1234, d_P | 0.516 | 0.830 | -0.315 [-0.378, -0.253] |
| facescape | SR | crop | GNM (visto) vB, mesh d'identita' SR | factorized s2345, d_P | 0.474 | 0.830 | -0.357 [-0.423, -0.293] |
| facescape | SR | all_cross | GNM (visto) vB, mesh d'identita' SR | factorized s1234, d_P | 0.482 | 0.839 | -0.357 [-0.406, -0.312] |
| facescape | SR | all_cross | GNM (visto) vB, mesh d'identita' SR | factorized s2345, d_P | 0.456 | 0.839 | -0.383 [-0.427, -0.341] |
| faceverse | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s1234, d_F cal. | 0.252 | 0.341 | -0.088 [-0.153, -0.015] |
| faceverse | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s2345, d_F cal. | 0.234 | 0.341 | -0.107 [-0.174, -0.038] |
| faceverse | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s1234, d_F non cal. | 0.265 | 0.341 | -0.076 [-0.145, -0.002] |
| faceverse | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s2345, d_F non cal. | 0.256 | 0.341 | -0.085 [-0.156, -0.013] |
| faceverse | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | ctrlfr s1234 | 0.256 | 0.341 | -0.085 [-0.152, -0.021] |
| faceverse | FR | crop | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | ctrlfr s2345 | 0.239 | 0.341 | -0.102 [-0.176, -0.031] |
| faceverse | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s1234, d_F cal. | 0.284 | 0.337 | -0.053 [-0.117, +0.018] |
| faceverse | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s2345, d_F cal. | 0.288 | 0.337 | -0.050 [-0.109, +0.016] |
| faceverse | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s1234, d_F non cal. | 0.286 | 0.337 | -0.052 [-0.119, +0.017] |
| faceverse | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | factorized s2345, d_F non cal. | 0.296 | 0.337 | -0.042 [-0.109, +0.026] |
| faceverse | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | ctrlfr s1234 | 0.273 | 0.337 | -0.065 [-0.130, -0.008] |
| faceverse | FR | all_cross | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | ctrlfr s2345 | 0.251 | 0.337 | -0.086 [-0.158, -0.016] |
| faceverse | SR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_P | 0.251 | 0.332 | -0.080 [-0.163, +0.000] |
| faceverse | SR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_P | 0.257 | 0.332 | -0.075 [-0.152, +0.003] |
| faceverse | SR | all_cross | GNM (visto) vB, mesh d'identita' SR | factorized s1234, d_P | 0.273 | 0.323 | -0.050 [-0.119, +0.028] |
| faceverse | SR | all_cross | GNM (visto) vB, mesh d'identita' SR | factorized s2345, d_P | 0.293 | 0.323 | -0.031 [-0.103, +0.040] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_F cal. | 0.282 | 0.481 | -0.198 [-0.304, -0.092] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_F cal. | 0.272 | 0.481 | -0.208 [-0.309, -0.101] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_F non cal. | 0.306 | 0.481 | -0.175 [-0.273, -0.074] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_F non cal. | 0.310 | 0.481 | -0.170 [-0.275, -0.066] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | ctrlfr s1234 | 0.296 | 0.481 | -0.185 [-0.279, -0.094] |
| faceverse_neutral | FR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | ctrlfr s2345 | 0.269 | 0.481 | -0.211 [-0.309, -0.113] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_F cal. | 0.319 | 0.479 | -0.160 [-0.259, -0.055] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_F cal. | 0.335 | 0.479 | -0.144 [-0.241, -0.044] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_F non cal. | 0.326 | 0.479 | -0.153 [-0.245, -0.055] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_F non cal. | 0.352 | 0.479 | -0.128 [-0.229, -0.027] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | ctrlfr s1234 | 0.316 | 0.479 | -0.163 [-0.259, -0.069] |
| faceverse_neutral | FR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | ctrlfr s2345 | 0.289 | 0.479 | -0.190 [-0.293, -0.091] |
| faceverse_neutral | SR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_P | 0.304 | 0.523 | -0.219 [-0.306, -0.132] |
| faceverse_neutral | SR | crop | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_P | 0.316 | 0.523 | -0.207 [-0.301, -0.109] |
| faceverse_neutral | SR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s1234, d_P | 0.323 | 0.521 | -0.198 [-0.282, -0.109] |
| faceverse_neutral | SR | all_cross | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | factorized s2345, d_P | 0.352 | 0.521 | -0.169 [-0.262, -0.078] |

### Scala dell'allineamento (crop): confronti principali

GT FR:

| dominio | passo | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- |
| hifi3d | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.204 [+0.134, +0.275] | +0.292 [+0.211, +0.372] | +0.307 [+0.211, +0.402] | +0.257 [+0.158, +0.356] |
| hifi3d | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.292 [-0.380, -0.204] | -0.285 [-0.373, -0.197] | - | - |
| hifi3d | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.044 [-0.075, -0.014] | -0.061 [-0.090, -0.032] | - | - |
| hifi3d | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.227 [-0.301, -0.161] | -0.231 [-0.304, -0.166] | - | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.207 [+0.148, +0.262] | +0.095 [+0.044, +0.145] | -0.011 [-0.068, +0.045] | +0.027 [-0.022, +0.077] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | +0.221 [+0.161, +0.279] | +0.114 [+0.062, +0.163] | -0.018 [-0.076, +0.040] | +0.046 [-0.005, +0.096] |
| hifi3d | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | +0.145 [+0.079, +0.201] | +0.028 [-0.029, +0.081] | -0.077 [-0.136, -0.017] | +0.001 [-0.055, +0.056] |
| hifi3d | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | +0.099 [+0.032, +0.161] | +0.005 [-0.056, +0.065] | -0.094 [-0.159, -0.025] | -0.010 [-0.074, +0.053] |
| facescape | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.068 [+0.031, +0.102] | +0.065 [+0.043, +0.086] | +0.079 [+0.046, +0.112] | +0.007 [-0.019, +0.033] |
| facescape | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.065 [-0.104, -0.031] | -0.057 [-0.095, -0.023] | - | - |
| facescape | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.004 [-0.036, +0.031] | +0.011 [-0.014, +0.038] | - | - |
| facescape | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.175 [-0.245, -0.100] | -0.112 [-0.179, -0.047] | - | - |
| facescape | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.336 [+0.281, +0.387] | +0.283 [+0.225, +0.343] | +0.125 [+0.060, +0.195] | +0.124 [+0.060, +0.190] |
| facescape | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | +0.269 [+0.216, +0.322] | +0.229 [+0.173, +0.287] | +0.063 [-0.005, +0.131] | +0.074 [+0.011, +0.136] |
| facescape | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | +0.195 [+0.128, +0.264] | +0.062 [+0.020, +0.106] | +0.008 [-0.061, +0.082] | +0.118 [+0.063, +0.170] |
| facescape | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | +0.174 [+0.103, +0.247] | +0.038 [-0.005, +0.083] | +0.028 [-0.047, +0.110] | +0.125 [+0.063, +0.183] |
| faceverse | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.002 [-0.051, +0.052] | +0.037 [-0.010, +0.089] | +0.041 [-0.008, +0.096] | +0.050 [-0.000, +0.103] |
| faceverse | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.039 [-0.082, +0.000] | -0.038 [-0.079, -0.000] | - | - |
| faceverse | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.042 [-0.003, +0.084] | +0.057 [+0.005, +0.104] | - | - |
| faceverse | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.108 [-0.198, -0.019] | -0.090 [-0.185, +0.003] | - | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.009 [-0.049, +0.066] | -0.023 [-0.085, +0.037] | -0.082 [-0.151, -0.019] | -0.051 [-0.117, +0.010] |
| faceverse | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.027 [-0.093, +0.043] | -0.054 [-0.126, +0.016] | -0.121 [-0.202, -0.046] | -0.079 [-0.155, -0.005] |
| faceverse | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.007 [-0.077, +0.067] | -0.026 [-0.098, +0.046] | -0.077 [-0.165, +0.022] | -0.043 [-0.118, +0.030] |
| faceverse | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.016 [-0.090, +0.055] | -0.016 [-0.096, +0.058] | -0.100 [-0.190, -0.015] | -0.034 [-0.116, +0.043] |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.017 [-0.066, +0.034] | +0.025 [-0.024, +0.074] | +0.032 [-0.021, +0.089] | +0.037 [-0.016, +0.092] |
| faceverse_neutral | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.024 [-0.063, +0.015] | -0.022 [-0.061, +0.017] | - | - |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.043 [-0.014, +0.099] | +0.061 [+0.002, +0.119] | - | - |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | +0.006 [-0.058, +0.069] | -0.052 [-0.119, +0.017] | -0.120 [-0.190, -0.051] | -0.097 [-0.167, -0.026] |
| faceverse_neutral | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.011 [-0.082, +0.061] | -0.061 [-0.138, +0.014] | -0.144 [-0.223, -0.068] | -0.106 [-0.187, -0.026] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.011 [-0.086, +0.066] | -0.047 [-0.122, +0.033] | -0.099 [-0.183, -0.008] | -0.070 [-0.148, +0.014] |
| faceverse_neutral | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.006 [-0.080, +0.067] | -0.022 [-0.101, +0.053] | -0.103 [-0.191, -0.019] | -0.045 [-0.129, +0.030] |

GT SR:

| dominio | passo | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- |
| hifi3d | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.073 [-0.127, -0.017] | -0.098 [-0.162, -0.035] | -0.201 [-0.280, -0.136] | -0.252 [-0.325, -0.186] |
| hifi3d | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.170 [+0.115, +0.232] | +0.182 [+0.126, +0.245] | - | - |
| hifi3d | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.071 [-0.106, -0.036] | -0.092 [-0.122, -0.060] | - | - |
| hifi3d | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.365 [-0.428, -0.307] | -0.366 [-0.425, -0.305] | - | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.231 [+0.168, +0.297] | +0.162 [+0.114, +0.217] | -0.064 [-0.124, -0.001] | -0.026 [-0.078, +0.029] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.227 [+0.167, +0.292] | +0.161 [+0.108, +0.216] | -0.101 [-0.164, -0.038] | -0.037 [-0.090, +0.017] |
| hifi3d | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.210 [+0.146, +0.269] | +0.132 [+0.088, +0.176] | -0.009 [-0.070, +0.049] | -0.001 [-0.050, +0.050] |
| hifi3d | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.175 [+0.110, +0.240] | +0.116 [+0.059, +0.166] | -0.032 [-0.098, +0.031] | -0.014 [-0.076, +0.044] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.159 [+0.088, +0.236] | +0.070 [+0.012, +0.135] | - | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.156 [+0.082, +0.228] | +0.068 [+0.005, +0.131] | - | - |
| hifi3d | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.139 [+0.068, +0.205] | +0.040 [-0.016, +0.095] | - | - |
| hifi3d | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.103 [+0.027, +0.177] | +0.023 [-0.039, +0.086] | - | - |
| facescape | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.037 [+0.008, +0.066] | +0.034 [+0.021, +0.047] | +0.007 [-0.012, +0.025] | -0.052 [-0.067, -0.038] |
| facescape | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.012 [-0.002, +0.027] | +0.020 [+0.005, +0.035] | - | - |
| facescape | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.005 [-0.040, +0.033] | +0.012 [-0.014, +0.041] | - | - |
| facescape | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.213 [-0.283, -0.136] | -0.144 [-0.214, -0.073] | - | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.562 [+0.512, +0.612] | +0.478 [+0.427, +0.527] | +0.189 [+0.121, +0.257] | +0.196 [+0.138, +0.253] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.538 [+0.477, +0.596] | +0.452 [+0.391, +0.507] | +0.152 [+0.073, +0.223] | +0.164 [+0.097, +0.225] |
| facescape | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.247 [+0.184, +0.310] | +0.121 [+0.078, +0.161] | +0.028 [-0.041, +0.096] | +0.082 [+0.029, +0.135] |
| facescape | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.205 [+0.137, +0.277] | +0.094 [+0.051, +0.138] | +0.053 [-0.016, +0.119] | +0.085 [+0.025, +0.145] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.556 [+0.494, +0.617] | +0.490 [+0.428, +0.551] | - | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.533 [+0.467, +0.599] | +0.464 [+0.395, +0.530] | - | - |
| facescape | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | +0.242 [+0.169, +0.313] | +0.133 [+0.082, +0.185] | - | - |
| facescape | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | +0.200 [+0.122, +0.283] | +0.107 [+0.052, +0.161] | - | - |
| faceverse | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.037 [-0.077, +0.005] | -0.003 [-0.042, +0.038] | +0.000 [-0.039, +0.045] | +0.007 [-0.035, +0.052] |
| faceverse | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.001 [-0.040, +0.039] | -0.002 [-0.041, +0.036] | - | - |
| faceverse | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.030 [-0.015, +0.073] | +0.048 [-0.003, +0.094] | - | - |
| faceverse | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.136 [-0.218, -0.045] | -0.107 [-0.202, -0.014] | - | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.028 [-0.031, +0.091] | +0.034 [-0.025, +0.086] | -0.029 [-0.103, +0.047] | +0.014 [-0.043, +0.069] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | -0.010 [-0.080, +0.062] | -0.002 [-0.074, +0.067] | -0.069 [-0.152, +0.022] | -0.019 [-0.091, +0.051] |
| faceverse | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | -0.036 [-0.106, +0.042] | -0.016 [-0.085, +0.053] | -0.074 [-0.175, +0.034] | -0.020 [-0.090, +0.054] |
| faceverse | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | -0.031 [-0.099, +0.038] | +0.003 [-0.063, +0.067] | -0.082 [-0.173, +0.006] | -0.004 [-0.073, +0.065] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.058 [-0.006, +0.122] | +0.081 [+0.012, +0.145] | - | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.020 [-0.054, +0.097] | +0.046 [-0.031, +0.122] | - | - |
| faceverse | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | -0.006 [-0.076, +0.063] | +0.032 [-0.037, +0.102] | - | - |
| faceverse | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | -0.001 [-0.077, +0.069] | +0.051 [-0.027, +0.121] | - | - |
| faceverse_neutral | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.057 [-0.100, -0.017] | -0.019 [-0.059, +0.021] | -0.013 [-0.057, +0.030] | -0.012 [-0.055, +0.032] |
| faceverse_neutral | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.018 [-0.018, +0.055] | +0.018 [-0.018, +0.055] | - | - |
| faceverse_neutral | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.034 [-0.024, +0.092] | +0.048 [-0.008, +0.103] | - | - |
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
| hifi3d | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.002 [-0.069, +0.074] | +0.094 [+0.020, +0.173] |
| hifi3d | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.079 [-0.189, +0.041] | +0.051 [-0.067, +0.166] |
| facescape | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | -0.031 [-0.071, +0.007] | +0.034 [+0.003, +0.068] |
| facescape | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.254 [-0.324, -0.182] | -0.085 [-0.156, -0.014] |
| faceverse | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.046 [-0.014, +0.107] | +0.103 [+0.038, +0.169] |
| faceverse | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.101 [-0.207, -0.001] | -0.032 [-0.144, +0.080] |
| faceverse_neutral | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.022 [-0.045, +0.084] | +0.090 [+0.014, +0.162] |
| faceverse_neutral | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.133 [-0.239, -0.029] | -0.088 [-0.205, +0.029] |

GT SR:

| dominio | passo | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- | --- |
| hifi3d | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | -0.150 [-0.217, -0.083] | -0.141 [-0.213, -0.068] | - | - |
| hifi3d | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.370 [-0.458, -0.278] | -0.401 [-0.498, -0.301] | - | - |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.304 [+0.239, +0.375] | +0.260 [+0.190, +0.333] | +0.137 [+0.052, +0.226] | +0.226 [+0.150, +0.306] |
| hifi3d | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.300 [+0.237, +0.370] | +0.259 [+0.193, +0.329] | +0.100 [+0.023, +0.185] | +0.215 [+0.143, +0.291] |
| hifi3d | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.283 [+0.217, +0.348] | +0.231 [+0.160, +0.300] | +0.191 [+0.111, +0.277] | +0.251 [+0.176, +0.325] |
| hifi3d | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.247 [+0.175, +0.319] | +0.214 [+0.138, +0.290] | +0.169 [+0.081, +0.262] | +0.238 [+0.159, +0.323] |
| facescape | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | -0.042 [-0.076, -0.004] | +0.026 [-0.001, +0.057] | - | - |
| facescape | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.300 [-0.364, -0.233] | -0.181 [-0.239, -0.116] | - | - |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.525 [+0.473, +0.569] | +0.444 [+0.390, +0.494] | +0.183 [+0.116, +0.247] | +0.247 [+0.189, +0.305] |
| facescape | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.502 [+0.438, +0.553] | +0.418 [+0.355, +0.473] | +0.145 [+0.063, +0.213] | +0.216 [+0.145, +0.279] |
| facescape | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.210 [+0.143, +0.276] | +0.087 [+0.041, +0.130] | +0.021 [-0.050, +0.085] | +0.134 [+0.077, +0.186] |
| facescape | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.168 [+0.096, +0.240] | +0.060 [+0.013, +0.104] | +0.046 [-0.022, +0.112] | +0.137 [+0.075, +0.195] |
| faceverse | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | +0.003 [-0.053, +0.057] | +0.055 [-0.002, +0.113] | - | - |
| faceverse | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | -0.163 [-0.264, -0.061] | -0.107 [-0.217, +0.005] | - | - |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | +0.065 [+0.002, +0.126] | +0.036 [-0.030, +0.096] | -0.029 [-0.114, +0.056] | +0.007 [-0.059, +0.070] |
| faceverse | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | +0.027 [-0.047, +0.104] | +0.001 [-0.073, +0.078] | -0.070 [-0.165, +0.030] | -0.025 [-0.102, +0.053] |
| faceverse | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | +0.001 [-0.070, +0.073] | -0.014 [-0.082, +0.058] | -0.074 [-0.168, +0.032] | -0.027 [-0.100, +0.049] |
| faceverse | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | +0.006 [-0.070, +0.073] | +0.006 [-0.071, +0.075] | -0.083 [-0.174, +0.008] | -0.011 [-0.092, +0.062] |
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
| FR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |
| FR | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | principale | 2 / 0 / 2 (su 4) | 3 / 0 / 1 (su 4) |
| FR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 0 / 1 / 3 (su 4) | 2 / 1 / 1 (su 4) |
| FR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 0 / 0 / 4 (su 4) | 4 / 0 / 0 (su 4) |
| FR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 0 / 3 / 1 (su 4) | 0 / 1 / 3 (su 4) |
| FR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 3 / 0 (su 3) | 0 / 2 / 1 (su 3) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | principale | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| FR | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | principale | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | principale | 2 / 0 / 2 (su 4) | 1 / 0 / 3 (su 4) |
| FR | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | principale | 2 / 0 / 2 (su 4) | 0 / 0 / 4 (su 4) |
| SR | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | principale | 1 / 2 / 1 (su 4) | 1 / 1 / 2 (su 4) |
| SR | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | principale | 0 / 1 / 3 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | principale | 0 / 1 / 3 (su 4) | 0 / 1 / 3 (su 4) |
| SR | 2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia) | secondario | 0 / 2 / 2 (su 4) | 0 / 1 / 3 (su 4) |
| SR | 3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia) | secondario | 0 / 4 / 0 (su 4) | 0 / 3 / 1 (su 4) |
| SR | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | principale | 0 / 3 / 0 (su 3) | 0 / 3 / 0 (su 3) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | principale | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | principale | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | principale | 0 / 3 / 1 (su 4) | 0 / 4 / 0 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | principale | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | principale | 2 / 0 / 2 (su 4) | 1 / 0 / 3 (su 4) |
| SR | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | principale | 2 / 0 / 2 (su 4) | 1 / 0 / 3 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [GNM (visto) vB, mesh d'identita' SR] | secondario | 0 / 4 / 0 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 4. fit 3DMM vB per mesh (mesh SR) - ICP mm (riferimento con la taglia) [FLAME 2023 Open vB, mesh d'identita' SR] | secondario | 0 / 2 / 2 (su 4) | 0 / 2 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s1234, d_P] | secondario | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |
| SR | 5. nostro (d_P) - ICP mm (riferimento con la taglia) [factorized s2345, d_P] | secondario | 2 / 0 / 2 (su 4) | 2 / 0 / 2 (su 4) |

### Fallimenti del NICP per coppia (crop)

| dominio | GT | insieme | metodo / passo | maschera comune | fallimento = distanza massima |
| --- | --- | --- | --- | --- | --- |
| hifi3d | FR | crop | NICP per coppia in mm | 0.416 [0.350, 0.484] (49306) | 0.412 [0.347, 0.480] (49500) |
| hifi3d | FR | crop | NICP per coppia cs | 0.257 [0.188, 0.333] (49306) | 0.254 [0.187, 0.329] (49500) |
| hifi3d | FR | crop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.044 [-0.075, -0.014] | -0.041 [-0.072, -0.012] |
| hifi3d | FR | crop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.227 [-0.301, -0.161] | -0.225 [-0.298, -0.160] |
| hifi3d | SR | crop | NICP per coppia in mm | 0.417 [0.357, 0.481] (49306) | 0.413 [0.355, 0.478] (49500) |
| hifi3d | SR | crop | NICP per coppia cs | 0.411 [0.363, 0.460] (49306) | 0.407 [0.360, 0.456] (49500) |
| hifi3d | SR | crop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.071 [-0.106, -0.036] | -0.068 [-0.103, -0.033] |
| hifi3d | SR | crop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.365 [-0.428, -0.307] | -0.362 [-0.425, -0.304] |
| facescape | FR | crop | NICP per coppia in mm | 0.324 [0.261, 0.387] (46931) | 0.289 [0.232, 0.345] (49500) |
| facescape | FR | crop | NICP per coppia cs | 0.228 [0.167, 0.291] (46931) | 0.203 [0.147, 0.262] (49500) |
| facescape | FR | crop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.004 [-0.036, +0.031] | +0.018 [-0.011, +0.048] |
| facescape | FR | crop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.175 [-0.245, -0.100] | -0.149 [-0.216, -0.081] |
| facescape | SR | crop | NICP per coppia in mm | 0.348 [0.282, 0.409] (46931) | 0.310 [0.251, 0.364] (49500) |
| facescape | SR | crop | NICP per coppia cs | 0.274 [0.213, 0.331] (46931) | 0.244 [0.189, 0.295] (49500) |
| facescape | SR | crop | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.005 [-0.040, +0.033] | +0.020 [-0.012, +0.054] |
| facescape | SR | crop | 3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs | -0.213 [-0.283, -0.136] | -0.183 [-0.246, -0.112] |

### Protocolli, landmark esatti e rilevati (crop)

HIFI3D:

| GT | insieme | protocollo | righe | landmark esatti (dichiarata) | **lmk_det (rilevatore)** | lmk_s1 | lmk_s2 | lmk_s3 | lmk_s5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FR | crop | now_metrical | 3923 | 0.705 [0.637, 0.757] | 0.697 [0.628, 0.753] | 0.704 [0.635, 0.757] | 0.700 [0.628, 0.752] | 0.695 [0.624, 0.747] | 0.683 [0.612, 0.737] |
| FR | crop | now_original | 3923 | 0.415 [0.317, 0.503] | 0.420 [0.323, 0.507] | 0.414 [0.316, 0.503] | 0.412 [0.316, 0.501] | 0.408 [0.312, 0.496] | 0.402 [0.308, 0.489] |
| FR | crop | realy | 3923 | 0.478 [0.373, 0.569] | 0.416 [0.323, 0.504] | 0.457 [0.353, 0.547] | 0.418 [0.319, 0.505] | 0.362 [0.265, 0.450] | 0.193 [0.110, 0.273] |
| FR | crop | stirling | 3923 | 0.473 [0.368, 0.570] | 0.400 [0.298, 0.487] | 0.456 [0.351, 0.559] | 0.413 [0.318, 0.506] | 0.351 [0.270, 0.429] | 0.216 [0.146, 0.289] |
| FR | all_cross | now_metrical | 11803 | 0.697 [0.628, 0.751] | 0.688 [0.621, 0.743] | 0.697 [0.627, 0.750] | 0.695 [0.626, 0.749] | 0.693 [0.624, 0.747] | 0.687 [0.618, 0.741] |
| FR | all_cross | now_original | 11803 | 0.412 [0.327, 0.497] | 0.415 [0.327, 0.500] | 0.411 [0.325, 0.496] | 0.410 [0.324, 0.494] | 0.408 [0.322, 0.492] | 0.405 [0.320, 0.488] |
| FR | all_cross | realy | 11803 | 0.485 [0.388, 0.573] | 0.418 [0.332, 0.498] | 0.472 [0.378, 0.555] | 0.440 [0.351, 0.520] | 0.376 [0.288, 0.455] | 0.195 [0.131, 0.253] |
| FR | all_cross | stirling | 11803 | 0.463 [0.366, 0.555] | 0.382 [0.292, 0.466] | 0.449 [0.351, 0.537] | 0.408 [0.323, 0.485] | 0.346 [0.273, 0.414] | 0.220 [0.164, 0.276] |
| SR | crop | now_metrical | 3923 | 0.435 [0.340, 0.520] | 0.450 [0.353, 0.534] | 0.433 [0.339, 0.519] | 0.432 [0.338, 0.517] | 0.430 [0.334, 0.513] | 0.423 [0.330, 0.508] |
| SR | crop | now_original | 3923 | 0.615 [0.543, 0.679] | 0.620 [0.546, 0.686] | 0.614 [0.542, 0.679] | 0.611 [0.539, 0.676] | 0.606 [0.535, 0.671] | 0.596 [0.525, 0.660] |
| SR | crop | realy | 3923 | 0.710 [0.629, 0.775] | 0.647 [0.576, 0.706] | 0.683 [0.605, 0.745] | 0.626 [0.548, 0.688] | 0.534 [0.457, 0.601] | 0.312 [0.243, 0.378] |
| SR | crop | stirling | 3923 | 0.654 [0.573, 0.726] | 0.571 [0.490, 0.648] | 0.635 [0.551, 0.707] | 0.573 [0.489, 0.647] | 0.481 [0.398, 0.555] | 0.301 [0.228, 0.375] |
| SR | all_cross | now_metrical | 11803 | 0.433 [0.348, 0.515] | 0.448 [0.360, 0.528] | 0.433 [0.347, 0.514] | 0.432 [0.345, 0.513] | 0.431 [0.343, 0.513] | 0.427 [0.338, 0.509] |
| SR | all_cross | now_original | 11803 | 0.618 [0.554, 0.675] | 0.616 [0.550, 0.673] | 0.617 [0.553, 0.673] | 0.615 [0.551, 0.672] | 0.613 [0.548, 0.670] | 0.608 [0.542, 0.665] |
| SR | all_cross | realy | 11803 | 0.705 [0.632, 0.766] | 0.629 [0.566, 0.683] | 0.681 [0.608, 0.740] | 0.624 [0.552, 0.683] | 0.521 [0.444, 0.588] | 0.296 [0.241, 0.352] |
| SR | all_cross | stirling | 11803 | 0.639 [0.559, 0.706] | 0.542 [0.465, 0.614] | 0.623 [0.546, 0.689] | 0.564 [0.492, 0.626] | 0.476 [0.407, 0.535] | 0.301 [0.240, 0.358] |

FaceScape (dev):

| GT | insieme | protocollo | righe | landmark esatti (dichiarata) | **lmk_det (rilevatore)** | lmk_s1 | lmk_s2 | lmk_s3 | lmk_s5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FR | crop | now_metrical | 3763 | 0.665 [0.585, 0.728] | 0.665 [0.585, 0.730] | 0.662 [0.579, 0.726] | 0.656 [0.574, 0.723] | 0.649 [0.564, 0.717] | 0.635 [0.552, 0.703] |
| FR | crop | now_original | 3763 | 0.594 [0.498, 0.680] | 0.593 [0.496, 0.681] | 0.591 [0.496, 0.678] | 0.586 [0.491, 0.673] | 0.579 [0.481, 0.666] | 0.565 [0.466, 0.651] |
| FR | crop | realy | 3763 | 0.633 [0.537, 0.713] | 0.512 [0.407, 0.606] | 0.623 [0.529, 0.703] | 0.559 [0.463, 0.641] | 0.442 [0.344, 0.531] | 0.262 [0.177, 0.351] |
| FR | crop | stirling | 3763 | 0.573 [0.461, 0.674] | 0.521 [0.407, 0.621] | 0.573 [0.464, 0.672] | 0.530 [0.420, 0.629] | 0.457 [0.352, 0.557] | 0.302 [0.206, 0.403] |
| FR | all_cross | now_metrical | 10851 | 0.671 [0.601, 0.727] | 0.671 [0.603, 0.729] | 0.669 [0.599, 0.726] | 0.667 [0.599, 0.724] | 0.665 [0.595, 0.723] | 0.659 [0.591, 0.718] |
| FR | all_cross | now_original | 10851 | 0.612 [0.525, 0.691] | 0.607 [0.520, 0.686] | 0.610 [0.524, 0.691] | 0.608 [0.522, 0.689] | 0.606 [0.520, 0.687] | 0.600 [0.513, 0.679] |
| FR | all_cross | realy | 10851 | 0.620 [0.534, 0.696] | 0.478 [0.386, 0.566] | 0.599 [0.510, 0.675] | 0.523 [0.432, 0.602] | 0.403 [0.315, 0.485] | 0.231 [0.163, 0.297] |
| FR | all_cross | stirling | 10851 | 0.575 [0.476, 0.668] | 0.488 [0.386, 0.588] | 0.573 [0.476, 0.663] | 0.526 [0.433, 0.613] | 0.446 [0.358, 0.532] | 0.273 [0.195, 0.353] |
| SR | crop | now_metrical | 3763 | 0.686 [0.607, 0.748] | 0.690 [0.612, 0.752] | 0.682 [0.602, 0.746] | 0.677 [0.596, 0.742] | 0.671 [0.590, 0.736] | 0.659 [0.579, 0.725] |
| SR | crop | now_original | 3763 | 0.694 [0.617, 0.756] | 0.696 [0.621, 0.758] | 0.691 [0.613, 0.753] | 0.685 [0.607, 0.749] | 0.678 [0.598, 0.740] | 0.663 [0.585, 0.726] |
| SR | crop | realy | 3763 | 0.730 [0.658, 0.789] | 0.574 [0.491, 0.652] | 0.708 [0.629, 0.770] | 0.626 [0.537, 0.700] | 0.492 [0.399, 0.577] | 0.278 [0.187, 0.363] |
| SR | crop | stirling | 3763 | 0.673 [0.586, 0.743] | 0.616 [0.528, 0.692] | 0.667 [0.584, 0.737] | 0.614 [0.529, 0.691] | 0.527 [0.434, 0.611] | 0.345 [0.256, 0.440] |
| SR | all_cross | now_metrical | 10851 | 0.699 [0.641, 0.752] | 0.700 [0.640, 0.752] | 0.697 [0.639, 0.751] | 0.696 [0.637, 0.750] | 0.694 [0.636, 0.748] | 0.688 [0.629, 0.743] |
| SR | all_cross | now_original | 10851 | 0.715 [0.657, 0.766] | 0.713 [0.654, 0.765] | 0.714 [0.656, 0.765] | 0.711 [0.654, 0.762] | 0.708 [0.651, 0.760] | 0.702 [0.644, 0.753] |
| SR | all_cross | realy | 10851 | 0.719 [0.653, 0.777] | 0.535 [0.458, 0.611] | 0.691 [0.618, 0.751] | 0.598 [0.519, 0.666] | 0.463 [0.384, 0.539] | 0.267 [0.204, 0.331] |
| SR | all_cross | stirling | 10851 | 0.676 [0.603, 0.740] | 0.564 [0.476, 0.644] | 0.669 [0.599, 0.733] | 0.612 [0.534, 0.679] | 0.518 [0.435, 0.590] | 0.322 [0.240, 0.394] |

### Varianti non dichiarate (crop)

GT FR:

| metodo | hifi3d crop | hifi3d all_cross | facescape crop | facescape all_cross | faceverse crop | faceverse all_cross | faceverse_neutral crop | faceverse_neutral all_cross |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), mm (FR) | 0.209 [0.145, 0.279] | 0.164 [0.114, 0.215] | 0.076 [0.025, 0.133] | 0.083 [0.050, 0.120] | 0.053 [0.012, 0.088] | 0.032 [0.005, 0.055] | 0.051 [0.011, 0.092] | 0.039 [0.010, 0.067] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.210 [0.143, 0.278] | 0.153 [0.102, 0.203] | 0.174 [0.115, 0.237] | 0.121 [0.078, 0.169] | 0.075 [0.037, 0.112] | 0.046 [0.020, 0.071] | 0.090 [0.047, 0.134] | 0.056 [0.027, 0.087] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.144 [0.087, 0.207] | 0.080 [0.049, 0.115] | 0.111 [0.064, 0.161] | 0.056 [0.033, 0.083] | 0.065 [0.035, 0.092] | 0.035 [0.017, 0.052] | 0.082 [0.046, 0.118] | 0.047 [0.028, 0.066] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.131 [0.081, 0.191] | 0.096 [0.063, 0.133] | 0.145 [0.087, 0.207] | 0.085 [0.052, 0.120] | 0.094 [0.056, 0.129] | 0.059 [0.032, 0.082] | 0.118 [0.070, 0.165] | 0.080 [0.048, 0.111] |
| DreamSim ensemble su render [profondita'] (non dichiarata), mm (FR) | 0.245 [0.173, 0.320] | 0.189 [0.139, 0.242] | 0.179 [0.125, 0.240] | 0.153 [0.118, 0.193] | 0.068 [0.026, 0.106] | 0.050 [0.018, 0.079] | 0.081 [0.025, 0.139] | 0.077 [0.026, 0.129] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.234 [0.167, 0.304] | 0.172 [0.121, 0.223] | 0.202 [0.149, 0.262] | 0.139 [0.104, 0.179] | 0.104 [0.065, 0.138] | 0.065 [0.039, 0.089] | 0.145 [0.100, 0.192] | 0.100 [0.066, 0.131] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), mm (FR) | 0.181 [0.138, 0.221] | 0.341 [0.271, 0.402] | 0.079 [0.039, 0.121] | 0.081 [0.064, 0.096] | 0.071 [0.043, 0.100] | 0.092 [0.057, 0.129] | 0.085 [0.058, 0.111] | 0.113 [0.074, 0.150] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), mm (FR) | 0.157 [0.112, 0.200] | 0.355 [0.282, 0.419] | 0.069 [0.034, 0.109] | 0.090 [0.075, 0.103] | 0.084 [0.056, 0.112] | 0.152 [0.106, 0.194] | 0.093 [0.066, 0.118] | 0.178 [0.127, 0.226] |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), mm (FR) | 0.335 [0.288, 0.375] | 0.317 [0.274, 0.354] | 0.120 [0.078, 0.170] | 0.100 [0.077, 0.124] | 0.131 [0.090, 0.171] | 0.101 [0.072, 0.127] | 0.180 [0.135, 0.222] | 0.144 [0.109, 0.175] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), mm (FR) | 0.384 [0.323, 0.437] | 0.385 [0.322, 0.439] | 0.121 [0.078, 0.171] | 0.119 [0.093, 0.145] | 0.147 [0.101, 0.190] | 0.117 [0.084, 0.147] | 0.193 [0.143, 0.238] | 0.161 [0.122, 0.197] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, mm (FR) | 0.018 [-0.023, 0.062] | 0.046 [0.018, 0.073] | 0.068 [0.007, 0.132] | 0.048 [0.028, 0.070] | 0.109 [0.045, 0.171] | 0.093 [0.050, 0.137] | 0.097 [0.029, 0.159] | 0.078 [0.031, 0.125] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT FR (FR) | -0.030 [-0.113, 0.062] | 0.001 [-0.052, 0.061] | 0.040 [-0.022, 0.105] | 0.049 [-0.017, 0.127] | 0.091 [0.033, 0.146] | 0.076 [0.028, 0.124] | 0.080 [0.016, 0.143] | 0.053 [0.002, 0.107] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, mm (FR) | 0.079 [0.045, 0.112] | 0.089 [0.068, 0.110] | 0.055 [0.006, 0.111] | 0.073 [0.055, 0.093] | 0.047 [-0.006, 0.103] | 0.061 [0.027, 0.096] | 0.036 [-0.014, 0.086] | 0.068 [0.032, 0.108] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT FR (FR) | 0.025 [-0.047, 0.091] | 0.054 [0.005, 0.098] | -0.004 [-0.051, 0.045] | 0.039 [0.005, 0.073] | 0.090 [0.020, 0.159] | 0.087 [0.027, 0.141] | 0.066 [-0.012, 0.145] | 0.081 [0.013, 0.144] |

GT SR:

| metodo | hifi3d crop | hifi3d all_cross | facescape crop | facescape all_cross | faceverse crop | faceverse all_cross | faceverse_neutral crop | faceverse_neutral all_cross |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.193 [0.137, 0.256] | 0.154 [0.113, 0.204] | 0.135 [0.082, 0.193] | 0.109 [0.073, 0.150] | 0.056 [0.014, 0.091] | 0.031 [0.002, 0.055] | 0.073 [0.032, 0.114] | 0.048 [0.020, 0.078] |
| CLIP ViT-B/32 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.236 [0.182, 0.286] | 0.173 [0.134, 0.212] | 0.167 [0.110, 0.232] | 0.112 [0.069, 0.160] | 0.091 [0.051, 0.130] | 0.051 [0.026, 0.075] | 0.111 [0.070, 0.156] | 0.071 [0.041, 0.105] |
| DINOv2 ViT-S/14 su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.127 [0.075, 0.181] | 0.085 [0.055, 0.118] | 0.119 [0.077, 0.168] | 0.060 [0.038, 0.086] | 0.073 [0.044, 0.100] | 0.035 [0.017, 0.052] | 0.097 [0.063, 0.130] | 0.049 [0.032, 0.066] |
| DINOv2 ViT-S/14 su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.113 [0.074, 0.153] | 0.105 [0.079, 0.133] | 0.151 [0.097, 0.209] | 0.087 [0.056, 0.123] | 0.109 [0.070, 0.145] | 0.063 [0.036, 0.088] | 0.139 [0.093, 0.185] | 0.089 [0.058, 0.118] |
| DreamSim ensemble su render [profondita'] (non dichiarata), a CS_ref (SR) | 0.286 [0.219, 0.357] | 0.219 [0.173, 0.271] | 0.206 [0.151, 0.269] | 0.162 [0.124, 0.204] | 0.069 [0.029, 0.107] | 0.054 [0.023, 0.083] | 0.088 [0.030, 0.147] | 0.087 [0.035, 0.139] |
| DreamSim ensemble su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.265 [0.212, 0.319] | 0.187 [0.147, 0.228] | 0.235 [0.180, 0.293] | 0.147 [0.111, 0.185] | 0.109 [0.069, 0.143] | 0.066 [0.040, 0.089] | 0.168 [0.121, 0.212] | 0.113 [0.080, 0.145] |
| Gromov-Wasserstein (POT) [euclid_256_cg] (non dichiarata), a CS_ref (SR) | 0.208 [0.162, 0.254] | 0.182 [0.144, 0.218] | 0.201 [0.158, 0.241] | 0.157 [0.127, 0.186] | 0.099 [0.064, 0.130] | 0.089 [0.059, 0.117] | 0.121 [0.085, 0.153] | 0.112 [0.077, 0.146] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] (non dichiarata), a CS_ref (SR) | 0.259 [0.206, 0.314] | 0.210 [0.169, 0.252] | 0.268 [0.217, 0.318] | 0.197 [0.164, 0.229] | 0.194 [0.140, 0.239] | 0.185 [0.135, 0.228] | 0.212 [0.155, 0.267] | 0.219 [0.166, 0.269] |
| LPIPS alex su render [ombreggiato per faccia] (non dichiarata), a CS_ref (SR) | 0.249 [0.213, 0.287] | 0.157 [0.136, 0.179] | 0.244 [0.185, 0.302] | 0.143 [0.111, 0.175] | 0.133 [0.090, 0.174] | 0.095 [0.068, 0.121] | 0.186 [0.142, 0.228] | 0.135 [0.104, 0.165] |
| LPIPS alex su render [ombreggiato con normali per vertice] (non dichiarata), a CS_ref (SR) | 0.253 [0.216, 0.291] | 0.171 [0.148, 0.195] | 0.259 [0.197, 0.317] | 0.157 [0.123, 0.191] | 0.160 [0.114, 0.208] | 0.113 [0.082, 0.143] | 0.202 [0.155, 0.247] | 0.147 [0.112, 0.181] |
| ULRSSM [faces_wks] (non dichiarata) per coppia, a CS_ref (SR) | 0.025 [-0.015, 0.065] | 0.058 [0.031, 0.083] | 0.065 [0.012, 0.126] | 0.051 [0.033, 0.073] | 0.121 [0.061, 0.182] | 0.103 [0.060, 0.147] | 0.113 [0.044, 0.172] | 0.091 [0.045, 0.135] |
| ULRSSM [faces_wks] (non dichiarata) via template, GT SR (d_P) (SR) | -0.003 [-0.086, 0.086] | 0.017 [-0.033, 0.067] | 0.024 [-0.029, 0.075] | 0.028 [-0.027, 0.080] | 0.104 [0.045, 0.159] | 0.089 [0.040, 0.139] | 0.094 [0.032, 0.155] | 0.065 [0.014, 0.117] |
| ULRSSM [zeroshot_faust] (non dichiarata) per coppia, a CS_ref (SR) | 0.099 [0.068, 0.128] | 0.103 [0.082, 0.124] | 0.073 [0.022, 0.129] | 0.078 [0.059, 0.099] | 0.064 [0.006, 0.124] | 0.070 [0.038, 0.102] | 0.048 [0.002, 0.097] | 0.075 [0.041, 0.111] |
| ULRSSM [zeroshot_faust] (non dichiarata) via template, GT SR (d_P) (SR) | 0.055 [0.005, 0.111] | 0.074 [0.045, 0.102] | -0.009 [-0.052, 0.032] | 0.026 [-0.010, 0.066] | 0.100 [0.034, 0.168] | 0.095 [0.042, 0.144] | 0.088 [0.013, 0.161] | 0.107 [0.044, 0.165] |

### HIFI3D (crop): Spearman per insieme di righe (seme 990708)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | crop (49306) | all_cross (147530) | stessa_topologia crop (4950) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.663 [0.589, 0.721] | 0.709 [0.634, 0.767] | 0.784 [0.721, 0.837] | 0.762 [0.688, 0.819] |
| **factorized s1234, d_F non cal.** | 0.563 [0.485, 0.630] | 0.584 [0.512, 0.647] | 0.651 [0.584, 0.717] | 0.648 [0.580, 0.712] |
| factorized s1234, d_P | 0.376 [0.281, 0.457] | 0.383 [0.291, 0.464] | 0.439 [0.341, 0.529] | 0.429 [0.337, 0.513] |
| **factorized s2345, d_F cal.*** | 0.552 [0.482, 0.614] | 0.661 [0.587, 0.722] | 0.755 [0.686, 0.817] | 0.746 [0.672, 0.807] |
| **factorized s2345, d_F non cal.** | 0.516 [0.436, 0.584] | 0.561 [0.486, 0.627] | 0.634 [0.559, 0.705] | 0.637 [0.566, 0.702] |
| factorized s2345, d_P | 0.347 [0.251, 0.436] | 0.370 [0.276, 0.457] | 0.415 [0.309, 0.515] | 0.418 [0.314, 0.513] |
| **ctrlfr s1234** | 0.236 [0.186, 0.281] | 0.515 [0.444, 0.577] | 0.788 [0.721, 0.845] | 0.780 [0.710, 0.839] |
| **ctrlfr s2345** | 0.279 [0.223, 0.329] | 0.529 [0.455, 0.593] | 0.789 [0.723, 0.842] | 0.768 [0.694, 0.829] |
| **C3M e123, d_F cal.*** | 0.506 [0.427, 0.574] | 0.641 [0.563, 0.703] | 0.763 [0.692, 0.820] | 0.766 [0.689, 0.823] |
| **C3M e123, d_F non cal.** | 0.413 [0.330, 0.491] | 0.418 [0.343, 0.488] | 0.521 [0.440, 0.607] | 0.538 [0.457, 0.614] |
| C3M e123, d_P | 0.316 [0.231, 0.397] | 0.313 [0.229, 0.390] | 0.405 [0.305, 0.503] | 0.406 [0.311, 0.492] |
| **C3M e205, d_F cal.*** | 0.472 [0.396, 0.538] | 0.629 [0.549, 0.692] | 0.759 [0.685, 0.816] | 0.757 [0.676, 0.815] |
| **C3M e205, d_F non cal.** | 0.405 [0.321, 0.484] | 0.413 [0.334, 0.488] | 0.500 [0.412, 0.585] | 0.519 [0.434, 0.597] |
| C3M e205, d_P | 0.329 [0.240, 0.410] | 0.325 [0.241, 0.406] | 0.412 [0.313, 0.507] | 0.411 [0.315, 0.502] |
| **ICP + Chamfer in mm (per coppia)** | 0.417 [0.361, 0.474] | 0.556 [0.491, 0.614] | 0.728 [0.662, 0.783] | 0.647 [0.580, 0.705] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.213 [0.155, 0.273] | 0.264 [0.198, 0.326] | 0.421 [0.333, 0.510] | 0.390 [0.304, 0.473] |
| **NICP per coppia in mm** | 0.416 [0.350, 0.484] | 0.462 [0.392, 0.532] | - | - |
| **NICP per coppia cs** | 0.257 [0.188, 0.333] | 0.325 [0.253, 0.410] | - | - |
| **NICP su template in mm** | 0.337 [0.241, 0.422] | 0.513 [0.409, 0.597] | 0.660 [0.560, 0.740] | 0.623 [0.516, 0.705] |
| **NICP su template cs** | 0.029 [-0.003, 0.064] | 0.095 [0.052, 0.138] | 0.245 [0.148, 0.340] | 0.203 [0.112, 0.287] |
| **varifold in mm (massa unitaria)** | 0.279 [0.225, 0.328] | 0.380 [0.307, 0.445] | 0.758 [0.688, 0.819] | 0.698 [0.623, 0.761] |
| **GNM (visto) preregistrato, coefficienti** | 0.443 [0.334, 0.534] | 0.550 [0.440, 0.639] | 0.652 [0.547, 0.734] | 0.624 [0.511, 0.713] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.537 [0.426, 0.623] | 0.583 [0.474, 0.667] | 0.663 [0.562, 0.743] | 0.617 [0.510, 0.701] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.262 [0.157, 0.349] | 0.378 [0.274, 0.467] | 0.380 [0.274, 0.471] | 0.456 [0.344, 0.550] |
| **GNM (visto) vA, coefficienti*** | 0.466 [0.356, 0.561] | 0.541 [0.433, 0.630] | 0.589 [0.485, 0.678] | 0.597 [0.487, 0.685] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.544 [0.435, 0.631] | 0.585 [0.476, 0.671] | 0.664 [0.563, 0.744] | 0.617 [0.509, 0.701] |
| GNM (visto) vA, mesh d'identita' SR* | 0.158 [0.072, 0.245] | 0.232 [0.140, 0.322] | 0.251 [0.158, 0.342] | 0.277 [0.175, 0.373] |
| **GNM (visto) vB, coefficienti*** | 0.524 [0.446, 0.600] | 0.539 [0.456, 0.617] | 0.535 [0.460, 0.610] | 0.546 [0.463, 0.623] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.625 [0.531, 0.701] | 0.651 [0.553, 0.724] | 0.717 [0.629, 0.788] | 0.674 [0.581, 0.747] |
| GNM (visto) vB, mesh d'identita' SR* | 0.380 [0.284, 0.470] | 0.395 [0.301, 0.481] | 0.414 [0.315, 0.504] | 0.410 [0.318, 0.498] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.661 [0.573, 0.733] | 0.683 [0.598, 0.755] | 0.737 [0.656, 0.806] | 0.705 [0.620, 0.775] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.401 [0.302, 0.486] | 0.506 [0.396, 0.598] | 0.568 [0.459, 0.668] | 0.578 [0.457, 0.675] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.546 [0.438, 0.631] | 0.607 [0.502, 0.692] | 0.667 [0.561, 0.751] | 0.648 [0.542, 0.733] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.172 [0.103, 0.239] | 0.236 [0.150, 0.325] | 0.242 [0.151, 0.333] | 0.278 [0.177, 0.379] |
| **FLAME 2023 Open vA, coefficienti*** | 0.423 [0.321, 0.511] | 0.524 [0.413, 0.617] | 0.587 [0.477, 0.682] | 0.592 [0.474, 0.688] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.565 [0.457, 0.652] | 0.616 [0.512, 0.700] | 0.669 [0.563, 0.751] | 0.649 [0.542, 0.731] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.179 [0.106, 0.252] | 0.236 [0.150, 0.325] | 0.244 [0.152, 0.337] | 0.274 [0.175, 0.371] |
| **FLAME 2023 Open vB, coefficienti*** | 0.542 [0.460, 0.616] | 0.570 [0.489, 0.646] | 0.555 [0.471, 0.630] | 0.577 [0.497, 0.652] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.639 [0.543, 0.713] | 0.670 [0.579, 0.741] | 0.710 [0.619, 0.785] | 0.693 [0.603, 0.766] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.433 [0.342, 0.516] | 0.448 [0.355, 0.531] | 0.442 [0.355, 0.525] | 0.456 [0.364, 0.539] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.671 [0.582, 0.740] | 0.697 [0.610, 0.766] | 0.726 [0.639, 0.797] | 0.716 [0.630, 0.786] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.140 [0.049, 0.239] | 0.135 [0.045, 0.234] | 0.144 [0.049, 0.245] | 0.114 [0.044, 0.193] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.182 [0.087, 0.284] | 0.171 [0.081, 0.269] | 0.192 [0.090, 0.298] | 0.148 [0.073, 0.227] |
| **ShapeDNA k=50** | 0.009 [-0.020, 0.037] | 0.057 [0.032, 0.083] | 0.200 [0.089, 0.311] | 0.250 [0.163, 0.340] |
| **HKS globale** | -0.008 [-0.020, 0.005] | 0.018 [0.001, 0.037] | 0.059 [-0.016, 0.153] | 0.104 [0.020, 0.195] |
| **WKS globale** | 0.034 [0.008, 0.066] | 0.023 [0.006, 0.041] | 0.164 [0.071, 0.269] | 0.199 [0.110, 0.290] |
| **OpenShape PointBERT** | 0.137 [0.085, 0.191] | 0.119 [0.075, 0.163] | 0.339 [0.225, 0.449] | 0.275 [0.182, 0.366] |
| **Uni3D-g** | 0.168 [0.089, 0.246] | 0.113 [0.056, 0.170] | 0.316 [0.190, 0.433] | 0.276 [0.159, 0.389] |
| **taglia stimata dalla mesh (non oracolo)** | 0.056 [0.026, 0.092] | 0.353 [0.276, 0.427] | 0.640 [0.544, 0.725] | 0.620 [0.518, 0.710] |
| **taglia oracolo (S di FR)** | 0.737 [0.657, 0.802] | 0.737 [0.657, 0.802] | 0.736 [0.658, 0.802] | 0.736 [0.658, 0.802] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.227 [0.158, 0.295] | 0.169 [0.120, 0.220] | 0.427 [0.312, 0.539] | 0.362 [0.257, 0.464] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.165 [0.103, 0.229] | 0.129 [0.081, 0.177] | 0.333 [0.216, 0.455] | 0.288 [0.187, 0.391] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.163 [0.110, 0.217] | 0.126 [0.087, 0.162] | 0.363 [0.249, 0.472] | 0.366 [0.267, 0.460] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.060 [0.012, 0.108] | 0.079 [0.046, 0.112] | 0.235 [0.131, 0.341] | 0.266 [0.175, 0.372] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.246 [0.176, 0.317] | 0.191 [0.139, 0.242] | 0.440 [0.326, 0.556] | 0.432 [0.321, 0.537] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.187 [0.119, 0.262] | 0.149 [0.100, 0.203] | 0.332 [0.221, 0.454] | 0.345 [0.239, 0.459] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.159 [0.115, 0.203] | 0.353 [0.280, 0.419] | - | - |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.161 [0.105, 0.213] | 0.128 [0.085, 0.169] | - | - |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.457 [0.386, 0.514] | 0.544 [0.464, 0.613] | 0.768 [0.694, 0.826] | 0.722 [0.644, 0.788] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.198 [0.138, 0.262] | 0.173 [0.125, 0.222] | 0.385 [0.305, 0.463] | 0.347 [0.268, 0.431] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.703 [0.636, 0.758] | 0.699 [0.631, 0.755] | - | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.411 [0.332, 0.489] | 0.413 [0.335, 0.492] | - | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.499 [0.407, 0.580] | 0.496 [0.403, 0.575] | - | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.474 [0.384, 0.560] | 0.467 [0.377, 0.552] | - | - |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.253 [0.202, 0.300] | 0.484 [0.409, 0.548] | 0.840 [0.783, 0.885] | 0.788 [0.719, 0.844] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.239 [0.166, 0.308] | 0.215 [0.156, 0.272] | 0.467 [0.373, 0.555] | 0.419 [0.320, 0.510] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.289 [0.235, 0.343] | 0.479 [0.404, 0.542] | 0.796 [0.735, 0.849] | 0.757 [0.681, 0.820] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.251 [0.172, 0.331] | 0.220 [0.147, 0.292] | 0.406 [0.312, 0.500] | 0.343 [0.256, 0.429] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | crop (49306) | all_cross (147530) | stessa_topologia crop (4950) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.297 [0.215, 0.378] | 0.303 [0.216, 0.389] | 0.373 [0.279, 0.461] | 0.332 [0.240, 0.421] |
| factorized s1234, d_F non cal. | 0.488 [0.400, 0.568] | 0.503 [0.418, 0.579] | 0.607 [0.526, 0.680] | 0.566 [0.480, 0.644] |
| **factorized s1234, d_P** | 0.550 [0.472, 0.623] | 0.557 [0.485, 0.624] | 0.666 [0.601, 0.727] | 0.631 [0.560, 0.695] |
| factorized s2345, d_F cal.* | 0.255 [0.182, 0.326] | 0.286 [0.199, 0.366] | 0.352 [0.258, 0.445] | 0.323 [0.229, 0.412] |
| factorized s2345, d_F non cal. | 0.444 [0.354, 0.525] | 0.482 [0.394, 0.560] | 0.583 [0.500, 0.661] | 0.551 [0.460, 0.632] |
| **factorized s2345, d_P** | 0.514 [0.435, 0.590] | 0.540 [0.463, 0.613] | 0.643 [0.567, 0.711] | 0.618 [0.540, 0.689] |
| ctrlfr s1234 | 0.163 [0.120, 0.203] | 0.244 [0.174, 0.306] | 0.418 [0.313, 0.507] | 0.358 [0.261, 0.445] |
| ctrlfr s2345 | 0.188 [0.144, 0.230] | 0.258 [0.189, 0.321] | 0.398 [0.299, 0.484] | 0.363 [0.266, 0.451] |
| C3M e123, d_F cal.* | 0.220 [0.149, 0.285] | 0.251 [0.167, 0.330] | 0.341 [0.248, 0.428] | 0.306 [0.210, 0.396] |
| C3M e123, d_F non cal. | 0.420 [0.333, 0.506] | 0.441 [0.357, 0.521] | 0.601 [0.519, 0.677] | 0.576 [0.486, 0.660] |
| **C3M e123, d_P** | 0.462 [0.381, 0.541] | 0.461 [0.383, 0.531] | 0.636 [0.559, 0.706] | 0.607 [0.525, 0.681] |
| C3M e205, d_F cal.* | 0.219 [0.152, 0.283] | 0.250 [0.161, 0.330] | 0.351 [0.257, 0.436] | 0.303 [0.201, 0.392] |
| C3M e205, d_F non cal. | 0.442 [0.351, 0.525] | 0.456 [0.370, 0.536] | 0.616 [0.534, 0.691] | 0.581 [0.493, 0.666] |
| **C3M e205, d_P** | 0.481 [0.399, 0.561] | 0.473 [0.393, 0.546] | 0.645 [0.574, 0.713] | 0.607 [0.528, 0.682] |
| **ICP + Chamfer in mm (per coppia)** | 0.267 [0.209, 0.329] | 0.326 [0.252, 0.403] | 0.475 [0.388, 0.564] | 0.380 [0.298, 0.466] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.339 [0.290, 0.389] | 0.425 [0.378, 0.471] | 0.675 [0.623, 0.727] | 0.631 [0.581, 0.682] |
| **NICP per coppia in mm** | 0.417 [0.357, 0.481] | 0.467 [0.403, 0.535] | - | - |
| **NICP per coppia cs** | 0.411 [0.363, 0.460] | 0.517 [0.470, 0.567] | - | - |
| **NICP su template in mm** | 0.046 [-0.014, 0.111] | 0.066 [-0.019, 0.154] | 0.100 [0.001, 0.197] | 0.086 [-0.014, 0.189] |
| **NICP su template cs** | 0.045 [0.009, 0.085] | 0.151 [0.095, 0.200] | 0.405 [0.303, 0.500] | 0.326 [0.221, 0.418] |
| **varifold in mm (massa unitaria)** | 0.165 [0.122, 0.211] | 0.164 [0.109, 0.218] | 0.380 [0.288, 0.467] | 0.312 [0.226, 0.396] |
| **GNM (visto) preregistrato, coefficienti** | 0.106 [0.020, 0.190] | 0.136 [0.040, 0.229] | 0.187 [0.076, 0.280] | 0.161 [0.051, 0.261] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.080 [-0.005, 0.169] | 0.088 [-0.004, 0.179] | 0.105 [0.002, 0.201] | 0.095 [-0.002, 0.190] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.199 [0.112, 0.289] | 0.264 [0.165, 0.364] | 0.336 [0.230, 0.437] | 0.321 [0.206, 0.431] |
| **GNM (visto) vA, coefficienti*** | 0.148 [0.060, 0.240] | 0.171 [0.078, 0.266] | 0.226 [0.122, 0.320] | 0.197 [0.096, 0.294] |
| GNM (visto) vA, mesh d'identita' FR* | 0.091 [0.005, 0.179] | 0.099 [0.007, 0.191] | 0.119 [0.014, 0.215] | 0.107 [0.010, 0.202] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.213 [0.127, 0.307] | 0.280 [0.188, 0.377] | 0.355 [0.259, 0.453] | 0.335 [0.232, 0.438] |
| **GNM (visto) vB, coefficienti*** | 0.651 [0.587, 0.711] | 0.664 [0.600, 0.724] | 0.679 [0.614, 0.736] | 0.673 [0.609, 0.732] |
| GNM (visto) vB, mesh d'identita' FR* | 0.205 [0.122, 0.291] | 0.212 [0.128, 0.300] | 0.233 [0.141, 0.323] | 0.220 [0.131, 0.309] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.570 [0.490, 0.647] | 0.587 [0.511, 0.661] | 0.612 [0.535, 0.683] | 0.605 [0.531, 0.677] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.332 [0.244, 0.420] | 0.341 [0.252, 0.429] | 0.371 [0.279, 0.462] | 0.352 [0.261, 0.441] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.096 [0.017, 0.168] | 0.137 [0.035, 0.228] | 0.144 [0.039, 0.244] | 0.163 [0.051, 0.271] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.095 [0.010, 0.181] | 0.114 [0.021, 0.211] | 0.112 [0.008, 0.213] | 0.125 [0.027, 0.226] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.199 [0.118, 0.281] | 0.253 [0.157, 0.353] | 0.296 [0.203, 0.392] | 0.299 [0.188, 0.408] |
| **FLAME 2023 Open vA, coefficienti*** | 0.096 [0.016, 0.170] | 0.135 [0.035, 0.228] | 0.141 [0.037, 0.241] | 0.159 [0.048, 0.269] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.094 [0.008, 0.181] | 0.110 [0.016, 0.205] | 0.107 [0.006, 0.205] | 0.119 [0.021, 0.217] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.220 [0.137, 0.302] | 0.268 [0.172, 0.366] | 0.310 [0.216, 0.403] | 0.310 [0.204, 0.418] |
| **FLAME 2023 Open vB, coefficienti*** | 0.551 [0.482, 0.615] | 0.573 [0.503, 0.636] | 0.573 [0.503, 0.638] | 0.582 [0.513, 0.647] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.208 [0.125, 0.294] | 0.223 [0.136, 0.311] | 0.219 [0.128, 0.312] | 0.230 [0.140, 0.319] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.567 [0.482, 0.647] | 0.585 [0.505, 0.662] | 0.575 [0.493, 0.654] | 0.594 [0.514, 0.670] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.320 [0.232, 0.408] | 0.337 [0.248, 0.426] | 0.334 [0.243, 0.428] | 0.345 [0.256, 0.435] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.225 [0.154, 0.301] | 0.221 [0.150, 0.293] | 0.235 [0.162, 0.311] | 0.183 [0.130, 0.238] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.249 [0.174, 0.322] | 0.241 [0.172, 0.308] | 0.264 [0.181, 0.345] | 0.208 [0.150, 0.266] |
| **ShapeDNA k=50** | 0.038 [0.008, 0.067] | 0.088 [0.066, 0.107] | 0.355 [0.251, 0.451] | 0.375 [0.299, 0.443] |
| **HKS globale** | 0.004 [-0.008, 0.016] | 0.039 [0.022, 0.056] | 0.142 [0.049, 0.228] | 0.192 [0.105, 0.278] |
| **WKS globale** | 0.062 [0.036, 0.089] | 0.037 [0.022, 0.054] | 0.268 [0.182, 0.358] | 0.304 [0.226, 0.384] |
| **OpenShape PointBERT** | 0.183 [0.140, 0.226] | 0.165 [0.126, 0.205] | 0.415 [0.325, 0.501] | 0.381 [0.301, 0.460] |
| **Uni3D-g** | 0.241 [0.181, 0.297] | 0.154 [0.110, 0.196] | 0.448 [0.341, 0.552] | 0.388 [0.298, 0.478] |
| **taglia stimata dalla mesh (non oracolo)** | 0.009 [-0.004, 0.022] | 0.017 [-0.038, 0.071] | 0.038 [-0.055, 0.127] | 0.026 [-0.068, 0.116] |
| **taglia oracolo (S di FR)** | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.165] | 0.074 [-0.024, 0.166] | 0.074 [-0.024, 0.166] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.259 [0.198, 0.318] | 0.190 [0.141, 0.236] | 0.465 [0.363, 0.557] | 0.420 [0.329, 0.511] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.233 [0.180, 0.283] | 0.186 [0.144, 0.228] | 0.455 [0.354, 0.548] | 0.429 [0.338, 0.519] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.176 [0.132, 0.221] | 0.141 [0.109, 0.174] | 0.447 [0.360, 0.527] | 0.409 [0.334, 0.484] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.114 [0.064, 0.166] | 0.124 [0.097, 0.153] | 0.418 [0.331, 0.504] | 0.428 [0.352, 0.503] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.284 [0.226, 0.341] | 0.213 [0.169, 0.259] | 0.486 [0.396, 0.580] | 0.484 [0.397, 0.579] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.279 [0.225, 0.334] | 0.215 [0.174, 0.257] | 0.498 [0.413, 0.587] | 0.499 [0.412, 0.590] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.059 [0.029, 0.089] | 0.069 [0.016, 0.123] | - | - |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.254 [0.202, 0.307] | 0.192 [0.150, 0.235] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | 0.213 [0.146, 0.283] | 0.200 [0.129, 0.271] | 0.314 [0.214, 0.404] | 0.278 [0.187, 0.368] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.322 [0.261, 0.391] | 0.265 [0.215, 0.319] | 0.574 [0.498, 0.647] | 0.520 [0.451, 0.589] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.437 [0.354, 0.516] | 0.432 [0.351, 0.512] | - | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.606 [0.541, 0.662] | 0.614 [0.549, 0.670] | - | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.721 [0.653, 0.776] | 0.716 [0.647, 0.771] | - | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.645 [0.567, 0.711] | 0.635 [0.556, 0.700] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.152 [0.109, 0.195] | 0.199 [0.134, 0.261] | 0.390 [0.297, 0.479] | 0.327 [0.232, 0.422] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.383 [0.315, 0.455] | 0.345 [0.289, 0.403] | 0.760 [0.707, 0.811] | 0.680 [0.613, 0.742] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.196 [0.151, 0.240] | 0.245 [0.188, 0.299] | 0.426 [0.328, 0.521] | 0.365 [0.279, 0.450] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.464 [0.388, 0.537] | 0.421 [0.356, 0.483] | 0.661 [0.586, 0.732] | 0.609 [0.539, 0.678] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.245 [+0.199, +0.294] | +0.153 [+0.110, +0.200] | +0.056 [+0.009, +0.107] | +0.115 [+0.070, +0.166] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.450 [+0.364, +0.530] | +0.445 [+0.356, +0.533] | +0.363 [+0.256, +0.472] | +0.372 [+0.269, +0.477] |
| NICP per coppia in mm | +0.247 [+0.168, +0.332] | +0.247 [+0.167, +0.334] | - | - |
| NICP per coppia cs | +0.406 [+0.306, +0.503] | +0.384 [+0.279, +0.488] | - | - |
| NICP su template in mm | +0.325 [+0.270, +0.382] | +0.195 [+0.148, +0.247] | +0.125 [+0.073, +0.183] | +0.139 [+0.090, +0.192] |
| NICP su template cs | +0.633 [+0.560, +0.693] | +0.614 [+0.541, +0.686] | +0.539 [+0.434, +0.642] | +0.559 [+0.462, +0.659] |
| varifold in mm (massa unitaria) | +0.384 [+0.331, +0.428] | +0.328 [+0.288, +0.369] | +0.027 [-0.012, +0.066] | +0.064 [+0.031, +0.102] |
| GNM (visto) preregistrato, coefficienti | +0.220 [+0.161, +0.280] | +0.159 [+0.111, +0.208] | +0.133 [+0.075, +0.199] | +0.138 [+0.088, +0.195] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.126 [+0.070, +0.184] | +0.125 [+0.076, +0.176] | +0.122 [+0.070, +0.182] | +0.145 [+0.093, +0.199] |
| GNM (visto) vA, coefficienti | +0.196 [+0.131, +0.262] | +0.167 [+0.114, +0.223] | +0.195 [+0.130, +0.273] | +0.165 [+0.111, +0.226] |
| GNM (visto) vA, mesh d'identita' FR | +0.119 [+0.064, +0.177] | +0.124 [+0.074, +0.174] | +0.121 [+0.071, +0.178] | +0.145 [+0.095, +0.200] |
| GNM (visto) vB, coefficienti | +0.139 [+0.044, +0.235] | +0.169 [+0.067, +0.269] | +0.249 [+0.150, +0.346] | +0.216 [+0.114, +0.318] |
| GNM (visto) vB, mesh d'identita' FR | +0.038 [-0.004, +0.080] | +0.058 [+0.018, +0.096] | +0.068 [+0.025, +0.111] | +0.088 [+0.046, +0.131] |
| GNM (visto) vB, composizione S_B + k d_P | +0.002 [-0.040, +0.044] | +0.026 [-0.012, +0.064] | +0.047 [+0.007, +0.088] | +0.057 [+0.016, +0.101] |
| FLAME 2023 Open preregistrato, coefficienti | +0.262 [+0.210, +0.320] | +0.202 [+0.145, +0.265] | +0.216 [+0.148, +0.294] | +0.184 [+0.119, +0.260] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.117 [+0.069, +0.173] | +0.101 [+0.057, +0.150] | +0.118 [+0.064, +0.179] | +0.114 [+0.067, +0.167] |
| FLAME 2023 Open vA, coefficienti | +0.239 [+0.189, +0.296] | +0.185 [+0.131, +0.247] | +0.198 [+0.131, +0.275] | +0.170 [+0.110, +0.244] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.097 [+0.050, +0.152] | +0.093 [+0.050, +0.141] | +0.116 [+0.063, +0.177] | +0.113 [+0.067, +0.164] |
| FLAME 2023 Open vB, coefficienti | +0.120 [+0.039, +0.209] | +0.139 [+0.051, +0.230] | +0.230 [+0.145, +0.318] | +0.185 [+0.096, +0.273] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.024 [-0.018, +0.065] | +0.039 [+0.000, +0.074] | +0.075 [+0.030, +0.120] | +0.069 [+0.029, +0.107] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.008 [-0.048, +0.032] | +0.011 [-0.026, +0.048] | +0.058 [+0.017, +0.103] | +0.046 [+0.007, +0.084] |
| ArcFace-render, normal map 3 viste (G0) | +0.523 [+0.403, +0.626] | +0.574 [+0.455, +0.679] | +0.641 [+0.524, +0.746] | +0.648 [+0.549, +0.739] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.480 [+0.358, +0.588] | +0.537 [+0.420, +0.640] | +0.593 [+0.472, +0.710] | +0.614 [+0.509, +0.709] |
| ShapeDNA k=50 | +0.654 [+0.574, +0.722] | +0.652 [+0.578, +0.715] | +0.585 [+0.459, +0.700] | +0.512 [+0.399, +0.619] |
| HKS globale | +0.670 [+0.594, +0.733] | +0.691 [+0.616, +0.752] | +0.725 [+0.627, +0.810] | +0.658 [+0.555, +0.758] |
| WKS globale | +0.629 [+0.556, +0.688] | +0.686 [+0.612, +0.746] | +0.620 [+0.498, +0.730] | +0.563 [+0.460, +0.670] |
| OpenShape PointBERT | +0.526 [+0.446, +0.601] | +0.590 [+0.509, +0.662] | +0.446 [+0.333, +0.567] | +0.487 [+0.383, +0.597] |
| Uni3D-g | +0.494 [+0.399, +0.585] | +0.596 [+0.512, +0.669] | +0.468 [+0.337, +0.598] | +0.486 [+0.364, +0.611] |
| taglia stimata dalla mesh (non oracolo) | +0.606 [+0.551, +0.651] | +0.356 [+0.311, +0.399] | +0.144 [+0.084, +0.206] | +0.142 [+0.085, +0.201] |
| taglia oracolo (S di FR) | -0.074 [-0.121, -0.021] | -0.028 [-0.072, +0.021] | +0.048 [+0.005, +0.099] | +0.025 [-0.021, +0.075] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.436 [+0.347, +0.517] | +0.539 [+0.464, +0.607] | +0.357 [+0.243, +0.470] | +0.400 [+0.291, +0.506] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.500 [+0.418, +0.585] | +0.583 [+0.502, +0.655] | +0.421 [+0.292, +0.557] | +0.396 [+0.272, +0.514] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.417 [+0.333, +0.496] | +0.518 [+0.439, +0.583] | +0.344 [+0.227, +0.456] | +0.330 [+0.221, +0.442] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.503 [+0.458, +0.543] | +0.356 [+0.316, +0.397] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.206 [+0.159, +0.252] | +0.165 [+0.122, +0.209] | +0.016 [-0.024, +0.058] | +0.040 [-0.002, +0.080] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.040 [-0.076, -0.006] | +0.010 [-0.023, +0.046] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.252 [+0.149, +0.351] | +0.295 [+0.190, +0.397] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.164 [+0.036, +0.289] | +0.213 [+0.092, +0.335] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.189 [+0.071, +0.306] | +0.242 [+0.123, +0.359] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.410 [+0.361, +0.450] | +0.225 [+0.190, +0.261] | -0.055 [-0.090, -0.024] | -0.026 [-0.060, +0.006] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.374 [+0.321, +0.422] | +0.230 [+0.186, +0.276] | -0.011 [-0.054, +0.030] | +0.005 [-0.043, +0.057] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.145 [+0.079, +0.201] | +0.028 [-0.029, +0.081] | -0.077 [-0.136, -0.017] | +0.001 [-0.055, +0.056] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.350 [+0.283, +0.418] | +0.320 [+0.254, +0.387] | +0.230 [+0.144, +0.318] | +0.258 [+0.178, +0.344] |
| NICP per coppia in mm | +0.147 [+0.079, +0.215] | +0.122 [+0.060, +0.187] | - | - |
| NICP per coppia cs | +0.306 [+0.228, +0.390] | +0.258 [+0.177, +0.337] | - | - |
| NICP su template in mm | +0.226 [+0.135, +0.316] | +0.070 [-0.018, +0.155] | -0.008 [-0.098, +0.088] | +0.026 [-0.058, +0.115] |
| NICP su template cs | +0.533 [+0.457, +0.596] | +0.489 [+0.423, +0.548] | +0.406 [+0.324, +0.489] | +0.446 [+0.364, +0.529] |
| varifold in mm (massa unitaria) | +0.284 [+0.226, +0.340] | +0.203 [+0.142, +0.262] | -0.106 [-0.174, -0.038] | -0.050 [-0.109, +0.010] |
| GNM (visto) preregistrato, coefficienti | +0.120 [+0.030, +0.216] | +0.033 [-0.045, +0.110] | -0.000 [-0.082, +0.093] | +0.025 [-0.049, +0.106] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.026 [-0.071, +0.123] | +0.000 [-0.089, +0.088] | -0.011 [-0.099, +0.086] | +0.031 [-0.054, +0.122] |
| GNM (visto) vA, coefficienti | +0.097 [+0.009, +0.191] | +0.042 [-0.037, +0.123] | +0.062 [-0.024, +0.153] | +0.052 [-0.025, +0.132] |
| GNM (visto) vA, mesh d'identita' FR | +0.019 [-0.075, +0.116] | -0.001 [-0.089, +0.086] | -0.012 [-0.099, +0.081] | +0.032 [-0.051, +0.121] |
| GNM (visto) vB, coefficienti | +0.039 [-0.041, +0.121] | +0.044 [-0.037, +0.127] | +0.116 [+0.030, +0.201] | +0.103 [+0.023, +0.190] |
| GNM (visto) vB, mesh d'identita' FR | -0.062 [-0.146, +0.021] | -0.067 [-0.142, +0.007] | -0.065 [-0.143, +0.014] | -0.026 [-0.096, +0.047] |
| GNM (visto) vB, composizione S_B + k d_P | -0.098 [-0.173, -0.026] | -0.100 [-0.165, -0.037] | -0.086 [-0.156, -0.014] | -0.057 [-0.118, +0.006] |
| FLAME 2023 Open preregistrato, coefficienti | +0.162 [+0.079, +0.245] | +0.077 [+0.001, +0.156] | +0.083 [-0.012, +0.175] | +0.071 [-0.012, +0.156] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.017 [-0.069, +0.108] | -0.024 [-0.104, +0.053] | -0.015 [-0.102, +0.075] | +0.000 [-0.075, +0.079] |
| FLAME 2023 Open vA, coefficienti | +0.140 [+0.055, +0.223] | +0.060 [-0.018, +0.139] | +0.065 [-0.030, +0.158] | +0.056 [-0.024, +0.141] |
| FLAME 2023 Open vA, mesh d'identita' FR | -0.003 [-0.091, +0.089] | -0.032 [-0.114, +0.048] | -0.017 [-0.103, +0.073] | -0.001 [-0.076, +0.076] |
| FLAME 2023 Open vB, coefficienti | +0.020 [-0.062, +0.107] | +0.014 [-0.062, +0.094] | +0.097 [+0.011, +0.187] | +0.071 [-0.008, +0.153] |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.076 [-0.154, +0.002] | -0.086 [-0.157, -0.020] | -0.058 [-0.137, +0.019] | -0.045 [-0.114, +0.021] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.108 [-0.179, -0.040] | -0.114 [-0.176, -0.054] | -0.074 [-0.148, -0.004] | -0.068 [-0.127, -0.010] |
| ArcFace-render, normal map 3 viste (G0) | +0.423 [+0.297, +0.530] | +0.448 [+0.334, +0.546] | +0.508 [+0.395, +0.616] | +0.535 [+0.441, +0.619] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.380 [+0.256, +0.488] | +0.412 [+0.303, +0.509] | +0.460 [+0.345, +0.571] | +0.501 [+0.407, +0.588] |
| ShapeDNA k=50 | +0.554 [+0.466, +0.627] | +0.526 [+0.457, +0.589] | +0.452 [+0.345, +0.552] | +0.399 [+0.298, +0.491] |
| HKS globale | +0.571 [+0.490, +0.639] | +0.566 [+0.495, +0.628] | +0.592 [+0.501, +0.679] | +0.544 [+0.448, +0.631] |
| WKS globale | +0.529 [+0.461, +0.593] | +0.561 [+0.495, +0.621] | +0.487 [+0.375, +0.595] | +0.449 [+0.357, +0.543] |
| OpenShape PointBERT | +0.426 [+0.353, +0.492] | +0.465 [+0.398, +0.522] | +0.313 [+0.219, +0.422] | +0.374 [+0.283, +0.458] |
| Uni3D-g | +0.394 [+0.301, +0.477] | +0.470 [+0.396, +0.531] | +0.335 [+0.220, +0.454] | +0.372 [+0.266, +0.477] |
| taglia stimata dalla mesh (non oracolo) | +0.507 [+0.434, +0.568] | +0.231 [+0.153, +0.302] | +0.011 [-0.085, +0.108] | +0.028 [-0.063, +0.123] |
| taglia oracolo (S di FR) | -0.174 [-0.252, -0.093] | -0.153 [-0.223, -0.079] | -0.085 [-0.156, -0.010] | -0.088 [-0.154, -0.016] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.336 [+0.252, +0.416] | +0.414 [+0.346, +0.479] | +0.224 [+0.116, +0.333] | +0.286 [+0.190, +0.386] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.400 [+0.315, +0.484] | +0.458 [+0.382, +0.522] | +0.288 [+0.172, +0.408] | +0.283 [+0.173, +0.387] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.317 [+0.235, +0.396] | +0.393 [+0.330, +0.451] | +0.212 [+0.106, +0.319] | +0.216 [+0.117, +0.317] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.404 [+0.342, +0.460] | +0.230 [+0.166, +0.294] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.106 [+0.037, +0.173] | +0.039 [-0.027, +0.106] | -0.117 [-0.186, -0.052] | -0.073 [-0.138, -0.013] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.140 [-0.201, -0.084] | -0.115 [-0.166, -0.068] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | +0.152 [+0.075, +0.224] | +0.170 [+0.094, +0.240] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.064 [-0.044, +0.165] | +0.088 [-0.019, +0.186] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.089 [-0.004, +0.179] | +0.117 [+0.027, +0.205] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.310 [+0.256, +0.358] | +0.099 [+0.044, +0.153] | -0.188 [-0.251, -0.131] | -0.139 [-0.199, -0.082] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.274 [+0.216, +0.327] | +0.105 [+0.050, +0.163] | -0.144 [-0.204, -0.084] | -0.108 [-0.171, -0.042] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.283 [+0.217, +0.348] | +0.231 [+0.160, +0.300] | +0.191 [+0.111, +0.277] | +0.251 [+0.176, +0.325] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.210 [+0.146, +0.269] | +0.132 [+0.088, +0.176] | -0.009 [-0.070, +0.049] | -0.001 [-0.050, +0.050] |
| NICP per coppia in mm | +0.133 [+0.058, +0.211] | +0.090 [+0.019, +0.164] | - | - |
| NICP per coppia cs | +0.139 [+0.068, +0.205] | +0.040 [-0.016, +0.095] | - | - |
| NICP su template in mm | +0.503 [+0.418, +0.583] | +0.491 [+0.396, +0.579] | +0.566 [+0.472, +0.669] | +0.545 [+0.448, +0.641] |
| NICP su template cs | +0.504 [+0.427, +0.574] | +0.406 [+0.353, +0.459] | +0.261 [+0.169, +0.361] | +0.304 [+0.231, +0.387] |
| varifold in mm (massa unitaria) | +0.384 [+0.328, +0.440] | +0.393 [+0.333, +0.451] | +0.286 [+0.205, +0.371] | +0.319 [+0.246, +0.395] |
| GNM (visto) preregistrato, coefficienti | +0.444 [+0.355, +0.527] | +0.421 [+0.337, +0.507] | +0.480 [+0.386, +0.591] | +0.470 [+0.379, +0.569] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.350 [+0.265, +0.435] | +0.293 [+0.219, +0.372] | +0.330 [+0.241, +0.435] | +0.310 [+0.227, +0.406] |
| GNM (visto) vA, coefficienti | +0.402 [+0.305, +0.492] | +0.386 [+0.300, +0.477] | +0.441 [+0.341, +0.552] | +0.434 [+0.348, +0.529] |
| GNM (visto) vA, mesh d'identita' SR | +0.336 [+0.253, +0.420] | +0.277 [+0.201, +0.355] | +0.311 [+0.223, +0.412] | +0.296 [+0.214, +0.389] |
| GNM (visto) vB, coefficienti | -0.101 [-0.184, -0.015] | -0.107 [-0.190, -0.024] | -0.012 [-0.097, +0.077] | -0.042 [-0.124, +0.042] |
| GNM (visto) vB, mesh d'identita' SR | -0.021 [-0.088, +0.043] | -0.030 [-0.085, +0.025] | +0.054 [-0.011, +0.118] | +0.025 [-0.028, +0.082] |
| FLAME 2023 Open preregistrato, coefficienti | +0.453 [+0.378, +0.529] | +0.420 [+0.340, +0.502] | +0.522 [+0.432, +0.621] | +0.468 [+0.381, +0.561] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.351 [+0.283, +0.419] | +0.304 [+0.233, +0.377] | +0.370 [+0.281, +0.460] | +0.332 [+0.250, +0.416] |
| FLAME 2023 Open vA, coefficienti | +0.453 [+0.378, +0.530] | +0.422 [+0.340, +0.506] | +0.526 [+0.435, +0.625] | +0.472 [+0.386, +0.566] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.330 [+0.260, +0.401] | +0.289 [+0.218, +0.363] | +0.356 [+0.266, +0.450] | +0.320 [+0.239, +0.407] |
| FLAME 2023 Open vB, coefficienti | -0.001 [-0.095, +0.090] | -0.016 [-0.105, +0.074] | +0.093 [+0.002, +0.183] | +0.048 [-0.039, +0.135] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.017 [-0.083, +0.045] | -0.029 [-0.086, +0.028] | +0.091 [+0.025, +0.164] | +0.036 [-0.020, +0.094] |
| ArcFace-render, normal map 3 viste (G0) | +0.324 [+0.211, +0.426] | +0.335 [+0.235, +0.432] | +0.432 [+0.327, +0.525] | +0.448 [+0.362, +0.530] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.301 [+0.196, +0.401] | +0.316 [+0.222, +0.406] | +0.403 [+0.297, +0.508] | +0.422 [+0.337, +0.506] |
| ShapeDNA k=50 | +0.512 [+0.428, +0.585] | +0.469 [+0.397, +0.533] | +0.311 [+0.213, +0.411] | +0.255 [+0.169, +0.338] |
| HKS globale | +0.546 [+0.467, +0.618] | +0.518 [+0.447, +0.583] | +0.525 [+0.422, +0.623] | +0.438 [+0.334, +0.542] |
| WKS globale | +0.488 [+0.421, +0.550] | +0.520 [+0.453, +0.580] | +0.398 [+0.305, +0.498] | +0.327 [+0.247, +0.414] |
| OpenShape PointBERT | +0.366 [+0.298, +0.430] | +0.392 [+0.326, +0.451] | +0.251 [+0.169, +0.339] | +0.249 [+0.175, +0.325] |
| Uni3D-g | +0.309 [+0.220, +0.387] | +0.403 [+0.331, +0.465] | +0.218 [+0.112, +0.325] | +0.242 [+0.150, +0.330] |
| taglia stimata dalla mesh (non oracolo) | +0.540 [+0.462, +0.610] | +0.540 [+0.457, +0.618] | +0.628 [+0.526, +0.727] | +0.604 [+0.500, +0.709] |
| taglia oracolo (S di FR) | +0.476 [+0.366, +0.583] | +0.483 [+0.378, +0.587] | +0.592 [+0.492, +0.696] | +0.556 [+0.454, +0.658] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.317 [+0.236, +0.394] | +0.371 [+0.302, +0.437] | +0.211 [+0.118, +0.308] | +0.201 [+0.115, +0.285] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.435 [+0.337, +0.522] | +0.433 [+0.354, +0.504] | +0.248 [+0.149, +0.355] | +0.202 [+0.097, +0.302] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.270 [+0.194, +0.341] | +0.342 [+0.275, +0.399] | +0.168 [+0.087, +0.251] | +0.132 [+0.046, +0.217] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.295 [+0.228, +0.354] | +0.365 [+0.307, +0.417] | - | - |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.227 [+0.150, +0.304] | +0.292 [+0.229, +0.355] | +0.092 [+0.023, +0.162] | +0.111 [+0.036, +0.184] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | +0.113 [+0.040, +0.186] | +0.125 [+0.056, +0.195] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.057 [-0.114, -0.005] | -0.057 [-0.105, -0.010] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.171 [-0.261, -0.090] | -0.159 [-0.245, -0.081] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.095 [-0.164, -0.019] | -0.078 [-0.147, -0.006] | - | - |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.167 [+0.107, +0.224] | +0.212 [+0.158, +0.260] | -0.094 [-0.153, -0.033] | -0.049 [-0.110, +0.010] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | +0.086 [+0.008, +0.161] | +0.136 [+0.074, +0.202] | +0.005 [-0.072, +0.090] | +0.021 [-0.046, +0.102] |

### FaceScape (dev) (crop): Spearman per insieme di righe (seme 796786)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | crop (46931) | all_cross (135655) | stessa_topologia crop (4950) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.355 [0.277, 0.433] | 0.365 [0.314, 0.415] | 0.700 [0.626, 0.767] | 0.688 [0.621, 0.749] |
| **factorized s1234, d_F non cal.** | 0.487 [0.401, 0.574] | 0.438 [0.371, 0.504] | 0.676 [0.593, 0.750] | 0.700 [0.629, 0.766] |
| factorized s1234, d_P | 0.457 [0.368, 0.545] | 0.438 [0.368, 0.507] | 0.659 [0.572, 0.737] | 0.676 [0.599, 0.745] |
| **factorized s2345, d_F cal.*** | 0.282 [0.214, 0.348] | 0.345 [0.303, 0.387] | 0.711 [0.635, 0.775] | 0.702 [0.633, 0.764] |
| **factorized s2345, d_F non cal.** | 0.467 [0.378, 0.552] | 0.415 [0.353, 0.475] | 0.695 [0.611, 0.768] | 0.707 [0.628, 0.778] |
| factorized s2345, d_P | 0.425 [0.336, 0.515] | 0.415 [0.351, 0.481] | 0.680 [0.595, 0.758] | 0.679 [0.593, 0.754] |
| **ctrlfr s1234** | 0.266 [0.199, 0.333] | 0.325 [0.286, 0.361] | 0.730 [0.675, 0.782] | 0.725 [0.665, 0.778] |
| **ctrlfr s2345** | 0.236 [0.176, 0.305] | 0.315 [0.276, 0.352] | 0.711 [0.648, 0.762] | 0.700 [0.635, 0.752] |
| **C3M e123, d_F cal.*** | 0.473 [0.402, 0.547] | 0.413 [0.355, 0.469] | 0.702 [0.626, 0.764] | 0.677 [0.612, 0.735] |
| **C3M e123, d_F non cal.** | 0.539 [0.460, 0.617] | 0.461 [0.391, 0.528] | 0.680 [0.595, 0.752] | 0.691 [0.625, 0.751] |
| C3M e123, d_P | 0.514 [0.434, 0.595] | 0.447 [0.377, 0.514] | 0.673 [0.585, 0.747] | 0.674 [0.606, 0.737] |
| **C3M e205, d_F cal.*** | 0.451 [0.381, 0.519] | 0.420 [0.370, 0.468] | 0.720 [0.648, 0.782] | 0.728 [0.669, 0.776] |
| **C3M e205, d_F non cal.** | 0.516 [0.439, 0.590] | 0.469 [0.403, 0.533] | 0.692 [0.611, 0.763] | 0.707 [0.642, 0.767] |
| C3M e205, d_P | 0.498 [0.419, 0.575] | 0.454 [0.387, 0.517] | 0.685 [0.603, 0.758] | 0.688 [0.625, 0.750] |
| **ICP + Chamfer in mm (per coppia)** | 0.292 [0.252, 0.332] | 0.376 [0.319, 0.429] | 0.667 [0.594, 0.734] | 0.582 [0.514, 0.645] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.225 [0.161, 0.288] | 0.311 [0.246, 0.373] | 0.588 [0.502, 0.672] | 0.575 [0.495, 0.653] |
| **NICP per coppia in mm** | 0.324 [0.261, 0.387] | 0.342 [0.277, 0.410] | - | - |
| **NICP per coppia cs** | 0.228 [0.167, 0.291] | 0.300 [0.236, 0.369] | - | - |
| **NICP su template in mm** | 0.070 [0.018, 0.124] | 0.257 [0.219, 0.292] | 0.637 [0.560, 0.698] | 0.579 [0.509, 0.639] |
| **NICP su template cs** | 0.053 [0.010, 0.096] | 0.188 [0.149, 0.229] | 0.507 [0.419, 0.584] | 0.459 [0.381, 0.537] |
| **varifold in mm (massa unitaria)** | 0.218 [0.172, 0.265] | 0.161 [0.134, 0.186] | 0.722 [0.648, 0.789] | 0.639 [0.561, 0.706] |
| **GNM (visto) preregistrato, coefficienti** | 0.328 [0.260, 0.386] | 0.417 [0.341, 0.482] | 0.510 [0.418, 0.593] | 0.504 [0.418, 0.575] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.375 [0.313, 0.433] | 0.429 [0.362, 0.494] | 0.589 [0.516, 0.653] | 0.499 [0.424, 0.569] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.399 [0.316, 0.476] | 0.465 [0.376, 0.550] | 0.534 [0.449, 0.621] | 0.535 [0.440, 0.625] |
| **GNM (visto) vA, coefficienti*** | 0.360 [0.288, 0.422] | 0.424 [0.343, 0.494] | 0.477 [0.387, 0.560] | 0.487 [0.401, 0.559] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.403 [0.341, 0.459] | 0.452 [0.383, 0.514] | 0.599 [0.529, 0.659] | 0.516 [0.442, 0.584] |
| GNM (visto) vA, mesh d'identita' SR* | 0.418 [0.335, 0.498] | 0.466 [0.369, 0.558] | 0.527 [0.436, 0.617] | 0.518 [0.420, 0.615] |
| **GNM (visto) vB, coefficienti*** | 0.549 [0.469, 0.622] | 0.559 [0.482, 0.631] | 0.551 [0.471, 0.628] | 0.562 [0.484, 0.634] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.628 [0.564, 0.684] | 0.659 [0.597, 0.719] | 0.792 [0.747, 0.831] | 0.706 [0.647, 0.763] |
| GNM (visto) vB, mesh d'identita' SR* | 0.755 [0.681, 0.822] | 0.758 [0.674, 0.830] | 0.779 [0.698, 0.849] | 0.767 [0.680, 0.842] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.761 [0.700, 0.812] | 0.783 [0.724, 0.831] | 0.851 [0.804, 0.889] | 0.814 [0.759, 0.859] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.283 [0.223, 0.346] | 0.345 [0.275, 0.415] | 0.510 [0.425, 0.594] | 0.436 [0.350, 0.519] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.343 [0.281, 0.402] | 0.408 [0.337, 0.471] | 0.606 [0.525, 0.678] | 0.495 [0.416, 0.564] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.362 [0.283, 0.446] | 0.387 [0.295, 0.480] | 0.491 [0.391, 0.591] | 0.445 [0.343, 0.545] |
| **FLAME 2023 Open vA, coefficienti*** | 0.285 [0.225, 0.346] | 0.348 [0.278, 0.417] | 0.514 [0.429, 0.595] | 0.440 [0.355, 0.520] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.355 [0.293, 0.414] | 0.416 [0.348, 0.477] | 0.604 [0.522, 0.676] | 0.496 [0.418, 0.565] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.375 [0.291, 0.462] | 0.395 [0.301, 0.491] | 0.492 [0.391, 0.590] | 0.448 [0.346, 0.553] |
| **FLAME 2023 Open vB, coefficienti*** | 0.514 [0.431, 0.591] | 0.535 [0.455, 0.612] | 0.538 [0.451, 0.619] | 0.548 [0.466, 0.625] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.562 [0.492, 0.625] | 0.605 [0.533, 0.668] | 0.730 [0.667, 0.785] | 0.656 [0.582, 0.717] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.703 [0.611, 0.789] | 0.707 [0.609, 0.792] | 0.715 [0.619, 0.802] | 0.713 [0.617, 0.799] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.685 [0.610, 0.750] | 0.713 [0.641, 0.777] | 0.778 [0.709, 0.835] | 0.746 [0.674, 0.806] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.308 [0.235, 0.383] | 0.303 [0.231, 0.374] | 0.342 [0.261, 0.427] | 0.274 [0.211, 0.336] |
| **ShapeDNA k=50** | 0.011 [-0.034, 0.054] | 0.043 [0.028, 0.059] | 0.208 [0.102, 0.307] | 0.287 [0.225, 0.352] |
| **HKS globale** | 0.011 [-0.026, 0.046] | 0.020 [0.007, 0.036] | 0.114 [0.014, 0.212] | 0.109 [0.046, 0.175] |
| **WKS globale** | 0.010 [-0.030, 0.050] | 0.011 [-0.001, 0.023] | 0.214 [0.087, 0.330] | 0.171 [0.113, 0.232] |
| **taglia stimata dalla mesh (non oracolo)** | 0.012 [-0.026, 0.051] | 0.088 [0.059, 0.114] | 0.265 [0.165, 0.355] | 0.296 [0.208, 0.376] |
| **taglia oracolo (S di FR)** | 0.464 [0.344, 0.573] | 0.464 [0.345, 0.572] | 0.464 [0.344, 0.573] | 0.464 [0.344, 0.573] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.189 [0.125, 0.255] | 0.141 [0.097, 0.188] | 0.361 [0.254, 0.470] | 0.324 [0.231, 0.420] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.162 [0.103, 0.224] | 0.123 [0.081, 0.171] | 0.368 [0.263, 0.482] | 0.304 [0.216, 0.402] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.143 [0.084, 0.210] | 0.101 [0.062, 0.142] | 0.451 [0.359, 0.544] | 0.369 [0.279, 0.460] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.135 [0.082, 0.192] | 0.102 [0.067, 0.139] | 0.398 [0.291, 0.504] | 0.357 [0.260, 0.449] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.220 [0.163, 0.281] | 0.166 [0.126, 0.209] | 0.472 [0.378, 0.570] | 0.446 [0.357, 0.537] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.229 [0.170, 0.291] | 0.162 [0.123, 0.206] | 0.497 [0.401, 0.595] | 0.433 [0.346, 0.526] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.067 [0.031, 0.109] | 0.080 [0.064, 0.094] | - | - |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.222 [0.173, 0.276] | 0.142 [0.109, 0.174] | - | - |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.166 [0.108, 0.233] | 0.175 [0.126, 0.221] | 0.512 [0.422, 0.595] | 0.493 [0.394, 0.577] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.292 [0.207, 0.380] | 0.223 [0.162, 0.285] | 0.476 [0.378, 0.570] | 0.461 [0.362, 0.554] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.659 [0.589, 0.717] | 0.669 [0.602, 0.726] | - | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.594 [0.509, 0.673] | 0.613 [0.527, 0.690] | - | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.616 [0.531, 0.691] | 0.616 [0.532, 0.691] | - | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.571 [0.478, 0.666] | 0.573 [0.477, 0.665] | - | - |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.119 [0.088, 0.152] | 0.153 [0.129, 0.176] | 0.633 [0.563, 0.699] | 0.672 [0.606, 0.727] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.210 [0.168, 0.253] | 0.161 [0.132, 0.190] | 0.572 [0.485, 0.650] | 0.593 [0.519, 0.662] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.131 [0.087, 0.180] | 0.158 [0.135, 0.181] | 0.631 [0.549, 0.711] | 0.504 [0.457, 0.549] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.163 [0.111, 0.216] | 0.176 [0.145, 0.209] | 0.594 [0.507, 0.682] | 0.461 [0.404, 0.513] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | crop (46931) | all_cross (135655) | stessa_topologia crop (4950) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.385 [0.315, 0.458] | 0.386 [0.335, 0.432] | 0.684 [0.612, 0.755] | 0.712 [0.648, 0.769] |
| factorized s1234, d_F non cal. | 0.535 [0.452, 0.612] | 0.474 [0.409, 0.534] | 0.707 [0.634, 0.770] | 0.754 [0.691, 0.804] |
| **factorized s1234, d_P** | 0.516 [0.430, 0.596] | 0.482 [0.416, 0.544] | 0.710 [0.635, 0.774] | 0.742 [0.678, 0.794] |
| factorized s2345, d_F cal.* | 0.310 [0.246, 0.376] | 0.358 [0.314, 0.398] | 0.684 [0.614, 0.748] | 0.712 [0.647, 0.770] |
| factorized s2345, d_F non cal. | 0.505 [0.417, 0.584] | 0.445 [0.388, 0.502] | 0.725 [0.655, 0.785] | 0.757 [0.695, 0.812] |
| **factorized s2345, d_P** | 0.474 [0.387, 0.557] | 0.456 [0.395, 0.515] | 0.735 [0.666, 0.793] | 0.745 [0.680, 0.800] |
| ctrlfr s1234 | 0.247 [0.180, 0.314] | 0.306 [0.263, 0.347] | 0.687 [0.616, 0.747] | 0.678 [0.608, 0.741] |
| ctrlfr s2345 | 0.243 [0.182, 0.311] | 0.306 [0.265, 0.344] | 0.658 [0.578, 0.725] | 0.668 [0.598, 0.731] |
| C3M e123, d_F cal.* | 0.501 [0.428, 0.572] | 0.431 [0.373, 0.485] | 0.670 [0.597, 0.738] | 0.692 [0.634, 0.747] |
| C3M e123, d_F non cal. | 0.584 [0.513, 0.652] | 0.500 [0.434, 0.560] | 0.709 [0.634, 0.776] | 0.746 [0.688, 0.795] |
| **C3M e123, d_P** | 0.576 [0.506, 0.644] | 0.494 [0.429, 0.554] | 0.720 [0.646, 0.784] | 0.740 [0.682, 0.788] |
| C3M e205, d_F cal.* | 0.465 [0.400, 0.534] | 0.423 [0.371, 0.471] | 0.687 [0.614, 0.753] | 0.720 [0.664, 0.771] |
| C3M e205, d_F non cal. | 0.556 [0.480, 0.625] | 0.506 [0.439, 0.565] | 0.735 [0.660, 0.800] | 0.763 [0.708, 0.811] |
| **C3M e205, d_P** | 0.555 [0.479, 0.622] | 0.499 [0.435, 0.557] | 0.746 [0.673, 0.808] | 0.756 [0.699, 0.805] |
| **ICP + Chamfer in mm (per coppia)** | 0.306 [0.267, 0.343] | 0.396 [0.339, 0.446] | 0.689 [0.622, 0.751] | 0.608 [0.544, 0.666] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.269 [0.211, 0.326] | 0.362 [0.304, 0.416] | 0.683 [0.613, 0.747] | 0.660 [0.594, 0.717] |
| **NICP per coppia in mm** | 0.348 [0.282, 0.409] | 0.370 [0.303, 0.433] | - | - |
| **NICP per coppia cs** | 0.274 [0.213, 0.331] | 0.350 [0.282, 0.411] | - | - |
| **NICP su template in mm** | 0.047 [-0.003, 0.097] | 0.189 [0.149, 0.225] | 0.483 [0.388, 0.563] | 0.425 [0.346, 0.503] |
| **NICP su template cs** | 0.061 [0.017, 0.107] | 0.205 [0.166, 0.244] | 0.547 [0.460, 0.626] | 0.494 [0.417, 0.566] |
| **varifold in mm (massa unitaria)** | 0.228 [0.182, 0.273] | 0.171 [0.145, 0.194] | 0.743 [0.677, 0.800] | 0.659 [0.595, 0.717] |
| **GNM (visto) preregistrato, coefficienti** | 0.298 [0.227, 0.367] | 0.389 [0.319, 0.461] | 0.484 [0.390, 0.568] | 0.475 [0.395, 0.553] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.299 [0.241, 0.359] | 0.342 [0.283, 0.401] | 0.437 [0.345, 0.525] | 0.392 [0.322, 0.463] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.412 [0.324, 0.495] | 0.503 [0.412, 0.586] | 0.580 [0.492, 0.664] | 0.587 [0.494, 0.668] |
| **GNM (visto) vA, coefficienti*** | 0.345 [0.269, 0.420] | 0.418 [0.340, 0.496] | 0.475 [0.383, 0.557] | 0.485 [0.402, 0.567] |
| GNM (visto) vA, mesh d'identita' FR* | 0.327 [0.269, 0.388] | 0.366 [0.305, 0.427] | 0.459 [0.369, 0.542] | 0.414 [0.346, 0.486] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.441 [0.351, 0.526] | 0.514 [0.417, 0.602] | 0.585 [0.495, 0.672] | 0.581 [0.481, 0.669] |
| **GNM (visto) vB, coefficienti*** | 0.601 [0.534, 0.664] | 0.611 [0.543, 0.676] | 0.614 [0.547, 0.675] | 0.615 [0.547, 0.677] |
| GNM (visto) vB, mesh d'identita' FR* | 0.566 [0.497, 0.635] | 0.592 [0.524, 0.664] | 0.694 [0.616, 0.769] | 0.628 [0.555, 0.705] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.830 [0.778, 0.873] | 0.839 [0.786, 0.881] | 0.872 [0.828, 0.910] | 0.855 [0.807, 0.895] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.744 [0.682, 0.801] | 0.766 [0.705, 0.822] | 0.833 [0.782, 0.878] | 0.796 [0.739, 0.849] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.279 [0.215, 0.348] | 0.351 [0.277, 0.425] | 0.524 [0.435, 0.608] | 0.447 [0.361, 0.527] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.294 [0.231, 0.355] | 0.346 [0.279, 0.411] | 0.502 [0.412, 0.593] | 0.417 [0.339, 0.495] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.405 [0.323, 0.489] | 0.444 [0.352, 0.532] | 0.574 [0.476, 0.663] | 0.517 [0.415, 0.607] |
| **FLAME 2023 Open vA, coefficienti*** | 0.279 [0.215, 0.346] | 0.351 [0.279, 0.424] | 0.522 [0.431, 0.606] | 0.448 [0.364, 0.528] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.297 [0.236, 0.359] | 0.344 [0.278, 0.409] | 0.483 [0.393, 0.571] | 0.406 [0.331, 0.482] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.424 [0.340, 0.509] | 0.458 [0.364, 0.547] | 0.578 [0.478, 0.670] | 0.525 [0.422, 0.618] |
| **FLAME 2023 Open vB, coefficienti*** | 0.554 [0.476, 0.629] | 0.577 [0.496, 0.648] | 0.595 [0.520, 0.665] | 0.594 [0.515, 0.665] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.531 [0.462, 0.600] | 0.568 [0.497, 0.638] | 0.673 [0.600, 0.741] | 0.610 [0.534, 0.683] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.807 [0.742, 0.859] | 0.814 [0.749, 0.865] | 0.834 [0.772, 0.885] | 0.824 [0.760, 0.874] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.696 [0.629, 0.757] | 0.725 [0.660, 0.784] | 0.795 [0.739, 0.843] | 0.757 [0.693, 0.812] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.369 [0.302, 0.440] | 0.355 [0.290, 0.422] | 0.413 [0.339, 0.493] | 0.322 [0.265, 0.381] |
| **ShapeDNA k=50** | -0.002 [-0.045, 0.036] | 0.052 [0.038, 0.067] | 0.186 [0.089, 0.281] | 0.327 [0.262, 0.388] |
| **HKS globale** | -0.006 [-0.043, 0.028] | 0.020 [0.007, 0.036] | 0.119 [0.032, 0.206] | 0.129 [0.068, 0.191] |
| **WKS globale** | 0.005 [-0.035, 0.043] | 0.010 [-0.002, 0.023] | 0.195 [0.081, 0.303] | 0.171 [0.109, 0.236] |
| **taglia stimata dalla mesh (non oracolo)** | 0.012 [-0.029, 0.055] | 0.041 [0.019, 0.062] | 0.088 [0.003, 0.180] | 0.126 [0.056, 0.196] |
| **taglia oracolo (S di FR)** | 0.118 [0.022, 0.228] | 0.119 [0.022, 0.228] | 0.118 [0.023, 0.229] | 0.118 [0.023, 0.229] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.199 [0.135, 0.265] | 0.145 [0.102, 0.192] | 0.384 [0.274, 0.498] | 0.337 [0.246, 0.434] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.179 [0.119, 0.242] | 0.133 [0.092, 0.179] | 0.406 [0.298, 0.517] | 0.329 [0.238, 0.428] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.157 [0.101, 0.219] | 0.105 [0.068, 0.144] | 0.464 [0.375, 0.557] | 0.362 [0.269, 0.458] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.139 [0.084, 0.198] | 0.107 [0.072, 0.145] | 0.429 [0.323, 0.538] | 0.361 [0.265, 0.459] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.249 [0.192, 0.308] | 0.178 [0.140, 0.221] | 0.511 [0.415, 0.610] | 0.483 [0.396, 0.572] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.258 [0.202, 0.317] | 0.179 [0.141, 0.220] | 0.552 [0.461, 0.645] | 0.485 [0.400, 0.575] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.061 [0.022, 0.105] | 0.067 [0.050, 0.083] | - | - |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.243 [0.194, 0.293] | 0.152 [0.121, 0.183] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | 0.172 [0.107, 0.240] | 0.176 [0.130, 0.225] | 0.461 [0.366, 0.550] | 0.479 [0.387, 0.566] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.311 [0.227, 0.394] | 0.237 [0.179, 0.299] | 0.505 [0.409, 0.589] | 0.491 [0.406, 0.575] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.677 [0.614, 0.735] | 0.693 [0.632, 0.745] | - | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.689 [0.627, 0.746] | 0.712 [0.656, 0.765] | - | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.716 [0.650, 0.773] | 0.717 [0.651, 0.773] | - | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.670 [0.601, 0.734] | 0.671 [0.602, 0.734] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.116 [0.083, 0.152] | 0.146 [0.121, 0.170] | 0.597 [0.525, 0.663] | 0.642 [0.577, 0.699] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.227 [0.183, 0.268] | 0.175 [0.150, 0.201] | 0.613 [0.540, 0.681] | 0.643 [0.583, 0.699] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.138 [0.091, 0.187] | 0.162 [0.138, 0.186] | 0.607 [0.525, 0.688] | 0.508 [0.455, 0.553] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.199 [0.145, 0.251] | 0.205 [0.176, 0.235] | 0.658 [0.582, 0.733] | 0.524 [0.480, 0.562] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.062 [-0.002, +0.130] | -0.011 [-0.049, +0.025] | +0.033 [-0.032, +0.102] | +0.106 [+0.055, +0.155] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.130 [+0.066, +0.197] | +0.054 [+0.013, +0.096] | +0.112 [+0.038, +0.196] | +0.113 [+0.055, +0.176] |
| NICP per coppia in mm | +0.031 [-0.039, +0.097] | +0.023 [-0.024, +0.069] | - | - |
| NICP per coppia cs | +0.126 [+0.055, +0.194] | +0.064 [+0.020, +0.110] | - | - |
| NICP su template in mm | +0.285 [+0.189, +0.379] | +0.108 [+0.050, +0.167] | +0.063 [-0.031, +0.170] | +0.109 [+0.026, +0.198] |
| NICP su template cs | +0.301 [+0.224, +0.393] | +0.177 [+0.130, +0.225] | +0.193 [+0.107, +0.291] | +0.229 [+0.151, +0.310] |
| varifold in mm (massa unitaria) | +0.136 [+0.076, +0.200] | +0.204 [+0.169, +0.235] | -0.023 [-0.098, +0.048] | +0.049 [-0.000, +0.102] |
| GNM (visto) preregistrato, coefficienti | +0.026 [-0.054, +0.121] | -0.053 [-0.125, +0.028] | +0.190 [+0.082, +0.302] | +0.184 [+0.097, +0.276] |
| GNM (visto) preregistrato, mesh d'identita' FR | -0.021 [-0.107, +0.076] | -0.065 [-0.142, +0.010] | +0.111 [+0.015, +0.204] | +0.189 [+0.104, +0.277] |
| GNM (visto) vA, coefficienti | -0.005 [-0.086, +0.093] | -0.059 [-0.130, +0.026] | +0.223 [+0.115, +0.342] | +0.201 [+0.115, +0.295] |
| GNM (visto) vA, mesh d'identita' FR | -0.049 [-0.136, +0.049] | -0.087 [-0.163, -0.012] | +0.101 [+0.007, +0.194] | +0.172 [+0.088, +0.259] |
| GNM (visto) vB, coefficienti | -0.194 [-0.291, -0.112] | -0.194 [-0.265, -0.132] | +0.148 [+0.076, +0.214] | +0.126 [+0.052, +0.191] |
| GNM (visto) vB, mesh d'identita' FR | -0.274 [-0.347, -0.197] | -0.295 [-0.354, -0.239] | -0.092 [-0.164, -0.022] | -0.018 [-0.081, +0.039] |
| GNM (visto) vB, composizione S_B + k d_P | -0.406 [-0.469, -0.343] | -0.418 [-0.461, -0.374] | -0.151 [-0.221, -0.080] | -0.126 [-0.179, -0.077] |
| FLAME 2023 Open preregistrato, coefficienti | +0.071 [-0.014, +0.164] | +0.019 [-0.051, +0.097] | +0.190 [+0.091, +0.296] | +0.253 [+0.162, +0.347] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.011 [-0.072, +0.104] | -0.043 [-0.114, +0.037] | +0.094 [-0.008, +0.195] | +0.193 [+0.108, +0.285] |
| FLAME 2023 Open vA, coefficienti | +0.070 [-0.013, +0.162] | +0.017 [-0.054, +0.090] | +0.186 [+0.088, +0.289] | +0.249 [+0.161, +0.343] |
| FLAME 2023 Open vA, mesh d'identita' FR | -0.001 [-0.087, +0.092] | -0.051 [-0.124, +0.027] | +0.096 [-0.005, +0.197] | +0.192 [+0.106, +0.282] |
| FLAME 2023 Open vB, coefficienti | -0.159 [-0.252, -0.070] | -0.170 [-0.237, -0.106] | +0.162 [+0.086, +0.233] | +0.140 [+0.071, +0.207] |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.207 [-0.278, -0.133] | -0.240 [-0.297, -0.182] | -0.030 [-0.113, +0.049] | +0.032 [-0.031, +0.091] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.330 [-0.392, -0.262] | -0.349 [-0.402, -0.296] | -0.079 [-0.158, +0.002] | -0.058 [-0.113, -0.005] |
| ArcFace-render, normal map 3 viste (G0) | +0.047 [-0.048, +0.135] | +0.062 [-0.004, +0.130] | +0.358 [+0.273, +0.435] | +0.414 [+0.349, +0.481] |
| ShapeDNA k=50 | +0.344 [+0.252, +0.433] | +0.322 [+0.276, +0.364] | +0.492 [+0.402, +0.588] | +0.401 [+0.338, +0.469] |
| HKS globale | +0.344 [+0.247, +0.442] | +0.344 [+0.295, +0.392] | +0.586 [+0.495, +0.678] | +0.579 [+0.499, +0.656] |
| WKS globale | +0.345 [+0.258, +0.440] | +0.354 [+0.305, +0.402] | +0.485 [+0.387, +0.589] | +0.517 [+0.443, +0.591] |
| taglia stimata dalla mesh (non oracolo) | +0.343 [+0.263, +0.426] | +0.276 [+0.216, +0.336] | +0.435 [+0.300, +0.570] | +0.392 [+0.275, +0.506] |
| taglia oracolo (S di FR) | -0.109 [-0.267, +0.056] | -0.100 [-0.233, +0.045] | +0.236 [+0.097, +0.383] | +0.224 [+0.072, +0.377] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.166 [+0.082, +0.255] | +0.224 [+0.175, +0.276] | +0.339 [+0.228, +0.451] | +0.364 [+0.279, +0.458] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.211 [+0.136, +0.284] | +0.264 [+0.213, +0.314] | +0.249 [+0.157, +0.346] | +0.320 [+0.235, +0.404] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.134 [+0.055, +0.215] | +0.199 [+0.152, +0.249] | +0.228 [+0.137, +0.318] | +0.242 [+0.165, +0.320] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.288 [+0.214, +0.364] | +0.285 [+0.238, +0.328] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.189 [+0.121, +0.259] | +0.190 [+0.147, +0.233] | +0.187 [+0.108, +0.273] | +0.195 [+0.119, +0.277] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.304 [-0.380, -0.228] | -0.305 [-0.356, -0.250] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.239 [-0.318, -0.163] | -0.248 [-0.305, -0.187] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.262 [-0.341, -0.181] | -0.252 [-0.319, -0.186] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.216 [-0.293, -0.136] | -0.208 [-0.279, -0.135] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.236 [+0.166, +0.307] | +0.212 [+0.173, +0.251] | +0.067 [+0.005, +0.129] | +0.017 [-0.034, +0.065] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.223 [+0.134, +0.312] | +0.207 [+0.165, +0.246] | +0.069 [-0.024, +0.162] | +0.184 [+0.132, +0.232] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.195 [+0.128, +0.264] | +0.062 [+0.020, +0.106] | +0.008 [-0.061, +0.082] | +0.118 [+0.063, +0.170] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.262 [+0.199, +0.328] | +0.127 [+0.088, +0.170] | +0.088 [+0.017, +0.164] | +0.126 [+0.069, +0.180] |
| NICP per coppia in mm | +0.163 [+0.095, +0.235] | +0.096 [+0.047, +0.146] | - | - |
| NICP per coppia cs | +0.259 [+0.191, +0.327] | +0.138 [+0.088, +0.188] | - | - |
| NICP su template in mm | +0.417 [+0.320, +0.513] | +0.181 [+0.110, +0.258] | +0.039 [-0.067, +0.154] | +0.121 [+0.031, +0.214] |
| NICP su template cs | +0.434 [+0.352, +0.522] | +0.250 [+0.191, +0.307] | +0.169 [+0.082, +0.273] | +0.241 [+0.159, +0.324] |
| varifold in mm (massa unitaria) | +0.269 [+0.209, +0.327] | +0.277 [+0.229, +0.325] | -0.047 [-0.122, +0.028] | +0.061 [+0.014, +0.108] |
| GNM (visto) preregistrato, coefficienti | +0.159 [+0.076, +0.250] | +0.021 [-0.060, +0.106] | +0.166 [+0.054, +0.290] | +0.196 [+0.105, +0.290] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.112 [+0.022, +0.202] | +0.009 [-0.083, +0.097] | +0.087 [-0.017, +0.191] | +0.201 [+0.102, +0.295] |
| GNM (visto) vA, coefficienti | +0.127 [+0.042, +0.222] | +0.014 [-0.068, +0.100] | +0.198 [+0.086, +0.324] | +0.213 [+0.119, +0.307] |
| GNM (visto) vA, mesh d'identita' FR | +0.084 [-0.002, +0.172] | -0.014 [-0.102, +0.073] | +0.076 [-0.026, +0.182] | +0.184 [+0.087, +0.278] |
| GNM (visto) vB, coefficienti | -0.061 [-0.157, +0.021] | -0.121 [-0.195, -0.056] | +0.124 [+0.050, +0.193] | +0.139 [+0.063, +0.202] |
| GNM (visto) vB, mesh d'identita' FR | -0.141 [-0.211, -0.073] | -0.221 [-0.290, -0.154] | -0.117 [-0.197, -0.036] | -0.005 [-0.078, +0.064] |
| GNM (visto) vB, composizione S_B + k d_P | -0.274 [-0.338, -0.210] | -0.345 [-0.398, -0.291] | -0.175 [-0.251, -0.098] | -0.114 [-0.174, -0.062] |
| FLAME 2023 Open preregistrato, coefficienti | +0.204 [+0.125, +0.286] | +0.093 [+0.017, +0.176] | +0.165 [+0.061, +0.276] | +0.265 [+0.173, +0.362] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.144 [+0.064, +0.226] | +0.030 [-0.053, +0.118] | +0.069 [-0.041, +0.178] | +0.205 [+0.105, +0.303] |
| FLAME 2023 Open vA, coefficienti | +0.202 [+0.126, +0.284] | +0.090 [+0.014, +0.171] | +0.162 [+0.059, +0.273] | +0.261 [+0.169, +0.357] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.132 [+0.051, +0.214] | +0.022 [-0.062, +0.109] | +0.072 [-0.038, +0.179] | +0.204 [+0.106, +0.303] |
| FLAME 2023 Open vB, coefficienti | -0.027 [-0.116, +0.060] | -0.097 [-0.168, -0.023] | +0.137 [+0.060, +0.220] | +0.152 [+0.079, +0.223] |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.075 [-0.141, -0.006] | -0.167 [-0.233, -0.105] | -0.054 [-0.141, +0.031] | +0.044 [-0.030, +0.114] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.198 [-0.265, -0.130] | -0.275 [-0.333, -0.220] | -0.103 [-0.187, -0.018] | -0.046 [-0.108, +0.010] |
| ArcFace-render, normal map 3 viste (G0) | +0.179 [+0.091, +0.265] | +0.135 [+0.060, +0.208] | +0.334 [+0.244, +0.419] | +0.426 [+0.356, +0.495] |
| ShapeDNA k=50 | +0.476 [+0.383, +0.562] | +0.395 [+0.335, +0.455] | +0.467 [+0.371, +0.569] | +0.413 [+0.346, +0.484] |
| HKS globale | +0.476 [+0.382, +0.572] | +0.418 [+0.354, +0.483] | +0.562 [+0.465, +0.662] | +0.591 [+0.513, +0.670] |
| WKS globale | +0.477 [+0.388, +0.566] | +0.428 [+0.364, +0.494] | +0.461 [+0.358, +0.569] | +0.529 [+0.452, +0.611] |
| taglia stimata dalla mesh (non oracolo) | +0.475 [+0.394, +0.560] | +0.350 [+0.277, +0.425] | +0.411 [+0.265, +0.553] | +0.404 [+0.285, +0.522] |
| taglia oracolo (S di FR) | +0.023 [-0.141, +0.197] | -0.026 [-0.170, +0.133] | +0.212 [+0.059, +0.369] | +0.236 [+0.085, +0.398] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.298 [+0.218, +0.385] | +0.297 [+0.239, +0.357] | +0.315 [+0.201, +0.427] | +0.377 [+0.288, +0.469] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.344 [+0.260, +0.423] | +0.337 [+0.280, +0.400] | +0.225 [+0.130, +0.329] | +0.332 [+0.249, +0.418] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.267 [+0.189, +0.348] | +0.272 [+0.215, +0.331] | +0.203 [+0.115, +0.301] | +0.254 [+0.173, +0.332] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.420 [+0.343, +0.499] | +0.358 [+0.297, +0.419] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.321 [+0.249, +0.394] | +0.263 [+0.215, +0.311] | +0.163 [+0.080, +0.251] | +0.207 [+0.130, +0.287] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.172 [-0.248, -0.091] | -0.231 [-0.290, -0.170] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.107 [-0.181, -0.029] | -0.175 [-0.233, -0.118] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.129 [-0.206, -0.054] | -0.178 [-0.244, -0.113] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.084 [-0.161, -0.004] | -0.134 [-0.202, -0.068] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.368 [+0.297, +0.437] | +0.286 [+0.233, +0.338] | +0.042 [-0.026, +0.112] | +0.029 [-0.023, +0.077] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.356 [+0.274, +0.437] | +0.280 [+0.226, +0.336] | +0.045 [-0.050, +0.142] | +0.196 [+0.146, +0.243] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.210 [+0.143, +0.276] | +0.087 [+0.041, +0.130] | +0.021 [-0.050, +0.085] | +0.134 [+0.077, +0.186] |
| ICP + Chamfer cs (per coppia, scala tolta) | +0.247 [+0.184, +0.310] | +0.121 [+0.078, +0.161] | +0.028 [-0.041, +0.096] | +0.082 [+0.029, +0.135] |
| NICP per coppia in mm | +0.168 [+0.100, +0.235] | +0.113 [+0.061, +0.161] | - | - |
| NICP per coppia cs | +0.242 [+0.169, +0.313] | +0.133 [+0.082, +0.185] | - | - |
| NICP su template in mm | +0.468 [+0.369, +0.560] | +0.294 [+0.232, +0.351] | +0.227 [+0.136, +0.318] | +0.317 [+0.233, +0.390] |
| NICP su template cs | +0.454 [+0.363, +0.538] | +0.277 [+0.217, +0.333] | +0.163 [+0.079, +0.254] | +0.247 [+0.171, +0.328] |
| varifold in mm (massa unitaria) | +0.287 [+0.222, +0.344] | +0.312 [+0.265, +0.354] | -0.033 [-0.099, +0.034] | +0.083 [+0.041, +0.126] |
| GNM (visto) preregistrato, coefficienti | +0.218 [+0.131, +0.303] | +0.093 [+0.017, +0.168] | +0.227 [+0.131, +0.328] | +0.267 [+0.180, +0.351] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.104 [+0.010, +0.199] | -0.021 [-0.107, +0.068] | +0.130 [+0.026, +0.241] | +0.154 [+0.065, +0.248] |
| GNM (visto) vA, coefficienti | +0.170 [+0.074, +0.267] | +0.064 [-0.020, +0.149] | +0.236 [+0.135, +0.346] | +0.257 [+0.165, +0.344] |
| GNM (visto) vA, mesh d'identita' SR | +0.074 [-0.021, +0.168] | -0.032 [-0.124, +0.063] | +0.126 [+0.015, +0.244] | +0.161 [+0.062, +0.261] |
| GNM (visto) vB, coefficienti | -0.086 [-0.178, +0.000] | -0.129 [-0.206, -0.063] | +0.097 [+0.021, +0.169] | +0.126 [+0.056, +0.193] |
| GNM (visto) vB, mesh d'identita' SR | -0.315 [-0.378, -0.253] | -0.357 [-0.406, -0.312] | -0.162 [-0.232, -0.091] | -0.113 [-0.161, -0.070] |
| FLAME 2023 Open preregistrato, coefficienti | +0.237 [+0.158, +0.316] | +0.131 [+0.048, +0.210] | +0.186 [+0.088, +0.290] | +0.294 [+0.203, +0.383] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.110 [+0.017, +0.203] | +0.038 [-0.053, +0.131] | +0.136 [+0.017, +0.267] | +0.225 [+0.124, +0.329] |
| FLAME 2023 Open vA, coefficienti | +0.237 [+0.157, +0.317] | +0.131 [+0.049, +0.209] | +0.189 [+0.091, +0.291] | +0.294 [+0.203, +0.382] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.091 [-0.007, +0.185] | +0.024 [-0.069, +0.117] | +0.132 [+0.011, +0.266] | +0.217 [+0.113, +0.321] |
| FLAME 2023 Open vB, coefficienti | -0.039 [-0.127, +0.044] | -0.094 [-0.162, -0.032] | +0.116 [+0.039, +0.187] | +0.148 [+0.081, +0.211] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.292 [-0.361, -0.221] | -0.331 [-0.387, -0.277] | -0.124 [-0.203, -0.034] | -0.082 [-0.139, -0.027] |
| ArcFace-render, normal map 3 viste (G0) | +0.146 [+0.057, +0.229] | +0.128 [+0.054, +0.195] | +0.297 [+0.198, +0.386] | +0.419 [+0.350, +0.479] |
| ShapeDNA k=50 | +0.517 [+0.424, +0.598] | +0.430 [+0.372, +0.486] | +0.524 [+0.429, +0.614] | +0.414 [+0.351, +0.478] |
| HKS globale | +0.522 [+0.435, +0.604] | +0.463 [+0.398, +0.521] | +0.591 [+0.498, +0.682] | +0.613 [+0.531, +0.690] |
| WKS globale | +0.511 [+0.422, +0.594] | +0.473 [+0.411, +0.530] | +0.516 [+0.416, +0.618] | +0.571 [+0.496, +0.646] |
| taglia stimata dalla mesh (non oracolo) | +0.503 [+0.421, +0.583] | +0.441 [+0.374, +0.501] | +0.622 [+0.509, +0.732] | +0.616 [+0.523, +0.704] |
| taglia oracolo (S di FR) | +0.397 [+0.255, +0.532] | +0.364 [+0.238, +0.482] | +0.592 [+0.469, +0.711] | +0.624 [+0.505, +0.735] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.337 [+0.252, +0.420] | +0.349 [+0.288, +0.404] | +0.304 [+0.189, +0.420] | +0.412 [+0.321, +0.494] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.376 [+0.301, +0.450] | +0.375 [+0.323, +0.428] | +0.282 [+0.175, +0.386] | +0.381 [+0.306, +0.459] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.257 [+0.175, +0.335] | +0.303 [+0.246, +0.359] | +0.158 [+0.058, +0.254] | +0.257 [+0.170, +0.340] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.273 [+0.213, +0.327] | +0.331 [+0.279, +0.376] | - | - |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.204 [+0.129, +0.275] | +0.246 [+0.193, +0.298] | +0.205 [+0.129, +0.286] | +0.251 [+0.185, +0.320] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.162 [-0.235, -0.089] | -0.210 [-0.263, -0.157] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.173 [-0.242, -0.104] | -0.230 [-0.279, -0.180] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.201 [-0.274, -0.130] | -0.234 [-0.292, -0.177] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | -0.155 [-0.222, -0.086] | -0.188 [-0.246, -0.134] | - | - |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.288 [+0.222, +0.348] | +0.307 [+0.258, +0.349] | +0.098 [+0.036, +0.158] | +0.098 [+0.057, +0.136] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | +0.316 [+0.234, +0.390] | +0.278 [+0.231, +0.322] | +0.053 [-0.030, +0.139] | +0.218 [+0.178, +0.253] |

### FaceVerse con espressioni (crop): Spearman per insieme di righe (seme 271049)

GT FR (grassetto = lettura dichiarata; * = post hoc):

| metodo | crop (49500) | all_cross (148500) | stessa_topologia crop (4950) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- |
| **factorized s1234, d_F cal.*** | 0.252 [0.175, 0.326] | 0.284 [0.213, 0.354] | 0.285 [0.180, 0.383] | 0.303 [0.225, 0.374] |
| **factorized s1234, d_F non cal.** | 0.265 [0.185, 0.339] | 0.286 [0.217, 0.352] | 0.288 [0.182, 0.381] | 0.300 [0.232, 0.365] |
| factorized s1234, d_P | 0.252 [0.173, 0.324] | 0.271 [0.198, 0.335] | 0.277 [0.173, 0.373] | 0.286 [0.212, 0.352] |
| **factorized s2345, d_F cal.*** | 0.234 [0.171, 0.292] | 0.288 [0.224, 0.345] | 0.256 [0.163, 0.350] | 0.309 [0.241, 0.369] |
| **factorized s2345, d_F non cal.** | 0.256 [0.187, 0.323] | 0.296 [0.226, 0.362] | 0.265 [0.172, 0.354] | 0.309 [0.238, 0.375] |
| factorized s2345, d_P | 0.251 [0.182, 0.316] | 0.287 [0.219, 0.352] | 0.260 [0.167, 0.350] | 0.300 [0.230, 0.364] |
| **ctrlfr s1234** | 0.256 [0.190, 0.314] | 0.273 [0.208, 0.327] | 0.316 [0.217, 0.404] | 0.302 [0.237, 0.363] |
| **ctrlfr s2345** | 0.239 [0.167, 0.303] | 0.251 [0.185, 0.315] | 0.302 [0.201, 0.398] | 0.276 [0.206, 0.341] |
| **C3M e123, d_F cal.*** | 0.253 [0.186, 0.316] | 0.277 [0.209, 0.339] | 0.286 [0.202, 0.368] | 0.312 [0.235, 0.380] |
| **C3M e123, d_F non cal.** | 0.289 [0.218, 0.356] | 0.296 [0.225, 0.365] | 0.308 [0.228, 0.385] | 0.314 [0.238, 0.386] |
| C3M e123, d_P | 0.287 [0.216, 0.352] | 0.297 [0.224, 0.363] | 0.302 [0.218, 0.382] | 0.312 [0.236, 0.385] |
| **C3M e205, d_F cal.*** | 0.250 [0.185, 0.307] | 0.276 [0.213, 0.336] | 0.294 [0.199, 0.377] | 0.326 [0.253, 0.395] |
| **C3M e205, d_F non cal.** | 0.297 [0.227, 0.365] | 0.302 [0.233, 0.370] | 0.316 [0.230, 0.398] | 0.326 [0.251, 0.395] |
| C3M e205, d_P | 0.297 [0.227, 0.364] | 0.305 [0.235, 0.372] | 0.312 [0.224, 0.397] | 0.325 [0.253, 0.393] |
| **ICP + Chamfer in mm (per coppia)** | 0.271 [0.207, 0.332] | 0.312 [0.242, 0.379] | 0.365 [0.274, 0.443] | 0.343 [0.269, 0.413] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.273 [0.209, 0.332] | 0.275 [0.211, 0.334] | 0.323 [0.234, 0.399] | 0.293 [0.223, 0.358] |
| **NICP per coppia in mm** | 0.225 [0.155, 0.299] | 0.209 [0.140, 0.278] | - | - |
| **NICP per coppia cs** | 0.231 [0.160, 0.303] | 0.217 [0.144, 0.294] | - | - |
| **NICP su template in mm** | 0.124 [0.053, 0.192] | 0.177 [0.096, 0.256] | 0.224 [0.106, 0.331] | 0.213 [0.116, 0.307] |
| **NICP su template cs** | 0.123 [0.068, 0.175] | 0.127 [0.066, 0.184] | 0.189 [0.110, 0.265] | 0.146 [0.072, 0.209] |
| **varifold in mm (massa unitaria)** | 0.237 [0.185, 0.285] | 0.248 [0.195, 0.296] | 0.356 [0.267, 0.449] | 0.285 [0.229, 0.333] |
| **GNM (visto) preregistrato, coefficienti** | 0.216 [0.145, 0.283] | 0.218 [0.146, 0.283] | 0.224 [0.147, 0.300] | 0.221 [0.150, 0.286] |
| **GNM (visto) preregistrato, mesh d'identita' FR** | 0.216 [0.130, 0.297] | 0.223 [0.136, 0.305] | 0.219 [0.123, 0.315] | 0.227 [0.140, 0.311] |
| GNM (visto) preregistrato, mesh d'identita' SR | 0.223 [0.150, 0.292] | 0.226 [0.148, 0.298] | 0.225 [0.147, 0.301] | 0.229 [0.147, 0.303] |
| **GNM (visto) vA, coefficienti*** | 0.222 [0.151, 0.289] | 0.222 [0.148, 0.286] | 0.232 [0.155, 0.309] | 0.225 [0.150, 0.289] |
| **GNM (visto) vA, mesh d'identita' FR*** | 0.219 [0.132, 0.302] | 0.226 [0.139, 0.312] | 0.221 [0.123, 0.318] | 0.230 [0.142, 0.314] |
| GNM (visto) vA, mesh d'identita' SR* | 0.215 [0.142, 0.287] | 0.218 [0.140, 0.292] | 0.217 [0.137, 0.294] | 0.220 [0.140, 0.296] |
| **GNM (visto) vB, coefficienti*** | 0.220 [0.126, 0.301] | 0.224 [0.133, 0.297] | 0.226 [0.116, 0.323] | 0.224 [0.133, 0.299] |
| **GNM (visto) vB, mesh d'identita' FR*** | 0.281 [0.192, 0.352] | 0.289 [0.199, 0.359] | 0.283 [0.186, 0.365] | 0.291 [0.202, 0.363] |
| GNM (visto) vB, mesh d'identita' SR* | 0.305 [0.226, 0.377] | 0.314 [0.233, 0.384] | 0.296 [0.210, 0.372] | 0.312 [0.231, 0.383] |
| **GNM (visto) vB, composizione S_B + k d_P*** | 0.325 [0.242, 0.394] | 0.334 [0.253, 0.402] | 0.322 [0.232, 0.399] | 0.335 [0.253, 0.403] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.162 [0.098, 0.225] | 0.170 [0.107, 0.233] | 0.161 [0.089, 0.236] | 0.176 [0.110, 0.241] |
| **FLAME 2023 Open preregistrato, mesh d'identita' FR** | 0.193 [0.117, 0.265] | 0.205 [0.131, 0.275] | 0.195 [0.110, 0.278] | 0.212 [0.135, 0.281] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | 0.161 [0.089, 0.230] | 0.164 [0.091, 0.232] | 0.164 [0.086, 0.245] | 0.169 [0.096, 0.238] |
| **FLAME 2023 Open vA, coefficienti*** | 0.162 [0.098, 0.225] | 0.170 [0.107, 0.233] | 0.161 [0.089, 0.236] | 0.176 [0.110, 0.241] |
| **FLAME 2023 Open vA, mesh d'identita' FR*** | 0.193 [0.117, 0.265] | 0.205 [0.131, 0.275] | 0.195 [0.110, 0.278] | 0.212 [0.135, 0.281] |
| FLAME 2023 Open vA, mesh d'identita' SR* | 0.161 [0.089, 0.230] | 0.164 [0.091, 0.232] | 0.164 [0.086, 0.245] | 0.169 [0.096, 0.238] |
| **FLAME 2023 Open vB, coefficienti*** | 0.179 [0.084, 0.268] | 0.195 [0.111, 0.272] | 0.160 [0.041, 0.278] | 0.194 [0.112, 0.270] |
| **FLAME 2023 Open vB, mesh d'identita' FR*** | 0.245 [0.166, 0.317] | 0.258 [0.180, 0.327] | 0.243 [0.149, 0.329] | 0.263 [0.186, 0.333] |
| FLAME 2023 Open vB, mesh d'identita' SR* | 0.270 [0.188, 0.348] | 0.283 [0.209, 0.355] | 0.258 [0.167, 0.343] | 0.285 [0.212, 0.358] |
| **FLAME 2023 Open vB, composizione S_B + k d_P*** | 0.283 [0.204, 0.355] | 0.295 [0.217, 0.365] | 0.277 [0.186, 0.360] | 0.300 [0.222, 0.367] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.160 [0.106, 0.217] | 0.137 [0.090, 0.186] | 0.202 [0.127, 0.277] | 0.150 [0.098, 0.204] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.161 [0.102, 0.222] | 0.131 [0.080, 0.183] | 0.209 [0.126, 0.293] | 0.146 [0.089, 0.202] |
| **ShapeDNA k=50** | 0.057 [0.010, 0.104] | 0.060 [0.032, 0.088] | 0.097 [-0.005, 0.198] | 0.127 [0.069, 0.184] |
| **HKS globale** | 0.042 [0.004, 0.082] | 0.035 [0.012, 0.058] | 0.066 [-0.025, 0.160] | 0.067 [0.024, 0.114] |
| **WKS globale** | 0.065 [0.027, 0.104] | 0.044 [0.020, 0.069] | 0.143 [0.056, 0.232] | 0.079 [0.030, 0.127] |
| **taglia stimata dalla mesh (non oracolo)** | 0.009 [-0.022, 0.043] | 0.077 [0.021, 0.136] | 0.120 [0.022, 0.212] | 0.130 [0.040, 0.217] |
| **taglia oracolo (S di FR)** | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] | 0.209 [0.114, 0.307] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR)** | 0.083 [0.040, 0.123] | 0.058 [0.028, 0.088] | 0.150 [0.058, 0.239] | 0.092 [0.044, 0.138] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.090 [0.048, 0.130] | 0.062 [0.033, 0.091] | 0.171 [0.076, 0.262] | 0.099 [0.052, 0.144] |
| **DINOv2 ViT-S/14 su render [profondita'], mm (FR)** | 0.102 [0.052, 0.147] | 0.077 [0.038, 0.115] | 0.169 [0.071, 0.262] | 0.126 [0.073, 0.177] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | 0.098 [0.051, 0.145] | 0.073 [0.034, 0.112] | 0.174 [0.071, 0.274] | 0.121 [0.067, 0.172] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR)** | 0.113 [0.074, 0.148] | 0.075 [0.047, 0.101] | 0.223 [0.136, 0.304] | 0.137 [0.087, 0.184] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | 0.112 [0.071, 0.147] | 0.075 [0.048, 0.100] | 0.212 [0.124, 0.294] | 0.135 [0.085, 0.182] |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR)** | 0.086 [0.057, 0.113] | 0.148 [0.099, 0.194] | - | - |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | 0.186 [0.136, 0.232] | 0.175 [0.124, 0.217] | - | - |
| **LPIPS alex su render [profondita'], mm (FR)** | 0.124 [0.072, 0.175] | 0.130 [0.087, 0.174] | 0.226 [0.134, 0.316] | 0.213 [0.150, 0.272] |
| LPIPS alex su render [profondita'], a CS_ref (SR) | 0.123 [0.070, 0.179] | 0.110 [0.071, 0.149] | 0.200 [0.110, 0.290] | 0.179 [0.122, 0.233] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.341 [0.266, 0.409] | 0.337 [0.266, 0.406] | - | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.301 [0.233, 0.366] | 0.299 [0.235, 0.364] | - | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.318 [0.252, 0.376] | 0.307 [0.242, 0.365] | - | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.193 [0.124, 0.259] | 0.198 [0.137, 0.260] | - | - |
| **ULRSSM [faces_xyz] per coppia, mm (FR)** | 0.171 [0.136, 0.205] | 0.156 [0.125, 0.185] | 0.383 [0.289, 0.464] | 0.292 [0.236, 0.342] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.183 [0.141, 0.223] | 0.151 [0.121, 0.178] | 0.364 [0.274, 0.452] | 0.278 [0.222, 0.327] |
| **ULRSSM [faces_xyz] via template, GT FR (FR)** | 0.219 [0.168, 0.267] | 0.178 [0.132, 0.225] | 0.360 [0.260, 0.449] | 0.249 [0.195, 0.305] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.225 [0.172, 0.275] | 0.177 [0.135, 0.218] | 0.351 [0.253, 0.444] | 0.236 [0.186, 0.287] |

GT SR (grassetto = lettura dichiarata; * = post hoc):

| metodo | crop (49500) | all_cross (148500) | stessa_topologia crop (4950) | stessa_topologia tutte (29700) |
| --- | --- | --- | --- | --- |
| factorized s1234, d_F cal.* | 0.220 [0.143, 0.294] | 0.251 [0.174, 0.325] | 0.241 [0.132, 0.346] | 0.267 [0.184, 0.344] |
| factorized s1234, d_F non cal. | 0.251 [0.172, 0.326] | 0.274 [0.202, 0.341] | 0.264 [0.160, 0.360] | 0.289 [0.216, 0.355] |
| **factorized s1234, d_P** | 0.251 [0.174, 0.324] | 0.273 [0.201, 0.339] | 0.267 [0.160, 0.367] | 0.288 [0.215, 0.354] |
| factorized s2345, d_F cal.* | 0.208 [0.134, 0.271] | 0.258 [0.180, 0.321] | 0.227 [0.121, 0.326] | 0.278 [0.198, 0.346] |
| factorized s2345, d_F non cal. | 0.249 [0.173, 0.317] | 0.288 [0.209, 0.357] | 0.254 [0.155, 0.343] | 0.300 [0.223, 0.367] |
| **factorized s2345, d_P** | 0.257 [0.181, 0.323] | 0.293 [0.219, 0.357] | 0.258 [0.161, 0.349] | 0.304 [0.231, 0.369] |
| ctrlfr s1234 | 0.239 [0.163, 0.306] | 0.261 [0.193, 0.324] | 0.291 [0.185, 0.392] | 0.287 [0.213, 0.356] |
| ctrlfr s2345 | 0.243 [0.171, 0.305] | 0.253 [0.188, 0.316] | 0.303 [0.190, 0.398] | 0.275 [0.205, 0.341] |
| C3M e123, d_F cal.* | 0.232 [0.159, 0.302] | 0.258 [0.182, 0.328] | 0.264 [0.171, 0.351] | 0.292 [0.208, 0.367] |
| C3M e123, d_F non cal. | 0.289 [0.215, 0.359] | 0.299 [0.224, 0.369] | 0.312 [0.231, 0.395] | 0.320 [0.240, 0.392] |
| **C3M e123, d_P** | 0.294 [0.224, 0.364] | 0.306 [0.234, 0.372] | 0.313 [0.227, 0.398] | 0.324 [0.250, 0.394] |
| C3M e205, d_F cal.* | 0.239 [0.170, 0.300] | 0.263 [0.196, 0.325] | 0.286 [0.189, 0.376] | 0.311 [0.232, 0.384] |
| C3M e205, d_F non cal. | 0.305 [0.232, 0.371] | 0.305 [0.232, 0.372] | 0.332 [0.249, 0.418] | 0.331 [0.256, 0.401] |
| **C3M e205, d_P** | 0.308 [0.241, 0.376] | 0.312 [0.244, 0.378] | 0.332 [0.251, 0.421] | 0.334 [0.263, 0.403] |
| **ICP + Chamfer in mm (per coppia)** | 0.251 [0.186, 0.311] | 0.287 [0.217, 0.352] | 0.341 [0.248, 0.423] | 0.315 [0.240, 0.383] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.288 [0.223, 0.347] | 0.290 [0.225, 0.353] | 0.341 [0.252, 0.417] | 0.308 [0.239, 0.375] |
| **NICP per coppia in mm** | 0.248 [0.180, 0.319] | 0.231 [0.158, 0.302] | - | - |
| **NICP per coppia cs** | 0.258 [0.190, 0.328] | 0.242 [0.172, 0.313] | - | - |
| **NICP su template in mm** | 0.085 [0.017, 0.153] | 0.125 [0.047, 0.205] | 0.164 [0.059, 0.274] | 0.153 [0.061, 0.249] |
| **NICP su template cs** | 0.122 [0.066, 0.173] | 0.135 [0.071, 0.191] | 0.182 [0.100, 0.261] | 0.153 [0.078, 0.219] |
| **varifold in mm (massa unitaria)** | 0.227 [0.174, 0.280] | 0.238 [0.185, 0.287] | 0.336 [0.246, 0.424] | 0.271 [0.216, 0.322] |
| **GNM (visto) preregistrato, coefficienti** | 0.177 [0.105, 0.248] | 0.181 [0.107, 0.248] | 0.183 [0.097, 0.265] | 0.183 [0.109, 0.251] |
| GNM (visto) preregistrato, mesh d'identita' FR | 0.161 [0.074, 0.248] | 0.168 [0.080, 0.253] | 0.160 [0.063, 0.257] | 0.170 [0.081, 0.257] |
| **GNM (visto) preregistrato, mesh d'identita' SR** | 0.203 [0.126, 0.278] | 0.207 [0.129, 0.283] | 0.205 [0.121, 0.286] | 0.210 [0.130, 0.288] |
| **GNM (visto) vA, coefficienti*** | 0.182 [0.109, 0.254] | 0.183 [0.107, 0.250] | 0.190 [0.103, 0.277] | 0.185 [0.109, 0.252] |
| GNM (visto) vA, mesh d'identita' FR* | 0.163 [0.077, 0.252] | 0.170 [0.082, 0.257] | 0.161 [0.065, 0.260] | 0.173 [0.083, 0.261] |
| **GNM (visto) vA, mesh d'identita' SR*** | 0.199 [0.122, 0.274] | 0.203 [0.124, 0.278] | 0.202 [0.118, 0.283] | 0.206 [0.124, 0.282] |
| **GNM (visto) vB, coefficienti*** | 0.224 [0.128, 0.303] | 0.223 [0.126, 0.300] | 0.237 [0.127, 0.330] | 0.222 [0.128, 0.301] |
| GNM (visto) vB, mesh d'identita' FR* | 0.235 [0.139, 0.316] | 0.241 [0.147, 0.321] | 0.238 [0.133, 0.329] | 0.243 [0.148, 0.325] |
| **GNM (visto) vB, mesh d'identita' SR*** | 0.316 [0.229, 0.388] | 0.323 [0.238, 0.397] | 0.312 [0.219, 0.397] | 0.322 [0.236, 0.395] |
| GNM (visto) vB, composizione S_B + k d_P* | 0.295 [0.199, 0.372] | 0.302 [0.206, 0.378] | 0.296 [0.192, 0.381] | 0.303 [0.207, 0.380] |
| **FLAME 2023 Open preregistrato, coefficienti** | 0.141 [0.071, 0.208] | 0.148 [0.077, 0.214] | 0.139 [0.062, 0.219] | 0.154 [0.081, 0.221] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | 0.155 [0.074, 0.234] | 0.165 [0.086, 0.241] | 0.152 [0.065, 0.241] | 0.170 [0.088, 0.245] |
| **FLAME 2023 Open preregistrato, mesh d'identita' SR** | 0.153 [0.071, 0.231] | 0.154 [0.076, 0.228] | 0.158 [0.075, 0.246] | 0.160 [0.078, 0.234] |
| **FLAME 2023 Open vA, coefficienti*** | 0.141 [0.071, 0.208] | 0.148 [0.077, 0.214] | 0.139 [0.062, 0.219] | 0.154 [0.081, 0.221] |
| FLAME 2023 Open vA, mesh d'identita' FR* | 0.155 [0.074, 0.234] | 0.165 [0.086, 0.241] | 0.152 [0.065, 0.241] | 0.170 [0.088, 0.245] |
| **FLAME 2023 Open vA, mesh d'identita' SR*** | 0.153 [0.071, 0.231] | 0.154 [0.076, 0.228] | 0.158 [0.075, 0.246] | 0.160 [0.078, 0.234] |
| **FLAME 2023 Open vB, coefficienti*** | 0.178 [0.078, 0.270] | 0.188 [0.099, 0.265] | 0.168 [0.044, 0.285] | 0.188 [0.101, 0.265] |
| FLAME 2023 Open vB, mesh d'identita' FR* | 0.202 [0.110, 0.286] | 0.213 [0.121, 0.294] | 0.199 [0.099, 0.294] | 0.217 [0.125, 0.300] |
| **FLAME 2023 Open vB, mesh d'identita' SR*** | 0.278 [0.189, 0.359] | 0.288 [0.203, 0.366] | 0.271 [0.172, 0.362] | 0.290 [0.202, 0.370] |
| FLAME 2023 Open vB, composizione S_B + k d_P* | 0.254 [0.163, 0.336] | 0.265 [0.179, 0.344] | 0.250 [0.145, 0.341] | 0.268 [0.181, 0.349] |
| **ArcFace-render, normal map 3 viste (G0)** | 0.173 [0.119, 0.233] | 0.149 [0.103, 0.199] | 0.219 [0.143, 0.296] | 0.162 [0.112, 0.220] |
| **ArcFace-render, ombreggiato 3 viste (E12)** | 0.180 [0.123, 0.241] | 0.145 [0.094, 0.196] | 0.236 [0.158, 0.320] | 0.164 [0.112, 0.220] |
| **ShapeDNA k=50** | 0.068 [0.023, 0.110] | 0.062 [0.033, 0.089] | 0.112 [0.010, 0.208] | 0.125 [0.065, 0.181] |
| **HKS globale** | 0.041 [0.001, 0.081] | 0.032 [0.009, 0.056] | 0.064 [-0.025, 0.159] | 0.061 [0.015, 0.109] |
| **WKS globale** | 0.062 [0.022, 0.102] | 0.044 [0.020, 0.067] | 0.120 [0.025, 0.219] | 0.071 [0.021, 0.120] |
| **taglia stimata dalla mesh (non oracolo)** | -0.000 [-0.028, 0.032] | 0.046 [-0.004, 0.099] | 0.080 [-0.019, 0.173] | 0.083 [-0.001, 0.172] |
| **taglia oracolo (S di FR)** | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] | 0.019 [-0.067, 0.106] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | 0.094 [0.050, 0.136] | 0.063 [0.034, 0.092] | 0.184 [0.090, 0.270] | 0.104 [0.055, 0.147] |
| **CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.101 [0.058, 0.141] | 0.066 [0.037, 0.094] | 0.206 [0.105, 0.296] | 0.109 [0.061, 0.154] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | 0.118 [0.067, 0.165] | 0.087 [0.047, 0.123] | 0.201 [0.100, 0.295] | 0.141 [0.084, 0.194] |
| **DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR)** | 0.111 [0.061, 0.157] | 0.083 [0.044, 0.120] | 0.203 [0.097, 0.297] | 0.136 [0.078, 0.187] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | 0.116 [0.075, 0.153] | 0.075 [0.046, 0.101] | 0.230 [0.135, 0.317] | 0.143 [0.089, 0.186] |
| **DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR)** | 0.116 [0.073, 0.151] | 0.076 [0.047, 0.101] | 0.226 [0.133, 0.309] | 0.142 [0.089, 0.186] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | 0.077 [0.046, 0.107] | 0.126 [0.080, 0.168] | - | - |
| **Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR)** | 0.189 [0.135, 0.234] | 0.179 [0.128, 0.222] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | 0.113 [0.059, 0.167] | 0.115 [0.071, 0.159] | 0.211 [0.123, 0.301] | 0.188 [0.128, 0.251] |
| **LPIPS alex su render [profondita'], a CS_ref (SR)** | 0.123 [0.069, 0.181] | 0.110 [0.069, 0.152] | 0.210 [0.118, 0.296] | 0.180 [0.121, 0.234] |
| **NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)** | 0.313 [0.235, 0.384] | 0.310 [0.235, 0.381] | - | - |
| **NoW originale (Procrustes + ICP con scala), mm (media dei due versi)** | 0.312 [0.241, 0.378] | 0.308 [0.238, 0.374] | - | - |
| **REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)** | 0.332 [0.270, 0.390] | 0.319 [0.260, 0.370] | - | - |
| **Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi)** | 0.204 [0.136, 0.268] | 0.200 [0.139, 0.260] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.168 [0.131, 0.203] | 0.153 [0.123, 0.181] | 0.372 [0.280, 0.456] | 0.282 [0.227, 0.335] |
| **ULRSSM [faces_xyz] per coppia, a CS_ref (SR)** | 0.190 [0.148, 0.229] | 0.160 [0.128, 0.187] | 0.379 [0.298, 0.460] | 0.292 [0.240, 0.339] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.213 [0.162, 0.264] | 0.171 [0.126, 0.217] | 0.359 [0.265, 0.445] | 0.243 [0.191, 0.298] |
| **ULRSSM [faces_xyz] via template, GT SR (d_P) (SR)** | 0.234 [0.182, 0.285] | 0.184 [0.142, 0.226] | 0.376 [0.285, 0.460] | 0.250 [0.201, 0.299] |

Delta appaiati, factorized s1234, d_F cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.019 [-0.083, +0.050] | -0.028 [-0.092, +0.040] | -0.080 [-0.165, +0.011] | -0.040 [-0.109, +0.030] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.021 [-0.097, +0.059] | +0.010 [-0.066, +0.089] | -0.038 [-0.145, +0.072] | +0.009 [-0.066, +0.091] |
| NICP per coppia in mm | +0.027 [-0.053, +0.103] | +0.075 [-0.006, +0.154] | - | - |
| NICP per coppia cs | +0.022 [-0.060, +0.101] | +0.067 [-0.017, +0.149] | - | - |
| NICP su template in mm | +0.128 [+0.049, +0.208] | +0.107 [+0.018, +0.192] | +0.061 [-0.034, +0.163] | +0.090 [-0.005, +0.185] |
| NICP su template cs | +0.130 [+0.036, +0.215] | +0.157 [+0.065, +0.249] | +0.096 [-0.014, +0.210] | +0.157 [+0.057, +0.259] |
| varifold in mm (massa unitaria) | +0.015 [-0.045, +0.082] | +0.036 [-0.020, +0.098] | -0.071 [-0.172, +0.030] | +0.018 [-0.041, +0.080] |
| GNM (visto) preregistrato, coefficienti | +0.037 [-0.064, +0.148] | +0.066 [-0.029, +0.164] | +0.061 [-0.074, +0.198] | +0.081 [-0.015, +0.181] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.036 [-0.046, +0.128] | +0.061 [-0.025, +0.156] | +0.066 [-0.037, +0.176] | +0.076 [-0.012, +0.174] |
| GNM (visto) vA, coefficienti | +0.030 [-0.068, +0.142] | +0.062 [-0.035, +0.162] | +0.053 [-0.080, +0.188] | +0.078 [-0.021, +0.183] |
| GNM (visto) vA, mesh d'identita' FR | +0.034 [-0.050, +0.128] | +0.058 [-0.028, +0.154] | +0.064 [-0.040, +0.175] | +0.073 [-0.015, +0.173] |
| GNM (visto) vB, coefficienti | +0.032 [-0.061, +0.133] | +0.061 [-0.029, +0.160] | +0.059 [-0.071, +0.187] | +0.078 [-0.017, +0.181] |
| GNM (visto) vB, mesh d'identita' FR | -0.028 [-0.097, +0.046] | -0.004 [-0.074, +0.075] | +0.002 [-0.087, +0.102] | +0.011 [-0.060, +0.090] |
| GNM (visto) vB, composizione S_B + k d_P | -0.073 [-0.143, +0.008] | -0.049 [-0.117, +0.030] | -0.037 [-0.130, +0.070] | -0.032 [-0.102, +0.050] |
| FLAME 2023 Open preregistrato, coefficienti | +0.091 [-0.006, +0.189] | +0.114 [+0.023, +0.206] | +0.124 [-0.011, +0.249] | +0.127 [+0.032, +0.222] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.059 [-0.026, +0.147] | +0.079 [-0.004, +0.165] | +0.090 [-0.021, +0.206] | +0.091 [+0.004, +0.181] |
| FLAME 2023 Open vA, coefficienti | +0.091 [-0.006, +0.189] | +0.114 [+0.023, +0.206] | +0.124 [-0.011, +0.249] | +0.127 [+0.032, +0.222] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.059 [-0.026, +0.147] | +0.079 [-0.004, +0.165] | +0.090 [-0.021, +0.206] | +0.091 [+0.004, +0.181] |
| FLAME 2023 Open vB, coefficienti | +0.074 [-0.029, +0.183] | +0.090 [-0.005, +0.186] | +0.125 [-0.028, +0.277] | +0.108 [+0.016, +0.206] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.007 [-0.072, +0.095] | +0.027 [-0.053, +0.115] | +0.042 [-0.070, +0.152] | +0.039 [-0.041, +0.132] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.031 [-0.116, +0.061] | -0.011 [-0.091, +0.076] | +0.008 [-0.108, +0.128] | +0.003 [-0.083, +0.095] |
| ArcFace-render, normal map 3 viste (G0) | +0.092 [+0.000, +0.181] | +0.147 [+0.068, +0.226] | +0.083 [-0.051, +0.200] | +0.152 [+0.064, +0.234] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.091 [-0.011, +0.187] | +0.153 [+0.063, +0.236] | +0.076 [-0.067, +0.213] | +0.157 [+0.061, +0.248] |
| ShapeDNA k=50 | +0.195 [+0.104, +0.284] | +0.224 [+0.148, +0.297] | +0.188 [+0.034, +0.325] | +0.176 [+0.083, +0.266] |
| HKS globale | +0.211 [+0.122, +0.295] | +0.249 [+0.174, +0.324] | +0.219 [+0.075, +0.359] | +0.236 [+0.149, +0.327] |
| WKS globale | +0.187 [+0.106, +0.268] | +0.240 [+0.165, +0.312] | +0.142 [+0.011, +0.271] | +0.223 [+0.138, +0.305] |
| taglia stimata dalla mesh (non oracolo) | +0.243 [+0.161, +0.320] | +0.207 [+0.133, +0.274] | +0.165 [+0.058, +0.270] | +0.173 [+0.080, +0.255] |
| taglia oracolo (S di FR) | +0.043 [-0.075, +0.158] | +0.075 [-0.045, +0.188] | +0.076 [-0.051, +0.204] | +0.093 [-0.025, +0.208] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.170 [+0.084, +0.250] | +0.226 [+0.149, +0.300] | +0.135 [-0.008, +0.271] | +0.210 [+0.123, +0.292] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.151 [+0.072, +0.233] | +0.208 [+0.136, +0.281] | +0.116 [-0.017, +0.254] | +0.176 [+0.091, +0.257] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.139 [+0.058, +0.220] | +0.209 [+0.134, +0.278] | +0.062 [-0.077, +0.204] | +0.165 [+0.082, +0.247] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.167 [+0.092, +0.236] | +0.136 [+0.075, +0.198] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.129 [+0.048, +0.211] | +0.154 [+0.084, +0.220] | +0.059 [-0.055, +0.168] | +0.090 [+0.012, +0.170] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.088 [-0.153, -0.015] | -0.053 [-0.117, +0.018] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.049 [-0.128, +0.027] | -0.015 [-0.086, +0.061] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.065 [-0.152, +0.026] | -0.023 [-0.105, +0.062] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.060 [-0.034, +0.156] | +0.086 [-0.005, +0.173] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.081 [+0.014, +0.153] | +0.128 [+0.064, +0.194] | -0.098 [-0.205, +0.015] | +0.010 [-0.058, +0.081] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.033 [-0.036, +0.108] | +0.107 [+0.043, +0.171] | -0.075 [-0.191, +0.040] | +0.053 [-0.018, +0.125] |

Delta appaiati, factorized s1234, d_F non cal. - concorrente, GT FR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | -0.007 [-0.077, +0.067] | -0.026 [-0.098, +0.046] | -0.077 [-0.165, +0.022] | -0.043 [-0.118, +0.030] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.008 [-0.082, +0.069] | +0.011 [-0.062, +0.083] | -0.036 [-0.139, +0.074] | +0.007 [-0.068, +0.082] |
| NICP per coppia in mm | +0.039 [-0.032, +0.117] | +0.076 [+0.003, +0.151] | - | - |
| NICP per coppia cs | +0.034 [-0.039, +0.107] | +0.068 [-0.006, +0.141] | - | - |
| NICP su template in mm | +0.140 [+0.046, +0.226] | +0.108 [+0.007, +0.208] | +0.063 [-0.045, +0.174] | +0.087 [-0.022, +0.196] |
| NICP su template cs | +0.142 [+0.054, +0.225] | +0.159 [+0.072, +0.246] | +0.098 [-0.009, +0.212] | +0.154 [+0.064, +0.253] |
| varifold in mm (massa unitaria) | +0.027 [-0.039, +0.097] | +0.038 [-0.022, +0.099] | -0.068 [-0.164, +0.031] | +0.015 [-0.044, +0.079] |
| GNM (visto) preregistrato, coefficienti | +0.049 [-0.058, +0.163] | +0.067 [-0.028, +0.168] | +0.064 [-0.071, +0.201] | +0.079 [-0.017, +0.181] |
| GNM (visto) preregistrato, mesh d'identita' FR | +0.049 [-0.046, +0.151] | +0.062 [-0.037, +0.163] | +0.068 [-0.046, +0.185] | +0.073 [-0.027, +0.179] |
| GNM (visto) vA, coefficienti | +0.043 [-0.063, +0.158] | +0.063 [-0.031, +0.164] | +0.056 [-0.076, +0.192] | +0.075 [-0.021, +0.177] |
| GNM (visto) vA, mesh d'identita' FR | +0.046 [-0.050, +0.149] | +0.059 [-0.040, +0.161] | +0.067 [-0.049, +0.185] | +0.070 [-0.031, +0.177] |
| GNM (visto) vB, coefficienti | +0.045 [-0.039, +0.138] | +0.062 [-0.015, +0.145] | +0.062 [-0.061, +0.183] | +0.076 [-0.003, +0.163] |
| GNM (visto) vB, mesh d'identita' FR | -0.016 [-0.093, +0.068] | -0.003 [-0.082, +0.077] | +0.004 [-0.098, +0.109] | +0.009 [-0.072, +0.091] |
| GNM (visto) vB, composizione S_B + k d_P | -0.060 [-0.133, +0.021] | -0.048 [-0.120, +0.026] | -0.034 [-0.135, +0.070] | -0.035 [-0.110, +0.043] |
| FLAME 2023 Open preregistrato, coefficienti | +0.103 [+0.007, +0.201] | +0.115 [+0.029, +0.204] | +0.126 [-0.002, +0.254] | +0.124 [+0.035, +0.216] |
| FLAME 2023 Open preregistrato, mesh d'identita' FR | +0.071 [-0.023, +0.166] | +0.080 [-0.010, +0.168] | +0.093 [-0.023, +0.215] | +0.089 [-0.003, +0.178] |
| FLAME 2023 Open vA, coefficienti | +0.103 [+0.007, +0.201] | +0.115 [+0.029, +0.204] | +0.126 [-0.002, +0.254] | +0.124 [+0.035, +0.216] |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.071 [-0.023, +0.166] | +0.080 [-0.010, +0.168] | +0.093 [-0.023, +0.215] | +0.089 [-0.003, +0.178] |
| FLAME 2023 Open vB, coefficienti | +0.086 [-0.010, +0.188] | +0.091 [+0.009, +0.176] | +0.128 [-0.017, +0.267] | +0.106 [+0.024, +0.192] |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.020 [-0.069, +0.113] | +0.028 [-0.055, +0.114] | +0.044 [-0.070, +0.158] | +0.037 [-0.050, +0.126] |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.018 [-0.102, +0.074] | -0.010 [-0.090, +0.074] | +0.011 [-0.106, +0.128] | +0.000 [-0.080, +0.087] |
| ArcFace-render, normal map 3 viste (G0) | +0.105 [+0.015, +0.195] | +0.148 [+0.068, +0.223] | +0.086 [-0.037, +0.197] | +0.150 [+0.068, +0.226] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.104 [+0.007, +0.198] | +0.154 [+0.068, +0.237] | +0.078 [-0.052, +0.206] | +0.154 [+0.067, +0.238] |
| ShapeDNA k=50 | +0.207 [+0.122, +0.292] | +0.226 [+0.152, +0.294] | +0.191 [+0.038, +0.322] | +0.173 [+0.088, +0.259] |
| HKS globale | +0.223 [+0.135, +0.311] | +0.250 [+0.174, +0.322] | +0.222 [+0.070, +0.366] | +0.233 [+0.144, +0.317] |
| WKS globale | +0.199 [+0.115, +0.282] | +0.241 [+0.167, +0.310] | +0.145 [+0.019, +0.268] | +0.221 [+0.138, +0.303] |
| taglia stimata dalla mesh (non oracolo) | +0.256 [+0.170, +0.333] | +0.209 [+0.119, +0.288] | +0.167 [+0.049, +0.288] | +0.171 [+0.063, +0.268] |
| taglia oracolo (S di FR) | +0.056 [-0.073, +0.175] | +0.076 [-0.060, +0.199] | +0.079 [-0.057, +0.209] | +0.091 [-0.045, +0.212] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], mm (FR) | +0.182 [+0.097, +0.262] | +0.228 [+0.153, +0.298] | +0.137 [-0.005, +0.267] | +0.208 [+0.123, +0.284] |
| DINOv2 ViT-S/14 su render [profondita'], mm (FR) | +0.163 [+0.084, +0.242] | +0.209 [+0.140, +0.279] | +0.118 [-0.007, +0.245] | +0.174 [+0.095, +0.246] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], mm (FR) | +0.151 [+0.068, +0.230] | +0.210 [+0.136, +0.276] | +0.065 [-0.071, +0.199] | +0.163 [+0.083, +0.239] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], mm (FR) | +0.179 [+0.102, +0.252] | +0.138 [+0.067, +0.202] | - | - |
| LPIPS alex su render [profondita'], mm (FR) | +0.141 [+0.058, +0.222] | +0.155 [+0.085, +0.226] | +0.062 [-0.057, +0.173] | +0.087 [+0.005, +0.170] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.076 [-0.145, -0.002] | -0.052 [-0.119, +0.017] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.037 [-0.109, +0.035] | -0.014 [-0.080, +0.050] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.053 [-0.142, +0.032] | -0.021 [-0.097, +0.052] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.072 [-0.016, +0.163] | +0.088 [+0.001, +0.170] | - | - |
| ULRSSM [faces_xyz] per coppia, mm (FR) | +0.094 [+0.025, +0.168] | +0.129 [+0.065, +0.192] | -0.095 [-0.202, +0.023] | +0.008 [-0.061, +0.075] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | +0.046 [-0.020, +0.117] | +0.108 [+0.045, +0.170] | -0.072 [-0.179, +0.041] | +0.051 [-0.018, +0.120] |

Delta appaiati, factorized s1234, d_P - concorrente, GT SR (tutti i riferimenti in `deltas.csv`):

| concorrente | crop | all_cross | stessa_topologia crop | stessa_topologia tutte |
| --- | --- | --- | --- | --- |
| ICP + Chamfer in mm (per coppia) | +0.001 [-0.070, +0.073] | -0.014 [-0.082, +0.058] | -0.074 [-0.168, +0.032] | -0.027 [-0.100, +0.049] |
| ICP + Chamfer cs (per coppia, scala tolta) | -0.036 [-0.106, +0.042] | -0.016 [-0.085, +0.053] | -0.074 [-0.175, +0.034] | -0.020 [-0.090, +0.054] |
| NICP per coppia in mm | +0.004 [-0.068, +0.081] | +0.042 [-0.034, +0.114] | - | - |
| NICP per coppia cs | -0.006 [-0.076, +0.063] | +0.032 [-0.037, +0.102] | - | - |
| NICP su template in mm | +0.167 [+0.074, +0.249] | +0.149 [+0.049, +0.238] | +0.103 [-0.013, +0.215] | +0.135 [+0.025, +0.236] |
| NICP su template cs | +0.130 [+0.044, +0.212] | +0.139 [+0.052, +0.224] | +0.085 [-0.029, +0.195] | +0.135 [+0.043, +0.233] |
| varifold in mm (massa unitaria) | +0.024 [-0.038, +0.092] | +0.035 [-0.019, +0.091] | -0.070 [-0.162, +0.034] | +0.017 [-0.038, +0.076] |
| GNM (visto) preregistrato, coefficienti | +0.074 [-0.027, +0.180] | +0.093 [+0.003, +0.188] | +0.083 [-0.051, +0.219] | +0.105 [+0.015, +0.202] |
| GNM (visto) preregistrato, mesh d'identita' SR | +0.049 [-0.041, +0.147] | +0.067 [-0.022, +0.165] | +0.061 [-0.061, +0.187] | +0.078 [-0.012, +0.181] |
| GNM (visto) vA, coefficienti | +0.069 [-0.031, +0.176] | +0.090 [+0.001, +0.188] | +0.076 [-0.057, +0.210] | +0.103 [+0.014, +0.201] |
| GNM (visto) vA, mesh d'identita' SR | +0.052 [-0.039, +0.153] | +0.071 [-0.019, +0.170] | +0.065 [-0.060, +0.187] | +0.083 [-0.008, +0.183] |
| GNM (visto) vB, coefficienti | +0.027 [-0.066, +0.128] | +0.051 [-0.034, +0.144] | +0.030 [-0.098, +0.167] | +0.066 [-0.021, +0.161] |
| GNM (visto) vB, mesh d'identita' SR | -0.065 [-0.141, +0.021] | -0.050 [-0.119, +0.028] | -0.045 [-0.155, +0.071] | -0.034 [-0.106, +0.043] |
| FLAME 2023 Open preregistrato, coefficienti | +0.110 [+0.019, +0.211] | +0.125 [+0.038, +0.211] | +0.127 [-0.005, +0.251] | +0.135 [+0.046, +0.224] |
| FLAME 2023 Open preregistrato, mesh d'identita' SR | +0.099 [-0.001, +0.209] | +0.119 [+0.027, +0.221] | +0.108 [-0.024, +0.236] | +0.129 [+0.035, +0.234] |
| FLAME 2023 Open vA, coefficienti | +0.110 [+0.019, +0.211] | +0.125 [+0.038, +0.211] | +0.127 [-0.005, +0.251] | +0.135 [+0.046, +0.224] |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.099 [-0.001, +0.209] | +0.119 [+0.027, +0.221] | +0.108 [-0.024, +0.236] | +0.129 [+0.035, +0.234] |
| FLAME 2023 Open vB, coefficienti | +0.073 [-0.031, +0.179] | +0.086 [-0.001, +0.179] | +0.099 [-0.052, +0.252] | +0.101 [+0.014, +0.193] |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.027 [-0.117, +0.066] | -0.014 [-0.099, +0.077] | -0.005 [-0.129, +0.116] | -0.001 [-0.086, +0.088] |
| ArcFace-render, normal map 3 viste (G0) | +0.078 [-0.011, +0.169] | +0.125 [+0.046, +0.203] | +0.048 [-0.079, +0.164] | +0.126 [+0.045, +0.204] |
| ArcFace-render, ombreggiato 3 viste (E12) | +0.072 [-0.019, +0.162] | +0.129 [+0.045, +0.208] | +0.030 [-0.099, +0.154] | +0.124 [+0.035, +0.209] |
| ShapeDNA k=50 | +0.183 [+0.100, +0.268] | +0.211 [+0.138, +0.281] | +0.155 [+0.014, +0.284] | +0.163 [+0.072, +0.251] |
| HKS globale | +0.211 [+0.125, +0.297] | +0.241 [+0.166, +0.310] | +0.203 [+0.057, +0.334] | +0.228 [+0.136, +0.310] |
| WKS globale | +0.189 [+0.109, +0.267] | +0.230 [+0.157, +0.298] | +0.146 [+0.020, +0.260] | +0.217 [+0.134, +0.300] |
| taglia stimata dalla mesh (non oracolo) | +0.252 [+0.167, +0.328] | +0.228 [+0.147, +0.298] | +0.187 [+0.066, +0.303] | +0.205 [+0.106, +0.297] |
| taglia oracolo (S di FR) | +0.233 [+0.122, +0.339] | +0.255 [+0.136, +0.363] | +0.248 [+0.128, +0.367] | +0.270 [+0.149, +0.376] |
| CLIP ViT-B/32 su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.150 [+0.066, +0.231] | +0.208 [+0.136, +0.276] | +0.061 [-0.077, +0.199] | +0.179 [+0.094, +0.255] |
| DINOv2 ViT-S/14 su render [profondita'], a CS_ref (SR) | +0.140 [+0.063, +0.219] | +0.191 [+0.123, +0.262] | +0.064 [-0.057, +0.202] | +0.153 [+0.078, +0.228] |
| DreamSim ensemble su render [ombreggiato con normali per vertice], a CS_ref (SR) | +0.135 [+0.055, +0.214] | +0.198 [+0.127, +0.263] | +0.040 [-0.101, +0.184] | +0.146 [+0.068, +0.221] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2], a CS_ref (SR) | +0.062 [-0.011, +0.134] | +0.094 [+0.030, +0.160] | - | - |
| LPIPS alex su render [profondita'], a CS_ref (SR) | +0.129 [+0.038, +0.218] | +0.163 [+0.094, +0.228] | +0.057 [-0.071, +0.178] | +0.108 [+0.030, +0.182] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.062 [-0.134, +0.016] | -0.037 [-0.107, +0.034] | - | - |
| NoW originale (Procrustes + ICP con scala), mm (media dei due versi) | -0.061 [-0.131, +0.010] | -0.034 [-0.100, +0.030] | - | - |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.080 [-0.163, +0.000] | -0.045 [-0.112, +0.022] | - | - |
| Stirling/ESRC (Procrustes con scala, RMSE), mm (media dei due versi) | +0.048 [-0.040, +0.136] | +0.074 [-0.009, +0.154] | - | - |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | +0.061 [-0.012, +0.133] | +0.114 [+0.051, +0.177] | -0.112 [-0.219, +0.003] | -0.004 [-0.068, +0.063] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | +0.017 [-0.048, +0.086] | +0.089 [+0.026, +0.145] | -0.109 [-0.226, +0.011] | +0.038 [-0.030, +0.100] |

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
| **C3M e205, d_F cal.*** | 0.299 [0.229, 0.364] | 0.324 [0.255, 0.387] | 0.378 [0.288, 0.462] | 0.395 [0.312, 0.470] |
| **C3M e205, d_F non cal.** | 0.345 [0.262, 0.420] | 0.353 [0.269, 0.429] | 0.382 [0.286, 0.468] | 0.393 [0.309, 0.472] |
| C3M e205, d_P | 0.345 [0.258, 0.419] | 0.357 [0.272, 0.434] | 0.380 [0.280, 0.467] | 0.394 [0.304, 0.474] |
| **ICP + Chamfer in mm (per coppia)** | 0.317 [0.248, 0.379] | 0.373 [0.297, 0.442] | 0.444 [0.363, 0.515] | 0.419 [0.337, 0.492] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.333 [0.262, 0.399] | 0.348 [0.274, 0.417] | 0.411 [0.335, 0.484] | 0.382 [0.302, 0.455] |
| **NICP per coppia in mm** | 0.295 [0.214, 0.371] | 0.283 [0.205, 0.362] | - | - |
| **NICP per coppia cs** | 0.290 [0.209, 0.368] | 0.287 [0.202, 0.368] | - | - |
| **NICP su template in mm** | 0.162 [0.098, 0.225] | 0.195 [0.112, 0.278] | 0.253 [0.158, 0.349] | 0.228 [0.129, 0.325] |
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
| C3M e205, d_F cal.* | 0.297 [0.225, 0.364] | 0.318 [0.246, 0.385] | 0.380 [0.289, 0.467] | 0.387 [0.303, 0.463] |
| C3M e205, d_F non cal. | 0.367 [0.286, 0.440] | 0.369 [0.289, 0.444] | 0.416 [0.331, 0.490] | 0.411 [0.328, 0.487] |
| **C3M e205, d_P** | 0.372 [0.294, 0.442] | 0.378 [0.304, 0.448] | 0.419 [0.338, 0.498] | 0.417 [0.339, 0.490] |
| **ICP + Chamfer in mm (per coppia)** | 0.299 [0.233, 0.363] | 0.351 [0.274, 0.420] | 0.421 [0.337, 0.493] | 0.393 [0.310, 0.467] |
| **ICP + Chamfer cs (per coppia, scala tolta)** | 0.356 [0.286, 0.420] | 0.370 [0.301, 0.435] | 0.434 [0.359, 0.502] | 0.405 [0.329, 0.472] |
| **NICP per coppia in mm** | 0.323 [0.245, 0.400] | 0.316 [0.237, 0.393] | - | - |
| **NICP per coppia cs** | 0.322 [0.243, 0.397] | 0.322 [0.242, 0.398] | - | - |
| **NICP su template in mm** | 0.122 [0.062, 0.181] | 0.137 [0.059, 0.212] | 0.190 [0.098, 0.288] | 0.161 [0.068, 0.254] |
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
| famos | SR | nocrop | 105 | C3M e205, d_P | 0.756 [0.512, 0.889] | FLAME 2023 Open vB, mesh d'identita' SR | 0.791 [0.626, 0.889] | -0.035 [-0.235, +0.116] |

Scala dell'allineamento, FaMoS (tutti i passi, principali e secondari):

GT FR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | +0.138 [-0.125, +0.418] | - | - | - | - | - | - |
| famos | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | -0.176 [-0.433, +0.055] | - | - | - | - | - | - |
| famos | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | +0.018 [-0.201, +0.248] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [GNM (visto) vB, mesh d'identita' FR] | -0.054 [-0.232, +0.095] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh FR) - ICP mm [FLAME 2023 Open vB, mesh d'identita' FR] | -0.071 [-0.235, +0.093] | - | - | - | - | - | - |
| famos | 5. nostro (d_F non cal.) - ICP mm [factorized s1234, d_F non cal.] | -0.123 [-0.312, +0.101] | - | - | - | - | - | - |
| famos | 5. nostro (d_F non cal.) - ICP mm [factorized s2345, d_F non cal.] | -0.127 [-0.309, +0.049] | - | - | - | - | - | - |

GT SR:

| dominio | passo | nocrop | con_perturbazione original | con_perturbazione remesh | con_perturbazione down8k | con_perturbazione up60k | con_perturbazione noisy | stessa_topologia senza_crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| famos | 1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs | -0.063 [-0.283, +0.161] | - | - | - | - | - | - |
| famos | 1b. togliere la taglia per coppia (scala del Procrustes + ICP): NoW originale - NoW metrical | +0.122 [-0.086, +0.347] | - | - | - | - | - | - |
| famos | 2. deformazione non rigida per coppia, cs: ICP cs - NICP cs | -0.164 [-0.368, +0.026] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [GNM (visto) vB, mesh d'identita' SR] | +0.155 [-0.048, +0.452] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - ICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.179 [-0.026, +0.477] | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP cs [factorized s1234, d_P] | +0.129 [-0.091, +0.417] | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - ICP cs [factorized s2345, d_P] | +0.117 [-0.084, +0.361] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [GNM (visto) vB, mesh d'identita' SR] | -0.009 [-0.187, +0.209] | - | - | - | - | - | - |
| famos | 4. fit 3DMM vB per mesh (mesh SR) - NICP cs [FLAME 2023 Open vB, mesh d'identita' SR] | +0.015 [-0.130, +0.213] | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - NICP cs [factorized s1234, d_P] | -0.035 [-0.207, +0.186] | - | - | - | - | - | - |
| famos | 5. nostro (d_P) - NICP cs [factorized s2345, d_P] | -0.047 [-0.208, +0.135] | - | - | - | - | - | - |

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
     "plugin:now_metrical/lmk_det|mm": "lmk_det",
     "plugin:now_metrical/lmk_s1|mm": "non dichiarata",
     "plugin:now_metrical/lmk_s2|mm": "non dichiarata",
     "plugin:now_metrical/lmk_s3|mm": "non dichiarata",
     "plugin:now_metrical/lmk_s5|mm": "non dichiarata",
     "plugin:now_original|mm": "dichiarata",
     "plugin:now_original/lmk_det|mm": "lmk_det",
     "plugin:now_original/lmk_s1|mm": "non dichiarata",
     "plugin:now_original/lmk_s2|mm": "non dichiarata",
     "plugin:now_original/lmk_s3|mm": "non dichiarata",
     "plugin:now_original/lmk_s5|mm": "non dichiarata",
     "plugin:realy|mm": "dichiarata",
     "plugin:realy/lmk_det|mm": "lmk_det",
     "plugin:realy/lmk_s1|mm": "non dichiarata",
     "plugin:realy/lmk_s2|mm": "non dichiarata",
     "plugin:realy/lmk_s3|mm": "non dichiarata",
     "plugin:realy/lmk_s5|mm": "non dichiarata",
     "plugin:stirling|mm": "dichiarata",
     "plugin:stirling/lmk_det|mm": "lmk_det",
     "plugin:stirling/lmk_s1|mm": "non dichiarata",
     "plugin:stirling/lmk_s2|mm": "non dichiarata",
     "plugin:stirling/lmk_s3|mm": "non dichiarata",
     "plugin:stirling/lmk_s5|mm": "non dichiarata",
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
     "plugin:gw/euclid_256_cgw2|sr",
     "plugin:ulrssm/zeroshot_faust|pair_fr",
     "plugin:ulrssm/zeroshot_faust|pair_sr"
    ],
    "righe_proprie": {
     "righe proprie (11803 righe a topologie diverse)": {
      "columns": [
       "plugin:now_metrical/lmk_det|mm",
       "plugin:now_metrical/lmk_s1|mm",
       "plugin:now_metrical/lmk_s2|mm",
       "plugin:now_metrical/lmk_s3|mm",
       "plugin:now_metrical/lmk_s5|mm",
       "plugin:now_original/lmk_det|mm",
       "plugin:now_original/lmk_s1|mm",
       "plugin:now_original/lmk_s2|mm",
       "plugin:now_original/lmk_s3|mm",
       "plugin:now_original/lmk_s5|mm",
       "plugin:realy/lmk_det|mm",
       "plugin:realy/lmk_s1|mm",
       "plugin:realy/lmk_s2|mm",
       "plugin:realy/lmk_s3|mm",
       "plugin:realy/lmk_s5|mm",
       "plugin:stirling/lmk_det|mm",
       "plugin:stirling/lmk_s1|mm",
       "plugin:stirling/lmk_s2|mm",
       "plugin:stirling/lmk_s3|mm",
       "plugin:stirling/lmk_s5|mm"
      ],
      "exact_landmarks_same_rows": [
       "plugin:now_metrical|mm",
       "plugin:now_original|mm",
       "plugin:realy|mm",
       "plugin:stirling|mm"
      ],
      "rows_cross": 11803,
      "rows_same": 0,
      "rows_lost_cross_vs_mask": 135727,
      "rows_lost_same_vs_mask": 29700
     }
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
   "n_methods": 68,
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
   "plugins_partial": [
    "plugin:now_metrical/lmk_det|mm",
    "plugin:now_metrical/lmk_s1|mm",
    "plugin:now_metrical/lmk_s2|mm",
    "plugin:now_metrical/lmk_s3|mm",
    "plugin:now_metrical/lmk_s5|mm",
    "plugin:now_original/lmk_det|mm",
    "plugin:now_original/lmk_s1|mm",
    "plugin:now_original/lmk_s2|mm",
    "plugin:now_original/lmk_s3|mm",
    "plugin:now_original/lmk_s5|mm",
    "plugin:realy/lmk_det|mm",
    "plugin:realy/lmk_s1|mm",
    "plugin:realy/lmk_s2|mm",
    "plugin:realy/lmk_s3|mm",
    "plugin:realy/lmk_s5|mm",
    "plugin:stirling/lmk_det|mm",
    "plugin:stirling/lmk_s1|mm",
    "plugin:stirling/lmk_s2|mm",
    "plugin:stirling/lmk_s3|mm",
    "plugin:stirling/lmk_s5|mm"
   ],
   "seconds": 699.0758872032166
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
     "plugin:now_metrical/lmk_det|mm": "lmk_det",
     "plugin:now_metrical/lmk_s1|mm": "non dichiarata",
     "plugin:now_metrical/lmk_s2|mm": "non dichiarata",
     "plugin:now_metrical/lmk_s3|mm": "non dichiarata",
     "plugin:now_metrical/lmk_s5|mm": "non dichiarata",
     "plugin:now_original|mm": "dichiarata",
     "plugin:now_original/lmk_det|mm": "lmk_det",
     "plugin:now_original/lmk_s1|mm": "non dichiarata",
     "plugin:now_original/lmk_s2|mm": "non dichiarata",
     "plugin:now_original/lmk_s3|mm": "non dichiarata",
     "plugin:now_original/lmk_s5|mm": "non dichiarata",
     "plugin:realy|mm": "dichiarata",
     "plugin:realy/lmk_det|mm": "lmk_det",
     "plugin:realy/lmk_s1|mm": "non dichiarata",
     "plugin:realy/lmk_s2|mm": "non dichiarata",
     "plugin:realy/lmk_s3|mm": "non dichiarata",
     "plugin:realy/lmk_s5|mm": "non dichiarata",
     "plugin:stirling|mm": "dichiarata",
     "plugin:stirling/lmk_det|mm": "lmk_det",
     "plugin:stirling/lmk_s1|mm": "non dichiarata",
     "plugin:stirling/lmk_s2|mm": "non dichiarata",
     "plugin:stirling/lmk_s3|mm": "non dichiarata",
     "plugin:stirling/lmk_s5|mm": "non dichiarata",
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
     "plugin:gw/euclid_256_cgw2|sr",
     "plugin:ulrssm/zeroshot_faust|pair_fr",
     "plugin:ulrssm/zeroshot_faust|pair_sr"
    ],
    "righe_proprie": {
     "righe proprie (10851 righe a topologie diverse)": {
      "columns": [
       "plugin:now_metrical/lmk_det|mm",
       "plugin:now_metrical/lmk_s1|mm",
       "plugin:now_metrical/lmk_s2|mm",
       "plugin:now_metrical/lmk_s3|mm",
       "plugin:now_metrical/lmk_s5|mm",
       "plugin:now_original/lmk_det|mm",
       "plugin:now_original/lmk_s1|mm",
       "plugin:now_original/lmk_s2|mm",
       "plugin:now_original/lmk_s3|mm",
       "plugin:now_original/lmk_s5|mm",
       "plugin:realy/lmk_det|mm",
       "plugin:realy/lmk_s1|mm",
       "plugin:realy/lmk_s2|mm",
       "plugin:realy/lmk_s3|mm",
       "plugin:realy/lmk_s5|mm",
       "plugin:stirling/lmk_det|mm",
       "plugin:stirling/lmk_s1|mm",
       "plugin:stirling/lmk_s2|mm",
       "plugin:stirling/lmk_s3|mm",
       "plugin:stirling/lmk_s5|mm"
      ],
      "exact_landmarks_same_rows": [
       "plugin:now_metrical|mm",
       "plugin:now_original|mm",
       "plugin:realy|mm",
       "plugin:stirling|mm"
      ],
      "rows_cross": 10851,
      "rows_same": 0,
      "rows_lost_cross_vs_mask": 124804,
      "rows_lost_same_vs_mask": 29700
     }
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
   "plugins_partial": [
    "plugin:now_metrical/lmk_det|mm",
    "plugin:now_metrical/lmk_s1|mm",
    "plugin:now_metrical/lmk_s2|mm",
    "plugin:now_metrical/lmk_s3|mm",
    "plugin:now_metrical/lmk_s5|mm",
    "plugin:now_original/lmk_det|mm",
    "plugin:now_original/lmk_s1|mm",
    "plugin:now_original/lmk_s2|mm",
    "plugin:now_original/lmk_s3|mm",
    "plugin:now_original/lmk_s5|mm",
    "plugin:realy/lmk_det|mm",
    "plugin:realy/lmk_s1|mm",
    "plugin:realy/lmk_s2|mm",
    "plugin:realy/lmk_s3|mm",
    "plugin:realy/lmk_s5|mm",
    "plugin:stirling/lmk_det|mm",
    "plugin:stirling/lmk_s1|mm",
    "plugin:stirling/lmk_s2|mm",
    "plugin:stirling/lmk_s3|mm",
    "plugin:stirling/lmk_s5|mm"
   ],
   "seconds": 665.0183401107788
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
     "plugin:gw/euclid_256_cgw2|sr",
     "plugin:ulrssm/zeroshot_faust|pair_fr",
     "plugin:ulrssm/zeroshot_faust|pair_sr"
    ],
    "righe_proprie": {}
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
   "n_methods": 66,
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
   "seconds": 518.2594723701477
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
   "n_methods": 61,
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
   "seconds": 291.16531324386597
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
   "seconds": 11.734086275100708
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

