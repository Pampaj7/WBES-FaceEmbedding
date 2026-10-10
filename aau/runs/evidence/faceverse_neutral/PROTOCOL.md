# FaceVerse neutra contro FaceVerse con espressioni: protocollo (scritto il 10 ottobre 2026, PRIMA di ogni calcolo)

Nessun numero della vista neutra esiste a quest'ora. Codice in `v3_work/faceverse_neutral/` (piu' un passo `fvn` in
`v3_work/trainer/ablations/c3f/eval_body.sh` e la vista `faceverse_neutral` in `aau/baselines_mm/blmm.py`), numeri
qui. Hash di questo file in `PROTOCOL.sha256`.

## 0. Domanda

Su FaceVerse con espressioni (`trainer_v3/factorized_results.md`, sezione FaceVerse) la graduata sta intorno a 0.3
per tutti i metodi (bracci 0.26-0.34 con FR, baseline in mm 0.20-0.34, taglia oracolo 0.209), mentre il rank-1 dei
bracci e' 0.93. Il limite viene dalle espressioni o dal dominio FaceVerse (CV della taglia 1.8%, unita' arbitrarie)?

## 1. Cosa si calcola

**Vista neutra.** `datasets/FACEVERSE_ZS/eval_view`: le stesse 500 identita' del pool (`id910000`-`id910499`), le
stesse 6 topologie per identita' (original, remesh, crop, noisy, down8k, up60k), mesh NEUTRE (`FACEVERSE_ZS/topo`).
La vista con espressioni (`expr_view`, `expr_topo`) differisce SOLO per l'espressione casuale di ogni mesh (3-8
blendshape, coefficienti 0.3-1.0, `expr_view/manifest.json`). Stessi nomi di file nelle due viste (verificato:
`diff` degli elenchi vuoto), quindi `zs_stage.select_subjects(seed 1234)` estrae gli stessi 100 soggetti valutati.

Interpretazione di "una mesh neutra per identita'": ogni mesh con espressione e' sostituita dalla mesh neutra della
stessa identita' NELLA STESSA TOPOLOGIA. Cosi' cambia una sola cosa, l'espressione; righe, coppie e topologie
restano quelle della graduata con espressioni. (La variante con una sola mesh per identita', la original, non si
calcola: toglierebbe anche il confronto fra topologie diverse e mescolerebbe due effetti.)

**Righe.** Graduata a livello di coppia di mesh, senza crop, topologie diverse (`nocrop_cross` / `mesh_pair_nocrop`):
coppie di mesh (s_a, t_a), (s_b, t_b) con s_a != s_b, t_a != t_b, nessuna delle due crop; 100 soggetti x 5 topologie,
99.000 righe. Ogni riga porta la GT della sua coppia di soggetti.

**GT (le stesse matrici della vista con espressioni).** FR e SR = `datasets/CANONICAL_GT/eval/faceverse_{fr,sr}.npz`,
maxabs = `FACEVERSE_ZS/eval_view/gt_matrix.npz` (stesso file, stesso sha256 c0ac3252..., di `expr_view/gt_matrix.npz`).
Tutte e tre sono gia' definite sull'identita' neutra (`aau/runs/evidence/e12/protocol.md` righe 18-21), quindi nella
vista neutra la GT descrive esattamente la geometria osservata.

**Metodi.**
- Bracci a 21.096 passi, pesi EMA: factorized s1234/s2345, factorized2 s1234/s2345 (d_F per FR e maxabs, d_P per
  SR, come `fact_summary.py`), ctrlfr s1234/s2345 (||z||); C3M (factorized, epoca 205, d_F / d_P). Embedding [s, u] (o
  z) dalla stessa pipeline del passo `form` di `eval_body.sh` (`eval_v3.py`, `WBES_V3_FACTORIZED_OUT=full`, convenzione
  BFM `_flip`, cache degli operatori), con la tabella di scala della vista neutra (`build_scale_table.py --view-dir
  FACEVERSE_ZS/eval_view/npz --domain faceverse`, scritta qui in `scale_tables/fv_eval.npz`).
- e108 (cieco alla taglia): embedding con `aau/zs3dmm/zs_zeroshot.sbatch` (`WBES_ZS_PART=embed`, `_flip`), la stessa
  pipeline degli embedding con espressioni (`ws_faceverse_expr/.../scale_e108_flip_topology`); distanza euclidea.
- Baseline (`aau/baselines_mm`, codice invariato, nuova vista `faceverse_neutral`): ICP + Chamfer in mm, NICP per
  coppia in mm, NICP su template in mm, NICP per coppia modo cs, taglia stimata (|log CS robusta|, dalla mesh
  osservata), taglia oracolo (|log S| della GT FR). L e CS_ref di FaceVerse restano quelli di `params.json` (gia'
  calcolati sulle original neutre).

**Bootstrap.** Per soggetto, 1000 repliche, seme 1234, `eval_factorized.boot_rows` (peso di riga = prodotto dei
conteggi dei due soggetti), IC 95% percentile. La stessa funzione e lo stesso seme per TUTTI i metodi e per ENTRAMBE
le viste: la tabella neutra contro espressioni si calcola qui con un solo codice. Righe con distanza non finita (NICP
fallito) escluse per quel metodo, contate e riportate.

**Controlli (prima di leggere la neutra).**
1. Vista con espressioni ricalcolata qui: i punti dei bracci e di C3M devono coincidere con `factorized_results.csv` e
   `c3f_eval/form/fv/v3factorizedc3me205/form_spearman.csv` (e anche gli IC: stessa funzione, stesso seme); i punti delle
   baseline e di e108 con `baselines_mm/spearman.csv` (`mesh_pair_nocrop`). Gli IC delle baseline pubblicate usavano
   il seme di E12: qui ricalcolati col seme 1234, la differenza si riporta.
2. Vista neutra: i punti dei bracci devono coincidere con l'uscita di `eval_factorized.py` del passo `fvn`
   (`form/<tag>/form_spearman.csv`).

## 2. Regola di lettura (fissata ora)

GT primaria FR; SR riportata e letta con la stessa regola come secondaria. "Modelli" = i 6 bracci + C3M; "tutti i
metodi" = modelli + e108 + le 6 baseline.

- **DOMINIO**: se nella vista neutra il punto della graduata resta <= 0.35 per tutti i metodi, il limite e' il
  dominio FaceVerse (taglia poco variabile, unita' arbitrarie, geometria), non le espressioni.
- **ESPRESSIONI**: se nella vista neutra il punto sale nettamente, cioe' > 0.5 per tutti i modelli, il limite sono le
  espressioni.
- Altrimenti (qualche metodo sopra 0.35 ma non tutti i modelli sopra 0.5): **INTERMEDIO**, entrambi i fattori
  contano; si riportano le differenze neutra - espressioni per metodo, senza un verdetto netto.

Si riporta anche, per descrizione e non per il verdetto: la taglia oracolo neutra (se resta bassa, FR su FaceVerse
e' quasi tutta forma), e il rank-1 non si ricalcola (non e' la domanda).
