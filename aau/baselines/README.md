# Baseline estese (WS1) — Tabella 2 estesa

Produce i numeri della **Tabella 2 estesa** (`paper/main_short.tex`,
`tab:alignment_effect`) per le baseline chieste dalle review: varifold, currents,
ArcFace, LPIPS, CLIP, DINOv2, con Chamfer raw come riferimento.  Niente qui tocca il
codice di ricerca del repo: gli script di ricerca vengono **importati**, non copiati.

Due passaggi, sempre gli stessi:

1. `*_matrix.py` → per ogni metrica e per ogni **coppia ordinata di topologie** una
   matrice 100×100 in npz, `D[i, j] = distanza(soggetto i in tA, soggetto j in tB)`, con i
   nomi dei soggetti dentro.  Solo la parte `i<j` e' riempita (tutte le metriche sono
   simmetriche nei due argomenti); il resto e' NaN.
2. `rank_from_matrix.py` → Spearman e Pearson vs `D_GT` con CI bootstrap subject-level al
   95%, 1000 repliche.

## Protocollo, e perche' e' identico a quello del paper

- **Quale insieme di 100 soggetti** (`--subject-set`, e' la cosa da non sbagliare):
  - `heldout` (default): i 100 soggetti dello split held-out, letti da
    `paper_artifacts/bootstrap_ci/table1_pairlevel_exact/*/pair_metrics.csv`, cioe' dai
    pair table dell'eval del repo.  Non si rifa' lo split con un seme: non c'e' modo di
    sbagliarlo.  E' il set su cui vanno i numeri nuovi.
  - `facebench_first100`: i primi 100 soggetti in ordine alfabetico (id0000..id0099).
    **E' il set con cui e' stata calcolata la Tabella 2 del paper**:
    `run_facebench_remesh.py` fa `sorted(...)[:max_subjects]` senza passare dallo split.
    Con lo split held-out condivide **solo 21 soggetti su 100**, quindi ~79 dei soggetti
    della Tabella 2 sono soggetti di training.  Serve a riprodurre lo 0.729, e a questo
    punto anche a documentare il problema.
- **`D_GT`**: `intrinsic_utils.load_gt_distance_matrix` sul `normalized_matrix_distances.npz`
  del repo.  Il probe ha verificato che coincide con la colonna `gt_distance` dei pair
  table entro 5e-7 (che e' l'arrotondamento a 6 decimali del CSV).
- **Setting (a) original→original**: 1 coppia di topologie × 4950 coppie di soggetti = 4950 righe.
- **Setting (b) no-crop cross-topology**: 20 coppie ordinate di topologie (le 5 non-crop,
  `tA != tB`) × 4950 = 99000 righe.  Sono gli stessi `n_pairs` di
  `paper_artifacts/bootstrap_ci/bootstrap_ci.csv`.
- **Setting (c) tessellation only** e **(d) perturbation**: la colonna (b) spezzata in due,
  perche' ci stanno dentro due cose diverse.  In (c) le due mesh sono la stessa superficie
  tassellata in modo diverso — `original`, `remesh`, `down8k`, `up60k` — 12 coppie ordinate
  × 4950 = 59400 righe.  In (d) una delle due e' `noisy`, cioe' la superficie e' cambiata:
  8 coppie × 4950 = 39600 righe.  12 + 8 = 20, quindi la decomposizione di (b) e' esatta.
  `crop` non entra in nessuna delle quattro colonne, come non entra in (b): nei due run non
  esiste nessuna matrice con `crop` fuori dalla diagonale, e calcolarle vorrebbe dire
  rifare render, calibrazione ArcFace ed embedding di tutte le topologie.  "Perturbation"
  qui vuol dire percio' `noisy`, e va letto cosi'.
- `n_topology_pairs` sta in ogni riga del csv e in testa al `.tex`: una metrica che copre
  meno coppie delle altre non ha una cella confrontabile con le loro.
- **Bootstrap**: `weighted_bootstrap_spearman` importata da `scripts/compute_bootstrap_ci.py`.
  Non e' riscritta.  Per Pearson quella funzione non esiste: si scambia temporaneamente
  `finite_spearman` dentro il modulo (`_corr_backend`), cosi' il ricampionamento dei
  soggetti resta lo stesso e cambia solo la correlazione.  Il seme dipende da
  (metrica, setting, correlazione), quindi i CI differiscono dal paper nella terza cifra
  per rumore Monte Carlo: quello che deve combaciare e' la **stima puntuale**.

- **Quale Chamfer**: nel repo ce ne sono due, vedi la tabella delle metriche.  Il numero
  della Tabella 2 e' la variante faceBench (media di distanze su 4096 punti campionati),
  non quella dell'eval (media di distanze al quadrato su tutti i vertici).

**Gate di validazione.** `rank_from_matrix.py` confronta da solo Chamfer raw con i numeri
pubblicati (0.729 [0.667, 0.788] e 0.552 [0.488, 0.606]) e stampa OK / FUORI CI.  Con
`--subject-set facebench_first100` e la variante faceBench il gate passa:
0.7295 [0.666, 0.787] e 0.5518 [0.490, 0.607], cioe' +0.0005 e -0.0002 rispetto al paper
(job 1019273/1019274).  Sullo stesso codice, cambiando **solo** l'insieme di soggetti a
`heldout`, Chamfer scende a 0.6436 e 0.4691: e' la misura di quanto la Tabella 2 stia
guardando soggetti di training.

## Metriche

| metrica | codice | dove gira | note |
|---|---|---|---|
| `chamfer` | `faceBench/latentVSpipeline/fg_metrics.symmetric_chamfer` | cpu, 24 core | `0.5 * (mean d(X->Y) + mean d(Y->X))` su 4096 punti campionati; **e' la Chamfer della Tabella 2** |
| `chamfer_sq` | `robustness/eval_utils.symmetric_chamfer_same_shape_batch` | L40S | `mean d^2` nei due versi su tutti i vertici; e' la Chamfer dell'eval del repo e delle celle fuori diagonale della Tabella 1.  **Copertura parziale**: esiste solo su 3 delle 20 coppie no-crop (14850 righe invece di 99000), quindi la sua cella cross e' una media su altre coppie di topologie e non si confronta con le altre righe.  Nel `.tex` sta sotto una riga a se' con il pugnale |
| `varifold`, `currents` | `v2_work/phase0/measure_distances.py` + `geometric_kernel.py` | A40 | `normalize="maxabs"` (dalla seconda revisione: vedi sotto), `sigmas=(0.5, 0.2, 0.1, 0.05)`, **nessun sottocampionamento casuale**: misura quantizzata su griglia |
| `bbox_proxy` | `proxy_matrix.py` | cpu, secondi | **riga di CONTROLLO, non una baseline**: quattro numeri per mesh (centro e diagonale del bbox) calcolati sulle mesh gia' normalizzate maxabs, distanza euclidea fra i due vettori.  Nessuna forma entra nel conto |
| `clip`, `dinov2` | `v2_work/phase0/perceptual_embed.py` | T4 | media degli embedding sulle 3 viste, rinormalizzazione L2, distanza `1 - coseno` |
| `arcface` | `arcface_fixed.py` (riconoscimento buffalo_l, **niente detector**) | cpu, onnxruntime | crop geometrico fisso per vista, vedi sotto |
| `lpips` | `lpips` (AlexNet) | L40S | pairwise per vista, poi media sulle 3 viste; render ridotti a 256 px |
| DPDist | — | — | **non implementata**, voce mancante della tabella |

**Render.** `v2_work/phase0/render_mesh.py`, 3 viste (yaw 0, ±30°), luce e sfondo fissi
del renderer, 512 px.  `render_cache.py` aggiunge due cose:

1. **la normalizzazione di Chamfer, per mesh**: centro sulla media dei vertici, divisione
   per il max valore assoluto (`common.maxabs_normalize`, la stessa funzione che usa
   `chamfer_matrix.py`).  Senza, le metriche percettive guardano una mesh e Chamfer
   un'altra, e le due colonne non sono confrontabili.  La prima versione di questa nota
   diceva che la posizione e la dimensione della mesh grezza "non sono informazione di
   identita' ma di topologia": e' falso, e la riga `bbox_proxy` lo misura.  Media dei
   vertici e bbox si spostano davvero quando cambia la densita' dei vertici, ma quei
   quattro numeri portano ANCHE identita': anche **dopo** la normalizzazione maxabs il
   proxy fa Spearman 0.284 su original→original e 0.117 sul cross no-crop (held-out,
   misurato).  La normalizzazione toglie la parte di scala e posizione assoluta, non tutto
   il segnale banale, ed e' per questo che il proxy resta in tabella come riga di
   controllo invece di essere dichiarato risolto.
2. **la camera unica per tutte le mesh**: centro = media dei centri di bbox, scala = la piu'
   piccola che non taglia nessuna mesh in nessuna vista, salvati in `renders/camera.json`
   assieme al marcatore `"normalize": "maxabs"` (una camera in cache senza quel marcatore
   viene rifatta, perche' i suoi render non stanno nello stesso spazio dei nuovi).  Col
   default del renderer (bbox della singola mesh) una topologia piu' stretta verrebbe
   disegnata piu' grande della sua stessa `original`.

**ArcFace: crop fisso, non detector.**  Il detector di insightface scatta sui render
sintetici in modo molto **diverso da topologia a topologia**: misurato sui 1500 render
held-out (100 soggetti x 5 topologie x 3 viste), fallisce su 4/300 render `down8k` (1.3%),
58/300 `original` (19.3%), 77/300 `up60k` (25.7%), 223/300 `remesh` (74.3%) e **300/300**
`noisy` (100%), cioe' 662/1500 in tutto (44.1%); su facebench_first100 sono 751/1500.  La
classe `perceptual_embed.ArcFaceExtractor` in quei casi ripiega su un center-crop
dell'immagine intera: un'altra inquadratura, quindi un altro spazio di embedding, mescolato
al primo dentro la stessa matrice di distanze — e con un contatore di fallback che nel ramo
a processo singolo era una costante `0`.

`arcface_fixed.py` la sostituisce.  Dato che la camera e' fissa e la mesh normalizzata, il
volto cade sempre nello stesso riquadro: si fa girare il detector UNA volta su tutti i
render (`calibrate_arcface`), si prende la mediana per vista dei suoi 5 landmark, da quella
si ricava con `face_align.estimate_norm` la similarita' verso il template `arcface_dst`, e
si congela.  Tutti i render della stessa vista passano da quella trasformazione e vanno
diritti al modello di riconoscimento.  La calibrazione salva in `renders/arcface_align.json`
i landmark mediani, il loro IQR e **il conteggio dei fallimenti del detector per
topologia**, che e' il numero che prima non veniva mai calcolato.

**Non esiste un ramo di ripiego, quindi non esiste un contatore di ripieghi.**  La versione
precedente di questo README diceva che "i ripieghi dell'estrattore sono `0/1500`, e non
perche' li si tace: la riga viene stampata leggendo l'attributo dell'estrattore, non una
costante".  Era falso: l'attributo (`arcface_fixed.py`, `self.n_fallback = 0`) era una
costante, inizializzata a zero e mai incrementata.  Ora l'attributo non c'e' piu' e
`perceptual_matrix` non stampa nessuna riga di ripieghi per ArcFace, perche' non c'e'
niente da contare: c'e' **una sola trasformazione di allineamento per vista**, applicata
a tutte le topologie.

Quella trasformazione ha pero' un limite da dichiarare, e non e' un ripiego: la mediana da
cui viene e' calcolata sulle sole detection riuscite, **838 su 1500** sull'held-out
(299 / 283 / 256 per yaw -30 / 0 / +30; 749/1500 su facebench_first100), e quelle 838 non
sono distribuite in modo uniforme sulle topologie — 296 da `down8k`, 242 da `original`,
223 da `up60k`, 77 da `remesh`, **zero da `noisy`**.  Il riquadro e' quindi calibrato su
quattro topologie e applicato a cinque.  Resta preferibile al ripiego, perche' tiene tutte
le distanze in un solo spazio di embedding, ma un riquadro un po' diverso sposterebbe
l'intera colonna ArcFace nello stesso verso.

**Varifold e currents: niente sottocampionamento casuale.**  `mesh_measure` di phase0
sottocampiona a caso `max_tris` triangoli con un seme, e la prima versione passava
`max_tris=4000` con `seed = indice del soggetto`.  Misurato: la stessa identica mesh con due
semi diversi dista **0.054** (original), 0.044 (remesh), 0.058 (down8k), 0.050 (noisy),
0.044 (up60k) da se' stessa, cioe' quanto due soggetti diversi.  Ora il sottocampionamento
casuale e' spento e la misura viene quantizzata su una griglia cubica di lato
`sigma_min / 4`, uguale per tutte le topologie: la self-distance e' **esattamente 0** su
tutte e cinque (verificato di nuovo dopo il passaggio a maxabs, job 1019724), e il numero di
atomi (12848 / 19575 / 20987 / 22884 / 25498 per down8k / remesh / original / noisy / up60k
con `--normalize maxabs`; 6832 / 7859 / 8008 / 4159 / 8744 con `--normalize area`) diventa
una proprieta' della superficie e non della tassellazione.
La decimazione della mesh con `igl.qslim` non era usabile: le mesh REMESH non sono
edge-manifold e sia `qslim` sia `igl.decimate` restituiscono una mesh vuota.  Che la
quantizzazione non sposti i risultati e' misurato: le distanze fra soggetti diversi cambiano
meno dell'1% rispetto ai triangoli interi, con Spearman 1.000.

**Varifold e currents: maxabs, non area.**  Il default di `measure_distances` e'
`normalize="area"` — centro sul baricentro pesato sulle aree, divisione per `sqrt(area
totale)` — e la prima versione lo teneva.  E' sbagliato proprio dove serve: su REMESH
`noisy` ha **2.28x** l'area della sua `original` (il rumore increspa i triangoli senza
toccare l'ingombro), quindi la normalizzazione area la rimpicciolisce di `1/sqrt(2.28)` e
accorcia le distanze del 34%.  Misurato sulle matrici vecchie (held-out, varifold, mediana
cross-soggetto): **0.0846** fra topologie di sola tassellazione contro **0.3227** appena
entra `noisy`, cioe' `d(original_i, noisy_i)` dello stesso soggetto cade dentro la nuvola
dei soggetti diversi.  Sulle stesse matrici lo Spearman vs D_GT per cella passa da ~0.57
(celle di sola tassellazione) a **0.14-0.19** (celle con noisy) per il varifold e da ~0.61 a
0.04-0.22 per currents.  Ora il default e' `--normalize maxabs`, la stessa normalizzazione
di Chamfer e dei render, e le matrici area sono conservate in
`aau/runs/ws1_old_geom/areanorm_*`.

**Ma il passaggio a maxabs non ripara il varifold.** Esito misurato (job 1019726/1019727, Spearman vs D_GT mediato sulle coppie di topologie, held-out, area -> maxabs): varifold same 0.577 -> 0.656, sola tassellazione 0.569 -> 0.503, con noisy 0.162 -> 0.145; currents same 0.614 -> 0.661, tassellazione 0.337 -> 0.328, con noisy 0.145 -> 0.323. Cioe' maxabs NON ripara il varifold sulle celle con noisy: il problema cambia segno invece di sparire. Con area la misura di noisy era rimpicciolita (mediana cross 0.32 contro 0.085); con maxabs ha 2.28x la massa della sua original, e la distanza e' dominata dalla differenza di massa (mediana cross 1.36 contro 0.34). Currents ne beneficia perche' le normali increspate di noisy si cancellano col segno; il varifold, che le eleva al quadrato, no. Nessuna normalizzazione per similarita' toglie una differenza di area vera: servirebbe normalizzare la massa (o confrontare misure a massa unitaria) senza rimpicciolire la geometria.

**`--normalize maxabs_unitmass`** (opzione, i default non cambiano) lo fa: frame maxabs come Chamfer, pesi dei triangoli divisi per l'area totale (massa 1), sigma di phase0 x 1.8482 (media di `sqrt(area)/maxabs` sulle 100 original held-out), matrici in `varifold_maxabs_unitmass` e `currents_maxabs_unitmass`. Held-out, same / no-crop / tassellazione / noisy, maxabs -> maxabs_unitmass: varifold 0.655/0.192/0.472/0.136 -> 0.646/0.258/0.496/0.307, currents 0.661/0.156/0.292/0.266 -> 0.663/0.210/0.287/0.289 (job 1054479/1054486); d(original_i, noisy_i) sotto la mediana fra soggetti diversi per 100/100 soggetti. Costo 48-50 ms/coppia su A10, 56 min per set.

Un effetto collaterale da dichiarare, perche' non e' solo un cambio di normalizzazione: le
sigma sono in unita' della normalizzazione, e la mesh maxabs e' ~1.6x piu' grande di quella
area-normalizzata.  A parita' di sigma il kernel guarda percio' un dettaglio fisico piu'
fine, gli atomi passano da ~8000 a ~21000 e il costo da 122 a **890 ms/coppia** su T4
(misurato, job 1019724).  Per questo `geometric.sbatch` chiede una A40 e 20 h invece di una
T4 e 8 h.

## Sequenza

```bash
aau/submit.sh baselines/probe_env.sbatch      # cpu, 2 min: dipendenze, rete, dati, tempi
aau/submit.sh baselines/setup_env.sbatch      # cpu, 5 min: .venv_baselines + cache dei pesi
aau/submit.sh baselines/chamfer.sbatch        # 24 core, 2 min
aau/submit.sh baselines/geometric.sbatch      # A40, ~9 h (maxabs; era 1.5 h a normalize=area)
aau/submit.sh baselines/render.sbatch         # 16 core, 2 min
WBES_BL_ARGS="--extractors clip,dinov2,lpips" \
  aau/submit.sh baselines/perceptual.sbatch --gres=gpu:t4:1   # 4 min, dopo render
WBES_BL_ARGS="--extractors arcface --device cpu --embed-workers 16" \
  aau/submit.sh baselines/perceptual.sbatch --gres=NONE --cpus-per-task=16   # calibrazione + embedding
aau/submit.sh baselines/rank.sbatch           # cpu, ~10 min (calcola anche bbox_proxy)
```

`rank.sbatch` chiama `proxy_matrix.py` prima di `rank_from_matrix.py`: la riga di controllo
costa quattro numeri per mesh e non merita un job suo.

ArcFace va sottomesso a parte: insightface gira su onnxruntime CPU (circa 26 s di CPU per
render), quindi non ha senso tenere occupata una L40S mentre macina, e conviene spargerlo
su piu' processi.  Per riprodurre la Tabella 2 la stessa sequenza con
`--subject-set facebench_first100 --out-root aau/runs/baselines_fb100`.

Opzioni extra allo script python via `WBES_BL_ARGS`:

```bash
WBES_BL_ARGS="--kinds varifold --max-tris 2000" aau/submit.sh baselines/geometric.sbatch
```

Tutti gli script sono **riprendibili**: saltano le matrici gia' su disco a meno di
`--overwrite`.  Output sotto `aau/runs/baselines/` (override `WBES_BASELINES_OUT`):
`matrices/<metrica>/<tA>__to__<tB>.npz`, `renders/`, `embeddings/`,
`ranking/table2_extended_<subject-set>.{csv,tex,json}`.

## Diagonale della Tabella 1 (WS0)

Le 6 celle **same-topology** della Tabella 1 (`tab:clean_xtopo_chamfer_latent_matrices`,
Chamfer 0.558-0.743 e latent 0.831-0.902) non le produce `eval_topology`: sono Spearman vs
`D_GT` sulle 4950 coppie i<j con **entrambe** le mesh nella stessa topologia, sui soggetti
`facebench_first100` e con la Chamfer variante faceBench.  Un job le fa tutte e dodici:

```bash
aau/submit.sh baselines/table1_diag.sbatch       # T4, ~9 min
```

Dentro ci sono i tre passi di sempre: `chamfer_matrix.py` e `human_study/latent_matrix.py`
con `--settings same_<topologia>` / `--topology crop,down8k,...`, poi `table1_diagonal.py`
(riusa `rank_from_matrix.compute_rows`, un setting per topologia) che scrive
`ranking/table1_diagonal_<subject-set>.{csv,tex,json}` con accanto il valore pubblicato e
il delta.  Riprodotta: tutte e 12 dentro il CI del paper, `max |delta| = 0.0078`
(latent su `noisy`, 0.8922 contro 0.900), job 1019690.

## Ambiente

`.venv_baselines` e' **separato** da `.venv_aau`: insightface/open_clip/lpips tirano
dietro numpy 2.x e un opencv che nel container non si importa, e non e' il caso di
metterli nel venv su cui girano training ed eval.  Due paletti nel `setup_env.sbatch`,
imparati sbagliando (job 1019231):

- constraint `numpy<2`: nel venv numpy 2.x shadowa quello del container e fa esplodere
  torch/torchvision NGC, compilati contro numpy 1.x;
- `insightface --no-deps`: la sua dipendenza `opencv-python` ha bisogno di `libxcb.so.1`,
  assente nel container; ma il `cv2` 4.7.0 del container e' una build ridotta senza
  modulo `dnn`, che ad ArcFace serve.  La combinazione che funziona e'
  `insightface --no-deps` piu' `opencv-python-headless<5`.

I nodi di calcolo vedono pypi.org, github.com e huggingface.co (verificato dal probe), ma
**non** `dl.insightface.net`.  I pesi `buffalo_l` arrivano dalle release GitHub di
insightface.  `warm_caches.py` scalda una volta sola le cache in home (CephFS), che sono
poi condivise da tutti i job.

## Cosa e' stato verificato, e come

- **`chamfer_sq` coincide con i valori pair-level gia' nel repo.**
  `check_vs_paper_pairs.py` confronta le matrici con la colonna `raw_chamfer` di
  `table1_pairlevel_exact`: errore relativo massimo 7e-4 su 3 coppie di topologie x 4950
  coppie, cioe' l'arrotondamento del CSV (job 1019265).  Caricamento delle mesh,
  normalizzazione e ordinamento delle coppie sono quindi giusti.
- **Chamfer riproduce la Tabella 2** con `--subject-set facebench_first100` (sopra).
- **La self-distance del varifold e' 0** con la regola nuova, contro 0.044-0.058 col
  sottocampionamento casuale, e il kernel di `geometric_kernel.py` coincide con quello di
  `measure_distances.py` entro 2e-7 relativi (`geometric_matrix.py --self-test`).
- **DPDist non e' implementata** ed e' la voce mancante della tabella.
- **La riga `bbox_proxy` e' un controllo, non un risultato**, e va letta per prima: sulle
  mesh gia' normalizzate maxabs fa Spearman 0.284 su original→original e 0.117 sul cross
  no-crop (held-out; 0.353 e 0.170 su facebench_first100).  Ogni riga della tabella che non
  batte questi numeri sta misurando qualcosa che quattro numeri banali gia' contengono.

## Quale tabella va nel paper

`table2_extended_heldout`, quattro colonne — **held-out e' la tabella primaria**.  `facebench_first100`
condivide solo 21 soggetti su 100 con lo split held-out, cioe' ~79 dei suoi soggetti sono
soggetti di training: serve unicamente come **gate di riproduzione** dello 0.729 pubblicato
e per documentare il problema, non come colonna di risultati.

Le colonne sono quattro e non due: original→original, no-crop cross-topology (quella del
paper), e le due in cui la seconda si spezza — tessellation only e perturbation.  La
seconda resta perche' e' il numero pubblicato e perche' le due nuove sommano esattamente a
lei; le due nuove ci sono perche' "cross-topology" mescola due domande diverse, e una
metrica puo' cavarsela sulla prima e crollare sulla seconda.  In fondo alla tabella, sotto
una riga a se' e con il pugnale, stanno le metriche a copertura parziale: la loro cella e'
una media su altre coppie di topologie e non si legge in colonna con le altre.
