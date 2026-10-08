# E11, parte dati: GT unificata per il training e per le valutazioni (8 ottobre 2026)

Codice: `v3_work/unified_gt/train_gt.py`, `check_train_gt.py`, `train_gt.sbatch`, `make_eval_gt.py`.
Definizione della GT: `summary.md`, sezione 3. Job 1061820: 6:42 di durata, 35.7 GB di RSS massimo.

## 1. GT di training: drop-in per il trainer v2

**File:** `datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm.npz` (17.2 GB) e il `.json` accanto.

**Formato:** lo stesso di `datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz`, la GT del run
`scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411`:
- `D_orig` float32, 65.600 x 65.600;
- `names` <U8, copiati dal file del run: stessi id, stesso ordine.

**Uso:** si cambia solo `--dist_npz`. Il massimo è 1, quindi la lettura è la stessa con e senza `--gt-keep-scale`.

**Valori:**
- `D_orig * mm_per_unit` (14.16) = g in mm.
- Mediane dentro i domini: BFM 3.98 mm, ICT 3.57, GNM 4.13, cioè 0.25-0.29 in unità del file.
- Coppie fra domini: riempite (mediana 4.3-4.5 mm). Il trainer a blocchi monodominio non le legge; la guardia NaN di train_v2 non ha nulla da segnalare.

**Sorgenti:** le `original` neutre lette dalle stesse sorgenti del run:

| dominio | identità | sorgente | chiave |
| --- | --- | --- | --- |
| BFM | 500 | `REMESH/npz_data_topo_500_withops_areanorm` | `verts` |
| ICT-5000 | 5.000 | `ICT/train_ready/npz_withops` | `verts` |
| ICT nuove | 50.000 | tar di `SCALE_ALL/shards`, per offset con `index.npz` | `V` |
| GNM | 10.100 | tar di `SCALE_ALL/shards`, per offset con `index.npz` | `V` |

La `original` GNM è la patch hockey_mask: i suoi vertici tornano agli indici della testa intera, dove sta la mappa. Tutti i vertici della mappa stanno nella patch: verificato.

**Controlli, eseguiti:**
- **Sorgenti del run contro `shapes.py`** (pesi del modello o original grezze), su TUTTE le identità: scarto massimo 7e-6 mm.
- **Loader del trainer in sola lettura** (`check_train_gt.py`). La catena è `train_steps.check_gt_scale` (passa in entrambi i modi) → `install_gt_keep_scale` → `train_v2._nan_guarded_loader`, chiamata con `dtype=np.float64` come in `train_runner`. Risultati:
  - `name_to_idx` identico a quello della GT del run;
  - stessi indici per held-out (1.200), valutazione online (16 BFM e 16 GNM) e train (64.400);
  - 0 valori non finiti, diagonale 0, simmetria esatta;
  - letture attraverso la guardia senza errori;
  - valori contro `s_train.npz`: scarto massimo 8e-7 mm.
- **s_i nell'ordine di `names`:** `train/s_train.npz` (1.2 GB).

**Da sapere per E11:**
- La GT unificata ha la scala di FLAME per tutti i domini.
- I margini della loss (p.es. `--rank_margin 0.05`) sono in unità GT: 0.05 corrisponde a 0.7 mm.
- La mediana delle distanze nel file (0.25-0.29) è nello stesso ordine di quella della GT attuale, ma le due scale non sono state tarate l'una sull'altra.

## 2. GT unificata dei set di valutazione, formato `aau/zs3dmm`

`make_eval_gt.py --domain hifi3d|faceverse|facescape [--ids ...]`. In Python si usa `unified_gt(dominio, ids)`.

Senza `--ids` usa i `names` della vista (`<vista>/gt_matrix.npz`), nello stesso ordine. Il file:
- si può mettere al posto di `--gt` / `ZS_DIST_NPZ`;
- oppure si passa a `zs_summarize.with_gt` sulle righe già calcolate, come fa `eval_methods.py`. Così non serve rifare gli embedding.

Prodotti, pool di 500:

| file | mediana | controlli |
| --- | --- | --- |
| `datasets/UNIFIED_GT/eval/hifi3d_gt_matrix.npz` | 4.31 mm | pesi contro patch salvata: 5e-7 |
| `datasets/UNIFIED_GT/eval/faceverse_gt_matrix.npz` | 7.02 mm | pesi contro patch salvata: 3e-8 |
| `datasets/UNIFIED_GT/eval/facescape_gt_matrix.npz` (dev, id930000+) | 2.68 mm | 4.010/4.010 vertici della mappa nella patch dev; pesi contro patch: 4e-6 mm |

- HIFI3D e FaceVerse coincidono con le GT usate in `summary.md`: scarto massimo 3e-8 dopo la normalizzazione.
- FaceScape: la base neutra ristretta ai vertici della mappa sta in `datasets/UNIFIED_GT/cache/` (solo locale, per la licenza FaceScape).

**Disco:** `datasets/UNIFIED_GT` occupa 19 GB in tutto.
