# Protocollo dichiarato prima delle eval: pilota della distillazione ArcFace-render -> DiffusionNet

Dichiarato il 2026-10-06, prima di qualunque embedding dello studente del pilota su FaceVerse o HIFI3D.
Piano: `paper/PLAN_DISTILL.md`. Gate: `aau/runs/arcface_render_zs/summary.md`.
Numeri gia' noti a quest'ora: tutte le righe del gate (insegnante ombreggiato su entrambi i domini,
normal map 3 viste su FaceVerse 0.867, NICP, ICP, Chamfer, congiunto). Il rank-1 della normal map su
HIFI3D NON e' noto: gli embedding sono stati calcolati (job 1057319, stessa pipeline e stesso crop del
gate) ma il riconoscimento non e' stato fatto. L'unico studente valutato prima di questa dichiarazione e'
uno smoke di 2 epoche su 48 soggetti (`aau/runs/distill_pilot/smoke`), usato solo per provare la catena
di eval su HIFI3D; non e' il pilota e i suoi numeri non entrano nel giudizio.

**Domanda.** Uno studente DiffusionNet addestrato solo a imitare ArcFace su normal map (sui dati BFM e
ICT-5000 di training del congiunto) batte il congiunto fuori dominio e si avvicina all'insegnante?

**Insegnante** (`aau/distill/teacher_labels.py`): la pipeline del gate importata (`zs_arcface_render.py`),
normal map, yaw 0/-30/+30, 512 px, embedding = media delle 3 viste rinormalizzata L2. Camera unica e crop
fisso PER DOMINIO DI TRAINING (BFM, ICT), il crop calibrato col detector sui render ombreggiati delle
`original` di 300 soggetti di training a caso (seed 1234) e copiato sulle normal map. Rotazione verso il
frame del renderer: BFM nessuna, ICT Rx(180), verificate con `--frame-check`. L'insegnante vede SOLO le
`original` (e, per i 418 soggetti ICT che le hanno, le 5 espressioni casuali `rexpr<k>`, stessa topologia).

**Dati.** Training: lo split del congiunto 1019532 (`aau/data_scale/split_scale.json`): 392 BFM, 4008
ICT-5000; di questi 411 ICT hanno le `rexpr`. Validazione (solo curve): i 16 BFM dell'eval online del
congiunto e 84 ICT held-out (seed 1234). FaceVerse e HIFI3D non entrano.

**Studente** (`aau/distill/train_distill.py`): `xyz_dn` con la ricetta del congiunto (width 128, 4 blocchi,
dropout 0.1, pool meanmax, rumore latente, augmentation xyz del congiunto, operatori ad area unitaria
k_eig 128 delle viste in uso, loader congelato), uscita a 512 dimensioni normalizzata L2. Ingresso: le 5
topologie senza crop (original, remesh, down8k, up60k, noisy) e le `rexpr`, frame nativi come il congiunto.
Bersaglio: l'embedding dell'insegnante della `original` dello stesso soggetto (della `rexpr<k>` stessa
per un'espressione). Loss: (1 - coseno) + 1.0 x MSE fra matrice dei coseni dello studente e
dell'insegnante nel batch (fuori diagonale). Batch: 16 soggetti a caso (domini mescolati) x 2 mesh
distinte. Adam, lr 1e-4 dimezzato dopo il 75% delle epoche, weight decay 1e-6, clip 1.0, seed 1234,
**50 epoche**. **Checkpoint valutato: l'ULTIMO** (`checkpoints/last.pth`), fissato ora; la validazione
held-out serve solo per le curve, nessuna selezione.

**Eval** (`aau/distill/eval_distill.sbatch`, `distill_summarize.py`): identica al gate. Stessi 100
soggetti (`select_subjects`, seed 1234), FaceVerse v2 con espressioni casuali e HIFI3D neutro, operatori
k_eig 128 ad area unitaria su /tmp (`zs_stage.py` + `areanorm_operators.py`), distanza = 1 - coseno
(euclidea sugli embedding L2, monotona). Funzioni di calcolo importate da `zs_expr_summarize.py` e
`zs_arcface_summarize.py`, STESSE repliche bootstrap del gate (1000, `stable_seed(1234,
"expr_recognition")`): le righe gia' nel gate devono coincidere col suo `recognition.csv` (controllo).

- **PRIMARIO:** riconoscimento sulle 5 topologie senza crop: rank-1, mAP, AUC di verifica, CI 95%.
- **Riga di riferimento: studente in convenzione BFM** (FaceVerse facce invertite; HIFI3D Rx(180) +
  facce invertite), cioe' lo STESSO ingresso della riga del congiunto nel gate.
- Righe di confronto: insegnante 3 viste normal map e ombreggiato, NICP P2Tri, ICP rigido + Chamfer,
  Chamfer faceBench, congiunto BFM+ICT (convenzione BFM). Delta APPAIATI studente - ciascuna.
- Secondario: studente in convenzione ICT (FaceVerse Rx(180); HIFI3D nativo), delta appaiati con il
  congiunto e con lo studente in convenzione BFM. Non sostituisce il riferimento.
- Crop: le stesse misure, a parte, mai nel primario.

**Criterio di successo del pilota, fissato ora (primario, riga di riferimento):**
1. "batte il congiunto": CI 95% del delta rank-1 studente - congiunto tutto sopra 0, su FaceVerse E su HIFI3D;
2. "almeno l'80% dell'insegnante": rank-1 studente / rank-1 insegnante normal map 3 viste >= 0.80 come
   stima puntuale, su FaceVerse E su HIFI3D (riportato anche il CI del rapporto, stesse repliche).
Il pilota passa se valgono entrambi su entrambi i domini. Le stesse letture sull'AUC si riportano ma non
decidono. Se passa solo in convenzione ICT, si riporta come tale e il pilota NON passa.

---

# Risultati: FaceVerse v2 con espressioni casuali

Soggetti: 100 (`select_subjects`, seed 1234), mesh da `datasets/FACEVERSE_ZS/expr_view/npz`; studente da `aau/runs/distill_pilot/eval/fv_expr`, insegnante da `aau/runs/arcface_render_zs/fv_expr`, baseline da `aau/runs/ws_faceverse_expr/data_736f96956a/baselines`, congiunto da `aau/runs/ws_faceverse_expr/data_736f96956a/joint_flip_topology/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse del gate).

## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | 0.332 [0.299, 0.366] | 0.432 [0.400, 0.465] | 0.797 [0.773, 0.821] | 0 |
| Studente distillato, convenzione ICT | 0.570 [0.520, 0.618] | 0.655 [0.611, 0.696] | 0.899 [0.875, 0.921] | 0 |
| ArcFace, normal map, 3 viste | 0.867 [0.843, 0.889] | 0.911 [0.893, 0.927] | 0.934 [0.921, 0.946] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.750 [0.724, 0.775] | 0.807 [0.785, 0.829] | 0.863 [0.847, 0.879] | 0 |
| Rigid ICP + NICP + P2Tri | 0.959 [0.940, 0.975] | 0.968 [0.953, 0.981] | 0.995 [0.992, 0.998] | 0 |
| Rigid ICP + Chamfer | 0.918 [0.895, 0.939] | 0.935 [0.916, 0.953] | 0.986 [0.979, 0.992] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.740 [0.698, 0.783] | 0.775 [0.737, 0.814] | 0.882 [0.853, 0.910] | 0 |
| BFM+ICT, convenzione BFM | 0.680 [0.634, 0.723] | 0.731 [0.688, 0.770] | 0.875 [0.847, 0.900] | 0 |

### Delta appaiati, studente (convenzione BFM) - baseline

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, normal map, 3 viste | -0.535 [-0.575, -0.496] (1.000) | -0.479 [-0.515, -0.443] (1.000) | -0.138 [-0.163, -0.112] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, ombreggiato, 3 viste | -0.418 [-0.457, -0.379] (1.000) | -0.375 [-0.411, -0.338] (1.000) | -0.067 [-0.093, -0.039] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + NICP + P2Tri | -0.627 [-0.660, -0.592] (1.000) | -0.536 [-0.566, -0.505] (1.000) | -0.198 [-0.220, -0.175] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + Chamfer | -0.586 [-0.621, -0.548] (1.000) | -0.503 [-0.535, -0.470] (1.000) | -0.190 [-0.211, -0.168] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Chamfer (faceBench, 4096 pt) | -0.408 [-0.454, -0.365] (1.000) | -0.343 [-0.383, -0.304] (1.000) | -0.086 [-0.110, -0.061] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | BFM+ICT, convenzione BFM | -0.348 [-0.395, -0.300] (1.000) | -0.300 [-0.342, -0.258] (1.000) | -0.078 [-0.104, -0.052] (1.000) |

### Criterio: rapporto con l'insegnante a normal map

- Studente distillato, convenzione BFM (riferimento): rank-1 / insegnante normal map = 0.383 [0.344, 0.422]
- Studente distillato, convenzione ICT: rank-1 / insegnante normal map = 0.657 [0.596, 0.717]

### Secondari

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione ICT | BFM+ICT, convenzione BFM | -0.110 [-0.154, -0.062] (1.000) | -0.077 [-0.117, -0.033] (1.000) | +0.024 [-0.005, +0.051] (0.047) |
| Studente distillato, convenzione ICT | Studente distillato, convenzione BFM (riferimento) | +0.238 [+0.186, +0.290] (0.000) | +0.223 [+0.180, +0.269] (0.000) | +0.102 [+0.076, +0.125] (0.000) |
| Studente distillato, convenzione ICT | ArcFace, normal map, 3 viste | -0.297 [-0.353, -0.242] (1.000) | -0.256 [-0.303, -0.210] (1.000) | -0.036 [-0.063, -0.010] (0.998) |

## A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | 0.031 [0.018, 0.046] | 0.106 [0.088, 0.126] | 0.651 [0.626, 0.678] | 0 |
| Studente distillato, convenzione ICT | 0.269 [0.218, 0.324] | 0.378 [0.325, 0.431] | 0.804 [0.773, 0.836] | 0 |
| ArcFace, normal map, 3 viste | 0.952 [0.937, 0.964] | 0.966 [0.955, 0.976] | 0.963 [0.954, 0.970] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.881 [0.867, 0.896] | 0.909 [0.897, 0.921] | 0.922 [0.910, 0.932] | 0 |
| Rigid ICP + NICP + P2Tri | 0.909 [0.876, 0.938] | 0.934 [0.910, 0.956] | 0.983 [0.974, 0.992] | 0 |
| Rigid ICP + Chamfer | 0.513 [0.455, 0.571] | 0.590 [0.540, 0.640] | 0.777 [0.734, 0.814] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.396 [0.336, 0.457] | 0.479 [0.421, 0.537] | 0.809 [0.781, 0.837] | 0 |
| BFM+ICT, convenzione BFM | 0.243 [0.205, 0.283] | 0.364 [0.325, 0.405] | 0.801 [0.771, 0.829] | 0 |

### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, normal map, 3 viste | -0.921 [-0.939, -0.902] (1.000) | -0.860 [-0.881, -0.838] (1.000) | -0.312 [-0.339, -0.284] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, ombreggiato, 3 viste | -0.850 [-0.870, -0.829] (1.000) | -0.803 [-0.824, -0.780] (1.000) | -0.271 [-0.298, -0.242] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + NICP + P2Tri | -0.878 [-0.911, -0.840] (1.000) | -0.828 [-0.857, -0.796] (1.000) | -0.332 [-0.358, -0.306] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + Chamfer | -0.482 [-0.541, -0.422] (1.000) | -0.484 [-0.536, -0.429] (1.000) | -0.126 [-0.168, -0.083] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Chamfer (faceBench, 4096 pt) | -0.365 [-0.425, -0.304] (1.000) | -0.373 [-0.430, -0.315] (1.000) | -0.158 [-0.194, -0.124] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | BFM+ICT, convenzione BFM | -0.212 [-0.254, -0.170] (1.000) | -0.258 [-0.303, -0.213] (1.000) | -0.150 [-0.181, -0.116] (1.000) |
| Studente distillato, convenzione ICT | BFM+ICT, convenzione BFM | +0.026 [-0.034, +0.091] (0.215) | +0.014 [-0.048, +0.076] (0.327) | +0.003 [-0.033, +0.040] (0.402) |
| Studente distillato, convenzione ICT | Studente distillato, convenzione BFM (riferimento) | +0.238 [+0.186, +0.291] (0.000) | +0.272 [+0.218, +0.321] (0.000) | +0.153 [+0.112, +0.194] (0.000) |
| Studente distillato, convenzione ICT | ArcFace, normal map, 3 viste | -0.683 [-0.733, -0.629] (1.000) | -0.588 [-0.640, -0.537] (1.000) | -0.159 [-0.190, -0.124] (1.000) |

- Studente distillato, convenzione BFM (riferimento): rank-1 / insegnante normal map = 0.033 [0.019, 0.049]
- Studente distillato, convenzione ICT: rank-1 / insegnante normal map = 0.283 [0.229, 0.340]

## Controlli

- riproduzione di `aau/runs/arcface_render_zs/fv_expr/recognition.csv` (stesse repliche): max |diff| su punto e CI = 1.11e-16 su 12 righe (arcface_normals_3v, arcface_shaded_3v, chamfer, joint@bfm, nicp_p2tri, rigid_icp_chamfer)

---

# Risultati: HIFI3D, neutre (senza espressioni)

Soggetti: 100 (`select_subjects`, seed 1234), mesh da `datasets/HIFI3D/eval_view/npz`; studente da `aau/runs/distill_pilot/eval/hifi3d`, insegnante da `aau/runs/arcface_render_zs/hifi3d`, baseline da `aau/runs/ws_hifi3d/data_328f2bfc1a/baselines`, congiunto da `aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_ranking/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse del gate).

## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | 0.224 [0.207, 0.244] | 0.279 [0.261, 0.299] | 0.603 [0.589, 0.617] | 0 |
| Studente distillato, convenzione ICT | 0.355 [0.322, 0.392] | 0.446 [0.413, 0.479] | 0.781 [0.761, 0.800] | 0 |
| ArcFace, normal map, 3 viste | 0.998 [0.995, 1.000] | 0.999 [0.998, 1.000] | 0.997 [0.996, 0.998] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.984 [0.974, 0.992] | 0.991 [0.985, 0.996] | 0.989 [0.985, 0.993] | 0 |
| Rigid ICP + NICP + P2Tri | 0.990 [0.982, 0.998] | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] | 1965 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] | 0 |
| BFM+ICT, convenzione BFM | 0.397 [0.363, 0.434] | 0.517 [0.489, 0.549] | 0.755 [0.740, 0.770] | 0 |

### Delta appaiati, studente (convenzione BFM) - baseline

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, normal map, 3 viste | -0.774 [-0.791, -0.754] (1.000) | -0.720 [-0.738, -0.700] (1.000) | -0.395 [-0.409, -0.380] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, ombreggiato, 3 viste | -0.760 [-0.778, -0.738] (1.000) | -0.712 [-0.731, -0.691] (1.000) | -0.387 [-0.400, -0.372] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + NICP + P2Tri | -0.766 [-0.785, -0.744] (1.000) | -0.711 [-0.733, -0.689] (1.000) | -0.397 [-0.411, -0.383] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + Chamfer | -0.772 [-0.790, -0.750] (1.000) | -0.718 [-0.737, -0.698] (1.000) | -0.397 [-0.411, -0.382] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Chamfer (faceBench, 4096 pt) | -0.253 [-0.285, -0.222] (1.000) | -0.285 [-0.311, -0.261] (1.000) | -0.180 [-0.197, -0.163] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | BFM+ICT, convenzione BFM | -0.173 [-0.207, -0.139] (1.000) | -0.238 [-0.267, -0.211] (1.000) | -0.153 [-0.167, -0.137] (1.000) |

### Criterio: rapporto con l'insegnante a normal map

- Studente distillato, convenzione BFM (riferimento): rank-1 / insegnante normal map = 0.224 [0.208, 0.245]
- Studente distillato, convenzione ICT: rank-1 / insegnante normal map = 0.356 [0.322, 0.392]

### Secondari

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione ICT | BFM+ICT, convenzione BFM | -0.042 [-0.079, -0.003] (0.986) | -0.071 [-0.102, -0.038] (1.000) | +0.026 [+0.008, +0.043] (0.002) |
| Studente distillato, convenzione ICT | Studente distillato, convenzione BFM (riferimento) | +0.131 [+0.099, +0.162] (0.000) | +0.167 [+0.138, +0.192] (0.000) | +0.179 [+0.157, +0.200] (0.000) |
| Studente distillato, convenzione ICT | ArcFace, normal map, 3 viste | -0.643 [-0.676, -0.607] (1.000) | -0.553 [-0.585, -0.519] (1.000) | -0.216 [-0.237, -0.197] (1.000) |

## A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | 0.038 [0.020, 0.060] | 0.110 [0.089, 0.136] | 0.579 [0.567, 0.592] | 0 |
| Studente distillato, convenzione ICT | 0.257 [0.219, 0.293] | 0.367 [0.333, 0.400] | 0.704 [0.685, 0.722] | 0 |
| ArcFace, normal map, 3 viste | 0.997 [0.994, 1.000] | 0.999 [0.997, 1.000] | 0.999 [0.998, 0.999] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.990 [0.984, 0.996] | 0.994 [0.991, 0.998] | 0.994 [0.992, 0.996] | 0 |
| Rigid ICP + NICP + P2Tri | 0.777 [0.740, 0.810] | 0.854 [0.829, 0.877] | 0.997 [0.995, 0.998] | 1965 |
| Rigid ICP + Chamfer | 0.284 [0.237, 0.338] | 0.423 [0.381, 0.470] | 0.888 [0.865, 0.911] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.472 [0.445, 0.499] | 0.536 [0.510, 0.564] | 0.699 [0.684, 0.715] | 0 |
| BFM+ICT, convenzione BFM | 0.085 [0.057, 0.117] | 0.201 [0.171, 0.233] | 0.716 [0.699, 0.734] | 0 |

### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, normal map, 3 viste | -0.959 [-0.978, -0.937] (1.000) | -0.889 [-0.910, -0.863] (1.000) | -0.419 [-0.432, -0.406] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, ombreggiato, 3 viste | -0.952 [-0.972, -0.929] (1.000) | -0.885 [-0.906, -0.859] (1.000) | -0.415 [-0.428, -0.401] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + NICP + P2Tri | -0.739 [-0.777, -0.700] (1.000) | -0.744 [-0.774, -0.713] (1.000) | -0.418 [-0.430, -0.405] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + Chamfer | -0.246 [-0.299, -0.202] (1.000) | -0.313 [-0.361, -0.273] (1.000) | -0.308 [-0.331, -0.284] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Chamfer (faceBench, 4096 pt) | -0.434 [-0.461, -0.408] (1.000) | -0.427 [-0.449, -0.404] (1.000) | -0.120 [-0.133, -0.106] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | BFM+ICT, convenzione BFM | -0.047 [-0.075, -0.022] (1.000) | -0.092 [-0.120, -0.067] (1.000) | -0.136 [-0.151, -0.121] (1.000) |
| Studente distillato, convenzione ICT | BFM+ICT, convenzione BFM | +0.172 [+0.130, +0.212] (0.000) | +0.166 [+0.132, +0.201] (0.000) | -0.012 [-0.030, +0.006] (0.901) |
| Studente distillato, convenzione ICT | Studente distillato, convenzione BFM (riferimento) | +0.219 [+0.184, +0.253] (0.000) | +0.257 [+0.226, +0.287] (0.000) | +0.124 [+0.107, +0.141] (0.000) |
| Studente distillato, convenzione ICT | ArcFace, normal map, 3 viste | -0.740 [-0.778, -0.703] (1.000) | -0.631 [-0.666, -0.599] (1.000) | -0.295 [-0.314, -0.276] (1.000) |

- Studente distillato, convenzione BFM (riferimento): rank-1 / insegnante normal map = 0.038 [0.020, 0.060]
- Studente distillato, convenzione ICT: rank-1 / insegnante normal map = 0.258 [0.220, 0.294]

## Controlli

- riproduzione di `aau/runs/arcface_render_zs/hifi3d/recognition.csv` (stesse repliche): max |diff| su punto e CI = 1.11e-16 su 10 righe (arcface_shaded_3v, chamfer, joint@bfm, nicp_p2tri, rigid_icp_chamfer); righe non presenti nel csv del gate (calcolate qui): arcface_normals_3v
