# Tabella dei frame, held-out BFM (OUTLINE_B §5, esperimento 3)

Spearman vs D_GT del repo (grezzo), CI 95% bootstrap per soggetto (1000 repliche).

## Chamfer (variante faceBench, 4096 punti) per frame

| frame | Same topology | Tessellation | Perturbation (noisy) | Crop |
|---|---|---|---|---|
| maxabs (per mesh) | 0.644 [0.567, 0.707] | 0.454 [0.383, 0.512] | 0.502 [0.430, 0.570] | -0.006 [-0.066, 0.051] |
| area (per mesh) | 0.558 [0.482, 0.629] | 0.571 [0.502, 0.637] | 0.097 [0.048, 0.146] | 0.231 [0.192, 0.269] |
| rms (per mesh) | 0.602 [0.535, 0.666] | 0.621 [0.547, 0.685] | 0.416 [0.341, 0.485] | 0.339 [0.270, 0.406] |
| global (one similarity) | 0.714 [0.650, 0.772] | 0.718 [0.661, 0.773] | 0.719 [0.665, 0.771] | 0.686 [0.620, 0.747] |
| global scale, per-mesh centroid | 0.679 [0.606, 0.743] | 0.509 [0.437, 0.580] | 0.487 [0.417, 0.561] | 0.509 [0.431, 0.577] |
| global centre, per-mesh scale (maxabs divisor) | 0.659 [0.595, 0.717] | 0.516 [0.461, 0.565] | 0.595 [0.535, 0.646] | 0.025 [-0.029, 0.078] |
| bbox control (global frame, 4 numbers) | 0.370 [0.287, 0.463] | 0.352 [0.262, 0.441] | 0.265 [0.197, 0.331] | 0.015 [-0.022, 0.051] |

Le ultime tre righe spezzano global: "global scale, per-mesh centroid" toglie a ogni mesh la propria media dei vertici e divide per lo s0 globale (via la posizione, resta la taglia); "global centre, per-mesh scale" sottrae il c0 globale e divide per il divisore maxabs della mesh (via la taglia, resta la posizione rispetto a c0, riscalata); il controllo bbox e' la distanza euclidea fra (centro del bounding box, diagonale) nel frame globale, senza forma.

## D_GT ricalcolato nel frame vs D_GT del repo (original, 4950 coppie)

| frame della GT | Spearman con D_GT grezzo |
|---|---|
| raw | 1.000 [1.000, 1.000] |
| maxabs (per mesh) | 0.785 [0.729, 0.835] |
| area (per mesh) | 0.795 [0.741, 0.839] |
| rms (per mesh) | 0.791 [0.733, 0.836] |
| global (one similarity) | 1.000 [1.000, 1.000] |

## Held-out ICT (seed 1234), contro la D_GT ICT grezza

Stessi frame e stesso codice, sulle mesh grezze di `datasets/ICT/topo` (vista `ict_heldout/raw_view`: quelle di eval_view_heldout sono gia' maxabs per mesh); c0, s0 stimati sulle altre 400 identita' del pool. D_GT: `datasets/ICT/gt/ict_matrix_distances_raw.npz` (quella di train_ready e' maxabs).

| frame | Same topology | Tessellation | Perturbation (noisy) | Crop |
|---|---|---|---|---|
| maxabs (per mesh) | 0.373 [0.261, 0.487] | 0.174 [0.110, 0.238] | 0.195 [0.135, 0.261] | 0.175 [0.108, 0.249] |
| global (one similarity) | 0.919 [0.899, 0.937] | 0.877 [0.848, 0.902] | 0.882 [0.852, 0.904] | 0.803 [0.763, 0.834] |
| global scale, per-mesh centroid | 0.944 [0.928, 0.956] | 0.442 [0.379, 0.497] | 0.459 [0.417, 0.500] | 0.244 [0.202, 0.282] |
| global centre, per-mesh scale (maxabs divisor) | 0.407 [0.297, 0.524] | 0.332 [0.242, 0.424] | 0.344 [0.250, 0.445] | 0.189 [0.137, 0.248] |
| bbox control (global frame, 4 numbers) | 0.691 [0.609, 0.761] | 0.667 [0.580, 0.740] | 0.656 [0.569, 0.727] | 0.134 [0.069, 0.202] |

| frame della GT ICT | Spearman con D_GT ICT grezzo |
|---|---|
| raw | 1.000 [1.000, 1.000] |
| maxabs (per mesh) | 0.361 [0.255, 0.472] |
| global (one similarity) | 1.000 [1.000, 1.000] |

## Controlli

```
  chamfer_maxabs vs Tabella 2 estesa (/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines): 21 coppie di topologie, max errore relativo 0.00e+00
  chamfer_maxabs vs pipeline faceBench (/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/outlineB/bfm_heldout): 31 coppie di topologie, max errore relativo 0.00e+00
  chamfer_maxabs vs pipeline faceBench ICT (/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/outlineB/ict_heldout): 31 coppie di topologie, max errore relativo 8.15e-07
```
