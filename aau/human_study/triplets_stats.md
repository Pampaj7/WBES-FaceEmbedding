# Triplette per lo studio umano (WS4)

Generato il 2026-09-11 02:25 UTC da `aau/human_study/select_triplets.py`, seed 1234.

| parametro | valore |
|---|---|
| soggetti | 100 (heldout, topologia `original`) |
| metriche | gt, chamfer, lpips, latent |
| pool completo | 485100 triplette (A, {B, C}) |
| margine relativo minimo | 0.15 (decisiva) / 0.50 (controllo) |
| triplette in disaccordo | 40699 |
| selezionate | 300 test + 30 controllo |

## Tipi di disaccordo

Squadre di metriche che si contraddicono, con il margine relativo richiesto su ognuna. L'etichetta e' invariante allo scambio B<->C.

| tipo | disponibili | scelte |
|---|---:|---:|
| `gt+latent_vs_lpips` | 6105 | 12 |
| `latent_vs_lpips` | 5900 | 12 |
| `chamfer+lpips_vs_gt` | 4531 | 12 |
| `chamfer+lpips_vs_latent` | 3547 | 12 |
| `gt_vs_lpips` | 3545 | 12 |
| `chamfer+lpips_vs_gt+latent` | 3150 | 12 |
| `chamfer_vs_gt` | 2727 | 12 |
| `chamfer_vs_gt+latent` | 2229 | 12 |
| `chamfer_vs_latent` | 1901 | 12 |
| `gt_vs_latent` | 1645 | 12 |
| `chamfer+lpips+latent_vs_gt` | 1123 | 12 |
| `gt+chamfer+latent_vs_lpips` | 779 | 12 |
| `chamfer_vs_lpips` | 559 | 12 |
| `chamfer+latent_vs_gt` | 558 | 12 |
| `chamfer+latent_vs_lpips` | 533 | 13 |
| `gt+lpips_vs_latent` | 499 | 13 |
| `gt_vs_lpips+latent` | 412 | 13 |
| `gt+chamfer+lpips_vs_latent` | 301 | 13 |
| `chamfer_vs_gt+lpips` | 189 | 13 |
| `gt+chamfer_vs_latent` | 150 | 13 |
| `chamfer_vs_gt+lpips+latent` | 119 | 13 |
| `gt+chamfer_vs_lpips` | 103 | 13 |
| `chamfer_vs_lpips+latent` | 54 | 13 |
| `chamfer+latent_vs_gt+lpips` | 38 | 13 |
| `gt+chamfer_vs_lpips+latent` | 2 | 2 |

## Margini nelle triplette scelte

| metrica | mediana test | min test | decisiva in test | mediana controllo |
|---|---:|---:|---:|---:|
| gt | 0.190 | 0.001 | 226/300 | 0.826 |
| chamfer | 0.180 | 0.000 | 226/300 | 0.828 |
| lpips | 0.194 | 0.002 | 227/300 | 0.837 |
| latent | 0.200 | 0.002 | 226/300 | 0.902 |

## Controlli (attention check)

- 30 triplette con accordo unanime di tutte le metriche e margine relativo >= 0.50 su ognuna.
- candidate: 7746; estratte dalle prime 300 per margine minimo.
- margine minimo (sulla metrica peggiore) nelle scelte: mediana 0.788, minimo 0.743.

## Copertura dei soggetti

- 100 soggetti distinti sui 100 disponibili.
- comparse per soggetto: min 1, mediana 9, max 30.

## Pacchetto

- 100 render frontali PNG 512x512 in `img/` (12.2 MB).
- `triplets.json` (dati + risposte attese) e `triplets.js` (stesso contenuto, caricato da `index.html` con un tag `<script>` perche' `fetch()` su `file://` e' bloccato dal browser).

## Riproduzione

```bash
srun -p cpu --mem=16G --time=00:20:00 \
    env AAU_NV= aau/run.sh aau/human_study/select_triplets.py \
    --seed 1234 --n-test 300 --n-control 30 \
    --margin 0.15 --control-margin 0.5
```
