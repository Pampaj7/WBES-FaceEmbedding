# Token di taglia Multiface (WS3a duro)

Statistiche del training da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ablations_v3/size_token_bfm_s1234.json`: media 10.92692, std 0.03544 (log r in unita' BFM).

Fattore di conversione (a): k = mediana r BFM original / mediana r Multiface tracked = 56575.7 / 123.504 mm = **458.089** (500 mesh BFM original, 2848 mesh tracked).

Token standardizzato per variante e topologia:

| variante | topologia | mesh | media | std | min | max |
|---|---|---|---|---|---|---|
| a_bfmunits | tracked | 2848 | +0.536 | 1.085 | -1.864 | +1.990 |
| a_bfmunits | remesh | 2848 | -0.553 | 1.110 | -2.953 | +0.979 |
| a_bfmunits | crop | 2848 | -3.994 | 1.024 | -5.905 | -2.164 |
| a_bfmunits | noisy | 2848 | -0.079 | 1.058 | -2.471 | +1.505 |
| a_bfmunits | down | 2848 | +0.540 | 1.086 | -1.864 | +1.996 |
| a_bfmunits | up | 2848 | +0.176 | 1.147 | -2.249 | +1.695 |
| b_neutral | tracked | 2848 | +0.000 | 0.000 | +0.000 | +0.000 |
| b_neutral | remesh | 2848 | +0.000 | 0.000 | +0.000 | +0.000 |
| b_neutral | crop | 2848 | +0.000 | 0.000 | +0.000 | +0.000 |
| b_neutral | noisy | 2848 | +0.000 | 0.000 | +0.000 | +0.000 |
| b_neutral | down | 2848 | +0.000 | 0.000 | +0.000 | +0.000 |
| b_neutral | up | 2848 | +0.000 | 0.000 | +0.000 | +0.000 |
