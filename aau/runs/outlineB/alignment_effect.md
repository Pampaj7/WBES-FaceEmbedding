# tab:alignment_effect su held-out (OUTLINE_B §4, esperimento 1)

Spearman vs D_GT, CI 95% bootstrap per soggetto (1000 repliche). Colonne: same = original->original (4950 coppie), cross = no-crop cross-topology (99000). Senza suffisso: 4096 punti campionati per mesh; '2048 pt': il campionamento della colonna cross del paper.

## original_to_original

| metodo | BFM held-out | BFM held-out, 2048 pt | ICT held-out (GT maxabs) | ICT held-out (GT maxabs), 2048 pt | ICT held-out (GT raw) | ICT held-out (GT raw), 2048 pt | fb100 (gate) | fb100 (gate), 2048 pt |
|---|---|---|---|---|---|---|---|---|
| Chamfer | 0.644 [0.580, 0.707] | 0.643 [0.579, 0.707] | 0.979 [0.969, 0.985] | 0.979 [0.969, 0.985] | 0.373 [0.259, 0.484] | 0.375 [0.261, 0.486] | 0.729 [0.666, 0.787] | 0.729 [0.666, 0.787] |
| Rigid ICP + Chamfer | 0.488 [0.411, 0.570] | 0.479 [0.406, 0.559] | 0.872 [0.832, 0.904] | 0.872 [0.832, 0.903] | 0.384 [0.274, 0.497] | 0.384 [0.276, 0.493] | 0.600 [0.528, 0.664] | 0.587 [0.516, 0.650] |
| Rigid ICP + NICP + P2P | 0.420 [0.327, 0.506] | 0.437 [0.346, 0.518] | 0.776 [0.712, 0.830] | 0.780 [0.714, 0.835] | 0.413 [0.298, 0.518] | 0.408 [0.293, 0.515] | 0.456 [0.361, 0.537] | 0.455 [0.358, 0.540] |
| Rigid ICP + NICP + P2Tri | 0.427 [0.335, 0.508] | 0.445 [0.357, 0.527] | 0.778 [0.712, 0.829] | 0.787 [0.721, 0.837] | 0.415 [0.287, 0.526] | 0.413 [0.288, 0.524] | 0.468 [0.378, 0.550] | 0.475 [0.386, 0.560] |
| M3DFB RLR + Chamfer | 0.501 [0.417, 0.580] | -- | 0.432 [0.347, 0.512] | -- | 0.428 [0.328, 0.524] | -- | -- | -- |

## nocrop_cross_topology

| metodo | BFM held-out | BFM held-out, 2048 pt | ICT held-out (GT maxabs) | ICT held-out (GT maxabs), 2048 pt | ICT held-out (GT raw) | ICT held-out (GT raw), 2048 pt | fb100 (gate) | fb100 (gate), 2048 pt |
|---|---|---|---|---|---|---|---|---|
| Chamfer | 0.469 [0.399, 0.532] | 0.470 [0.400, 0.533] | 0.433 [0.357, 0.498] | 0.437 [0.360, 0.501] | 0.178 [0.116, 0.245] | 0.179 [0.117, 0.247] | 0.552 [0.490, 0.607] | 0.552 [0.491, 0.607] |
| Rigid ICP + Chamfer | 0.415 [0.347, 0.487] | 0.413 [0.347, 0.486] | 0.645 [0.577, 0.702] | 0.642 [0.573, 0.700] | 0.304 [0.217, 0.388] | 0.300 [0.213, 0.383] | 0.521 [0.454, 0.573] | 0.514 [0.448, 0.567] |
| Rigid ICP + NICP + P2P | 0.388 [0.300, 0.473] | 0.391 [0.309, 0.471] | 0.600 [0.508, 0.673] | 0.596 [0.500, 0.672] | 0.356 [0.255, 0.455] | 0.341 [0.242, 0.443] | 0.422 [0.334, 0.507] | 0.414 [0.327, 0.496] |
| Rigid ICP + NICP + P2Tri | 0.395 [0.311, 0.479] | 0.403 [0.322, 0.484] | 0.613 [0.523, 0.684] | 0.606 [0.513, 0.679] | 0.367 [0.262, 0.462] | 0.350 [0.246, 0.443] | 0.432 [0.335, 0.518] | 0.433 [0.334, 0.516] |
| M3DFB RLR + Chamfer | 0.473 [0.393, 0.549] | -- | 0.229 [0.176, 0.282] | -- | 0.223 [0.166, 0.288] | -- | -- | -- |

## tessellation_cross_topology

| metodo | BFM held-out | BFM held-out, 2048 pt | ICT held-out (GT maxabs) | ICT held-out (GT maxabs), 2048 pt | ICT held-out (GT raw) | ICT held-out (GT raw), 2048 pt | fb100 (gate) | fb100 (gate), 2048 pt |
|---|---|---|---|---|---|---|---|---|
| Chamfer | 0.454 [0.383, 0.519] | 0.454 [0.383, 0.518] | 0.417 [0.339, 0.486] | 0.420 [0.341, 0.489] | 0.174 [0.113, 0.239] | 0.176 [0.114, 0.241] | 0.534 [0.472, 0.586] | 0.533 [0.471, 0.586] |
| Rigid ICP + Chamfer | 0.399 [0.329, 0.468] | 0.397 [0.327, 0.464] | 0.629 [0.558, 0.689] | 0.625 [0.552, 0.687] | 0.300 [0.219, 0.393] | 0.296 [0.216, 0.388] | 0.504 [0.440, 0.563] | 0.497 [0.431, 0.557] |
| Rigid ICP + NICP + P2P | 0.381 [0.286, 0.462] | 0.382 [0.293, 0.462] | 0.593 [0.495, 0.670] | 0.593 [0.490, 0.674] | 0.359 [0.261, 0.453] | 0.346 [0.244, 0.441] | 0.415 [0.325, 0.499] | 0.407 [0.319, 0.488] |
| Rigid ICP + NICP + P2Tri | 0.388 [0.301, 0.473] | 0.393 [0.311, 0.476] | 0.603 [0.515, 0.676] | 0.600 [0.509, 0.674] | 0.369 [0.269, 0.469] | 0.353 [0.249, 0.457] | 0.426 [0.340, 0.514] | 0.425 [0.339, 0.512] |
| M3DFB RLR + Chamfer | 0.501 [0.422, 0.577] | -- | 0.222 [0.165, 0.273] | -- | 0.214 [0.154, 0.273] | -- | -- | -- |

## perturbation_cross_topology

| metodo | BFM held-out | BFM held-out, 2048 pt | ICT held-out (GT maxabs) | ICT held-out (GT maxabs), 2048 pt | ICT held-out (GT raw) | ICT held-out (GT raw), 2048 pt | fb100 (gate) | fb100 (gate), 2048 pt |
|---|---|---|---|---|---|---|---|---|
| Chamfer | 0.502 [0.431, 0.566] | 0.500 [0.431, 0.565] | 0.480 [0.416, 0.541] | 0.481 [0.417, 0.542] | 0.195 [0.126, 0.258] | 0.195 [0.126, 0.259] | 0.586 [0.526, 0.644] | 0.585 [0.524, 0.643] |
| Rigid ICP + Chamfer | 0.445 [0.373, 0.515] | 0.441 [0.370, 0.509] | 0.681 [0.613, 0.734] | 0.678 [0.609, 0.732] | 0.315 [0.223, 0.405] | 0.311 [0.220, 0.401] | 0.551 [0.485, 0.614] | 0.543 [0.474, 0.605] |
| Rigid ICP + NICP + P2P | 0.401 [0.316, 0.484] | 0.407 [0.323, 0.490] | 0.621 [0.539, 0.695] | 0.611 [0.526, 0.687] | 0.357 [0.260, 0.451] | 0.340 [0.246, 0.432] | 0.434 [0.342, 0.520] | 0.425 [0.333, 0.514] |
| Rigid ICP + NICP + P2Tri | 0.408 [0.327, 0.482] | 0.419 [0.343, 0.492] | 0.635 [0.555, 0.707] | 0.623 [0.539, 0.695] | 0.369 [0.266, 0.462] | 0.349 [0.250, 0.443] | 0.444 [0.355, 0.529] | 0.445 [0.356, 0.530] |
| M3DFB RLR + Chamfer | 0.448 [0.366, 0.521] | -- | 0.264 [0.201, 0.320] | -- | 0.260 [0.191, 0.329] | -- | -- | -- |

## Gate su facebench_first100 contro tab:alignment_effect pubblicata

```
  [OK] bfm_fb100 chamfer/original_to_original: qui 0.7295 [0.666, 0.787] vs paper 0.7295 [0.667, 0.788] (delta +0.00000)
  [info] bfm_fb100 chamfer/nocrop_cross_topology: qui 0.5518 [0.490, 0.607] vs paper 0.5525 [0.488, 0.606] (delta -0.00064)
  [OK] bfm_fb100 rigid_icp_chamfer/original_to_original: qui 0.5995 [0.528, 0.664] vs paper 0.5995 [0.528, 0.663] (delta +0.00000)
  [info] bfm_fb100 rigid_icp_chamfer/nocrop_cross_topology: qui 0.5206 [0.454, 0.573] vs paper 0.5144 [0.449, 0.573] (delta +0.00624)
  [OK] bfm_fb100 nicp_p2p/original_to_original: qui 0.4561 [0.361, 0.537] vs paper 0.4561 [0.357, 0.544] (delta +0.00000)
  [info] bfm_fb100 nicp_p2p/nocrop_cross_topology: qui 0.4216 [0.334, 0.507] vs paper 0.4140 [0.326, 0.499] (delta +0.00760)
  [OK] bfm_fb100 nicp_p2tri/original_to_original: qui 0.4676 [0.378, 0.550] vs paper 0.4676 [0.373, 0.552] (delta +0.00000)
  [info] bfm_fb100 nicp_p2tri/nocrop_cross_topology: qui 0.4324 [0.335, 0.518] vs paper 0.4327 [0.350, 0.514] (delta -0.00034)
  [info] bfm_fb100_s2048 chamfer/original_to_original: qui 0.7290 [0.666, 0.787] vs paper 0.7295 [0.667, 0.788] (delta -0.00043)
  [OK] bfm_fb100_s2048 chamfer/nocrop_cross_topology: qui 0.5525 [0.491, 0.607] vs paper 0.5525 [0.488, 0.606] (delta +0.00000)
  [info] bfm_fb100_s2048 rigid_icp_chamfer/original_to_original: qui 0.5870 [0.516, 0.650] vs paper 0.5995 [0.528, 0.663] (delta -0.01251)
  [OK] bfm_fb100_s2048 rigid_icp_chamfer/nocrop_cross_topology: qui 0.5144 [0.448, 0.567] vs paper 0.5144 [0.449, 0.573] (delta +0.00000)
  [info] bfm_fb100_s2048 nicp_p2p/original_to_original: qui 0.4550 [0.358, 0.540] vs paper 0.4561 [0.357, 0.544] (delta -0.00110)
  [OK] bfm_fb100_s2048 nicp_p2p/nocrop_cross_topology: qui 0.4140 [0.327, 0.496] vs paper 0.4140 [0.326, 0.499] (delta +0.00000)
  [info] bfm_fb100_s2048 nicp_p2tri/original_to_original: qui 0.4745 [0.386, 0.560] vs paper 0.4676 [0.373, 0.552] (delta +0.00699)
  [OK] bfm_fb100_s2048 nicp_p2tri/nocrop_cross_topology: qui 0.4327 [0.334, 0.516] vs paper 0.4327 [0.350, 0.514] (delta +0.00000)
```
