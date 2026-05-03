# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.525033 | 0.570883 | -0.045850 | 0.532839 | 0.543788 | -0.010950 | no |
| mixed | 1.000000e-01 | 1.000000e-01 | 1.000000e-01 | 1.700000 | 0.004000 | 0.468511 | 0.457312 | 0.011199 | 0.459995 | 0.489019 | -0.029024 | yes |
