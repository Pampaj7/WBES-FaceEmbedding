# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.523979 | 0.570883 | -0.046904 | 0.532734 | 0.543788 | -0.011054 | no |
| mixed | 1.000000e-01 | 1.000000e-01 | 1.000000e-01 | 1.700000 | 0.004000 | 0.480369 | 0.457312 | 0.023057 | 0.476668 | 0.489019 | -0.012350 | yes |
