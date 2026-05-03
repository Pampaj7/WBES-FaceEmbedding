# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.590774 | 0.514582 | 0.076192 | 0.616605 | 0.549931 | 0.066674 | yes |
| jitter | 1.000000e-01 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.482397 | 0.592527 | -0.110130 | 0.525135 | 0.633615 | -0.108480 | no |
