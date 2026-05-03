# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.525033 | 0.570883 | -0.045850 | 0.532839 | 0.543788 | -0.010950 | no |
| mixed | 1.500000e-01 | 1.500000e-01 | 1.500000e-01 | 2.300000 | 0.005500 | 0.426877 | 0.425823 | 0.001054 | 0.401163 | 0.418540 | -0.017378 | yes |
