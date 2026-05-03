# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.527009 | 0.570883 | -0.043874 | 0.532420 | 0.543788 | -0.011368 | no |
| mixed | 1.500000e-01 | 1.500000e-01 | 1.500000e-01 | 2.300000 | 0.005500 | 0.427141 | 0.425823 | 0.001318 | 0.400387 | 0.418540 | -0.018153 | yes |
