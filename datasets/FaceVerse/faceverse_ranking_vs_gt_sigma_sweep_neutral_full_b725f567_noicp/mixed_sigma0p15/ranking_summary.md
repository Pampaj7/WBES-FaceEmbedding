# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.545040 | 0.639175 | -0.094135 | 0.539079 | 0.633116 | -0.094037 | no |
| mixed | 1.500000e-01 | 1.500000e-01 | 1.500000e-01 | 4.000000 | 0.009500 | 0.216860 | 0.471810 | -0.254950 | 0.222394 | 0.529995 | -0.307601 | no |
