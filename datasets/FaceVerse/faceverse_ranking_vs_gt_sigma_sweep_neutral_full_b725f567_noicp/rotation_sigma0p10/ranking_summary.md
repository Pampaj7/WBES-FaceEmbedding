# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.545040 | 0.639175 | -0.094135 | 0.539079 | 0.633116 | -0.094037 | no |
| rotation | 0.000000e+00 | 1.000000e-01 | 0.000000e+00 | 3.000000 | 0.000000 | 0.499399 | 0.611587 | -0.112188 | 0.500413 | 0.621758 | -0.121345 | no |
