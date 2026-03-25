# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.577300 | 0.639175 | -0.061875 | 0.588041 | 0.633116 | -0.045075 | no |
| mixed | 1.500000e-01 | 1.500000e-01 | 1.500000e-01 | 4.000000 | 0.009500 | 0.243023 | 0.471810 | -0.228787 | 0.252070 | 0.529995 | -0.277925 | no |
