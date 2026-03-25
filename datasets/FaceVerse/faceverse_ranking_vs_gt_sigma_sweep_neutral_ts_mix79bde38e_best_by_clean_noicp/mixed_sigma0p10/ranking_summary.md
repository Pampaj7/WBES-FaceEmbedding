# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.577300 | 0.639175 | -0.061875 | 0.588041 | 0.633116 | -0.045075 | no |
| mixed | 1.000000e-01 | 1.000000e-01 | 1.000000e-01 | 3.000000 | 0.007000 | 0.344280 | 0.555300 | -0.211020 | 0.367685 | 0.598607 | -0.230922 | no |
