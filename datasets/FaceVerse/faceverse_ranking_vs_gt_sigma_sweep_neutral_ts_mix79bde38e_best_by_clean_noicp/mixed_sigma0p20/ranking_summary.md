# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.577300 | 0.639175 | -0.061875 | 0.588041 | 0.633116 | -0.045075 | no |
| mixed | 2.000000e-01 | 2.000000e-01 | 2.000000e-01 | 5.000000 | 0.012000 | 0.173594 | 0.392031 | -0.218437 | 0.177035 | 0.444181 | -0.267147 | no |
