# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.111549 | 0.597833 | -0.486284 | 0.115549 | 0.586700 | -0.471151 | no |
| mixed | 2.000000e-01 | 2.000000e-01 | 2.000000e-01 | 2.900000 | 0.007000 | 0.093476 | 0.379113 | -0.285637 | 0.091642 | 0.415110 | -0.323468 | no |
