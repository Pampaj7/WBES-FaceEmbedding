# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.111549 | 0.597833 | -0.486284 | 0.115549 | 0.586700 | -0.471151 | no |
| rotation | 0.000000e+00 | 1.000000e-01 | 0.000000e+00 | 1.700000 | 0.000000 | 0.110400 | 0.598595 | -0.488195 | 0.115038 | 0.594338 | -0.479300 | no |
