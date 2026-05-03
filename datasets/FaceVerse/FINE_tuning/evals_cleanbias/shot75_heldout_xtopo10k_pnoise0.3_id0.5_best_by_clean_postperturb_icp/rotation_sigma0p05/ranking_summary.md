# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.575031 | 0.514582 | 0.060448 | 0.597241 | 0.549931 | 0.047310 | yes |
| rotation | 0.000000e+00 | 5.000000e-02 | 0.000000e+00 | 1.100000 | 0.000000 | 0.566975 | 0.531130 | 0.035846 | 0.594047 | 0.570598 | 0.023449 | yes |
