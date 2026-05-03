# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.575031 | 0.514582 | 0.060448 | 0.597241 | 0.549931 | 0.047310 | yes |
| translation | 0.000000e+00 | 0.000000e+00 | 1.500000e-01 | 0.000000 | 0.005500 | 0.554897 | 0.531604 | 0.023293 | 0.579913 | 0.565907 | 0.014006 | yes |
