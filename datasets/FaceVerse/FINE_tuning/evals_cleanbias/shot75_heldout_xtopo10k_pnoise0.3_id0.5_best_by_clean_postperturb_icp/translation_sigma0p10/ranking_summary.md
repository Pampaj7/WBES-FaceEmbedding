# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.575031 | 0.514582 | 0.060448 | 0.597241 | 0.549931 | 0.047310 | yes |
| translation | 0.000000e+00 | 0.000000e+00 | 1.000000e-01 | 0.000000 | 0.004000 | 0.563575 | 0.530675 | 0.032900 | 0.588215 | 0.568593 | 0.019622 | yes |
