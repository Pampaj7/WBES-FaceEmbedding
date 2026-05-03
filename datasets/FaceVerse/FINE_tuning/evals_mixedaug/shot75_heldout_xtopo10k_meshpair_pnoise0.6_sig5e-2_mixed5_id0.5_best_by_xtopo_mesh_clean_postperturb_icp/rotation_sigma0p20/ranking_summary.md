# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.537433 | 0.403222 | 0.134211 | 0.551494 | 0.406753 | 0.144741 | yes |
| rotation | 0.000000e+00 | 2.000000e-01 | 0.000000e+00 | 5.000000 | 0.000000 | 0.443250 | 0.410675 | 0.032575 | 0.462829 | 0.434862 | 0.027968 | yes |
