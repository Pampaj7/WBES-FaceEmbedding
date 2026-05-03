# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.511733 | 0.403222 | 0.108511 | 0.520214 | 0.406753 | 0.113461 | yes |
| translation | 0.000000e+00 | 0.000000e+00 | 1.500000e-01 | 0.000000 | 0.009500 | 0.429283 | 0.416080 | 0.013203 | 0.446314 | 0.423381 | 0.022933 | yes |
