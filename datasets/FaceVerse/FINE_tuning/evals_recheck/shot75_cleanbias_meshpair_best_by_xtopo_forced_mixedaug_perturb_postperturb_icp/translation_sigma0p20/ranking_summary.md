# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.511733 | 0.403222 | 0.108511 | 0.520214 | 0.406753 | 0.113461 | yes |
| translation | 0.000000e+00 | 0.000000e+00 | 2.000000e-01 | 0.000000 | 0.012000 | 0.391764 | 0.416705 | -0.024941 | 0.407124 | 0.422747 | -0.015623 | no |
