# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.537433 | 0.403222 | 0.134211 | 0.551494 | 0.406753 | 0.144741 | yes |
| translation | 0.000000e+00 | 0.000000e+00 | 2.000000e-01 | 0.000000 | 0.012000 | 0.407103 | 0.416705 | -0.009603 | 0.420111 | 0.422747 | -0.002636 | no |
