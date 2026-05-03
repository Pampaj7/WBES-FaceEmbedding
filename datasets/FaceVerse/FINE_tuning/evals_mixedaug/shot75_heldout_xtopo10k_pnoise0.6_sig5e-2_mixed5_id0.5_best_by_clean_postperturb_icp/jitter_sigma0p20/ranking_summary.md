# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.590712 | 0.514582 | 0.076130 | 0.616718 | 0.549931 | 0.066787 | yes |
| jitter | 2.000000e-01 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.329115 | 0.498846 | -0.169731 | 0.372453 | 0.554975 | -0.182522 | no |
