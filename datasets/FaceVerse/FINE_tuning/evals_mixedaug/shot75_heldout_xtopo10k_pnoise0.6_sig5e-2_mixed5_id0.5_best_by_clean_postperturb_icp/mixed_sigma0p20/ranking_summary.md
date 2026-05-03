# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.590712 | 0.514582 | 0.076130 | 0.616718 | 0.549931 | 0.066787 | yes |
| mixed | 2.000000e-01 | 2.000000e-01 | 2.000000e-01 | 5.000000 | 0.012000 | 0.304433 | 0.447222 | -0.142789 | 0.342260 | 0.488275 | -0.146015 | no |
