# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.590712 | 0.514582 | 0.076130 | 0.616718 | 0.549931 | 0.066787 | yes |
| translation | 0.000000e+00 | 0.000000e+00 | 5.000000e-02 | 0.000000 | 0.004500 | 0.573694 | 0.532889 | 0.040806 | 0.604093 | 0.568599 | 0.035494 | yes |
