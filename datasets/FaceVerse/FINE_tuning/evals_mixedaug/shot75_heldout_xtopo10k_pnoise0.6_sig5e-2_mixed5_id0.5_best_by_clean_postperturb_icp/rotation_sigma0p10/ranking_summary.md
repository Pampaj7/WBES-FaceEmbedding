# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.590712 | 0.514582 | 0.076130 | 0.616718 | 0.549931 | 0.066787 | yes |
| rotation | 0.000000e+00 | 1.000000e-01 | 0.000000e+00 | 3.000000 | 0.000000 | 0.553102 | 0.524510 | 0.028592 | 0.590420 | 0.567240 | 0.023180 | yes |
