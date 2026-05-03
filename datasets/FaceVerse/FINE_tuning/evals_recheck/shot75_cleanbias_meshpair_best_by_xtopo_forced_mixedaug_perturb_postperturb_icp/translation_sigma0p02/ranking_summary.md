# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.511733 | 0.403222 | 0.108511 | 0.520214 | 0.406753 | 0.113461 | yes |
| translation | 0.000000e+00 | 0.000000e+00 | 2.000000e-02 | 0.000000 | 0.003000 | 0.498404 | 0.413499 | 0.084905 | 0.508515 | 0.422060 | 0.086455 | yes |
