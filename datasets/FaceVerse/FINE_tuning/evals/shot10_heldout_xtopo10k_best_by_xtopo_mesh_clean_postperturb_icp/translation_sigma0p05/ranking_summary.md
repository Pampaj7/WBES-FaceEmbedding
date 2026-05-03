# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.121101 | 0.598907 | -0.477807 | 0.121568 | 0.590748 | -0.469180 | no |
| translation | 0.000000e+00 | 0.000000e+00 | 5.000000e-02 | 0.000000 | 0.002500 | 0.118421 | 0.602252 | -0.483832 | 0.118393 | 0.593893 | -0.475500 | no |
