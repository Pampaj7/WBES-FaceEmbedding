# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.590774 | 0.514582 | 0.076192 | 0.616605 | 0.549931 | 0.066674 | yes |
| rotation | 0.000000e+00 | 5.000000e-02 | 0.000000e+00 | 2.000000 | 0.000000 | 0.569850 | 0.524036 | 0.045814 | 0.603942 | 0.564367 | 0.039575 | yes |
