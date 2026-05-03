# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.590774 | 0.514582 | 0.076192 | 0.616605 | 0.549931 | 0.066674 | yes |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 2.000000 | 0.004500 | 0.571506 | 0.600352 | -0.028846 | 0.600280 | 0.626735 | -0.026455 | no |
