# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.473790 | 0.539292 | -0.065502 | 0.471090 | 0.526986 | -0.055896 | no |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 1.100000 | 0.002500 | 0.492092 | 0.592701 | -0.100609 | 0.488726 | 0.588973 | -0.100246 | no |
