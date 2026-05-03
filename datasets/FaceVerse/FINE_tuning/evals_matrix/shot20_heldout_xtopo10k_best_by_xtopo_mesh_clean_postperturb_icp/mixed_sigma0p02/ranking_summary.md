# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.374977 | 0.608244 | -0.233267 | 0.392028 | 0.602030 | -0.210002 | no |
| mixed | 2.000000e-02 | 2.000000e-02 | 2.000000e-02 | 0.740000 | 0.001600 | 0.396931 | 0.645270 | -0.248338 | 0.407593 | 0.642902 | -0.235309 | no |
