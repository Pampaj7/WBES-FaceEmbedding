# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.374977 | 0.608244 | -0.233267 | 0.392028 | 0.602030 | -0.210002 | no |
| mixed | 1.000000e-01 | 1.000000e-01 | 1.000000e-01 | 1.700000 | 0.004000 | 0.305875 | 0.572943 | -0.267068 | 0.297510 | 0.622081 | -0.324571 | no |
