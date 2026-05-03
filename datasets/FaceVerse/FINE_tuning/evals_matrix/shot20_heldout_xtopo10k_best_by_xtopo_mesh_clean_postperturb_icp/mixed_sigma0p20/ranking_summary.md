# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.374977 | 0.608244 | -0.233267 | 0.392028 | 0.602030 | -0.210002 | no |
| mixed | 2.000000e-01 | 2.000000e-01 | 2.000000e-01 | 2.900000 | 0.007000 | 0.211817 | 0.421057 | -0.209240 | 0.187578 | 0.456391 | -0.268813 | no |
