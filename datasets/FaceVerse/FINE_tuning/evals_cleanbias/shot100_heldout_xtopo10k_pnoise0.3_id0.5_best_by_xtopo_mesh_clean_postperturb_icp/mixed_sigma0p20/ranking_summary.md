# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.522661 | 0.570883 | -0.048221 | 0.531925 | 0.543788 | -0.011863 | no |
| mixed | 2.000000e-01 | 2.000000e-01 | 2.000000e-01 | 2.900000 | 0.007000 | 0.362714 | 0.289723 | 0.072991 | 0.366245 | 0.347600 | 0.018645 | yes |
