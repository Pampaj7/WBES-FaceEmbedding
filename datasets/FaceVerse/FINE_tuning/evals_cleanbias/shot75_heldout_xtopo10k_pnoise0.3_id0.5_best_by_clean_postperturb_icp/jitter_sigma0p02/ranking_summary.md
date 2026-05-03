# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.575031 | 0.514582 | 0.060448 | 0.597241 | 0.549931 | 0.047310 | yes |
| jitter | 2.000000e-02 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.559601 | 0.581747 | -0.022146 | 0.585628 | 0.606787 | -0.021158 | no |
