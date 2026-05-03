# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.121101 | 0.598907 | -0.477807 | 0.121568 | 0.590748 | -0.469180 | no |
| rotation | 0.000000e+00 | 1.000000e-01 | 0.000000e+00 | 1.700000 | 0.000000 | 0.122895 | 0.601115 | -0.478220 | 0.123407 | 0.597810 | -0.474402 | no |
