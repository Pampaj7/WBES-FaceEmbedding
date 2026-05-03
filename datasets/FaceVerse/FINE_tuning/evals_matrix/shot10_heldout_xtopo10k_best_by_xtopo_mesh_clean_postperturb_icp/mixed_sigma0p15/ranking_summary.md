# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.121101 | 0.598907 | -0.477807 | 0.121568 | 0.590748 | -0.469180 | no |
| mixed | 1.500000e-01 | 1.500000e-01 | 1.500000e-01 | 2.300000 | 0.005500 | 0.101079 | 0.460382 | -0.359303 | 0.101544 | 0.507688 | -0.406144 | no |
