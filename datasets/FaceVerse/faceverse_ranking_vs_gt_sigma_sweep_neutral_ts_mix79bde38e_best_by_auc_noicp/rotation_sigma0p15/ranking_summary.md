# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.585670 | 0.639175 | -0.053504 | 0.600368 | 0.633116 | -0.032748 | no |
| rotation | 0.000000e+00 | 1.500000e-01 | 0.000000e+00 | 4.000000 | 0.000000 | 0.514300 | 0.582002 | -0.067701 | 0.535674 | 0.604934 | -0.069260 | no |
