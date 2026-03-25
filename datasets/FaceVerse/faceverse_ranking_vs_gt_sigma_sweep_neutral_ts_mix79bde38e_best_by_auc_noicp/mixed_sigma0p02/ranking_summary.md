# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.585670 | 0.639175 | -0.053504 | 0.600368 | 0.633116 | -0.032748 | no |
| mixed | 2.000000e-02 | 2.000000e-02 | 2.000000e-02 | 1.400000 | 0.003000 | 0.552906 | 0.639612 | -0.086706 | 0.578271 | 0.626612 | -0.048340 | no |
