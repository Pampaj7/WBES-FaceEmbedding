# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.525033 | 0.570883 | -0.045850 | 0.532839 | 0.543788 | -0.010950 | no |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 1.100000 | 0.002500 | 0.526877 | 0.632411 | -0.105534 | 0.510137 | 0.634834 | -0.124697 | no |
