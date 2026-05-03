# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.525033 | 0.570883 | -0.045850 | 0.532839 | 0.543788 | -0.010950 | no |
| mixed | 2.000000e-02 | 2.000000e-02 | 2.000000e-02 | 0.740000 | 0.001600 | 0.528854 | 0.644796 | -0.115942 | 0.524713 | 0.621161 | -0.096448 | no |
