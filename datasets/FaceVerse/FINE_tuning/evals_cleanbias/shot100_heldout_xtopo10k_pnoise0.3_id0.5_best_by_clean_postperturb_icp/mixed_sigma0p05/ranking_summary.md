# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.523979 | 0.570883 | -0.046904 | 0.532734 | 0.543788 | -0.011054 | no |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 1.100000 | 0.002500 | 0.539262 | 0.632411 | -0.093149 | 0.513916 | 0.634834 | -0.120917 | no |
