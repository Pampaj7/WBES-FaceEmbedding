# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.527009 | 0.570883 | -0.043874 | 0.532420 | 0.543788 | -0.011368 | no |
| rotation | 0.000000e+00 | 5.000000e-02 | 0.000000e+00 | 1.100000 | 0.000000 | 0.535310 | 0.631094 | -0.095784 | 0.522841 | 0.572958 | -0.050118 | no |
