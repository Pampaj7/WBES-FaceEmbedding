# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.121101 | 0.598907 | -0.477807 | 0.121568 | 0.590748 | -0.469180 | no |
| mixed | 2.000000e-02 | 2.000000e-02 | 2.000000e-02 | 0.740000 | 0.001600 | 0.137528 | 0.634249 | -0.496721 | 0.139191 | 0.630321 | -0.491130 | no |
