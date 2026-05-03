# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.111549 | 0.597833 | -0.486284 | 0.115549 | 0.586700 | -0.471151 | no |
| mixed | 2.000000e-02 | 2.000000e-02 | 2.000000e-02 | 0.740000 | 0.001600 | 0.127560 | 0.631415 | -0.503855 | 0.133781 | 0.625946 | -0.492165 | no |
