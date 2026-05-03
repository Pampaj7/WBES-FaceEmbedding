# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.537433 | 0.403222 | 0.134211 | 0.551494 | 0.406753 | 0.144741 | yes |
| mixed | 2.000000e-02 | 2.000000e-02 | 2.000000e-02 | 1.400000 | 0.003000 | 0.515111 | 0.463925 | 0.051187 | 0.530682 | 0.486240 | 0.044442 | yes |
