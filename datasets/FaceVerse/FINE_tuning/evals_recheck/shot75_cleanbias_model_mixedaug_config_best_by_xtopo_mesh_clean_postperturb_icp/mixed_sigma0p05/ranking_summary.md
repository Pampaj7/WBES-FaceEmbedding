# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.575304 | 0.514582 | 0.060722 | 0.597577 | 0.549931 | 0.047646 | yes |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 1.100000 | 0.002500 | 0.599311 | 0.608408 | -0.009097 | 0.615554 | 0.632144 | -0.016590 | no |
