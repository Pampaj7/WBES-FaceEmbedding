# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.575304 | 0.514582 | 0.060722 | 0.597577 | 0.549931 | 0.047646 | yes |
| rotation | 0.000000e+00 | 2.000000e-01 | 0.000000e+00 | 5.000000 | 0.000000 | 0.509918 | 0.532205 | -0.022287 | 0.540493 | 0.575089 | -0.034596 | no |
