# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.575304 | 0.514582 | 0.060722 | 0.597577 | 0.549931 | 0.047646 | yes |
| jitter | 1.000000e-01 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.468604 | 0.592527 | -0.123923 | 0.492450 | 0.633615 | -0.141165 | no |
