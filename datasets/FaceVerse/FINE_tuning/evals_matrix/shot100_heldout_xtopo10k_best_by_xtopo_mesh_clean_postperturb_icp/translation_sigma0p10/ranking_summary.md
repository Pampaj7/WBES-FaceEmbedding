# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.525033 | 0.570883 | -0.045850 | 0.532839 | 0.543788 | -0.010950 | no |
| translation | 0.000000e+00 | 0.000000e+00 | 1.000000e-01 | 0.000000 | 0.004000 | 0.544005 | 0.634651 | -0.090646 | 0.520120 | 0.598301 | -0.078181 | no |
