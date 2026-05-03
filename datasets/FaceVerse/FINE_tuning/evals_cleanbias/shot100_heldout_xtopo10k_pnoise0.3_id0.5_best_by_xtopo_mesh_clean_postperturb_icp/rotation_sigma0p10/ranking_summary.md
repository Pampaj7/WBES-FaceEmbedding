# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.522661 | 0.570883 | -0.048221 | 0.531925 | 0.543788 | -0.011863 | no |
| rotation | 0.000000e+00 | 1.000000e-01 | 0.000000e+00 | 1.700000 | 0.000000 | 0.533070 | 0.632016 | -0.098946 | 0.522064 | 0.581389 | -0.059326 | no |
