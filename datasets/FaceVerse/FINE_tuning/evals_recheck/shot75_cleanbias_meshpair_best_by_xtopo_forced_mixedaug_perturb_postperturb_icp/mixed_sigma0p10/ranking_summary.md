# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.511733 | 0.403222 | 0.108511 | 0.520214 | 0.406753 | 0.113461 | yes |
| mixed | 1.000000e-01 | 1.000000e-01 | 1.000000e-01 | 3.000000 | 0.007000 | 0.362153 | 0.463479 | -0.101326 | 0.399167 | 0.493235 | -0.094068 | no |
