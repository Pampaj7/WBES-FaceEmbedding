# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.590712 | 0.514582 | 0.076130 | 0.616718 | 0.549931 | 0.066787 | yes |
| mixed | 2.000000e-02 | 2.000000e-02 | 2.000000e-02 | 1.400000 | 0.003000 | 0.580812 | 0.578568 | 0.002243 | 0.604837 | 0.610466 | -0.005629 | yes |
