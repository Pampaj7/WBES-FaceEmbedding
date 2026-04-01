# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.531864 | 0.639175 | -0.107311 | 0.527528 | 0.633116 | -0.105588 | no |
| jitter | 5.000000e-02 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.492245 | 0.647659 | -0.155414 | 0.501052 | 0.635451 | -0.134399 | no |
| translation | 0.000000e+00 | 0.000000e+00 | 5.000000e-02 | 0.000000 | 0.004500 | 0.522663 | 0.633143 | -0.110481 | 0.518774 | 0.627198 | -0.108424 | no |
| rotation | 0.000000e+00 | 5.000000e-02 | 0.000000e+00 | 2.000000 | 0.000000 | 0.501169 | 0.630175 | -0.129006 | 0.500988 | 0.629645 | -0.128657 | no |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 2.000000 | 0.004500 | 0.476996 | 0.627053 | -0.150057 | 0.486990 | 0.626425 | -0.139435 | no |
