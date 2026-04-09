# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.531864 | 0.605202 | -0.073337 | 0.527528 | 0.561029 | -0.033501 | no |
| jitter | 5.000000e-02 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.492245 | 0.649123 | -0.156879 | 0.501052 | 0.640778 | -0.139726 | no |
| translation | 0.000000e+00 | 0.000000e+00 | 5.000000e-02 | 0.000000 | 0.004500 | 0.522663 | 0.607038 | -0.084375 | 0.518774 | 0.558635 | -0.039860 | no |
| rotation | 0.000000e+00 | 5.000000e-02 | 0.000000e+00 | 2.000000 | 0.000000 | 0.501169 | 0.604593 | -0.103425 | 0.500988 | 0.557252 | -0.056264 | no |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 2.000000 | 0.004500 | 0.476996 | 0.638449 | -0.161453 | 0.486990 | 0.634137 | -0.147147 | no |
