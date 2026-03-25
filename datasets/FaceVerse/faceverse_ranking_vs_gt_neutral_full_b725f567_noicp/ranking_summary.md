# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.545040 | 0.639175 | -0.094135 | 0.539079 | 0.633116 | -0.094037 | no |
| jitter | 5.000000e-02 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.541008 | 0.647659 | -0.106651 | 0.537570 | 0.635451 | -0.097880 | no |
| translation | 0.000000e+00 | 0.000000e+00 | 5.000000e-02 | 0.000000 | 0.004500 | 0.413430 | 0.633143 | -0.219713 | 0.428555 | 0.627198 | -0.198643 | no |
| rotation | 0.000000e+00 | 5.000000e-02 | 0.000000e+00 | 2.000000 | 0.000000 | 0.523121 | 0.630175 | -0.107054 | 0.519332 | 0.629645 | -0.110314 | no |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 2.000000 | 0.004500 | 0.428430 | 0.627053 | -0.198623 | 0.446805 | 0.626425 | -0.179620 | no |
