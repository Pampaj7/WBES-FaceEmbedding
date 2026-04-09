# FaceVerse Model vs Chamfer Ranking Summary

Reference metric: GT distance matrix from original clean FaceVerse meshes.

| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | Model > Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.114807 | 0.598032 | -0.483225 | 0.124456 | 0.591699 | -0.467243 | no |
| jitter | 5.000000e-02 | 0.000000e+00 | 0.000000e+00 | 0.000000 | 0.000000 | 0.087487 | 0.609911 | -0.522424 | 0.101043 | 0.647043 | -0.546000 | no |
| translation | 0.000000e+00 | 0.000000e+00 | 5.000000e-02 | 0.000000 | 0.004500 | 0.113514 | 0.598813 | -0.485299 | 0.119388 | 0.592716 | -0.473329 | no |
| rotation | 0.000000e+00 | 5.000000e-02 | 0.000000e+00 | 2.000000 | 0.000000 | 0.120113 | 0.598910 | -0.478797 | 0.130958 | 0.598361 | -0.467402 | no |
| mixed | 5.000000e-02 | 5.000000e-02 | 5.000000e-02 | 2.000000 | 0.004500 | 0.132258 | 0.598688 | -0.466430 | 0.144418 | 0.636562 | -0.492144 | no |
