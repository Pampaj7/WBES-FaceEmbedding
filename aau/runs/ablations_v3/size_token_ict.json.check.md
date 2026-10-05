# Token di taglia, ict: consistenza fra topologie dello stesso soggetto

Sorgente `datasets/ICT/topo`, 30000 mesh. Standardizzazione: soggetti di datasets/ICT/train_ready/split_train.txt (training ICT, nessun held-out); 27000 mesh, media 2.20611, std 0.06086.

Std fra soggetti del log r sull'original: 0.04915.

| topologia | n | media Δlog r | std Δlog r | max abs | rapporto raggi | media in std token | max abs in std token |
|---|---|---|---|---|---|---|---|
| remesh | 5000 | -0.01734 | 0.00078 | 0.02041 | 0.9828 | -0.285 | 0.335 |
| crop | 5000 | -0.10033 | 0.00598 | 0.12345 | 0.9045 | -1.649 | 2.028 |
| noisy | 5000 | -0.04212 | 0.00425 | 0.05888 | 0.9588 | -0.692 | 0.967 |
| down8k | 5000 | +0.00040 | 0.00005 | 0.00073 | 1.0004 | +0.007 | 0.012 |
| up60k | 5000 | -0.00027 | 0.00001 | 0.00031 | 0.9997 | -0.004 | 0.005 |
