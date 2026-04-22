# Same-vs-Different Gap Probe: clean

- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.168577 | 0.426232 | 0.257655 | 0.997778 | 0.878962 |
| raw_chamfer | 0.003569 | 0.005236 | 0.001667 | 0.689136 | 0.253003 |
| rigid_registered_chamfer | 0.003019 | 0.005285 | 0.002266 | 0.701481 | 0.325289 |
| cpd_registered_chamfer | 0.009323 | 0.007960 | -0.001363 | 0.462222 | -0.055001 |
