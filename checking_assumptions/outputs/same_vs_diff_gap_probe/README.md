# Same-vs-Different Gap Probe

This probe asks a more direct question than the GT ranking benchmark:

- do same-subject cross-topology pairs stay closer than different-subject pairs?
- does rigid ICP improve or shrink that separation?
- does CPD collapse it even more?

| scenario | metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | latent_distance | 0.168577 | 0.426232 | 0.257655 | 0.997778 | 0.878962 |
| clean | raw_chamfer | 0.003569 | 0.005236 | 0.001667 | 0.689136 | 0.253003 |
| clean | rigid_registered_chamfer | 0.003019 | 0.005285 | 0.002266 | 0.701481 | 0.325289 |
| clean | cpd_registered_chamfer | 0.009323 | 0.007960 | -0.001363 | 0.462222 | -0.055001 |
| translation | latent_distance | 0.177145 | 0.433815 | 0.256670 | 0.994074 | 0.862200 |
| translation | raw_chamfer | 0.003543 | 0.005292 | 0.001749 | 0.693580 | 0.227336 |
| translation | rigid_registered_chamfer | 0.002977 | 0.005210 | 0.002233 | 0.707901 | 0.323718 |
| translation | cpd_registered_chamfer | 0.009178 | 0.008059 | -0.001119 | 0.459012 | -0.041381 |
