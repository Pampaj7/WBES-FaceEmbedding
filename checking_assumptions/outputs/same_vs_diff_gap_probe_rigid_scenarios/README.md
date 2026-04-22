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
| clean | nicp_correspondence | 0.091850 | 0.098124 | 0.006274 | 0.653580 | 0.233622 |
| clean | cpd_registered_chamfer |  |  |  |  |  |
| translation | latent_distance | 0.177145 | 0.433815 | 0.256670 | 0.994074 | 0.862200 |
| translation | raw_chamfer | 0.003543 | 0.005292 | 0.001749 | 0.693580 | 0.227336 |
| translation | rigid_registered_chamfer | 0.002977 | 0.005210 | 0.002233 | 0.707901 | 0.323718 |
| translation | nicp_correspondence | 0.091485 | 0.097973 | 0.006487 | 0.657531 | 0.241479 |
| translation | cpd_registered_chamfer |  |  |  |  |  |
| rotation | latent_distance | 0.174279 | 0.429812 | 0.255533 | 0.996049 | 0.871105 |
| rotation | raw_chamfer | 0.003591 | 0.005260 | 0.001669 | 0.688889 | 0.258765 |
| rotation | rigid_registered_chamfer | 0.003059 | 0.005377 | 0.002318 | 0.700741 | 0.343099 |
| rotation | nicp_correspondence | 0.091985 | 0.098390 | 0.006406 | 0.653827 | 0.258765 |
| rotation | cpd_registered_chamfer |  |  |  |  |  |
| mixed | latent_distance | 0.178296 | 0.412905 | 0.234609 | 0.999012 | 0.894676 |
| mixed | raw_chamfer | 0.001911 | 0.002640 | 0.000729 | 0.659012 | 0.234669 |
| mixed | rigid_registered_chamfer | 0.001920 | 0.002722 | 0.000802 | 0.649877 | 0.182811 |
| mixed | nicp_correspondence | 0.096675 | 0.101767 | 0.005092 | 0.626667 | 0.203764 |
| mixed | cpd_registered_chamfer |  |  |  |  |  |
