# Same-vs-Different Gap Probe: mixed

- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.178296 | 0.412905 | 0.234609 | 0.999012 | 0.894676 |
| raw_chamfer | 0.001911 | 0.002640 | 0.000729 | 0.659012 | 0.234669 |
| rigid_registered_chamfer | 0.001920 | 0.002722 | 0.000802 | 0.649877 | 0.182811 |
| nicp_correspondence | 0.096675 | 0.101767 | 0.005092 | 0.626667 | 0.203764 |
| cpd_registered_chamfer |  |  |  |  |  |
