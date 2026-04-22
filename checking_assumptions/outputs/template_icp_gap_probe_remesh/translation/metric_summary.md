# Template ICP Gap Probe: translation

- template sample: `id0478_GTready_remesh`
- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.190715 | 0.442840 | 0.252125 | 0.989136 | 0.846485 |
| raw_chamfer | 0.003565 | 0.005384 | 0.001819 | 0.691852 | 0.209002 |
| template_icp_chamfer | 0.002801 | 0.004246 | 0.001444 | 0.672099 | 0.099525 |
