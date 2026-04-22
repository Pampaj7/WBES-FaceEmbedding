# Template ICP Gap Probe: clean

- template sample: `id0478_GTready_original`
- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.168577 | 0.426232 | 0.257655 | 0.997778 | 0.878962 |
| raw_chamfer | 0.003569 | 0.005236 | 0.001667 | 0.689136 | 0.253003 |
| template_icp_chamfer | 0.003603 | 0.005276 | 0.001673 | 0.669383 | 0.053953 |
