# Template ICP Gap Probe: mixed

- template sample: `id0478_GTready_down8k`
- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.256305 | 0.430645 | 0.174340 | 0.978272 | 0.806675 |
| raw_chamfer | 0.001462 | 0.001695 | 0.000233 | 0.580988 | 0.145097 |
| template_icp_chamfer | 0.001685 | 0.001939 | 0.000254 | 0.571605 | 0.106334 |
