# Template ICP Gap Probe: mixed

- template sample: `id0478_GTready_remesh`
- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.256305 | 0.430645 | 0.174340 | 0.978272 | 0.806675 |
| raw_chamfer | 0.001462 | 0.001695 | 0.000233 | 0.580988 | 0.145097 |
| template_icp_chamfer | 0.001688 | 0.001863 | 0.000175 | 0.570617 | 0.136716 |
