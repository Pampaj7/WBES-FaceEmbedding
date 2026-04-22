# Same-vs-Different Gap Probe: translation

- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.177145 | 0.433815 | 0.256670 | 0.994074 | 0.862200 |
| raw_chamfer | 0.003543 | 0.005292 | 0.001749 | 0.693580 | 0.227336 |
| rigid_registered_chamfer | 0.002977 | 0.005210 | 0.002233 | 0.707901 | 0.323718 |
| nicp_correspondence | 0.091485 | 0.097973 | 0.006487 | 0.657531 | 0.241479 |
| cpd_registered_chamfer |  |  |  |  |  |
