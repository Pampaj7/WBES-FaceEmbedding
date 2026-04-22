# Template ICP Gap Probe: rotation

- template sample: `id0478_GTready_up60k`
- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.199032 | 0.441828 | 0.242795 | 0.992099 | 0.837057 |
| raw_chamfer | 0.003686 | 0.005358 | 0.001673 | 0.685679 | 0.248288 |
| template_icp_chamfer | 0.002764 | 0.003911 | 0.001146 | 0.687654 | 0.128859 |
