# Template ICP Gap Probe: rotation

- template sample: `id0478_GTready_down8k`
- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.199032 | 0.441828 | 0.242795 | 0.992099 | 0.837057 |
| raw_chamfer | 0.003686 | 0.005358 | 0.001673 | 0.685679 | 0.248288 |
| template_icp_chamfer | 0.003258 | 0.004443 | 0.001184 | 0.618272 | 0.179145 |
