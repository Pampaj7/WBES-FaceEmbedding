# Template ICP Gap Probe

This probe aligns each mesh to a fixed template face with rigid ICP, then computes pairwise Chamfer in that common frame.

- template sample: `id0478_GTready_up60k`

| scenario | metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | latent_distance | 0.168577 | 0.426232 | 0.257655 | 0.997778 | 0.878962 |
| clean | raw_chamfer | 0.003569 | 0.005236 | 0.001667 | 0.689136 | 0.253003 |
| clean | template_icp_chamfer | 0.004159 | 0.005296 | 0.001136 | 0.692840 | 0.132001 |
| translation | latent_distance | 0.190715 | 0.442840 | 0.252125 | 0.989136 | 0.846485 |
| translation | raw_chamfer | 0.003565 | 0.005384 | 0.001819 | 0.691852 | 0.209002 |
| translation | template_icp_chamfer | 0.002805 | 0.004008 | 0.001203 | 0.695309 | 0.072286 |
| rotation | latent_distance | 0.199032 | 0.441828 | 0.242795 | 0.992099 | 0.837057 |
| rotation | raw_chamfer | 0.003686 | 0.005358 | 0.001673 | 0.685679 | 0.248288 |
| rotation | template_icp_chamfer | 0.002764 | 0.003911 | 0.001146 | 0.687654 | 0.128859 |
| mixed | latent_distance | 0.256305 | 0.430645 | 0.174340 | 0.978272 | 0.806675 |
| mixed | raw_chamfer | 0.001462 | 0.001695 | 0.000233 | 0.580988 | 0.145097 |
| mixed | template_icp_chamfer | 0.001637 | 0.001880 | 0.000243 | 0.558519 | 0.040858 |
