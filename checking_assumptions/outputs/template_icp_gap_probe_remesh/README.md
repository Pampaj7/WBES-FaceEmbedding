# Template ICP Gap Probe

This probe aligns each mesh to a fixed template face with rigid ICP, then computes pairwise Chamfer in that common frame.

- template sample: `id0478_GTready_remesh`

| scenario | metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | latent_distance | 0.168577 | 0.426232 | 0.257655 | 0.997778 | 0.878962 |
| clean | raw_chamfer | 0.003569 | 0.005236 | 0.001667 | 0.689136 | 0.253003 |
| clean | template_icp_chamfer | 0.002860 | 0.004212 | 0.001351 | 0.650370 | 0.114192 |
| translation | latent_distance | 0.190715 | 0.442840 | 0.252125 | 0.989136 | 0.846485 |
| translation | raw_chamfer | 0.003565 | 0.005384 | 0.001819 | 0.691852 | 0.209002 |
| translation | template_icp_chamfer | 0.002801 | 0.004246 | 0.001444 | 0.672099 | 0.099525 |
| rotation | latent_distance | 0.199032 | 0.441828 | 0.242795 | 0.992099 | 0.837057 |
| rotation | raw_chamfer | 0.003686 | 0.005358 | 0.001673 | 0.685679 | 0.248288 |
| rotation | template_icp_chamfer | 0.002680 | 0.004097 | 0.001417 | 0.697531 | 0.084858 |
| mixed | latent_distance | 0.256305 | 0.430645 | 0.174340 | 0.978272 | 0.806675 |
| mixed | raw_chamfer | 0.001462 | 0.001695 | 0.000233 | 0.580988 | 0.145097 |
| mixed | template_icp_chamfer | 0.001688 | 0.001863 | 0.000175 | 0.570617 | 0.136716 |
