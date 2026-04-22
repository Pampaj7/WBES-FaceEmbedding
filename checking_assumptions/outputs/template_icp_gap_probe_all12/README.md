# Template ICP Gap Probe

This probe aligns each mesh to a fixed template face with rigid ICP, then computes pairwise Chamfer in that common frame.

- template sample: `id0478_GTready_original`

| scenario | metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | latent_distance | 0.168577 | 0.426232 | 0.257655 | 0.997778 | 0.878962 |
| clean | raw_chamfer | 0.003569 | 0.005236 | 0.001667 | 0.689136 | 0.253003 |
| clean | template_icp_chamfer | 0.003603 | 0.005276 | 0.001673 | 0.669383 | 0.053953 |
| translation | latent_distance | 0.190715 | 0.442840 | 0.252125 | 0.989136 | 0.846485 |
| translation | raw_chamfer | 0.003565 | 0.005384 | 0.001819 | 0.691852 | 0.209002 |
| translation | template_icp_chamfer | 0.003654 | 0.005170 | 0.001515 | 0.662222 | 0.039810 |
| rotation | latent_distance | 0.199032 | 0.441828 | 0.242795 | 0.992099 | 0.837057 |
| rotation | raw_chamfer | 0.003686 | 0.005358 | 0.001673 | 0.685679 | 0.248288 |
| rotation | template_icp_chamfer | 0.003367 | 0.004651 | 0.001284 | 0.646420 | 0.014667 |
| mixed | latent_distance | 0.256305 | 0.430645 | 0.174340 | 0.978272 | 0.806675 |
| mixed | raw_chamfer | 0.001462 | 0.001695 | 0.000233 | 0.580988 | 0.145097 |
| mixed | template_icp_chamfer | 0.001566 | 0.001866 | 0.000300 | 0.624691 | 0.143002 |
