# Template ICP Gap Probe

This probe aligns each mesh to a fixed template face with rigid ICP, then computes pairwise Chamfer in that common frame.

- template sample: `id0478_GTready_down8k`

| scenario | metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | latent_distance | 0.168577 | 0.426232 | 0.257655 | 0.997778 | 0.878962 |
| clean | raw_chamfer | 0.003569 | 0.005236 | 0.001667 | 0.689136 | 0.253003 |
| clean | template_icp_chamfer | 0.003225 | 0.004510 | 0.001285 | 0.615802 | 0.202193 |
| translation | latent_distance | 0.190715 | 0.442840 | 0.252125 | 0.989136 | 0.846485 |
| translation | raw_chamfer | 0.003565 | 0.005384 | 0.001819 | 0.691852 | 0.209002 |
| translation | template_icp_chamfer | 0.003729 | 0.004853 | 0.001123 | 0.626420 | 0.162383 |
| rotation | latent_distance | 0.199032 | 0.441828 | 0.242795 | 0.992099 | 0.837057 |
| rotation | raw_chamfer | 0.003686 | 0.005358 | 0.001673 | 0.685679 | 0.248288 |
| rotation | template_icp_chamfer | 0.003258 | 0.004443 | 0.001184 | 0.618272 | 0.179145 |
| mixed | latent_distance | 0.256305 | 0.430645 | 0.174340 | 0.978272 | 0.806675 |
| mixed | raw_chamfer | 0.001462 | 0.001695 | 0.000233 | 0.580988 | 0.145097 |
| mixed | template_icp_chamfer | 0.001685 | 0.001939 | 0.000254 | 0.571605 | 0.106334 |
