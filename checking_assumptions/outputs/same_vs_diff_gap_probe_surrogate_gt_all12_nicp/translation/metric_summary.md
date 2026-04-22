# Same-vs-Different Gap Probe: translation

- pairs: `480`
- same-subject pairs: `180`
- different-subject pairs: `300`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.171235 | 0.516027 | 0.344792 | 0.997222 | 0.756707 |
| raw_chamfer | 0.003193 | 0.006785 | 0.003592 | 0.774148 | 0.366804 |
| rigid_registered_chamfer | 0.002780 | 0.008039 | 0.005260 | 0.784889 | 0.383331 |
| nicp_correspondence | 0.092292 | 0.103777 | 0.011485 | 0.703241 | 0.325598 |
| cpd_registered_chamfer |  |  |  |  |  |
