# Same-vs-Different Gap Probe: rotation

- pairs: `135`
- same-subject pairs: `45`
- different-subject pairs: `90`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.174279 | 0.429812 | 0.255533 | 0.996049 | 0.871105 |
| raw_chamfer | 0.003591 | 0.005260 | 0.001669 | 0.688889 | 0.258765 |
| rigid_registered_chamfer | 0.003059 | 0.005377 | 0.002318 | 0.700741 | 0.343099 |
| nicp_correspondence | 0.091985 | 0.098390 | 0.006406 | 0.653827 | 0.258765 |
| cpd_registered_chamfer |  |  |  |  |  |
