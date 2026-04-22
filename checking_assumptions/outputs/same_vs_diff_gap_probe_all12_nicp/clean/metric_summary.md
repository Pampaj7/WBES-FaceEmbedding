# Same-vs-Different Gap Probe: clean

- pairs: `480`
- same-subject pairs: `180`
- different-subject pairs: `300`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.165992 | 0.512756 | 0.346764 | 0.997741 | 0.848272 |
| raw_chamfer | 0.003215 | 0.006793 | 0.003578 | 0.772037 | 0.243155 |
| rigid_registered_chamfer | 0.002917 | 0.009562 | 0.006645 | 0.782056 | 0.235961 |
| nicp_correspondence | 0.092537 | 0.104001 | 0.011464 | 0.697019 | 0.161189 |
| cpd_registered_chamfer |  |  |  |  |  |
