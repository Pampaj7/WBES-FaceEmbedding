# Same-vs-Different Gap Probe

This probe asks a more direct question than the GT ranking benchmark:

- do same-subject cross-topology pairs stay closer than different-subject pairs?
- does rigid ICP improve or shrink that separation?
- does CPD collapse it even more?

| scenario | metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean | latent_distance | 0.165992 | 0.512756 | 0.346764 | 0.997741 | 0.761397 |
| clean | raw_chamfer | 0.003215 | 0.006793 | 0.003578 | 0.772037 | 0.368014 |
| clean | rigid_registered_chamfer | 0.002917 | 0.009562 | 0.006645 | 0.782056 | 0.385298 |
| clean | cpd_registered_chamfer |  |  |  |  | -0.102302 |
| translation | latent_distance | 0.171235 | 0.516027 | 0.344792 | 0.997222 | 0.756707 |
| translation | raw_chamfer | 0.003193 | 0.006785 | 0.003592 | 0.774148 | 0.366804 |
| translation | rigid_registered_chamfer | 0.002780 | 0.008039 | 0.005260 | 0.784889 | 0.383331 |
| translation | cpd_registered_chamfer |  |  |  |  | -0.102302 |
