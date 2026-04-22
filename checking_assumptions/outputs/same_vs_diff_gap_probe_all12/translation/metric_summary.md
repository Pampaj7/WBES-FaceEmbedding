# Same-vs-Different Gap Probe: translation

- pairs: `480`
- same-subject pairs: `180`
- different-subject pairs: `300`

| metric | same mean | diff mean | gap diff-same | auc(same if smaller) | diff-only spearman vs gt |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent_distance | 0.171235 | 0.516027 | 0.344792 | 0.997222 | 0.843282 |
| raw_chamfer | 0.003193 | 0.006785 | 0.003592 | 0.774148 | 0.248963 |
| rigid_registered_chamfer | 0.002780 | 0.008039 | 0.005260 | 0.784889 | 0.244316 |
| cpd_registered_chamfer | 0.008225 | 0.008240 | 0.000015 | 0.528630 | 0.075291 |
