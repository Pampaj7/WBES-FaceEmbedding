# Normalized GT Surrogate

This surrogate GT matrix is built directly from REMESH `original` meshes after applying the same per-mesh normalization used by `GTReadyDatasetNPZ`.

- topology source: `original`
- subject count: `500`
- output: `/dtu/p1/leopam/WBES-FaceEmbedding/v2_work/phase0/gt_surrogate_normalized_500/normalized_matrix_distances_surrogate.npz`

Distance definition:
- subtract per-mesh centroid
- divide each mesh by its own max absolute coordinate
- compute mean per-vertex L2 distance
- normalize the final matrix by its positive max
