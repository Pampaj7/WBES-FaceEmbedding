# WS5 rifatto: espressioni casuali per soggetto su ICT

## Regime x metrica (coppie di tutte le k), CI 95% bootstrap subject-level

| regime | condition | model_seed | metric | spearman | ci_low | ci_high | ranking_spearman | baseline_spearman | delta_vs_baseline | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same_k | all_k | 1234 | latent | 0.739 | 0.682 | 0.790 | nan | 0.841 | -0.101 | 24750 |
| same_k | all_k | 1234 | chamfer | 0.887 | 0.844 | 0.918 | nan | 0.951 | -0.064 | 24750 |
| expr_vs_neutral | all_k | 1234 | latent | 0.784 | 0.727 | 0.831 | nan | 0.841 | -0.056 | 49500 |
| expr_vs_neutral | all_k | 1234 | chamfer | 0.914 | 0.884 | 0.938 | nan | 0.951 | -0.036 | 49500 |
| mixed | all | 1234 | latent | 0.739 | 0.680 | 0.790 | 0.816 | 0.841 | -0.101 | 99000 |
| mixed | all | 1234 | chamfer | 0.886 | 0.846 | 0.917 | 0.930 | 0.951 | -0.065 | 99000 |

Baseline neutra-contro-neutra sugli stessi 100 soggetti (job 1019710): latent 0.8405, chamfer 0.9506. `delta_vs_baseline` e' la perdita rispetto a quella riga, ed e' il numero che WS5 vuole.

`spearman` (con il suo CI) e' il punto del bootstrap sulle pair_metrics del breakdown: UNA osservazione per coppia (soggetti, coppia di etichette), cioe' una sola mesh per soggetto, che e' il caso reale. `ranking_spearman` e' il punto di compare_model_vs_chamfer_rankings.py, che aggrega su TUTTE le mesh pair della coppia di soggetti: 1 per same_k (e infatti i due numeri coincidono), 2 per expr_vs_neutral, 20 per mixed. Mediare piu' mesh pair toglie rumore e alza lo Spearman, quindi lo scarto fra le due colonne cresce col numero di mesh pair mediate e non e' un disaccordo. Sulle righe aggregate `all_k` la colonna e' vuota: nessuna singola run di ranking copre tutte le k insieme.


## Dettaglio per espressione k

| regime | condition | model_seed | metric | spearman | ci_low | ci_high | ranking_spearman | baseline_spearman | delta_vs_baseline | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expr_vs_neutral | k1 | 1234 | chamfer | 0.914 | 0.878 | 0.938 | 0.932 | 0.951 | -0.037 | 9900 |
| expr_vs_neutral | k1 | 1234 | latent | 0.776 | 0.709 | 0.831 | 0.809 | 0.841 | -0.065 | 9900 |
| expr_vs_neutral | k2 | 1234 | chamfer | 0.907 | 0.872 | 0.939 | 0.923 | 0.951 | -0.044 | 9900 |
| expr_vs_neutral | k2 | 1234 | latent | 0.792 | 0.737 | 0.839 | 0.819 | 0.841 | -0.049 | 9900 |
| expr_vs_neutral | k3 | 1234 | chamfer | 0.916 | 0.884 | 0.940 | 0.931 | 0.951 | -0.034 | 9900 |
| expr_vs_neutral | k3 | 1234 | latent | 0.788 | 0.731 | 0.837 | 0.813 | 0.841 | -0.053 | 9900 |
| expr_vs_neutral | k4 | 1234 | chamfer | 0.925 | 0.898 | 0.946 | 0.939 | 0.951 | -0.025 | 9900 |
| expr_vs_neutral | k4 | 1234 | latent | 0.788 | 0.734 | 0.835 | 0.815 | 0.841 | -0.053 | 9900 |
| expr_vs_neutral | k5 | 1234 | chamfer | 0.909 | 0.867 | 0.942 | 0.925 | 0.951 | -0.041 | 9900 |
| expr_vs_neutral | k5 | 1234 | latent | 0.778 | 0.708 | 0.832 | 0.807 | 0.841 | -0.063 | 9900 |
| mixed | all | 1234 | chamfer | 0.886 | 0.846 | 0.917 | 0.930 | 0.951 | -0.065 | 99000 |
| mixed | all | 1234 | latent | 0.739 | 0.680 | 0.790 | 0.816 | 0.841 | -0.101 | 99000 |
| same_k | k1 | 1234 | chamfer | 0.889 | 0.846 | 0.924 | 0.889 | 0.951 | -0.061 | 4950 |
| same_k | k1 | 1234 | latent | 0.730 | 0.647 | 0.797 | 0.730 | 0.841 | -0.110 | 4950 |
| same_k | k2 | 1234 | chamfer | 0.873 | 0.818 | 0.916 | 0.873 | 0.951 | -0.077 | 4950 |
| same_k | k2 | 1234 | latent | 0.747 | 0.677 | 0.811 | 0.747 | 0.841 | -0.093 | 4950 |
| same_k | k3 | 1234 | chamfer | 0.889 | 0.843 | 0.925 | 0.889 | 0.951 | -0.061 | 4950 |
| same_k | k3 | 1234 | latent | 0.743 | 0.666 | 0.807 | 0.743 | 0.841 | -0.098 | 4950 |
| same_k | k4 | 1234 | chamfer | 0.905 | 0.866 | 0.933 | 0.905 | 0.951 | -0.045 | 4950 |
| same_k | k4 | 1234 | latent | 0.748 | 0.681 | 0.807 | 0.748 | 0.841 | -0.093 | 4950 |
| same_k | k5 | 1234 | chamfer | 0.878 | 0.815 | 0.925 | 0.878 | 0.951 | -0.073 | 4950 |
| same_k | k5 | 1234 | latent | 0.729 | 0.638 | 0.806 | 0.729 | 0.841 | -0.111 | 4950 |
