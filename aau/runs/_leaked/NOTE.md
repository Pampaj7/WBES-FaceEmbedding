# Eval buttate via perche' lo split era quello di un altro modello

`aau/eval_common.sh` passava `--seed 1234` fisso in `common_args`, e il seed e' quello con cui
`rebuild_subject_split` estrae i 100 soggetti held-out. I ranking dei checkpoint addestrati
con seed 2345 e 3456 sono quindi stati calcolati sullo split del seed 1234, cioe' su soggetti
che quei due modelli avevano visto in training: l'intersezione fra lo split 1234 e quello
giusto e' 21/100 per il 2345 e 14/100 per il 3456, quindi 79 e 86 soggetti valutati su 100
erano di training (misurato confrontando `selected_subjects` dei tre `ranking_summary.json`:
sono identici e riportano tutti `"seed": 1234`).

Niente e' stato cancellato: i numeri qui dentro (Spearman ~0.85 latent, delta ~0.37) sono
gonfiati e non vanno nel paper, ma servono a documentare di quanto lo erano. Le eval rifatte
con lo split giusto stanno nella `eval_.../ranking_{rm,cjt}` originale, un livello sopra.

Gli eval del seed 1234 NON sono qui: per quel modello il seed dell'override coincideva con il
suo, quindi lo split era corretto (verificato: stessa lista di 100 soggetti).
