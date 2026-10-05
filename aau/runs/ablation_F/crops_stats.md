# Variante F: crop casuali per i soggetti di training

Sorgente `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500` (original), split rebuild_subject_split(0.2, seed 1234): 400 soggetti di training, 5 crop ciascuno, seme 6. Held-out esclusi: 100. Regola in aau/models/random_crops_F.py.

Crop: 2000 (400 soggetti), vuoti: 0.
Frazione di vertici tenuti (finale, dopo la componente piu' grande): media 0.752, min 0.598, p5 0.614, p25 0.678, mediana 0.755, p75 0.826, p95 0.887, max 0.900.
Frazione obiettivo: media 0.752, min 0.600, p5 0.614, p25 0.678, mediana 0.755, p75 0.826, p95 0.887, max 0.900; |finale - obiettivo| max 0.0152.
Vertici: min 14044, mediana 17713, max 21120 (original 23470; crop canonico ~0.90).
Tagli: uno 993, due 1007 (50.3%).
Primo taglio, |phi| dal mento: <30 gradi 1041, 30-60 700, 60-90 251, 90-110 8.
Tentativi per crop: media 5.5, max 168.

| frazione tenuta | crop |
|---|---|
| 0.60-0.65 | 318 |
| 0.65-0.70 | 332 |
| 0.70-0.75 | 325 |
| 0.75-0.80 | 318 |
| 0.80-0.85 | 374 |
| 0.85-0.90 | 331 |

Obiettivo 0.6-0.7: |phi| mediano 23 gradi, due tagli 48%, tentativi medi 11.0.

Obiettivo 0.7-0.8: |phi| mediano 29 gradi, due tagli 51%, tentativi medi 4.0.

Obiettivo 0.8-0.9: |phi| mediano 39 gradi, due tagli 52%, tentativi medi 1.8.

Operatori (k_eig 128): 2000 crop, tutti finiti, massa positiva, vertici identici al mesh-only.

vista /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/REMESH/view_F: 5000 npz (3000 standard + 2000 crop casuali); etichette del trainer = le 6 standard per tutti i 500 soggetti; file 'crop' per soggetto: training {6: 400}, held-out {1: 100}
