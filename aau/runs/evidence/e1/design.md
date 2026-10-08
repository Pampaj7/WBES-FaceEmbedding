# E1: disegno delle celle (calcolato prima dei numeri)

Generato da `aau/evidence/e1_factorial/e1_design.py` con le funzioni del trainer (`partition_blocks`, `epoch_subset`) e la formula della cache del trainer. S = 293 passi per epoca, batch 5, seme 1234, quota BFM 0.0887372. Checkpoint alle epoche (36, 72) = passi 10548 e 21096.

| cella | training (per dominio) | blocchi K | E nominale | passi/epoca per dominio | cambi di blocco (epoca) | realizzabile |
| --- | --- | --- | --- | --- | --- | --- |
| c3m | {'bfm': 392, 'gnm': 10000, 'ict': 54008} | 46 | 360 | {'bfm': 26, 'gnm': 42, 'ict': 225} | [9, 17, 25, 33, 41, 48, 56, 64, 72] | True |
| c2m | {'bfm': 392, 'ict': 54008} | 40 | 313 | {'bfm': 26, 'ict': 267} | [9, 17, 25, 33, 41, 48, 56, 64, 72] | True |
| c2f | {'bfm': 392, 'ict': 5401} | 4 | 72 | {'bfm': 26, 'ict': 267} | [19, 37, 55] | True |
| c3f | {'bfm': 392, 'gnm': 844, 'ict': 4557} | 4 | 72 | {'bfm': 26, 'gnm': 42, 'ict': 225} | [19, 37, 55] | True |
| g1 | {'gnm': 10000} | 6 | 72 | {'gnm': 293} | [13, 25, 37, 49, 61] | True |

## Identita' VISTE entro il checkpoint

Un'identita' e' vista se compare in almeno un'epoca fino a quel checkpoint; esposizioni = epoche in cui compare (una per epoca, fino a 6 mesh ciascuna).

| cella | passi | blocchi toccati | viste per dominio | viste totali | non-BFM | esposizioni mediana / media |
| --- | --- | --- | --- | --- | --- | --- |
| c3m | 10548 | 5 di 46 | {'bfm': 392, 'gnm': 1090, 'ict': 5874} | 7356 | 6964 | 8 / 7.2 |
| c3m | 21096 | 10 di 46 | {'bfm': 392, 'gnm': 2172, 'ict': 11695} | 14259 | 13867 | 8 / 7.4 |
| c2m | 10548 | 5 di 40 | {'bfm': 392, 'ict': 6755} | 7147 | 6755 | 8 / 7.4 |
| c2m | 21096 | 10 di 40 | {'bfm': 392, 'ict': 13493} | 13885 | 13493 | 8 / 7.6 |
| c2f | 10548 | 2 di 4 | {'bfm': 392, 'ict': 2701} | 3093 | 2701 | 18 / 17.1 |
| c2f | 21096 | 4 di 4 | {'bfm': 392, 'ict': 5401} | 5793 | 5401 | 18 / 18.2 |
| c3f | 10548 | 2 di 4 | {'bfm': 392, 'gnm': 422, 'ict': 2279} | 3093 | 2701 | 18 / 17.1 |
| c3f | 21096 | 4 di 4 | {'bfm': 392, 'gnm': 844, 'ict': 4557} | 5793 | 5401 | 18 / 18.2 |
| g1 | 10548 | 3 di 6 | {'gnm': 5001} | 5001 | 5001 | 11 / 10.5 |
| g1 | 21096 | 6 di 6 | {'gnm': 10000} | 10000 | 10000 | 11 / 10.5 |

## Memoria

Cache esatta per blocco (formula del trainer) e picco previsto = RSS di base 40.0 GiB (run su scala, dopo il rilascio del blocco al cambio 0->1: GT 65.600^2 float64 piu' il resto) + cache + /tmp del blocco successivo a 5.92 MiB per mesh dai tar (picco di shmem del run su scala 96.6 GiB / 16698 mesh del blocco piu' piccolo fra 1 e 20). Riferimento misurato: picco rss+shmem del run su scala 361.8 GiB.

| cella | cache max per blocco (GiB) | mesh dai tar per blocco (max) | picco previsto rss+shmem (GiB) |
| --- | --- | --- | --- |
| c3m | 213.3 | 17069 | 352.0 |
| c2m | 219.7 | 17640 | 361.7 |
| c2f | 219.8 | 17668 | 360.6 |
| c3f | 208.0 | 16446 | 343.2 |
| g1 | 112.3 | 12554 | 224.6 |
