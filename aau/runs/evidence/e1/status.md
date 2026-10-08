# E1: stato al 8 ottobre 2026, 16:10 (dettagli del report al PI)

## Cosa e' pronto e verificato

- Split delle celle (`aau/evidence/e1_factorial/split_*.json`, `subsets.json`): heldout, online_eval e online_eval_extra
  identici a `split_scale_all.json` (controllato); C3F annidata in C2F; nessun soggetto congelato in training.
  C2F: BFM 392 + 401 ICT-5000 + 5.000 nuove; C3F: BFM 392 + 338 + 4.219 + 844 GNM (419 con 1 espressione, 425 con 2).
- Disegno (`design.md`) calcolato con `partition_blocks` ed `epoch_subset` del trainer: tutte le 72 epoche di ogni
  cella sono realizzabili; C2M cambia blocco alle stesse epoche di C3M (9, 17, 25, 33, 41, 48, 56, 64, 72); passi per
  epoca BFM 26, ICT 267 (2 domini) o 225 + GNM 42 (3 domini), G1 GNM 293.
- Smoke del training sul nodo cpu (job 1061646 = C3F, 1061647 = G1): split esplicito, blocchi, BFM residente (C3F) o
  assente (G1), checkpoint, arresto dopo l'epoca voluta (rc 143 trattato come voluto), sync, mem_job.log: rc 0.
- Riepilogo (`e1_summarize.py`) provato sui dati di C3M: HIFI3D (3 scenari, 2 passi), riconoscimento HIFI3D e NoW
  riproducono `aau/runs/data_scale_ood` a 0 / 1e-16 (`controls.csv`); il percorso delle differenze appaiate provato
  con un alias (`scratch/test_summarize_alias.py`), anche per FLAME (risultati del congiunto e del BFM-only) e
  FaceVerse.
- Corpo della valutazione (`e1_eval_body.sh`) provato sul nodo cpu per C3M, passi FaceVerse e NoW, in una cartella di
  prova poi cancellata (job 1061694, rc 0): NoW tau 0.299 / 0.348 identici a curve.md; FaceVerse rank-1 e036 0.519
  [0.476, 0.566] contro 0.518 [0.475, 0.565] pubblicato (embedding su CPU, max |diff| 5e-4). HIFI3D e FLAME non
  provati qui (stesse chiamate gia' usate da data_scale_ood / ws_flame, con la sola cartella di uscita cambiata):
  FLAME lo esercita per prima la eval di C3M (1061667), HIFI3D la prima eval di una cella.

## Deviazioni e scelte (da confermare o correggere prima che i job partano)

1. **Quantita' effettiva ~2.5x, non 10x.** C3M (e quindi C2M) a 21.096 passi ha visto 10 blocchi su 46: 14.259
   identita' (13.867 non-BFM), a 10.548 passi 7.356. C2F/C3F ne vedono 5.793 / 3.093. Ho seguito la specifica
   (ICT a 1/10 del totale). Per un contrasto ~10x a 21.096 passi servirebbero celle F con ICT ~1/40 (1.350, un solo
   blocco, RAM ~260 GiB, 2 run in piu'): non lanciate.
2. **C2M fermato all'epoca 72 di un T nominale 91.709** (non `--total-steps 21096`): con 21.096 i 54.008 ICT
   sarebbero stati spalmati su tutti i blocchi (54k identita' viste, 1.8 esposizioni) e C3M - C2M confonderebbe
   varieta' e quantita'. Con K=40 / E=313 lo schedule dei blocchi e' quello di C3M (97% delle identita' viste).
3. K per cella imposto dal trainer (un blocco deve contenere i soggetti di un'epoca): C2F/C3F 4 blocchi da 18 epoche,
   G1 6 da 12. Nelle celle F a 10.548 passi e' stata vista meta' delle identita'.
4. Nessuna modifica al trainer; stessa spec, GT e flag di C3M (controllo automatico nel job).
5. Valutazione: HIFI3D e NoW di C3M riusati da `aau/runs/data_scale_ood` (stessi file); FaceVerse ed FLAME di C3M
   rifatti (FLAME non c'era; FaceVerse e072 di C3M era ancora in coda per un altro agente). I bracci HIFI3D nuovi
   stanno in `aau/runs/evidence/e1/hifi_runs` (WBES_HIFI_RUNS), non in `ws_hifi3d`: il summary di ws_hifi3d,
   se rigenerato, non li raccoglie con un'etichetta sbagliata.

## Memoria (--mem)

Prevista con componenti MISURATE (cache esatta per blocco con la formula del trainer; RSS di base 40.0 GiB e /tmp
5.92 MiB per mesh misurati su 1060130; la previsione per C3M, 352.0 GiB, sta 9.8 GiB sotto il picco misurato 361.8):
C2M 361.7, C2F 360.6, C3F 343.2, G1 224.6 GiB di rss+shmem. Richiesti 430G, 430G, 410G, 270G (~15% sopra
previsione + 9.8). Le celle "piccole" non costano meno di C3M: la RAM la decide la grandezza del blocco (i soggetti
di un'epoca, piu' i 392 BFM residenti), non il numero di identita'. Il picco vero lo scrive `mem_job.log` di ogni
run e il riepilogo lo riporta.

## Job (anche in `jobs.md`)

Training 1061659 (C2M), 1061661 (C2F), 1061663 (C3F), 1061665 (G1); eval 1061700-1061703 (afterok, si cancellano se
il training fallisce), eval C3M 1061667; riepilogo 1061704 (afterany) -> `summary.md`.
**Coda L40S:** Slurm stima la partenza dei training al 9 ottobre alle 22:08 (tutte le L40S occupate, ~30 job
L40S di altri utenti davanti in FIFO). Libere ora: il nodo A100 nv-ai-04 (8 GPU, 980 GB, 256 CPU: 2 celle per volta)
e 9 V100 su nv-ai-02 (1.2 TB liberi, container 24.10 compatibile). Il vincolo "solo L40S" e' del PI: non li ho usati.
I job in coda si possono spostare senza perdere niente (`scontrol update JobId=... Partition=... Gres=...`; il
corpo dello sbatch e' letto all'avvio).

## Quando i job finiscono

`summary.md` viene riscritto dal job di riepilogo; a mano:
`aau/submit.sh evidence/e1_factorial/e1_summarize.sbatch`. Se un training fallisce (OOM, tempo), la sua eval si
cancella e il riepilogo esce con la cella mancante e la regola "non valutabile".
