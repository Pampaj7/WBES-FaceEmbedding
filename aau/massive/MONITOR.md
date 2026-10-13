# Monitoraggio del run del modello da rilasciare

Progetto in `paper/PLAN_MASSIVE.md`, sezione 23. Licenze in `aau/massive/LICENZE.md`. Tutti i percorsi sono relativi a `~/WBES-FaceEmbedding`.

## Job

| cosa | job | GPU | stato al 13 ottobre |
|---|---|---|---|
| run principale | **JOB_MAIN** (`wbes-massive-release`) | 8 L40S, 1 nodo, QoS normal | vedi sotto |
| smoke e ripresa (codice) | 1074653 | 2 A10 | COMPLETED, 600/600 passi, ripresa provata |
| smoke del throughput | 1074876 (`wbes-massive-rel-smoke`) | 2 L40S | vedi sotto |
| held-out | 1074610 | CPU | COMPLETED, 1.380 viste |

Lancio (una sola riga per run e continuazioni): `aau/massive/launch.sh main 1 8 450000 3-12:00:00`.

## Percorsi

`OUT=aau/runs/massive_release/main`

| file | contenuto |
|---|---|
| `$OUT/steps.jsonl` | rank 0, una riga ogni 10 passi: `step`, `epoch`, `t` (unix), `loss`, `loss_mean` della finestra, `nonfinite`, `lr`, `s_per_step`, `terms` (stress, rank, size_mse ...). A ogni partenza una riga `{"event": "start", "step", "job", "segment"}`. Dopo una ripresa i passi fra l'ultimo checkpoint e l'interruzione ricompaiono: vale l'ultima riga di ogni passo |
| `$OUT/monitor/heldout.jsonl` | una riga per checkpoint EMA (ogni 8.000 passi): `mean` e `domains.<d>` con `sp_fr`, `sp_sr`, `rank1_fr`, `rank1_sr`, `size_mape`, `c`; `error` se la valutazione è fallita |
| `$OUT/node0/train.log` | log del trainer (`[v3] passo N` ogni 50 passi, epoche, ripresa) |
| `$OUT/node0/node.log` | partenze, tentativi, SIGTERM, ripresa |
| `$OUT/node0/producers.log` | viste fresche/s, anello, fallimenti |
| `$OUT/node0/heldout_watch.log` | sorvegliante dell'held-out |
| `$OUT/node0/gpu.csv`, `mem.csv` | uso delle GPU ogni 5 s; memoria del job e anello ogni 30 s |
| `$OUT/segments.log` | una riga per partenza (riavvii e continuazioni) |
| `$OUT/runs/v3_*/checkpoints/` | `last.pth` (ripresa), `epochNNN.pth` e `epochNNN_ema.pth` |
| `aau/logs/wbes-massive-release-<job>.{out,err}` | stdout e stderr di Slurm |

## Controllo in un comando

```
python3 aau/massive/status.py                 # riassunto + ALLARMI; uscita 2 se c'è un allarme
python3 aau/massive/status.py --json          # lo stesso, in JSON
```

## Segnali d'allarme e comandi

| allarme | come si vede | comando | cosa fare |
|---|---|---|---|
| **NaN** | `nonfinite > 0` nel jsonl; `finite: false` nell'held-out | `grep -n '"nonfinite": [1-9]' $OUT/steps.jsonl \| tail` | Se si ripete: `scancel`. Si riparte da un checkpoint precedente, decide il PI |
| **Loss esplosa** | `loss_mean` dell'ultima finestra oltre 3 volte la mediana delle ultime 200 righe | `tail -n 30 $OUT/steps.jsonl \| python3 -c "import sys,json; [print(r['step'], round(r['loss_mean'],4), r['lr'], round(r['s_per_step'],3)) for r in map(json.loads, sys.stdin) if 'loss_mean' in r]"` | Guardare `terms`: `size_mse` o `stress` |
| **Validazione che crolla** | `sp_fr` o `rank1_fr` medi scesi di oltre 0,10 sotto il massimo visto | `tail -n 3 $OUT/monitor/heldout.jsonl \| python3 -c "import sys,json; [print(r['ckpt'].split('/')[-1], r.get('mean'), r.get('error')) for r in map(json.loads, sys.stdin)]"` | Solo un segnale, non una regola di selezione; avvisare il PI |
| **Run fermo** | job RUNNING ma nessuna riga di passo da oltre 20 minuti | `tail -n 2 $OUT/steps.jsonl; tail -n 20 $OUT/node0/train.log; tail -n 3 $OUT/node0/producers.log` | Produttori morti o anello vuoto: `grep ERRORE $OUT/node0/node.log` |
| **Nodo morto** | nodo `down`/`drain`, job sparito senza COMPLETED | `squeue -u $USER -n wbes-massive-release; sacct -j <job> -X -o JobID,State,Elapsed,NodeList,ExitCode; sinfo -n <nodo>` | Con `--requeue` Slurm lo rimette in coda e la ripresa parte da `last.pth`; vedi le righe `start` del jsonl |
| **Fine del tempo** | `TIMEOUT` in sacct, passi < 450.000 | `sacct -j <job> -X -o State` | Continuazione: `STREAM_OUT=$PWD/aau/runs/massive_release/main aau/massive/launch.sh main 1 8 450000 3-12:00:00` (stessi argomenti, stessa run dir) |
| **Disco** | meno di 300 GB liberi sulla quota da 2 TiB | `getfattr -n ceph.dir.rbytes ~` (usati, byte); `getfattr -n ceph.dir.rbytes $OUT` | Il run scrive circa 15 MB per checkpoint (57 in tutto) più il registro delle viste usate. Se serve spazio si tolgono altri dati, mai i checkpoint del paper |
| **GPU ferme** | utilizzo basso in `gpu.csv` | `gputil <job>`; `tail -n 16 $OUT/node0/gpu.csv` | Se i produttori non bastano, il riuso sale: vedi `reuse_epoch` in `runs/v3_*/stream_stats_rank0.jsonl` |

Non usare mai `--qos=unprivileged` per questo run e non superare le 8 GPU.
