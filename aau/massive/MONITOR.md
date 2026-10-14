# Monitoraggio del run del modello da rilasciare

Progetto in `paper/PLAN_MASSIVE.md`, sezione 23. Licenze in `aau/massive/LICENZE.md`. Tutti i percorsi sono relativi a `~/WBES-FaceEmbedding`.

## Job

| cosa | job | GPU | stato al 14 ottobre |
|---|---|---|---|
| run principale, prima parte (fino al passo 110.972) | 1075583 (`wbes-massive-release`) | 3 A100, nv-ai-04, `aicentre-a100`, QoS unprivileged (prelazionabile, `--requeue`) | 13 ottobre 19:16 - 14 ottobre 16:38; prelazionato, poi cancellato in coda |
| stato del run principale al 14 ottobre, 16:45 | 1075583 | — | **PRELAZIONATO due volte** (13 ottobre 20:54 al passo circa 8.190; 14 ottobre 16:38 al passo circa 110.970), rimesso in coda da solo, ripartenza stimata da Slurm il 16 ottobre alle 08:54; ripresa automatica da `last.pth` (righe `start` del jsonl) |
| **run principale, continuazione** | **1078908** (`wbes-massive-release`) | 3 A10, i256-a10-08, prioritized, QoS normal | RUNNING dal 14 ottobre 16:51, ripreso al passo 110.972; 1,04 s/passo, fine stimata il 17 ottobre in mattinata; `OUT=aau/runs/massive_release/main_a100` |
| **nucleo aperto** (PLAN_MASSIVE 24) | **1079557** (`wbes-massive-release-open`) | 3 A10, i256-a10-09, prioritized, QoS normal | RUNNING dal 2026-10-14 17:49, T 230.000, 0,95 s/passo, fine stimata il 17 ottobre in mattinata; `OUT=aau/runs/massive_release/release_open` |
| nucleo aperto, tentativi annullati | 1078821, 1078909 | 3 A100; 4 A10 | 1078821 cancellato prima di partire; 1078909 a 2,9 s/passo (32 CPU per 4 rank), cancellato al passo 1.070 |
| primo lancio su 4 A100, annullato | 1075219 | 4 A100 | cancellato dopo 20 min (Ava-256 vuole una A100 libera) |
| run L40S, sostituito | 1075178 | 2 × 4 L40S | cancellato (PLAN_MASSIVE 23.4) |
| smoke e ripresa (codice) | 1074653 | 2 A10 | COMPLETED, 600/600 passi, ripresa provata |
| smoke del throughput | 1074876 (`wbes-massive-rel-smoke`) | 2 L40S | 500 passi, 0,434-0,453 s/passo a regime |
| held-out | 1074610 | CPU | COMPLETED, 1.380 viste |

Lancio e continuazioni (una sola riga per run):
- run principale: `STREAM_OUT=$PWD/aau/runs/massive_release/main_a100 aau/massive/launch.sh a10 3 330000 3-12:00:00`;
- nucleo aperto: `RELEASE_SET=open aau/massive/launch.sh a10 3 230000 3-12:00:00`.

Stato: `python3 aau/massive/status.py` (run principale) e `python3 aau/massive/status.py --out aau/runs/massive_release/release_open`.

## Percorsi

`OUT=aau/runs/massive_release/main_a100`

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
| **Fine del tempo** | `TIMEOUT` in sacct, passi < 330.000 | `sacct -j <job> -X -o State` | Continuazione: la riga di lancio del run con lo stesso `STREAM_OUT` (vedi sopra) |
| **Disco** | meno di 300 GB liberi sulla quota da 2 TiB | `getfattr -n ceph.dir.rbytes ~` (usati, byte); `getfattr -n ceph.dir.rbytes $OUT` | Il run scrive circa 15 MB per checkpoint (42 in tutto) più il registro delle viste usate. Se serve spazio si tolgono altri dati, mai i checkpoint del paper |
| **GPU ferme** | utilizzo basso in `gpu.csv` | `gputil <job>`; `tail -n 16 $OUT/node0/gpu.csv` | Se i produttori non bastano, il riuso sale: vedi `reuse_epoch` in `runs/v3_*/stream_stats_rank0.jsonl` |

Dal 14 ottobre entrambi i run girano su A10 con QoS normal, non prelazionabili (PLAN_MASSIVE 24.1). La prima parte del run principale era su A100 `unprivileged` ed è stata prelazionata due volte. Ogni ripartenza aggiunge una riga `start` al jsonl e una a `segments.log`; ripartenze frequenti vanno segnalate al PI. Non superare le 8 GPU.
