# Ripresa del run principale: correzioni del critic su efffd74, verificate (11 ottobre)

Codice: commit `ce0ae82`. La prova a due nodi ha eseguito la copia congelata fatta prima del commit: i 158 file
di `v3_work` hanno lo stesso sha256 di `ce0ae82` (`run2n/code_vs_commit.json`).

## Invarianza (parzialità spenta, run dir validata)

| test | job | esito | file |
|---|---|---|---|
| `test_partial.py` (base fissa 3e8cae7, non più HEAD) | 1068077 | PASSA: geometria, invarianza, on/off, run dir, regen 12/12, skip | `partial_check.json`, `partial_views.csv` |
| `test_recipe_c3m.py` | 1068074 | PASSA | `regression_recipe_c3m_check.json` |
| `test_provenance.py` | 1068075 | PASSA | `regression_provenance_check.json` |
| controlli unitari + guardia end-to-end | 1068076 | PASSA | `unit_and_guard_checks.json` |

- Invarianza contro 3e8cae7, a parzialità spenta: ricetta uguale, 348 array identici bit per bit, header dello shard
  uguale, dati dello shard con lo stesso sha256 (268.654.160 byte).
- Run dir della riga di lancio validata (smoke 1067645): `...__78085a52` con il codice di base, con quello di
  adesso, con quello di adesso e `WBES_PREEMPT_SAVE=1`, e con la copia congelata (`code_snapshot.py`).
- Guardia end-to-end (`WBES_CHECK_RESUME_ONLY=1`):
  - sulla riga validata trova `last.pth` al passo 1000 di `78085a52`;
  - con un altro job id nell'anello la run dir diventa `1e41334f` e la guardia esce con un errore esplicito che elenca
    `78085a52`.

## Prova a due nodi (2 × 1 L40S, a768-l40s-02 e -03, QoS normal)

Configurazione: ricetta c3m, T = 600, 100 passi per epoca. `STREAM_PREEMPTIBLE` e `STREAM_CKPT_MIN` ai default
(1 e 20 minuti): ogni ripresa viene quindi dal solo salvataggio su SIGTERM.

| segmento | job | partenza | interruzione | salvataggio | ripresa |
|---|---|---|---|---|---|
| 0 | 1068078, riavvio 0 | copia del codice (986 file, 275 link) | `scontrol requeue` alle 02:40:42, SIGTERM ai due rank al passo 254 | passo 255 (epoca 3, batch 55), dopo 0.2 s | - |
| 1 | 1068078, riavvio 1 | copia verificata (sha256) | `scancel` alle 02:46:36 (come un TIMEOUT), SIGTERM al passo 456 | passo 457 (epoca 5, batch 57), dopo 0.2 s | passo 255, generatori ripristinati |
| 2 | 1068090, job nuovo, stesso STREAM_OUT | `run_tag` letto: 1068078 | - | fine: 600/600 passi | passo 457, generatori ripristinati |

- **Run dir:** una sola, `...__59f6f46f`, in tutti i segmenti.
- **Passi:** `steps_loss.csv` ha i passi da 1 a 600, una volta ciascuno, in ordine; salvataggi [255, 457] = riprese
  [255, 457]. Loss media dei 50 passi prima e dopo: 0.0903 → 0.0775 a 255, 0.0647 → 0.0648 a 457.
- **Registro delle viste:** un file per checkpoint (passi 100, 200, 255, 300, 400, 457, 500, 600).
  - Righe = viste uniche = Σ `views_logged`: rank 0 3492, rank 1 2646.
  - Rank 1: Σ `d_unique_views_used` = 2658. Le 12 in più sono il piano successivo già estratto all'interruzione del
    passo 457: mai addestrato, e infatti non registrato.
  - Semi per segmento giusti: 20261009 / 20262009 nel segmento 0, poi +100000 e +200000. La continuazione non
    ripete i semi del segmento 0.
- **Riga di lancio:** contro il mini-smoke c3m 1067709 (riga generata dagli script prima di `ce0ae82`) cambiano solo
  le chiavi del test (stream, total_steps, epochs, eval_every, runs_root). Coi valori di 1067709 l'hash dà
  `8a0a69d7`, la sua run dir.
  - Contro lo smoke validato 1067645 cambia anche `--stream-rot` (30,15,10 → 0): è la ricetta c3m di d310782, non
    queste modifiche.
- **Passo:** 0.438-0.471 s (1 + 1 L40S), GPU 74-79%.
- **Tempi di ripartenza:**
  - 71-74 s dall'avvio del job al lancio di torchrun (verifica della copia, anello, controllo di `last.pth` in 7 s);
  - alla prima partenza la copia del codice ha preso circa 60 s (job avviato alle 02:35:45, copia pronta alle
    02:36:45).

Guardia negativa (job 1068100, `run2n/guard_negative_1068100.json`): continuazione con `STREAM_RUN_TAG=999`. La run dir
diventa `9d8c1885`, senza `last.pth`, e il job esce con "[resume] ERRORE ..." e codice 3 sullo step: nessun
addestramento, nessuna run dir nuova.

## Vertici a farfalla (`butterfly_vs_igl.json`)

- 89 viste parziali su 432 hanno vertici a farfalla (più di un ventaglio di facce), fino a 28 per vista, 754 in tutto.
  Lo stesso conteggio di `igl.is_vertex_manifold` su tutte le 432 viste.
- Il critic riportava 95 viste e fino a 23 per vista: probabilmente con un'altra definizione, non riconciliato.
- Gli operatori DiffusionNet, con valori grezzi finiti, reggono su tutte le 89 viste.
- Le farfalle si contano ma non invalidano la vista: il generatore non è cambiato.

## Non verificato

- 6 + 6 GPU (la prova ha usato 1 + 1).
- Limite di tempo vero di Slurm (simulato con scancel, stesso percorso di SIGTERM e KillWait).
- Prelazione su A100.
- Un salvataggio interrotto dal SIGKILL: se il checkpoint non finisce entro KillWait (30 s) si riparte dall'ultimo
  checkpoint periodico.
- Un SIGTERM che arriva prima del primo passo (riempimento dell'anello): nessun `last.pth`, quindi alla ripartenza
  errore esplicito.
- La versione del codice negli shard: `WBES_CODE_VERSION` è passata ai produttori, ma non l'ho letta da uno shard
  (lo step di lettura non è partito prima della fine del segmento).
