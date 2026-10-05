# Porting su AAU AI Cloud (Slurm)

Il frontend `ai-fe02` non ha GPU e non ha `singularity`, `pip` o `conda`: **tutto quello che
esegue python gira dentro un job Slurm**, nel container NGC `pytorch_24.10.sif`
(python 3.10.12, torch 2.5.0a0+e000cf0ad9.nv24.10 — soddisfa `environment.twotower_robust.yml`).
Nessuno script originale del repo e' stato modificato: tutto sta qui dentro.

## Sequenza

```bash
cd ~/WBES-FaceEmbedding
git clone --depth 1 https://github.com/nmwsharp/diffusion-net.git diffusion-net   # frontend
aau/submit.sh setup_env.sbatch        # partizione cpu, ~20 s: crea .venv_aau
aau/submit.sh smoke_test.sbatch       # L40S, 5 min: import + nome GPU
aau/submit.sh train_remesh.sbatch     # L40S, 72 h
aau/submit.sh eval_ranking.sbatch     # L40S, 10 h   \
aau/submit.sh eval_topology.sbatch    # L40S,  3 h    > indipendenti, anche in parallelo
aau/submit.sh eval_sigma.sbatch       # L40S, 36 h   /
```

I tre `eval_*` **non** valutano il modello appena addestrato: `WBES_CKPT` punta di default
al checkpoint v1 pubblicato nel repo
(`.../mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth`), cioe'
riproducono i numeri del paper. Per valutare la run nuova bisogna dirlo esplicitamente:

```bash
WBES_CKPT=aau/runs/remesh_v1recipe_<jobid>/<rundir>/checkpoints/best_by_xtopo_mesh_clean.pth \
  aau/submit.sh eval_ranking.sbatch
```

(`<rundir>` e' la dir col fingerprint, la stampa il log del training alla riga `Run dir:`.)
Il nome del file e' lo stesso in tutte le run, ma la chiave della output dir e' il nome
della run dir piu' un hash di checkpoint+dataset+matrice, quindi i due eval non si pestano.

### Il seed dell'eval e' quello del checkpoint (bug corretto l'11 settembre 2026)

`eval_common.sh` passava `--seed 1234` fisso in `common_args`. Quel seed non e' un dettaglio
di riproducibilita': `rebuild_subject_split` (`robustness/data_utils.py:76`) estrae con
`np.random.default_rng(seed)` i 100 soggetti held-out, gli stessi che il training aveva
tenuto fuori usando lo stesso seed. Valutare con 1234 un modello addestrato con un altro
seed significa misurarlo sui suoi soggetti di **training**: fra lo split 1234 e quello giusto
l'intersezione e' 21/100 per il seed 2345 e 14/100 per il 3456, cioe' 79 e 86 soggetti
valutati su 100 erano di training (misurato). I ranking sbagliati sono in
`aau/runs/_leaked/` con la nota del caso, non cancellati.

Ora `common_args` non passa `--seed`: gli script posthoc prendono il seed dal `config.json`
del checkpoint (`if cli_args.seed >= 0` in `posthoc_runner.py:253`), che e' quello giusto per
costruzione. `aau_eval_begin` chiama `aau/eval_split_info.py` e scrive nel log, prima delle
ore di GPU, seed, origine del seed e i primi soggetti valutati:

```
[eval-split] seed=2345, da il checkpoint: mixed_xtopo_..._seed2345__1b40106f/config.json
[eval-split] pool=500 train=400 held-out=100 -> valutati(eval)=100
[eval-split] primi 5 valutati: id0001 id0003 id0008 id0015 id0018
```

`WBES_EVAL_SEED=<n>` forza un seed a mano quando serve davvero — p.es. per valutare piu'
modelli sulla stessa selezione di soggetti di un dataset che nessuno di loro ha visto
(zero-shot cross-3DMM, `aau/ict/`), dove non c'e' niente da tenere fuori. Con l'override il
seed entra nell'hash della out dir e in `eval_key.txt`, perche' altrimenti due split diversi
dello stesso checkpoint finirebbero negli stessi file; senza override l'hash e' identico a
prima, quindi le out dir esistenti non si spostano (verificato sul v1: `47e4a6be`).

Per il v1 non cambia nulla: e' addestrato con seed 1234, quindi la lista dei 100 soggetti e'
la stessa di prima, byte per byte (confrontata con `selected_subjects` di
`ranking_rm/ranking_summary.json`).

**Sottometti sempre con `aau/submit.sh`**, da qualunque directory: crea `aau/logs/` e
`aau/runs/` e passa `--chdir/--output/--error` assoluti. Con `sbatch` diretto funziona solo
se il repo e' nel checkout indicato nelle direttive `#SBATCH` dei file (path hardcoded: se
sposti il repo vanno aggiornati, oppure usa `submit.sh` che li ricalcola). Opzioni extra dopo
il nome dello script: `aau/submit.sh eval_sigma.sbatch --time=48:00:00`.

`env.sh` e' la sorgente unica dei percorsi; `run.sh` esegue python nel container col venv
attivo (`aau/run.sh -c "import torch"`) e serve solo su un nodo di calcolo.

### Spezzare `eval_ranking` e `eval_sigma` in piu' job

I `--time` scritti nei due file **non bastano**, misurato su L40S: un blocco di 148500 mesh
pair costa 2.3 h (~18.5 pair/s), quindi ranking = 5 scenari = ~11.8 h contro 10 h, e sigma =
21 blocchi (1 clean + 4 scenari x 5 sigma non nulle; le sigma=0.00 sono clonate dal clean e
costano zero) = ~48 h contro 36 h.

La differenza fra i due e' cosa resta dopo il kill. `compare_model_vs_chamfer_rankings.py`
accumula le righe in memoria e scrive `ranking_summary.json` solo dopo l'ultimo scenario: un
job ucciso non lascia **niente**, nemmeno gli scenari gia' finiti. Il sigma sweep invece gira
con `--progressive_output_layout` e ha gia' scritto la sottocartella di ogni coppia
scenario/sigma, quindi conserva tutto quello che aveva chiuso.

Si spezza per scenari, una `WBES_EVAL_STAGE` per job (e' la sottocartella sotto la out dir:
due job nella stessa si sovrascriverebbero `run.log` e `.done`):

```bash
WBES_EVAL_SCENARIOS=clean,jitter,translation WBES_EVAL_STAGE=ranking_clean_jitter_translation \
  aau/submit.sh eval_ranking.sbatch --time=09:00:00
WBES_EVAL_SCENARIOS=rotation,mixed WBES_EVAL_STAGE=ranking_rotation_mixed \
  aau/submit.sh eval_ranking.sbatch --time=07:00:00

WBES_SIGMA_SCENARIOS=mixed WBES_EVAL_STAGE=sigma_mixed \
  aau/submit.sh eval_sigma.sbatch --time=17:00:00
```

Per il ranking le tabelle vanno poi rimesse insieme (solo stdlib, gira sul frontend):

```bash
aau/merge_ranking_scenarios.py --out_dir <RUN>/ranking_merged \
    <RUN>/ranking_clean_jitter_translation <RUN>/ranking_rotation_mixed
```

Rifiuta di fondere out dir con checkpoint, dataset, matrice GT, seed o pair context diversi.

## Dati

| cosa | percorso | override | da dove |
|---|---|---|---|
| REMESH mesh-only | `datasets/REMESH/npz_data_topo_500/` | `WBES_REMESH_NOOPS_DIR` | HF `Pampaj`, `.tar.zst` |
| REMESH con operatori | `datasets/REMESH/npz_data_topo_500_withops/` | `WBES_DATA_DIR` | calcolato qui |
| matrice GT normalizzata | `.../autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz` | `WBES_DIST_NPZ` | snapshot workspace |
| checkpoint v1 | `...mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth` | `WBES_CKPT` | **gia' nel repo** |

Preparazione a partire dai `.tar.zst` scaricati in `hf_downloads/`:

```bash
aau/submit.sh extract_remesh.sbatch     # cpu, ~1 min: 3000 npz in layout PIATTO
aau/submit.sh precompute_ops.sbatch     # cpu 24 core, ~40 min: operatori DiffusionNet
```

Il layout e' piatto per forza: `precompute_operators_npz.py` usa `Path.iterdir()` e
`GTReadyDatasetNPZ` usa `os.listdir()`, nessuno dei due e' ricorsivo. I nomi sono
`id<NNNN>_GTready_<topologia>.npz` con topologia in
`original|remesh|crop|noisy|down8k|up60k`, ed e' da li' che
`infer_topology_label_from_name` ricava l'etichetta. Nessuna conversione di chiavi
necessaria: l'archivio HF ha `V`/`F` e `load_geometry_from_npz` accetta sia `V`/`F` sia
`verts`/`faces`, scrivendo poi sempre `verts`/`faces` come vuole il loader.

`k_eig=128` per gli operatori (default dello script, e il valore usato dal progetto —
`v2_work/STATUS.md`). Il `--eig_k 300` della ricetta di training non c'entra: lo usa solo
il modello `intrinsic_dn`, mentre il top model e' `xyz_dn`, che riceve `evals`/`evecs`
interi; nel training entra solo nel fingerprint.

Se i dati mancano, training ed eval si fermano subito con il percorso mancante stampato
nel log, senza stack trace.

## Tempi e riavvii

I `--time` partono dalle misure della run v1: training ~11.5 min/epoca x 120 epoche
(~23.3 h di solo loop) → 72 h, perche' alle 23.3 h vanno aggiunte ~60 eval online
(`--eval_every 2`) di durata **non misurata**; e' una stima conservativa, da tarare con
`seff` dopo la prima run vera su L40S. Eval: 28h10m in totale, ripartite in ranking 5h13m,
topology 1h03m, sigma sweep 21h54m → 10 h / 3 h / 36 h. MaxWall su `prioritized` con QoS
`normal` e' 6 giorni, quindi non c'e' motivo di stringere.

- **`train_runner.py` non ha resume**: se il job muore per walltime si riparte da zero.
  Non e' stato aggiunto un resume perche' vorrebbe dire toccare il codice di ricerca.
- I tre stage di eval hanno **output dir stabile**
  (`aau/runs/eval_<rundir-del-ckpt>_<hash8>/<stage>/`) e **skip-if-exists**: il job esce 0
  se trova `<stage>/.done`, che viene scritta solo dopo che python e' uscito con 0 (non
  basta l'esistenza degli artefatti: un kill a meta' scrittura ne lascia solo una parte).
  In `eval_key.txt` ci sono i tre input in chiaro; se la dir esiste con una chiave diversa
  il job **fallisce** invece di riusare i numeri sbagliati.
- Il training invece mette lo `SLURM_JOB_ID` nel `runs_root`: il fingerprint della run dir
  dipende solo dagli iperparametri, quindi due sottomissioni identiche si sovrascriverebbero
  i checkpoint a vicenda.

## Note

- Il venv e' creato dentro il container con `--system-site-packages --without-pip` (il
  container NGC non ha `ensurepip`; il pip di sistema installa comunque nel venv perche'
  `sys.prefix` punta li'). Nel venv finiscono solo `libigl`, `potpourri3d`,
  `robust-laplacian`: torch, numpy, scipy, sklearn, tqdm, tensorboard e lo stesso pip sono
  gia' nel container. Rifarlo da zero: `WBES_FORCE_VENV=1 aau/submit.sh setup_env.sbatch`.
- `train_remesh.sbatch` riproduce il fingerprint della v1 (`...__9a81466d`). Entrano nel
  fingerprint anche `epochs` e `preload_eval_workers_train`: `WBES_EPOCHS=60` o
  `WBES_PRELOAD_WORKERS=8` producono una run dir con hash diverso, non confrontabile a
  colpo d'occhio con la v1.
- `WBES_STAGE_TMP=1 aau/submit.sh train_remesh.sbatch` copia gli npz su `/tmp` prima di
  partire (`dataset_gtready.py` fa un `np.load` per campione senza worker, e CephFS
  mono-processo fa 69 MB/s). **`/tmp` e' RAM e conta contro `--mem`**: lo script misura il
  dataset con `du` e aborta se non ci sta lasciando 32 GB al training, suggerendo il `--mem`
  giusto. Una `trap EXIT` ripulisce `/tmp/$SLURM_JOB_ID`.
- Non usare `--qos=unprivileged`: rende il job prelazionabile.
