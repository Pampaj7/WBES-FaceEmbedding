# Ablazioni del trainer v3: protocollo dichiarato prima dei numeri

Scritto l'8 ottobre 2026 dal coder del trainer v3 su decisione del PI, PRIMA di lanciare qualunque braccio e
prima di qualunque metrica. L'impronta sha256 di questo file sta in `ablation_protocol.sha256`; ogni modifica
successiva va in un emendamento datato, separato, senza toccare questo file.

## Domanda

Quali flag del trainer v3 adottare nel run massivo, rispetto al default v2 (centro e pooling medio per vertice,
scala maxabs, campionatore attuale, GT maxabs con batch a dominio singolo, loss v2).

## Bracci (un job ciascuno, trainer v3, `v3_work/trainer/train_v3.py`)

| braccio | flag rispetto a ctrl | domanda |
|---|---|---|
| ctrl | nessuno (default = v2) | riferimento; controllo incrociato con la cella E1 C3F |
| area | `--area on`: centro, scala sqrt(area) e pooling medio pesati per area (massa) | E7 |
| arearobust | `--area robust --area-robust smooth`: come area, aree sulla geometria passa-basso (64 autovettori) | E7 |
| bal | `--sampler balanced --domain-alpha 0`: passi uniformi fra i domini | E5 |
| ugtmix | `--gt unified --sampler balanced --domain-alpha 1 --batch-domains mixed`: GT unificata, batch multi-dominio | GT come target |

Non si lancia `ugt` (GT unificata a dominio singolo): la copre la cella E1 C3F-UGT.

## Cosa resta uguale in tutti i bracci

- Dati e ordine: quelli della cella E1 C3F (`aau/evidence/e1_factorial/split_c3f.json`: 392 BFM, 844 GNM,
  4.557 ICT; spec del run su scala con 4 blocchi, `block_seed` 1234, BFM residente con l'8.87% dei passi).
- Passi: T = 21.096, S = 293 per epoca, 72 epoche; lr 1e-4 costante (il dimezzamento del run su scala, al
  passo 81.747, non arriva); ricetta v1 del run su scala (`aau/data_scale/recipe_v1.sh`); seme 1234.
- Checkpoint ed eval online alle epoche 36 e 72 (passi 10.548 e 21.096); EMA dei pesi 0.999.
- Operatori: calcolati UNA volta (pre-pass con il `build_grad` vettorizzato di E9) e letti da tutti i bracci
  dallo stesso store; nessun braccio rifa' il pre-pass, quindi i segni degli autovettori sono gli stessi per tutti.
- Forward sequenziale con `--fast-data` (stessi tensori e stessi gradienti di v2, verificato).
- Hardware: A100 di nv-ai-04 (`-p aicentre-a100 --qos=unprivileged --requeue`), ripresa da checkpoint testata
  (requeue vero, traiettoria identica bit per bit su dati con operatori identici).

## Valutazione

Di ogni braccio, ai passi 10.548 e 21.096, con i pesi EMA (`epoch036_ema.pth`, `epoch072_ema.pth`), con le
pipeline esistenti invariate (`aau/zs3dmm/zs_zeroshot.sbatch` via `v3_work/trainer/eval_v3.py`, scenario clean):

- dev FaceScape (`datasets/DEV_FACESCAPE/`, `aau/zs3dmm/dev_facescape_env.sh`);
- HIFI3D, distanza graduata senza crop con le due GT: maxabs e unificata (`datasets/UNIFIED_GT/eval/`); rank-1;
- FaceVerse con espressioni (convenzione BFM, `_flip`): rank-1;
- NoW: tau.

## Criterio primario e regola di adozione

- **Punteggio dev:** sul dev FaceScape, media fra lo Spearman graduato senza crop (vista neutra, GT maxabs) e
  il rank-1 con espressioni senza crop (definizione di `aau/runs/evidence/dev_facescape/results.md`),
  all'ULTIMO checkpoint EMA (21.096 passi). Il checkpoint a 10.548 e' solo descrittivo.
- **Differenza braccio - ctrl:** appaiata, sulle stesse repliche bootstrap (1.000, soggetti ricampionati con
  reinserimento, seme 1234, come `aau/zs3dmm/zs_summarize.py` e `zs_expr_summarize.py`), IC 95% percentile.
- **Un flag si adotta se:**
  1. il punteggio dev sale di ALMENO 0.03 rispetto a ctrl, con IC 95% della differenza che esclude lo 0; e
  2. nessuna altra metrica (HIFI3D graduata con GT maxabs, HIFI3D graduata con GT unificata, FaceVerse con
     espressioni rank-1) scende di PIU' di 0.05 rispetto a ctrl (stima puntuale).
- **Altrimenti si tiene il default v2.** NoW e il rank-1 HIFI3D sono riportati, non entrano nella regola.
- area e arearobust si giudicano entrambe contro ctrl; se passano entrambe si adotta quella col punteggio dev
  piu' alto.

## Limiti dichiarati

- Un solo seme per braccio: la variabilita' fra semi non e' stimata (gli IC sono solo sui soggetti di test).
- Il controllo incrociato ctrl contro E1 C3F confronta trainer diversi su stessi soggetti, ordine e passi, ma
  con operatori ricalcolati separatamente (segni degli autovettori) e su GPU diverse (A100 contro V100): ci si
  aspetta un accordo entro il rumore, non l'identita'.
- I bracci area/arearobust cambiano anche la scala dell'input: le sigma del rumore di training restano nelle
  unita' del frame, quindi l'ampiezza relativa del rumore non e' identica a ctrl.
