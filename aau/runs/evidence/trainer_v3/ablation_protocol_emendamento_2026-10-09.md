# Ablazioni del trainer v3: emendamento del 9 ottobre 2026 (braccio di loss `loginv`, limite di `ugtmix`)

Scritto il 2026-10-09 alle 00:45 CEST, su richiesta del PI dopo la revisione del critic, PRIMA di lanciare il
braccio nuovo e prima di qualunque numero delle ablazioni: a quest'ora nessun braccio ha un checkpoint EMA ne'
una valutazione (i cinque bracci sono partiti il 2026-10-08 alle 23:35 e sono all'epoca ~18 di 72; esistono solo
le eval online del training, che non entrano nella regola). Il protocollo originale
(`ablation_protocol.md`, sha256 `b68df254bdaeca12d6f705f4033be649c6d30df4e08b766646ca524019354f13`, ricontrollato
ora) resta invariato. L'impronta di questo file sta in `ablation_protocol_emendamento_2026-10-09.sha256`.

## 1. Braccio aggiunto: `loginv`

Motivo (critic, 9 ottobre): il calo di e108 dalla distanza graduata globale a quella locale e' maggiore di quello
delle baseline, e il vantaggio di e108 sta solo nel decile di coppie piu' lontane. E' il sintomo atteso da una loss
in distanze assolute (stress v2), che una loss relativa (residui in log) potrebbe correggere. La conclusione
precedente "la loss non va cambiata" e' ritirata finche' questo braccio non risponde.

| braccio | flag rispetto a ctrl | domanda |
|---|---|---|
| loginv | `--loss log+inv` | loss relativa (log) con ancora di scala e termine d'invarianza al posto della loss v2 |

- Pesi: i default di `train_v3.py`, gli stessi del test anti-collasso superato
  (`a100/anti_collapse_loginv_log.json`, `extra` vuoto): `--lambda-scale 1.0 --scale-target 1.0 --lambda-inv 1.0
  --log-huber-delta 0.25 --log-eps 1e-6 --log-eps-gt 1e-4 --log-pair-mode all --w-cross 1.0`. Nessuno e' scelto
  guardando le ablazioni.
- Tutto il resto IDENTICO a ctrl: stessa riga di lancio di `v3_work/trainer/ablations/c3f/train_body.sh` (stessi
  dati, split, spec, blocchi, ordine, seme 1234, ricetta v1, EMA 0.999, GT maxabs `c3f/gt.npz` con
  `--gt-keep-scale`), stessi passi (T = 21.096, S = 293, checkpoint ed eval ai passi 10.548 e 21.096), stesso store
  di operatori condiviso (`/tmp/wbes_v3_store_c3f` su nv-ai-04, job 1062011), stessa eval
  (`c3f/eval.sbatch`, stesse pipeline e stessi set). Run dir `ablations/c3f_runs/loginv`.
- I flag della ricetta che riguardano solo la loss v2 (`--lambda_subject`, `--lambda_mesh`, `--lambda_rank`,
  `--rank_*`, `--use_id_loss`, `--lambda_id`) restano sulla riga ma non hanno effetto: `log+inv` sostituisce
  l'intera loss v2 (stress, rank e identita'). Il braccio confronta quindi il PACCHETTO loss v2 con il pacchetto
  log+inv, non un singolo termine.

**Regola di adozione:** la stessa degli altri bracci, contro ctrl e con le stesse misure: punteggio dev
all'ultimo checkpoint EMA su ALMENO +0.03 con IC 95% appaiato della differenza che esclude lo 0, e nessuna delle
altre metriche (HIFI3D graduata GT maxabs, HIFI3D graduata GT unificata, FaceVerse con espressioni rank-1) giu'
di piu' di 0.05 (stima puntuale). Altrimenti resta la loss v2. `loginv` si giudica da solo contro ctrl (asse
diverso da area/arearobust/bal/ugtmix): l'adozione di un altro flag non dipende da questo braccio e viceversa;
la combinazione di due flag adottati non e' testata da questo giro.

## 2. Limite dichiarato di `ugtmix` (confondente di scala)

`ugtmix` usa la GT unificata `c3f/gt_unified.npz` (divisa per il massimo GLOBALE, `mm_per_unit` 14.16) con i
margini della loss v2 fissi in unita' di GT (`--rank_margin 0.05`, `--rank_tau 0.02`) e lo stress in distanze
assolute. Le mediane per dominio della GT unificata non coincidono con quelle della GT maxabs per dominio, quindi
a parita' di flag i margini e lo stress pesano in modo diverso da ctrl e diverso fra i domini: e' lo stesso
confondente di scala segnalato per la GT unificata nelle valutazioni.

Una GT unificata TARATA (mediane per dominio allineate a quella maxabs,
`datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm_calib.npz`) e' in preparazione da un altro agente, ma alle
00:42 del 9 ottobre il file non esiste, e `ugtmix` (job 1061984) e' in esecuzione dal 2026-10-08 23:35: il braccio
NON si cambia. Conseguenza dichiarata ora: un esito di `ugtmix` diverso da ctrl (in un verso o nell'altro) non si
attribuisce alla sola scelta della GT come target, perche' e' confuso con la scala effettiva dei margini. Se
`ugtmix` superasse la regola di adozione, l'adozione e' sospesa finche' un braccio con la GT tarata (stessa riga
di `ugtmix`, sola GT cambiata) non la conferma con la stessa regola.

`loginv` non ha questo problema nella stessa forma: usa la GT maxabs di ctrl, e la sua loss e' in log con offset
mediano, quindi invariante a una scala globale della GT (resta solo `--log-eps-gt` come scala assoluta).
