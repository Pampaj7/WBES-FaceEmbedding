# Protocollo dichiarato prima delle eval: pilota della distillazione ArcFace-render -> DiffusionNet

Dichiarato il 2026-10-06, prima di qualunque embedding dello studente del pilota su FaceVerse o HIFI3D.
Piano: `paper/PLAN_DISTILL.md`. Gate: `aau/runs/arcface_render_zs/summary.md`.
Numeri gia' noti a quest'ora: tutte le righe del gate (insegnante ombreggiato su entrambi i domini,
normal map 3 viste su FaceVerse 0.867, NICP, ICP, Chamfer, congiunto). Il rank-1 della normal map su
HIFI3D NON e' noto: gli embedding sono stati calcolati (job 1057319, stessa pipeline e stesso crop del
gate) ma il riconoscimento non e' stato fatto. L'unico studente valutato prima di questa dichiarazione e'
uno smoke di 2 epoche su 48 soggetti (`aau/runs/distill_pilot/smoke`), usato solo per provare la catena
di eval su HIFI3D; non e' il pilota e i suoi numeri non entrano nel giudizio.

**Domanda.** Uno studente DiffusionNet addestrato solo a imitare ArcFace su normal map (sui dati BFM e
ICT-5000 di training del congiunto) batte il congiunto fuori dominio e si avvicina all'insegnante?

**Insegnante** (`aau/distill/teacher_labels.py`): la pipeline del gate importata (`zs_arcface_render.py`),
normal map, yaw 0/-30/+30, 512 px, embedding = media delle 3 viste rinormalizzata L2. Camera unica e crop
fisso PER DOMINIO DI TRAINING (BFM, ICT), il crop calibrato col detector sui render ombreggiati delle
`original` di 300 soggetti di training a caso (seed 1234) e copiato sulle normal map. Rotazione verso il
frame del renderer: BFM nessuna, ICT Rx(180), verificate con `--frame-check`. L'insegnante vede SOLO le
`original` (e, per i 418 soggetti ICT che le hanno, le 5 espressioni casuali `rexpr<k>`, stessa topologia).

**Dati.** Training: lo split del congiunto 1019532 (`aau/data_scale/split_scale.json`): 392 BFM, 4008
ICT-5000; di questi 411 ICT hanno le `rexpr`. Validazione (solo curve): i 16 BFM dell'eval online del
congiunto e 84 ICT held-out (seed 1234). FaceVerse e HIFI3D non entrano.

**Studente** (`aau/distill/train_distill.py`): `xyz_dn` con la ricetta del congiunto (width 128, 4 blocchi,
dropout 0.1, pool meanmax, rumore latente, augmentation xyz del congiunto, operatori ad area unitaria
k_eig 128 delle viste in uso, loader congelato), uscita a 512 dimensioni normalizzata L2. Ingresso: le 5
topologie senza crop (original, remesh, down8k, up60k, noisy) e le `rexpr`, frame nativi come il congiunto.
Bersaglio: l'embedding dell'insegnante della `original` dello stesso soggetto (della `rexpr<k>` stessa
per un'espressione). Loss: (1 - coseno) + 1.0 x MSE fra matrice dei coseni dello studente e
dell'insegnante nel batch (fuori diagonale). Batch: 16 soggetti a caso (domini mescolati) x 2 mesh
distinte. Adam, lr 1e-4 dimezzato dopo il 75% delle epoche, weight decay 1e-6, clip 1.0, seed 1234,
**50 epoche**. **Checkpoint valutato: l'ULTIMO** (`checkpoints/last.pth`), fissato ora; la validazione
held-out serve solo per le curve, nessuna selezione.

**Eval** (`aau/distill/eval_distill.sbatch`, `distill_summarize.py`): identica al gate. Stessi 100
soggetti (`select_subjects`, seed 1234), FaceVerse v2 con espressioni casuali e HIFI3D neutro, operatori
k_eig 128 ad area unitaria su /tmp (`zs_stage.py` + `areanorm_operators.py`), distanza = 1 - coseno
(euclidea sugli embedding L2, monotona). Funzioni di calcolo importate da `zs_expr_summarize.py` e
`zs_arcface_summarize.py`, STESSE repliche bootstrap del gate (1000, `stable_seed(1234,
"expr_recognition")`): le righe gia' nel gate devono coincidere col suo `recognition.csv` (controllo).

- **PRIMARIO:** riconoscimento sulle 5 topologie senza crop: rank-1, mAP, AUC di verifica, CI 95%.
- **Riga di riferimento: studente in convenzione BFM** (FaceVerse facce invertite; HIFI3D Rx(180) +
  facce invertite), cioe' lo STESSO ingresso della riga del congiunto nel gate.
- Righe di confronto: insegnante 3 viste normal map e ombreggiato, NICP P2Tri, ICP rigido + Chamfer,
  Chamfer faceBench, congiunto BFM+ICT (convenzione BFM). Delta APPAIATI studente - ciascuna.
- Secondario: studente in convenzione ICT (FaceVerse Rx(180); HIFI3D nativo), delta appaiati con il
  congiunto e con lo studente in convenzione BFM. Non sostituisce il riferimento.
- Crop: le stesse misure, a parte, mai nel primario.

**Criterio di successo del pilota, fissato ora (primario, riga di riferimento):**
1. "batte il congiunto": CI 95% del delta rank-1 studente - congiunto tutto sopra 0, su FaceVerse E su HIFI3D;
2. "almeno l'80% dell'insegnante": rank-1 studente / rank-1 insegnante normal map 3 viste >= 0.80 come
   stima puntuale, su FaceVerse E su HIFI3D (riportato anche il CI del rapporto, stesse repliche).
Il pilota passa se valgono entrambi su entrambi i domini. Le stesse letture sull'AUC si riportano ma non
decidono. Se passa solo in convenzione ICT, si riporta come tale e il pilota NON passa.
