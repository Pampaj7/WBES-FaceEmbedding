# E1: stato all'8 ottobre 2026, 17:30 (catena corretta dopo il critic)

Regole: `protocol.md` (15:30) + `protocol_amendment.md` (16:55; scritto dopo il critic e prima di ogni numero delle
celle nuove), entrambi con sha256. Job: `jobs.md` (l'ultima sottomissione e' quella valida). Codice:
`aau/evidence/e1_factorial/` (README).

## Catena in coda (17:25)

| priorita' | cella | training (V100) | eval (A100) |
| --- | --- | --- | --- |
| 1 | C2F / C3F / C2F-GNM / C3F-UGT | 1061843 / 1061845 / 1061847 / 1061850 | 1061844 / 1061846 / 1061848 / 1061851 |
| 2 | C2F s2 / C3F s2 (seme 2345) | 1061852 / 1061854 | 1061853 / 1061855 |
| 3 | C3M rifatta / C2M | 1061856 / 1061858 | 1061857 / 1061859 |
| 4 | C2F40 / C3F40 / G1 | 1061860 / 1061862 / 1061864 | 1061861 / 1061863 / 1061865 |
| rif. | C3M L40S (1060130) | - | 1061841 (in corso su nv-ai-04) |

- **Cancelli.** Smoke V100 1061840 (in corso), poi il cancello 1061842: la loss dell'epoca 1 deve stare entro il
  5% di quella del run su scala su L40S (`e1_gate_smoke.py` -> `smoke_v100.json`). Solo allora partono i
  training; se non torna, i training e le loro eval si cancellano da soli. La GT unificata e' verificata
  (cancello 1061849 passato alle 17:25).
- **Riepilogo:** 1061866, afterany su tutte le eval -> `summary.md`.
- **Dipendenze:** eval afterok sul suo training, `--kill-on-invalid-dep`; un training fallito non blocca il
  riepilogo.

## Cosa e' cambiato rispetto alla prima catena (cancellata: 1061659-1061704)

- **Celle nuove:**
  - C2F-GNM: BFM 392 + 5.401 GNM, nessuna ICT; le GNM di C3F sono un suo sottoinsieme;
  - C3F-UGT: C3F con `--dist_npz` = GT unificata;
  - C2F s2 e C3F s2;
  - C3M rifatta su V100: split del run su scala, K=46, fermata all'epoca 72;
  - C2F40 e C3F40: 1.350 non-BFM, un blocco, annidate in C2F e C3F.

  Gli split vecchi sono identici byte per byte (sha256 controllato).
- **Regola:** C3F > max(C2F, C2F-GNM), con tre esiti, effetto minimo +0.05, dev FaceScape, non inferiorita' su
  FaceVerse, pavimento del rumore e secondo seme (emendamento, sezione 2). C3M - C2M resta solo descrittivo.
- **Valutazione:**
  - dev FaceScape in tutte le celle (`dev_facescape_env.sh`, uscita in `devfs/`);
  - GT unificata applicata alle stesse righe (`datasets/UNIFIED_GT/eval/*_gt_matrix.npz`): HIFI3D e dev
    FaceScape. FLAME non ce l'ha: colonna "-";
  - tutte le eval girano sulle A100 (QoS unprivileged, `--requeue`), compresa C3M L40S.
- **Hardware e staging:**
  - training su V100 (container 24.10, controllato nel job);
  - staging del pre-pass su `/raid/$USER` (disco locale, bind esplicito nel container), quindi fuori da
    `--mem`: 320G per le celle con ICT, 240G per C2F-GNM e G1 (previsione in `design.md`, +20%).
- **Robustezza:** ogni job esegue una copia privata del corpo presa all'avvio. Il primo smoke V100 si era
  corrotto perche' ho modificato il corpo mentre girava: cancellato e rifatto (1061840).

## Misure di questa sessione che contano per i tempi

- **Il pre-pass sui nodi V100 (Xeon 8168) e' il collo di bottiglia.** Una identita' ICT (14 mesh):
  - 1 processo: 4.98 CPU-s per mesh, contro 3.9 su L40S (EPYC 9354);
  - 16 processi: 10.0 contro 4.9 CPU-s per mesh, cioe' 1.58 contro 3.2 mesh/s;
  - nello smoke a 36 processi, sul nodo carico: 1.2-2.2 mesh/s.

  Script: `scratch/prepass_bench/bench.sh`, job 1061829-1061837.
- **Stima, non misurata:** un blocco F (~17.500 mesh) richiede circa 2-3 h di pre-pass, quindi una cella F
  circa 12-15 h e una cella M (10 blocchi) circa 25-30 h. Limiti richiesti: 30 h e 60 h, poi abbassati a 24 h e 40 h con grad_vec (sotto). La velocita' del
  training su V100 la misura lo smoke (`smoke_v100.json`, riga "epoca 1").
- **Capienza:**
  - nv-ai-02/03 hanno 96 CPU e 1.47 TB ciascuno: con 32 CPU per job girano circa 3 celle alla volta, il resto
    aspetta in ordine di priorita';
  - il tetto delle 12 GPU vale anche qui;
  - blocchi piu' piccoli non servono (la RAM sta) e non sono possibili senza cambiare S per tutte le celle
    (C3M compresa).
- **A100 per il training: no.** Il trainer non riprende da un checkpoint: `run_training` riparte sempre
  dall'epoca 1 e `--init_checkpoint` carica solo i pesi, non l'ottimizzatore, i passi o il blocco. Su un job
  prelazionabile si perderebbe tutto. Non ho fatto lo smoke "interrompi e rilancia": il codice non ha un
  percorso di ripresa.

## Gia' verificato

- Smoke di plumbing su CPU (1061646 C3F, 1061647 G1): rc 0.
- `e1_summarize.py` provato con alias su dati esistenti:
  - C3M L40S su HIFI3D (3 scenari, GT maxabs) identica a curve.md;
  - GT unificata identica a E8 (ICP+Chamfer 0.522, NICP 0.577, e036 0.340);
  - NoW identico;
  - regola a tre esiti e domanda su C3F-UGT calcolate.
- Corpo della eval su CPU (1061694): NoW identico, FaceVerse 0.519 contro 0.518.
- HIFI3D, dev FaceScape e FLAME li esercita per prima la eval di C3M L40S (1061841).

## build_grad vettorizzato di E9: ADOTTATO in tutte le celle (decisione del PI, 17:47)

Il controllo (`gradvec_check.json`, 57 mesh) ha trovato operatori NON identici bit per bit:
- 19 array su 798 diversi, il 6.3e-5 degli elementi;
- |differenza| massima 1.2e-10, relativa al massimo 3.7e-13;
- guadagno sul pre-pass 1.64x.

Dopo il controllo il PI ha abbassato la soglia e ha deciso di adottarlo, perche' nessun training era partito.
Nota tecnica datata con sha256: `nota_tecnica_gradvec.md`.
- **Attivazione:** `e1_train_body.sh` esporta `WBES_E1_GRADVEC=1` e mette `gradvec_site/` in testa a PYTHONPATH.
  I training lo prendono all'avvio (copia privata del corpo); `e1_cell.json` registra `gradvec: 1` e il job
  conta le righe `[e1-gradvec]` nei log del pre-pass.
- **Limiti di tempo abbassati** con `scontrol`: celle F e G1 da 30 a 24 h, C3M rifatta e C2M da 60 a 40 h.
- **Valutazioni e smoke:** usano il `build_grad` originale; vale per tutte le celle allo stesso modo.

## Emendamento 2 (9 ottobre, 00:55): GT unificata tarata per C3F-UGT
Fattori per dominio (mediana maxabs / mediana unificata): BFM 1.229, ICT 0.913, GNM 1.024; verifica col loader del trainer ok (`datasets/UNIFIED_GT/train/check_calib.json`). C3F-UGT (1061850) usa la tarata; C3F-UGT non tarata (1062087) in fondo alla coda. Riepilogo ora 1062089.
