# paper.tex v1 → v2: mappa scoperta → modifica richiesta

Stato: 2026-08-17. Lettura completa di `paper.tex` (1247 righe) incrociata coi risultati
in `v2_work/STATUS.md` e `board_diary.tex`. Ordinato per impatto sul rischio di reject.

---

## A. Interventi che disinnescano una critica esistente

### A1. Tab.~`clean_xtopo_chamfer_latent_matrices` — i Chamfer negativi su `crop` sono in parte un artefatto di normalizzazione
La tabella riporta Chamfer $\approx 0$ o negativo su tutte le coppie che coinvolgono
`crop` (es. crop→down8k $-0.093$). È il dato più vistoso a favore della tesi, ed è
**attaccabile**: un reviewer può obiettare che dipende dalla normalizzazione per-mesh
scelta (maxabs), non dalla metrica.

Misurato (`v2_work/phase0/normalization_confound.csv`, 30 coppie, 13.050 coppie soggetto):
coppie con `crop` passano da $+0.087$ (maxabs) a $+0.279$ (area-weighted); coppie con
`noisy` crollano da $+0.510$ a $+0.105$; in aggregato maxabs $+0.253$ batte area $+0.172$.

**Azione**: nuova sottosezione (dopo `sec:distance_compression`) sulla non-neutralità
della normalizzazione, con la tabella a tre gruppi; e nel testo di `sec:xtopo_results`
una frase che dichiara la scelta e rimanda alla sottosezione. Riportare il Chamfer con
**entrambe** le normalizzazioni almeno nella tabella aggregata.
**Perché conviene**: la critica arriva comunque; arrivarci prima con il dato in mano
(e con maxabs che vince in aggregato, quindi nessun cherry-picking) la trasforma in un
secondo meccanismo a supporto della tesi — indipendente dalla registrazione.

### A2. Tab.~`alignment_effect` — manca l'allineatore rigido forte (uomo di paglia)
Il paper confronta solo `Rigid ICP + Chamfer` ($0.600$) e NICP ($0.456$/$0.468$).
M3DFB fornisce **RLR** (Procrustes su landmark, senza scala): sullo stesso protocollo
ristretto (20 soggetti, 5700 coppie) dà $+0.456$ contro $+0.284$ dell'ICP.

**Azione**: aggiungere le righe M3DFB (E1, E2, E9, E10) alla tabella, dichiarando che i
loro estimator ricevono 51 landmark iBUG **che il nostro setting non avrebbe** (i landmark
sono esatti sulle topologie BFM-ordinate e trasferiti per vertice più vicino sulle altre)
→ i loro numeri sono un **upper bound generoso**. Il migliore resta $0.25$ sotto il latente.
**Perché conviene**: risponde alla richiesta di "breadth of comparisons" del meta-review
e rimuove l'accusa di aver battuto solo la baseline debole.

### A3. `sec:ood_transfer` + Tab.~`faceverse_summary` — il fallimento cross-topology va riformulato
Il paper concede $0.115$ su FaceScape cross-topology zero-shot; è il numero su cui
Z1mX e YJz1 hanno costruito il reject. Con lo **stesso checkpoint e lo stesso protocollo**
(`v2_work/transfer/eval_transfer.py`): FaceScape $+0.057$ (riproduce il fallimento),
FLAME (600 identità nostre, crop trasportato per corrispondenza dal BFM) $+0.478$.

**Azione (in attesa dell'esperimento decisivo, in corso)**: se il re-crop di FaceScape a
supporto BFM-comparabile alza $\rho$ da $0.057$ verso $0.3$–$0.4$, riscrivere la sezione
come *support mismatch*, non *domain shift generico*, con la matrice transfer
$\{$BFM, FLAME$\}\times\{$BFM, FLAME, FaceScape, FaceScape-recropped$\}$.
Se il re-crop non muove nulla, il gap è reale e la sezione resta, ma con FLAME come
evidenza che il transfer cross-3DMM *è* possibile quando il supporto combacia.
**Perché conviene**: in entrambi gli esiti si passa da "non generalizza" a una
caratterizzazione precisa di *quando* generalizza.

---

## B. Interventi che aggiungono contributo

### B1. Nuova sezione: baseline correspondence-free e percettive
Richieste esplicitamente da YJz1 (varifolds, currents, DPDist) e bhBZ (ArcFace 2D).
Misurato sugli **stessi pair set** della Tab.~1 (148.500 coppie): ArcFace $+0.213$,
varifold $+0.151$, LPIPS $+0.120$, CLIP $+0.105$, DINOv2 $+0.074$, currents $+0.074$
(latente $+0.751$, Chamfer $+0.237$).
Da notare nel testo: le percettive 2D sono state **corrette a favore della baseline**
(scala condivisa per soggetto + 3 viste) prima del confronto; il guadagno viene dalla
scala condivisa, non dal multi-vista.
**Nota metodologica**: `currents` non ordina i soggetti (ratio same/diff $0.97$) perché
il termine di normale orientata è sensibile alla tassellatura quanto all'identità →
si riporta come diagnostica di orientazione, non come baseline di ranking.

### B2. `sec:distance_compression` — formalizzare il meccanismo
Il paper mostra la compressione empiricamente (IQR/raw $0.537$ rigido, $0.219$ NICP).
Manca il perché: $D_{\text{reg}}(i,j)=\min_T d(T(X_i),X_j)$ è un inf su una famiglia di
trasformazioni, quindi una contrazione; l'entità della contrazione cresce coi gradi di
libertà (rigido 6–7, NICP $\sim\infty$), che è esattamente l'ordine osservato.
**Azione**: proposizione con bound sulla riduzione di varianza inter-soggetto in funzione
dei DOF + verifica empirica sulla tabella IQR già presente.
**Perché conviene**: risponde a bhBZ ("lack of technical novelty") spostando il
contributo da "un encoder" a "un meccanismo spiegato e quantificato".

### B3. REMESH-2: secondo 3DMM nel benchmark
600 identità FLAME × 6 topologie, crop trasportato per corrispondenza,
varianti a **rapporto costante** (non conteggio assoluto: coi target BFM assoluti
`down8k` su FLAME sarebbe stato un no-op e metà benchmark degenere).
Dedup identità: coppia più vicina al $30.8\%$ della mediana (risposta diretta a Z1mX Q2).
**Azione**: sezione benchmark estesa + la matrice transfer come tabella principale.

### B4. GT matrix: dichiarare lo spazio
Su FLAME, Spearman(GT-raw, GT-normalizzato) $= 0.571$ (su BFM era $0.81$): due GT
costruiti dalle stesse mesh, differenti solo per normalizzazione, ordinano le identità in
modo sostanzialmente diverso.
**Azione**: in `sec:problem_formulation`, dopo l'Eq.~(2), dichiarare in quale spazio vive
$D_{\mathrm{GT}}$ e perché (deve essere lo stesso in cui vivono le metriche valutate).

---

## C. Correzioni di claim richieste dai reviewer (testo, costo nullo)

| # | Richiesta | Dove |
|---|---|---|
| C1 | Nominare il 3DMM (BFM) nel corpo del paper, non solo nel codice — Z1mX non riusciva a determinarlo | `sec:remesh_benchmark`, prima menzione |
| C2 | "cross-topology" vale **entro lo stesso 3DMM**: dirlo in abstract e introduzione, non solo in fondo | abstract, intro, conclusioni |
| C3 | DiffusionNet **non** è invariante a moto rigido: dichiararlo esplicitamente (la robustezza viene dall'augmentation) — Z1mX lo aveva dato per scontato al contrario | `sec:mesh_agnostic_encoder` |
| C4 | Solo facce neutre: scoping esplicito + esperimento minimo sulle espressioni | `sec:limitations` |
| C5 | Bias demografici ereditati dal 3DMM: paragrafo dedicato (citare il lavoro 2026 sul bias geometrico dei 3DMM) | `sec:limitations` |
| C6 | Split a 3 vie (train/selezione/test): la v1 selezionava il checkpoint sullo stesso benchmark dei numeri finali | `sec:evaluation_protocol` |
| C7 | Più di un seed sul modello finale | `sec:training_details` |

---

## D. Blocchi tecnici alla compilazione locale (non bloccano il lavoro, ma vanno sistemati)

1. `neurips_2026.sty` non è presente né in repo né in texmf → il paper compila solo dove
   c'è lo stile (Overleaf?). Serve il file per compilare qui.
2. `\usepackage{to-be-determined}` (riga 17) — pacchetto inesistente, evidentemente un
   segnaposto. Va rimosso.
3. `references.bib`: **35 chiavi citate, 8 presenti**. Mancano tutte le principali
   (now, REALY, facescape, diffusionet, lpips, urbach2020dpdist, fg2025, 10448898,
   kaltenmark2017general, pierson20223d, e i ~20 metodi di ricostruzione).
   Ho copiato il bib della v1 in questa cartella; va completato.
4. Le immagini erano in `img/` con suffissi ` (1)`; ho creato `imgs/` con symlink
   dai nomi che `paper.tex` si aspetta. 4 figure referenziate, tutte risolte.
   Le altre 11 immagini in `img/` non sono referenziate: probabilmente candidate per le
   nuove sezioni (`icp_fail.png`, `pipeline5.5.png`, `post_ICP3DMM.png`, `raw3DMM.png`,
   `clean_registration_compression_ratios.png`, `perturbation_iqr_compression_lines.png`, …).
