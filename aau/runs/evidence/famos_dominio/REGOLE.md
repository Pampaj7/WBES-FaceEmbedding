# FaMoS come confine di dominio: perche' il nostro e' sotto ICP mm e NoW metrical — regole (scritte PRIMA dei numeri)

Domanda (PI, 14 ottobre): sul riconoscimento dell'identita' su FaMoS (`esterna_famos_id`, protocollo `29baaa7`) la d_F
non cal. del nostro e' sotto ICP + Chamfer in mm e NoW metrical, mentre su Ava-256 e' pari ai migliori. Serve una
spiegazione causale e verificabile. Tre ipotesi, non esclusive. Codice: commit `219e8f3` e `3bb6adf`
(`aau/famos_dominio/fd.py` sha256 `f78bccd842d21dc734b3f5c91a5876c4ab9ec68ff2028b0c688670a9d3b50080`, `fd.sbatch`
`4729769082266095f2bea68f1ac5aa2cd82776abd7d547dd13cf5aab9121368e`, figura `fig_famos_expr.py` qui accanto
`6d80d9866b53e53129333d25cb91d0a75f005785a8a8ca3edcf7f8e001540f08`), collaudato solo su distanze, statistiche e taglie
casuali.

Stato al commit di questo file:
- **noti**: tutti i numeri di `esterna_famos_id/results.md` (metriche globali e delta di ogni colonna, comprese
  ICP mm / cs e NoW metrical / originale, che entrano in H3);
- **noti, solo etichette**: gli `expr_mm` delle 150 scansioni valutate (sotto), di cui ho guardato solo la
  distribuzione per fissare i terzili; nessuna distanza letta per strato;
- **inesistenti**: metriche per strato, statistiche geometriche delle scansioni, predittore a sola taglia.

**Dati**: solo le distanze gia' calcolate in `datasets/FAMOS/esterna_id/eval/dist` (14 colonne, 11.175 coppie) e i file
FaMoS gia' su disco. Niente Ava-256, nemmeno le sue mesh. Valori per scansione (etichette, statistiche) solo in
`datasets/FAMOS/esterna_id/eval/dominio` (fuori da git); in git solo aggregati.

**Statistica** (comune): IC 95% percentile, 1000 repliche per soggetto, le stesse di `esterna_famos_id`
(`famos_eval.bootstrap_counts(15, 1000, 1234)`, replica 0 = punto). Pesi come in `ei.py eval`: query pesate col
conteggio del soggetto; coppie della stessa persona c_s, di persone diverse c_a c_b. Delta = nostro - concorrente
(rank-1, AUC; positivo = il nostro meglio), sulle stesse repliche. Una replica in cui uno strato ha peso nullo esce
dal percentile di quello strato; il conteggio va nei risultati. Letture del verdetto: **d_F non cal. s1234 e s2345**
contro **ICP + Chamfer in mm** e **NoW metrical** (le due migliori): 4 combinazioni. Le altre colonne sono riportate.

## H1, espressione

Il nostro e' sensibile all'espressione; su FaMoS le coppie della stessa persona differiscono in espressione.
Previsione: il nostro errore di verifica cresce con l'intensita' dell'espressione piu' di quello dei concorrenti.

- **Etichetta d'intensita'** e_i (mm): i parametri FLAME 2023 di FaMoS NON ci sono (la release ha solo registrazioni
  ply in topologia FLAME 2020, nessun parametro su disco). Si usa il ripiego: `expr_mm` di
  `datasets/FAMOS/test_view/manifest.csv` (riga `scan` con lo stesso soggetto, sequenza e fotogramma; 150/150
  presenti) = distanza g (GT unificata) fra la registrazione dello stesso fotogramma e la forma neutra di riferimento
  del soggetto, posa esclusa (Procrustes; `famos_test_view.py`). Solo etichetta, mai input di un metodo.
- **Strati**: terzili delle 150 e_i, `statistics.quantiles(e, n=3)`: t1 = 0,862 mm, t2 = 3,315 mm. Scansione in L
  (e <= t1), M, H (e > t2). Coppia nello strato della scansione PIU' espressiva (max(e_i, e_j)).
- **AUC_k (primaria)**: coppie della stessa persona dello strato k contro TUTTE le coppie di persone diverse
  (`famos_eval.weighted_auc`). Si ha 1 - AUC = media, sulle coppie della stessa persona, della quota di coppie di
  persone diverse piu' vicine: l'AUC globale si scompone sugli strati, e si riporta la quota del divario globale che
  viene da ciascuno strato (descrittiva).
- **rank-1_k**: query con e_q nello strato k, galleria = le altre 149 (la regola di `ei.py`). Riportato, non entra nel
  verdetto.
- **Interazione** I = Delta_H - Delta_L (delta nostro - concorrente). H1 prevede I < 0.
- **Sottoinsieme quasi neutro**: le scansioni di L sole (query e galleria in L; query senza un'altra scansione della
  stessa persona in L escluse e contate): rank-1 e AUC, delta. Fra le 10 sequenze valutate non ci sono sequenze
  neutre: L e' il quasi neutro disponibile (soggetto 080 senza scansioni in L).
- Secondaria, descrittiva: AUC_k con le sole coppie di persone diverse dello stesso strato k.
- **Verdetto**: SUPPORTATA se l'IC di I(AUC) e' tutto sotto 0 in tutte e 4 le combinazioni; NON SUPPORTATA se lo
  include (o e' sopra 0) in tutte e 4; PARZIALE altrimenti. **Spiega il divario** solo se, in piu', l'IC di Delta_AUC
  su L e sul quasi neutro include 0 (o e' sopra) in tutte e 4; se H1 e' supportata ma Delta_L resta sotto 0,
  l'espressione spiega una parte, non il divario.

## H2, qualita' della scansione

Scansioni grezze con buchi, bordi irregolari e densita' variabile, contro le mesh tracciate di Ava.

- **Statistiche per scansione** (job CPU, `fd.py stats`): sulla GREZZA (obj) vertici usati, triangoli, area (mm^2),
  densita' (vertici per cm^2), componenti connesse, lunghezza del bordo (mm, spigoli di un solo triangolo), anelli di
  bordo; sulla PATCH NoW in mm (quella che vedono tutti i metodi) area, componenti, anelli di bordo, lunghezza del
  bordo e **irregolarita' q = bordo / (2 sqrt(pi area))** (1 = disco; buchi interni e bordi frastagliati la alzano).
- **Primaria**: q della patch. Coppia nello strato della scansione PEGGIORE (max(q_i, q_j)), terzili di q delle 150
  con la stessa regola di H1 (valori nei risultati). Stessa AUC_k, stessa I, stesso verdetto di H1.
- Descrittiva: per ogni statistica (ed e_i), Spearman fra la statistica e l'eccesso d'errore per scansione
  x_i = c_i(nostro) - c_i(concorrente), con c_i = media su tutte le coppie della stessa persona che contengono i della
  quota (non pesata) di coppie di persone diverse piu' vicine (pari = 1/2). IC per soggetto (scansioni dei soggetti
  ripetute col conteggio della replica).
- Limite dichiarato ora: senza Ava si verifica solo se la variazione di qualita' DENTRO FaMoS spiega il nostro
  eccesso d'errore. Uno spostamento di qualita' uguale per tutte le grezze non si vede qui.

## H3, taglia

Su FaMoS la taglia pesa molto (`size_share`: 39,6% di FR^2). Quanto del vantaggio di ICP mm viene dalla sola taglia?

- **Predittore a sola taglia (oracolo)**: d_ij = |CS_i - CS_j|, CS = `cs` di `scalars.npz` (centroid size robusta della
  patch in mm, la stessa di ICP cs, ULRSSM cs e GW cs). rank-1, mAP, AUC, EER come `ei.py eval`, e delta contro il
  nostro. Nuovo.
- **Quota del vantaggio dovuta alla taglia** (rank-1 e AUC): f_ICP = (M_ICPmm - M_ICPcs) / (M_ICPmm - M_nostro) e
  f_NoW = (M_NoWmet - M_NoWorig) / (M_NoWmet - M_nostro), per replica; IC del rapporto, del numeratore e del
  denominatore. I punti sono gia' noti da results.md: H3 NON e' cieca, nuovi sono solo gli IC e l'oracolo.
- **Verdetto** (AUC, contro ICP): SUPPORTATA (la taglia da' la maggior parte del vantaggio) se l'IC del numeratore
  ICPmm - ICPcs e' tutto sopra 0 e f_ICP >= 0,5 nel punto per entrambi i semi; PARZIALE se il numeratore e' sopra 0 ma
  f_ICP < 0,5 per almeno un seme; NON SUPPORTATA se l'IC del numeratore include 0. NoW: descrittivo.

## Lettura

Descrittiva, 15 persone, IC larghi; nessun p-value. Dopo aver visto i numeri non si cambiano etichette, strati, letture
o criteri. I risultati sfavorevoli si riportano come gli altri; se nessuna ipotesi spiega il divario, lo si scrive.
Correzioni di bug dopo questo commit: in results.md con commit e motivo.

Uscite: `results.md` (generato da `fd.py analyze`, sezione a mano conservata), `strata.csv`, `size.csv`,
`quality.csv`, `summary.json`, `fig_famos_expr.pdf` (da `fig_famos_expr.py`).
