# Lettura (revisione 1; scritta dopo i risultati, con le regole fissate nel protocollo)

La lettura della prima tornata e' in `lettura_tornata1.md`. Su crop e ICP quella e' superata da questa.

**Leak.** Nessun soggetto valutato e' nel training del congiunto: 0/108 BFM, 0/89 ICT, 0/992 ICT nella
galleria grande. Restano due note.
- 16 dei 108 soggetti BFM erano nell'eval online del congiunto, cioe' hanno pesato sulla scelta del
  checkpoint.
- Il BFM-only e' confrontabile solo su 19 soggetti.

Il congiunto ha visto in training le topologie crop e noisy dei soggetti di training; nessuna baseline ha
avuto un'augmentation simile.

**1. Con l'ICP di similarita' il vantaggio del congiunto sul crop sparisce: era l'artefatto di scala.**
- **Con il crop.** Rank-1 congiunto contro ICP di similarita' + NICP P2Tri:
  - BFM: 0.944 contro 1.000, delta -0.056 [-0.089, -0.030], **sotto**;
  - ICT: 0.993 contro 1.000, delta -0.007 [-0.020, +0.000], pari.

  Anche la sola **ICP di similarita' + Chamfer**, senza NICP, fa 1.000 in tutti e due i domini, con e
  senza crop. Con l'ICP rigido, sul crop, la stessa Chamfer scendeva a 0.11 (BFM) e 0.31 (ICT).
- **Senza crop.** Il congiunto e' pari alle due varianti di similarita' sul rank-1 (BFM 0.998 contro
  1.000; ICT 1.000 contro 1.000). Sul TAR@FAR 1e-3 e' sotto in BFM (0.969 contro 1.000) e alla pari o
  sopra in ICT (1.000 contro 0.997 NICP e 0.962 Chamfer). BFM-19: congiunto, BFM-only e similarita' tutti a
  1.000.

In dominio, quindi, una pipeline geometrica configurata bene riconosce le persone attraverso topologie e
crop **almeno quanto** il congiunto. La differenza della prima tornata (congiunto 0.944 / 0.993 contro
NICP rigido 0.686 / 0.600 sul crop) veniva dalla scala non corretta.

**2. La galleria grande non rompe il soffitto.**
- **Blocchi rettangolari (100 query, galleria di 992).**
  - remesh -> original (primario): congiunto e ICP di similarita' + NICP P2Tri entrambi a 1.000 su
    rank-1, AUC e TAR@FAR 1e-3 e 1e-4. La Chamfer cade a 0.16, il template a 0.82.
  - crop -> original: rank-1 1.000 per entrambi; il TAR del congiunto e' 0.98 / 0.97, quello di NICP
    1.00 / 1.00.
- **Tutte le 992 query, solo congiunto.**
  - senza crop: rank-1 1.000, TAR@FAR 1e-4 0.995;
  - con crop: rank-1 0.996, TAR@FAR 1e-4 0.926.
- **noisy -> original e' degenere.** La noisy e' la original perturbata: anche la Chamfer grezza fa
  1.000. Lo stesso vale in parte per crop -> original, che e' un ritaglio della original. Il blocco
  remesh -> original e' stato aggiunto per questo, prima di calcolarlo.

Con 992 identita' ICT il compito resta saturo per i due metodi migliori; per separarli servono dati piu'
difficili, non piu' soggetti dello stesso 3DMM.

**3. Espressioni (rexpr), invariato.** Il congiunto resta il metodo peggiore fra i forti: rank-1 0.822,
contro 0.905 di NICP rigido e 1.000 di ArcFace. Il template fa 0.388. Le varianti di similarita' non sono
state calcolate su rexpr (niente crop, stessa topologia; dichiarato nel protocollo).

**4. Baseline "NICP su template".**
- **Senza crop.** Rank-1 0.856 (BFM) e 0.898 (ICT) sui 108/89; sulla galleria grande 0.82 (remesh) e
  0.34 (noisy).
- **Con il crop.** Crolla: 0.00-0.05. Il template copre tutta la faccia, e registrandolo su un ritaglio la
  NICP inventa la parte mancante, che poi entra nella distanza media per vertice.

E' l'implementazione semplice dichiarata, con i parametri NICP di faceBench non ritoccati: va letta come
limite inferiore della famiglia "iscrivi una volta", non come il suo valore migliore.

**5. ArcFace migliore.** Con l'inquadratura per mesh e le normali smussate:
- rank-1 BFM 0.987 senza crop (0.981 prima) e 0.988 con il crop; ICT 0.981 e 0.980;
- il TAR@FAR resta inchiodato a 0.600 senza crop e 0.800 con il crop: le coppie con la `noisy` non
  passano mai una soglia stretta, perche' la media sull'1-anello non basta a togliere quel rumore;
- su rexpr fa 1.000.

Il congiunto e' **sopra** ArcFace migliore senza crop (BFM +0.012 [+0.003, +0.021], ICT +0.019), **sotto**
con il crop in BFM (-0.044 [-0.077, -0.018]) e **pari** in ICT.

**6. Tempi** (job 1060419, `a768-l40s-06` in `--exclusive`, AMD EPYC 9454 + L40S; CPU a un thread).

| fase | metodo | tempo (mediana) |
| --- | --- | --- |
| iscrizione, per mesh | congiunto (operatori CPU 2.03 s + embedding GPU 55 ms) | 2.08 s [1.27-4.42] |
| iscrizione, per mesh | NICP su template | 1.15 s |
| iscrizione, per mesh | ArcFace migliore | 0.57 s |
| ricerca 1:10.000, per query | congiunto | 1.96 ms su CPU, 68 us su GPU |
| ricerca 1:10.000, per query | template (L2 media per vertice) | 811 ms |
| ricerca 1:10.000, per query | ArcFace | 0.88 ms |
| per coppia | ICP di similarita' + Chamfer | 45 ms |
| per coppia | ICP di similarita' + NICP P2Tri | 1.29 s |

Per i metodi a coppie, una ricerca 1:10.000 costerebbe circa 7.5 minuti (ICP + Chamfer) e 3.6 ore (NICP).
E' una STIMA (N x tempo per coppia), non una misura.

**Conclusione per il paper.** In dominio, il riconoscimento non e' piu' un argomento d'accuratezza a favore
del congiunto: ICP di similarita' + Chamfer lo eguaglia o lo supera (crop BFM) a 45 ms per coppia.
L'argomento che resta e' il costo della ricerca su gallerie grandi:
- il congiunto si iscrive una volta (circa 2 s) e poi confronta in microsecondi;
- la Chamfer va rifatta per ogni coppia, quindi il congiunto conviene da circa 50 confronti per query in
  su (stima: 2.08 s / 45 ms);
- fra i metodi che si iscrivono una volta, il congiunto domina il template (accuratezza) e ArcFace sulle
  topologie senza crop (TAR).

Restano a sfavore: espressioni e crop BFM contro ICP di similarita'.
