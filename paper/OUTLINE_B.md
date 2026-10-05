# Outline B — paper di metodologia di valutazione (CVPR 2027)

Scritto il 4 ottobre 2026, prima dei risultati di stasera (tabella cross-3DMM ICT-only / congiunto; frame rms contro standard).
Fonti dei numeri: `BOARD_DIARY.md` (BD, con data), `DOPO_NEURIPS.html` (DN), `main_cvpr_draft.tex` (TEX, con label), `review.md` (RV), `literature/REVIEW_2026-09-10.md` (LIT).
Nessun numero qui è nuovo: ogni valore ha il suo riferimento. Dove manca, è scritto "manca".

---

## 1. Titoli proposti

1. **Not Just Alignment: How Frames, Support and Registration Distort 3D Face Evaluation** ← raccomandato
2. **Five Frames, One Table: Hidden Choices in 3D Face Reconstruction Evaluation**
3. **Alignment Is Not Neutral, and Neither Is Normalization: Auditing 3D Face Distances**
4. **What the Bounding Box Knows: Controls for Meta-Evaluating 3D Face Metrics**
5. **Before the Distance: Normalization, Support and Alignment in 3D Face Evaluation**

Perché il primo: dice la cosa che i dati sostengono davvero. Il paper NeurIPS diceva "alignment hurts"; i dati reali (WS3b) dicono che a parità di supporto e scala l'allineamento non cambia la classifica dei metodi, e che il colpevole sono frame e supporto. Un titolo solo sull'allineamento verrebbe smentito dalla nostra stessa Tabella 5. Il 2 è più memorabile ma mette in vetrina un difetto del nostro paper precedente, e LIT/DN (31 ago) notano che "normalizzazioni diverse nella stessa tabella" è già noto (REALY, M3DFB).

## 2. Tesi e contributi

**Tesi (3 frasi, in inglese come nell'abstract).**
(1) Distances used to evaluate 3D face reconstruction depend on choices made before any distance is computed: the normalization frame, the surface support, and pairwise alignment; each can change which pair, subject or method looks closer.
(2) On controlled faces with known correspondence from two independent 3DMMs, pairwise registration compresses inter-subject distances and degrades identity ranking, per-mesh frames let render-based metrics read position and scale, and kernel distances are fair only with unit mass; on real reconstructions, once support and scale are equalized, alignment no longer changes the ranking of methods.
(3) We release a meta-evaluation protocol with explicit controls (bounding-box proxy, equal patch, single frame, held-out subjects, subject bootstrap) and show that neither perceptual metrics on renders nor a registration-free learned metric escape these confounds outside their training distribution.

**Contributi.**

| # | Claim (inglese) | Prova | Stato |
|---|---|---|---|
| C1 | *A controlled meta-evaluation protocol with trivial-signal controls, replicated on two independent 3DMMs and on real scans.* | Reference sets TEX `sec:ref_sets`; controllo bbox: Spearman 0.47 con D_GT e AUC 0.996 su Multiface prima della normalizzazione (BD 11 set, notte); riproduzione del paper a max \|Δ\| 0.0008/0.008 (BD 11 set); contaminazione fb100 79/100 (BD 11 set) | BFM e Multiface misurati; baseline estese su ICT **da fare** |
| C2 | *Pairwise registration compresses inter-subject distances and degrades identity ranking, even at fixed topology.* | TEX `tab:alignment_effect` (Chamfer 0.729 → rigid ICP 0.600 → NICP 0.456/0.468, fb100); ICP rigido da solo −0.135 a topologia fissa, −0.032 cross (DN, 31 ago); compressione IQR: rigid 53.7%, NICP P2P 33.9% nel frame corretto (TEX `tab:distance_compression`, DN 31 ago); M3DFB: RLR+Chamfer 0.456, ICP 0.284 su 20 soggetti (DN, 17 ago) | misurato su BFM/fb100; **da rifare** su held-out e su ICT; 33.9% da confermare con tabella nel frame corretto |
| C3 | *Normalization frame and support are first-order confounds, larger than alignment on real reconstructions.* | GT dipende dal frame: 0.865 grezze vs 0.573 maxabs (DN, 17 ago); GT mai normalizzata, cinque frame nella stessa tabella (DN, 31 ago); LPIPS 0.739 → 0.590 normalizzando i render (BD 11 set); varifold sul rumore 0.136 (maxabs) → 0.307 (massa unitaria) (BD 4 ott); WS3b: Chamfer normalizzato per mesh ordina 3DDFA < SynergyNet < PRNet, su patch uguale in mm 3DDFA < PRNet < SynergyNet = ICP, τ 0.80 [0.59, 0.95] (TEX `tab:multiface_recon`, BD 4 ott); 3DDFA +14.8% di superficie nella sfera (BD 11 set) | misurato; frame rms per il modello appreso **in arrivo stasera** |
| C4 | *Negative results: learned and perceptual metrics do not escape these confounds out of distribution.* | Zero-shot BFM→ICT 0.30 vs Chamfer 0.44 (BD 11 set); espressioni −0.101 vs −0.065 (TEX `tab:expressions`); WS3b metrica appresa vs Chamfer τ 0.13–0.18 (TEX `tab:multiface_recon`); fattore Poisson FaceScape 0.406 → 0.109 (DN, 17 ago); CLIP/DINOv2 cross sotto il controllo bbox (TEX `tab:extended_baselines`); ArcFace 0.293 su REMESH ma 1.000 sul crop Multiface (BD 4 ott) | misurato; modelli ICT-only e congiunto **in arrivo stasera** |

## 3. Struttura (8 pagine CVPR)

**§1 Introduction (1 p).**
Claim: *Evaluation pipelines contain degrees of freedom (frame, support, alignment) that are rarely reported and can reorder results.*
Figura 1: schema della pipeline con i tre punti di scelta + un esempio di inversione reale: WS3b, stessi tre metodi, ordine che cambia passando da normalizzazione per mesh a patch uguale in mm.
Numeri disponibili: ordini WS3b (TEX `tab:multiface_recon`, BD 4 ott). Mancano: nessuno.

**§2 Related work (0.5 p).** Meta-valutazione (Sariyanidi, M3DFB, REALY), bias da Procrustes in morfometria (Koska 2026, Daboul 2018, Courtenay 2026, Taheri 2021: LIT §a, §g), metriche apprese/percettive (Shilova 2026 che allinea comunque con ICP, TGE, AlignFace, Lee 2025, Besnier 2023: LIT §i), meta-valutazione con umani (Jozwik 2022, arXiv:2503.16264: LIT §c, §f). Correggere la frase sul kernel (TEX riga 103) contestata da YJz1.

**§3 Meta-evaluation protocol (1.25 p).**
Claim: *A metric is judged by identity-ranking fidelity against references that break circularity along different axes, always next to a control that sees only the bounding box.*
Tabella 1: i quattro set di riferimento (REMESH-BFM, ICT-5000, Multiface identità, Multiface ricostruzioni) × cosa rompono (stesso 3DMM, 3DMM indipendente, nessuna D_GT, metodi reali) × n soggetti × statistica.
Contenuto: D_GT in un frame unico dichiarato; held-out per seed; bootstrap per soggetto; τ contro il nullo (0 ± 0.177); split-half. Paragrafo "evaluation pitfalls we found": i nove difetti di DN (sottocampionamento varifold, detector ArcFace, render non normalizzati, seed dello split, fb100 contaminato, raggio di ritaglio, falso "NoW", Chamfer normalizzato per mesh, varifold per area).
Numeri disponibili: dimensioni dei set (TEX `sec:ref_sets`); 79/100 (BD 11 set); bbox 0.47 / 0.996, dispersione da 48.3 a 0.18 (BD 11 set). Mancano: bbox su render non normalizzati nella tabella di REMESH held-out (todo TEX `sec:baseline_protocol`).

**§4 Alignment compresses identity differences (1.25 p).**
Claim: *Pair-specific registration shrinks the dynamic range of inter-subject distances non-uniformly, so ranking fidelity drops even when topology is fixed.*
Tabella 2: Chamfer / rigid ICP / NICP P2P / NICP P2Tri / M3DFB-RLR × {BFM held-out, ICT held-out} × {stessa topologia, cross no-crop}. Figura 2: densità raw vs registered (esiste, TEX `fig:registration_distance_compression`, da ricalcolare nel frame corretto).
Disponibili: 0.729/0.600/0.456/0.468 e 0.552/0.514/0.414/0.433 su fb100 (TEX `tab:alignment_effect`; per le baseline geometriche fb100 non è contaminato perché non addestrano, ma va detto); −0.135 / −0.032 (DN); IQR 53.7% e 33.9% (TEX, DN); M3DFB 0.456/0.284 su 20 soggetti (DN). Mancano: tabella su held-out BFM; tutta la colonna ICT; compressione nel frame corretto con CI.

**§5 Frames and normalization (1.25 p).**
Claim: *No per-mesh normalization is neutral; the frame decides what each metric can see, and mixing frames inflates margins.*
Tabella 3: stessa metrica sotto frame diversi. Righe: GT (grezza vs maxabs), Chamfer (maxabs vs area), varifold/currents (area vs maxabs vs massa unitaria), LPIPS/ArcFace (render grezzi vs normalizzati), bbox proxy. Colonne: stessa topologia, tassellazione, perturbazione.
Disponibili: 0.865 vs 0.573 (DN); LPIPS 0.739 → 0.590 normalizzando i render (BD 11 set); varifold cross no-crop 0.192 (maxabs) → 0.258 (massa unitaria), perturbazione 0.136 → 0.307 (BD 4 ott); "nessuna normalizzazione per mesh è neutra: area aiuta sul crop, maxabs sul rumore" (DN, 17 ago, senza numeri in tabella). Frame rms per il modello appreso: crop +5.2, tutte +3.1 su n=2 (DN, 19 ago); area unitaria +0.032 ± 0.031 su 3 seed (BD 4 ott). Mancano: Chamfer maxabs vs area su held-out in tabella; bbox su render grezzi in Spearman; risultato rms su 3 seed (stasera).

**§6 Support and real reconstructions (1 p).**
Claim: *On real reconstructions, apparent disagreement between aligned and unaligned criteria is a support and scale artifact; at equal patch and scale, they agree.*
Tabella 4: TEX `tab:multiface_recon` (classifiche + τ). Box laterale: WS3a crop.
Disponibili: 1.242/1.260/1.519 mm ICP; 2.368/2.488/2.559 mm Chamfer patch uguale; τ 0.80 [0.59, 0.95], 69% ordine identico; τ 0.85 Chamfer in mm (CI mancante); split-half 0.86–0.98 (TEX, BD 4 ott). WS3a: controllo bbox 0.95 su 4 celle su 6 e 0.42 sul crop; geometriche 0.58–0.65; ArcFace 1.000, LPIPS 0.73–0.84 sul crop (BD 4 ott). Mancano: CI per τ 0.85 e 0.13 (todo TEX); test della metrica appresa con input a patch uguale.

**§7 Do learned and perceptual metrics escape the confounds? (1.25 p).**
Claim: *Perceptual metrics on renders and a registration-free learned metric help only inside their training distribution.*
Tabella 5: baseline estese REMESH held-out (TEX `tab:extended_baselines`) + righe del modello appreso (BFM-only, ICT-only, congiunto) × {BFM, ICT}. Tabella 6 (piccola): espressioni + zero-shot + FaceScape Poisson. Studio umano, se arriva: accordo con la maggioranza umana per GT, Chamfer, LPIPS, metrica appresa.
Disponibili: tutta `tab:extended_baselines`; 0.30 vs 0.44 e 0.83 vs 0.91 su coppie di soggetti clean (DN); espressioni (TEX); Poisson 0.406 → 0.109 (DN); margine latent − Chamfer in-domain sulle 30 coppie 0.421 (v1) (BD 4 ott). Mancano: righe ICT-only e congiunto (stasera); riga appresa su held-out in `tab:extended_baselines` (todo TEX); studio umano (zero risposte, BD 4 ott).

**§8 Recommendations and limitations (0.5 p).**
Claim: *A five-line checklist: report the frame, equalize support, use unit-mass kernels, add a bounding-box control, evaluate on held-out subjects.* Limiti: 13 soggetti reali, 3 metodi senza FLAME (DECA, MICA esclusi), D_GT sintetico, nessun NoW.

Appendice: riproduzione della Tabella 1 NeurIPS (gate), metodo DiffusionNet completo, matrice 6×6, distinzione delle identità train/held-out, FaceScape.

## 4. Obiezioni dei reviewer

| Obiezione | Chi | In B | Coperta? |
|---|---|---|---|
| Solo dati sintetici, nessun benchmark reale | tutti, AC | §6: Multiface ricostruzioni (3 metodi, 3075 immagini per metodo) e identità (13 soggetti); il risultato reale è parte della tesi, anche se va contro "alignment hurts" | **parziale**: niente NoW, 13 soggetti |
| Mai applicata a ricostruzione image-to-3D | YJz1, Z1mX | §6 è esattamente questo, con tre criteri | sì, su piccola scala |
| Circolarità D_GT | Z1mX | In B la metrica appresa non è il contributo, quindi la circolarità del training pesa meno. D_GT resta il riferimento sintetico: si replica su ICT (D_GT indipendente), si cita Jozwik 2022 (LIT §iv), e Multiface non usa D_GT | **scoperta** la validazione umana di D_GT finché lo studio ha zero risposte |
| Poche baseline | bhBZ, YJz1, AC | §5 e §7: varifold, currents, LPIPS, ArcFace, CLIP, DINOv2 + controllo bbox | sì, tranne **DPDist** (non implementato: dichiararlo o implementarlo) |
| Frase sul kernel (Besnier) | YJz1 | riscrivere TEX riga 103 citando Besnier 2023 | da fare (testo) |
| 3000 volti, un solo 3DMM; 10^5 mesh; FLAME | YJz1 | ICT-5000 (5000 identità × 6 topologie) e modello congiunto | **parziale**: niente FLAME (licenza), niente 10^5; in B conta meno perché il modello non è il contributo |
| Solo neutri | Z1mX | espressioni per soggetto su ICT (TEX `tab:expressions`); identità Multiface con 10 espressioni | sì |
| User study | bhBZ, Z1mX | pacchetto pronto, 300 triplette | **scoperta**: zero risposte al 4 ottobre |
| Novità tecnica limitata | bhBZ | B rinuncia al claim di metodo: il contributo è protocollo + controlli + risultati | risposta di framing, non di dati |
| Claim cross-topologia troppo largo | Z1mX | già ridimensionato nel draft; ICT lo misura su un secondo 3DMM | sì |
| Invarianza rigida non dichiarata; 3DMM non nominato; bias demografico | Z1mX | già nel draft (BD 10 set, WS6) | sì (testo) |
| Identità train/held-out distinte | Z1mX | appendice esistente con `\todo{numeri}` | **da fare** (calcolo di minuti) |
| Dataset non disponibile | YJz1 | rilascio di coefficienti, split, matrici GT, codice dei controlli | promessa, da eseguire |

## 5. Dal draft attuale: cosa resta e cosa si toglie

**Si riusa** (quasi invariato): Problem Formulation (D_GT come riferimento, non come verità); REMESH construction; `sec:heldout_contamination`; `sec:ref_sets`; `sec:baseline_protocol`; `sec:statistics`; `tab:extended_baselines` e il suo testo; `tab:multiface_recon` e il testo del risultato negativo; `tab:expressions`; il paragrafo cross-3DMM; la sezione Distance Compression con figure (numeri da aggiornare a 33.9%); i paragrafi sui dati reali in Limitations; appendici bootstrap e distinzione delle identità; related work su Shilova/TGE/AlignFace.

**Si riscrive**: abstract, introduzione e lista dei contributi (oggi incentrati sul metodo); `tab:alignment_effect` rifatta su held-out (riga latent fuori o in grigio); Conclusions.

**Si toglie o si sposta in appendice**: §Method intera (encoder e loss) → mezza colonna in §7 + appendice; `tab:clean_xtopo_chamfer_latent_matrices` (fb100, contaminata per la riga latent) → appendice come gate di riproduzione; `sec:computational_cost` e la tabella dei tempi → una frase o nulla; FaceScape OOD (`tab:faceverse_summary`, few-shot, forgetting) → una riga sul fattore Poisson, il resto fuori; paragrafo hardware RTX 5090 → appendice; ogni frase "the learned metric consistently preserves rankings better".

## 6. Rischi e mitigazioni (budget: meno di 2 settimane, 4–18 ottobre)

| Critica attesa | Mitigazione | Costo |
|---|---|---|
| "Solo analisi, nessun metodo" | Rendere il protocollo un artefatto: toolkit di controlli (bbox, patch uguale, massa unitaria, frame unico) + checklist; mostrare che applicarla ricompone l'accordo tra criteri (già vero in WS3b: τ 0.80). Citare il precedente di track metodologici (LIT §c). | 3–4 giorni di codice/packaging |
| "13 soggetti reali" | (a) FaceScape: 110 soggetti reali con variante remesh e matrice GT già recuperati (PLAN, agg. 10 set): ripetere su di loro l'esperimento di compressione e di frame; (b) bootstrap per soggetto e split-half già riportati; (c) dichiararlo nel titolo della sezione | (a) 1–2 giorni CPU |
| "Già noto (REALY, M3DFB)" | Posizionarsi su ciò che manca a loro (LIT, DN 31 ago): controllo bbox, massa unitaria, meccanismo del bordo nel divisore di scala, replica su due 3DMM, inversione sui dati reali | testo |
| "Il metodo non funziona, quindi hanno scritto un paper di analisi" | Il modello appreso esce dal titolo e dai contributi; resta come sonda con risultati riportati per intero, anche positivi in-domain | testo |
| "D_GT sintetico è comunque arbitrario" | Replica su ICT; studio umano: reclutamento in laboratorio questa settimana (azione dell'autore); se a 18 ottobre ha meno partecipanti del minimo del piano (25–30), si riporta come pilota o si toglie | 0 calcolo, dipende dall'autore |
| "ArcFace su render a 1.000 sul crop: la soluzione è percettiva" | Mostrare che su REMESH ArcFace fa 0.293 a parità di topologia; aggiungere le percettive su ICT held-out | ~6 h di T4 (stima da BD 11 set) |
| "Compressione = banale riscalatura" | Spearman cala (cambia l'ordine, non solo la scala); aggiungere rapporto IQR per gruppo di coppie nel frame corretto | ore di CPU |
| "La metrica appresa discorda su Multiface per l'input, non per l'identità" | Rifare WS3b con input appresi a patch uguale (TEX dice "we do not test": va testato) | 1 giorno |

Esperimenti in ordine di priorità, tutti dentro le 2 settimane: (1) `tab:alignment_effect` su held-out BFM e ICT; (2) compressione nel frame corretto; (3) tabella dei frame (§5) su held-out; (4) baseline estese su ICT; (5) FaceScape-110 per compressione e frame; (6) WS3b con input a patch uguale; (7) distinzione delle identità; (8) DPDist o esclusione motivata.

## 7. Venue alternative

- **3DV 2027**: deadline 28 agosto 2026, già passata ([3DV 2027 dates](https://3dvconf.github.io/2027/dates/)); 3DV 2028 non annunciato (non verificato).
- **WACV 2027**: round 2 chiuso il 28 agosto 2026 ([WACV 2027 dates](https://wacv.thecvf.com/Conferences/2027/Dates)); WACV 2028 non verificato.
- **ICCV 2027** (Hong Kong): deadline stimata attorno al 5 marzo 2027 da aggregatori, call ufficiale non uscita ([fonte](https://researchtheta.com/conferences/iccv-2027/)): **non verificato**. È la riserva naturale per B, con lo studio umano finito.
- **TPAMI / IJCV**: sottomissione continua, nessuna scadenza; adatti a B in versione lunga (con FaceScape, appendici e risultati negativi per intero), tempi di review di mesi.
- **NeurIPS 2027, track Evaluations & Datasets** (ex D&B, rinominato: [NeurIPS su X](https://x.com/NeurIPSConf/status/2036184924001022405)): B è più vicino allo spirito del track del paper rifiutato lì nel 2026, ma è lo stesso track. Un aggregatore indica 21 gennaio 2027 come stima ([mlciv](https://mlciv.com/ai-deadlines/)), mentre di solito la scadenza è in primavera: **non verificato**.

## 8. Varianti condizionali (tabella cross-3DMM di stasera)

Criterio di "batte" fissato prima di guardare: Spearman del congiunto sopra Chamfer su BFM held-out e su ICT held-out, sul protocollo mesh-pair cross-topologia, con CI per soggetto che non si sovrappongono (PLAN WS2: "congiunto sopra Chamfer su tutte le coppie").

**Se il congiunto batte Chamfer su entrambi.** C4 cambia segno e diventa: *"Learned registration-free metrics escape the confounds only when the training distribution covers the target family: BFM-only loses zero-shot (0.30 vs 0.44), joint training recovers it."* È una raccomandazione costruttiva e risponde a "solo analisi". La metrica appresa sale da sonda a "strumento raccomandato con condizioni"; §7 cresce di mezza pagina a scapito di §4. Restano in B, non in A: il gate del diario (BD 4 ott) vuole anche lo studio umano favorevole, e WS3b (τ 0.13–0.18) va rifatto con il congiunto prima di dire qualunque cosa sui dati reali. Da fare subito: congiunto su espressioni e su WS3b (ore di GPU).

**Se non lo batte.** C4 resta un risultato negativo pulito: *"A registration-free learned metric does not by itself fix evaluation; its advantage is in-distribution only."* La raccomandazione del paper diventa Chamfer in frame unico e su patch uguale, più i controlli; la metrica appresa scende a una riga per tabella e il metodo va tutto in appendice. Il titolo raccomandato regge; il 4 ("What the Bounding Box Knows") diventa un'alternativa forte perché sposta il peso sui controlli. A è escluso al gate del 24 ottobre.

In entrambi i casi il frame rms (stasera) entra in §5: se il guadagno di n=2 (crop +5.2, DN) regge su 3 seed appaiati, è la prova che il frame conta anche per un modello appreso; se no, lo si riporta in una riga e si chiude la questione.
