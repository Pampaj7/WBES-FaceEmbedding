# Form, shape e normalizzazione della GT: letteratura verificata (9 ottobre 2026)

Ricerca di un agente Sonnet, con metadati verificati su Crossref, PubMed e testi integrali. Le voci non verificate sono indicate.

## Risultati chiave per il paper
- **Distanza size-and-shape, verificata:**
  - d_S² = S1² + S2² − 2 S1 S2 cos ρ (rotazioni per coppia, nessuna scala; ρ è la distanza riemanniana di shape);
  - equivalente: **d_S² = (S1−S2)² + S1·S2·d_P²**, con d_P = 2 sin(ρ/2). Taglia e forma si separano esattamente;
  - fonti: Dryden & Mardia 2016 (doi:10.1002/9781119072492), pacchetto R `shapes` (`ssriemdist`), Varano et al. arXiv:1510.00708 (nella v1 il segno di eq. 0.8 è un refuso).
- **Nessun lavoro di valutazione della ricostruzione 3D del volto separa form e shape:** lo spazio per il nostro contributo c'è.
- **Il filone metrico va nella direzione "taglia = tratto":**
  - MICA (Zielonka et al. 2022, doi:10.1007/978-3-031-19778-9_15) mostra che la scala ottimizzata gonfia i punteggi NoW (FLAME medio 1.92 → 1.53 mm) e propone un protocollo solo rigido;
  - Klingenberg 2020 (doi:10.1007/s11692-020-09520-y): l'effetto Pinocchio riguarda la localizzazione delle differenze, non la distanza globale; la taglia è biologicamente rilevante;
  - Russ et al. 2006 (doi:10.1109/CVPR.2006.13): normalizzare elimina la taglia 3D utile al riconoscimento;
  - Cole et al. 2016 (doi:10.1371/journal.pgen.1006174): la taglia del volto è ereditabile.
- **Percezione:** tolleranza a stiramenti e scala globali (Hole et al. 2002, doi:10.1068/p3252; Sandford & Burton 2014, doi:10.1016/j.cognition.2014.04.005; Zhao & Chubb 2001, doi:10.1016/S0042-6989(01)00202-4), ma sensibilità a piccoli spostamenti dei tratti (Haig 1984, doi:10.1068/p130505). Jozwik et al. 2022 (doi:10.1073/pnas.2115047119): la distanza euclidea nel BFM predice i giudizi umani. Nessuno studio su taglia 3D assoluta e identità: la percezione non decide la questione da sola.

## Riferimenti
- **Form/shape:**
  - Kendall 1984, doi:10.1112/blms/16.2.81;
  - Kendall 1989, doi:10.1214/ss/1177012582;
  - Dryden, Koloydenko & Zhou 2009, doi:10.1214/09-AOAS249;
  - Klingenberg 2016, doi:10.1007/s00427-016-0539-2 (terminologia size, shape, form, conformation);
  - Mitteroecker et al. 2013, doi:10.4404/hystrix-24.1-6369 (letto solo il titolo);
  - Bookstein 1989, doi:10.2307/2992387 (solo titolo).
- **Pinocchio e resistant fit:**
  - Chapman 1990, Michigan Morphometrics Workshop, pp. 251-267 (non letto direttamente);
  - Siegel & Benson 1982, Biometrics 38:341, PMID 6810969;
  - Rohlf & Slice 1990, doi:10.2307/2992207;
  - Walker 2000, doi:10.1080/106351500750049770;
  - von Cramon-Taubadel et al. 2007, doi:10.1002/ajpa.20616.
- **Metrica appresa su Procrustes:** Gill, Ritov & Dror 2007, doi:10.1007/978-3-540-76858-6_63.
- **EDMA:**
  - Lele & Richtsmeier 1991, doi:10.1002/ajpa.1330860307;
  - libro 2001, ISBN 9780849303197;
  - Rohlf 2000, PMID 10727966 (bassa potenza dei test EDMA).
- **Antropometria:**
  - Garson 1885, doi:10.2307/2841484 (Accordo di Francoforte; data 1882 o 1884 da verificare);
  - Farkas 1994 (non letto);
  - Gupta, Markey & Bovik 2010, doi:10.1007/s11263-010-0360-8 (abstract non letto).
- **Valutazione della ricostruzione:**
  - NoW, Sanyal et al. 2019, doi:10.1109/CVPR.2019.00795 (rotazione, traslazione, scala opzionale);
  - REALY, Chai et al. 2022, doi:10.1007/978-3-031-20074-8_5 (GT riscalata per scansione, poi ICP rigido per regioni);
  - Sariyanidi et al. 2023, IJCB, doi:10.1109/IJCB57857.2023.10448898 (allineamento rigido su 5 punti);
  - M3DFB, Sariyanidi et al. 2025, doi:10.1109/FG61629.2025.11099357.
