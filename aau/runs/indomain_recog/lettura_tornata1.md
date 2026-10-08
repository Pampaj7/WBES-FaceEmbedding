# Lettura (scritta dopo i risultati, con le regole fissate nel protocollo)

**Leak.** Nessun soggetto valutato e' nel training del congiunto (0/108 BFM, 0/89 ICT, ricontrollato su
`splits.json`). Restano le due note del protocollo: 16 dei 108 soggetti BFM (5 dei 19 di BFM-19) erano
nell'eval online del congiunto, cioe' hanno pesato sulla scelta del checkpoint; il BFM-only e' confrontabile
solo su 19 soggetti, perche' 89 dei 108 erano nel suo training.

**1. In dominio, senza crop, il compito e' saturo.** Il congiunto fa rank-1 0.998 (BFM) e 1.000 (ICT),
AUC 1.000; NICP P2Tri fa lo stesso (1.000 / 1.000 su BFM, 1.000 / 0.994 su ICT). Lettura del protocollo:
congiunto **pari** a NICP sul rank-1 in entrambi i domini, pari sull'AUC in BFM e **sopra** in ICT
(+0.006 [+0.004, +0.009]); **sopra** a Chamfer, ICP + Chamfer e ArcFace su normal map su rank-1 e AUC in
entrambi i domini. Il riconoscimento in dominio fra topologie diverse non separa il congiunto da NICP: lo
separa da tutto il resto. Anche il congiunto contro il BFM-only (BFM-19) e' pari, 1.000 contro 1.000.

**2. Con il crop da un lato il congiunto stacca le pipeline geometriche, non ArcFace.** Rank-1 0.944 (BFM)
e 0.993 (ICT) contro 0.686 e 0.600 di NICP P2Tri (delta +0.257 e +0.393, **sopra**), e Chamfer / ICP sotto
0.31. Contro ArcFace su normal map: **sotto** sul rank-1 in BFM (-0.041 [-0.073, -0.014]), pari in ICT, e
**sopra** sull'AUC in entrambi. BFM-19, crop: congiunto 0.979 contro BFM-only 0.942, delta +0.037
[-0.005, +0.095], pari (19 soggetti: il test e' senza potenza).

**3. Con le espressioni il congiunto e' il metodo peggiore.** Espressione contro espressione (stessa
topologia ICT): rank-1 0.822, **sotto** Chamfer (0.867), ICP + Chamfer (0.896), NICP P2Tri (0.905) e ArcFace
(1.000); AUC 0.978, **pari** alle tre geometriche e **sotto** ArcFace (1.000). Con la galleria neutra il
congiunto sale a 0.962 ma resta **sotto** tutti (gli altri da 0.998 a 1.000). Due avvertenze che non cambiano
il segno: (a) qui non varia la topologia, cioe' manca proprio la difficolta' in cui le pipeline geometriche
crollano (punto 2), e Chamfer su mesh con la stessa connettivita' e' quasi una distanza vertice-vertice;
(b) il congiunto non ha mai visto espressioni ICT in training. ArcFace a 1.000 non e' un errore di
indicizzazione: i png di controllo (`arcface/rexpr/normals/control`) mostrano espressioni diverse per lo
stesso soggetto, e su ICT neutro fra topologie lo stesso ArcFace fa 0.981. Per il paper: il riconoscimento
d'identita' attraverso le espressioni e' un limite da dichiarare, non un risultato.

**4. Tempi (stesso nodo L40S, AMD EPYC 9454; parti CPU a un thread).** Congiunto: 2.34 s per mesh
[IQR 1.31-4.89], quasi tutto operatori DiffusionNet su CPU (2.28 s; l'embedding sulla GPU e' 57 ms, sulla CPU
443 ms); confronto fra due embedding 2.3 us; retrieval 1:100 / 1:1.000 / 1:10.000 in 0.02 / 0.19 / 2.2 ms
su CPU (0.04 / 0.05 / 0.07 ms su GPU). NICP P2Tri 1.42 s per coppia [1.37-1.82], ICP + Chamfer 34 ms per
coppia, ArcFace su normal map 0.54 s per mesh (render 0.20 s + 3 embedding 0.35 s). Quindi, a parita' di
accuratezza senza crop, un confronto 1:1 da zero costa al congiunto due iscrizioni (~4.7 s) contro 1.4 s di
NICP, ma con la galleria gia' iscritta una query 1:N costa ~2.3 s + N x 2.3 us, contro N x 1.42 s per NICP
(stima, non misurata: 1:10.000 ~ 4 ore a un core). Il punto di pareggio e' a N = 2.
Avvertenze: (i) il nodo dei tempi ospitava nello stesso momento un mio job faceBench a 48 core (core
diversi, ma memoria e frequenza condivise): tutte le voci sono state misurate in quelle condizioni, in fila
nello stesso job, e i tempi assoluti possono essere gonfiati; (ii) l'embedding comprende la lettura degli
operatori precalcolati da /tmp, che in uso reale verrebbero dalla memoria: totale per mesh leggermente
sovrastimato.

**Controlli.** Le distanze del congiunto dagli embedding coincidono con `latent_distance` delle eval WS2 a
meno di 3.1e-3 (mediana delle distanze 0.76-1.27, cioe' < 0.5%): stesso modello e stessa catena, differenze
numeriche fra esecuzioni GPU. faceBench: 0 coppie fallite su 412.590.
