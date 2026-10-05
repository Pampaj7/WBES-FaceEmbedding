# WS3b Multiface: la classifica dei metodi di ricostruzione cambia se si allinea?

Tre metodi a pesi pubblici (3DDFA_V2, SynergyNet, PRNet) su tutte le immagini frontali di Multiface con mesh tracciata: 3ddfa_v2 3075, synergynet 3075, prnet 3075 ricostruzioni.
CI 95% bootstrap su 1000 repliche, ricampionando i 13 soggetti.

## (a) Classifica per criterio

Punteggio = media sui soggetti della mediana delle loro ricostruzioni; piccolo = meglio. `p_first` = frazione di repliche bootstrap in cui il metodo e' primo.

**sim_icp_p2s_median** — similarita' da ICP su tutta la superficie, punto-superficie, mm

| rango | metodo | punteggio [CI 95%] | p_first |
|---|---|---|---|
| 1 | 3ddfa_v2 | 1.242 [1.098, 1.372] | 0.62 |
| 2 | prnet | 1.26 [1.115, 1.41] | 0.38 |
| 3 | synergynet | 1.519 [1.354, 1.69] | 0.00 |

**sim_icp_p2s_mean** — come sopra, media invece che mediana, mm

| rango | metodo | punteggio [CI 95%] | p_first |
|---|---|---|---|
| 1 | 3ddfa_v2 | 1.489 [1.324, 1.646] | 0.64 |
| 2 | prnet | 1.506 [1.347, 1.653] | 0.36 |
| 3 | synergynet | 1.8 [1.624, 1.981] | 0.00 |

**chamfer_raw** — Chamfer grezza (sola normalizzazione maxabs)

| rango | metodo | punteggio [CI 95%] | p_first |
|---|---|---|---|
| 1 | 3ddfa_v2 | 0.06723 [0.06247, 0.07176] | 1.00 |
| 2 | synergynet | 0.07289 [0.06496, 0.07938] | 0.00 |
| 3 | prnet | 0.07574 [0.07076, 0.08106] | 0.00 |

**chamfer_icp_mm** — Chamfer in mm, scala dalla similarita' ICP invece che maxabs per mesh

| rango | metodo | punteggio [CI 95%] | p_first |
|---|---|---|---|
| 1 | 3ddfa_v2 | 2.491 [2.326, 2.651] | 0.91 |
| 2 | prnet | 2.565 [2.398, 2.731] | 0.06 |
| 3 | synergynet | 2.604 [2.417, 2.785] | 0.04 |

**chamfer_gtclip_mm** — Chamfer in mm sulla recon allineata e RITAGLIATA alla patch GT (supporto uguale per costruzione)

| rango | metodo | punteggio [CI 95%] | p_first |
|---|---|---|---|
| 1 | 3ddfa_v2 | 2.368 [2.2, 2.538] | 1.00 |
| 2 | prnet | 2.488 [2.342, 2.643] | 0.00 |
| 3 | synergynet | 2.559 [2.401, 2.723] | 0.00 |

**latent_v1** — distanza latente, checkpoint v1

| rango | metodo | punteggio [CI 95%] | p_first |
|---|---|---|---|
| 1 | 3ddfa_v2 | 0.7211 [0.6658, 0.7779] | 0.93 |
| 2 | synergynet | 0.7488 [0.6962, 0.8029] | 0.07 |
| 3 | prnet | 0.9191 [0.8601, 0.9809] | 0.00 |

**now_median** — DIAGNOSTICO, fuori scope: NoW da 7 landmark (vedi nota)

| rango | metodo | punteggio [CI 95%] | p_first |
|---|---|---|---|
| 1 | prnet | 1.816 [1.635, 2.018] | 0.76 |
| 2 | synergynet | 1.839 [1.666, 2.044] | 0.24 |
| 3 | 3ddfa_v2 | 6.332 [5.598, 7.056] | 0.00 |

**now_mean** — DIAGNOSTICO, fuori scope: NoW da 7 landmark, media

| rango | metodo | punteggio [CI 95%] | p_first |
|---|---|---|---|
| 1 | synergynet | 2.099 [1.906, 2.303] | 0.52 |
| 2 | prnet | 2.103 [1.896, 2.326] | 0.48 |
| 3 | 3ddfa_v2 | 6.223 [5.495, 6.912] | 0.00 |

> **`now_median` / `now_mean` sono diagnostici, non risultati.** Il protocollo NoW vero (similarita' da 7 landmark, nessun ICP) e' implementato in `ws3b_geometric.py`, ma ha bisogno di 7 landmark sulla scansione e Multiface non li distribuisce. Quelli stimati da `ws3b_landmarks.py` trasferendoli dai 68 iBUG dei tre metodi hanno 5-7 mm di dispersione fra campioni e 8-10 mm di scarto fra metodi, contro un errore di ricostruzione di 1.2-1.6 mm: l'errore dell'annotazione e' 4 volte il segnale. Per questo i due criteri non entrano ne' nel Kendall tau ne' nella classifica primaria, e la colonna resta nei csv solo come traccia. Il criterio allineato di riferimento e' `sim_icp_p2s_median`.

### Classifica per soggetto

Ordine dei tre metodi per ciascun soggetto e criterio (dal migliore).

| soggetto | sim_icp_p2s_median | chamfer_raw | latent_v1 |
|---|---|---|---|
| 002421669 | prnet > 3ddfa_v2 > synergynet | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > synergynet > prnet |
| 002539136 | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > prnet > synergynet | synergynet > 3ddfa_v2 > prnet |
| 002643814 | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > synergynet > prnet |
| 002645310 | prnet > 3ddfa_v2 > synergynet | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > prnet > synergynet |
| 002757580 | 3ddfa_v2 > synergynet > prnet | synergynet > 3ddfa_v2 > prnet | synergynet > 3ddfa_v2 > prnet |
| 002914589 | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > synergynet > prnet | 3ddfa_v2 > synergynet > prnet |
| 2183941 | synergynet > prnet > 3ddfa_v2 | synergynet > 3ddfa_v2 > prnet | 3ddfa_v2 > synergynet > prnet |
| 5067077 | synergynet > prnet > 3ddfa_v2 | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > synergynet > prnet |
| 5372021 | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > synergynet > prnet |
| 6674443 | 3ddfa_v2 > prnet > synergynet | synergynet > 3ddfa_v2 > prnet | synergynet > 3ddfa_v2 > prnet |
| 6795937 | 3ddfa_v2 > prnet > synergynet | 3ddfa_v2 > synergynet > prnet | 3ddfa_v2 > synergynet > prnet |
| 7889059 | prnet > synergynet > 3ddfa_v2 | 3ddfa_v2 > synergynet > prnet | 3ddfa_v2 > synergynet > prnet |
| 8870559 | prnet > 3ddfa_v2 > synergynet | 3ddfa_v2 > synergynet > prnet | 3ddfa_v2 > prnet > synergynet |

## (b) Accordo fra le classifiche: Kendall tau

tau-b fra le classifiche dei tre metodi, calcolato dentro ogni soggetto e mediato sui soggetti. Su tre elementi tau vale 1, 1/3, -1/3 o -1.

Le due colonne "caso" sono il metro: sotto l'ipotesi nulla di classifiche indipendenti il tau medio ha valore atteso 0 e deviazione standard **0.177** su 13 soggetti (varianza esatta 11/27 sulle 6 permutazioni di tre elementi), e la frazione di soggetti con ordine identico vale **16.7%** (1 permutazione su 6).

| criteri | tau medio per soggetto [CI 95%] | caso: 0 +- sd | tau/sd | tau fra le classifiche globali | soggetti con ordine identico | caso |
|---|---|---|---|---|---|---|
| sim_icp_p2s_median vs chamfer_raw | 0.179 [-0.179, 0.538] | 0 +- 0.177 | 1.01 | 0.333 | 23% | 16.7% |
| sim_icp_p2s_median vs latent_v1 | -0.026 [-0.282, 0.179] | 0 +- 0.177 | -0.14 | 0.333 | 0% | 16.7% |
| chamfer_raw vs latent_v1 | 0.590 [0.383, 0.795] | 0 +- 0.177 | 3.33 | 1.000 | 46% | 16.7% |

### Tetto di affidabilita': split-half dentro ogni criterio

Le immagini di ogni soggetto spezzate a caso in due meta', classifica ricalcolata su ciascuna, tau fra le due, mediato sui soggetti e su 200 ripetizioni. Nessun tau FRA criteri della tabella qui sopra puo' superare questi valori.

| criterio | tau split-half [2.5%, 97.5%] |
|---|---|
| sim_icp_p2s_median | 0.982 [0.897, 1.000] |
| chamfer_raw | 0.914 [0.846, 1.000] |
| latent_v1 | 0.924 [0.846, 1.000] |

## (c) La ricostruzione conserva l'identita'?

AUC = P(distanza fra ricostruzioni dello stesso soggetto < distanza fra ricostruzioni di soggetti diversi), 0.5 = caso. Nessuna verita' a terra entra nel conto. Classi: (a) stesso soggetto stessa espressione, (b) stesso soggetto espressione diversa, (c) soggetti diversi stessa espressione, (d) tutto diverso.

| metrica | metodo | ab_vs_cd | b_vs_c | a_vs_c |
|---|---|---|---|---|
| chamfer_raw | 3ddfa_v2 | 0.913 [0.852, 0.961] | 0.842 [0.709, 0.935] | 0.964 [0.938, 0.984] |
| chamfer_raw | synergynet | 0.930 [0.876, 0.968] | 0.877 [0.758, 0.950] | 0.974 [0.956, 0.989] |
| chamfer_raw | prnet | 0.937 [0.889, 0.968] | 0.890 [0.795, 0.953] | 0.978 [0.959, 0.992] |
| latent_v1 | 3ddfa_v2 | 0.899 [0.837, 0.952] | 0.824 [0.698, 0.918] | 0.958 [0.928, 0.979] |
| latent_v1 | synergynet | 0.921 [0.866, 0.963] | 0.868 [0.753, 0.940] | 0.971 [0.952, 0.988] |
| latent_v1 | prnet | 0.918 [0.868, 0.955] | 0.857 [0.754, 0.933] | 0.968 [0.942, 0.986] |

## Limiti di questo confronto

**1. I supporti non coincidono.** Il rapporto fra l'area della patch ricostruita e quella della patch GT vale 3ddfa_v2 **1.144**, synergynet **1.059**, prnet **1.083**. Il raggio del ritaglio e' ormai giusto (mediana 94.84 mm, 95.71 mm, 95.10 mm contro i 95.0 della GT), quindi la differenza residua e' forma, non supporto: un metodo mette piu' superficie dentro la stessa sfera. `chamfer_raw`, che guarda il supporto attraverso la maxabs, paga quella differenza; la riga `chamfer_gtclip_mm` e' il controllo che la toglie, perche' ritaglia la ricostruzione allineata con la stessa sfera della GT.

**2. Il criterio allineato e' unidirezionale e su ricostruzione NON ritagliata.** `sim_icp_p2s_*` misura la distanza dai vertici della GT ritagliata alla superficie ricostruita INTERA, in un verso solo. E' la convenzione di NoW (superficie in piu' non disturba), ma vuol dire che un metodo che ricostruisce piu' faccia del necessario non viene mai penalizzato, e che meta' dell'informazione -- quanto della ricostruzione non ha un corrispondente nella GT -- non entra nel numero.

**3. I 7 landmark della GT sono stimati, non misurati.** Multiface non distribuisce landmark: quelli usati vengono trasferiti dai 68 iBUG dei tre metodi con l'ICP inverso, e il disaccordo fra i tre metodi sullo stesso punto e' di **8.9 mm** in mediana e **10.6 mm** sul punto peggiore, contro un errore di ricostruzione di 1.2-1.6 mm. Da li' viene anche l'ex-ex che fissa il raggio del ritaglio. E' il motivo per cui `now_*` resta diagnostico, e una ragione in piu' per non fidarsi del terzo decimale di nessuna riga.

**4. Primo e secondo non sono distinguibili sul criterio allineato.** Su `sim_icp_p2s_median` 3ddfa_v2 ha p_first 0.62 e prnet ha p_first 0.38: il bootstrap sui 13 soggetti mette 3ddfa_v2 primo in poco piu' della meta' delle repliche. La distanza fra il primo e il secondo (1.242 contro 1.260) e' dentro i CI di tutti e due. La differenza che si puo' sostenere e' fra i primi due e il terzo, non fra il primo e il secondo.

