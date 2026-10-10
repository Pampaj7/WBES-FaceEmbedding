# Diagnostica D, emendamento 1 (prima dei numeri): IC di sola valutazione in D2, dettagli della sonda

Scritto l'11 ottobre 2026 dal coder, dopo il protocollo (`PROTOCOL_D.md`, sha256 `b9f5378a...`, commit 251bdd6) e
mentre si scriveva il codice, PRIMA di qualunque numero di D1 o D2 (nessuno Spearman, embedding o fit nuovo e' stato
calcolato). Non cambia insiemi, righe, semi, regole o soglie del protocollo: aggiunge una colonna descrittiva e
precisa due dettagli d'implementazione della sonda che il protocollo lasciava impliciti. L'impronta sha256 sta nel
messaggio del commit che lo introduce e in `PROTOCOL_D_emendamento_1.sha256`.

## 1. D2: IC di sola valutazione (descrittivo, fuori dalla regola)

Nel bootstrap dell'intera procedura (sez. 4 del protocollo) ogni replica estrae 100 soggetti con reinserimento: i
soggetti distinti sono circa 63 e ogni fold di training ne ha circa 32 (con pesi che sommano a circa 50), contro i 50
distinti della stima puntuale. Una sonda addestrata su meno soggetti distinti ordina peggio: l'IC del protocollo e'
quindi spostato verso il basso rispetto alla stima puntuale (conservativo, oltre alla variabilita' dello split gia'
dichiarata). Si aggiunge, **solo descrittivo**, l'IC di sola valutazione: predizioni fuori fold FISSE dei 50 split
della stima puntuale; 1000 repliche di conteggi per soggetto `bincount(default_rng(SeedSequence([20261117, b]))
.integers(0, 100, 100))`, uguali per tutti gli split; per replica, rho pesato c_a c_b sulle righe di test di ogni fold,
media sui 100 fold; IC 95% percentile del delta sonda - riferimento. Colonne `delta_evalci_low`, `delta_evalci_high`
di `d2_probe.csv`. **R3 resta con l'IC del protocollo**; se i due IC portano a letture diverse lo si scrive.

## 2. Dettagli della sonda

- CV interna: standardizzazione delle feature e base PCA sono quelle del fold esterno di training (i soggetti di
  validazione interna stanno nella base, quindi l'errore sui punteggi e' l'errore sul vettore intero); la CV interna
  sceglie solo lambda.
- Bootstrap: lo split a 2 fold divide a meta' (parte intera e resto) i soggetti DISTINTI estratti; i pesi c_s entrano in
  standardizzazione, PCA, ridge e CV interna (fold interni sui soggetti distinti); righe di test con entrambi i
  soggetti nel fold di test, peso c_a c_b.
- Controllo K1: lo split della ripetizione r viene da `SeedSequence([20261116, r])`, la permutazione dei bersagli fra
  i soggetti di training dallo stesso generatore.

## 3. D2: l'atteso di K1 era sbagliato

Il protocollo dice "K1 permutazione: rho atteso ~ 0". E' sbagliato, e lo mostra la prova sintetica di
`aau/diagnostics/test_diag.py` (dati finti, nessun dato valutato; job 1067961): una ridge addestrata su bersagli
permutati e' comunque una mappa LINEARE dell'embedding, e le distanze fra le sue previsioni sono una proiezione delle
distanze fra gli embedding, che ne conserva in parte l'ordine (prova: segnale di rango 8, sonda vera rho 0.99,
sonda permutata rho 0.78). K1 quindi **non** e' un controllo di fuga: si riporta come riferimento "lettura lineare
senza la supervisione giusta" (la sonda vera deve stare sopra, nessuna soglia). L'assenza di fuga dei soggetti di test
si garantisce per costruzione (soggetti di training e di test disgiunti: verificato nel codice a ogni fold) e si prova
sui dati finti (ingresso di solo rumore: rho della sonda vicino a 0, prova di `test_diag.py`).
