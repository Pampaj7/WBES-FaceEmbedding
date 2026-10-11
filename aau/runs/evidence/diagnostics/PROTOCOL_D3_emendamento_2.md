# Diagnostica D3, emendamento 2 (POST HOC, DOPO i numeri): griglia di lambda estesa, perdita riscalata, testa congiunta

Scritto l'11 ottobre 2026, mattina, dal coder, su richiesta del PI, DOPO i risultati preregistrati di D3
(`results_d3.md`, commit f0c643b; protocollo 21afea8, emendamento 1 49fead6) e DOPO la revisione del critic (RISERVE,
`/home/create.aau.dk/ga41wf/critic_scratch_657e6a75/d3c/`, fuori dal repo). E' un'analisi di sensibilita' POST HOC:
**non cambia il verdetto preregistrato** (R_LOGO: NO in entrambi i semi), e ogni suo numero si legge come post hoc,
accanto ai preregistrati e mai al loro posto. L'impronta sha256 di questo file sta nel messaggio del commit che lo
introduce e in `PROTOCOL_D3_emendamento_2.sha256`.

## 1. Perche' (rilievi del critic, con i suoi numeri gia' visti)

1. **La griglia di lambda era troppo stretta.** La penalita' lambda ||W||^2 / alpha0^2 (`d3_head.py`) agisce su x non
   standardizzate: con lambda >= 1e-2 W va a zero, e dei 5 valori preregistrati {1e-4, ..., 1} solo il bordo 1e-4 lascia
   una correzione. Nel run preregistrato tutte le scelte con r > 0 stavano su quel bordo (riportato come "bordo"). Il
   critic, con lambda = 1e-5, trova r = 32 o 64 scelto in tutte le 8 impostazioni primarie, con guadagno di CV da
   +0.017 a +0.055. Quindi "la CV sceglie spesso r = 0", la curva piatta di s2345 e il "contributo di FaMoS" del run
   preregistrato sono in parte artefatti della griglia.
2. **La perdita usa una sola scala per sorgenti con alpha0 diversi.** alpha0 (scala di d_P sulla GT, per sorgente) va da
   0.355 (FaceScape) a 0.565 (FaceVerse) per s1234: con una sola alpha la perdita stress spende la correzione W a
   compensare le scale fra sorgenti, che lo Spearman dentro un dominio ignora. Il critic, con la GT di ogni sorgente
   divisa per il suo alpha0, vede il delta LOGO salire in 8 celle su 8 (configurazione fissa r = 32, lambda = 1e-5:
   s1234 FaceScape +0.050, HIFI3D +0.022, FaceVerse +0.027, FLAME +0.033; s2345 +0.031, +0.026, +0.008, +0.018).
   Catena completa del critic (CV ridotta: r in {8, 32, 64} x lambda in {1e-5, 1e-4, 1e-3}, 2 ripetizioni): media sui 4
   bersagli s1234 +0.033 [+0.016, +0.051], s2345 +0.022 [+0.001, +0.043]; s1234 per bersaglio FaceScape +0.049,
   HIFI3D +0.022, FaceVerse +0.027, FLAME +0.033; n_SR = 0 in entrambi i semi.
3. **"Una correzione lineare comune non esiste" e' falso.** Una testa congiunta su tutte le 8 sorgenti, compresi i pool
   non valutati dei bersagli (FLAME: soggetti 0-99 in training, 100-199 in test), migliora tutti e 4 i bersagli insieme
   (critic, s1234, perdita riscalata, r = 32, lambda = 1e-5: FaceScape +0.133 [+0.101, +0.173], HIFI3D +0.122 [+0.079,
   +0.164], FaceVerse +0.091 [+0.036, +0.140], FLAME +0.076 [+0.051, +0.107]). Esiste una correzione comune quando il
   generatore e' visto; e' l'estrapolazione a un generatore mai visto che resta piccola.
4. **La media sulle etichette era letta male.** Il guadagno viene dall'avere la STESSA discretizzazione ai due lati
   della riga (critic: FaceScape d_P remesh-remesh 0.800, up60k-up60k 0.802, contro 0.747 sulle righe incrociate),
   non dalla riduzione del rumore ne' dalla original.

## 2. Cosa si ricalcola (con la pipeline di D3, codice `aau/diagnostics/d3_e2.py`)

- **(i-e2) catena LOGO completa con le due correzioni**: le 9 impostazioni dell'emendamento 1 (4 bersagli primari x
  {tutte tranne T, senza FaMoS}, FaMoS come bersaglio), testa (i) con
  - griglia **lambda in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}**, r in {2, 4, 8, 16, 32, 64}, piu' r = 0 (43
    configurazioni);
  - **perdita riscalata per sorgente**: nella perdita stress la GT di ogni blocco e' divisa per l'alpha0 del blocco
    (scala ai minimi quadrati di d_P sulla GT, sulle sole coppie di fit del blocco: in CV sui soggetti di fit),
    come il critic (`G / Problem([b]).alpha0`); lo Spearman dentro un blocco non cambia;
  - tutto il resto come l'emendamento 1: CV per soggetto 5 fold x 3 ripetizioni con i fold per sorgente, scelta
    (massimo, entro 1e-3 r minore e lambda maggiore), rifit su tutte le sorgenti, c_h dalle sorgenti con la GT
    ORIGINALE.
- **(J) testa congiunta, descrittiva**: la stessa catena (i-e2) sulle 8 sorgenti insieme, FLAME limitato ai soggetti
  0-99 (i primi 100 per id, `id710000`-`id710099`); valutata su FaceScape, HIFI3D, FaceVerse neutra (i 100 soggetti
  valutati, mai fra le sorgenti: le sorgenti sono i pool non valutati) e su FLAME sulle righe fra i soggetti 100-199,
  con bootstrap per soggetto sui soli 100 di test (`diag.boot_counts(100, 0, seed=20261124)`); FaceVerse con
  espressioni secondaria. NON e' LOGO: misura se una correzione comune esiste quando il generatore e' visto.
- **Accanto**: la curva sul numero di sorgenti con (i-e2) (stessa definizione dell'emendamento 1, iperparametri di
  "tutte tranne T" della catena e2); il contributo di FaMoS ((i-e2) - (i-e2) senza FaMoS); lo Spearman di d_P sulle
  righe con la STESSA etichetta ai due lati (original, remesh, down8k, noisy, up60k: una riga per coppia di soggetti di
  test) accanto alle incrociate e alle mediate, per il punto 4.
- **Controlli**: le teste (i) preregistrate rifittate con le loro configurazioni (`d3_heads.csv`) riproducono i delta
  preregistrati (punto e IC); alla configurazione fissa del critic (r = 32, lambda = 1e-5, perdita riscalata) i delta
  puntuali LOGO e congiunti si confrontano coi suoi (`exp2.json`); se non coincidono si dice perche'. (ii) e CORAL non
  hanno lambda ne' perdita: non si ricalcolano.
- Stesse righe, GT, semi e bootstrap dell'emendamento 1; FaMoS TEST e Ava-256 esclusi.

## 3. Letture (post hoc, descrittive)

- Il verdetto preregistrato resta **R_LOGO: NO**. Alla catena (i-e2) si applica la stessa regola (n_SR >= 3 bersagli
  con delta SR >= +0.05 e IC > 0, FR >= -0.03 su tutti e 4) solo come descrizione, etichettata post hoc.
- Si riportano: delta per bersaglio con IC, media sui 4 bersagli con IC, delta (i-e2) - (i) preregistrata, testa
  congiunta per bersaglio con IC, curva, contributo di FaMoS, d_P con la stessa etichetta ai due lati.
- Lettura attesa dal critic, da confermare o smentire coi numeri: l'estrapolazione LOGO e' piccola (+0.02/+0.03, sotto
  la soglia); una correzione comune esiste se il generatore e' visto.
