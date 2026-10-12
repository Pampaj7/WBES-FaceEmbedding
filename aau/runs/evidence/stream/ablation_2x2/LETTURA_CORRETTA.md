# Ablazione 2x2 dello stream: lettura corretta (POST HOC)

Scritta il 12 ottobre 2026 dal coder, su decisione del PI, DOPO i numeri e dopo il critic sui risultati (RISERVE: il
calcolo e l'applicazione del protocollo sono corretti, le letture no). **Post hoc: niente qui e' preregistrato.** I
verdetti preregistrati restano quelli di `em1/results.md` (emendamento 1, codice congelato al commit 7a398c9) e, con le
regole originali, di `results.md`; questo file dice come vanno letti. Analisi del critic in
`/home/create.aau.dk/ga41wf/critic_scratch_657e6a75/abl/` (`vs_c3f.py` e `vs_c3f.out`: stream contro C3F sulle stesse
righe; `abar.py` e `abar.out`: contrasti contro la media dei due semi di A, Ā = (A + A') / 2; sha256 di `vs_c3f.py`
9de241b6..., di `abar.py` 278f1116...). I numeri citati sotto vengono da quelle uscite e da `em1/`.

## Verdetti preregistrati (emendamento 1, `em1/`)

- Controllo FLAME: B - A su `flame2023_s1` SR +0.098 [+0.078, +0.120], D - C +0.095, B' - A +0.111: FLAME e' imparato,
  R_F e R_F' sono interpretabili.
- **R_F: NO**, nessun danno. **R_F': NO**, danno confermato su FaceVerse (FR -0.108 [-0.167, -0.048], SR -0.081
  [-0.154, -0.013]).
- **R_P1: non confermato** (beneficio solo in HIFI3D FR, +0.173 [+0.149, +0.202]). **R_P2: non costa**. **R_I:** nessuna.
- Con le regole originali (`results.md`): R_F PARZIALE, R_P1 PARZIALE, R_P2 non costa, interazione su FaceScape SR.

## Correzioni del critic

1. **"Lo stream batte C3F" e' falso in generale.** Media di A e A' meno media dei due semi di C3F, sulle stesse righe
   (`vs_c3f.out`, d_F calibrata con FR, d_P con SR):

   | righe | GT | stream - C3F |
   |---|---|---|
   | HIFI3D nocrop | FR | +0.045 [+0.018, +0.074] |
   | HIFI3D nocrop | SR | +0.040 [-0.005, +0.085] |
   | FaceScape nocrop | SR | -0.063 [-0.093, -0.030] |
   | FaceScape nocrop | FR | -0.028 [-0.072, +0.016] |
   | FaceScape righe col crop | FR | +0.244 [+0.186, +0.295] |
   | HIFI3D righe col crop | FR | -0.117 [-0.156, -0.084] |
   | FaceVerse nocrop | FR, SR | circa 0 (+0.004, +0.016, IC attraverso 0) |

   Il confronto non e' pulito: budget circa 5.7 volte, 2 GPU contro 1, dati freschi contro identita' fisse.
   **Lettura: misto.**
2. **R_P1: il +0.173 su HIFI3D col crop viene dalla taglia S.** Con la sola taglia C - A = +0.241 [+0.204, +0.281]; con
   la sola forma (d_P, SR) +0.060 [+0.020, +0.105] contro A e +0.089 [+0.040, +0.138] contro A'. **Lettura: la parzialita'
   rende robusta al crop la stima della taglia; sulla forma l'effetto e' piccolo.**
3. **"FaceScape va in senso opposto" e' un artefatto di A**, che sulle righe col crop vale 0.639 (FR) contro 0.478 di
   A'. Contro Ā: C - Ā FR +0.040 [+0.002, +0.075], SR +0.053 [+0.018, +0.089] (`abar.out`). **Lettura: inconclusivo,
   debolmente positivo.**
4. **A e' un controllo anomalo su FaceScape nocrop SR.** B, C, D e B' stanno fra 0.771 e 0.783; A vale 0.708 e A' 0.666.
   Ogni modifica batte entrambi i semi di A. L'interazione I = -0.068 [-0.101, -0.038] dice che gli effetti non si
   sommano. **R_F va letto come "FLAME non da' un guadagno specifico e coerente fra le famiglie"**, non come "FLAME non
   trasferisce" (che il controllo FLAME smentisce).
5. **R_F' su FaceVerse.** Il danno c'e' contro entrambi i semi di A, ma con un solo seme di B' e con le due "celle" sulla
   stessa famiglia. Con sigma_run stimato su tutte e 6 le celle (0.029, punto 7) la FR e' al limite e la SR non passa.
   **Lettura: danno possibile su FaceVerse, non confermato.**
6. **Costo della parzialita' in distribuzione:** piccolo ma reale su ict e gnm (C - A e D - B da -0.013 a -0.031, IC
   sotto 0); su bfm (C - A -0.071 / -0.065) circa meta' e' rumore (A' - A gia' -0.045 / -0.051). La parzialita' alza c
   di circa il 30% (C 0.438 e D 0.460 contro A 0.335 e B 0.357).
7. **Rumore fra semi dello stream:** dalle 6 differenze A' - A senza crop, sigma_run circa 0.029, circa 2.6 volte quello
   di C3F (0.011) su cui erano tarate le soglie della tabella. Ipotesi NON verificata: con lr costante l'iterato finale
   oscilla, e il consumo dello stream non e' deterministico. Prova in corso: valutazione di `epoch090_ema` di A e A'
   (sotto).

## Rumore all'epoca 90 contro 100 (punto 7)

Esito in `epoch_noise/` (aggiunto quando i job finiscono).
