# Ablazione k_eig, emendamento 1 (10 ottobre 2026): solo k64 contro k128

Decisione dell'utente, riferita dal PI il 10 ottobre alle 18:20 circa: **k256 non si prova**. Questo emendamento
sostituisce la regola di `PROTOCOL.md` (sha256 8b775a93..., commit 932184b), che resta invariato.

**Non e' una preregistrazione.** Quando l'ho scritto avevo GIA' calcolato e letto i numeri FR/SR/maxabs di k64 e
k128 (`results.md` generato alle 18:10 con la regola a scala di `PROTOCOL.md`, che dava k* = 64). La regola qui sotto
e' quella indicata dal PI, non scelta da me dopo i numeri; va comunque letta come regola dichiarata a posteriori.

## Cosa cambia

- Bracci: solo robal_k64 (training 1065295) e robal_k128 (training 1065293). Niente k256: niente store, training o eval.
- Valutazione: invariata (righe, GT FR, SR e maxabs, maschera, 1.000 repliche bootstrap per soggetto e semi di
  `fact_paired.py`; checkpoint e072 primario, e036 descrittivo). Delta appaiato: k64 - k128.
- **Regola:** **k128 resta la scelta di base.** k64 si adotta solo se **in tutte** le nove celle (HIFI3D, dev
  FaceScape, FaceVerse x FR, SR, maxabs) il delta k64 - k128 e' **>= -0.03** (stima puntuale). Il motivo per
  adottarlo sarebbe la velocita' (s/passo), che si riporta ma non entra nel criterio.
- Assunzione mia: il margine si applica alla stima puntuale, come il criterio "senza perdere piu' di 0.03" di
  `PROTOCOL.md`; l'estremo inferiore dell'IC si riporta accanto. Un solo seme (1234): vale quanto detto in
  `PROTOCOL.md`.
