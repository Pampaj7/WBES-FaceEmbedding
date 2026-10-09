# factorized_protocol.md, emendamento 3 (9 ottobre 2026, 16:12, PRIMA di qualunque numero dei run del protocollo)

Protocollo ed emendamenti 1-2 invariati. Nessun run del protocollo ha prodotto righe d'eval finora (C3F non partiti,
C3M senza eval online). Su decisione del PI si aggiunge un braccio:

**`factorized2`** (C3F, s1234 e s2345, 1 A100 ciascuno; stessa riga di `factorized`: GT-SR x kappa per u, log S_i di
E12 per s con BFM a peso 0, arearobust + bal, ingresso globale, scala 0.8-1.25, dropout 0, 21.096 passi, eval a
10.548 e 21.096) con la testa `--head factorized2` (`factorized_v3.FactorizedNormEncoderV3`): dall'ingresso globale
X, c e R = baricentro e centroid size pesati per area robusta (pesi `smooth`), Xn = (X - c) / R;
u = f(Xn) invariante per costruzione, s = log(R x L0) + delta(Xn) equivariante per costruzione (delta, ultimo strato a
zero, corregge il supporto). Motivo: nel run breve (`factorized.md`, sez. 5) l'invarianza di u di `factorized` era
solo parziale. Controllo numerico senza training (`tests/test_factorized.py`, `factorized/units_f2.json`): a in
{0.8, 1, 1.25} con traslazione, |s(aX + t) - s(X) - log a| <= 5.0e-7 e |u(aX + t) - u(X)| <= 6.7e-8.

Valutazione identica agli altri bracci (FR con la distanza derivata, SR, maxabs, dev FaceScape, FaceVerse, FaMoS
TEST, NoW con u). Regola: la stessa del protocollo, per factorized2 come per factorized; in piu' si riporta
factorized2 - factorized (stesso seme), descrittivo.
