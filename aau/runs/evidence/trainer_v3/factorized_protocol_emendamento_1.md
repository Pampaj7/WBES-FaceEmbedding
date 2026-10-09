# factorized_protocol.md, emendamento 1 (9 ottobre 2026, ~14:50, PRIMA di qualunque passo di training)

Protocollo originale invariato (sha256 `514f10e154c6add3aacf468cf22d94d5cb8678f7d5a31e21752a5f452e1acfa4`). Stato a
quest'ora: nessun run del protocollo ha fatto un passo (C3F in attesa dello store ricostruito; C3M 1062766 fermato
durante lo staging del primo blocco e rilanciato con questo emendamento). Avvertenze di E12 dopo `READY`:

1. **BFM fuori dalla loss di taglia.** Le original REMESH sono normalizzate per similarita' una per una (CV della
   taglia BFM 1.9%): le S_i di BFM non sono taglie vere. In `factorized` la MSE di s ha peso 0 sulle identita' BFM
   (`--size-mask-domains bfm`, C3F e C3M); u (loss di forma) e i dati sono invariati.
2. **Termine del centroide.** Con FR (rigida robusta per identita') vale d_FR^2 = ||c_i - c_j||^2 + (S_i - S_j)^2 +
   S_i S_j d_P^2: la distanza derivata d_F omette ||c_i - c_j|| (errore relativo mediana 0.5%, p95 7.5%, E12). Scelta
   (b): si valuta d_F cosi' com'e' contro GT-FR, approssimazione dichiarata. Motivo: un ramo per c sotto
   l'augmentation di scala non ha un bersaglio definito (c dipende dalla rigida verso mu, che non scala con a);
   un braccio con il ramo c e' una proposta separata, non questo protocollo.
3. `--gt-keep-scale` in tutti i run (FR tarata ha massimo 1.30, SR tarata 1.02; SR grezza 1.0): gia' cosi'.
4. Ingresso globale verificato: area(X) x L0^2 = area_mm2 (err rel <= 1.3e-8) e X = u_d R_d V_grezza / L0 entro
   1.5e-5 mm (tests/test_factorized.py); log dei run: "ingresso globale ... L0 100 mm".

Metrica primaria invariata: Spearman della distanza form derivata (d_F) contro GT-FR di valutazione, HIFI3D
`nocrop_cross`, regola come nel protocollo.
