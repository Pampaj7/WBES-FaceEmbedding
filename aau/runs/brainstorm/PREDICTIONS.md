# Brainstorm H1/H3: predizioni fissate prima dei numeri (4 ottobre 2026)

Scritte prima che `h1_size.py` e `h3_table.py` girassero; i due script le copiano in testa al
proprio output.  Le soglie in corsivo sono precisazioni operative dell'esecutore dove la
richiesta non dava un numero.

- **H1 confermata** se Spearman(D_GT, |Δlog taglia|) > 0.3 e il residuo latent correla con
  |Δlog taglia|.  *Operativamente: per almeno una delle tre misure di taglia (divisore maxabs,
  radice dell'area, raggio rms) Spearman(D_GT, |Δlog taglia|) > 0.3 sulle 4950 coppie
  held-out, e sulla stessa misura Spearman(rango D_GT − rango latent, |Δlog taglia|) > 0.1 su
  original→original.  Le coppie cross-topologia sono riportate a parte.*
- **H3 confermata** se (b) robusto riduce rispetto ad (a) cotangente sia la dispersione degli
  autovalori sia l'errore HKS di almeno il 20% su down8k e noisy.  *Operativamente: sulle
  coppie original–down8k e original–noisy, riduzione relativa (a−b)/a ≥ 0.20 per entrambe le
  misure e per entrambe le coppie.*
