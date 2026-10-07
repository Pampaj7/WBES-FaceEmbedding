# Besnier et al. 2023 e Ma et al. 2021 (le baseline geometriche di Lee et al. 2025): non integrate

Verificato il 2026-10-07. Regola del protocollo: integrare solo se codice e pesi pubblici girano in meno di
mezza giornata di lavoro.

**Besnier, Arguillère, Pierson, Daoudi, "Toward mesh-invariant 3D generative deep learning with geometric
measures", Computers & Graphics 115 (2023), arXiv:2306.15762.**
- Codice: nessun link nel paper (testo dell'arXiv controllato: c'e' solo "the Python code is built over the one
  from [3]"). Nessun repository nell'account GitHub dell'autore (`tbesnier`: ScanTalk, ScanMove, STM, PaNDaS,
  bm-shapes, deep_deformer, robustAE). `robustAE` (MIT) contiene solo README e LICENSE, un "Initial commit" del
  19-11-2025; `deep_deformer` non ha README e non e' dichiarato come codice del paper. Ricerca GitHub per titolo:
  0 risultati.
- Pesi: nessuno pubblicato. Il modello e' un autoencoder addestrato su COMA (topologia FLAME): rifarlo vuol dire
  reimplementare il paper e ottenere COMA (licenza MPI, registrazione dell'utente), cioe' ben oltre mezza giornata,
  e il risultato sarebbe la nostra reimplementazione, non la loro baseline.
- Cosa abbiamo gia': la componente di misura del loro metodo e' la distanza kernel fra varifold/currents, che e'
  gia' in tabella come baseline (`varifold`, `currents`, WS1, `aau/runs/baselines/ranking/table2_extended_heldout.csv`).

**Ma, Liang, Liang, Wu, "3D facial similarity measure based on deformation field", IEEE RCAR 2021, pp. 364-369.**
- Codice: nessun repository trovato (ricerca GitHub per titolo e parole chiave: 0 risultati), nessun link noto.
- Paper IEEE dietro paywall; il metodo (campo di deformazione fra mesh in corrispondenza) presuppone una
  registrazione, cioe' la famiglia gia' coperta da ICP rigido + NICP (P2P, P2Tri) nelle nostre tabelle.
- Reimplementarlo dal solo paper richiede piu' di mezza giornata e non darebbe la loro baseline.

**Conclusione:** nessuna delle due entra. Nel rebuttal: "nessuna delle due ha codice o pesi pubblici; le loro
famiglie (misure geometriche varifold/currents, registrazione + deformazione) sono rappresentate da varifold,
currents, ICP e NICP".
