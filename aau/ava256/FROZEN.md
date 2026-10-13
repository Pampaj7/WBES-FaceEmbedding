# Ava-256: congelamento della prova confermativa

- **Documento**: `aau/ava256/PROTOCOL_CONFERMATIVO_candidato.md`, v9, sha256 `8690afbe6c8d7b6530cc928ca9a78a79416e82e0c133b213ce6ed64096a2c7d2`
- **Commit del documento**: `067f9c50c00f1f7cfbfc60337478e4825fa082a3` (tag git `ava256-frozen`)
- **Codice congelato**: commit `8a6e184d2951d73df7ea55e48296c4df36b92aef` (`8a6e184`); sha256 dei file nella sez. 11.1
- **Congelato il**: 2026-10-13 19:11:14 CEST

## Esecuzione (sez. 7, Esecuzione; sez. 11.9)

Solo da un worktree al commit del codice congelato, con il commit nell'ambiente di ogni job:

```bash
cd ~/WBES-FaceEmbedding
aau/ava256/conf_worktree.sh 8a6e184d2951d73df7ea55e48296c4df36b92aef ~/wt_ava256_frozen
cd ~/wt_ava256_frozen
export CONF_FROZEN_COMMIT=8a6e184d2951d73df7ea55e48296c4df36b92aef
export CONF_SPEC=aau/ava256/spec/ava256.json
CONF_STEPS="<passi>" sbatch --chdir="$PWD" [vincoli di nodo e GPU della sez. 7] aau/ava256/conf.sbatch
```

Vincoli: rilevatore e render dei protocolli su `a256-t4-04`; fit B su `i256-a10-08`; ArcFace su `i256-a10-09`;
testa di R6 su `i256-a10-10`; embedding su A100 (`-p aicentre-a100 --qos=unprivileged --gres=gpu:a100:1`);
varifold e ULRSSM su L40S (`--gres=gpu:l40s:1`). Il primo passo di ogni job e' `conf_common.py verify` (sez. 11).
