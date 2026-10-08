# E10: operatori DiffusionNet su GPU

Stessa convenzione di `aau/data_scale/prepass_ops.py::ops_areanorm` (mesh centrata ad area 1, `compute_operators`
di diffusion-net), calcolata su GPU. Risultati: `aau/runs/evidence/e10/` (`tables.md` generato, `E10.md` note e
verdetto). Nessuna modifica a `diffusion-net/`, `v2_work/`, `face_embedding/` ne' a E9: `e9_bench` e' importato.

| file | cosa fa |
| --- | --- |
| `gpu_ops.py` | geometria su GPU (Laplaciano cotangente, massa, normali, frame, gradienti: scatter-add, stessi dtype di `compute_operators`) e i risolutori degli autovettori: `dense` (eigh), `lobpcg` (torch), `clobpcg` (cupyx), `chfsi` (Chebyshev senza shift), `si` (shift-invert cuDSS + Chebyshev), `bk` (shift-invert cuDSS + Krylov a blocchi, la via scelta); `compute_batch` restituisce i dict del npz di `ops_areanorm` |
| `check_ops.py` | 60 mesh BFM/ICT x 6 etichette: operatori, autovalori, angoli fra sottospazi, embedding e108 contro la CPU e contro il rumore della CPU stessa (`cpu_v0`) |
| `bench.py` | `classes` (mesh/s, latenza, memoria per V ~3.3k/9.4k/24k/60k, k 64/128), `methods` (le vie (a)/(b)/Chebyshev), `e2e` (pipeline completa sui campioni di E9) |
| `e10.sbatch` | check + bench su una GPU (default V100; `--gres` al lancio per le altre) |
| `summarize.py` | `aau/runs/evidence/e10/tables.md` dai JSON, col confronto E9 |

## Ambiente: `.venv_e10` (cuDSS)

cuDSS non e' nel container. Venv dentro il container, con i pacchetti di `.venv_aau` via `.pth`, e la sola wheel di
cuDSS (`--no-deps`: le dipendenze di pip porterebbero numpy 2 e cuda-bindings 13, incompatibili col container):

    srun ... singularity exec --nv /home/container/pytorch/pytorch_24.10.sif bash -c '
      python3 -m venv --system-site-packages --without-pip .venv_e10
      echo $PWD/.venv_aau/lib/python3.10/site-packages > .venv_e10/lib/python3.10/site-packages/aau_base.pth
      source .venv_e10/bin/activate
      python3 -m pip install --no-deps nvidia-cudss-cu12'      # 0.8.0.10, 146 MB

`gpu_ops.CuDSS` carica `nvidia/cu12/lib/libcudss.so.0` con ctypes (nessun binding Python). Si lancia con
`VENV=.venv_e10 aau/run.sh ...` (lo fa `e10.sbatch`).

## Lancio

    sbatch aau/evidence/e10_gpu_ops/e10.sbatch                                        # V100: check k 128 e 64 + bench
    E10_STEPS="bench check" E10_CHECK_K=128 E10_CHECK_ARGS="--variants bk" \
        sbatch --gres=gpu:l40s:1 aau/evidence/e10_gpu_ops/e10.sbatch                  # L40S
    python3 aau/evidence/e10_gpu_ops/summarize.py                                     # sul frontend
