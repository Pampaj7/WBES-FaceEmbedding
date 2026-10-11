#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl2x2-em1
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Ablazione 2x2, letture dell'emendamento 1 (aau/runs/evidence/stream/ablation_2x2/PROTOCOL_emendamento_1.md):
# v3_work/stream/eval_ablation_em1.py, uscite in ablation_2x2/em1. Prima, fuori dal container, il codice d'analisi
# congelato (sez. 7): per ogni file di ablation_2x2/analysis_freeze.json lo sha256 del file e quello del blob al commit
# fissato devono coincidere col valore congelato, altrimenti uscita 1; commit e stato git in em1/code_state.json.
#   provvisorio: sbatch --dependency=afterok:<valutazioni A-D>:<FLAME di B e D> v3_work/stream/slurm/ablation_2x2_summary_em1.sbatch
#   definitivo:  sbatch --dependency=afterok:<valutazioni A' e B'> ... (stessa riga: le celle pronte entrano da sole)
# Argomenti in piu' passati a eval_ablation_em1.py (prova: --root <dir> --epoch <N> --n-boot 50 --out <dir> --no-freeze).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
W="${SLURM_CPUS_PER_TASK:-8}"
ABL="$AAU_RUNS/evidence/stream/ablation_2x2"
STATE="${ABL_EM1_STATE:-$ABL/em1/code_state.json}"
mkdir -p "$(dirname "$STATE")"
echo "[abl-em1] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date +%T) argomenti: $*"
if [[ "${ABL_EM1_NOFREEZE:-0}" == 1 ]]; then       # solo prove (con --no-freeze): uscite marcate "codice non congelato"
  echo '{"pass": false, "note": "PROVA: ABL_EM1_NOFREEZE=1, nessun controllo del codice"}' > "$STATE"
  exec env UGT_THREADS=1 v3_work/unified_gt/run.sh v3_work/stream/eval_ablation_em1.py --workers "$W" --code-state "$STATE" "$@"
fi
python3 - "$ABL/analysis_freeze.json" "$STATE" "$WBES_ROOT" <<'PY'
import hashlib, json, subprocess, sys
fz_path, state, root = sys.argv[1:4]
fz = json.load(open(fz_path))
git = lambda *a: subprocess.run(["git", "--no-optional-locks", "-C", root, *a], capture_output=True, check=True).stdout
rows, bad = {}, []
for rel, want in fz["sha256"].items():
    work = hashlib.sha256(open(f"{root}/{rel}", "rb").read()).hexdigest()
    blob = hashlib.sha256(git("show", f"{fz['commit']}:{rel}")).hexdigest()
    rows[rel] = {"frozen": want, "working_tree": work, "at_commit": blob, "equal": work == want == blob}
    if not rows[rel]["equal"]:
        bad.append(rel)
out = {"freeze_commit": fz["commit"], "head": git("rev-parse", "HEAD").decode().strip(),
       "status_porcelain": git("status", "--porcelain", "--untracked-files=no").decode().splitlines(),
       "files": rows, "pass": not bad}
json.dump(out, open(state, "w"), indent=1)
print(f"[abl-em1] codice congelato {fz['commit'][:12]}: {'uguale' if not bad else 'DIVERSO: ' + str(bad)} (HEAD {out['head'][:12]})")
sys.exit(1 if bad else 0)
PY
UGT_THREADS=1 v3_work/unified_gt/run.sh v3_work/stream/eval_ablation_em1.py --workers "$W" --code-state "$STATE" "$@"
