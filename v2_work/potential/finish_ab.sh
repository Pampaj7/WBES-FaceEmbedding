#!/bin/bash
# Closes the potential-well A/B without supervision: waits for both arms to stop, then
# evaluates each with the SAME model-selection rule and the SAME per-group protocol.
#
# Selection is `best_by_xtopo_mesh_clean.pth` for both arms. Fixing one rule for both is what
# makes the comparison fair; picking each arm's own best group score would manufacture the
# effect we are trying to test.
#
# Evaluation must be per-group (crop / noisy / resample), not pooled: the well is predicted to
# help where the boundary moves and to do nothing or slightly hurt where only the sampling
# changes, and a pooled score cancels exactly those two effects.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
EPOCHS=60
LAST=$(printf 'epoch%03d.pth' "$EPOCHS")
DIST=face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz

# Each arm is submitted to p1i AND to two public queues as backup, so an arm can leave several
# run dirs at different stages. Waiting for *all* pot_* jobs to end would block on a backup that
# is still pending hours after the real run finished, so the condition is instead: both arms
# have a run that reached the last epoch. Any leftover duplicates are killed at that point.

# Pick the run dir with the MOST training progress, not the most recent one: a backup that
# starts late would be newer while being barely trained, which would silently compare an
# undertrained arm against a finished one.
pick_run () {
  # Only a run that REACHED THE LAST EPOCH is eligible. Without this the picker happily
  # returned the best-so-far checkpoint of a run that was still training (or had been killed),
  # and scoring it produced a results file that looks exactly like a real one: it happened,
  # writing a pot_w55.json from a model at epoch 5/60 whose numbers (all=0.6929) would have
  # been read as "the well, placed correctly". An undertrained arm must be MISSING, not weak.
  local best="" bestn=-1 d n
  for d in v2_work/runs/${1}_*/*/checkpoints/; do
    [ -d "$d" ] || continue
    [ -f "${d}${LAST}" ] || continue
    n=$(ls "$d" 2>/dev/null | grep -c '^epoch')
    if [ "$n" -gt "$bestn" ]; then bestn=$n; best="$d"; fi
  done
  [ -n "$best" ] && echo "${best}best_by_xtopo_mesh_clean.pth"
}

while true; do
  np=$(ls v2_work/runs/pot_plain_*/*/checkpoints/$LAST 2>/dev/null | wc -l)
  nw=$(ls v2_work/runs/pot_w55_*/*/checkpoints/$LAST 2>/dev/null | wc -l)
  echo "[$(date +%T)] finished runs: plain=$np well=$nw (need 1 each)"
  [ "$np" -ge 1 ] && [ "$nw" -ge 1 ] && break
  # if every job died without reaching the last epoch, stop waiting forever
  # Must list EVERY arm. An earlier version checked only pot_plain|pot_well while the kill
  # loop below already covered all four: with plain finished and well finished, it concluded
  # "no arm alive", fell through, and then KILLED the running pot_w55 and the queued pot_m55 --
  # the two arms it existed to wait for. Updating one grep and not the other is exactly the
  # kind of half-applied change that produces a confident wrong result.
  if [ "$(bjobs -w 2>/dev/null | grep -cE 'pot_plain|pot_well|pot_w55|pot_m55')" -eq 0 ]; then
    echo "[$(date +%T)] no pot_* job alive and an arm never finished — evaluating what exists"
    break
  fi
  sleep 300
done

# Kill ONLY duplicates of arms that already have a finished run. The previous version killed
# every pot_* job, which three times over destroyed arms that had not run yet: pot_m55 was
# queued, never dispatched, and got bkill'd the moment two OTHER arms finished. A job for an
# arm with no epoch060 anywhere is not a duplicate -- it is the only copy.
for arm in plain well w55 m55; do
  [ "$(ls v2_work/runs/pot_${arm}_*/*/checkpoints/$LAST 2>/dev/null | wc -l)" -ge 1 ] || continue
  for j in $(bjobs -w 2>/dev/null | awk -v a="pot_${arm}" '$7==a {print $1}'); do
    bkill "$j" >/dev/null 2>&1 && echo "[$(date +%T)] pot_${arm}: duplicato $j rimosso (run già completa)"
  done
done
echo "[$(date +%T)] evaluating"

# Four arms separating three distinct claims: whether the well is there (plain vs w55),
# where it is placed (well vs w55), and what the embedding is pooled over (w55 vs m55).
# Each arm MUST be read with the operators it was trained on, and m55 additionally needs
# --masked-pooling: masking adds no parameters, so its checkpoint would load into the
# unmasked model without error and be scored as if it had never been masked.
for arm in plain well w55 m55; do
  ckpt=$(pick_run "pot_${arm}")
  if [ -z "$ckpt" ]; then echo "MISSING checkpoint for $arm — skipped"; continue; fi
  extra=""
  case "$arm" in
    plain) data=datasets/REMESH/npz_data_topo_500_withops ;;
    well)  data=v2_work/potential/bfm_withwell ;;
    w55)   data=v2_work/potential/bfm_well055 ;;
    m55)   data=v2_work/potential/bfm_well055; extra="--masked-pooling --roi-threshold 0.5" ;;
  esac
  echo "[$(date +%T)] eval $arm: $ckpt"
  .conda_env/bin/python v2_work/potential/eval_by_topology.py \
      --checkpoint "$ckpt" --data-dir "$data" --dist-npz "$DIST" \
      --tag "pot_${arm}" --use-eval-split --n-subjects 100 $extra \
      2>&1 | tee "v2_work/logs/eval_pot_${arm}.log"
done

echo "[$(date +%T)] tutti i bracci valutati"
.conda_env/bin/python - <<'PY'
import json, pathlib
rows = []
for arm, label in [("plain", "no well"), ("well", "well misplaced (per-mesh)"),
                   ("w55", "well alpha=0.55"), ("m55", "well 0.55 + masked pooling")]:
    f = pathlib.Path(f"v2_work/potential/results/pot_{arm}.json")
    if not f.exists():
        continue
    g = json.loads(f.read_text())["groups"]
    rows.append((label, g))
if rows:
    groups = ["crop", "noisy", "resample", "all"]
    print(f"\n{'arm':30s}" + "".join(f"{k:>11s}" for k in groups))
    for label, g in rows:
        print(f"{label:30s}" + "".join(
            f"{g[k]['spearman']:11.4f}" if k in g else f"{'-':>11s}" for k in groups))
PY
