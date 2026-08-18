#!/usr/bin/env bash
# Each ablation arm is queued on two congested queues at once (gpua10 estimated a Friday start;
# gpuv100 has worse queue depth but 136 concurrent slots). Whichever dispatches first wins and
# its twin is killed, so we pay for one run per arm.
#
# This is not the same thing as the duplicate pot_well submissions killed earlier: those were
# copies of a job that was ALREADY RUNNING, i.e. pure waste. Here nothing is running and the
# duplication buys latency against schedulers we do not control -- provided the loser dies,
# which is what this does. Two copies of one arm would also leave two run dirs and make
# checkpoint selection ambiguous.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
ARMS="abl_intrinsic abl_xyzhks abl_specmlp"

while true; do
  alive=0
  for arm in $ARMS; do
    # exact-name match: "abl_intrinsic" must not also match "abl_intrinsic_v"
    running=$(bjobs -w 2>/dev/null | awk -v a="$arm" -v b="${arm}_v" \
              '$3=="RUN" && ($7==a || $7==b) {print $1}' | head -1)
    if [ -n "$running" ]; then
      for j in $(bjobs -w 2>/dev/null | awk -v a="$arm" -v b="${arm}_v" \
                 '($7==a || $7==b) && $3=="PEND" {print $1}'); do
        bkill "$j" >/dev/null 2>&1 && echo "[$(date +%T)] $arm: gemello $j ucciso (vince $running)"
      done
    fi
    if bjobs -w 2>/dev/null | awk -v a="$arm" -v b="${arm}_v" '($7==a || $7==b)' | grep -q .; then
      alive=$((alive + 1))
    fi
  done
  [ "$alive" -eq 0 ] && { echo "[$(date +%T)] nessun job ablation residuo, esco"; break; }
  sleep 180
done
