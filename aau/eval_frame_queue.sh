#!/usr/bin/env bash
# Sottomette le eval per gruppo di topologia del confronto frame rms contro maxabs, e i job che
# ne fanno la tabella appaiata per seed. Si lancia sul frontend, e si puo' rilanciare:
#
#   aau/eval_frame_queue.sh            # sottomette
#   aau/eval_frame_queue.sh --dry-run  # stampa i comandi senza sottometterli
#
# Bracci: i training da aau/scratch/frame_training_jobs.txt (vedi sotto). Il v1 pubblicato si
# valuta sempre: e' il controllo maxabs del seed 1234 solo se il file non ne ha uno suo, altrimenti
# e' una riga di riferimento separata nella tabella, fuori dal Delta appaiato.
#
# Ogni eval di un training non ancora finito parte con --dependency=afterok:<job training>,
# quindi solo se il training esce 0, e con --kill-on-invalid-dep=yes: se il training muore
# l'eval viene cancellata invece di restare in coda per sempre, e la tabella che la aspetta
# (afterany) parte lo stesso e dice cosa manca. Un training gia' COMPLETED non mette
# dipendenze; un training "final" in qualunque altro stato blocca tutto prima di sottomettere.
# Un'eval con .done gia' scritta non si risottomette, una gia' in coda si riusa (stesso nome).
#
# Training interrotti: una riga con quinta colonna "partial" e' un training cancellato prima
# della fine, valutato sul suo best_*.pth di allora. Non aspetta niente, e l'eval porta in
# eval_key.txt e in .done una riga partial= con epoca raggiunta, epoche previste ed epoca del
# best. Con righe partial le tabelle sono due:
#   _paired_interim/  ogni braccio dalla riga partial se c'e', altrimenti dalla riga final;
#                     parte appena finiscono le eval che servono;
#   _paired/          solo righe final, cioe' budget pieni; parte dopo le eval dei rilanci.
# Senza righe partial c'e' solo _paired/. Il budget di ogni braccio e' una colonna della tabella.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$WBES_ROOT"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# Esplicito e non $WBES_CKPT: una WBES_CKPT esportata nella shell non deve cambiare il controllo.
V1_CKPT="$WBES_ROOT/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth"

# I training si leggono da un file, una riga per training: "seed frame jobid rundir [stato]",
# con frame rms|maxabs, jobid "-" se non c'e' un job da aspettare, rundir = il runs_root sotto
# aau/runs (remesh_v1recipe_rms_s2345_1054482), stato final (default) | partial. Per ogni seed e
# frame al piu' una riga final e una partial. Righe vuote e # ignorate. Il v1 si aggiunge qui,
# non nel file: ruolo "pair" se manca un "1234 maxabs" final, "ref" se c'e'.
JOBS_FILE="${WBES_FRAME_JOBS:-$AAU_DIR/scratch/frame_training_jobs.txt}"
if [[ ! -f "$JOBS_FILE" ]]; then
    echo "ERRORE: manca $JOBS_FILE (formato: seed frame jobid rundir [final|partial])" >&2
    exit 2
fi
# seed|frame|root|jid|stato|ruolo
ENTRIES=()
declare -A SEEN=()
while read -r seed frame jid root status _; do
    [[ -z "${seed:-}" || "$seed" == \#* ]] && continue
    case "$frame" in rms|maxabs) ;; *) echo "ERRORE: frame '$frame' in $JOBS_FILE" >&2; exit 2 ;; esac
    status="${status:-final}"
    case "$status" in final|partial) ;; *) echo "ERRORE: stato '$status' in $JOBS_FILE" >&2; exit 2 ;; esac
    if [[ -n "${SEEN[$seed-$frame-$status]:-}" ]]; then
        echo "ERRORE: due righe '$seed $frame $status' in $JOBS_FILE, quale sia il braccio appaiato e' ambiguo" >&2
        exit 2
    fi
    SEEN[$seed-$frame-$status]=1
    [[ "$jid" == "-" ]] && jid=""
    ENTRIES+=("$seed|$frame|$(basename "$root")|$jid|$status|pair")
done < "$JOBS_FILE"
if [[ -n "${SEEN[1234-maxabs-final]:-}" ]]; then
    ENTRIES+=("1234|maxabs|v1||final|ref")
else
    ENTRIES+=("1234|maxabs|v1||final|pair")
fi
HAS_PARTIAL=0
printf '%s\n' "${ENTRIES[@]}" | cut -d'|' -f5 | grep -qx partial && HAS_PARTIAL=1

# Stessa regola di nome di eval_frame_topology.sbatch: il runs_root per i training sotto
# aau/runs, la dir che contiene checkpoints/ per il v1.
out_dir_of() {
    local root="$1" frame="$2"
    if [[ "$root" == "v1" ]]; then
        echo "$AAU_RUNS/eval_frame/mixed_xtopo_rank0p5_id0p25_bs5_best_${frame}"
    else
        echo "$AAU_RUNS/eval_frame/${root}_${frame}"
    fi
}

# Il WBES_CKPT da passare allo sbatch. Con la run dir gia' creata, il suo best_*.pth (che puo'
# non esserci ancora: lo scrive la prima eval online, a epoca 2; lo verifica lo sbatch quando
# parte). Con un training ancora in coda la run dir col fingerprint non esiste: si passa il
# runs_root e lo sbatch lo risolve all'avvio.
ckpt_of() {
    local root="$1" jid="$2"
    if [[ "$root" == "v1" ]]; then
        echo "$V1_CKPT"
        return
    fi
    local runs=("$AAU_RUNS/$root"/*/config.json)
    if [[ ! -f "${runs[0]}" && -n "$jid" ]]; then
        echo "$AAU_RUNS/$root"
        return
    fi
    if [[ ${#runs[@]} -ne 1 || ! -f "${runs[0]}" ]]; then
        echo "ERRORE: attesa una sola run dir con config.json in $AAU_RUNS/$root, trovate: ${runs[*]}" >&2
        return 1
    fi
    local ckpt
    ckpt="$(dirname "${runs[0]}")/checkpoints/best_by_xtopo_mesh_clean.pth"
    if [[ -z "$jid" && ! -f "$ckpt" ]]; then
        echo "ERRORE: $ckpt non esiste e non c'e' un training da aspettare" >&2
        return 1
    fi
    echo "$ckpt"
}

# Il seed dichiarato qui e' quello su cui la tabella appaia i bracci: deve essere quello del
# training, letto dal config.json della run dir (per il v1 sta accanto a checkpoints/), o dal
# --seed di wrapper_launch.txt se la run dir non c'e' ancora. Se non c'e' nemmeno quello lo
# controlla lo sbatch all'avvio (WBES_EXPECT_SEED, passata sempre).
check_seed() {
    local ckpt="$1" seed="$2" root="$3" cfg got=""
    if [[ -d "$ckpt" ]]; then
        if [[ -f "$AAU_RUNS/$root/wrapper_launch.txt" ]]; then
            got="$(grep -oE -- '--seed [0-9]+' "$AAU_RUNS/$root/wrapper_launch.txt" | awk '{print $2}' | tail -1 || true)"
        fi
        if [[ -z "$got" ]]; then
            echo "[seed] s$seed $root: run dir assente, seed verificato dallo sbatch all'avvio"
            return 0
        fi
        cfg="$AAU_RUNS/$root/wrapper_launch.txt"
    else
        cfg="$(dirname "$(dirname "$ckpt")")/config.json"
        got="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("args", d).get("seed"))' "$cfg")"
    fi
    if [[ "$got" != "$seed" ]]; then
        echo "ERRORE: $cfg ha seed=$got, dichiarato $seed" >&2
        return 1
    fi
}

# "epoca 108/120, best epoca 106": ultima riga di train_log.csv, epochs di config.json,
# best_epoch di best_by_xtopo_mesh_clean.txt (accanto a checkpoints/).
budget_of() {
    local run_dir="$1"
    python3 - "$run_dir" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
last = (d / "train_log.csv").read_text().strip().splitlines()[-1].split(",")[0]
cfg = json.loads((d / "config.json").read_text())
planned = cfg.get("args", cfg)["epochs"]
best = dict(l.split("=", 1) for l in (d / "best_by_xtopo_mesh_clean.txt").read_text().split())
print(f"epoca {last}/{planned}, best epoca {best['best_epoch']}")
PY
}

dependency_of() {
    local jid="$1" state
    [[ -z "$jid" ]] && return 0
    state="$(sacct -j "$jid" -X -n -o State | head -1 | tr -d ' ')"
    case "$state" in
        PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING) echo "--dependency=afterok:$jid" ;;
        COMPLETED) ;;
        *) echo "ERRORE: training $jid in stato '${state:-sconosciuto}', non sottometto" >&2; return 1 ;;
    esac
}

# Prima tutti i controlli, poi le sottomissioni: un errore a meta' non lascia mezza coda.
declare -a PLAN=()
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r seed frame root jid status _ <<< "$entry"
    partial=""
    if [[ "$status" == "partial" ]]; then
        # Un training interrotto non si aspetta: si valuta il best_*.pth che ha lasciato.
        ckpt="$(ckpt_of "$root" "")"
        partial="$(budget_of "$(dirname "$(dirname "$ckpt")")")${jid:+, training $jid interrotto}"
        dep=""
    else
        ckpt="$(ckpt_of "$root" "$jid")"
        dep="$(dependency_of "$jid")"
    fi
    check_seed "$ckpt" "$seed" "$root"
    PLAN+=("$seed|$frame|$root|$ckpt|$dep|$status|$partial")
done

# eval_job[<root>_<frame>] = job id da aspettare (vuoto se gia' completa)
declare -A EVAL_JOB=()
for item in "${PLAN[@]}"; do
    IFS='|' read -r seed frame root ckpt dep status partial <<< "$item"
    out="$(out_dir_of "$root" "$frame")"
    if [[ -f "$out/.done" ]]; then
        echo "[skip] s$seed $frame $status: gia' completa ($out/.done)"
        continue
    fi
    # Rilanciare la coda non deve duplicare le eval gia' in coda: si riusa il job con lo stesso
    # nome per la dipendenza delle tabelle. Il v1 ha un nome suo: con un controllo s1234 nel
    # file, "s1234-maxabs" e' quello; le parziali hanno il suffisso -partial.
    name="wbes-evalframe-s${seed}-${frame}"
    [[ "$status" == "partial" ]] && name="$name-partial"
    [[ "$root" == "v1" ]] && name="wbes-evalframe-v1-maxabs"
    queued="$(squeue -u "$USER" -h -n "$name" -o '%i' | head -1)"
    if [[ -n "$queued" ]]; then
        EVAL_JOB[${root}_$frame]="$queued"
        echo "[keep] s$seed $frame $status: gia' in coda come job $queued"
        continue
    fi
    cmd=("$AAU_DIR/submit.sh" eval_frame_topology.sbatch --parsable "--job-name=$name"
         --kill-on-invalid-dep=yes)
    [[ -n "$dep" ]] && cmd+=("$dep")
    if (( DRY )); then
        echo "[dry] WBES_CKPT=$ckpt WBES_FRAME=$frame WBES_EXPECT_SEED=$seed${partial:+ WBES_EVAL_PARTIAL='$partial'} ${cmd[*]}"
        EVAL_JOB[${root}_$frame]="<$name>"
        continue
    fi
    jobid="$(WBES_CKPT="$ckpt" WBES_FRAME="$frame" WBES_EXPECT_SEED="$seed" \
             WBES_EVAL_PARTIAL="$partial" "${cmd[@]}")"
    jobid="${jobid%%;*}"
    EVAL_JOB[${root}_$frame]="$jobid"
    echo "[eval] s$seed $frame $status -> job $jobid ${dep:-(nessuna dipendenza)}${partial:+ [PARZIALE: $partial]}  $out"
done

# submit_table <nome job> <out dir> <prefer>: prefer=partial prende per ogni braccio la riga
# partial se c'e', prefer=final solo le righe final. Appaia per seed i bracci con ruolo "pair";
# un seed con un solo braccio resta fuori dalla tabella, e lo si dice. Il v1 "ref" va a parte.
submit_table() {
    local job_name="$1" out_dir="$2" prefer="$3"
    local pair_args=() wait_jobs=() seed s f r st role rms_root max_root
    echo "[table] $job_name -> ${out_dir#$WBES_ROOT/}"
    for seed in $(printf '%s\n' "${ENTRIES[@]}" | cut -d'|' -f1 | sort -u); do
        rms_root="" max_root=""
        for f in rms maxabs; do
            local pick=""
            for entry in "${ENTRIES[@]}"; do
                IFS='|' read -r s ff r _ st role <<< "$entry"
                [[ "$s" != "$seed" || "$ff" != "$f" || "$role" != "pair" ]] && continue
                if [[ "$st" == "final" && -z "$pick" ]]; then pick="$r"; fi
                if [[ "$st" == "partial" && "$prefer" == "partial" ]]; then pick="$r"; fi
            done
            [[ "$f" == "rms" ]] && rms_root="$pick" || max_root="$pick"
        done
        if [[ -z "$rms_root" || -z "$max_root" ]]; then
            echo "[table]   seed $seed senza entrambi i bracci (rms='$rms_root' maxabs='$max_root'): fuori"
            continue
        fi
        pair_args+=(--pair "$seed" "$(out_dir_of "$rms_root" rms)" "$(out_dir_of "$max_root" maxabs)")
        for r in "${rms_root}_rms" "${max_root}_maxabs"; do
            [[ -n "${EVAL_JOB[$r]:-}" ]] && wait_jobs+=("${EVAL_JOB[$r]}")
        done
        echo "[table]   coppia s$seed: rms=$rms_root  maxabs=$max_root"
    done
    for entry in "${ENTRIES[@]}"; do
        IFS='|' read -r s f r _ st role <<< "$entry"
        [[ "$role" != "ref" ]] && continue
        pair_args+=(--reference "v1_s$s" "$(out_dir_of "$r" "$f")")
        [[ -n "${EVAL_JOB[${r}_$f]:-}" ]] && wait_jobs+=("${EVAL_JOB[${r}_$f]}")
        echo "[table]   riferimento (fuori dal Delta): v1 s$s $f"
    done
    if (( ${#pair_args[@]} == 0 )); then
        echo "ERRORE: nessun seed con entrambi i bracci, niente tabella $job_name" >&2
        return 1
    fi

    local table_cmd=(sbatch --parsable "--job-name=$job_name" --partition=cpu
                     --cpus-per-task=1 --mem=2G --time=00:10:00
                     "--chdir=$WBES_ROOT" "--output=$AAU_LOGS/%x-%j.out" "--error=$AAU_LOGS/%x-%j.err")
    if (( ${#wait_jobs[@]} )); then
        table_cmd+=("--dependency=afterany:$(IFS=:; echo "${wait_jobs[*]}")")
    fi
    # Il nodo cpu non ha GPU: AAU_NV vuota, come in setup_env.sbatch. %q: la riga passa da sh -c.
    local wrap
    wrap="AAU_NV= $(printf '%q ' "$AAU_DIR/run.sh" "$AAU_DIR/eval_frame_table.py" \
        --out-dir "$out_dir" "${pair_args[@]}")"
    table_cmd+=(--wrap "$wrap")

    if (( DRY )); then
        echo "[dry] ${table_cmd[*]}"
        return 0
    fi
    mkdir -p "$AAU_LOGS"
    # Una tabella di un lancio precedente ancora in attesa e' superata da questa.
    local old table_job
    for old in $(squeue -u "$USER" -h -n "$job_name" -t PENDING -o '%i'); do
        scancel "$old" && echo "[table]   cancellato il job tabella precedente $old (sostituito)"
    done
    table_job="$("${table_cmd[@]}")"
    echo "[table]   job ${table_job%%;*} dopo: ${wait_jobs[*]:-(nessuna eval da aspettare)}"
}

if (( HAS_PARTIAL )); then
    submit_table wbes-evalframe-table-interim "$AAU_RUNS/eval_frame/_paired_interim" partial
fi
submit_table wbes-evalframe-table "$AAU_RUNS/eval_frame/_paired" final
