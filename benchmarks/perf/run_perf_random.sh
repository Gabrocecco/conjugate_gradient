#!/usr/bin/env bash
# run_random_perf.sh   – build scalar+vector, profile cache-misses on vector only,
#                        merge results into a single CSV with computed speed-ups.
set -euo pipefail

# ───────────────────────────── paths ─────────────────────────────
proj_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$proj_root/build"
src_dir="$proj_root/src"
inc_dir="$proj_root/include"
data_dir="$proj_root/benchmarks/perf/data"
mkdir -p "$build_dir" "$data_dir"

# ─────────────────────── compile two profiles ────────────────────
common_sources=(
  "$src_dir/vectorized.c" "$src_dir/common.c" "$src_dir/coo.c"
  "$src_dir/ell.c" "$src_dir/csr.c" "$proj_root/benchmarks/perf/mv_random_perf.c"
)

gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" -DRUN_SCALAR \
    -o "$build_dir/mv_scalar"   "${common_sources[@]}"

gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" -DRUN_VECTOR \
    -o "$build_dir/mv_vector"   "${common_sources[@]}"

# ─────────────────────── experiment matrix ───────────────────────
sizes=(1024 2048 4096 8192 16384 32768)
sparsities=(0.01 0.02 0.05 0.10 0.20)

# ───────────────────── header of final CSV -----------------------
out="$data_dir/mv_random_perf_full.csv"
printf '%s\n' \
"n,sparsity,max_nnz_row,"\
"time_serial,time_vectorized,speedup_time,"\
"cycles_serial,cycles_vector,speedup_cycles,"\
"inst_serial,inst_vector,speedup_inst,"\
"ipc_serial,ipc_vector,"\
"L1-loads,L1-misses,LLC-loads,LLC-misses" \
> "$out"

# ───────────────────── helper: merge two rows --------------------
merge_line() {          # $1 = scalar CSV row,  $2 = vector CSV row,  $3-$6 = perf ctrs
  IFS=',' read -ra S <<<"$1"   # array S[0-13]   (scalar build)
  IFS=',' read -ra V <<<"$2"   # array V[0-13]   (vector build)

  # ---- common identifiers ---------------------------------------
  n=${S[0]}    ;  spars=${S[1]}  ;  max=${S[2]}

  # ---- scalar metrics -------------------------------------------
  tS=${S[3]}   ;  cS=${S[6]}    ;  iS=${S[9]}    ;  ipcS=${S[12]}

  # ---- vector metrics -------------------------------------------
  tV=${V[4]}   ;  cV=${V[7]}    ;  iV=${V[10]}   ;  ipcV=${V[13]}

  # ---- guard against empty or NaN strings -----------------------
  nz() { [[ $1 =~ ^([0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?)$ ]] && echo "$1" || echo 0; }

  tSnum=$(nz "$tS") ; tVnum=$(nz "$tV")
  cSnum=$(nz "$cS") ; cVnum=$(nz "$cV")
  iSnum=$(nz "$iS") ; iVnum=$(nz "$iV")

  # ---- speed-ups -------------------------------------------------
  spdT=$(awk "BEGIN{print ($tSnum>0&&$tVnum>0)?$tSnum/$tVnum:NaN}")
  spdC=$(awk "BEGIN{print ($cSnum>0&&$cVnum>0)?$cSnum/$cVnum:NaN}")
  spdI=$(awk "BEGIN{print ($iSnum>0&&$iVnum>0)?$iSnum/$iVnum:NaN}")

  # ---- emit merged CSV row --------------------------------------
  printf '%s,%s,%s,'  "$n" "$spars" "$max"
  printf '%.6f,%.6f,%.6f,' "$tS" "$tV" "$spdT"
  printf '%s,%s,%.6f,'     "$cS" "$cV" "$spdC"
  printf '%s,%s,%.6f,'     "$iS" "$iV" "$spdI"
  printf '%s,%s,'          "$ipcS" "$ipcV"
  printf '%s,%s,%s,%s\n'   "$3" "$4" "$5" "$6"
}

# ───────────────────── main loop ---------------------------------
for spars in "${sparsities[@]}"; do
  for n in "${sizes[@]}"; do
    echo "n=$n  sparsity=$spars"

    tmpS=$(mktemp)  # scalar CSV
    tmpV=$(mktemp)  # vector CSV
    tmpP=$(mktemp)  # perf stat

    # --- scalar run (no perf) ------------------------------------
    MV_RND_CSV="$tmpS" "$build_dir/mv_scalar" "$n" "$spars"

    # --- vector run under perf -----------------------------------
    MV_RND_CSV="$tmpV" \
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        -x, --output "$tmpP" -- "$build_dir/mv_vector" "$n" "$spars"

    # --- extract perf counters -----------------------------------
    read l1 l1m llc llcm <<<"$(awk -F',' '
        $3=="L1-dcache-loads"       {gsub(/[^0-9]/,"",$1); l1=$1}
        $3=="L1-dcache-load-misses" {gsub(/[^0-9]/,"",$1); l1m=$1}
        $3=="LLC-loads"             {gsub(/[^0-9]/,"",$1); llc=$1}
        $3=="LLC-load-misses"       {gsub(/[^0-9]/,"",$1); llcm=$1}
        END{print l1,l1m,llc,llcm}' "$tmpP")"

    scalar_line=$(tail -n1 "$tmpS")
    vector_line=$(tail -n1 "$tmpV")

    merge_line "$scalar_line" "$vector_line" "$l1" "$l1m" "$llc" "$llcm" >> "$out"
    rm "$tmpS" "$tmpV" "$tmpP"
  done
done

echo "Done  →  $out"
