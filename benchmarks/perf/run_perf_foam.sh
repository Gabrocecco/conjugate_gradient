#!/usr/bin/env bash
# run_foam_perf.sh – build scalar / vector, tre pass di perf stat
#                    (L1, LLC, L2) e CSV con contatori grezzi.
set -euo pipefail

# ───────────────────────── paths ─────────────────────────
proj_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$proj_root/build"
src_dir="$proj_root/src"
inc_dir="$proj_root/include"
data_dir="$proj_root/benchmarks/perf/data"
mkdir -p "$build_dir" "$data_dir"

# ────────────────────── build binaries ───────────────────
common_sources=(
  "$src_dir/vectorized.c" "$src_dir/parser.c" "$src_dir/common.c"
  "$src_dir/ell.c"        "$proj_root/benchmarks/perf/mv_foam_perf.c"
)

gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" -DRUN_SCALAR \
    -o "$build_dir/mv_foam_scalar" "${common_sources[@]}"

gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" -DRUN_VECTOR \
    -o "$build_dir/mv_foam_vector" "${common_sources[@]}"

# ───────────────────── dataset OpenFOAM ──────────────────
dataset_dir="$proj_root/data/cylinder"
mats=(
  "$dataset_dir/2000.system"
  "$dataset_dir/8000.system"
  "$dataset_dir/32k.system"
  "$dataset_dir/128k.system"
)

# ───────────────────── CSV header ────────────────────────
out="$data_dir/mv_foam_perf_full.csv"
printf '%s\n' \
"file,n,nnz_max,"\
"time_serial,time_vectorized,speedup_time,"\
"cycles_serial,cycles_vector,speedup_cycles,"\
"pass,"\
"L1-loads,L1-misses,LLC-loads,LLC-misses,L2-loads,L2-misses" \
> "$out"

# ───────────────────── helper merge_line ─────────────────
merge_line() {          # merge_line scalar_csv vector_csv l1 l1m llc llcm l2a l2m
  IFS=',' read -ra S <<<"$1"; IFS=',' read -ra V <<<"$2"
  n=${S[0]} nnz=${S[1]}
  tS=${S[2]} cS=${S[5]}
  tV=${V[3]} cV=${V[6]} pass=${V[8]}
  nz(){ [[ $1 =~ ^([0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?)$ ]] && echo "$1" || echo 0; }
  spdT=$(awk "BEGIN{a=$(nz "$tS");b=$(nz "$tV");print (a>0&&b>0)?a/b:NaN}")
  spdC=$(awk "BEGIN{a=$(nz "$cS");b=$(nz "$cV");print (a>0&&b>0)?a/b:NaN}")
  printf '%s,%s,' "$n" "$nnz"
  printf '%.6f,%.6f,%.6f,' "$tS" "$tV" "$spdT"
  printf '%s,%s,%.6f,'     "$cS" "$cV" "$spdC"
  printf '%s,'              "$pass"
  printf '%s,%s,%s,%s,%s,%s\n' "$3" "$4" "$5" "$6" "$7" "$8"
}

# ───────────────────── main loop ────────────────────────
for f in "${mats[@]}"; do
  echo "Profiling $(basename "$f") …"

  tmpS=$(mktemp) tmpV=$(mktemp)
  tmpP1=$(mktemp) tmpP2=$(mktemp) tmpP3=$(mktemp)

  # ---------- scalar ------------------------------------
  MV_CSV="$tmpS" "$build_dir/mv_foam_scalar" "$f"

  # ---------- vector – PASS 1 (L1) -----------------------
  perf stat -e '{L1-dcache-loads,L1-dcache-load-misses}' \
    -x, --output "$tmpP1" -- \
    env MV_CSV="$tmpV" "$build_dir/mv_foam_vector" "$f"

  # ---------- vector – PASS 2 (LLC alias) ---------------
  perf stat -e '{LLC-loads,LLC-load-misses}' \
    -x, --output "$tmpP2" -- \
    "$build_dir/mv_foam_vector" "$f"

  # ---------- vector – PASS 3 (L2) -----------------------
  perf stat -e '{r10,r11}' \
    -x, --output "$tmpP3" -- \
    "$build_dir/mv_foam_vector" "$f"

  # ---------- estrazione contatori ----------------------
  read l1 l1m <<<"$(awk -F',' '
     $3=="L1-dcache-loads"       {gsub(/[^0-9]/,"",$1); l1=$1}
     $3=="L1-dcache-load-misses" {gsub(/[^0-9]/,"",$1); l1m=$1}
     END{print l1,l1m}' "$tmpP1")"

  read llc llcm <<<"$(awk -F',' '
     $3=="LLC-loads"       {gsub(/[^0-9]/,"",$1); llc=$1}
     $3=="LLC-load-misses" {gsub(/[^0-9]/,"",$1); llcm=$1}
     END{print llc,llcm}' "$tmpP2")"

  read l2a l2m <<<"$(awk -F',' '
     $3=="r10" {gsub(/[^0-9]/,"",$1); l2a=$1}
     $3=="r11" {gsub(/[^0-9]/,"",$1); l2m=$1}
     END{print l2a,l2m}' "$tmpP3")"

  merge_line "$(tail -n1 "$tmpS")" "$(tail -n1 "$tmpV")" \
             "$l1" "$l1m" "$llc" "$llcm" "$l2a" "$l2m" \
  | sed "s#^#$(basename "$f"),#" >> "$out"

  rm "$tmpS" "$tmpV" "$tmpP1" "$tmpP2" "$tmpP3"
done

echo "Done  →  $out"
