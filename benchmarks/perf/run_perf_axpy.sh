#!/usr/bin/env bash
# run_axpy_perf.sh – build scalar / vector, tre pass di perf stat
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
  "$src_dir/vectorized.c" "$src_dir/common.c"
  "$proj_root/benchmarks/perf/axpy_perf.c"
)

gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" -DRUN_SCALAR \
    -o "$build_dir/axpy_scalar"   "${common_sources[@]}"

gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" -DRUN_VECTOR \
    -o "$build_dir/axpy_vector"   "${common_sources[@]}"

# ─────────────────── sizes to test ───────────────────────
sizes=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

# ─────────────────── CSV header ──────────────────────────
out="$data_dir/axpy_perf_full.csv"
printf '%s\n' \
"n,"\
"time_scalar,time_vector,speedup_time,"\
"cycles_scalar,cycles_vector,speedup_cycles,"\
"pass,"\
"L1-loads,L1-misses,LLC-loads,LLC-misses,L2-loads,L2-misses" \
> "$out"

# ─────────────────── helper merge_line ───────────────────
merge_line() {          # merge_line scalar_csv vector_csv l1 l1m llc llcm l2a l2m
  IFS=',' read -ra S <<<"$1"; IFS=',' read -ra V <<<"$2"
  n=${S[0]}
  tS=${S[1]} cS=${S[4]}
  tV=${V[2]} cV=${V[5]}
  pass=${V[7]}
  nz(){ [[ $1 =~ ^([0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?)$ ]] && echo "$1" || echo 0; }
  spdT=$(awk "BEGIN{a=$(nz "$tS");b=$(nz "$tV");print (a>0&&b>0)?a/b:NaN}")
  spdC=$(awk "BEGIN{a=$(nz "$cS");b=$(nz "$cV");print (a>0&&b>0)?a/b:NaN}")
  printf '%s,' "$n"
  printf '%.6f,%.6f,%.6f,' "$tS" "$tV" "$spdT"
  printf '%s,%s,%.6f,'     "$cS" "$cV" "$spdC"
  printf '%s,'              "$pass"
  printf '%s,%s,%s,%s,%s,%s\n' "$3" "$4" "$5" "$6" "$7" "$8"
}

# ─────────────────── main loop ──────────────────────────
for n in "${sizes[@]}"; do
  echo "Profiling n=$n …"

  tmpS=$(mktemp) tmpV=$(mktemp)
  tmpP1=$(mktemp) tmpP2=$(mktemp) tmpP3=$(mktemp)

  # ----- scalar run -------------------------------------
  AXPY_CSV="$tmpS" "$build_dir/axpy_scalar" "$n"

  # ----- vector – PASS 1 : L1 ---------------------------
  perf stat -e '{L1-dcache-loads,L1-dcache-load-misses}' \
    -x, --output "$tmpP1" -- \
    env AXPY_CSV="$tmpV" "$build_dir/axpy_vector" "$n"

  # ----- vector – PASS 2 : LLC (alias) ------------------
  perf stat -e '{LLC-loads,LLC-load-misses}' \
    -x, --output "$tmpP2" -- \
    "$build_dir/axpy_vector" "$n"

  # ----- vector – PASS 3 : L2 ---------------------------
  perf stat -e '{r10,r11}' \
    -x, --output "$tmpP3" -- \
    "$build_dir/axpy_vector" "$n"

  # ----- estrazione contatori ---------------------------
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
             "$l1" "$l1m" "$llc" "$llcm" "$l2a" "$l2m" >> "$out"

  rm "$tmpS" "$tmpV" "$tmpP1" "$tmpP2" "$tmpP3"
done

echo "Done  →  $out"
