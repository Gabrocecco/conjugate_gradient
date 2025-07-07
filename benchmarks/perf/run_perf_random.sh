#!/usr/bin/env bash
# run_random_perf.sh – scalar vs vector, due pass di perf stat (L1+LLC)
#                      e merge dei contatori in un unico CSV “grezzo”.
set -euo pipefail

# ──────────────────────── paths ─────────────────────────
proj_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$proj_root/build"
src_dir="$proj_root/src"
inc_dir="$proj_root/include"
data_dir="$proj_root/benchmarks/perf/data"
mkdir -p "$build_dir" "$data_dir"

# ────────────────────── build binaries ──────────────────
common_sources=(
  "$src_dir/vectorized.c" "$src_dir/common.c"
  "$src_dir/coo.c" "$src_dir/ell.c" "$src_dir/csr.c"
  "$proj_root/benchmarks/perf/mv_random_perf.c"
)

gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" -DRUN_SCALAR \
    -o "$build_dir/mv_scalar"   "${common_sources[@]}"

gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" -DRUN_VECTOR \
    -o "$build_dir/mv_vector"   "${common_sources[@]}"

# ───────────────────── experiment matrix ────────────────
# sizes=(1024 2048 4096 8192 16384)
# sparsities=(0.01 0.02 0.05 0.10 0.20)
sizes=(1024 2048 4096 8192 16384 32768 65536)
sparsities=(0.01 0.02 0.05)

# ───────────────────── CSV header ───────────────────────
out="$data_dir/mv_random_perf_full.csv"
printf '%s\n' \
"n,sparsity,max_nnz_row,"\
"time_serial,time_vectorized,speedup_time,"\
"cycles_serial,cycles_vector,speedup_cycles,"\
"inst_serial,inst_vector,speedup_inst,"\
"ipc_serial,ipc_vector,"\
"L1-loads,L1-misses,LLC-loads,LLC-misses" \
> "$out"

# ───────────────────── helper merge_line ────────────────
merge_line() {
  IFS=',' read -ra S <<<"$1"; IFS=',' read -ra V <<<"$2"

  n=${S[0]} spars=${S[1]} max=${S[2]}
  tS=${S[3]} cS=${S[6]} iS=${S[9]}  ipcS=${S[12]}
  tV=${V[4]} cV=${V[7]} iV=${V[10]} ipcV=${V[13]}

  nz(){ [[ $1 =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] && echo "$1" || echo 0; }

  spdT=$(awk "BEGIN{a=$(nz "$tS"); b=$(nz "$tV"); print (a>0&&b>0)?a/b:NaN}")
  spdC=$(awk "BEGIN{a=$(nz "$cS"); b=$(nz "$cV"); print (a>0&&b>0)?a/b:NaN}")
  spdI=$(awk "BEGIN{a=$(nz "$iS"); b=$(nz "$iV"); print (a>0&&b>0)?a/b:NaN}")

  printf '%s,%s,%s,' "$n" "$spars" "$max"
  printf '%.6f,%.6f,%.6f,' "$tS" "$tV" "$spdT"
  printf '%s,%s,%.6f,'     "$cS" "$cV" "$spdC"
  printf '%s,%s,%.6f,'     "$iS" "$iV" "$spdI"
  printf '%s,%s,'           "$ipcS" "$ipcV"
  printf '%s,%s,%s,%s\n'    "$3" "$4" "$5" "$6"
}

# ───────────────────── main loop ────────────────────────
for spars in "${sparsities[@]}"; do
  for n in "${sizes[@]}"; do
    echo "→ n=$n  sparsity=$spars"

    tmpS=$(mktemp) tmpV=$(mktemp)
    tmpP1=$(mktemp) tmpP2=$(mktemp)   # L1 / LLC

    # 1) scalar (no perf) ------------------------------------------
    if ! MV_RND_CSV="$tmpS" "$build_dir/mv_scalar" "$n" "$spars"; then
        echo "  [skip] scalar run failed"; rm -f "$tmpS" "$tmpV" "$tmpP1" "$tmpP2"; continue
    fi

    # 2) vector – PASS 1 (L1)  + CSV line --------------------------
    if ! MV_RND_CSV="$tmpV" \
         perf stat -e '{L1-dcache-loads,L1-dcache-load-misses}' \
                   -x, --output "$tmpP1" -- \
                   "$build_dir/mv_vector" "$n" "$spars"; then
        echo "  [skip] vector run failed"; rm -f "$tmpS" "$tmpV" "$tmpP1" "$tmpP2"; continue
    fi

    # la riga CSV deve iniziare con un numero
    vec_line=$(tail -n1 "$tmpV")
    [[ "$vec_line" =~ ^[0-9]+, ]] || { 
        echo "  [warn] no data line – skipped"; rm -f "$tmpS" "$tmpV" "$tmpP1" "$tmpP2"; continue; }

    # 3) vector – PASS 2 (LLC)  ------------------------------------
    perf stat -e '{LLC-loads,LLC-load-misses}' \
              -x, --output "$tmpP2" -- \
              "$build_dir/mv_vector" "$n" "$spars" 2>/dev/null

    # -------- estrazione contatori -------------------------------
    read l1 l1m <<<"$(awk -F',' '
        $3=="L1-dcache-loads"       {gsub(/[^0-9]/,"",$1); l1=$1}
        $3=="L1-dcache-load-misses" {gsub(/[^0-9]/,"",$1); l1m=$1}
        END{print l1,l1m}' "$tmpP1")"

    read llc llcm <<<"$(awk -F',' '
        $3=="LLC-loads"       {gsub(/[^0-9]/,"",$1); llc=$1}
        $3=="LLC-load-misses" {gsub(/[^0-9]/,"",$1); llcm=$1}
        END{print llc,llcm}' "$tmpP2")"

    merge_line "$(tail -n1 "$tmpS")" "$vec_line" \
               "$l1" "$l1m" "$llc" "$llcm" >> "$out"

    rm -f "$tmpS" "$tmpV" "$tmpP1" "$tmpP2"
  done
done

echo "CSV pronto → $out"
