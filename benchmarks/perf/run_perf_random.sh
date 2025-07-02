#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────
# 1.  Percorsi
# ──────────────────────────────────────────────────────
proj_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$proj_root/build"
src_dir="$proj_root/src"
inc_dir="$proj_root/include"
data_dir="$proj_root/benchmarks/perf/data"

mkdir -p "$build_dir" "$data_dir"

# ──────────────────────────────────────────────────────
# 2.  Compilazione (output = random_mv_perf)
# ──────────────────────────────────────────────────────
gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" \
    -o "$build_dir/random_mv_perf" \
    "$src_dir/vectorized.c" \
    "$src_dir/common.c" \
    "$src_dir/coo.c" \
    "$src_dir/ell.c" \
    "$src_dir/csr.c" \
    mv_random_perf.c            # <-- il tuo .c completo

# ──────────────────────────────────────────────────────
# 3.  Parametri da testare
# ──────────────────────────────────────────────────────
sizes=(1024 2048 4096 8192 16384 32768)
sparsities=(0.01 0.02 0.05 0.10 0.20)

# ──────────────────────────────────────────────────────
# 4.  CSV finale
# ──────────────────────────────────────────────────────
out="$data_dir/mv_random_perf_full.csv"
echo "n,sparsity,max_nnz_row,time_serial,time_vectorized,speedup_time,"\
"cycles_serial,cycles_vector,speedup_cycles,pass,"\
"L1-loads,L1-misses,LLC-loads,LLC-misses" > "$out"

# ──────────────────────────────────────────────────────
# 5.  Loop principale
# ──────────────────────────────────────────────────────
for spars in "${sparsities[@]}"; do
  for n in "${sizes[@]}"; do
    echo "Profiling n=$n  sparsity=$spars …"

    tmp_csv=$(mktemp)
    tmp_perf=$(mktemp)

    # — run unica sotto perf —
    MV_RND_CSV="$tmp_csv" \
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        -x, --output "$tmp_perf" \
        -- "$build_dir/random_mv_perf" "$n" "$spars"

    prog_line=$(tail -n 1 "$tmp_csv")

    read l1 l1m llc llcm <<< "$(
      awk -F',' '
        $3=="L1-dcache-loads"       { gsub(/[^0-9]/,"",$1); l1=$1 }
        $3=="L1-dcache-load-misses" { gsub(/[^0-9]/,"",$1); l1m=$1 }
        $3=="LLC-loads"             { gsub(/[^0-9]/,"",$1); llc=$1 }
        $3=="LLC-load-misses"       { gsub(/[^0-9]/,"",$1); llcm=$1 }
        END { print l1, l1m, llc, llcm }
      ' "$tmp_perf"
    )"

    echo "$prog_line,$l1,$l1m,$llc,$llcm" >> "$out"
    rm "$tmp_csv" "$tmp_perf"
  done
done

echo "Done ➜ $out"
