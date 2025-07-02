#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------
# 1.  Percorsi
# --------------------------------------------------------------
proj_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$proj_root/build"
inc_dir="$proj_root/include"
src_dir="$proj_root/src"
data_dir="$proj_root/benchmarks/perf/data"

mkdir -p "$build_dir" "$data_dir"

# --------------------------------------------------------------
# 2.  Compilazione di mv_foam_perf
# --------------------------------------------------------------
gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" \
    -o "$build_dir/mv_foam_perf" \
    "$src_dir/vectorized.c" "$src_dir/parser.c" "$src_dir/common.c" "$src_dir/ell.c" mv_foam_perf.c

# --------------------------------------------------------------
# 3.  Dataset da testare (file OpenFOAM *.system)
# --------------------------------------------------------------
dataset_dir="$proj_root/data/cylinder"
sizes=(
  "$dataset_dir/2000.system"
  "$dataset_dir/8000.system"
  "$dataset_dir/32k.system"
  "$dataset_dir/128k.system"
)
# --------------------------------------------------------------
# 4.  CSV finale (viene sovrascritto ogni volta)
# --------------------------------------------------------------
out="$data_dir/mv_foam_perf_full.csv"
echo "file,n,nnz_max,time_serial,time_vectorized,speedup_time,"\
"cycles_serial,cycles_vector,speedup_cycles,pass,"\
"L1-loads,L1-misses,LLC-loads,LLC-misses" > "$out"

# --------------------------------------------------------------
# 5.  Loop principale ― **una sola esecuzione** per file
# --------------------------------------------------------------
for f in "${sizes[@]}"; do
    echo "Profiling $f …"

    tmp_csv=$(mktemp)
    tmp_perf=$(mktemp)

    MV_CSV="$tmp_csv" \
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        -x, --output "$tmp_perf" \
        -- "$build_dir/mv_foam_perf" "$f"

    line=$(tail -n 1 "$tmp_csv")

    read l1 l1m llc llcm <<< "$(
        awk -F',' '
            $3=="L1-dcache-loads"       { gsub(/[^0-9]/,"",$1); l1=$1 }
            $3=="L1-dcache-load-misses" { gsub(/[^0-9]/,"",$1); l1m=$1 }
            $3=="LLC-loads"             { gsub(/[^0-9]/,"",$1); llc=$1 }
            $3=="LLC-load-misses"       { gsub(/[^0-9]/,"",$1); llcm=$1 }
            END { print l1, l1m, llc, llcm }
        ' "$tmp_perf"
    )"

    fname=$(basename "$f")            # ← solo il nome del file
    echo "$fname,$line,$l1,$l1m,$llc,$llcm" >> "$out"

    rm "$tmp_csv" "$tmp_perf"
done

echo "Done ➜ $out"
