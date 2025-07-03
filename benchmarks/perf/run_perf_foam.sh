#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------
# 1.  Folders 
# --------------------------------------------------------------
proj_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$proj_root/build"
inc_dir="$proj_root/include"
src_dir="$proj_root/src"
data_dir="$proj_root/benchmarks/perf/data"

mkdir -p "$build_dir" "$data_dir"

# --------------------------------------------------------------
# 2.  COmpilation of mv_foam_perf
# --------------------------------------------------------------
gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" \
    -o "$build_dir/mv_foam_perf" \
    "$src_dir/vectorized.c" "$src_dir/parser.c" "$src_dir/common.c" "$src_dir/ell.c" mv_foam_perf.c

# --------------------------------------------------------------
# 3.  Foam matrices to test on (file OpenFOAM *.system)
# --------------------------------------------------------------
dataset_dir="$proj_root/data/cylinder"
sizes=(
  "$dataset_dir/2000.system"
  "$dataset_dir/8000.system"
  "$dataset_dir/32k.system"
  "$dataset_dir/128k.system"
)
# --------------------------------------------------------------
# 4.  final CSV (overwritten every time)
# --------------------------------------------------------------
out="$data_dir/mv_foam_perf_full.csv"
echo "file,n,nnz_max,time_serial,time_vectorized,speedup_time,"\
"cycles_serial,cycles_vector,speedup_cycles,pass,"\
"L1-loads,L1-misses,LLC-loads,LLC-misses" > "$out"

# --------------------------------------------------------------
# 5.  Main loop 
# --------------------------------------------------------------
for f in "${sizes[@]}"; do  # for each file in the dataset
    echo "Profiling $f …"

    tmp_csv=$(mktemp)   # will produce the CSV line from mv_foam_perf
    tmp_perf=$(mktemp)  # will contain data from perf stat

    # set the environment variable for the CSV output and run perf stat
    MV_CSV="$tmp_csv" \
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        -x, --output "$tmp_perf" \
        -- "$build_dir/mv_foam_perf" "$f"   

    # extract the last line from the temporary CSV file
    line=$(tail -n 1 "$tmp_csv")

    # extract the performance counters from the perf output using awk
    read l1 l1m llc llcm <<< "$(
        awk -F',' '
            $3=="L1-dcache-loads"       { gsub(/[^0-9]/,"",$1); l1=$1 }
            $3=="L1-dcache-load-misses" { gsub(/[^0-9]/,"",$1); l1m=$1 }
            $3=="LLC-loads"             { gsub(/[^0-9]/,"",$1); llc=$1 }
            $3=="LLC-load-misses"       { gsub(/[^0-9]/,"",$1); llcm=$1 }
            END { print l1, l1m, llc, llcm }
        ' "$tmp_perf"
    )"

    fname=$(basename "$f")            # extract the filename from the full path
    echo "$fname,$line,$l1,$l1m,$llc,$llcm" >> "$out"   # append the line to the CSV file

    rm "$tmp_csv" "$tmp_perf"   # remove temporary files
done

echo "Done -> $out"
