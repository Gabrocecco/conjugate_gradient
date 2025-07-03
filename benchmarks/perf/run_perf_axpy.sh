#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------
# 1.  Folders
# --------------------------------------------------------------
proj_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$proj_root/build"
src_dir="$proj_root/src"
inc_dir="$proj_root/include"
data_dir="$proj_root/benchmarks/perf/data"

mkdir -p "$build_dir" "$data_dir"

# --------------------------------------------------------------
# 2.  Compilation of axpy_perf
# --------------------------------------------------------------
gcc -O3 -std=c11 -march=rv64gc_xtheadvector -mabi=lp64d \
    -Wall -pedantic -I"$inc_dir" \
    -o "$build_dir/axpy_perf" \
    "$src_dir/vectorized.c" "$src_dir/common.c" axpy_perf.c

# --------------------------------------------------------------
# 3.  Sizes to test
# --------------------------------------------------------------
sizes=(1024 2048 4096 8192 16384 32768 65536 \
       131072 262144 524288 1048576)

# --------------------------------------------------------------
# 4.  Final CSV file (overwritten every time)
# --------------------------------------------------------------
out="$data_dir/axpy_perf_full.csv"
echo "n,time_serial,time_vectorized,speedup_time,"\
"cycles_serial,cycles_vector,speedup_cycles,pass,"\
"L1-loads,L1-misses,LLC-loads,LLC-misses" > "$out"

# --------------------------------------------------------------
# 5.  Main loop 
# --------------------------------------------------------------
for n in "${sizes[@]}"; do
    echo "Profiling n=$n …"

    # ── temp files ─────────────────────────────────────────
    tmp_csv=$(mktemp)   # will contain the CSV line from axpy_perf
    tmp_perf=$(mktemp)  # will contain data from perf stat

    # run the program under perf stat
    AXPY_CSV="$tmp_csv" \
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        -x, --output "$tmp_perf" \
        -- "$build_dir/axpy_perf" "$n"

    # extract the last line from the temporary CSV file
    prog_line=$(tail -n 1 "$tmp_csv")

    # read the performance counters from the perf output using awk
    read l1_loads l1_miss llc_loads llc_miss <<< "$(
        awk -F',' '
            $3=="L1-dcache-loads"       { gsub(/[^0-9]/,"",$1); l1=$1 }
            $3=="L1-dcache-load-misses" { gsub(/[^0-9]/,"",$1); l1m=$1 }
            $3=="LLC-loads"             { gsub(/[^0-9]/,"",$1); llc=$1 }
            $3=="LLC-load-misses"       { gsub(/[^0-9]/,"",$1); llcm=$1 }
            END { print l1, l1m, llc, llcm }
        ' "$tmp_perf"
    )"

    # write the results to the final CSV file
    echo "${prog_line},${l1_loads},${l1_miss},${llc_loads},${llc_miss}" >> "$out"

    # clean up temporary files
    rm "$tmp_csv" "$tmp_perf"
done

echo "Results in $out"
